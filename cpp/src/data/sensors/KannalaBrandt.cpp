#include "isaeslam/data/sensors/KannalaBrandt.h"
#include "utilities/geometry.h"

#include <cmath>

namespace isae {

bool KannalaBrandt::projectCamera(const Eigen::Vector3d &p_cam, Eigen::Vector2d &p2d,
                                  Eigen::Matrix<double, 2, 3> *J) const {
    const double x = p_cam.x(), y = p_cam.y(), z = p_cam.z();
    const double r2 = x * x + y * y, rho2 = r2 + z * z;
    if (!(rho2 > 1e-18))
        return false;
    const double r     = std::sqrt(r2);
    const double theta = std::atan2(r, z); // angle from the optical axis, in [0, pi]
    if (theta > _max_theta)
        return false;
    const double t2 = theta * theta;
    const double poly  = 1 + t2 * (_k(0) + t2 * (_k(1) + t2 * (_k(2) + t2 * _k(3))));
    const double th_d  = theta * poly;
    const double fx = _calibration(0, 0), fy = _calibration(1, 1), cx = _calibration(0, 2), cy = _calibration(1, 2);

    // Near the axis (r -> 0): theta_d / r -> 1 / z, the pinhole limit
    const bool near_axis = r < 1e-9 * std::sqrt(rho2);
    const double s       = near_axis ? 1.0 / z : th_d / r;
    p2d                  = Eigen::Vector2d(fx * s * x + cx, fy * s * y + cy);

    if (J) {
        Eigen::Vector3d ds;
        if (near_axis) {
            ds = Eigen::Vector3d(0, 0, -1.0 / (z * z));
        } else {
            // s = theta_d(theta) / r: ds = (theta_d' dtheta - s dr) / r
            const double dth_d = 1 + t2 * (3 * _k(0) + t2 * (5 * _k(1) + t2 * (7 * _k(2) + t2 * 9 * _k(3))));
            const Eigen::Vector3d dtheta(x * z / (r * rho2), y * z / (r * rho2), -r / rho2);
            const Eigen::Vector3d dr(x / r, y / r, 0);
            ds = (dth_d * dtheta - s * dr) / r;
        }
        J->row(0) = fx * (s * Eigen::Vector3d::UnitX() + x * ds).transpose();
        J->row(1) = fy * (s * Eigen::Vector3d::UnitY() + y * ds).transpose();
    }

    if (!std::isfinite(p2d.x()) || !std::isfinite(p2d.y()))
        return false;
    if (p2d.x() < 0 || p2d.y() < 0 || p2d.x() > _raw_data.cols || p2d.y() > _raw_data.rows)
        return false;
    return true;
}

cv::Mat KannalaBrandt::imageCircleMask() const {
    // theta_d grows with theta over the lens' range, so the circle is theta_d <= theta_d(max_theta) in normalized units
    const double t2    = _max_theta * _max_theta;
    const double th_dm = _max_theta * (1 + t2 * (_k(0) + t2 * (_k(1) + t2 * (_k(2) + t2 * _k(3)))));
    const double fx = _calibration(0, 0), fy = _calibration(1, 1), cx = _calibration(0, 2), cy = _calibration(1, 2);
    cv::Mat m(_raw_data.rows, _raw_data.cols, CV_8UC1);
    for (int v = 0; v < m.rows; v++) {
        uchar *row     = m.ptr<uchar>(v);
        const double my = (v - cy) / fy;
        for (int u = 0; u < m.cols; u++) {
            const double mx = (u - cx) / fx;
            row[u]          = (mx * mx + my * my <= th_dm * th_dm) ? 255 : 0;
        }
    }
    return m;
}

Eigen::Vector3d KannalaBrandt::getRayCamera(Eigen::Vector2d f) {
    const double mx   = (f(0) - _calibration(0, 2)) / _calibration(0, 0);
    const double my   = (f(1) - _calibration(1, 2)) / _calibration(1, 1);
    const double th_d = std::sqrt(mx * mx + my * my);
    if (th_d < 1e-12)
        return Eigen::Vector3d::UnitZ();

    // theta from theta_d by Newton (theta_d is monotonic in theta over the lens' range)
    double theta = std::min(th_d, M_PI);
    for (int it = 0; it < 20; it++) {
        const double t2  = theta * theta;
        const double f_t = theta * (1 + t2 * (_k(0) + t2 * (_k(1) + t2 * (_k(2) + t2 * _k(3))))) - th_d;
        const double df  = 1 + t2 * (3 * _k(0) + t2 * (5 * _k(1) + t2 * (7 * _k(2) + t2 * 9 * _k(3))));
        if (std::abs(df) < 1e-12)
            break;
        const double step = f_t / df;
        theta -= step;
        if (std::abs(step) < 1e-12)
            break;
    }
    const double sin_t = std::sin(theta);
    return Eigen::Vector3d(sin_t * mx / th_d, sin_t * my / th_d, std::cos(theta));
}

Eigen::Vector3d KannalaBrandt::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world = this->getSensor2WorldTransform().rotation() * this->getRayCamera(f);
    ray_world.normalize();
    return ray_world;
}

bool KannalaBrandt::project(const Eigen::Affine3d &T_w_lmk,
                            const std::shared_ptr<AModel3d> ldmk_model,
                            std::vector<Eigen::Vector2d> &p2d_vector) {
    for (const auto &p3d_model : ldmk_model->getModel()) {
        const Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * (T_w_lmk * p3d_model);
        Eigen::Vector2d p2d;
        const bool ok = projectCamera(t_cam_lmk, p2d);
        p2d_vector.push_back(p2d);
        if (!ok)
            return false;
    }
    return true;
}

bool KannalaBrandt::project(const Eigen::Affine3d &T_w_lmk,
                            const std::shared_ptr<AModel3d> ldmk_model,
                            const Eigen::Affine3d &T_f_w,
                            std::vector<Eigen::Vector2d> &p2d_vector) {
    for (const auto &p3d_model : ldmk_model->getModel()) {
        const Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * (T_w_lmk * p3d_model);
        Eigen::Vector2d p2d;
        if (!projectCamera(t_cam_lmk, p2d))
            return false;
        p2d_vector.push_back(p2d);
    }
    return true;
}

bool KannalaBrandt::project(const Eigen::Affine3d &T_w_lmk,
                            const Eigen::Affine3d &T_f_w,
                            const Eigen::Matrix2d sqrt_info,
                            Eigen::Vector2d &p2d,
                            double *J_proj_frame,
                            double *J_proj_lmk) {
    const Eigen::Vector3d t_w_lmk   = T_w_lmk.translation();
    const Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;
    Eigen::Matrix<double, 2, 3> J_cam = Eigen::Matrix<double, 2, 3>::Zero(); // stays zero if invalid
    const bool ok = projectCamera(t_cam_lmk, p2d, (J_proj_frame || J_proj_lmk) ? &J_cam : nullptr);

    if (J_proj_frame || J_proj_lmk) {
        // Same parameterization as Camera::project and DoubleSphere::project
        if (J_proj_frame != NULL) {
            Eigen::MatrixXd J_int   = Eigen::MatrixXd::Zero(3, 6);
            J_int.block(0, 0, 3, 3) = -T_f_w.linear() * isae::geometry::skewMatrix(t_w_lmk) *
                                      geometry::so3_rightJacobian(isae::geometry::se3_RTtoVec6d(T_f_w).block<3, 1>(0, 0));
            J_int.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_frame(J_proj_frame);
            J_frame = sqrt_info * J_cam * this->getFrame2SensorTransform().linear() * J_int;
        }
        if (J_proj_lmk != NULL) {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_lmk(J_proj_lmk);
            J_lmk = sqrt_info * J_cam * this->getFrame2SensorTransform().linear() * T_f_w.linear();
        }
    }
    return ok;
}

} // namespace isae
