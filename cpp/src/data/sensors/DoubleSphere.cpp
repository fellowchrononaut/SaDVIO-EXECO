#include "isaeslam/data/sensors/DoubleSphere.h"

namespace isae {

Eigen::Vector3d DoubleSphere::getRay(Eigen::Vector2d f) {
    Eigen::Vector3d ray_world;
    Eigen::Vector3d ray_cam = this->getRayCamera(f);

    ray_world = this->getSensor2WorldTransform().rotation() * ray_cam;
    ray_world.normalize();
    return ray_world;
}

Eigen::Vector3d DoubleSphere::getRayCamera(Eigen::Vector2d f) {
    Eigen::Vector3d ray_cam;

    // double sphere model as defined by Usenko et. al.
    double mx = (f(0) - _calibration(0, 2)) / _calibration(0, 0);
    double my = (f(1) - _calibration(1, 2)) / _calibration(1, 1);
    double r2 = mx * mx + my * my;

    double mz  = (1 - _alpha * _alpha * r2) / (_alpha * std::sqrt(1 - (2 * _alpha - 1) * r2) + 1 - _alpha);
    double mz2 = mz * mz;
    double k   = (mz * _xi + std::sqrt(mz2 + (1 - _xi * _xi) * r2)) / (mz2 + r2);

    ray_cam[0] = k * mx;
    ray_cam[1] = k * my;
    ray_cam[2] = k * mz - _xi;
    
    return ray_cam;
}

bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const std::shared_ptr<AModel3d> ldmk_model,
                           std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(Eigen::Vector3d::Ones());
        Eigen::Vector3d t_cam_lmk = this->getWorld2SensorTransform() * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // double sphere model as defined by Usenko et. al.
        double d1 =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        double d2 = std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() +
                              (_xi * d1 + t_cam_lmk.z()) * (_xi * d1 + t_cam_lmk.z()));
        Eigen::Vector3d sph_pix(t_cam_lmk.x() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                t_cam_lmk.y() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                1);
        Eigen::Vector2d p2d = (this->getCalibration() * sph_pix).block<2, 1>(0, 0);

        p2d_vector.push_back(p2d);

        // Check validity
        double w1;
        if (_alpha <= 0.5)
            w1 = _alpha / (1 - _alpha);
        else
            w1 = (1 - _alpha) / _alpha;

        double w2 = (w1 + _xi) / std::sqrt(2 * w1 * _xi + _xi * _xi + 1);
        if (t_cam_lmk.z() <= -w2 * d1)
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;
    }

    return true;
}

bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const std::shared_ptr<AModel3d> ldmk_model,
                           const Eigen::Affine3d &T_f_w,
                           std::vector<Eigen::Vector2d> &p2d_vector) {

    for (const auto &p3d_model : ldmk_model->getModel()) {
        // conversion to the camera coordinate system
        Eigen::Vector3d t_w_lmk   = T_w_lmk * p3d_model.cwiseProduct(Eigen::Vector3d::Ones());
        Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;

        if (t_cam_lmk[2] < 0.1) // point behind the camera
            return false;

        // double sphere model as defined by Usenko et. al.
        double d1 =
            std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() + t_cam_lmk.z() * t_cam_lmk.z());
        double d2 = std::sqrt(t_cam_lmk.x() * t_cam_lmk.x() + t_cam_lmk.y() * t_cam_lmk.y() +
                              (_xi * d1 + t_cam_lmk.z()) * (_xi * d1 + t_cam_lmk.z()));
        Eigen::Vector3d sph_pix(t_cam_lmk.x() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                t_cam_lmk.y() / (_alpha * d2 + (1 - _alpha) * (_xi * d1 + t_cam_lmk.z())),
                                1);
        Eigen::Vector2d p2d = (this->getCalibration() * sph_pix).block<2, 1>(0, 0);

        // Check validity
        double w1;
        if (_alpha <= 0.5)
            w1 = _alpha / (1 - _alpha);
        else
            w1 = (1 - _alpha) / _alpha;

        double w2 = (w1 + _xi) / std::sqrt(2 * w1 * _xi + _xi * _xi + 1);
        if (t_cam_lmk.z() <= -w2 * d1)
            return false;

        if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows) // out of image
            return false;
        
        if (!std::isfinite(p2d[0]) || !std::isfinite(p2d[1])) // Nan returned
            return false;

        p2d_vector.push_back(p2d);
    }

    return true;
}
bool DoubleSphere::project(const Eigen::Affine3d &T_w_lmk,
                           const Eigen::Affine3d &T_f_w,
                           const Eigen::Matrix2d sqrt_info,
                           Eigen::Vector2d &p2d,
                           double *J_proj_frame,
                           double *J_proj_lmk) {

    // conversion to the camera coordinate system
    const Eigen::Vector3d t_w_lmk   = T_w_lmk.translation();
    const Eigen::Vector3d t_cam_lmk = this->getFrame2SensorTransform() * T_f_w * t_w_lmk;
    const double x = t_cam_lmk.x(), y = t_cam_lmk.y(), z = t_cam_lmk.z();

    // double sphere model as defined by Usenko et. al.
    const double d1    = t_cam_lmk.norm();
    const double w     = _xi * d1 + z;
    const double d2    = std::sqrt(x * x + y * y + w * w);
    const double denom = _alpha * d2 + (1 - _alpha) * w;
    const Eigen::Matrix3d K = this->getCalibration();
    p2d = Eigen::Vector2d(K(0, 0) * x / denom + K(0, 2), K(1, 1) * y / denom + K(1, 2));

    if (J_proj_frame != NULL || J_proj_lmk != NULL) {
        // d projection / d point in the camera frame
        const Eigen::Vector3d d_d1 = t_cam_lmk / d1;
        const Eigen::Vector3d d_w  = _xi * d_d1 + Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d d_d2 = (Eigen::Vector3d(x, y, 0) + w * d_w) / d2;
        const Eigen::Vector3d d_den = _alpha * d_d2 + (1 - _alpha) * d_w;
        Eigen::Matrix<double, 2, 3> J_cam;
        J_cam.row(0) = K(0, 0) * (Eigen::Vector3d::UnitX() / denom - x * d_den / (denom * denom)).transpose();
        J_cam.row(1) = K(1, 1) * (Eigen::Vector3d::UnitY() / denom - y * d_den / (denom * denom)).transpose();

        // Same parameterization as Camera::project
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

    // Validity: point in the valid region of the model and inside the image
    double w1 = (_alpha <= 0.5) ? _alpha / (1 - _alpha) : (1 - _alpha) / _alpha;
    double w2 = (w1 + _xi) / std::sqrt(2 * w1 * _xi + _xi * _xi + 1);
    if (z <= -w2 * d1 || !std::isfinite(p2d[0]) || !std::isfinite(p2d[1]))
        return false;
    if (p2d[0] < 0 || p2d[1] < 0 || p2d[0] > _raw_data.cols || p2d[1] > _raw_data.rows)
        return false;

    return true;
}

} // namespace isae