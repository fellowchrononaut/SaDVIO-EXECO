#ifndef KANNALABRANDT_H
#define KANNALABRANDT_H

#include "isaeslam/data/sensors/ASensor.h"
#include <opencv2/core/types.hpp>

namespace isae {

/*!
 * @brief An image sensor with the Kannala-Brandt model, 4 coefficients (KB4, OpenCV fisheye, COLMAP OPENCV_FISHEYE)
 *
 * A point at angle theta from the optical axis maps to the image radius theta_d = theta (1 + k1 theta^2 + k2 theta^4 +
 * k3 theta^6 + k4 theta^8), in normalized units, along the point's azimuth: u = fx theta_d x / r + cx,
 * v = fy theta_d y / r + cy, with r = sqrt(x^2 + y^2). Projection and unprojection go through the angle, not x / z,
 * so the model holds beyond 90 deg from the axis (field of view over 180 deg: points behind the image plane, z <= 0).
 * A point is valid up to max_theta from the axis (the lens' image circle); the unprojection returns unit bearings.
 * (doc/nofovvio/NOFOVVIO_LEDGER.md, Insta360 X5: image circle 95-100 deg from the axis)
 */
class KannalaBrandt : public ImageSensor {

  public:
    KannalaBrandt(const cv::Mat &image, Eigen::Matrix3d K, const Eigen::Vector4d &k, double max_theta_deg = 100.0)
        : ImageSensor() {
        _calibration = K;
        _raw_data    = image.clone();
        _k           = k;
        _max_theta   = max_theta_deg * M_PI / 180.0;
        _has_depth   = false;
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
    }

    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Affine3d &T_f_w,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const Eigen::Affine3d &T_f_w,
                 const Eigen::Matrix2d sqrt_info,
                 Eigen::Vector2d &p2d,
                 double *J_proj_frame,
                 double *J_proj_lmk) override;

    //! Unit bearing of a pixel in the camera frame (z <= 0 beyond 90 deg from the axis)
    Eigen::Vector3d getRayCamera(Eigen::Vector2d f) override;
    Eigen::Vector3d getRay(Eigen::Vector2d f) override;
    double getFocal() override { return (_calibration(0, 0) + _calibration(1, 1)) / 2; }

    double maxTheta() const { return _max_theta; }

    //! 8-bit mask of the image circle: 255 where the pixel's ray is within max_theta of the axis, 0 outside
    cv::Mat imageCircleMask() const;

    /*!
     * @brief Pixel of a point in the camera frame, and optionally the Jacobian d pixel / d point (2 x 3, row-major)
     * @return false outside max_theta, the image, or for a point at the optical centre
     */
    bool projectCamera(const Eigen::Vector3d &p_cam, Eigen::Vector2d &p2d, Eigen::Matrix<double, 2, 3> *J = nullptr) const;

  private:
    Eigen::Vector4d _k; //!< k1..k4
    double _max_theta;  //!< Valid angle from the optical axis (rad)
};

} // namespace isae

#endif
