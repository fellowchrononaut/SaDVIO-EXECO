#ifndef ESKFESTIMATOR_H
#define ESKFESTIMATOR_H
#include "isaeslam/estimator/APoseEstimator.h"

namespace isae {

/*!
 * @brief ESKFEstimator class for estimating the transformation between two frames using an EKF
 *
 * The estimation is done by perfoming an ESKF update on each landmark (the matches must be associated with landmarks).
 * This allows closed form computation of the covariance of the estimated transformation.
 */
class ESKFEstimator : public APoseEstimator {
  public:
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  vec_match &matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  typed_vec_match &typed_matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;

    /*!
     * @brief Refine the triangulation of the landmarks with an ESKF BEWARE: currently being tested
     */
    bool refineTriangulation(std::shared_ptr<Frame> &frame);

    /*!
     * @brief Ad_{M^-1} for perturbations E(theta, rho) = [Exp(theta), rho] (order: rotation, translation):
     * E(xi) M = M E(Ad_{M^-1} xi) to first order
     */
    static Eigen::Matrix<double, 6, 6> adjointOfInverse(const Eigen::Affine3d &M);

    /*!
     * @brief Jacobian from the IMU-step error of dT = T_f1_f2 (R Exp(dtheta), t + dt) to the right perturbation of
     * T_cam2_cam1 = T_cam2_f2 dT^-1 T_cam1_f1^-1
     */
    static Eigen::Matrix<double, 6, 6> imuToCameraErrorJacobian(const Eigen::Affine3d &dT,
                                                                const Eigen::Affine3d &T_cam1_f1);

    /*!
     * @brief Jacobian from the right perturbation of T_cam2_cam1 to the right perturbation dT E(xi) of
     * dT = T_cam1_f1^-1 T_cam2_cam1^-1 T_cam2_f2
     */
    static Eigen::Matrix<double, 6, 6> cameraToFrameErrorJacobian(const Eigen::Affine3d &T_cam2_cam1,
                                                                  const Eigen::Affine3d &T_cam2_f2);

    /*!
     * @brief Kalman update of the rotation of dT = T_f1_f2 with the preintegrated rotation delta: one
     * Gauss-Newton step of the MAP cost, error R = R_pred Exp(d), residual Log(R^T dR) (Jacobian -Jl^-1)
     * @param R_pred Predicted rotation of dT
     * @param dR Preintegrated rotation delta
     * @param P Covariance of d, updated
     * @param cov_dR Covariance of dR
     * @return The updated rotation
     */
    static Eigen::Matrix3d updateRotation(const Eigen::Matrix3d &R_pred,
                                          const Eigen::Matrix3d &dR,
                                          Eigen::Matrix3d &P,
                                          const Eigen::Matrix3d &cov_dR);

    /*!
     * @brief Kalman update of the world velocity of frame 2 with the preintegrated velocity delta.
     *
     * Measurement model (in frame 1's axes): dV = R_w_f1^T (v2 - v1 - g dt), so H = R_w_f1^T. The
     * predicted velocity has an identity prior covariance and the measurement covariance is cov_dv.
     *
     * @param R_w_f1 Rotation of frame 1 (frame to world)
     * @param v_pred Predicted world velocity of frame 2
     * @param v1 World velocity of frame 1
     * @param dV Preintegrated velocity delta between frame 1 and frame 2
     * @param cov_dv Covariance of dV
     * @param dt Time between frame 1 and frame 2
     * @return The updated world velocity of frame 2
     */
    static Eigen::Vector3d updateVelocity(const Eigen::Matrix3d &R_w_f1,
                                          const Eigen::Vector3d &v_pred,
                                          const Eigen::Vector3d &v1,
                                          const Eigen::Vector3d &dV,
                                          const Eigen::Matrix3d &cov_dv,
                                          double dt);
};

} // namespace isae
#endif // ESKFESTIMATOR_H