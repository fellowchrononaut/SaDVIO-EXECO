#include "isaeslam/estimator/ESKFEstimator.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/IMU.h"
#include "utilities/geometry.h"
#include <iostream>

namespace isae {

// Function to compute the Jacobian of the switch from homogeneous point to 2D point
Eigen::MatrixXd jac_homogeneous(const Eigen::Vector3d &point) {
    Eigen::MatrixXd jac(2, 3);
    double inv_z    = 1.0 / point(2);
    double inv_z_sq = inv_z * inv_z;

    jac(0, 0) = inv_z;
    jac(0, 1) = 0;
    jac(0, 2) = -point(0) * inv_z_sq;

    jac(1, 0) = 0;
    jac(1, 1) = inv_z;
    jac(1, 2) = -point(1) * inv_z_sq;

    return jac;
}

// Function to compute the Jacobian of the update of the pose T \delta \tau w.r.t the delta
Eigen::MatrixXd jac_delta_update(const Eigen::Vector3d &dtheta, const Eigen::Matrix3d &rotation) {
    Eigen::MatrixXd jr = geometry::so3_rightJacobian(dtheta);
    Eigen::MatrixXd jac(6, 6);

    jac.block<3, 3>(0, 0) = jr;
    jac.block<3, 3>(0, 3) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 0) = Eigen::MatrixXd::Zero(3, 3);
    jac.block<3, 3>(3, 3) = rotation;

    return jac;
}

std::tuple<Eigen::Vector2d, Eigen::MatrixXd, Eigen::MatrixXd> jac_projection(const Eigen::Matrix3d &camera_matrix,
                                                                             const Eigen::Vector3d &point_3d,
                                                                             const Eigen::Matrix3d &rotation,
                                                                             const Eigen::Vector3d &translation,
                                                                             const Eigen::Vector3d &dtheta) {
    // Project point
    Eigen::MatrixXd extrinsic_matrix(3, 4);
    extrinsic_matrix.block<3, 3>(0, 0) = rotation;
    extrinsic_matrix.col(3)            = translation;

    Eigen::Vector4d homogeneous_point(point_3d(0), point_3d(1), point_3d(2), 1);
    Eigen::Vector3d proj = camera_matrix * extrinsic_matrix * homogeneous_point;

    // Compute jacs
    Eigen::MatrixXd jac_proj_R = -camera_matrix * rotation * geometry::skewMatrix(point_3d);
    Eigen::MatrixXd jac_proj_T(3, 6);
    jac_proj_T.block<3, 3>(0, 0) = jac_proj_R;
    jac_proj_T.block<3, 3>(0, 3) = Eigen::MatrixXd::Identity(3, 3);

    Eigen::MatrixXd J_T = jac_homogeneous(proj) * jac_proj_T * jac_delta_update(dtheta, rotation);
    Eigen::MatrixXd J_p = jac_homogeneous(proj) * camera_matrix * rotation;

    // Compute 2D projection
    Eigen::Vector2d proj_2d(proj(0) / proj(2), proj(1) / proj(2));

    return {proj_2d, J_T, J_p};
}

Eigen::Matrix<double, 6, 6> ESKFEstimator::adjointOfInverse(const Eigen::Affine3d &M) {
    // E(xi) M = M E(Ad_{M^-1} xi), with E(theta, rho) = [Exp(theta), rho] (perturbation order: rotation, translation)
    const Eigen::Matrix3d Rt         = M.rotation().transpose();
    Eigen::Matrix<double, 6, 6> Ad   = Eigen::Matrix<double, 6, 6>::Zero();
    Ad.block<3, 3>(0, 0)             = Rt;
    Ad.block<3, 3>(3, 0)             = -Rt * geometry::skewMatrix(M.translation());
    Ad.block<3, 3>(3, 3)             = Rt;
    return Ad;
}

Eigen::Matrix<double, 6, 6> ESKFEstimator::imuToCameraErrorJacobian(const Eigen::Affine3d &dT,
                                                                    const Eigen::Affine3d &T_cam1_f1) {
    // IMU step error of dT = T_f1_f2: R <- R Exp(dtheta), t <- t + dt. The visual step perturbs
    // T_cam2_cam1 = T_cam2_f2 dT^-1 T_cam1_f1^-1 on the right (the left factor T_cam2_f2 does not change it)
    const Eigen::Affine3d B         = T_cam1_f1.inverse();
    Eigen::Matrix<double, 6, 6> J   = Eigen::Matrix<double, 6, 6>::Zero();
    J.leftCols<3>()                 = -adjointOfInverse(dT.inverse() * B).leftCols<3>();
    J.rightCols<3>()                = -adjointOfInverse(B).rightCols<3>();
    return J;
}

Eigen::Matrix<double, 6, 6> ESKFEstimator::cameraToFrameErrorJacobian(const Eigen::Affine3d &T_cam2_cam1,
                                                                      const Eigen::Affine3d &T_cam2_f2) {
    // dT = T_cam1_f1^-1 T_cam2_cam1^-1 T_cam2_f2; a right perturbation of T_cam2_cam1 gives the right perturbation
    // dT E(xi_dT) of dT
    return -adjointOfInverse(T_cam2_cam1.inverse() * T_cam2_f2);
}

Eigen::Matrix3d ESKFEstimator::updateRotation(const Eigen::Matrix3d &R_pred,
                                              const Eigen::Matrix3d &dR,
                                              Eigen::Matrix3d &P,
                                              const Eigen::Matrix3d &cov_dR) {
    // One Gauss-Newton step of the MAP cost |r(d)|^2_cov + |d|^2_P with R = R_pred Exp(d) and
    // r(d) = Log((R_pred Exp(d))^T dR) ~ e - Jl^-1(e) d
    Eigen::Vector3d e  = geometry::log_so3(R_pred.transpose() * dR);
    Eigen::Matrix3d H  = geometry::so3_leftJacobian(e).inverse();
    Eigen::Matrix3d K  = P * H.transpose() * (H * P * H.transpose() + cov_dR).inverse();
    Eigen::Matrix3d Ru = R_pred * geometry::exp_so3(K * e);
    P                  = (Eigen::Matrix3d::Identity() - K * H) * P;
    return Ru;
}

Eigen::Vector3d ESKFEstimator::updateVelocity(const Eigen::Matrix3d &R_w_f1,
                                              const Eigen::Vector3d &v_pred,
                                              const Eigen::Vector3d &v1,
                                              const Eigen::Vector3d &dV,
                                              const Eigen::Matrix3d &cov_dv,
                                              double dt) {
    // The innovation lives in frame 1's axes and the state in world axes: H = R_w_f1^T, P = I
    Eigen::Matrix3d H    = R_w_f1.transpose();
    Eigen::Vector3d errv = dV - H * (v_pred - v1 - g * dt);
    Eigen::Matrix3d K    = H.transpose() * (H * H.transpose() + cov_dv).inverse();
    return v_pred + K * errv;
}

bool isae::ESKFEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                                   const std::shared_ptr<Frame> &frame2,
                                                   vec_match &matches,
                                                   Eigen::Affine3d &dT,
                                                   Eigen::MatrixXd &covdT) {
    if (matches.size() < 5)
        return false;

    // Get matched features from frame 1 with existing 3D landmarks
    std::vector<Eigen::Vector3d> p3d_vector;
    p3d_vector.reserve(matches.size());
    std::vector<Eigen::Vector2d> p2d_vector;
    p2d_vector.reserve(matches.size());

    vec_match init_matches, noninit_matches;
    init_matches.reserve(matches.size());
    noninit_matches.reserve(matches.size());

    Eigen::Affine3d T_cam1_w = matches.at(0).first->getSensor()->getWorld2SensorTransform();
    for (auto &m : matches) {
        if (m.first->getLandmark().lock()) {

            // Ignore non initialized landmarks to keep them in track
            if (!m.first->getLandmark().lock()->isInitialized()) {
                noninit_matches.push_back(m);
                continue;
            }
            init_matches.push_back(m);

            // Get p3d in camera one frame
            Eigen::Vector3d t_w_lmk    = m.first->getLandmark().lock()->getPose().translation();
            Eigen::Vector3d t_cam1_lmk = T_cam1_w * t_w_lmk;
            p3d_vector.push_back(t_cam1_lmk);

            // Get corresponding detection in homogeneous coordinates in frame 2
            Eigen::Vector3d ray_cam2 = m.second->getBearingVectors().at(0);
            p2d_vector.push_back(Eigen::Vector2d(ray_cam2.x() / ray_cam2.z(), ray_cam2.y() / ray_cam2.z()));
        }
    }

    Eigen::Matrix3d intrinsic     = Eigen::Matrix3d::Identity();
    Eigen::Matrix2d R             = 0.1 * Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
    P.block(0, 0, 3, 3)           = Eigen::Matrix3d::Identity();
    P.block(3, 3, 3, 3)           = Eigen::Matrix3d::Identity();

    // Perform a first update with IMU in VIO case
    if (frame2->getIMU()) {
        if (frame1 == frame2->getIMU()->getLastKF()) {

            double dt          = (frame2->getTimestamp() - frame1->getTimestamp()) * 1e-9;
            Eigen::Matrix3d R1 = frame1->getFrame2WorldTransform().rotation();

            // Update velocity
            Eigen::Vector3d v_cst =
                (frame2->getFrame2WorldTransform().translation() - frame1->getFrame2WorldTransform().translation()) /
                dt;
            Eigen::Vector3d vu = updateVelocity(R1,
                                                v_cst,
                                                frame1->getIMU()->getVelocity(),
                                                frame2->getIMU()->getDeltaV(),
                                                frame2->getIMU()->getCov().block(3, 3, 3, 3),
                                                dt);
            frame2->getIMU()->setVelocity(vu);

            // Update translation
            Eigen::Vector3d deltaP_est = dT.translation() - R1.transpose() * frame1->getIMU()->getVelocity() * dt -
                                         0.5 * R1.transpose() * g * dt * dt;
            Eigen::Vector3d errt = frame2->getIMU()->getDeltaP() - deltaP_est;
            Eigen::Matrix3d Kt   = P.block(3, 3, 3, 3) *
                                 (P.block(3, 3, 3, 3) + 1000 * frame2->getIMU()->getCov().block(6, 6, 3, 3)).inverse();
            dT.translation()    = dT.translation() + Kt * errt;
            P.block(3, 3, 3, 3) = (Eigen::Matrix3d::Identity() - Kt) * P.block(3, 3, 3, 3);

            // Update rotation (convention e = DeltaR ominus R)
            Eigen::Matrix3d P_rot         = P.block(0, 0, 3, 3);
            dT.affine().block(0, 0, 3, 3) = updateRotation(
                dT.linear(), frame2->getIMU()->getDeltaR(), P_rot, 1000 * frame2->getIMU()->getCov().block(0, 0, 3, 3));
            P.block(0, 0, 3, 3) = P_rot;
        }
    }

    // Init the transformation
    Eigen::Affine3d T_cam1_f1   = matches.at(0).first->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_f2   = matches.at(0).second->getSensor()->getFrame2SensorTransform();
    Eigen::Affine3d T_cam2_cam1 = T_cam2_f2 * dT.inverse() * T_cam1_f1.inverse();

    // The IMU step's covariance is expressed on the error of dT: move it to the error of T_cam2_cam1
    const Eigen::Matrix<double, 6, 6> J_imu_cam = imuToCameraErrorJacobian(dT, T_cam1_f1);
    P                                           = J_imu_cam * P * J_imu_cam.transpose();

    // We estimate T_cam2_cam1 using the ESKF
    for (uint i = 0; i < p3d_vector.size(); ++i) {

        // Projection
        Eigen::Vector2d proj;
        Eigen::MatrixXd J_T, J_p;
        std::tie(proj, J_T, J_p) = jac_projection(
            intrinsic, p3d_vector.at(i), T_cam2_cam1.linear(), T_cam2_cam1.translation(), Eigen::Vector3d::Zero());
        Eigen::Vector2d err = p2d_vector[i] - proj;

        // Kalman Equations
        Eigen::Matrix<double, 6, 2> K  = P * J_T.transpose() * (J_T * P * J_T.transpose() + R).inverse();
        Eigen::Matrix<double, 6, 1> dx = K * err;
        P                              = (Eigen::Matrix<double, 6, 6>::Identity() - K * J_T) * P;

        // Update
        Eigen::Affine3d dtau            = Eigen::Affine3d::Identity();
        dtau.translation()              = dx.bottomRows<3>();
        dtau.affine().block(0, 0, 3, 3) = geometry::exp_so3(dx.topRows<3>());
        T_cam2_cam1                     = T_cam2_cam1 * dtau;
    }

    // Covariance of the right perturbation dT E(xi) of the estimate (order: rotation, translation)
    const Eigen::Matrix<double, 6, 6> J_cam_dT = cameraToFrameErrorJacobian(T_cam2_cam1, T_cam2_f2);
    covdT                                      = J_cam_dT * P * J_cam_dT.transpose();
    dT                                         = T_cam1_f1.inverse() * T_cam2_cam1.inverse() * T_cam2_f2;

    return true;
}

bool ESKFEstimator::estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                             const std::shared_ptr<Frame> &frame2,
                                             typed_vec_match &typed_matches,
                                             Eigen::Affine3d &dT,
                                             Eigen::MatrixXd &covdT) {
    return false;
}

bool ESKFEstimator::refineTriangulation(std::shared_ptr<Frame> &frame) {

    // Iterated Kalman update (Gauss-Newton on the prior + all observations) of each initialized landmark of the frame,
    // with a 1.5 px measurement noise on the normalized image plane. The single update with a 0.1 (normalized,
    // ~40 px) noise it replaced barely moved the landmark.
    typed_vec_landmarks landmarks = frame->getLandmarks();
    for (auto &landmark_list : landmarks) {
        for (auto &landmark : landmark_list.second) {
            if (!landmark->isInitialized() || landmark->isOutlier())
                continue;

            const Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Identity();
            const Eigen::Matrix3d P0        = Eigen::Matrix3d::Identity();
            const Eigen::Vector3d t_prior   = landmark->getPose().translation();
            Eigen::Vector3d t_w_lmk         = t_prior;

            // Observations still attached to a sensor
            std::vector<std::shared_ptr<AFeature>> feats;
            for (std::weak_ptr<AFeature> &wfeature : landmark->getFeatures()) {
                std::shared_ptr<AFeature> f = wfeature.lock();
                if (f && f->getSensor())
                    feats.push_back(f);
            }
            if (feats.empty())
                continue;

            for (int it = 0; it < 5; it++) {
                Eigen::VectorXd err = Eigen::VectorXd::Zero(2 * feats.size());
                Eigen::MatrixXd J_p = Eigen::MatrixXd::Zero(2 * feats.size(), 3);
                Eigen::VectorXd r_var(2 * feats.size());
                for (size_t k = 0; k < feats.size(); ++k) {
                    std::shared_ptr<ImageSensor> cam = feats[k]->getSensor();
                    Eigen::Affine3d T_s_w            = cam->getWorld2SensorTransform();
                    Eigen::Vector3d ray_cam          = feats[k]->getBearingVectors().at(0);
                    Eigen::Vector2d ray_cam_h(ray_cam.x() / ray_cam.z(), ray_cam.y() / ray_cam.z());
                    Eigen::Vector2d proj;
                    Eigen::MatrixXd J_T, J_lmk;
                    std::tie(proj, J_T, J_lmk) = jac_projection(
                        intrinsic, t_w_lmk, T_s_w.linear(), T_s_w.translation(), Eigen::Vector3d::Zero());
                    err.segment<2>(2 * k)     = ray_cam_h - proj;
                    J_p.block<2, 3>(2 * k, 0) = J_lmk;
                    const double sigma        = 1.5 / cam->getFocal();
                    r_var.segment<2>(2 * k)   = Eigen::Vector2d::Constant(sigma * sigma);
                }
                // MAP step around the current estimate: prior N(t_prior, P0), observations linearized here
                const Eigen::MatrixXd W = r_var.cwiseInverse().asDiagonal();
                const Eigen::Matrix3d H = J_p.transpose() * W * J_p + P0.inverse();
                const Eigen::Vector3d g = J_p.transpose() * W * err - P0.inverse() * (t_w_lmk - t_prior);
                const Eigen::Vector3d dx = H.ldlt().solve(g);
                t_w_lmk += dx;
                if (dx.norm() < 1e-9)
                    break;
            }

            // Update the landmark and check if it is valid
            landmark->setPosition(t_w_lmk);
            landmark->sanityCheck();
        }
    }

    return true;
}
} // namespace isae