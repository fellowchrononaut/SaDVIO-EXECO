#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/estimator/ESKFEstimator.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/residuals.hpp"
#include <gtest/gtest.h>

#include "jacobian_check.h"

namespace isae {

class ResidualTest : public testing::Test {
  public:
    void SetUp() override {
        std::srand(12345u); // same random state for every test, whatever the run order
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame1 = std::shared_ptr<Frame>(new Frame());

        // Generates a random pose + a small displacement
        Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
        Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
        Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
        T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
        T_f0_w.translation()              = t_rand;
        _dT                               = Eigen::Affine3d::Identity();
        _dT.translation()                 = Eigen::Vector3d(0.05, 0.05, 0);

        // Set Intrinsics
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;
        _sensor0 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor1 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));

        // Init frames
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0);
        _frame0->init(sensors_frame0, 0);
        _frame0->setWorld2FrameTransform(T_f0_w);
        _sensor0->setFrame2SensorTransform(Eigen::Affine3d::Identity());

        std::vector<std::shared_ptr<ImageSensor>> sensors_frame1;
        sensors_frame1.push_back(_sensor1);
        _frame1->init(sensors_frame1, 0);
        _frame1->setWorld2FrameTransform(_dT.inverse() * T_f0_w);
        _sensor1->setFrame2SensorTransform(Eigen::Affine3d::Identity());

        // Init a random landmark in the FOV
        _rand_lmk               = Eigen::Affine3d::Identity();
        _rand_lmk.translation() = Eigen::Vector3d::Random();

        while (!_sensor0->project(
                   _rand_lmk, _frame0->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), _p2d0, NULL, NULL) ||
               !_sensor1->project(
                   _rand_lmk, _frame1->getWorld2FrameTransform(), Eigen::Matrix2d::Identity(), _p2d1, NULL, NULL)) {
            _rand_lmk.translation() = Eigen::Vector3d::Random();
        }

        // Init features
        std::vector<Eigen::Vector2d> p2d0, p2d1;
        p2d0.push_back(_p2d0);
        p2d1.push_back(_p2d1);
        std::shared_ptr<AFeature> f0 = std::shared_ptr<AFeature>(new Point2D(p2d0));
        _sensor0->addFeature("pointxd", f0);
        f0->computeBearingVectors();
        std::shared_ptr<AFeature> f1 = std::shared_ptr<AFeature>(new Point2D(p2d1));
        _sensor1->addFeature("pointxd", f1);
        f1->computeBearingVectors();

        // Init landmark
        _lmk = std::shared_ptr<Point3D>(new Point3D());
        std::vector<std::shared_ptr<AFeature>> features;
        features.push_back(f0);
        features.push_back(f1);

        _lmk->init(_rand_lmk, features);
        _frame0->addLandmark(_lmk);
        _frame1->addLandmark(_lmk);
    }

    std::shared_ptr<Frame> _frame0, _frame1;
    std::shared_ptr<Camera> _sensor0, _sensor1;
    Eigen::Affine3d _rand_lmk, _dT;
    Eigen::Matrix3d _K;
    Eigen::Vector2d _p2d0, _p2d1;
    std::shared_ptr<ALandmark> _lmk;
};

TEST_F(ResidualTest, PriorResidual) {

    // Generates a random pose
    Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_f0_w.translation()              = t_rand;

    // Create a Cost function
    ceres::CostFunction *cost_fct =
        new PosePriordx(Eigen::Affine3d::Identity(), T_f0_w, 100 * Vector6d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX.values());

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));
}

TEST_F(ResidualTest, reprojTest) {

    // Create a Cost function
    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(_p2d0, _sensor0, _rand_lmk, 1);

    // Set variables
    // ceres::Manifold *nullptr = new SE3RightParameterization();
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dlmk.values());

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));
}

TEST_F(ResidualTest, angularTest) {

    // Get depth
    Eigen::Vector3d t_s0_lmk = _sensor0->getWorld2SensorTransform() * _rand_lmk.translation();
    double depth             = t_s0_lmk.norm();

    // Get bearing vectors
    Eigen::Vector3d b0 = _sensor0->getRayCamera(_p2d0);
    Eigen::Vector3d b1 = _sensor1->getRayCamera(_p2d1);

    // Create a Cost function
    ceres::CostFunction *cost_fct = new AngularErrCeres_pointxd_depth(b1,
                                                                      b0,
                                                                      Eigen::Affine3d::Identity(),
                                                                      _frame0->getWorld2FrameTransform(),
                                                                      _frame1->getWorld2FrameTransform(),
                                                                      depth,
                                                                      1);

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PoseParametersBlock dXa(Eigen::Affine3d::Identity());
    double ddepth[1] = {0.0};

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dXa.values());
    parameters_blocks.push_back(ddepth);

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));
    ASSERT_NEAR(isae_test::ResidualNorm(*cost_fct, parameters_blocks), 0, 1e-5);
}

TEST_F(ResidualTest, scaleTest) {

    Eigen::Vector3d b1      = _sensor1->getRayCamera(_p2d1);
    Eigen::Affine3d T_c0_c0 = Eigen::Affine3d::Identity();

    // Create a Cost function
    ceres::CostFunction *cost_fct =
        new AngularErrorScaleCam0(b1, _rand_lmk.translation(), _sensor0->getWorld2SensorTransform(), _dT, T_c0_c0, 1);

    // Set variables
    double lambda[1] = {1.0};
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(lambda);
    parameters_blocks.push_back(dlmk.values());

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));

    // Check the cost function with scales
    Eigen::Vector3d t_s_lmk = _sensor0->getWorld2SensorTransform() * _rand_lmk.translation();
    double depth            = t_s_lmk.norm();
    Eigen::Vector3d b0      = _sensor0->getRayCamera(_p2d0);

    // Check if the bearing vector * depth gives the actual landmark pose
    ASSERT_NEAR((t_s_lmk - b0 * depth).norm(), 0, 1e-5);

    ceres::CostFunction *cost_fct1 = new AngularErrorScaleDepth(b1, b0, _dT, T_c0_c0, depth, 1);

    // Set variables
    double ddepth[1] = {0.0};

    std::vector<double *> parameters_blocks1;
    parameters_blocks1.push_back(lambda);
    parameters_blocks1.push_back(ddepth);

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct1, parameters_blocks1));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct1, parameters_blocks1, 0.05));
    ASSERT_NEAR(isae_test::ResidualNorm(*cost_fct1, parameters_blocks1), 0, 1e-5);
}

TEST_F(ResidualTest, PoseToLandmarkResidual) {

    // Generates a random pose
    Eigen::Affine3d T_f0_w            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_f0_w.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_f0_w.translation()              = t_rand;

    // Generates a random lmk
    Eigen::Vector3d t_w_lmk  = Eigen::Vector3d::Random();
    Eigen::Vector3d t_f0_lmk = T_f0_w * t_w_lmk;

    // Create a Cost function
    ceres::CostFunction *cost_fct = new PoseToLandmarkFactor(
        t_f0_lmk, T_f0_w, t_w_lmk + 0.01 * Eigen::Vector3d::Random(), 100 * Eigen::Vector3d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dl(Eigen::Vector3d::Zero());

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX.values());
    parameters_blocks.push_back(dl.values());

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));
}

TEST_F(ResidualTest, RelativePose6DResidual) {

    // Generate random poses
    Eigen::Affine3d T_w_f0            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q_rand         = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t_rand            = Eigen::Vector3d::Random();
    T_w_f0.affine().block(0, 0, 3, 3) = q_rand.toRotationMatrix();
    T_w_f0.translation()              = t_rand;

    Eigen::Affine3d T_w_f1            = Eigen::Affine3d::Identity();
    Eigen::Quaterniond q1_rand        = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t1_rand           = Eigen::Vector3d::Random();
    T_w_f1.affine().block(0, 0, 3, 3) = q1_rand.toRotationMatrix();
    T_w_f1.translation()              = t1_rand;

    // Create a Cost function
    Eigen::Affine3d T_f0_f1       = T_w_f0.inverse() * T_w_f1;
    ceres::CostFunction *cost_fct = new Relative6DPose(T_w_f0, T_w_f1, T_f0_f1, Vector6d::Ones().asDiagonal());

    // Set variables
    PoseParametersBlock dX_f0(Eigen::Affine3d::Identity());
    PoseParametersBlock dX_f1(Eigen::Affine3d::Identity());
    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX_f0.values());
    parameters_blocks.push_back(dX_f1.values());

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));
}

TEST(ESKFVelocityTest, updateVelocityFrames) {

    // Issue 11: the preintegrated velocity delta is expressed in frame 1's axes; the update must
    // map the innovation back to world axes with R_w_f1, whatever the orientation.
    const double dt          = 0.1;
    const Eigen::Vector3d v1(0.4, -0.1, 0.2), v2_true(0.6, 0.3, -0.1), err(0.3, -0.2, 0.1);
    const std::vector<Eigen::Vector3d> rots = {Eigen::Vector3d(0, 0, M_PI / 2), Eigen::Vector3d(0.3, -1.1, 0.7),
                                               Eigen::Vector3d(2.0, 0.5, -0.4)};
    for (const auto &w : rots) {
        const Eigen::Matrix3d R  = geometry::exp_so3(w);
        const Eigen::Vector3d dV = R.transpose() * (v2_true - v1 - g * dt);

        // Accurate measurement: the update recovers the true velocity
        Eigen::Vector3d vu =
            ESKFEstimator::updateVelocity(R, v2_true + err, v1, dV, 1e-8 * Eigen::Matrix3d::Identity(), dt);
        EXPECT_LT((vu - v2_true).norm(), 1e-6) << "rotation " << w.transpose();

        // Anisotropic noise: only the error along the poorly measured body axis survives,
        // vu - v2 = R diag(c / (1 + c)) R^T err
        const Eigen::Vector3d c(1e-8, 1.0, 1e-8);
        vu = ESKFEstimator::updateVelocity(R, v2_true + err, v1, dV, c.asDiagonal(), dt);
        const Eigen::Vector3d kept = R * (c.array() / (1 + c.array())).matrix().asDiagonal() * R.transpose() * err;
        EXPECT_LT((vu - v2_true - kept).norm(), 1e-6) << "rotation " << w.transpose();
    }
}

namespace {
// Right difference of two poses: T2 = T1 E(theta, rho) with E(theta, rho) = [Exp(theta), rho]
Vector6d rightDiff(const Eigen::Affine3d &T1, const Eigen::Affine3d &T2) {
    Vector6d xi;
    xi.head<3>() = geometry::log_so3(T1.rotation().transpose() * T2.rotation());
    xi.tail<3>() = T1.rotation().transpose() * (T2.translation() - T1.translation());
    return xi;
}

Eigen::Affine3d makePose(const Eigen::Vector3d &w, const Eigen::Vector3d &t) {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear()        = geometry::exp_so3(w);
    T.translation()   = t;
    return T;
}
} // namespace

TEST(ESKFCovarianceTest, errorJacobiansMatchFiniteDifferences) {

    // The ESKF carries its covariance from the IMU step (error of dT = T_f1_f2: R Exp(dtheta), t + dt) to the
    // visual step (right perturbation of T_cam2_cam1 = T_cam2_f2 dT^-1 T_cam1_f1^-1) and returns it on the right perturbation of dT
    const Eigen::Affine3d T_cam1_f1 = makePose(Eigen::Vector3d(0.1, -1.5, 0.3), Eigen::Vector3d(0.05, -0.02, 0.1));
    const Eigen::Affine3d T_cam2_f2 = makePose(Eigen::Vector3d(-0.2, 1.2, 0.4), Eigen::Vector3d(-0.11, 0.03, 0.02));
    const Eigen::Affine3d dT        = makePose(Eigen::Vector3d(0.3, 0.2, -0.5), Eigen::Vector3d(0.4, -0.3, 0.8));
    const Eigen::Affine3d X         = T_cam2_f2 * dT.inverse() * T_cam1_f1.inverse();
    const double eps                = 1e-6;

    const Eigen::Matrix<double, 6, 6> J_ic = ESKFEstimator::imuToCameraErrorJacobian(dT, T_cam1_f1);
    const Eigen::Matrix<double, 6, 6> J_cf = ESKFEstimator::cameraToFrameErrorJacobian(X, T_cam2_f2);
    for (int k = 0; k < 6; k++) {
        Vector6d d = Vector6d::Zero();
        d(k)       = eps;

        Eigen::Affine3d dT_p = dT;
        dT_p.linear()        = dT.linear() * geometry::exp_so3(d.head<3>());
        dT_p.translation()   = dT.translation() + d.tail<3>();
        const Eigen::Affine3d X_p = T_cam2_f2 * dT_p.inverse() * T_cam1_f1.inverse();
        EXPECT_LT((rightDiff(X, X_p) / eps - J_ic.col(k)).cwiseAbs().maxCoeff(), 1e-5) << "imu->cam column " << k;

        Eigen::Affine3d E = Eigen::Affine3d::Identity();
        E.linear()        = geometry::exp_so3(d.head<3>());
        E.translation()   = d.tail<3>();
        const Eigen::Affine3d dT_from_X = T_cam1_f1.inverse() * (X * E).inverse() * T_cam2_f2;
        EXPECT_LT((rightDiff(dT, dT_from_X) / eps - J_cf.col(k)).cwiseAbs().maxCoeff(), 1e-5)
            << "cam->frame column " << k;
    }
}

TEST(ESKFCovarianceTest, rotationUpdateIsOneGaussNewtonStep) {

    // MAP cost |r(d)|^2_C + |d|^2_P with r(d) = Log((R_pred Exp(d))^T dR): one Gauss-Newton step from d = 0 with
    // the finite-difference Jacobian of r must give the ESKF update, for a large innovation and anisotropic P, C
    const Eigen::Matrix3d R_pred = geometry::exp_so3(Eigen::Vector3d(0.4, -0.7, 0.2));
    const Eigen::Matrix3d dR     = R_pred * geometry::exp_so3(Eigen::Vector3d(0.25, 0.1, -0.3));
    const Eigen::Vector3d p(1.0, 0.3, 2.0), c(0.2, 0.05, 0.6);
    const Eigen::Matrix3d Q = geometry::exp_so3(Eigen::Vector3d(0.3, 0.1, 0.5));
    const Eigen::Matrix3d P = Q * p.asDiagonal() * Q.transpose();
    const Eigen::Matrix3d C = Q.transpose() * c.asDiagonal() * Q;

    auto r = [&](const Eigen::Vector3d &d) {
        return Eigen::Vector3d(geometry::log_so3((R_pred * geometry::exp_so3(d)).transpose() * dR));
    };
    Eigen::Matrix3d Jr_num;
    for (int k = 0; k < 3; k++) {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        d(k)              = 1e-7;
        Jr_num.col(k)     = (r(d) - r(-d)) / 2e-7;
    }
    const Eigen::Vector3d e    = r(Eigen::Vector3d::Zero());
    const Eigen::Matrix3d Cinv = C.inverse();
    const Eigen::Vector3d d_gn =
        -(Jr_num.transpose() * Cinv * Jr_num + P.inverse()).inverse() * Jr_num.transpose() * Cinv * e;

    Eigen::Matrix3d P_upd = P;
    const Eigen::Matrix3d R_upd = ESKFEstimator::updateRotation(R_pred, dR, P_upd, C);
    EXPECT_LT(geometry::log_so3((R_pred * geometry::exp_so3(d_gn)).transpose() * R_upd).norm(), 1e-6);

    // Posterior covariance = inverse of the Gauss-Newton Hessian
    const Eigen::Matrix3d P_gn = (Jr_num.transpose() * Cinv * Jr_num + P.inverse()).inverse();
    EXPECT_LT((P_upd - P_gn).cwiseAbs().maxCoeff(), 1e-6);
}

TEST_F(ResidualTest, ESKFLmkTest) {
    ESKFEstimator estimator;

    Eigen::Vector3d t_lmk       = _lmk->getPose().translation();
    Eigen::Vector3d t_lmk_noise = _lmk->getPose().translation() + 0.1 * Eigen::Vector3d::Random();
    _lmk->setPosition(t_lmk_noise);
    estimator.refineTriangulation(_frame1);
    ASSERT_NEAR((_lmk->getPose().translation() - t_lmk).norm(), 0, 0.02);
}

// Issue 19: residuals of the online camera-IMU time offset estimation
TEST_F(ResidualTest, timeOffsetResidualJacobians) {
    std::shared_ptr<imu_config> cfg = std::make_shared<imu_config>();
    cfg->gyr_noise = cfg->acc_noise = cfg->bgyr_noise = cfg->bacc_noise = 0.01;
    cfg->rate_hz                                                         = 200;
    std::shared_ptr<IMU> imu = std::make_shared<IMU>(cfg, Eigen::Vector3d(0, 0, 9.81), Eigen::Vector3d(0.3, -0.5, 0.8));
    imu->setVelocity(Eigen::Vector3d(0.6, -0.2, 0.4));

    // Angular error with time offset (AngularAnalytic optimizer)
    Eigen::Vector3d b0 = _sensor0->getRayCamera(_p2d0);
    ceres::CostFunction *ang = new AngularErrCeres_pointxd_td(b0,
                                                              imu,
                                                              _sensor0->getFrame2SensorTransform(),
                                                              _frame0->getWorld2FrameTransform(),
                                                              _rand_lmk.translation(),
                                                              1.0 / _sensor0->getFocal());
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());
    double td[1] = {0.01};
    std::vector<double *> params = {dX.values(), dlmk.values(), td};
    EXPECT_TRUE(isae_test::JacobiansMatch(*ang, params, 1e-5));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*ang, params, 0.02, 5, 1e-5));

    // Reprojection error with time offset (Analytic optimizer)
    ceres::CostFunction *rep = new ReprojectionErrCeres_pointxd_dx_td(_p2d0, _sensor0, imu, _rand_lmk, 1);
    td[0] = 0.01;
    EXPECT_TRUE(isae_test::JacobiansMatch(*rep, params, 1e-5));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*rep, params, 0.02, 6, 1e-5));

    // The parameter is the absolute offset: a frame read with offset a and parameter a + x behaves like a frame
    // read with offset 0 and parameter x
    const double a = 0.007, x = 0.004;
    AngularErrCeres_pointxd_td ang_a(b0, imu, _sensor0->getFrame2SensorTransform(), _frame0->getWorld2FrameTransform(),
                                     _rand_lmk.translation(), 1.0 / _sensor0->getFocal(), a);
    ReprojectionErrCeres_pointxd_dx_td rep_a(_p2d0, _sensor0, imu, _rand_lmk, 1, a);
    td[0] = a + x;
    Eigen::Vector2d r_ang_a, r_rep_a, r_ang, r_rep;
    ang_a.Evaluate(params.data(), r_ang_a.data(), nullptr);
    rep_a.Evaluate(params.data(), r_rep_a.data(), nullptr);
    td[0] = x;
    ang->Evaluate(params.data(), r_ang.data(), nullptr);
    rep->Evaluate(params.data(), r_rep.data(), nullptr);
    EXPECT_NEAR((r_ang_a - r_ang).norm(), 0, 1e-12);
    EXPECT_NEAR((r_rep_a - r_rep).norm(), 0, 1e-12);
    delete ang;
    delete rep;
}

// The reprojection-error optimizers (Analytic, Numeric) need the projection Jacobians of every camera model;
// the double-sphere one was a stub returning false (no visual constraint with fisheye rigs such as TUM-VI)
TEST_F(ResidualTest, reprojTestDoubleSphere) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 158.3;
    K(1, 1) = 158.3;
    K(0, 2) = 255.0;
    K(1, 2) = 256.9;
    std::shared_ptr<DoubleSphere> ds = std::make_shared<DoubleSphere>(cv::Mat::zeros(512, 512, CV_8U), K, 0.593, -0.172);
    std::vector<std::shared_ptr<ImageSensor>> sensors = {ds};
    std::shared_ptr<Frame> f = std::make_shared<Frame>();
    f->init(sensors, 0);
    ds->setFrame2SensorTransform(Eigen::Affine3d::Identity());
    Eigen::Affine3d T_f_w = Eigen::Affine3d::Identity();
    T_f_w.linear()        = geometry::exp_so3(Eigen::Vector3d(0.1, -0.2, 0.05));
    f->setWorld2FrameTransform(T_f_w);

    // A landmark well inside the field of view, observed with some noise
    Eigen::Affine3d T_w_lmk = Eigen::Affine3d::Identity();
    T_w_lmk.translation()   = T_f_w.inverse() * Eigen::Vector3d(0.8, -0.5, 2.0);
    Eigen::Vector2d p2d;
    ASSERT_TRUE(ds->project(T_w_lmk, T_f_w, Eigen::Matrix2d::Identity(), p2d, NULL, NULL));

    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(p2d + Eigen::Vector2d(1.5, -0.7), ds, T_w_lmk, 1);
    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dlmk(Eigen::Vector3d::Zero());
    std::vector<double *> parameters_blocks = {dX.values(), dlmk.values()};
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks, 1e-5));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.02, 7, 1e-5));
    delete cost_fct;
}

} // namespace isae