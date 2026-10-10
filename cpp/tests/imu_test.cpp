#include <fstream>
#include <gtest/gtest.h>

#include "jacobian_check.h"
#include <random>

#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/dataproviders/adataprovider.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESNumeric.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/sensors/Camera.h"
#include <functional>

namespace isae {

void read_line_euroc(std::string line, Eigen::Affine3d &pose, Eigen::Vector3d &v, double &ts) {
    std::istringstream s(line);
    std::string svalue;
    std::string::size_type sz;

    std::vector<double> values;
    while (getline(s, svalue, ',')) {
        values.push_back(std::stod(svalue, &sz)); // convert to double
    }

    // Deal with pose
    Eigen::Quaterniond q(values[4], values[5], values[6], values[7]);
    Eigen::Matrix3d R = q.toRotationMatrix();

    pose                            = Eigen::Affine3d::Identity();
    pose.affine().block(0, 0, 3, 3) = R;
    pose.translation()              = Eigen::Vector3d(values[1], values[2], values[3]);

    // Deal with ts (in seconds)
    ts = values[0] * 1e-9;

    // Deal with velocity
    v = Eigen::Vector3d(values[8], values[9], values[10]);
}

void write_result(std::shared_ptr<Frame> f) {

    // Write in a txt file for evaluation
    std::ofstream fw_res("result.txt", std::ofstream::out | std::ofstream::app);
    const Eigen::Matrix3d R = f->getFrame2WorldTransform().rotation();
    Eigen::Vector3d twc     = f->getFrame2WorldTransform().translation();
    fw_res << f->getTimestamp() << " " << 0 << " " << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << twc.x()
           << " " << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << twc.y() << " " << R(2, 0) << " " << R(2, 1)
           << " " << R(2, 2) << " " << twc.z() << "\n";
    fw_res.close();
}

void write_imu_data(double ts, Eigen::Vector3d acc) {

    // Write in a txt file for evaluation
    std::ofstream fw_res("acc.txt", std::ofstream::out | std::ofstream::app);
    fw_res << ts << "," << acc(0) << "," << acc(1) << "," << acc(2) << "\n";
    fw_res.close();
}

class ImuTest : public testing::Test {
  public:
    void SetUp() override {
        std::srand(12345u); // same random state for every test, whatever the run order
        // Set Imu Config
        _imu_cfg             = std::shared_ptr<imu_config>(new imu_config());
        _imu_cfg->gyr_noise  = (0.5 * M_PI) / (180 * 60);
        _imu_cfg->bgyr_noise = 1.9393e-05;
        _imu_cfg->acc_noise  = 0.1 / 60;
        _imu_cfg->bacc_noise = 3.0000e-3;
        _imu_cfg->rate_hz    = 200;
        _imu_cfg->T_s_f      = Eigen::Affine3d::Identity();
        _acc                 = Eigen::Vector3d(0.5, 1.0, 10.81);
        _gyr                 = Eigen::Vector3d(0.1, 0.3, 0.1);

        // Set Frames and IMU
        _imu0   = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame0->init(_imu0, 1e9);
        _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
        _frame0->setKeyFrame();
        _imu0->setLastKF(_frame0);

        _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _imu1->setLastIMU(_imu0);
        _imu1->setLastKF(_frame0);
        _frame1 = std::shared_ptr<Frame>(new Frame());
        _frame1->init(_imu1, 1.5e9);

        _imu2 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
        _imu2->setLastIMU(_imu1);
        _imu2->setLastKF(_frame0);
        _frame2 = std::shared_ptr<Frame>(new Frame());
        _frame2->init(_imu2, 2e9);
    }

    std::shared_ptr<imu_config> _imu_cfg;
    Eigen::Vector3d _acc;
    Eigen::Vector3d _gyr;
    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<IMU> _imu0;
    std::shared_ptr<Frame> _frame1;
    std::shared_ptr<IMU> _imu1;
    std::shared_ptr<Frame> _frame2;
    std::shared_ptr<IMU> _imu2;
};

TEST_F(ImuTest, ImuTestBase) {

    // Check measurements
    ASSERT_EQ(_imu0->getAcc(), Eigen::Vector3d(0.5, 1.0, 10.81));
    ASSERT_EQ(_imu0->getGyr(), Eigen::Vector3d(0.1, 0.3, 0.1));

    // Check preintegration measurement (without biases)
    _imu1->processIMU();
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr / 2);
    ASSERT_EQ((dR * _imu1->getDeltaR().transpose()).trace(), 3);

    Eigen::Vector3d dv = _acc / 2;
    ASSERT_EQ((dv - _imu1->getDeltaV()).norm(), 0);

    Eigen::Vector3d dp = 0.5 * _acc * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu1->getDeltaP()).norm(), 0);

    // Check pose estimation
    Eigen::Affine3d T_f0_f1;
    _imu1->estimateTransformIMU(T_f0_f1);
    ASSERT_EQ((dR * T_f0_f1.rotation().transpose()).trace(), 3);
    Eigen::Vector3d t_f0_f1 = dp + 0.5 * g * 0.5 * 0.5;
    ASSERT_EQ((T_f0_f1.translation() - t_f0_f1).norm(), 0);

    // Check preintegration measurement with biases
    Eigen::Vector3d ba(0.1, 0.2, 0.3);
    Eigen::Vector3d bg(0.2, 0.3, 0.1);
    _imu0->setBa(ba);
    _imu0->setBg(bg);
    _imu1->processIMU();

    // Check preintegration measurement (without biases)
    dR = geometry::exp_so3((_gyr - bg) * 0.5);
    ASSERT_EQ((dR * _imu1->getDeltaR().transpose()).trace(), 3);

    dv = (_acc - ba) / 2;
    ASSERT_EQ((dv - _imu1->getDeltaV()).norm(), 0);

    dp = 0.5 * (_acc - ba) * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu1->getDeltaP()).norm(), 0);
}

TEST_F(ImuTest, ImuNewMeas) {

    // Check with two measurements
    _imu1->processIMU();
    _imu2->processIMU();

    // Check preintegration meas
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr);
    ASSERT_EQ((dR * _imu2->getDeltaR().transpose()).trace(), 3);
    Eigen::Vector3d dv = _acc * 0.5 + _imu1->getDeltaR() * _acc * 0.5;
    ASSERT_EQ((dv - _imu2->getDeltaV()).norm(), 0);
    Eigen::Vector3d dp =
        0.5 * _acc * 0.5 * 0.5 + _imu1->getDeltaV() * 0.5 + 0.5 * _imu1->getDeltaR() * _acc * 0.5 * 0.5;
    ASSERT_EQ((dp - _imu2->getDeltaP()).norm(), 0);
}

// Here we use the same tests as in gtsam:
// https://github.com/borglab/gtsam/blob/develop/gtsam/navigation/tests/testImuFactor.cpp

TEST_F(ImuTest, checkCov) {

    // Set Frame and IMU according to gtsam test values
    Eigen::Vector3d acc(0.1, 0.0, 0.0);
    Eigen::Vector3d gyr(M_PI / 100.0, 0.0, 0.0);
    _imu_cfg->rate_hz = 2;
    _imu0             = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    _frame0           = std::shared_ptr<Frame>(new Frame());
    _frame0->init(_imu0, 1e9);
    _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame0->setKeyFrame();

    _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, _acc, _gyr));
    _imu1->setLastIMU(_imu0);
    _imu1->setLastKF(_frame0);
    _frame1 = std::shared_ptr<Frame>(new Frame());
    _frame1->init(_imu1, 1.5e9);
    _imu1->processIMU();
    Eigen::MatrixXd expected_cov = Eigen::Matrix<double, 9, 9>::Zero();
    expected_cov << 1.0577e-08, 0, 0, 0, 0, 0, 0, 0, 0, //
        0, 1.0577e-08, 0, 0, 0, 0, 0, 0, 0,             //
        0, 0, 1.0577e-08, 0, 0, 0, 0, 0, 0,             //
        0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07, 0, 0,  //
        0, 0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07, 0,  //
        0, 0, 0, 0, 0, 1.38889e-06, 0, 0, 3.47222e-07,  //
        0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05, 0, 0,  //
        0, 0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05, 0,  //
        0, 0, 0, 0, 0, 3.47222e-07, 0, 0, 5.00868e-05;
    ASSERT_NEAR((expected_cov - _imu1->getCov()).trace(), 0, 1e-9);
}

TEST_F(ImuTest, accelerating) {
    const double a = 0.2, v = 50;

    // Set the initial pose
    Eigen::Vector3d initial_velocity(v, 0, 0);
    Eigen::Affine3d initial_pose;
    initial_pose.matrix() << 1, 0, 0, 10, //
        0, 1, 0, 20,                      //
        0, 0, 1, 0,                       //
        0, 0, 0, 1;

    // We integrate the IMU
    Eigen::Vector3d vel, gyr, acc, ba, bg;
    vel << v, 0.0, 0.0;
    gyr.setZero();
    acc << a, 0.0, 9.81;
    ba.setZero();
    bg.setZero();
    double dt         = 0.01;
    _imu_cfg->rate_hz = 100;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(initial_pose.inverse());
    cur_frame->setKeyFrame();
    cur_imu->setBa(ba);
    cur_imu->setBg(bg);
    cur_imu->setVelocity(vel);
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 301; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Set last imu
        last_imu = cur_imu;
    }

    // Save predictions
    Eigen::MatrixXd prediction_cov = last_imu->getCov();
    Eigen::Vector3d prediction_r   = geometry::log_so3(last_imu->getDeltaR());
    Eigen::Vector3d prediction_v   = last_imu->getDeltaV();
    Eigen::Vector3d prediction_p   = last_imu->getDeltaP();

    // Compute covariance with MC
    std::vector<Eigen::Vector3d> v_vec, p_vec, r_vec;
    Eigen::Vector3d v_mean, p_mean, r_mean;
    r_mean = Eigen::Vector3d::Zero();
    v_mean = Eigen::Vector3d::Zero();
    p_mean = Eigen::Vector3d::Zero();

    int N = 100;
    for (int k = 0; k < N; k++) {

        // We create noisy IMU measurements
        // random device class instance, source of 'true' randomness for initializing random seed
        std::random_device rd;

        // Mersenne twister PRNG, initialized with seed from previous random device instance
        std::mt19937 gen(rd());
        std::normal_distribution<double> d_acc(0, _imu_cfg->acc_noise / std::sqrt(dt));
        std::normal_distribution<double> d_gyr(0, _imu_cfg->gyr_noise / std::sqrt(dt));
        Eigen::Vector3d acc_noise = Eigen::Vector3d(d_acc(gen), d_acc(gen), d_acc(gen));
        Eigen::Vector3d gyr_noise = Eigen::Vector3d(d_gyr(gen), d_gyr(gen), d_gyr(gen));

        // We set the first IMU
        std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc + acc_noise, gyr + gyr_noise));
        std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9);
        cur_frame->setWorld2FrameTransform(initial_pose.inverse());
        cur_frame->setKeyFrame();
        cur_imu->setBa(ba);
        cur_imu->setBg(bg);
        cur_imu->setVelocity(vel);
        std::shared_ptr<IMU> last_imu = cur_imu;

        std::shared_ptr<Frame> lastKF = cur_frame;

        // We integrate during 3 seconds
        for (int i = 1; i < 301; i++) {
            acc_noise = Eigen::Vector3d(d_acc(gen), d_acc(gen), d_acc(gen));
            gyr_noise = Eigen::Vector3d(d_gyr(gen), d_gyr(gen), d_gyr(gen));
            cur_imu   = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc + acc_noise, gyr + gyr_noise));
            cur_imu->setLastIMU(last_imu);
            cur_imu->setLastKF(lastKF);

            // Create frame and process IMU
            cur_frame = std::shared_ptr<Frame>(new Frame());
            cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
            cur_imu->processIMU();

            // Set last imu
            last_imu = cur_imu;
        }

        // Compute deltas and fill vectors for statistics
        Eigen::Vector3d v = last_imu->getDeltaV();
        Eigen::Vector3d p = last_imu->getDeltaP();
        Eigen::Vector3d r = geometry::log_so3(last_imu->getDeltaR());
        v_vec.push_back(v);
        p_vec.push_back(p);
        r_vec.push_back(r);
        v_mean += v;
        p_mean += p;
        r_mean += r;
    }

    // Compute statistics
    v_mean /= (double)N;
    p_mean /= (double)N;
    r_mean /= (double)N;
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(9, 9);
    for (int k = 0; k < N; k++) {
        Eigen::VectorXd xi   = Eigen::VectorXd::Zero(9);
        xi.block(0, 0, 3, 1) = r_vec.at(k) - prediction_r;
        xi.block(3, 0, 3, 1) = v_vec.at(k) - prediction_v;
        xi.block(6, 0, 3, 1) = p_vec.at(k) - prediction_p;
        Sigma += xi * xi.transpose();
    }

    Sigma /= (double)(N - 1);
    ASSERT_NEAR((Sigma - prediction_cov).trace(), 0, 1e-3);
}

TEST_F(ImuTest, checkJacobiansBiasGyr) {

    // First processing
    _imu1->processIMU();
    double dt                         = 0.5;
    Eigen::Matrix3d J_rk              = geometry::so3_rightJacobian((_gyr)*dt);
    Eigen::Matrix3d expected_J_dbg    = -J_rk * dt;
    Eigen::Matrix3d expected_J_dv_dba = -Eigen::Matrix3d::Identity() * dt;
    Eigen::Matrix3d expected_J_dv_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d expected_J_dp_dba = -0.5 * Eigen::Matrix3d::Identity() * dt * dt;
    Eigen::Matrix3d expected_J_dp_dbg = Eigen::Matrix3d::Zero();

    ASSERT_NEAR((_imu1->_J_dR_bg * expected_J_dbg.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu1->_J_dv_ba * expected_J_dv_dba.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu1->_J_dv_bg - expected_J_dv_dbg).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu1->_J_dp_ba - expected_J_dp_dba).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu1->_J_dp_bg - expected_J_dp_dbg).sum(), 0, 1e-9);

    // Second processing
    _imu2->processIMU();
    Eigen::Matrix3d dR = geometry::exp_so3(_gyr * dt);
    expected_J_dbg     = dR.transpose() * _imu1->_J_dR_bg - J_rk * dt;
    expected_J_dv_dba  = expected_J_dv_dba - _imu1->getDeltaR() * dt;
    expected_J_dv_dbg  = -_imu1->getDeltaR() * geometry::skewMatrix(_acc) * _imu1->_J_dR_bg * dt;
    expected_J_dp_dba  = expected_J_dp_dba + _imu1->_J_dv_ba * dt - 0.5 * _imu1->getDeltaR() * dt * dt;
    expected_J_dp_dbg =
        _imu1->_J_dv_bg * dt - 0.5 * _imu1->getDeltaR() * geometry::skewMatrix(_acc) * _imu1->_J_dR_bg * dt * dt;

    ASSERT_NEAR((_imu2->_J_dR_bg * expected_J_dbg.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu2->_J_dv_ba * expected_J_dv_dba.inverse()).trace(), 3, 1e-9);
    ASSERT_NEAR((_imu2->_J_dv_bg - expected_J_dv_dbg).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu2->_J_dp_ba - expected_J_dp_dba).sum(), 0, 1e-9);
    ASSERT_NEAR((_imu2->_J_dp_bg - expected_J_dp_dbg).sum(), 0, 1e-9);
}

TEST_F(ImuTest, predictionPositionVelocity) {

    // We set a rotated inertial frame
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 0.38001193, 0.16469125, 0.91020202, 0.03067918, -0.9857245, 0.16554758,
        0.92447267, -0.0349858, -0.37963966;
    T_i_f.affine().block(0, 3, 3, 1) = Eigen::Vector3d::Ones();

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0, 0, 0;
    acc << 0, 0, 10.81;
    acc = T_i_f.rotation().transpose() * acc;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt         = 0.001;
    _imu_cfg->rate_hz = 1000;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    cur_imu->setBa(ba);
    cur_imu->setBg(bg);
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 1001; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(1, 1, 1.5)).norm(), 0, 1e-5);
    ASSERT_NEAR((cur_imu->getVelocity() - Eigen::Vector3d(0, 0, 1)).norm(), 0, 1e-5);
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().rotation().transpose() * T_i_f.rotation()).trace(), 3, 1e-5);

    // Test IMU Factor

    // Create parameter blocks
    PointXYZParametersBlock dvi(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dvj(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dba(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dbg(Eigen::Vector3d::Zero());
    PoseParametersBlock dX_i(Eigen::Affine3d::Identity());
    PoseParametersBlock dX_j(Eigen::Affine3d::Identity());
    double lambda[1] = {0.0};

    std::vector<double *> parameters_blocks;
    parameters_blocks.push_back(dX_i.values());
    parameters_blocks.push_back(dX_j.values());
    parameters_blocks.push_back(dvi.values());
    parameters_blocks.push_back(dvj.values());
    parameters_blocks.push_back(dba.values());
    parameters_blocks.push_back(dbg.values());

    // Create cost fct
    ceres::CostFunction *cost_fct = new IMUFactor(lastKF->getIMU(), cur_frame->getIMU());

    // Create residual and jaocobian objects
    Eigen::VectorXd residuals;
    residuals.resize(cost_fct->num_residuals());

    std::vector<int> block_sizes = cost_fct->parameter_block_sizes();
    double **raw_jacobians       = new double *[block_sizes.size()];
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    jacobians.resize(block_sizes.size());

    for (size_t i = 0; i < block_sizes.size(); i++) {
        jacobians[i].resize(cost_fct->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }

    cost_fct->Evaluate(parameters_blocks.data(), residuals.data(), raw_jacobians);
    ASSERT_NEAR(residuals.norm(), 0, 1e-3);

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05));

    // Test Inertial optimization
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0);
    lastKF->setKeyFrame();
    lastKF->setPrior(T_i_f.inverse(), 100 * Vector6d::Ones());                         // To constrain the problem
    cur_frame->setPrior(cur_frame->getWorld2FrameTransform(), 100 * Vector6d::Ones()); // To constrain the problem
    cur_frame->setKeyFrame();
    local_map->addFrame(lastKF);
    local_map->addFrame(cur_frame);

    // Add error in states
    Vector6d err_pose;
    err_pose << 0.0000, -0.000, 0.000, 0.1, 0.05, -0.01;
    lastKF->getIMU()->setVelocity(lastKF->getIMU()->getVelocity());
    cur_frame->getIMU()->setVelocity(cur_frame->getIMU()->getVelocity() + Eigen::Vector3d(0.04, 0.02, -0.02));
    cur_frame->setWorld2FrameTransform(cur_frame->getWorld2FrameTransform() * geometry::se3_Vec6dtoRT(err_pose));

    // Solve the SLAM problem
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.localMapVIOptimization(local_map, 0);

    // Check states
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(1, 1, 1.5)).norm(), 0, 1e-2);
    ASSERT_NEAR((cur_imu->getVelocity() - Eigen::Vector3d(0, 0, 1)).norm(), 0, 1e-2);
    ASSERT_NEAR((cur_frame->getFrame2WorldTransform().rotation().transpose() * T_i_f.rotation()).trace(), 3, 1e-5);

    // Change the scale
    double scale = 0.5;
    Eigen::Affine3d T_flast_w = lastKF->getWorld2FrameTransform();
    Eigen::Affine3d T_fcurr_w = cur_frame->getWorld2FrameTransform();
    T_fcurr_w.translation() *= scale;
    T_flast_w.translation() *= scale;
    lastKF->setWorld2FrameTransform(T_flast_w);
    cur_frame->setWorld2FrameTransform(T_fcurr_w);

    // Test IMU Factor INIT
    ceres::CostFunction *cost_fct1 = new IMUFactorInit(lastKF->getIMU(), cur_frame->getIMU());


    double r_w_i[2] = {0.0, 0.0};
    std::vector<double *> parameters_blocks1;
    parameters_blocks1.push_back(r_w_i);
    parameters_blocks1.push_back(dvi.values());
    parameters_blocks1.push_back(dvj.values());
    parameters_blocks1.push_back(dba.values());
    parameters_blocks1.push_back(dbg.values());
    parameters_blocks1.push_back(lambda);

    // Check the Jacobians against finite differences, at zero and away from zero
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct1, parameters_blocks1));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct1, parameters_blocks1, 0.05));

    // Solve
    ceres::Problem problem;
    problem.AddResidualBlock(cost_fct1, nullptr, parameters_blocks1);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    ASSERT_NEAR(scale, 1 / std::exp(lambda[0]), 1e-2);
}

TEST_F(ImuTest, biasEstimation) {

    // Let's see if the bias are estimated with a pose prior
    _frame0->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());
    _imu0->setBa(Eigen::Vector3d(0.5, 1., 1.0));
    _imu0->setBg(Eigen::Vector3d(0.1, 0.3, 0.1));
    _frame1->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame1->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    // Build the local map
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0);
    local_map->addFrame(_frame0);
    _imu1->processIMU();
    _frame1->setKeyFrame();
    local_map->addFrame(_frame1);

    // Solve the SLAM problem
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.localMapVIOptimization(local_map, 0);

    // check bias
    ASSERT_NEAR((_imu0->getBg() - Eigen::Vector3d(0.1, 0.3, 0.1)).norm(), 0, 1e-5);
    ASSERT_NEAR((_imu0->getBa() - Eigen::Vector3d(0.5, 1, 1)).norm(), 0, 1e-5);
}

// Combination of a rotation and a translation
// The results of the free integration comes from https://github.com/Aceinna/gnss-ins-sim/tree/master

TEST_F(ImuTest, predictionWithRotation) {

    // The IMU is set on the frame 1 from Aceinna
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0.5, 0, 0;
    acc << -1, 0, -9.81;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt = 0.005;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 202; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransformIMU(dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR(
        (T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(-0.505, 0.813, 0.1))
            .norm(),
        0,
        1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-1, 2.41, 0.4)).norm(), 0, 1e-2);

    // Second set of measurements
    cur_frame->setKeyFrame();
    lastKF = cur_frame;
    gyr << 0.5, 0.2, 0.04;
    acc << -1, 0.05, -9.81;

    for (int i = 202; i < 401; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransformIMU(dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR(
        (T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() - Eigen::Vector3d(-2.31, 6.18, 1.62))
            .norm(),
        0,
        1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-2.95, 8.91, 3.24)).norm(), 0, 1e-2);
}

TEST_F(ImuTest, predictionWithRotation2) {

    // The IMU is set on the frame 1 from Aceinna
    Eigen::Affine3d T_i_f = Eigen::Affine3d::Identity();
    T_i_f.affine().block(0, 0, 3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

    Eigen::Vector3d gyr, acc, ba, bg;
    gyr << 0.5, 0.2, 0.04;
    acc << -1, 0.05, -9.81;
    ba << 0.0, 0, 0;
    bg << 0, 0, 0;
    double dt = 0.005;

    std::shared_ptr<IMU> cur_imu     = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
    std::shared_ptr<Frame> cur_frame = std::shared_ptr<Frame>(new Frame());
    cur_frame->init(cur_imu, 1e9);
    cur_frame->setWorld2FrameTransform(T_i_f.inverse());
    cur_frame->setKeyFrame();
    std::shared_ptr<IMU> last_imu = cur_imu;

    std::shared_ptr<Frame> lastKF = cur_frame;

    for (int i = 1; i < 201; i++) {
        cur_imu = std::shared_ptr<IMU>(new IMU(_imu_cfg, acc, gyr));
        cur_imu->setLastIMU(last_imu);
        cur_imu->setLastKF(lastKF);

        // Create frame and process IMU
        cur_frame = std::shared_ptr<Frame>(new Frame());
        cur_frame->init(cur_imu, 1e9 + (dt * i) * 1e9);
        cur_imu->processIMU();

        // Estimate transform
        Eigen::Affine3d dT;
        cur_frame->getIMU()->estimateTransformIMU(dT);
        cur_frame->setWorld2FrameTransform(dT.inverse() * lastKF->getWorld2FrameTransform());

        // Set last imu
        last_imu = cur_imu;
    }

    ASSERT_NEAR((T_i_f.inverse() * cur_frame->getFrame2WorldTransform().translation() -
                 Eigen::Vector3d(-0.82143062, 0.80412303, 0.15357111))
                    .norm(),
                0,
                1e-2);
    ASSERT_NEAR((T_i_f.inverse() * cur_imu->getVelocity() - Eigen::Vector3d(-1.97810799, 2.38035184, 0.5780088)).norm(),
                0,
                1e-2);
}

TEST_F(ImuTest, simuEuroc) {

    // Get the txt file for the groundtruth
    std::string gt_path = "../tests/euroc_gt.csv";
    std::fstream gt_file;

    // Imu measurement and poses as vectors
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> meas_vec;
    std::vector<Eigen::Affine3d> pose_vec;
    std::vector<Eigen::Vector3d> vel_vec;
    std::vector<double> ts_vec;

    gt_file.open(gt_path, std::ios::in);
    if (gt_file.is_open()) { // checking whether the file is open
        std::string tp;

        double ts, tsp;           // current and previous timestamp
        Eigen::Vector3d v, vp;    // current and previous velocity
        Eigen::Affine3d T, Tp;    // current and previous pose
        Eigen::Vector3d acc, gyr; // accelero and gyro meas

        getline(gt_file, tp); // skip the first line

        // Init the derivation
        getline(gt_file, tp);
        read_line_euroc(tp, Tp, vp, tsp);

        getline(gt_file, tp);
        read_line_euroc(tp, T, v, ts);

        Tp  = T;
        vp  = v;
        tsp = ts;

        while (getline(gt_file, tp)) { // read data from file object and put it into string.
            read_line_euroc(tp, T, v, ts);

            // Compute accelero and gyro measurements
            // v = (1 / (ts - tsp)) * (T.translation() - Tp.translation());
            acc = (1 / (ts - tsp)) * (T.rotation().transpose() * (v - vp)) - T.rotation().transpose() * g;
            gyr = (1 / (ts - tsp)) * geometry::log_so3((Tp.rotation().transpose() * T.rotation()));

            // Fill vectors
            meas_vec.push_back(std::make_pair(acc, gyr));
            vel_vec.push_back(vp);
            pose_vec.push_back(Tp);
            ts_vec.push_back(tsp * 1e9);

            // Set previous values
            Tp  = T;
            vp  = v;
            tsp = ts;
        }

        gt_file.close(); // close the file object.
    }

    // TEST IMU INTEGRATION

    // Set the first frame
    std::shared_ptr<Frame> frame0 = std::shared_ptr<Frame>(new Frame());
    std::shared_ptr<IMU> imu0     = std::make_shared<IMU>(_imu_cfg, meas_vec.at(0).first, meas_vec.at(0).second);
    frame0->init(imu0, ts_vec.at(0));
    frame0->setWorld2FrameTransform(pose_vec.at(0).inverse());
    frame0->setKeyFrame();
    imu0->setLastKF(frame0);
    imu0->setVelocity(vel_vec.at(0));
    write_result(frame0);

    double ts, tsp, dt;    // current and previous timestamp
    Eigen::Affine3d T, Tp; // current and previous pose
    Eigen::Vector3d v, vp; // current and previous velocity
    tsp = ts_vec.at(0);
    vp  = vel_vec.at(0);
    Tp  = pose_vec.at(0);

    for (uint i = 1; i < meas_vec.size(); i++) {
        std::shared_ptr<Frame> frame = std::shared_ptr<Frame>(new Frame());
        ts                           = ts_vec.at(i);
        dt                           = (ts - tsp) * 1e-9;

        // Create and process IMU
        std::shared_ptr<IMU> imu = std::make_shared<IMU>(_imu_cfg, meas_vec.at(i).first, meas_vec.at(i).second);
        frame->init(imu, ts);
        imu->setLastKF(frame0);
        imu->setLastIMU(imu0);
        imu->processIMU();
        imu0 = imu;

        // Simple integration
        Eigen::Matrix3d dR           = geometry::exp_so3(meas_vec.at(i - 1).second * dt);
        Eigen::Matrix3d R_w_f        = Tp.rotation();
        v                            = vp + g * dt + R_w_f * meas_vec.at(i - 1).first * dt;
        T.affine().block(0, 0, 3, 3) = Tp.rotation() * dR;
        T.translation() =
            Tp.translation() + vp * dt + 0.5 * g * dt * dt + 0.5 * R_w_f * meas_vec.at(i - 1).first * dt * dt;
        vp  = v;
        Tp  = T;
        tsp = ts;

        ASSERT_NEAR(
            ((T * frame->getWorld2FrameTransform()).matrix() - Eigen::Affine3d::Identity().matrix()).norm(), 0, 0.1);

        // if (i % 50 == 0) {
        //     write_result(frame);
        // }
    }

    // TEST IMU INITIALIZATION

    int idx_start                       = 2000;                                 // Start after 10 seconds
    double dt_kf                        = 0.5;                                  // t between kf
    std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0); // The sliding window
    double scale_factor                 = 0.5;                                  // Scale factor to be recovered
    std::unordered_map<double, Eigen::Affine3d> map_ts_gt; // A map to stack the gt poses of the local map

    // Set the first frame
    frame0 = std::shared_ptr<Frame>(new Frame());
    imu0   = std::make_shared<IMU>(_imu_cfg, meas_vec.at(idx_start).first, meas_vec.at(idx_start).second);
    frame0->init(imu0, ts_vec.at(idx_start));
    Eigen::Affine3d T_w_f0 = pose_vec.at(idx_start);
    map_ts_gt.emplace(ts_vec.at(idx_start), T_w_f0);
    T_w_f0.translation() *= scale_factor;
    frame0->setWorld2FrameTransform(T_w_f0.inverse());
    frame0->setKeyFrame();
    imu0->setLastKF(frame0);
    imu0->setVelocity(vel_vec.at(idx_start) * scale_factor);
    local_map->addFrame(frame0);
    tsp = ts_vec.at(idx_start);

    for (uint i = idx_start + 1; i < meas_vec.size(); i++) {
        std::shared_ptr<Frame> frame = std::shared_ptr<Frame>(new Frame());
        ts                           = ts_vec.at(i);
        dt                           = (ts - tsp) * 1e-9;

        // Create and process IMU
        std::shared_ptr<IMU> imu = std::make_shared<IMU>(_imu_cfg, meas_vec.at(i).first, meas_vec.at(i).second);
        frame->init(imu, ts);
        imu->setLastKF(frame0);
        imu->setLastIMU(imu0);
        imu->processIMU();
        imu0 = imu;

        // Compute scaled pose and velocity
        Eigen::Affine3d T_w_f = pose_vec.at(i);
        map_ts_gt.emplace(ts, T_w_f);
        T_w_f.translation() *= scale_factor;
        frame->setWorld2FrameTransform(T_w_f.inverse());
        imu0->setVelocity(vel_vec.at(i) * scale_factor);

        // Vote KF
        if (dt > dt_kf) {
            frame->setKeyFrame();
            local_map->addFrame(frame);
            frame0 = frame;
            tsp    = ts;
        }

        // Break the loop if enough KF
        if (local_map->getMapSize() == 10)
            break;
    }

    // Solve the init problem
    Eigen::Matrix3d R_w_i;
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    ceres_ba.VIInit(local_map, R_w_i, true);

    std::cout << "Rotation Matrix : \n" << R_w_i << std::endl;

    for (auto frame : local_map->getFrames()) {
        Eigen::Affine3d T_w_f = map_ts_gt.at(frame->getTimestamp());
        ASSERT_NEAR(((T_w_f * frame->getWorld2FrameTransform()).matrix() - Eigen::Affine3d::Identity().matrix()).norm(),
                    0,
                    0.02);
    }

    // TEST Bias Estimation
    _imu_cfg->bgyr_noise                    = 1.9393e-03;
    _imu_cfg->acc_noise                     = 3.0e-2;
    _imu_cfg->bacc_noise                    = 3.0000e-2;
    idx_start                               = 2000;                                 // Start after 10 seconds
    dt_kf                                   = 0.5;                                  // t between kf
    std::shared_ptr<LocalMap> local_map_vio = std::make_shared<LocalMap>(0, 10, 0); // The sliding window
    std::unordered_map<double, Eigen::Affine3d> map_ts_gt_vio; // A map to stack the gt poses of the local map
    Eigen::Vector3d ba(0.1, 0.2, -0.1);
    Eigen::Vector3d bg(0.4, -0.2, 0.01);

    // Set the first frame
    frame0 = std::shared_ptr<Frame>(new Frame());
    imu0   = std::make_shared<IMU>(_imu_cfg, meas_vec.at(idx_start).first + ba, meas_vec.at(idx_start).second + bg);
    frame0->init(imu0, ts_vec.at(idx_start));
    T_w_f0 = pose_vec.at(idx_start);
    map_ts_gt_vio.emplace(ts_vec.at(idx_start), T_w_f0);
    frame0->setWorld2FrameTransform(T_w_f0.inverse());
    frame0->setPrior(T_w_f0.inverse(), 100 * Vector6d::Ones());
    frame0->setKeyFrame();
    imu0->setLastKF(frame0);
    imu0->setVelocity(vel_vec.at(idx_start));
    local_map_vio->addFrame(frame0);
    tsp = ts_vec.at(idx_start);

    for (uint i = idx_start + 1; i < meas_vec.size(); i++) {
        std::shared_ptr<Frame> frame = std::shared_ptr<Frame>(new Frame());
        ts                           = ts_vec.at(i);
        dt                           = (ts - tsp) * 1e-9;

        // Create and process IMU
        std::shared_ptr<IMU> imu =
            std::make_shared<IMU>(_imu_cfg, meas_vec.at(i).first + ba, meas_vec.at(i).second + bg);
        frame->init(imu, ts);
        imu->setLastKF(frame0);
        imu->setBa(frame0->getIMU()->getBa());
        imu->setBg(frame0->getIMU()->getBg());
        imu->setLastIMU(imu0);
        imu->processIMU();
        imu0 = imu;

        // Compute scaled pose and velocity
        Eigen::Affine3d T_w_f = pose_vec.at(i);
        map_ts_gt_vio.emplace(ts, T_w_f);
        frame->setPrior(T_w_f.inverse(), 100 * Vector6d::Ones());
        frame->setWorld2FrameTransform(T_w_f.inverse());
        imu0->setVelocity(vel_vec.at(i));

        // Vote KF
        if (dt > dt_kf) {
            frame->setKeyFrame();
            local_map_vio->addFrame(frame);
            ceres_ba.localMapVIOptimization(local_map_vio, 1);
            frame0 = frame;
            tsp    = ts;
        }

        // Break the loop if enough KF
        if (local_map_vio->getMapSize() == 30)
            break;
    }
    
    ASSERT_NEAR((local_map_vio->getFrames().at(29)->getIMU()->getBa() - ba).norm(), 0, 0.02);
    ASSERT_NEAR((local_map_vio->getFrames().at(29)->getIMU()->getBg() - bg).norm(), 0, 0.02);
}

TEST_F(ImuTest, TestPreInteg) {
    // Measurements
    const double a = 0.1, w = M_PI / 100.0;
    Eigen::Vector3d measured_acc(a, 0.0, 0.0);
    Eigen::Vector3d measured_gyr(w, 0.0, 0.0);
    double deltaT = 0.5;

    // Expected pre-integrated values
    Eigen::Vector3d expectedDeltaR1(w * deltaT, 0.0, 0.0);
    Eigen::Vector3d expectedDeltaP1(0.5 * a * deltaT * deltaT, 0, 0);
    Eigen::Vector3d expectedDeltaV1(0.05, 0.0, 0.0);

    // Set Frames and IMU
    _imu0   = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _frame0 = std::shared_ptr<Frame>(new Frame());
    _frame0->init(_imu0, 1e9);
    _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame0->setKeyFrame();
    _imu0->setLastKF(_frame0);

    _imu1 = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _imu1->setLastIMU(_imu0);
    _imu1->setLastKF(_frame0);
    _frame1 = std::shared_ptr<Frame>(new Frame());
    _frame1->init(_imu1, 1.5e9);

    // Pre integrate
    _imu1->processIMU();
    ASSERT_EQ((_imu1->getDeltaR() - geometry::exp_so3(expectedDeltaR1)).sum(), 0);
    ASSERT_EQ((_imu1->getDeltaP() - expectedDeltaP1).norm(), 0);
    ASSERT_EQ((_imu1->getDeltaV() - expectedDeltaV1).norm(), 0);

    // Integrate again
    Eigen::Vector3d expectedDeltaR2(2.0 * 0.5 * M_PI / 100.0, 0.0, 0.0);
    Eigen::Vector3d expectedDeltaP2(0.025 + expectedDeltaP1(0) + 0.5 * 0.1 * 0.5 * 0.5, 0, 0);
    Eigen::Vector3d expectedDeltaV2 =
        Eigen::Vector3d(0.05, 0.0, 0.0) + geometry::exp_so3(expectedDeltaR1) * measured_acc * 0.5;

    _imu2 = std::shared_ptr<IMU>(new IMU(_imu_cfg, measured_acc, measured_gyr));
    _imu2->setLastIMU(_imu1);
    _imu2->setLastKF(_frame0);
    _frame2 = std::shared_ptr<Frame>(new Frame());
    _frame2->init(_imu2, 2e9);
    _imu2->processIMU();
    ASSERT_NEAR((_imu2->getDeltaR() - geometry::exp_so3(expectedDeltaR2)).sum(), 0, 0.000001);
    ASSERT_EQ((_imu2->getDeltaP() - expectedDeltaP2).norm(), 0);
    ASSERT_EQ((_imu2->getDeltaV() - expectedDeltaV2).norm(), 0);
}

/*!
 * @brief A preintegration chain from a keyframe over a rotating, accelerating motion.
 *
 * The frames are kept alive here because IMUs only hold weak pointers to them.
 */
struct ImuChain {
    std::vector<std::shared_ptr<Frame>> frames;
    std::shared_ptr<IMU> imu_i, imu_j;
};

static ImuChain integrateChain(const std::shared_ptr<imu_config> &cfg,
                               const Eigen::Vector3d &ba,
                               const Eigen::Vector3d &bg,
                               int n_samples  = 40,
                               int gap_after  = -1,
                               double gap_s   = 0) {
    const double dt = 1.0 / cfg->rate_hz;
    auto acc_at     = [](int k) { return Eigen::Vector3d(0.4 * std::sin(0.3 * k), 0.2 + 0.1 * std::cos(0.2 * k), 9.6); };
    auto gyr_at     = [](int k) { return Eigen::Vector3d(0.5 * std::sin(0.1 * k), 0.3, -0.4 * std::cos(0.15 * k)); };

    ImuChain c;
    Eigen::Affine3d T_w_i              = Eigen::Affine3d::Identity();
    T_w_i.linear()                     = geometry::exp_so3(Eigen::Vector3d(0.2, -0.1, 0.3));
    T_w_i.translation()                = Eigen::Vector3d(1.0, -2.0, 0.5);
    std::shared_ptr<IMU> imu           = std::make_shared<IMU>(cfg, acc_at(0), gyr_at(0));
    std::shared_ptr<Frame> kf          = std::make_shared<Frame>();
    kf->init(imu, 1e9);
    kf->setWorld2FrameTransform(T_w_i.inverse());
    kf->setKeyFrame();
    imu->setLastKF(kf);
    imu->setBa(ba);
    imu->setBg(bg);
    imu->setVelocity(Eigen::Vector3d(0.3, -0.2, 0.1));
    c.frames.push_back(kf);
    c.imu_i = imu;

    std::shared_ptr<IMU> last = imu;
    for (int k = 1; k <= n_samples; k++) {
        std::shared_ptr<IMU> cur = std::make_shared<IMU>(cfg, acc_at(k), gyr_at(k));
        cur->setLastIMU(last);
        cur->setLastKF(kf);
        std::shared_ptr<Frame> f = std::make_shared<Frame>();
        double t = 1.0 + k * dt + ((gap_after >= 0 && k > gap_after) ? gap_s : 0.0);
        f->init(cur, (unsigned long long)(t * 1e9));
        cur->processIMU();
        c.frames.push_back(f);
        last = cur;
    }
    c.imu_j = last;
    return c;
}

TEST_F(ImuTest, biasJacobiansMatchReintegration) {

    // The preintegration bias Jacobians must predict how the deltas change when the chain is
    // re-integrated with a slightly different bias (independent of the Jacobian formulas).
    const Eigen::Vector3d ba(0.05, -0.02, 0.03), bg(0.01, -0.02, 0.015);
    ImuChain ref = integrateChain(_imu_cfg, ba, bg);
    const double h = 1e-6;

    for (int axis = 0; axis < 3; axis++) {
        Eigen::Vector3d e = Eigen::Vector3d::Zero();
        e(axis)           = h;

        ImuChain cg = integrateChain(_imu_cfg, ba, bg + e);
        Eigen::Vector3d num_dR_dbg = geometry::log_so3(ref.imu_j->getDeltaR().transpose() * cg.imu_j->getDeltaR()) / h;
        Eigen::Vector3d num_dv_dbg = (cg.imu_j->getDeltaV() - ref.imu_j->getDeltaV()) / h;
        Eigen::Vector3d num_dp_dbg = (cg.imu_j->getDeltaP() - ref.imu_j->getDeltaP()) / h;
        EXPECT_LT((num_dR_dbg - ref.imu_j->_J_dR_bg.col(axis)).norm(), 1e-4) << "dR/dbg axis " << axis;
        EXPECT_LT((num_dv_dbg - ref.imu_j->_J_dv_bg.col(axis)).norm(), 1e-4) << "dv/dbg axis " << axis;
        EXPECT_LT((num_dp_dbg - ref.imu_j->_J_dp_bg.col(axis)).norm(), 1e-4) << "dp/dbg axis " << axis;

        ImuChain ca = integrateChain(_imu_cfg, ba + e, bg);
        Eigen::Vector3d num_dv_dba = (ca.imu_j->getDeltaV() - ref.imu_j->getDeltaV()) / h;
        Eigen::Vector3d num_dp_dba = (ca.imu_j->getDeltaP() - ref.imu_j->getDeltaP()) / h;
        EXPECT_LT((num_dv_dba - ref.imu_j->_J_dv_ba.col(axis)).norm(), 1e-4) << "dv/dba axis " << axis;
        EXPECT_LT((num_dp_dba - ref.imu_j->_J_dp_ba.col(axis)).norm(), 1e-4) << "dp/dba axis " << axis;
    }
}

TEST_F(ImuTest, imuFactorJacobiansAwayFromZero) {

    // Issue 12: the pose-j rotation block is only right at a zero increment.
    ImuChain c = integrateChain(_imu_cfg, Eigen::Vector3d(0.05, -0.02, 0.03), Eigen::Vector3d(0.01, -0.02, 0.015));
    ceres::CostFunction *cost_fct = new IMUFactor(c.imu_i, c.imu_j);

    PoseParametersBlock dX_i(Eigen::Affine3d::Identity()), dX_j(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dvi(Eigen::Vector3d::Zero()), dvj(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dba(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero());
    std::vector<double *> parameters_blocks = {
        dX_i.values(), dX_j.values(), dvi.values(), dvj.values(), dba.values(), dbg.values()};

    EXPECT_NEAR(isae_test::ResidualNorm(*cost_fct, parameters_blocks), 0, 1e-3);
    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05, 1));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05, 2));
    delete cost_fct;
}

TEST_F(ImuTest, imuFactorInitJacobiansAtNonUnitScale) {

    // Issue 3: the scale Jacobian omits exp(lambda), which is invisible at lambda = 0.
    ImuChain c = integrateChain(_imu_cfg, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    ceres::CostFunction *cost_fct = new IMUFactorInit(c.imu_i, c.imu_j);

    double r_w_i[2]  = {0.02, -0.03};
    double lambda[1] = {std::log(3.0)};
    PointXYZParametersBlock dvi(Eigen::Vector3d::Zero()), dvj(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dba(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero());
    std::vector<double *> parameters_blocks = {r_w_i, dvi.values(), dvj.values(), dba.values(), dbg.values(), lambda};

    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05, 3));
    delete cost_fct;
}

TEST_F(ImuTest, imuPriorJacobiansWithDenseWhitening) {

    // Issue 13: every Jacobian block must carry the square-root information, which is dense after
    // sparsification.
    Eigen::Affine3d T       = Eigen::Affine3d::Identity();
    T.linear()              = geometry::exp_so3(Eigen::Vector3d(0.3, -0.2, 0.1));
    T.translation()         = Eigen::Vector3d(0.5, 1.0, -0.4);
    Eigen::Affine3d T_prior = T * geometry::se3_Vec6dtoRT((Vector6d() << 0.01, -0.02, 0.01, 0.05, 0.0, -0.03).finished());

    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1, 1);
    Eigen::MatrixXd A(15, 15);
    for (int i = 0; i < 15; i++)
        for (int j = 0; j < 15; j++)
            A(i, j) = dist(gen);
    Eigen::MatrixXd sqrt_inf = A + 5 * Eigen::MatrixXd::Identity(15, 15);

    ceres::CostFunction *cost_fct = new IMUPriordx(T,
                                                   T_prior,
                                                   Eigen::Vector3d(0.1, 0.2, 0.3),
                                                   Eigen::Vector3d(0.12, 0.18, 0.31),
                                                   Eigen::Vector3d(0.01, 0.0, -0.01),
                                                   Eigen::Vector3d::Zero(),
                                                   Eigen::Vector3d(0.001, 0.002, 0.0),
                                                   Eigen::Vector3d::Zero(),
                                                   sqrt_inf);

    PoseParametersBlock dX(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dv(Eigen::Vector3d::Zero()), dba(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero());
    std::vector<double *> parameters_blocks = {dX.values(), dv.values(), dba.values(), dbg.values()};

    EXPECT_TRUE(isae_test::JacobiansMatch(*cost_fct, parameters_blocks));
    EXPECT_TRUE(isae_test::JacobiansMatchAtRandomPoint(*cost_fct, parameters_blocks, 0.05, 4));
    delete cost_fct;
}

TEST_F(ImuTest, imuDataGapsAreIntegratedAndNotTurnedIntoFactors) {

    // Issue 18: a step longer than 1 s used to be replaced by 1 / rate, so the preintegration covered
    // less time than the factor built on it.
    const Eigen::Vector3d ba = Eigen::Vector3d::Zero(), bg = Eigen::Vector3d::Zero();
    const double dt          = 1.0 / _imu_cfg->rate_hz;
    ImuChain ok  = integrateChain(_imu_cfg, ba, bg, 40);
    ImuChain gap = integrateChain(_imu_cfg, ba, bg, 40, 20, 2.0);

    EXPECT_EQ(ok.imu_j->getGapSteps(), 0);
    EXPECT_NEAR(ok.imu_j->getIntegratedDt(), 40 * dt, 1e-9);
    EXPECT_EQ(gap.imu_j->getGapSteps(), 1);
    EXPECT_TRUE(ok.imu_j->hasValidCovariance());
    EXPECT_FALSE(std::make_shared<IMU>(_imu_cfg, _acc, _gyr)->hasValidCovariance()) << "zero covariance";
    EXPECT_NEAR(gap.imu_j->getIntegratedDt(), 40 * dt + 2.0, 1e-9);

    // The factor needs a preintegration covering exactly the KF interval: a preintegration that does not start
    // at fi (here, fj stamped 10 ms later than its integrated measurements) is not used
    EXPECT_TRUE(AOptimizer::imuFactorUsable(ok.frames.front(), ok.frames.back()));
    {
        ImuChain shifted          = integrateChain(_imu_cfg, ba, bg, 40);
        std::shared_ptr<Frame> fj = std::make_shared<Frame>();
        fj->init(shifted.imu_j, shifted.frames.back()->getTimestamp() + 10'000'000ULL);
        EXPECT_FALSE(AOptimizer::imuFactorUsable(shifted.frames.front(), fj));
    }

    // No preintegration factor across the gap, one otherwise
    for (ImuChain *c : {&ok, &gap}) {
        std::shared_ptr<Frame> kf_i = c->frames.front(), kf_j = c->frames.back();
        kf_i->setPrior(kf_i->getWorld2FrameTransform(), 100 * Vector6d::Ones());
        kf_j->setPrior(kf_j->getWorld2FrameTransform(), 100 * Vector6d::Ones());
        kf_j->setKeyFrame();
        std::shared_ptr<LocalMap> local_map = std::make_shared<LocalMap>(0, 10, 0);
        local_map->addFrame(kf_i);
        local_map->addFrame(kf_j);
        isae::AngularAdjustmentCERESAnalytic ceres_ba;
        ceres_ba.localMapVIOptimization(local_map, 0);
        EXPECT_EQ(ceres_ba.getLastVIStats().n_imu_factors, c == &ok ? 1u : 0u);
    }
}


/*!
 * @brief Simulated smooth motion with keyframes at their true poses and an IMU chain whose
 * measurements integrate exactly to the motion (zero-order hold), with the given biases.
 */
struct SimKFs {
    std::vector<std::shared_ptr<Frame>> frames, kfs;
    std::vector<Eigen::Vector3d> kf_velocities;
    std::shared_ptr<LocalMap> map;
};

static Eigen::Vector3d simP(double t) {
    return Eigen::Vector3d(0.5 * std::sin(0.8 * t), 0.3 * std::cos(0.6 * t) - 0.3, 0.1 * std::sin(1.1 * t));
}
static Eigen::Vector3d simV(double t) {
    return Eigen::Vector3d(0.4 * std::cos(0.8 * t), -0.18 * std::sin(0.6 * t), 0.11 * std::cos(1.1 * t));
}
static Eigen::Matrix3d simR(double t) {
    return geometry::exp_so3(Eigen::Vector3d(0.3 * std::sin(0.5 * t), 0.2 * std::cos(0.7 * t) - 0.2, 0.4 * t));
}

// Constant velocity along a straight line, no rotation: the IMU only measures gravity
static Eigen::Vector3d straightP(double t) { return Eigen::Vector3d(0.4 * t, 0.1 * t, 0); }
static Eigen::Vector3d straightV(double) { return Eigen::Vector3d(0.4, 0.1, 0); }
static Eigen::Matrix3d straightR(double) { return Eigen::Matrix3d::Identity(); }

static SimKFs simulateKFs(const std::shared_ptr<imu_config> &cfg,
                          const Eigen::Vector3d &ba_true,
                          const Eigen::Vector3d &bg_true,
                          int n_kf      = 10,
                          int kf_every  = 60,
                          bool straight = false) {
    const double dt = 1.0 / cfg->rate_hz;
    auto simP       = straight ? straightP : ::isae::simP;
    auto simV       = straight ? straightV : ::isae::simV;
    auto simR       = straight ? straightR : ::isae::simR;
    auto meas       = [&](int k, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {
        double t = k * dt;
        gyr      = geometry::log_so3(simR(t).transpose() * simR(t + dt)) / dt + bg_true;
        acc      = simR(t).transpose() * ((simV(t + dt) - simV(t)) / dt - g) + ba_true;
    };
    auto pose = [&](double t) {
        Eigen::Affine3d T_w_f = Eigen::Affine3d::Identity();
        T_w_f.linear()        = simR(t);
        T_w_f.translation()   = simP(t);
        return T_w_f;
    };

    SimKFs sim;
    sim.map = std::make_shared<LocalMap>(0, n_kf + 1, 0);
    Eigen::Vector3d acc, gyr;
    meas(0, acc, gyr);
    std::shared_ptr<IMU> imu  = std::make_shared<IMU>(cfg, acc, gyr);
    std::shared_ptr<Frame> kf = std::make_shared<Frame>();
    kf->init(imu, 1e9);
    kf->setWorld2FrameTransform(pose(0).inverse());
    kf->setKeyFrame();
    imu->setLastKF(kf);
    sim.frames.push_back(kf);
    sim.kfs.push_back(kf);
    sim.kf_velocities.push_back(simV(0));
    sim.map->addFrame(kf);

    std::shared_ptr<IMU> last = imu;
    for (int k = 1; k <= (n_kf - 1) * kf_every; k++) {
        meas(k, acc, gyr);
        std::shared_ptr<IMU> cur = std::make_shared<IMU>(cfg, acc, gyr);
        cur->setLastIMU(last);
        cur->setLastKF(kf);
        std::shared_ptr<Frame> f = std::make_shared<Frame>();
        f->init(cur, (unsigned long long)(1e9 + k * dt * 1e9 + 0.5));
        cur->processIMU();
        sim.frames.push_back(f);
        if (k % kf_every == 0) {
            f->setWorld2FrameTransform(pose(k * dt).inverse());
            f->setKeyFrame();
            sim.map->addFrame(f);
            sim.kfs.push_back(f);
            sim.kf_velocities.push_back(simV(k * dt));
            kf = f;
        }
        last = cur;
    }
    return sim;
}

TEST_F(ImuTest, viInitRecoversGyroBiasAndVelocities) {

    // Issue 2: VIInit applied the accelerometer bias change twice and dropped the gyro bias change.
    const Eigen::Vector3d ba_true(0.0, 0.0, 0.0), bg_true(0.02, -0.015, 0.01);
    SimKFs sim = simulateKFs(_imu_cfg, ba_true, bg_true);

    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    Eigen::Matrix3d R_w_i;
    ceres_ba.VIInit(sim.map, R_w_i, false);

    EXPECT_LT(geometry::log_so3(R_w_i).norm(), 1e-2) << "the world was already gravity-aligned";
    for (size_t i = 0; i < sim.kfs.size(); i++) {
        std::shared_ptr<IMU> imu = sim.kfs[i]->getIMU();
        EXPECT_LT((imu->getBg() - bg_true).norm(), 2e-3) << "KF " << i << " bg " << imu->getBg().transpose();
        EXPECT_LT((imu->getBa() - ba_true).norm(), 2e-2) << "KF " << i << " ba " << imu->getBa().transpose();
        EXPECT_LT((imu->getVelocity() - sim.kf_velocities[i]).norm(), 5e-2) << "KF " << i;
    }

    // The deltas must have been corrected to the new bias: the IMU factors are then consistent with
    // the true states (residual small compared to the uncorrected deltas)
    for (size_t i = 1; i < sim.kfs.size(); i++) {
        Eigen::Matrix3d dR_true = sim.kfs[i - 1]->getWorld2FrameTransform().rotation() *
                                  sim.kfs[i]->getFrame2WorldTransform().rotation();
        EXPECT_LT(geometry::log_so3(sim.kfs[i]->getIMU()->getDeltaR().transpose() * dR_true).norm(), 2e-3)
            << "KF " << i;
    }
}

TEST_F(ImuTest, viInitModelMatchesTiltedScaledWorld) {

    // Issue 3: the inertial-initialization factor must vanish at the true gravity tilt, scale,
    // velocities and biases, when the KF poses come from a tilted, scaled visual world.
    const Eigen::Vector3d bg_true(0.01, -0.02, 0.005), w_true(0.08, -0.05, 0);
    const double s_true        = 2.5;
    const Eigen::Matrix3d R_wi = geometry::exp_so3(w_true);
    SimKFs sim                 = simulateKFs(_imu_cfg, Eigen::Vector3d::Zero(), bg_true, 20); // scale well determined

    // Visual world: p_w = R_wi p_I / s, R_w_f = R_wi R_I_f (VIInit maps it back with p_I = s R_wi^T p_w)
    std::vector<Eigen::Vector3d> p_true;
    for (auto &kf : sim.kfs) {
        Eigen::Affine3d T_I_f = kf->getFrame2WorldTransform(), T_w_f = Eigen::Affine3d::Identity();
        p_true.push_back(T_I_f.translation());
        T_w_f.linear()      = R_wi * T_I_f.linear();
        T_w_f.translation() = R_wi * T_I_f.translation() / s_true;
        kf->setWorld2FrameTransform(T_w_f.inverse());
    }

    for (size_t i = 1; i < sim.kfs.size(); i++) {
        std::shared_ptr<IMU> imu_i = sim.kfs[i - 1]->getIMU(), imu_j = sim.kfs[i]->getIMU();
        IMUFactorInit factor(imu_i, imu_j);
        double r_wi[2]   = {w_true.x(), w_true.y()};
        double lambda[1] = {std::log(s_true)};
        Eigen::Vector3d dvi = sim.kf_velocities[i - 1] - imu_i->getVelocity();
        Eigen::Vector3d dvj = sim.kf_velocities[i] - imu_j->getVelocity();
        Eigen::Vector3d dba = Eigen::Vector3d::Zero(), dbg = bg_true - imu_i->getBg();
        std::vector<double *> params = {r_wi, dvi.data(), dvj.data(), dba.data(), dbg.data(), lambda};
        EXPECT_LT(isae_test::ResidualNorm(factor, params), 0.1) << "KF pair " << i;
    }

    // End to end: the initialization recovers scale and tilt, and the positions become the true ones
    isae::AngularAdjustmentCERESAnalytic ceres_ba;
    Eigen::Matrix3d R_w_i;
    double s = ceres_ba.VIInit(sim.map, R_w_i, true);
    EXPECT_NEAR(s, s_true, 0.01 * s_true);
    EXPECT_LT(geometry::log_so3(R_w_i.transpose() * R_wi).norm(), 1e-2);
    for (size_t i = 0; i < sim.kfs.size(); i++)
        EXPECT_LT((sim.kfs[i]->getFrame2WorldTransform().translation() - p_true[i]).norm(), 1e-2) << "KF " << i;
}

TEST_F(ImuTest, viInitRefusesAnUnobservableScale) {

    // Mono inertial initialization: at constant velocity without rotation, any scale fits the visual positions with
    // a matching velocity. Only a positive scale was required, so whatever scale the solver stopped at was accepted.
    const double s_true = 2.5;
    for (bool straight : {true, false}) {
        SimKFs sim = simulateKFs(_imu_cfg, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 20, 60, straight);
        for (auto &kf : sim.kfs) { // visual world at an unknown scale
            Eigen::Affine3d T_w_f = kf->getFrame2WorldTransform();
            T_w_f.translation() /= s_true;
            kf->setWorld2FrameTransform(T_w_f.inverse());
        }
        isae::AngularAdjustmentCERESAnalytic ceres_ba;
        Eigen::Matrix3d R_w_i;
        const double s = ceres_ba.VIInit(sim.map, R_w_i, true);
        if (straight)
            EXPECT_LE(s, 0) << "unobservable scale accepted: " << s;
        else
            EXPECT_NEAR(s, s_true, 0.01 * s_true) << "an exciting motion must still initialize";
    }
}

TEST(ImuInitTest, staticInitializationAlignsGravityAndGuessesBiases) {

    // Level start: the old Rodrigues formula normalized a zero cross product (NaN orientation)
    std::vector<Eigen::Vector3d> accs(10, Eigen::Vector3d(0, 0, 9.81)), gyrs(10, Eigen::Vector3d(0.01, 0, -0.02));
    Eigen::Matrix3d R;
    Eigen::Vector3d ba, bg;
    EXPECT_TRUE(staticImuInitialization(accs, gyrs, R, ba, bg));
    ASSERT_TRUE(R.allFinite());
    EXPECT_NEAR((R * Eigen::Vector3d(0, 0, 9.81) - Eigen::Vector3d(0, 0, 9.81)).norm(), 0, 1e-9);
    EXPECT_NEAR(ba.norm(), 0, 1e-9);
    EXPECT_NEAR((bg - Eigen::Vector3d(0.01, 0, -0.02)).norm(), 0, 1e-12);

    // Upside down
    accs.assign(10, Eigen::Vector3d(0, 0, -9.81));
    EXPECT_TRUE(staticImuInitialization(accs, gyrs, R, ba, bg));
    ASSERT_TRUE(R.allFinite());
    EXPECT_NEAR((R * Eigen::Vector3d(0, 0, -9.81) - Eigen::Vector3d(0, 0, 9.81)).norm(), 0, 1e-9);

    // Tilted, with an accelerometer bias along gravity: the measured up direction maps to world up
    const Eigen::Matrix3d R_true = geometry::exp_so3(Eigen::Vector3d(0.3, -0.2, 0.5));
    const Eigen::Vector3d up_f   = R_true.transpose() * Eigen::Vector3d(0, 0, 1);
    accs.assign(10, up_f * 9.91);
    EXPECT_TRUE(staticImuInitialization(accs, gyrs, R, ba, bg));
    EXPECT_NEAR((R * up_f - Eigen::Vector3d(0, 0, 1)).norm(), 0, 1e-9);
    EXPECT_NEAR((ba - 0.1 * up_f).norm(), 0, 1e-9);

    // Moving: no bias guess
    for (int i = 0; i < 10; i++)
        gyrs[i] = Eigen::Vector3d(0.3 * std::sin(i), 0, 0);
    EXPECT_FALSE(staticImuInitialization(accs, gyrs, R, ba, bg));
    EXPECT_EQ(ba.norm(), 0);
    EXPECT_EQ(bg.norm(), 0);
}

// Constant world acceleration and constant body rate, measured with biases: held measurements integrate them exactly
struct ConstantMotion {
    Eigen::Matrix3d R0 = geometry::exp_so3(Eigen::Vector3d(0.1, -0.2, 0.7));
    Eigen::Vector3d p0{1, -2, 0.5}, v0{0.8, 0.3, -0.1}, a_w{0.4, -0.6, 0.2}, w_b{0.2, -0.1, 0.5};
    Eigen::Vector3d ba{0.05, -0.02, 0.08}, bg{0.004, -0.003, 0.002};
    Eigen::Affine3d pose(double t) const {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.linear()        = R0 * geometry::exp_so3(w_b * t);
        T.translation()   = p0 + v0 * t + 0.5 * a_w * t * t;
        return T;
    }
    Eigen::Vector3d acc(double t) const { return pose(t).linear().transpose() * (a_w - g) + ba; }
    Eigen::Vector3d gyr() const { return w_b + bg; }
};

TEST(ImuDeadReckoningTest, deadReckoningIntegratesConstantMotionExactly) {
    const ConstantMotion m;
    Eigen::Affine3d T = m.pose(0);
    Eigen::Vector3d v = m.v0;
    const double dt   = 0.005;
    for (int i = 0; i < 400; i++)
        IMU::deadReckon(T, v, m.acc(i * dt), m.gyr(), m.ba, m.bg, dt);
    EXPECT_NEAR((T.translation() - m.pose(2.0).translation()).norm(), 0, 1e-9);
    EXPECT_NEAR(geometry::log_so3(T.linear().transpose() * m.pose(2.0).linear()).norm(), 0, 1e-9);
    EXPECT_NEAR((v - (m.v0 + m.a_w * 2.0)).norm(), 0, 1e-9);
}

TEST(ImuDeadReckoningTest, carriedStateFollowsTheImuUntilItIsStale) {
    const ConstantMotion m;
    const unsigned long long t0 = 1000000000ULL, step = 5000000ULL; // 1 s, 5 ms (200 Hz)
    const double max_step       = IMU::maxStepDtForRate(200);
    auto at                     = [&](int i) { return t0 + i * step; };

    // The last visual pose 0.5 s before the dropout ended in a re-initialization
    CarriedImuState c;
    ASSERT_TRUE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(0), t0 - 500000000ULL, max_step, 2.0));

    // It follows the measurements; at() extrapolates within a step without moving the state
    for (int i = 1; i <= 200; i++)
        c.propagate(m.acc(i * 0.005), m.gyr(), at(i), false);
    ASSERT_TRUE(c.valid);
    EXPECT_NEAR((c.T_w_f.translation() - m.pose(1.0).translation()).norm(), 0, 1e-9);
    EXPECT_NEAR((c.v - (m.v0 + m.a_w)).norm(), 0, 1e-9);
    CarriedImuState c_img;
    ASSERT_TRUE(c.at(at(200) + 2000000ULL, c_img));
    EXPECT_NEAR((c_img.T_w_f.translation() - m.pose(1.002).translation()).norm(), 0, 1e-9);
    EXPECT_EQ(c.ts, at(200));
    EXPECT_FALSE(c.at(at(200) + 2 * (unsigned long long)(max_step * 1e9), c_img)); // too far from a measurement

    // Stale 2 s after the last visual pose (here 1.5 s of dead reckoning after the capture)
    for (int i = 201; i <= 299; i++)
        c.propagate(m.acc(i * 0.005), m.gyr(), at(i), false);
    EXPECT_TRUE(c.valid);
    for (int i = 300; i <= 302; i++)
        c.propagate(m.acc(i * 0.005), m.gyr(), at(i), false);
    EXPECT_FALSE(c.valid);
    EXPECT_FALSE(c.at(at(302), c_img));

    // Missing IMU data: a long step, or a sample interpolated across a raw gap
    ASSERT_TRUE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(0), at(0), max_step, 2.0));
    c.propagate(m.acc(0.1), m.gyr(), at(20), false); // 100 ms > max_step (50 ms)
    EXPECT_FALSE(c.valid);
    ASSERT_TRUE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(0), at(0), max_step, 2.0));
    c.propagate(m.acc(0.005), m.gyr(), at(1), true);
    EXPECT_FALSE(c.valid);

    // Not captured: implausible biases, non-finite state, already stale, no visual pose yet
    EXPECT_FALSE(c.capture(m.pose(0), m.v0, Eigen::Vector3d(3, 0, 0), m.bg, m.acc(0), m.gyr(), at(0), at(0), max_step, 2.0));
    EXPECT_FALSE(c.capture(m.pose(0), m.v0, m.ba, Eigen::Vector3d(0, 0.6, 0), m.acc(0), m.gyr(), at(0), at(0), max_step, 2.0));
    EXPECT_FALSE(c.capture(m.pose(0), Eigen::Vector3d(NAN, 0, 0), m.ba, m.bg, m.acc(0), m.gyr(), at(0), at(0), max_step, 2.0));
    EXPECT_FALSE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(500), at(0), max_step, 2.0));
    EXPECT_FALSE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(0), 0, max_step, 2.0));

    // The limit is the configured one (reinit_carry_max_age_vio): 0.5 s since the last visual pose
    EXPECT_TRUE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(100), at(0), max_step, 1.0));
    EXPECT_FALSE(c.capture(m.pose(0), m.v0, m.ba, m.bg, m.acc(0), m.gyr(), at(100), at(0), max_step, 0.4));
}

TEST_F(ImuTest, imuFactorAccountsForBiasChangesAfterIntegration) {

    // Issue 4: the back end changes the bias of KF i after its preintegration was computed. The factor
    // must correct the deltas for (bias of KF i) - (linearization bias), not ignore it.
    const Eigen::Vector3d b0a(0.05, -0.02, 0.03), b0g(0.01, -0.02, 0.015);
    const Eigen::Vector3d da(0.004, -0.003, 0.002), dg(0.0015, 0.001, -0.002);
    ImuChain c0 = integrateChain(_imu_cfg, b0a, b0g);           // integrated at b0
    ImuChain c1 = integrateChain(_imu_cfg, b0a + da, b0g + dg); // reference: integrated at b0 + d
    EXPECT_NEAR((c0.imu_j->getBgLin() - b0g).norm(), 0, 1e-12);

    // Same KF poses and velocities for both (those of c1, where the reference factor is ~0)
    c0.frames.front()->setWorld2FrameTransform(c1.frames.front()->getWorld2FrameTransform());
    c0.frames.back()->setWorld2FrameTransform(c1.frames.back()->getWorld2FrameTransform());
    c0.imu_i->setVelocity(c1.imu_i->getVelocity());
    c0.imu_j->setVelocity(c1.imu_j->getVelocity());
    c0.imu_i->setBa(b0a + da); // the back end moved the bias of KF i
    c0.imu_i->setBg(b0g + dg);

    PoseParametersBlock dX_i(Eigen::Affine3d::Identity()), dX_j(Eigen::Affine3d::Identity());
    PointXYZParametersBlock dvi(Eigen::Vector3d::Zero()), dvj(Eigen::Vector3d::Zero());
    PointXYZParametersBlock dba(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero());
    std::vector<double *> params = {dX_i.values(), dX_j.values(), dvi.values(), dvj.values(), dba.values(), dbg.values()};

    IMUFactor f0(c0.imu_i, c0.imu_j), f1(c1.imu_i, c1.imu_j);
    Eigen::Matrix<double, 9, 1> r0, r1;
    f0.Evaluate(params.data(), r0.data(), nullptr);
    f1.Evaluate(params.data(), r1.data(), nullptr);

    // Ignoring the bias change (previous behaviour) = an increment cancelling it
    Eigen::Map<Eigen::Vector3d>(dba.values()) = -da;
    Eigen::Map<Eigen::Vector3d>(dbg.values()) = -dg;
    Eigen::Matrix<double, 9, 1> r_ignored;
    f0.Evaluate(params.data(), r_ignored.data(), nullptr);

    EXPECT_LT((r0 - r1).norm(), 0.02 * (r_ignored - r1).norm())
        << "first-order correction " << (r0 - r1).norm() << " vs ignored " << (r_ignored - r1).norm();
}

TEST_F(ImuTest, repropagateEqualsFreshIntegration) {

    const Eigen::Vector3d b0a(0.05, -0.02, 0.03), b0g(0.01, -0.02, 0.015);
    const Eigen::Vector3d b1a(0.25, 0.1, -0.1), b1g(0.04, 0.03, -0.05); // beyond first order
    ImuChain c0 = integrateChain(_imu_cfg, b0a, b0g);
    ImuChain c1 = integrateChain(_imu_cfg, b1a, b1g);

    ASSERT_TRUE(c0.imu_j->repropagate(b1a, b1g));
    EXPECT_NEAR((c0.imu_j->getDeltaR() - c1.imu_j->getDeltaR()).norm(), 0, 1e-12);
    EXPECT_NEAR((c0.imu_j->getDeltaV() - c1.imu_j->getDeltaV()).norm(), 0, 1e-12);
    EXPECT_NEAR((c0.imu_j->getDeltaP() - c1.imu_j->getDeltaP()).norm(), 0, 1e-12);
    EXPECT_NEAR((c0.imu_j->getCov() - c1.imu_j->getCov()).norm(), 0, 1e-12);
    EXPECT_NEAR((c0.imu_j->_J_dR_bg - c1.imu_j->_J_dR_bg).norm(), 0, 1e-12);
    EXPECT_NEAR((c0.imu_j->_J_dp_bg - c1.imu_j->_J_dp_bg).norm(), 0, 1e-12);
    EXPECT_NEAR(c0.imu_j->getIntegratedDt(), c1.imu_j->getIntegratedDt(), 1e-12);
    EXPECT_NEAR((c0.imu_j->getBgLin() - b1g).norm(), 0, 1e-12);
}

TEST_F(ImuTest, droppingAKeyframeKeepsItsImuInformation) {

    // Issue 5: when a low-parallax KF is removed, the next KF's preintegration is integrated from the KF
    // before it, instead of cutting the IMU link.
    const Eigen::Vector3d ba(0.02, -0.01, 0.03), bg(0.01, 0.005, -0.01);
    SimKFs three = simulateKFs(_imu_cfg, ba, bg, 3, 60); // KFs at samples 0, 60, 120
    SimKFs two   = simulateKFs(_imu_cfg, ba, bg, 2, 120); // KFs at samples 0, 120: the reference

    std::shared_ptr<Frame> kf0 = three.kfs[0], kf1 = three.kfs[1], kf2 = three.kfs[2];
    EXPECT_TRUE(AOptimizer::imuFactorUsable(kf1, kf2));
    EXPECT_FALSE(AOptimizer::imuFactorUsable(kf0, kf2));

    kf2->getIMU()->setLastKF(kf0);
    ASSERT_TRUE(kf2->getIMU()->repropagate(kf0->getIMU()->getBa(), kf0->getIMU()->getBg()));
    std::shared_ptr<IMU> ref = two.kfs[1]->getIMU();
    EXPECT_NEAR((kf2->getIMU()->getDeltaR() - ref->getDeltaR()).norm(), 0, 1e-12);
    EXPECT_NEAR((kf2->getIMU()->getDeltaV() - ref->getDeltaV()).norm(), 0, 1e-12);
    EXPECT_NEAR((kf2->getIMU()->getDeltaP() - ref->getDeltaP()).norm(), 0, 1e-12);
    EXPECT_NEAR((kf2->getIMU()->getCov() - ref->getCov()).norm(), 0, 1e-12);
    EXPECT_NEAR(kf2->getIMU()->getIntegratedDt(), ref->getIntegratedDt(), 1e-12);

    // The factor admission follows the new link
    EXPECT_TRUE(AOptimizer::imuFactorUsable(kf0, kf2));
    EXPECT_FALSE(AOptimizer::imuFactorUsable(kf1, kf2));
}

/*!
 * @brief Simulated KFs (simulateKFs) with cameras (stereo: left at the frame origin, right 10 cm aside) and a grid
 * of landmarks in front of each KF, observed by it and the next three KFs where they project. Features are the
 * exact projections, so the true states are the optimum of every window problem.
 */
struct VIScene {
    SimKFs sim;
    std::shared_ptr<LocalMap> map;
    std::vector<Eigen::Affine3d> T_f_w_true;
};

static VIScene buildVIScene(const std::shared_ptr<imu_config> &cfg,
                            bool stereo,
                            int n_kf,
                            double pixel_noise = 0,
                            size_t window   = 0) {
    VIScene sc;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0, pixel_noise > 0 ? pixel_noise : 1);
    sc.sim = simulateKFs(cfg, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), n_kf, 60);
    std::vector<std::shared_ptr<Frame>> &kfs = sc.sim.kfs;

    Eigen::Matrix3d K;
    K << 300, 0, 400, 0, 300, 400, 0, 0, 1;
    for (auto &kf : kfs) {
        std::vector<std::shared_ptr<ImageSensor>> cams;
        cams.push_back(std::make_shared<Camera>(cv::Mat::zeros(800, 800, CV_8U), K));
        if (stereo)
            cams.push_back(std::make_shared<Camera>(cv::Mat::zeros(800, 800, CV_8U), K));
        kf->init(cams, kf->getTimestamp()); // keeps the IMU, adds the cameras
        cams[0]->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        if (stereo) {
            Eigen::Affine3d T_r_f   = Eigen::Affine3d::Identity();
            T_r_f.translation().x() = -0.1;
            cams[1]->setFrame2SensorTransform(T_r_f);
        }
        sc.T_f_w_true.push_back(kf->getWorld2FrameTransform());
    }

    for (size_t i = 0; i < kfs.size(); i++) {
        for (int a = -2; a <= 2; a++) {
            for (int b = -2; b <= 2; b++) {
                Eigen::Affine3d T_w_l = Eigen::Affine3d::Identity();
                T_w_l.translation() =
                    kfs[i]->getFrame2WorldTransform() * Eigen::Vector3d(0.6 * a, 0.6 * b, 4.0 + 0.3 * ((a + b + 5) % 3));
                std::shared_ptr<ALandmark> lmk =
                    std::make_shared<Point3D>(T_w_l, std::vector<std::shared_ptr<AFeature>>());
                for (size_t j = i; j < std::min(i + 4, kfs.size()); j++) {
                    bool seen = false;
                    for (auto &cam : kfs[j]->getSensors()) {
                        std::vector<Eigen::Vector2d> p2d;
                        if (!cam->project(T_w_l, lmk->getModel(), p2d))
                            continue;
                        if (pixel_noise > 0)
                            p2d[0] += Eigen::Vector2d(noise(rng), noise(rng));
                        std::shared_ptr<AFeature> feat = std::make_shared<Point2D>(p2d);
                        cam->addFeature("pointxd", feat);
                        lmk->addFeature(feat);
                        seen = true;
                    }
                    if (seen)
                        kfs[j]->addLandmark(lmk);
                }
                lmk->setInlier(); // LocalMap::addFrame puts it in the map
            }
        }
    }

    // Window with the landmarks (the first `window` KFs, all by default), first KF fixed
    sc.map = std::make_shared<LocalMap>(0, n_kf + 1, 1);
    for (size_t i = 0; i < (window ? window : kfs.size()); i++)
        sc.map->addFrame(kfs[i]);

    // Start away from the truth (except the fixed first KF)
    for (size_t i = 1; i < kfs.size(); i++) {
        Eigen::Affine3d T_w_f = kfs[i]->getFrame2WorldTransform();
        Eigen::Affine3d dT    = Eigen::Affine3d::Identity();
        dT.linear()           = geometry::exp_so3(0.01 * Eigen::Vector3d(1, -1, 0.5));
        dT.translation()      = Eigen::Vector3d(0.02, -0.01, 0.015);
        kfs[i]->setWorld2FrameTransform((T_w_f * dT).inverse());
        kfs[i]->getIMU()->setVelocity(sc.sim.kf_velocities[i] + Eigen::Vector3d(0.03, -0.02, 0.01));
    }
    kfs[0]->getIMU()->setVelocity(sc.sim.kf_velocities[0]);
    return sc;
}

static double maxPoseError(const VIScene &sc, size_t from) {
    double err = 0;
    for (size_t i = from; i < sc.sim.kfs.size(); i++) {
        Eigen::Affine3d d = sc.T_f_w_true[i] * sc.sim.kfs[i]->getFrame2WorldTransform();
        err = std::max({err, d.translation().norm(), geometry::log_so3(d.rotation()).norm()});
    }
    return err;
}

// Issue 20: every supported combination of optimizer x marginalization x sparsification x estimate_td, mono and
// stereo, inserts the expected factors and recovers the true window states. The Numeric optimizer has no
// marginalization and no time offset (refused at config load), so those combinations are not run.
TEST_F(ImuTest, everySupportedOptionCombinationInsertsItsFactors) {
    struct Optim {
        std::string name;
        std::function<std::shared_ptr<AOptimizer>()> make;
        bool supports_marg_td;
    };
    const std::vector<Optim> optims = {
        {"AngularAnalytic", [] { return std::make_shared<AngularAdjustmentCERESAnalytic>(); }, true},
        {"Analytic", [] { return std::make_shared<BundleAdjustmentCERESAnalytic>(); }, true},
        {"Numeric", [] { return std::make_shared<BundleAdjustmentCERESNumeric>(); }, false}};
    const int n_kf = 7;

    for (bool stereo : {true, false}) {
        for (const Optim &o : optims) {
            for (int marg = 0; marg <= 1; marg++) {
                for (int sparse = 0; sparse <= marg; sparse++) {
                    for (int td = 0; td <= 1; td++) {
                        if (!o.supports_marg_td && (marg || td))
                            continue;
                        const std::string label = std::string(stereo ? "stereo " : "mono ") + o.name +
                                                  " marg=" + std::to_string(marg) + " sparse=" +
                                                  std::to_string(sparse) + " td=" + std::to_string(td);
                        VIScene sc                        = buildVIScene(_imu_cfg, stereo, n_kf);
                        std::shared_ptr<AOptimizer> optim = o.make();
                        auto optimize = [&]() {
                            if (td) {
                                double t = 0;
                                optim->localMapVIOptimizationTd(sc.map, t, 1);
                                EXPECT_LT(std::abs(t), 1e-3) << label << ": time offset " << t;
                            } else {
                                optim->localMapVIOptimization(sc.map, 1);
                            }
                        };

                        optimize();
                        VIOptimStats st = optim->getLastVIStats();
                        EXPECT_TRUE(st.usable) << label;
                        EXPECT_EQ(st.n_imu_factors, uint(n_kf - 1)) << label;
                        EXPECT_EQ(st.n_prior_factors, 0u) << label;
                        EXPECT_GT(st.n_other_factors, 100u) << label << ": visual factors";
                        EXPECT_LT(maxPoseError(sc, 1), 2e-3) << label;

                        if (!marg)
                            continue;
                        ASSERT_TRUE(optim->marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), sparse))
                            << label;
                        sc.map->discardLastFrame();
                        optimize();
                        st = optim->getLastVIStats();
                        EXPECT_TRUE(st.usable) << label << " after marginalization";
                        EXPECT_EQ(st.n_imu_factors, uint(n_kf - 2)) << label << " after marginalization";
                        EXPECT_GT(st.n_prior_factors, 0u) << label << " after marginalization";
                        EXPECT_LT(maxPoseError(sc, 1), 2e-3) << label << " after marginalization";
                    }
                }
            }
        }
    }
}

static double maxVelocityError(const VIScene &sc, size_t from) {
    double err = 0;
    for (size_t i = from; i < sc.sim.kfs.size(); i++)
        err = std::max(err, (sc.sim.kfs[i]->getIMU()->getVelocity() - sc.sim.kf_velocities[i]).norm());
    return err;
}

static std::vector<Eigen::Affine3d> windowPoses(const VIScene &sc, size_t from) {
    std::vector<Eigen::Affine3d> T;
    for (size_t i = from; i < sc.sim.kfs.size(); i++)
        T.push_back(sc.sim.kfs[i]->getWorld2FrameTransform());
    return T;
}

// The dense marginalization prior is linearized at the states of the moment it is built. When it is used
// again after the window moved those states (the next marginalization folds it in; a low-parallax KF drop
// re-uses it in a second window), it must measure the states from its linearization point, not from the
// state at the start of each window.
TEST_F(ImuTest, marginalizationPriorKeepsItsLinearizationPoint) {
    for (bool stereo : {true, false}) {
        const std::string label = stereo ? "stereo" : "mono";
        VIScene sc = buildVIScene(_imu_cfg, stereo, 8);
        // KF 1 becomes the fixed (gauge) frame of the window once KF 0 is marginalized: start it at the truth
        sc.sim.kfs[1]->setWorld2FrameTransform(sc.T_f_w_true[1]);
        AngularAdjustmentCERESAnalytic optim;

        // Prior built away from the optimum (perturbed states), then the window converges
        ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), false)) << label;
        sc.map->discardLastFrame();
        optim.localMapVIOptimization(sc.map, 1);
        EXPECT_GT(optim.getLastVIStats().n_prior_factors, 0u) << label;

        // Re-optimizing the converged window with the same prior must not move it
        const std::vector<Eigen::Affine3d> T_before = windowPoses(sc, 1);
        optim.localMapVIOptimization(sc.map, 1);
        const std::vector<Eigen::Affine3d> T_after = windowPoses(sc, 1);
        double moved = 0;
        for (size_t i = 0; i < T_before.size(); i++)
            moved = std::max(moved, (T_before[i] * T_after[i].inverse()).translation().norm());
        EXPECT_LT(moved, 1e-3) << label << ": a converged window moved when optimized again";

        // A second marginalization folds the first prior in: the chain still ends at the truth (exact data)
        ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), false)) << label;
        sc.map->discardLastFrame();
        optim.localMapVIOptimization(sc.map, 1);
        EXPECT_LT(maxPoseError(sc, 2), 3e-3) << label;
        EXPECT_LT(maxVelocityError(sc, 2), 1e-2) << label;
    }
}

// Gives the test access to the dense prior built by marginalize()
struct MarginalizationAccess : public AngularAdjustmentCERESAnalytic {
    std::shared_ptr<Marginalization> prior() const { return _marginalization; }
};

TEST_F(ImuTest, marginalizationFactorJacobiansAwayFromZero) {
    // The parameter blocks of the dense prior are additive increments: its rotation Jacobian
    // d Log(A Exp(x)) / dx = Jr^-1(Log(A Exp(x))) Jr(x) was exact only at x = 0 (missing Jr(x)).
    VIScene sc = buildVIScene(_imu_cfg, true, 8);
    MarginalizationAccess optim;
    ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), false));
    sc.map->discardLastFrame();
    optim.localMapVIOptimization(sc.map, 1); // the kept states move away from the linearization point

    MarginalizationFactor f(optim.prior());
    ASSERT_GT(f.parameter_block_sizes().size(), 4u) << "frame blocks and landmarks";
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<std::vector<double>> blocks;
    for (size_t i = 0; i < f.parameter_block_sizes().size(); i++) {
        std::vector<double> b(f.parameter_block_sizes()[i]);
        for (double &x : b)
            x = (i == 0 ? 0.15 : 0.05) * u(rng); // pose increment up to 0.15 rad / 0.15 m
        blocks.push_back(b);
    }
    std::vector<double *> params;
    for (auto &b : blocks)
        params.push_back(b.data());
    EXPECT_TRUE(isae_test::JacobiansMatch(f, params, 1e-5));
}

TEST_F(ImuTest, inertialPriorWithoutKeptLandmarks) {
    // A VIO marginalization that keeps no landmark (here all are outliers, as when vision fails) still gives a
    // prior on the kept frame's pose, velocity and biases. It was dropped: the window, the next marginalization and
    // the sparse path all required kept landmarks to consider that a prior existed.
    for (bool sparse : {false, true}) {
        VIScene sc = buildVIScene(_imu_cfg, true, 8);
        for (auto &f : sc.map->getFrames())
            for (auto &tl : f->getLandmarks())
                for (auto &lmk : tl.second)
                    lmk->setOutlier();
        MarginalizationAccess optim;
        ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), sparse));
        EXPECT_TRUE(optim.prior()->_lmk_to_keep.empty());
        EXPECT_TRUE(optim.prior()->_frame_to_keep);
        EXPECT_TRUE(optim.prior()->_has_prior);
        if (sparse) {
            // The kept frame's prior leaves the gauge unobserved: its sparse factor came from a plain inverse of a
            // rank-deficient covariance (NaN on EuRoC, absurd weights here). It must not claim more information
            // than the dense prior has on those states.
            std::shared_ptr<Marginalization> m = optim.prior();
            const Eigen::MatrixXd S            = m->_map_frame_inf.at(m->_frame_to_keep);
            const Eigen::MatrixXd Jf =
                m->_marginalization_jacobian.middleCols(m->_map_frame_idx.at(m->_frame_to_keep), 15);
            const double dense_max  = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Jf.transpose() * Jf).eigenvalues().maxCoeff();
            const double sparse_max = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(S.transpose() * S).eigenvalues().maxCoeff();
            ASSERT_TRUE(S.allFinite());
            EXPECT_LT(sparse_max, 10 * dense_max) << "sparse " << sparse_max << " vs dense " << dense_max;
            // Nor less: with no kept landmark the absolute factor is the whole prior, so it must represent it
            // exactly. The covariance route lost the precise directions (rotation, velocity, biases) next to the
            // unobserved ones
            const Eigen::Matrix<double, 15, 15> Ja =
                Marginalization::absolutePriorJacobian(m->_frame_to_keep->getWorld2FrameTransform());
            const Eigen::MatrixXd A_sparse = Ja.transpose() * S.transpose() * S * Ja;
            EXPECT_LT((A_sparse - Jf.transpose() * Jf).norm() / (Jf.transpose() * Jf).norm(), 1e-6);
        }
        sc.map->discardLastFrame();
        optim.localMapVIOptimization(sc.map, 1);
        EXPECT_GT(optim.getLastVIStats().n_prior_factors, 0u) << (sparse ? "sparse" : "dense");
        // Without landmarks the kept frame's prior leaves the gauge unobserved: the sparse factor's information
        // came from a plain inverse of a rank-deficient covariance (NaN, every window solve failed)
        EXPECT_TRUE(optim.getLastVIStats().usable) << (sparse ? "sparse" : "dense") << ": "
                                                   << optim.getLastVIStats().termination;

        // The next marginalization folds it in and still has a prior
        ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), sparse));
        EXPECT_TRUE(optim.prior()->_has_prior);
    }
}

// A loop closure moves the sliding window into the corrected frame by C (yaw + translation, or a full rotation): the
// marginalization prior, expressed in the new frame (Marginalization::transformWorld), must give the moved states the
// cost the old states had. It was dropped at each correction instead; the prior left as it was does not follow
TEST_F(ImuTest, priorFollowsTheWindowWhenTheWorldMoves) {
    for (int variant = 0; variant < 2; variant++) {
        VIScene sc = buildVIScene(_imu_cfg, true, 8);
        MarginalizationAccess optim;
        ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), false));
        sc.map->discardLastFrame();
        optim.localMapVIOptimization(sc.map, 1); // the kept states leave the linearization point
        ASSERT_TRUE(optim.prior()->_has_prior);
        ASSERT_FALSE(optim.prior()->_lmk_to_keep.empty()) << "the test needs kept landmarks";

        Eigen::Affine3d C = Eigen::Affine3d::Identity();
        C.linear()        = Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        if (variant == 1)
            C.linear() = C.linear() * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                         Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
        C.translation() = Eigen::Vector3d(3, -2, 0.5);

        // The window's states, restored before each perturbation
        std::vector<std::shared_ptr<Frame>> frames(sc.map->getFrames().begin(), sc.map->getFrames().end());
        std::vector<Eigen::Affine3d> T0;
        std::vector<Eigen::Vector3d> v0;
        for (auto &f : frames) {
            T0.push_back(f->getWorld2FrameTransform());
            v0.push_back(f->getIMU() ? f->getIMU()->getVelocity() : Eigen::Vector3d::Zero());
        }
        std::vector<std::shared_ptr<ALandmark>> lmks;
        for (auto &tl : sc.map->getLandmarks())
            for (auto &l : tl.second)
                lmks.push_back(l);
        std::vector<Eigen::Affine3d> L0;
        for (auto &l : lmks)
            L0.push_back(l->getPose());

        auto residual = [](const std::shared_ptr<Marginalization> &m) {
            MarginalizationFactor f(m); // offsets from the current states
            std::vector<std::vector<double>> blocks;
            std::vector<double *> params;
            for (int sz : f.parameter_block_sizes())
                blocks.emplace_back(sz, 0.0);
            for (auto &b : blocks)
                params.push_back(b.data());
            Eigen::VectorXd r(f.num_residuals());
            f.Evaluate(params.data(), r.data(), nullptr);
            return r;
        };
        const auto orig  = std::make_shared<Marginalization>(*optim.prior());
        const auto moved = std::make_shared<Marginalization>(*optim.prior());
        moved->transformWorld(C);

        std::mt19937 rng(3 + variant);
        std::uniform_real_distribution<double> u(-0.02, 0.02);
        auto rnd = [&]() { return Eigen::Vector3d(u(rng), u(rng), u(rng)); };
        for (int k = 0; k < 3; k++) {
            // Perturbed states (up to 0.02 rad / m / m/s away from where the window left them)
            for (size_t i = 0; i < frames.size(); i++) {
                Eigen::Affine3d T = T0[i];
                T.linear()        = T.linear() * geometry::exp_so3(rnd());
                T.translation() += rnd();
                frames[i]->setWorld2FrameTransform(T);
                if (frames[i]->getIMU())
                    frames[i]->getIMU()->setVelocity(v0[i] + rnd());
            }
            for (size_t i = 0; i < lmks.size(); i++) {
                Eigen::Affine3d T = L0[i];
                T.translation() += rnd();
                lmks[i]->setPose(T);
            }
            const Eigen::VectorXd r_old = residual(orig);

            // The same states in the moved world (points only translate)
            for (auto &f : frames) {
                f->setWorld2FrameTransform((C * f->getFrame2WorldTransform()).inverse());
                if (f->getIMU())
                    f->getIMU()->setVelocity(C.rotation() * f->getIMU()->getVelocity());
            }
            for (auto &l : lmks) {
                Eigen::Affine3d T = l->getPose();
                T.translation()   = C * T.translation();
                l->setPose(T);
            }
            const Eigen::VectorXd r_new  = residual(moved);
            const Eigen::VectorXd r_kept = residual(orig);
            const double tol             = 1e-4 * r_old.norm() + 1e-6;
            EXPECT_LT((r_new - r_old).norm(), tol) << "variant " << variant << ", state " << k;
            EXPECT_GT((r_kept - r_old).norm(), 100 * tol) << "the prior left as it was must not follow";
        }

        // Its Jacobians with the world change (mapped increments, the translation's dependence on the rotation)
        MarginalizationFactor fm(moved);
        std::vector<std::vector<double>> blocks;
        for (size_t i = 0; i < fm.parameter_block_sizes().size(); i++) {
            std::vector<double> b(fm.parameter_block_sizes()[i]);
            for (double &x : b)
                x = (i == 0 ? 0.1 : 0.05) * u(rng) / 0.02;
            blocks.push_back(b);
        }
        std::vector<double *> params;
        for (auto &b : blocks)
            params.push_back(b.data());
        EXPECT_TRUE(isae_test::JacobiansMatch(fm, params, 1e-5)) << "variant " << variant;
    }
}

// The sparse prior's absolute factor (IMUPriordx, built as addSparsePriorResiduals does) follows the window too: its
// pose error is relative to the linearization pose, its velocity error is rotated into the current world
TEST_F(ImuTest, sparsePriorFollowsTheWindowWhenTheWorldMoves) {
    VIScene sc = buildVIScene(_imu_cfg, true, 8);
    MarginalizationAccess optim;
    ASSERT_TRUE(optim.marginalize(sc.map->getFrames().at(0), sc.map->getFrames().at(1), true));
    sc.map->discardLastFrame();
    optim.localMapVIOptimization(sc.map, 1);
    const auto m = optim.prior();
    const std::shared_ptr<Frame> f = m->_frame_to_keep;
    ASSERT_TRUE(f && f->getIMU() && m->_map_frame_inf.count(f));

    auto residual = [&]() {
        Eigen::MatrixXd S = m->_map_frame_inf.at(f);
        S.middleCols(6, 3) = S.middleCols(6, 3) * m->_W_prior.rotation();
        IMUPriordx c(f->getWorld2FrameTransform(), m->linPoseCurrent(), f->getIMU()->getVelocity(),
                     m->linVelocityCurrent(), f->getIMU()->getBa(), m->_ba_lin, f->getIMU()->getBg(), m->_bg_lin, S);
        double zero[6] = {0, 0, 0, 0, 0, 0}, z3[3] = {0, 0, 0};
        const double *params[4] = {zero, z3, z3, z3};
        Eigen::Matrix<double, 15, 1> r;
        c.Evaluate(params, r.data(), nullptr);
        return r;
    };
    const Eigen::Matrix<double, 15, 1> r_old = residual();
    Eigen::Affine3d C = Eigen::Affine3d::Identity();
    C.linear()        = Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    C.translation()   = Eigen::Vector3d(3, -2, 0.5);
    f->setWorld2FrameTransform((C * f->getFrame2WorldTransform()).inverse());
    f->getIMU()->setVelocity(C.rotation() * f->getIMU()->getVelocity());
    m->transformWorld(C);
    EXPECT_LT((residual() - r_old).norm(), 1e-9 * (1 + r_old.norm()));
}

TEST(ImuConfigTest, noiseKeysAreReadIntoTheRightFields) {

    // Issue 1: the gyroscope random walk was read from accelerometer_random_walk.
    const std::string path = "imu_config_test.yaml";
    std::ofstream f(path);
    f << "ncam: 0\n"
      << "imu:\n"
      << "  topic: /imu0\n"
      << "  T_BS:\n"
      << "    data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n"
      << "  rate_hz: 200\n"
      << "  gyroscope_noise_density: 0.1\n"
      << "  gyroscope_random_walk: 0.2\n"
      << "  accelerometer_noise_density: 0.3\n"
      << "  accelerometer_random_walk: 0.4\n"
      << "  dt_imu_cam: 0.0\n";
    f.close();

    Config cfg;
    cfg.slam_mode = "bimonovio";
    ADataProvider prov(path, cfg);
    std::shared_ptr<imu_config> c = prov.getIMUConfig();
    ASSERT_TRUE(c);
    EXPECT_DOUBLE_EQ(c->gyr_noise, 0.1);
    EXPECT_DOUBLE_EQ(c->bgyr_noise, 0.2);
    EXPECT_DOUBLE_EQ(c->acc_noise, 0.3);
    EXPECT_DOUBLE_EQ(c->bacc_noise, 0.4);
    std::remove(path.c_str());
}

TEST(ImuConfigTest, measurementsAreRotatedIntoTheBodyFrame) {

    // Issue 6: T_BS must be applied by every reader; it now happens in createImuSensor.
    const std::string path = "imu_tbs_test.yaml";
    std::ofstream f(path);
    f << "ncam: 0\n"
      << "imu:\n"
      << "  topic: /imu0\n"
      << "  T_BS:\n" // R_x(+90 deg): sensor y -> body z
      << "    data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n"
      << "  rate_hz: 200\n"
      << "  gyroscope_noise_density: 0.1\n"
      << "  gyroscope_random_walk: 0.2\n"
      << "  accelerometer_noise_density: 0.3\n"
      << "  accelerometer_random_walk: 0.4\n"
      << "  dt_imu_cam: 0.0\n";
    f.close();

    Config cfg;
    cfg.slam_mode = "monovio";
    ADataProvider prov(path, cfg);
    std::shared_ptr<IMU> imu = prov.createImuSensor(Eigen::Vector3d(0, 9.81, 0), Eigen::Vector3d(0.1, 0.2, 0.3));
    EXPECT_NEAR((imu->getAcc() - Eigen::Vector3d(0, 0, 9.81)).norm(), 0, 1e-12);
    EXPECT_NEAR((imu->getGyr() - Eigen::Vector3d(0.1, -0.3, 0.2)).norm(), 0, 1e-12);
    std::remove(path.c_str());
}

} // namespace isae
