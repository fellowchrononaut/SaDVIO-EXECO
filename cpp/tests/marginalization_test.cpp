#include <gtest/gtest.h>

#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"

#include <opencv2/core.hpp>

#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"

namespace isae {

class MarginalizationTest : public testing::Test {
    /*
    * Simple Toy Example for Marginalization

    *   l0   l1
    *   |  /    \
    *   | /      \
    *   x0        x1
    *     \      /
    *      \    /
    *        l2
    * */
  public:
    void SetUp() override {
        std::srand(12345u); // same random state for every test, whatever the run order

        // Set Frames
        _frame0 = std::shared_ptr<Frame>(new Frame());
        _frame1 = std::shared_ptr<Frame>(new Frame());

        // Set Sensors
        _K       = Eigen::Matrix3d::Identity();
        _K(0, 0) = 100;
        _K(1, 1) = 100;
        _K(0, 2) = 400;
        _K(1, 2) = 400;

        // We set the two sensors for frame 0 (We only marginalize stereo factors)
        _sensor0l = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor0r = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0l);
        sensors_frame0.push_back(_sensor0r);
        _frame0->init(sensors_frame0, 0);
        _sensor0l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_s01_f0   = Eigen::Affine3d::Identity();
        T_s01_f0.translation().y() = 0.2;
        _sensor0r->setFrame2SensorTransform(T_s01_f0);

        // We set the two sensors for frame 0 (We only marginalize stereo factors)
        _sensor1l = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        _sensor1r = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(800, 800, CV_16F), _K));
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame1;
        sensors_frame1.push_back(_sensor1l);
        sensors_frame1.push_back(_sensor1r);
        _frame1->init(sensors_frame1, 1.0);
        _sensor1l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_s11_f1   = Eigen::Affine3d::Identity();
        T_s11_f1.translation().y() = 0.2;
        _sensor1r->setFrame2SensorTransform(T_s11_f1);

        // Set frame pose
        _frame0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
        Eigen::Affine3d T_w_f1 = Eigen::Affine3d::Identity();
        T_w_f1.translation()   = Eigen::Vector3d(0, 0, 1);
        _frame1->setWorld2FrameTransform(T_w_f1.inverse());

        // Set landmark 0
        Eigen::Affine3d T_w_l0 = Eigen::Affine3d::Identity();
        T_w_l0.translation()   = Eigen::Vector3d(0.5, 0, 2);
        std::vector<std::shared_ptr<AFeature>> feat_vec;
        _lmk_0 = std::shared_ptr<Point3D>(new Point3D(T_w_l0, feat_vec));
        _frame0->addLandmark(_lmk_0);
        _lmk_0->setInMap();
        _lmk_0->setInlier();

        // Set landmark 1
        Eigen::Affine3d T_w_l1 = Eigen::Affine3d::Identity();
        T_w_l1.translation()   = Eigen::Vector3d(-1, 0, 2);
        _lmk_1                 = std::shared_ptr<Point3D>(new Point3D(T_w_l1, feat_vec));
        _frame0->addLandmark(_lmk_1);
        _frame1->addLandmark(_lmk_1);
        _lmk_1->setInMap();
        _lmk_1->setInlier();

        // Set landmark 2
        Eigen::Affine3d T_w_l2 = Eigen::Affine3d::Identity();
        T_w_l2.translation()   = Eigen::Vector3d(1, 0, 2);
        _lmk_2                 = std::shared_ptr<Point3D>(new Point3D(T_w_l2, feat_vec));
        _frame0->addLandmark(_lmk_2);
        _frame1->addLandmark(_lmk_2);
        _lmk_2->setInMap();
        _lmk_2->setInlier();

        // Projections and add features for lmk 0
        std::vector<Eigen::Vector2d> projection_0_0_l;
        _sensor0l->project(_lmk_0->getPose(), _lmk_0->getModel(), projection_0_0_l);

        std::vector<Eigen::Vector2d> projection_0_0_r;
        _sensor0r->project(_lmk_0->getPose(), _lmk_0->getModel(), projection_0_0_r);

        _feat_0_0_l = std::shared_ptr<Point2D>(new Point2D(projection_0_0_l));
        _lmk_0->addFeature(_feat_0_0_l);
        _sensor0l->addFeature("pointxd", _feat_0_0_l);

        _feat_0_0_r = std::shared_ptr<Point2D>(new Point2D(projection_0_0_r));
        _lmk_0->addFeature(_feat_0_0_r);
        _sensor0r->addFeature("pointxd", _feat_0_0_r);

        // Projections and add features for lmk 1
        std::vector<Eigen::Vector2d> projection_0_1_l;
        _sensor0l->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_0_1_l);

        std::vector<Eigen::Vector2d> projection_0_1_r;
        _sensor0r->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_0_1_r);

        std::vector<Eigen::Vector2d> projection_1_1_l;
        _sensor1l->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_1_1_l);

        std::vector<Eigen::Vector2d> projection_1_1_r;
        _sensor1r->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_1_1_r);

        _feat_0_1_l = std::shared_ptr<Point2D>(new Point2D(projection_0_1_l));
        _lmk_1->addFeature(_feat_0_1_l);
        _sensor0l->addFeature("pointxd", _feat_0_1_l);

        _feat_0_1_r = std::shared_ptr<Point2D>(new Point2D(projection_0_1_r));
        _lmk_1->addFeature(_feat_0_1_r);
        _sensor0r->addFeature("pointxd", _feat_0_1_r);

        _feat_1_1_l = std::shared_ptr<Point2D>(new Point2D(projection_1_1_l));
        _lmk_1->addFeature(_feat_1_1_l);
        _sensor1l->addFeature("pointxd", _feat_1_1_l);

        _feat_1_1_r = std::shared_ptr<Point2D>(new Point2D(projection_1_1_r));
        _lmk_1->addFeature(_feat_1_1_r);
        _sensor1r->addFeature("pointxd", _feat_1_1_r);

        // Projections and add features for lmk 2
        std::vector<Eigen::Vector2d> projection_0_2_l;
        _sensor0l->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_0_2_l);

        std::vector<Eigen::Vector2d> projection_0_2_r;
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_0_2_r);

        std::vector<Eigen::Vector2d> projection_1_2_l;
        _sensor1l->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_1_2_l);

        std::vector<Eigen::Vector2d> projection_1_2_r;
        _sensor1r->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_1_2_r);

        _feat_0_2_l = std::shared_ptr<Point2D>(new Point2D(projection_0_2_l));
        _lmk_2->addFeature(_feat_0_2_l);
        _sensor0l->addFeature("pointxd", _feat_0_2_l);

        _feat_0_2_r = std::shared_ptr<Point2D>(new Point2D(projection_0_2_r));
        _lmk_2->addFeature(_feat_0_2_r);
        _sensor0r->addFeature("pointxd", _feat_0_2_r);

        _feat_1_2_l = std::shared_ptr<Point2D>(new Point2D(projection_1_2_l));
        _lmk_2->addFeature(_feat_1_2_l);
        _sensor1l->addFeature("pointxd", _feat_1_2_l);

        _feat_1_2_r = std::shared_ptr<Point2D>(new Point2D(projection_1_2_r));
        _lmk_2->addFeature(_feat_1_2_r);
        _sensor1r->addFeature("pointxd", _feat_1_2_r);
    }

    Eigen::Matrix3d _K;
    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<ImageSensor> _sensor0l;
    std::shared_ptr<ImageSensor> _sensor0r;
    std::shared_ptr<Frame> _frame1;
    std::shared_ptr<ImageSensor> _sensor1l;
    std::shared_ptr<ImageSensor> _sensor1r;
    std::shared_ptr<Point3D> _lmk_0;
    std::shared_ptr<Point2D> _feat_0_0_l;
    std::shared_ptr<Point2D> _feat_0_0_r;
    std::shared_ptr<Point3D> _lmk_1;
    std::shared_ptr<Point2D> _feat_0_1_l;
    std::shared_ptr<Point2D> _feat_0_1_r;
    std::shared_ptr<Point2D> _feat_1_1_l;
    std::shared_ptr<Point2D> _feat_1_1_r;
    std::shared_ptr<Point3D> _lmk_2;
    std::shared_ptr<Point2D> _feat_0_2_l;
    std::shared_ptr<Point2D> _feat_0_2_r;
    std::shared_ptr<Point2D> _feat_1_2_l;
    std::shared_ptr<Point2D> _feat_1_2_r;

    std::unordered_map<std::shared_ptr<Frame>, PoseParametersBlock> _map_frame_posepar;
    std::unordered_map<std::shared_ptr<ALandmark>, PointXYZParametersBlock> _map_lmk_ptpar;
    std::unordered_map<std::shared_ptr<ALandmark>, PoseParametersBlock> _map_lmk_posepar;

    Marginalization _marg;
};

// Is the setup all right
TEST_F(MarginalizationTest, setupTest) {

    // Check number of feats
    ASSERT_EQ(_sensor0r->getFeatures()["pointxd"].size(), 3);
    ASSERT_EQ(_sensor0l->getFeatures()["pointxd"].size(), 3);

    ASSERT_EQ(_sensor1l->getFeatures()["pointxd"].size(), 2);
    ASSERT_EQ(_sensor1r->getFeatures()["pointxd"].size(), 2);

    // Check if the map is good
    ASSERT_EQ(_frame0->getInMapLandmarksNumber(), 3);
}

// Pre Marginalization test
TEST_F(MarginalizationTest, preMargTest) {

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(_frame0, _frame1, marg_last);

    // Check all the variables in _marg
    ASSERT_EQ(_marg._n, 6);
    ASSERT_EQ(_marg._m, 9);
    ASSERT_EQ(_marg._frame_to_marg, _frame0);
    ASSERT_EQ(_marg._lmk_to_marg["pointxd"].at(0), _lmk_0);
    ASSERT_EQ(_marg._lmk_to_keep["pointxd"].size(), 2);
}

// We test the gradient and info mat computation as in the solver
TEST_F(MarginalizationTest, margTest) {

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(_frame0, _frame1, marg_last);

    // Create pose parameters for the frame to marginalize
    _map_frame_posepar.emplace(_frame0, PoseParametersBlock(_frame0->getWorld2FrameTransform()));

    ASSERT_EQ(_marg._lmk_to_keep["pointxd"].size(), 2);
    ASSERT_EQ(_marg._lmk_to_marg["pointxd"].size(), 1);

    // Create Marginalization Blocks with landmarks to keep
    for (auto tlmk : _marg._lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == _frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marg._map_frame_idx.at(feature.lock()->getSensor()->getFrame()));
                    parameter_blocks.push_back(_map_frame_posepar.at(feature.lock()->getSensor()->getFrame()).values());

                    // For the lmk
                    parameter_idx.push_back(_marg._map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    std::cout << feature.lock()->getPoints().size() << std::endl;
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());
                    _marg._marginalization_blocks.push_back(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marg._lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == _frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marg._map_frame_idx.at(feature.lock()->getSensor()->getFrame()));
                    parameter_blocks.push_back(_map_frame_posepar.at(feature.lock()->getSensor()->getFrame()).values());

                    // For the lmk
                    parameter_idx.push_back(_marg._map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());
                    _marg._marginalization_blocks.push_back(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Check the indices of the variables
    ASSERT_EQ(_marg._map_frame_idx.at(_frame0), 0);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_0), 6);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_1), 9);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_2), 12);

    _marg.computeSchurComplement();

    // Are indices updated ?
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_1), 0);
    ASSERT_EQ(_marg._map_lmk_idx.at(_lmk_2), 3);

    // Info Mat check
    ASSERT_EQ(_marg._Ak.size(), 36);
    ASSERT_NEAR((_marg._Ak - _marg._Ak.transpose()).norm(), 0, 1e-8);

    // We test off diag
    double off = _marg.computeOffDiag(_lmk_1, _lmk_2);
    ASSERT_GT(off, 0);
}

// Test the case where the frame to marginalize is not corelated
TEST_F(MarginalizationTest, margFailTest) {

    std::shared_ptr<Frame> frame2   = std::shared_ptr<Frame>(new Frame());
    std::shared_ptr<Camera> sensor2 = std::shared_ptr<Camera>(new Camera(cv::Mat::zeros(400, 400, CV_16F), _K));

    std::vector<std::shared_ptr<ImageSensor>> sensors_frame2;
    sensors_frame2.push_back(sensor2);
    frame2->init(sensors_frame2, 2.0);
    sensor2->setFrame2SensorTransform(Eigen::Affine3d::Identity());

    std::shared_ptr<Marginalization> marg_last = std::shared_ptr<Marginalization>(new Marginalization());
    _marg.preMarginalize(frame2, frame2, marg_last);

    ASSERT_EQ(_marg.computeSchurComplement(), false);
}

// Monocular frames: single-view landmarks seen by other frames are kept (Issue 16)
TEST(MarginalizationMonoTest, preMargKeepsSingleViewLandmarks) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = K(1, 1) = 100;
    K(0, 2) = K(1, 2) = 400;
    std::shared_ptr<Frame> f0 = std::make_shared<Frame>(), f1 = std::make_shared<Frame>();
    std::shared_ptr<Camera> c0 = std::make_shared<Camera>(cv::Mat::zeros(800, 800, CV_16F), K);
    std::shared_ptr<Camera> c1 = std::make_shared<Camera>(cv::Mat::zeros(800, 800, CV_16F), K);
    f0->init(std::vector<std::shared_ptr<ImageSensor>>{c0}, 0);
    f1->init(std::vector<std::shared_ptr<ImageSensor>>{c1}, 1);
    c0->setFrame2SensorTransform(Eigen::Affine3d::Identity());
    c1->setFrame2SensorTransform(Eigen::Affine3d::Identity());
    f0->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    Eigen::Affine3d T_w_f1 = Eigen::Affine3d::Identity();
    T_w_f1.translation()   = Eigen::Vector3d(0.2, 0, 0);
    f1->setWorld2FrameTransform(T_w_f1.inverse());

    // lmk 0 seen by f0 only, lmks 1 and 2 by both
    std::vector<std::shared_ptr<Point3D>> lmks;
    for (int i = 0; i < 3; i++) {
        Eigen::Affine3d T_w_l    = Eigen::Affine3d::Identity();
        T_w_l.translation()      = Eigen::Vector3d(-0.5 + 0.5 * i, 0.1, 2);
        std::shared_ptr<Point3D> l = std::make_shared<Point3D>(T_w_l, std::vector<std::shared_ptr<AFeature>>());
        l->setInMap();
        l->setInlier();
        for (auto [f, c] : {std::make_pair(f0, c0), std::make_pair(f1, c1)}) {
            if (i == 0 && f == f1)
                continue;
            std::vector<Eigen::Vector2d> p;
            c->project(l->getPose(), l->getModel(), p);
            std::shared_ptr<Point2D> feat = std::make_shared<Point2D>(p);
            l->addFeature(feat);
            c->addFeature("pointxd", feat);
            f->addLandmark(l);
        }
        lmks.push_back(l);
    }

    Marginalization marg;
    std::shared_ptr<Marginalization> marg_last = std::make_shared<Marginalization>();
    marg.preMarginalize(f0, f1, marg_last);
    EXPECT_EQ(marg._lmk_to_keep["pointxd"].size(), 2u);
    ASSERT_EQ(marg._lmk_to_marg["pointxd"].size(), 1u);
    EXPECT_EQ(marg._lmk_to_marg["pointxd"].at(0), lmks[0]);
    EXPECT_EQ(marg._n, 6);
}

// Issue 14: the information recovered by sparsification must use the Jacobians of the factors it builds
TEST(MarginalizationJacobianTest, recoveryJacobiansMatchTheFactors) {
    Eigen::Affine3d T_f_w = Eigen::Affine3d::Identity();
    T_f_w.linear()        = geometry::exp_so3(Eigen::Vector3d(0.2, 0.1, -0.3));
    T_f_w.translation()   = Eigen::Vector3d(0.5, -0.4, 0.2);
    const Eigen::Vector3d p(2, 1, 4);

    // Pose to landmark
    PoseToLandmarkFactor f_pl(T_f_w * p, T_f_w, p, Eigen::Matrix3d::Identity());
    double pose[6] = {0, 0, 0, 0, 0, 0}, dl[3] = {0, 0, 0};
    std::vector<double *> params_pl = {pose, dl};
    Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J_pose;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J_lmk;
    double *J_pl[2] = {J_pose.data(), J_lmk.data()};
    Eigen::Vector3d r3;
    f_pl.Evaluate(params_pl.data(), r3.data(), J_pl);
    Eigen::Matrix<double, 3, 9> J_rec = Marginalization::poseToLandmarkJacobian(T_f_w, p);
    EXPECT_NEAR((J_rec.block(0, 0, 3, 6) - J_pose).cwiseAbs().maxCoeff(), 0, 1e-12);
    EXPECT_NEAR((J_rec.block(0, 6, 3, 3) - J_lmk).cwiseAbs().maxCoeff(), 0, 1e-12);

    // Absolute inertial prior
    const Eigen::Vector3d v(0.1, 0.2, 0.3), ba(0.01, 0, 0), bg(0, 0.001, 0);
    IMUPriordx f_abs(T_f_w, T_f_w, v, v, ba, ba, bg, bg, Eigen::MatrixXd::Identity(15, 15));
    double dv[3] = {0, 0, 0}, dba[3] = {0, 0, 0}, dbg[3] = {0, 0, 0};
    std::vector<double *> params_abs = {pose, dv, dba, dbg};
    Eigen::Matrix<double, 15, 6, Eigen::RowMajor> J_p;
    Eigen::Matrix<double, 15, 3, Eigen::RowMajor> J_v, J_ba, J_bg;
    double *J_abs[4] = {J_p.data(), J_v.data(), J_ba.data(), J_bg.data()};
    Eigen::Matrix<double, 15, 1> r15;
    f_abs.Evaluate(params_abs.data(), r15.data(), J_abs);
    Eigen::Matrix<double, 15, 15> J_full;
    J_full << J_p, J_v, J_ba, J_bg;
    EXPECT_NEAR((Marginalization::absolutePriorJacobian(T_f_w) - J_full).cwiseAbs().maxCoeff(), 0, 1e-12);
}

// Issue 14: the sparse VIO prior (absolute inertial factor + pose-to-landmark factors) against the dense prior.
// With these factors the stacked Jacobian J is square and invertible, so the KLD-optimal information of each
// factor has the closed form (J_i Sigma J_i^T)^-1 (Mazuran et al.) used by sparsifyVIO():
// - a dense prior that has exactly this structure is recovered (KLD ~ 0);
// - for any dense prior, scaling the information of one factor away from it increases the KLD.
namespace {

struct SparseVIOSetup {
    Marginalization marg;
    std::vector<std::shared_ptr<ALandmark>> lmks;
    Eigen::MatrixXd J; // stacked Jacobian of the sparse factors (abs prior rows first)

    explicit SparseVIOSetup(int n_lmk) {
        Eigen::Affine3d T_f_w = Eigen::Affine3d::Identity();
        T_f_w.linear()        = geometry::exp_so3(Eigen::Vector3d(0.3, -0.2, 0.5));
        T_f_w.translation()   = Eigen::Vector3d(0.4, 0.3, -0.6);
        std::shared_ptr<Frame> f = std::make_shared<Frame>();
        f->setWorld2FrameTransform(T_f_w);

        marg._frame_to_keep = f;
        marg._map_frame_idx.emplace(f, 0);
        marg._n = 15 + 3 * n_lmk;
        J       = Eigen::MatrixXd::Zero(marg._n, marg._n);
        J.block(0, 0, 15, 15) = Marginalization::absolutePriorJacobian(T_f_w);
        for (int k = 0; k < n_lmk; k++) {
            Eigen::Affine3d T_w_l = Eigen::Affine3d::Identity();
            T_w_l.translation()   = Eigen::Vector3d(1.0 + k, -0.5 * k, 3.0 + 0.7 * k);
            std::shared_ptr<ALandmark> l = std::make_shared<Point3D>(T_w_l, std::vector<std::shared_ptr<AFeature>>());
            lmks.push_back(l);
            marg._lmk_to_keep["pointxd"].push_back(l);
            marg._map_lmk_idx.emplace(l, 15 + 3 * k);
            Eigen::Matrix<double, 3, 9> J_pl = Marginalization::poseToLandmarkJacobian(T_f_w, T_w_l.translation());
            J.block(15 + 3 * k, 0, 3, 6)          = J_pl.block(0, 0, 3, 6);
            J.block(15 + 3 * k, 15 + 3 * k, 3, 3) = J_pl.block(0, 6, 3, 3);
        }
    }

    void setDensePrior(const Eigen::MatrixXd &A) {
        marg.rankReveallingDecomposition(A, marg._U, marg._Lambda);
        marg._Sigma   = marg._Lambda.array().inverse();
        marg._n_full  = marg._U.cols();
        marg._Sigma_k = marg._U * marg._Sigma.asDiagonal() * marg._U.transpose();
    }

    // Information of the sparse prior, with the information of factor `scaled` (-1: none, 0: abs prior,
    // k > 0: landmark k-1) multiplied by `s`
    Eigen::MatrixXd sparseInformation(int scaled = -1, double s = 1) const {
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(marg._n, marg._n);
        Eigen::MatrixXd S = marg._map_frame_inf.at(marg._frame_to_keep);
        D.block(0, 0, 15, 15) = S * S * (scaled == 0 ? s : 1);
        for (size_t k = 0; k < lmks.size(); k++) {
            Eigen::Matrix3d Sl = marg._map_lmk_inf.at(lmks[k]);
            D.block(15 + 3 * k, 15 + 3 * k, 3, 3) = Sl * Sl * (scaled == int(k) + 1 ? s : 1);
        }
        return J.transpose() * D * J;
    }
};

} // namespace

TEST(MarginalizationSparseKLDTest, recoversADensePriorWithTheSparseStructure) {
    std::srand(12345u);
    SparseVIOSetup setup(4);

    // Dense prior generated by the sparse topology: J^T D J with SPD blocks
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(setup.marg._n, setup.marg._n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(15, 15);
    D.block(0, 0, 15, 15) = B * B.transpose() + 0.5 * Eigen::MatrixXd::Identity(15, 15);
    for (int k = 0; k < 4; k++) {
        Eigen::Matrix3d Bl = Eigen::Matrix3d::Random();
        D.block(15 + 3 * k, 15 + 3 * k, 3, 3) = 10 * (Bl * Bl.transpose() + 0.5 * Eigen::Matrix3d::Identity());
    }
    Eigen::MatrixXd A_dense = setup.J.transpose() * D * setup.J;
    setup.setDensePrior(A_dense);
    ASSERT_TRUE(setup.marg.sparsifyVIO());

    Eigen::MatrixXd A_sparse = setup.sparseInformation();
    EXPECT_LT((A_sparse - A_dense).norm() / A_dense.norm(), 1e-8);
    EXPECT_NEAR(setup.marg.computeKLD(A_dense, A_sparse), 0, 1e-6);
}

TEST(MarginalizationSparseKLDTest, sparsePriorIsTheKLDMinimum) {
    std::srand(12345u);
    SparseVIOSetup setup(4);

    // Generic dense prior (correlations the sparse topology cannot represent)
    Eigen::MatrixXd B       = Eigen::MatrixXd::Random(setup.marg._n, setup.marg._n);
    Eigen::MatrixXd A_dense = B * B.transpose() + 0.1 * Eigen::MatrixXd::Identity(setup.marg._n, setup.marg._n);
    setup.setDensePrior(A_dense);
    ASSERT_TRUE(setup.marg.sparsifyVIO());

    const double kld = setup.marg.computeKLD(A_dense, setup.sparseInformation());
    ASSERT_TRUE(std::isfinite(kld));
    EXPECT_GT(kld, 1e-3); // not representable exactly
    for (int factor = 0; factor <= 4; factor++) {
        for (double s : {0.95, 1.05}) {
            EXPECT_GT(setup.marg.computeKLD(A_dense, setup.sparseInformation(factor, s)), kld)
                << "factor " << factor << " scaled by " << s;
        }
    }
}

// Robust marginalization: a block with a loss gives the robust gradient rho' J^T r (and is unchanged where the
// loss is quadratic), so the information of an outlier is down-weighted before it enters the prior
namespace {
struct LinearResidual : public ceres::SizedCostFunction<2, 2> {
    Eigen::Vector2d target;
    explicit LinearResidual(const Eigen::Vector2d &t) : target(t) {}
    bool Evaluate(double const *const *x, double *r, double **J) const override {
        const Eigen::Matrix2d W = Eigen::Vector2d(2, 0.5).asDiagonal();
        Eigen::Map<Eigen::Vector2d> res(r);
        Eigen::Map<const Eigen::Vector2d> xv(x[0]);
        res = W * (xv - target);
        if (J && J[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> jac(J[0]);
            jac = W;
        }
        return true;
    }
};
} // namespace

TEST(MarginalizationRobustTest, robustBlockGivesTheRobustGradient) {
    for (double far : {0.1, 5.0}) { // inside / outside the quadratic zone of the Huber loss
        double x[2] = {0, 0};
        std::vector<double *> blocks = {x};
        std::shared_ptr<ceres::LossFunction> loss(new ceres::HuberLoss(1.0));
        MarginalizationBlockInfo plain(new LinearResidual(Eigen::Vector2d(far, -0.3 * far)), {0}, blocks);
        MarginalizationBlockInfo robust(new LinearResidual(Eigen::Vector2d(far, -0.3 * far)), {0}, blocks, loss);
        plain.Evaluate();
        robust.Evaluate();
        double rho[3];
        loss->Evaluate(plain._residuals.squaredNorm(), rho);
        const Eigen::Vector2d g_plain  = plain._jacobians[0].transpose() * plain._residuals;
        const Eigen::Vector2d g_robust = robust._jacobians[0].transpose() * robust._residuals;
        EXPECT_LT((g_robust - rho[1] * g_plain).norm(), 1e-9 * (1 + g_plain.norm())) << "far " << far;
        if (far < 1)
            EXPECT_LT((robust._jacobians[0] - plain._jacobians[0]).norm(), 1e-12);
        else
            EXPECT_LT(rho[1], 0.5); // the outlier is down-weighted
        delete plain._cost_function;
        delete robust._cost_function;
        delete[] plain._raw_jacobians;
        delete[] robust._raw_jacobians;
    }
}

// Only well-conditioned landmarks are linearized into the prior
TEST_F(MarginalizationTest, wellConditionedLandmarks) {
    // Fixture: landmarks ~2 m in front of two stereo frames 1 m apart: well conditioned
    EXPECT_TRUE(Marginalization::wellConditioned(_lmk_1));

    // Moved behind the cameras: the predicted bearings disagree with the measured ones
    Eigen::Affine3d T = _lmk_1->getPose();
    _lmk_1->setPose(Eigen::Translation3d(0, 0, -4) * T);
    EXPECT_FALSE(Marginalization::wellConditioned(_lmk_1));
    _lmk_1->setPose(T);

    // Same bearings but only a 0.2 m stereo baseline at 40 m: rays span ~0.3 deg, too little
    std::shared_ptr<Point3D> far = std::make_shared<Point3D>(
        Eigen::Affine3d(Eigen::Translation3d(0.5, 0, 40)), std::vector<std::shared_ptr<AFeature>>());
    for (auto &cam : {_sensor0l, _sensor0r}) {
        std::vector<Eigen::Vector2d> p2d;
        cam->project(far->getPose(), far->getModel(), p2d);
        std::shared_ptr<AFeature> f = std::make_shared<Point2D>(p2d);
        cam->addFeature("pointxd", f);
        far->addFeature(f);
    }
    EXPECT_FALSE(Marginalization::wellConditioned(far));
    EXPECT_TRUE(Marginalization::wellConditioned(far, 2.0 * M_PI / 180, 0.1 * M_PI / 180));
}

// Issue 15: a point-only map gets its sparse prior (the gating used to count landmark types)
struct SparsePriorProbe : public BundleAdjustmentCERESAnalytic {
    using AOptimizer::addSparsePriorResiduals;
};

// The dense VO prior was never added to the window by the AngularAnalytic optimizer: it skipped the prior when the
// frame to keep was not in the problem, and VO keeps no frame (find(nullptr) never succeeds)
struct DensePriorProbe : public AngularAdjustmentCERESAnalytic {
    using AngularAdjustmentCERESAnalytic::addMarginalizationResiduals;
};

TEST_F(MarginalizationTest, denseVOPriorIsAddedToTheWindow) {
    DensePriorProbe optim;
    ASSERT_TRUE(optim.marginalize(_frame0, _frame1, false));
    ceres::Problem problem;
    ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;
    optim.addMarginalizationResiduals(problem, nullptr, ordering);
    EXPECT_EQ(problem.NumResidualBlocks(), 1) << "the dense prior factor";
    delete ordering;
}

TEST_F(MarginalizationTest, sparsePriorForPointOnlyMap) {
    SparsePriorProbe ba;
    ASSERT_TRUE(ba.marginalize(_frame0, _frame1, true));
    ceres::Problem problem;
    ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;
    ba.addSparsePriorResiduals(problem, nullptr, ordering);
    EXPECT_GT(problem.NumResidualBlocks(), 0);
    delete ordering;
}

} // namespace isae