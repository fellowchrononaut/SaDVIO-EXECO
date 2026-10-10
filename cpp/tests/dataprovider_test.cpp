#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <opencv2/imgcodecs.hpp>
#include <unistd.h>

#include "isaeslam/data/frame.h"
#include "isaeslam/dataproviders/adataprovider.h"

namespace isae {

/*!
 * @brief A tiny synthetic EuRoC folder: stereo images, IMU at 200 Hz with a 2 ms clock offset.
 *
 * IMU sample k (raw stamp t0 + 5k ms) has acc = k (1, 2, 3) and gyr = 0.1 k (1, 1, 1), so a linear
 * interpolation at any time is exact. Camera stamps (camera clock): 10, 37 and 60 ms after t0 for
 * cam0; cam1 is 1 ms later and has an extra unmatched image at 50 ms.
 */
class DataProviderTest : public testing::Test {
  public:
    void SetUp() override {
        _dir = std::filesystem::temp_directory_path() / ("isae_grabber_test_" + std::to_string(::getpid()));
        std::filesystem::remove_all(_dir);
        for (const char *sub : {"mav0/cam0/data", "mav0/cam1/data", "mav0/imu0"})
            std::filesystem::create_directories(_dir / sub);

        cv::Mat img(16, 16, CV_8UC1, cv::Scalar(128));
        auto write_cam = [&](const std::string &cam, const std::vector<long long> &stamps) {
            std::ofstream csv(_dir / "mav0" / cam / "data.csv");
            csv << "#timestamp [ns],filename\n";
            for (long long t : stamps) {
                csv << t << "," << t << ".png\n";
                cv::imwrite((_dir / "mav0" / cam / "data" / (std::to_string(t) + ".png")).string(), img);
            }
        };
        _cam_stamps = {_t0 + 10 * _ms, _t0 + 37 * _ms, _t0 + 60 * _ms};
        write_cam("cam0", _cam_stamps);
        write_cam("cam1", {_t0 + 11 * _ms, _t0 + 38 * _ms, _t0 + 50 * _ms, _t0 + 61 * _ms});

        std::ofstream imu(_dir / "mav0/imu0/data.csv");
        imu << "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n";
        for (int k = 0; k <= 20; k++)
            imu << _t0 + 5 * k * _ms << "," << 0.1 * k << "," << 0.1 * k << "," << 0.1 * k << "," << k << ","
                << 2 * k << "," << 3 * k << "\n";
    }

    void TearDown() override { std::filesystem::remove_all(_dir); }

    std::shared_ptr<ADataProvider> makeProvider(const std::string &slam_mode) {
        std::string yaml = (_dir / "dataset.yaml").string();
        std::ofstream f(yaml);
        for (int i = 0; i < 2; i++)
            f << "camera_" << i << ":\n"
              << "  topic: /cam" << i << "\n"
              << "  T_BS:\n    data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n"
              << "  resolution: [16, 16]\n  projection_model: pinhole\n  intrinsics: [10.0, 10.0, 8.0, 8.0]\n"
              << "  distortion_model: none\n  distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n";
        f << "ncam: 2\nstereo_sync_tolerance_ms: 5\n"
          << "imu:\n  topic: /imu0\n"
          << "  T_BS:\n    data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n"
          << "  rate_hz: 200\n  gyroscope_noise_density: 0.001\n  gyroscope_random_walk: 0.0001\n"
          << "  accelerometer_noise_density: 0.01\n  accelerometer_random_walk: 0.001\n  dt_imu_cam: 0.002\n";
        f.close();

        Config cfg;
        cfg.slam_mode         = slam_mode;
        cfg.contrast_enhancer = 0;
        cfg.downsampling      = 1;
        return std::make_shared<ADataProvider>(yaml, cfg);
    }

    std::vector<std::shared_ptr<Frame>> readAll(const std::shared_ptr<ADataProvider> &prov) {
        EUROCGrabber grabber((_dir / "mav0").string(), prov);
        grabber.load_filenames();
        grabber.addAllFrames();
        std::vector<std::shared_ptr<Frame>> frames;
        while (prov->queueSize() > 0)
            frames.push_back(prov->next());
        return frames;
    }

    std::filesystem::path _dir;
    const long long _t0 = 1000000000000LL, _ms = 1000000LL;
    std::vector<long long> _cam_stamps;
};

TEST_F(DataProviderTest, imuAndImagesMergedInTimeOrder) {

    std::vector<std::shared_ptr<Frame>> frames = readAll(makeProvider("bimonovio"));

    // IMU stamps in the camera clock: t0 - 2 ms + 5k ms. Expected: 3 IMU frames before 10 ms, the
    // image, 5 IMU frames, the image at 37 ms, 5 IMU frames, the image at 60 ms; then the end.
    ASSERT_EQ(frames.size(), 16u);
    int n_imu_only = 0;
    std::vector<std::shared_ptr<Frame>> image_frames;
    for (size_t i = 0; i < frames.size(); i++) {
        if (i > 0)
            EXPECT_LT(frames[i - 1]->getTimestamp(), frames[i]->getTimestamp()) << "frame " << i;
        if (frames[i]->getSensors().empty()) {
            ASSERT_TRUE(frames[i]->getIMU());
            n_imu_only++;
        } else
            image_frames.push_back(frames[i]);
    }

    // No IMU sample lost (13 raw samples fall before the last image), unmatched cam1 image dropped
    EXPECT_EQ(n_imu_only, 13);
    ASSERT_EQ(image_frames.size(), 3u);

    for (size_t i = 0; i < 3; i++) {
        std::shared_ptr<Frame> f = image_frames[i];
        EXPECT_EQ((long long)f->getTimestamp(), _cam_stamps[i]) << "image frames keep the image stamp";
        EXPECT_EQ(f->getSensors().size(), 2u);
        ASSERT_TRUE(f->getIMU()) << "image frame " << i << " carries an IMU measurement";

        // Interpolated at the image time: k = (t_cam + 2 ms - t0) / 5 ms
        double k = double(_cam_stamps[i] + 2 * _ms - _t0) / double(5 * _ms);
        EXPECT_NEAR((f->getIMU()->getAcc() - k * Eigen::Vector3d(1, 2, 3)).norm(), 0, 1e-9);
        EXPECT_NEAR((f->getIMU()->getGyr() - 0.1 * k * Eigen::Vector3d(1, 1, 1)).norm(), 0, 1e-9);
    }
}

TEST_F(DataProviderTest, voModeNeedsNoImuFile) {

    std::filesystem::remove_all(_dir / "mav0/imu0");
    std::vector<std::shared_ptr<Frame>> frames = readAll(makeProvider("bimono"));
    ASSERT_EQ(frames.size(), 3u);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ((long long)frames[i]->getTimestamp(), _cam_stamps[i]);
        EXPECT_FALSE(frames[i]->getIMU());
    }
}

TEST_F(DataProviderTest, mergerRejectsOutOfOrderData) {

    // Live streams (ROS) can deliver late data: the merger must keep the frame queue in time order.
    std::shared_ptr<ADataProvider> prov = makeProvider("bimonovio");
    ImuImageMerger merger(prov);
    const Eigen::Vector3d a(0, 0, 9.81), w(0, 0, 0.1);
    EXPECT_TRUE(merger.addImu(100, a, w));
    EXPECT_TRUE(merger.addImu(200, a, w));
    EXPECT_FALSE(merger.addImu(150, a, w)); // late IMU message dropped
    EXPECT_FALSE(merger.addImu(200, a, w)); // duplicate dropped
    EXPECT_EQ(merger.droppedImu(), 2);
    EXPECT_TRUE(merger.imuReached(200));
    EXPECT_FALSE(merger.imuReached(201));

    EXPECT_TRUE(merger.emitImageFrame(150, {})); // IMU at 100, then image at 150 with IMU interpolated
    EXPECT_FALSE(merger.emitImageFrame(120, {})); // image older than the emitted IMU: rejected
    ASSERT_EQ(prov->queueSize(), 2u);
    std::shared_ptr<Frame> f0 = prov->next(), f1 = prov->next();
    EXPECT_EQ(f0->getTimestamp(), 100u);
    EXPECT_EQ(f1->getTimestamp(), 150u);
    ASSERT_TRUE(f1->getIMU());
}

TEST_F(DataProviderTest, rawImuGapIsNotHiddenByImageInterpolation) {

    // A 2 s outage of the raw IMU stream while images keep coming at 20 Hz: the measurements interpolated at the
    // image times are one image period apart, so the integrator saw no gap and the factors trusted invented
    // motion. The merger marks them; every KF interval touching the outage must report a gap, the others none.
    std::shared_ptr<ADataProvider> prov = makeProvider("bimonovio");
    ImuImageMerger merger(prov);
    const long long T0 = 1'000'000'000LL;
    const Eigen::Vector3d a(0.1, 0.2, 9.81), w(0.01, -0.02, 0.03);
    for (long long t = 0; t <= 100; t += 5) // raw IMU at 200 Hz: 0-100 ms, then 2100-2300 ms
        merger.addImu(T0 + t * _ms, a, w);
    for (long long t = 2100; t <= 2300; t += 5)
        merger.addImu(T0 + t * _ms, a, w);
    for (long long t = 50; t <= 2250; t += 50)
        ASSERT_TRUE(merger.emitImageFrame(T0 + t * _ms, {}));

    // Chain the measurements as the pipelines do, with KFs at 50 ms, 1000 ms (in the outage), 2150 ms and 2200 ms
    const std::set<long long> kf_ms = {50, 1000, 2150, 2200};
    std::map<long long, int> gaps_at_kf; // KF -> gap steps of the interval ending there
    std::shared_ptr<Frame> kf;
    std::shared_ptr<IMU> last;
    while (prov->queueSize() > 0) {
        std::shared_ptr<Frame> f = prov->next();
        const long long t_ms     = ((long long)f->getTimestamp() - T0) / _ms;
        std::shared_ptr<IMU> imu = f->getIMU();
        ASSERT_TRUE(imu);
        if (kf) {
            imu->setLastIMU(last);
            imu->setLastKF(kf);
            ASSERT_TRUE(imu->processIMU());
        }
        last = imu;
        if ((long long)f->getTimestamp() == T0 + t_ms * _ms && kf_ms.count(t_ms)) { // KF at these image times
            if (kf)
                gaps_at_kf[t_ms] = imu->getGapSteps();
            f->setKeyFrame();
            kf = f;
        }
    }
    ASSERT_EQ(gaps_at_kf.size(), 3u);
    EXPECT_GT(gaps_at_kf[1000], 0) << "interval 50-1000 ms covers the start of the outage";
    EXPECT_GT(gaps_at_kf[2150], 0) << "interval 1000-2150 ms covers its end";
    EXPECT_EQ(gaps_at_kf[2200], 0) << "interval 2150-2200 ms has only raw data";
}

TEST(ConfigTest, unsupportedOptionCombinationsAreReported) {

    // Issue 20: option combinations that cannot work are refused instead of silently doing nothing.
    auto check = [](Config cfg, int ncam, bool imu, size_t n_err, size_t n_warn) {
        std::vector<std::string> warnings;
        std::vector<std::string> errors = validateConfig(cfg, ncam, imu, warnings);
        std::string all;
        for (auto &e : errors)
            all += "E: " + e + "\n";
        for (auto &w : warnings)
            all += "W: " + w + "\n";
        EXPECT_EQ(errors.size(), n_err) << all;
        EXPECT_EQ(warnings.size(), n_warn) << all;
    };

    Config cfg;
    cfg.slam_mode       = "bimonovio";
    cfg.optimizer       = "AngularAnalytic";
    cfg.marginalization = 1;
    cfg.sparsification  = true;
    cfg.estimate_td     = true;
    check(cfg, 2, true, 0, 0); // valid stereo VIO

    check(cfg, 2, false, 1, 0); // VIO without imu block
    check(cfg, 1, true, 1, 0);  // stereo mode with one camera

    Config c = cfg;
    c.slam_mode = "stereo";
    check(c, 2, true, 1, 1); // unknown mode (estimate_td then counts as non-VIO: warning)

    c           = cfg;
    c.optimizer = "Numeric";
    check(c, 2, true, 2, 0); // Numeric: no marginalization, no estimate_td

    c                 = cfg;
    c.marginalization = 0;
    check(c, 2, true, 0, 1); // sparsification without marginalization

    c             = cfg;
    c.slam_mode   = "bimono";
    c.estimate_td = true;
    check(c, 2, false, 0, 1); // estimate_td in VO

    c           = cfg;
    c.slam_mode = "monovio";
    check(c, 1, true, 0, 0); // mono VIO marginalizes too (Issue 16)

    c                 = cfg;
    c.max_lost_frames = -1;
    check(c, 2, true, 1, 0); // re-initialization threshold: >= 1, or 0 for the mode default
    c.max_lost_frames = 10;
    check(c, 2, true, 0, 0);

    c                    = cfg;
    c.reinit_carry_state = 2;
    check(c, 2, true, 1, 0); // re-initialization from the last state: 0 or 1
    c.reinit_carry_state = 0;
    check(c, 2, true, 0, 0);
    c                          = cfg;
    c.reinit_carry_max_age_vio = 0;
    check(c, 2, true, 1, 0); // carry limits: > 0 s
    c.reinit_carry_max_age_vio = 5;
    c.reinit_carry_max_age_vo  = -1;
    check(c, 2, true, 1, 0);

    // Loop closure (doc/loop_closure): proximity needs no vocabulary file
    c                   = cfg;
    c.loop_closure      = 1;
    c.loop_detector     = "proximity";
    check(c, 2, true, 0, 0);
    c.loop_correct_window = -1; // automatic
    check(c, 2, true, 0, 0);
    c.loop_correct_window = 2;
    check(c, 2, true, 1, 0);
    c.loop_correct_window = 0;
    c.loop_graph_dof      = 5;
    check(c, 2, true, 1, 0);
    c.loop_graph_dof = 4;
    check(c, 2, true, 0, 0);
    c.slam_mode = "bimono";
    c.estimate_td = false;
    check(c, 2, false, 1, 0); // 4-DoF needs an IMU
    c                 = cfg;
    c.loop_closure    = 1;
    c.loop_detector   = "bow";
    c.loop_vocabulary = "/nonexistent/ORBvoc.dbow3";
    check(c, 2, true, 1, 0); // unreadable vocabulary
    c.slam_mode   = "mono";
    c.estimate_td = false;
    check(c, 1, false, 0, 1); // mono VO: skipped with a warning (no Sim3 graph), nothing else checked
}

TEST_F(DataProviderTest, onlineTimeOffsetAppliesToLaterImu) {

    // Issue 19: an online estimate of dt_imu_cam must reach the IMU read afterwards (the offline reader used
    // to shift every IMU stamp once, at load time)
    std::shared_ptr<ADataProvider> prov = makeProvider("bimonovio");
    EUROCGrabber grabber((_dir / "mav0").string(), prov);
    grabber.load_filenames();
    ASSERT_TRUE(grabber.addNextFrame()); // up to the first image (10 ms), dt_imu_cam = 2 ms
    prov->getIMUConfig()->dt_imu_cam = 0.004;
    grabber.addAllFrames();

    std::vector<long long> imu_only;
    while (prov->queueSize() > 0) {
        std::shared_ptr<Frame> f = prov->next();
        if (f->getSensors().empty())
            imu_only.push_back((long long)f->getTimestamp() - _t0);
    }
    // raw stamps 5k ms: k = 0..3 were read with the old 2 ms offset (k = 3, raw 15 ms, was needed for the
    // interpolation at the 10 ms image), the later ones with the new 4 ms offset
    ASSERT_GE(imu_only.size(), 6u);
    EXPECT_EQ(imu_only[0], -2 * _ms);
    EXPECT_EQ(imu_only[3], 13 * _ms);
    EXPECT_EQ(imu_only[4], 16 * _ms); // raw 20 ms, now shifted by 4 ms
}

} // namespace isae
