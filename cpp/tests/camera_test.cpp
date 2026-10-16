#include <gtest/gtest.h>

#include <cmath>

#include <opencv2/calib3d.hpp>

#include "isaeslam/data/sensors/KannalaBrandt.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "utilities/geometry.h"

namespace isae {

namespace {

// Insta360 X5 front lens (GUTWrap COLMAP rig BA, VID_012) scaled to 1920 px: fx fy cx cy, k1..k4
std::shared_ptr<KannalaBrandt> x5Lens() {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 1034.563 / 2, K(1, 1) = 1035.349 / 2, K(0, 2) = 1937.498 / 2, K(1, 2) = 1916.321 / 2;
    const Eigen::Vector4d k(0.034111, -0.001103, 0.004174, -0.001841);
    return std::make_shared<KannalaBrandt>(cv::Mat(1920, 1920, CV_8UC1, cv::Scalar(0)), K, k, 100.0);
}

// A unit direction at angle theta from the optical axis, azimuth phi
Eigen::Vector3d dir(double theta_deg, double phi_deg) {
    const double t = theta_deg * M_PI / 180, p = phi_deg * M_PI / 180;
    return Eigen::Vector3d(std::sin(t) * std::cos(p), std::sin(t) * std::sin(p), std::cos(t));
}

} // namespace

// Pixel -> bearing -> pixel, up to the X5's image circle (100 deg from the axis: points behind the image plane)
TEST(KannalaBrandtTest, roundTripBeyondNinetyDegrees) {
    auto cam = x5Lens();
    for (double th : {0.0, 1.0, 30.0, 60.0, 85.0, 89.9, 90.0, 90.1, 95.0, 99.5})
        for (double ph : {0.0, 37.0, 135.0, 250.0}) {
            const Eigen::Vector3d d = dir(th, ph);
            Eigen::Vector2d px;
            ASSERT_TRUE(cam->projectCamera(3.0 * d, px)) << th << " deg";
            const Eigen::Vector3d b = cam->getRayCamera(px);
            EXPECT_NEAR(b.norm(), 1.0, 1e-12);
            const double angle = std::atan2(b.cross(d).norm(), b.dot(d)); // stable for tiny angles (acos is not)
            EXPECT_LT(angle, 1e-9) << th << " deg, " << ph << " deg";
            if (th > 90)
                EXPECT_LT(b.z(), 0) << "beyond 90 deg the bearing points behind the image plane";
        }
}

// Beyond the image circle (max_theta) or straight behind, a point does not project
TEST(KannalaBrandtTest, invalidBeyondTheImageCircle) {
    auto cam = x5Lens();
    Eigen::Vector2d px;
    EXPECT_TRUE(cam->projectCamera(dir(99.0, 10.0), px));
    EXPECT_FALSE(cam->projectCamera(dir(101.0, 10.0), px));
    EXPECT_FALSE(cam->projectCamera(Eigen::Vector3d(0, 0, -1), px));
    EXPECT_FALSE(cam->projectCamera(Eigen::Vector3d::Zero(), px));
}

// Same pixels as OpenCV's fisheye model (the convention of the calibration) within its range (z > 0)
TEST(KannalaBrandtTest, matchesOpenCvFisheye) {
    auto cam = x5Lens();
    const Eigen::Matrix3d K = cam->getCalibration();
    cv::Matx33d Kcv(K(0, 0), 0, K(0, 2), 0, K(1, 1), K(1, 2), 0, 0, 1);
    cv::Vec4d D(0.034111, -0.001103, 0.004174, -0.001841);
    std::vector<cv::Point3d> obj;
    std::vector<Eigen::Vector2d> ours;
    for (double th : {5.0, 40.0, 70.0, 85.0})
        for (double ph : {0.0, 100.0, 220.0}) {
            const Eigen::Vector3d p = 2.0 * dir(th, ph);
            obj.emplace_back(p.x(), p.y(), p.z());
            Eigen::Vector2d px;
            ASSERT_TRUE(cam->projectCamera(p, px));
            ours.push_back(px);
        }
    std::vector<cv::Point2d> ref;
    cv::fisheye::projectPoints(obj, ref, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), Kcv, D);
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(ours[i].x(), ref[i].x, 1e-6);
        EXPECT_NEAR(ours[i].y(), ref[i].y, 1e-6);
    }
}

// Analytic Jacobian d pixel / d point against central differences, near the axis, at 60 deg and beyond 90 deg
TEST(KannalaBrandtTest, jacobianMatchesNumeric) {
    auto cam = x5Lens();
    for (const Eigen::Vector3d &p : {Eigen::Vector3d(1e-6, -2e-6, 2.0), Eigen::Vector3d(dir(60, 30) * 1.5),
                                     Eigen::Vector3d(dir(95, 200) * 4.0)}) {
        Eigen::Vector2d px;
        Eigen::Matrix<double, 2, 3> J;
        ASSERT_TRUE(cam->projectCamera(p, px, &J));
        for (int c = 0; c < 3; c++) {
            const double h   = 1e-6;
            Eigen::Vector3d e = Eigen::Vector3d::Zero();
            e(c)              = h;
            Eigen::Vector2d pp, pm;
            cam->projectCamera(p + e, pp);
            cam->projectCamera(p - e, pm);
            const Eigen::Vector2d num = (pp - pm) / (2 * h);
            EXPECT_LT((J.col(c) - num).norm(), 1e-3 * std::max(1.0, num.norm())) << "column " << c;
        }
    }
}

// The planar estimators' ray limit is off by default (every ray, unchanged behaviour for every camera model) and,
// once set (KB4 cameras: 80 deg), leaves out the rays beyond it
TEST(KannalaBrandtTest, planarRayLimitIsOptIn) {
    const double saved = geometry::planarRayMinCos();
    geometry::planarRayMinCos() = -2.0;
    EXPECT_TRUE(geometry::planarRayUsable(dir(85, 0)));
    EXPECT_TRUE(geometry::planarRayUsable(dir(120, 0))) << "off: even rays behind the image plane pass";
    geometry::planarRayMinCos() = std::cos(80 * M_PI / 180);
    EXPECT_TRUE(geometry::planarRayUsable(dir(79, 30)));
    EXPECT_FALSE(geometry::planarRayUsable(dir(81, 30)));
    EXPECT_FALSE(geometry::planarRayUsable(dir(95, 30)));
    geometry::planarRayMinCos() = saved;
}

// The image-circle mask keeps the pixels within max_theta of the axis: the centre and a pixel at 95 deg, not the
// corners of the 1920 x 1920 image (beyond the 100 deg circle)
TEST(KannalaBrandtTest, imageCircleMask) {
    auto cam        = x5Lens();
    const cv::Mat m = cam->imageCircleMask();
    ASSERT_EQ(m.rows, 1920);
    Eigen::Vector2d px95;
    ASSERT_TRUE(cam->projectCamera(dir(95, 45), px95));
    EXPECT_EQ(m.at<uchar>(960, 960), 255);
    EXPECT_EQ(m.at<uchar>(static_cast<int>(px95.y()), static_cast<int>(px95.x())), 255);
    EXPECT_EQ(m.at<uchar>(5, 5), 0);
    EXPECT_EQ(m.at<uchar>(1914, 1914), 0);
}

} // namespace isae
