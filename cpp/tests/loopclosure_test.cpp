#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "isaeslam/loopclosure/LoopClosure.h"
#include "isaeslam/loopclosure/PoseGraph.h"

namespace isae {

namespace {

// Ground truth: two laps of a circle (radius 5 m) with a gentle up and down, frames tilted a little (roll, pitch)
std::vector<Eigen::Affine3d> circle(int n) {
    std::vector<Eigen::Affine3d> T(n);
    for (int k = 0; k < n; k++) {
        const double a = 4 * M_PI * k / n;
        T[k]           = Eigen::Affine3d::Identity();
        T[k].translation() << 5 * std::cos(a), 5 * std::sin(a), 0.5 * std::sin(2 * a);
        T[k].linear() = (Eigen::AngleAxisd(a + M_PI / 2, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()))
                            .toRotationMatrix();
    }
    return T;
}

// Odometry: the true relative poses with a small constant yaw and translation bias per step (drift), roll and
// pitch kept exact when four_dof (gravity makes them observable)
std::vector<Eigen::Affine3d> drift(const std::vector<Eigen::Affine3d> &gt, bool four_dof) {
    std::vector<Eigen::Affine3d> od(gt.size());
    od[0] = gt[0];
    for (size_t k = 1; k < gt.size(); k++) {
        Eigen::Affine3d d  = gt[k - 1].inverse() * gt[k];
        d.translation()   *= 1.01;
        d.translation()   += Eigen::Vector3d(0.002, -0.001, 0.0005);
        Eigen::Affine3d T  = od[k - 1] * d;
        T.linear()         = Eigen::AngleAxisd(0.002, Eigen::Vector3d::UnitZ()).toRotationMatrix() * T.linear();
        if (!four_dof)
            T.linear() = Eigen::AngleAxisd(0.001, Eigen::Vector3d::UnitX()).toRotationMatrix() * T.linear();
        od[k] = T;
    }
    return od;
}

double posError(const PoseGraph &g, const std::vector<Eigen::Affine3d> &gt, int i) {
    return (g.pose(i).translation() - gt[i].translation()).norm();
}

} // namespace

// One loop between the end and the start removes most of the accumulated drift (4-DoF and 6-DoF graphs)
TEST(PoseGraphTest, loopCorrectsTheDrift) {
    for (bool four_dof : {true, false}) {
        const int n                            = 200;
        const std::vector<Eigen::Affine3d> gt = circle(n);
        const std::vector<Eigen::Affine3d> od = drift(gt, four_dof);
        PoseGraph g(four_dof);
        for (int k = 0; k < n; k++)
            g.addNode(od[k], 0);
        const double before = (od[n - 1].translation() - gt[n - 1].translation()).norm();
        // Loops every 10 KFs of the second lap to the first lap (same places)
        for (int k = n / 2; k < n; k += 10)
            g.addLoop(k - n / 2, k, gt[k - n / 2].inverse() * gt[k]);
        EXPECT_EQ(g.optimize(), 0) << "no loop should be rejected";
        const double after = posError(g, gt, n - 1);
        EXPECT_GT(before, 1.0) << "the test needs real drift";
        EXPECT_LT(after, 0.2 * before) << (four_dof ? "4-DoF" : "6-DoF") << ": " << before << " -> " << after;
        // The correction maps the latest odometry pose onto the corrected one
        EXPECT_LT(((g.correction(0) * od[n - 1]).translation() - g.pose(n - 1).translation()).norm(), 1e-9);
    }
}

// A wrong loop (between two places 10 m apart) is disabled by the pose graph and does not distort the trajectory
TEST(PoseGraphTest, wrongLoopIsRejected) {
    const int n                            = 200;
    const std::vector<Eigen::Affine3d> gt = circle(n);
    const std::vector<Eigen::Affine3d> od = drift(gt, true);
    PoseGraph g(true);
    for (int k = 0; k < n; k++)
        g.addNode(od[k], 0);
    for (int k = n / 2; k < n; k += 10)
        g.addLoop(k - n / 2, k, gt[k - n / 2].inverse() * gt[k]);
    g.addLoop(10, 150, Eigen::Affine3d::Identity()); // claims KF 150 is at KF 10: the opposite side of the circle
    EXPECT_GE(g.optimize(), 1);
    EXPECT_EQ(g.nActiveLoops(), g.nLoops() - 1);
    EXPECT_LT(posError(g, gt, n - 1), 0.5);
}

// Two segments (a re-initialization with a world frame of its own) are joined by a loop between them
TEST(PoseGraphTest, loopJoinsTwoSegments) {
    const int n                            = 200;
    const std::vector<Eigen::Affine3d> gt = circle(n);
    std::vector<Eigen::Affine3d> od       = drift(gt, true);
    // Second lap restarted in a frame moved by 3 m and rotated by 40 deg about gravity
    Eigen::Affine3d T_w2_w   = Eigen::Affine3d::Identity();
    T_w2_w.linear()          = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_w2_w.translation()     = Eigen::Vector3d(3, -1, 0.2);
    PoseGraph g(true);
    for (int k = 0; k < n; k++)
        g.addNode(k < n / 2 ? od[k] : T_w2_w * od[k], k < n / 2 ? 0 : 1);
    for (int k = n / 2; k < n; k += 10)
        g.addLoop(k - n / 2, k, gt[k - n / 2].inverse() * gt[k]);
    // Before: the two laps are 3 m and 40 deg apart at the same places
    EXPECT_GT((g.pose(n - 1).translation() - g.pose(n / 2 - 1).translation()).norm(), 1.0);
    EXPECT_EQ(g.optimize(), 0);
    // After: the second segment is brought into the first one's frame, the two laps agree at the same places.
    // Nothing corrects the first lap's own drift (no loop within it), so the comparison is between the laps
    for (int k = n / 2; k < n; k += 10) {
        EXPECT_LT((g.pose(k).translation() - g.pose(k - n / 2).translation()).norm(), 0.3) << "KF " << k;
        EXPECT_LT(Eigen::AngleAxisd(g.pose(k).rotation().transpose() * g.pose(k - n / 2).rotation()).angle(), 0.05);
    }
}

// Verification of a loop candidate (LoopClosure::verify): each KF's landmarks are matched to the other KF's
// features, PnP RANSAC gives the relative pose both ways, and the two must agree
struct LoopClosureTestAccess {
    using Record = LoopClosure::Record;
    static bool verify(LoopClosure &lc, const Record &q, const Record &c, Eigen::Affine3d &T_fc_fq) {
        int a = 0, b = 0;
        return lc.verify(q, c, T_fc_fq, a, b);
    }
};

namespace {

using Record = LoopClosureTestAccess::Record;

// A scene of random points with random 256-bit descriptors; a view of it is a camera pose, whose descriptors differ
// from the scene's by a few bits (viewpoint)
struct Scene {
    std::vector<Eigen::Vector3d> pts;
    std::vector<cv::Mat> desc;
    explicit Scene(unsigned seed, int n = 200) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ux(-3, 3), uy(-2, 2), uz(4, 8);
        std::uniform_int_distribution<int> byte(0, 255);
        for (int i = 0; i < n; i++) {
            pts.emplace_back(ux(rng), uy(rng), uz(rng));
            cv::Mat d(1, 32, CV_8U);
            for (int k = 0; k < 32; k++)
                d.at<uchar>(0, k) = static_cast<uchar>(byte(rng));
            desc.push_back(d);
        }
    }
};

cv::Mat flipBits(const cv::Mat &d, std::mt19937 &rng, int n_bits) {
    cv::Mat o = d.clone();
    std::uniform_int_distribution<int> bit(0, 255);
    for (int k = 0; k < n_bits; k++) {
        const int b = bit(rng);
        o.at<uchar>(0, b / 8) ^= static_cast<uchar>(1 << (b % 8));
    }
    return o;
}

// A record of the scene seen from camera pose T_w_c (frame = camera): landmarks [l0, l1) in the camera, and every
// point in front of the camera as a feature (normalized coordinates, 0.5 px noise at focal 400). lmk_offset moves the
// landmarks (not the features): a candidate whose structure disagrees with its image
Record view(const Scene &s, const Eigen::Affine3d &T_w_c, int l0, int l1, unsigned seed,
            const Eigen::Vector3d &lmk_offset = Eigen::Vector3d::Zero()) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> px(0, 0.5 / 400);
    Record r;
    r.ts = 0, r.segment = 0, r.node = 0;
    r.T_s_f = Eigen::Affine3d::Identity();
    r.focal = 400;
    const Eigen::Affine3d T_c_w = T_w_c.inverse();
    for (size_t i = 0; i < s.pts.size(); i++) {
        const Eigen::Vector3d p = T_c_w * s.pts[i];
        if (p.z() < 0.5)
            continue;
        r.kp_n.emplace_back(static_cast<float>(p.x() / p.z() + px(rng)), static_cast<float>(p.y() / p.z() + px(rng)));
        r.desc.push_back(flipBits(s.desc[i], rng, 6));
        if (static_cast<int>(i) >= l0 && static_cast<int>(i) < l1) {
            r.lmk_c.push_back(p + lmk_offset);
            r.lmk_desc.push_back(flipBits(s.desc[i], rng, 6));
        }
    }
    return r;
}

LoopClosure::Options verifyOptions() {
    LoopClosure::Options o;
    o.detector    = "proximity"; // no vocabulary needed to test the verification
    o.min_inliers = 12;
    return o;
}

Eigen::Affine3d candidatePose() {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear()        = Eigen::AngleAxisd(0.17, Eigen::Vector3d::UnitY()).toRotationMatrix();
    T.translation()   = Eigen::Vector3d(0.4, 0.1, -0.2);
    return T;
}

} // namespace

// The same place seen again from 0.5 m and 10 deg away: accepted, with the relative pose of the two cameras
TEST(LoopVerificationTest, revisitIsAcceptedWithItsRelativePose) {
    LoopClosure lc(verifyOptions());
    const Scene s(1);
    const Eigen::Affine3d T_w_q = Eigen::Affine3d::Identity(), T_w_c = candidatePose();
    const Record q = view(s, T_w_q, 0, 120, 2), c = view(s, T_w_c, 60, 180, 3);
    Eigen::Affine3d T_fc_fq;
    ASSERT_TRUE(LoopClosureTestAccess::verify(lc, q, c, T_fc_fq));
    const Eigen::Affine3d T_true = T_w_c.inverse() * T_w_q;
    EXPECT_LT((T_fc_fq.translation() - T_true.translation()).norm(), 0.05);
    EXPECT_LT(Eigen::AngleAxisd(T_fc_fq.rotation().transpose() * T_true.rotation()).angle() * 180 / M_PI, 0.5);
}

// Two different places (other points, other descriptors): rejected
TEST(LoopVerificationTest, differentPlaceIsRejected) {
    LoopClosure lc(verifyOptions());
    const Scene s1(1), s2(7);
    const Record q = view(s1, Eigen::Affine3d::Identity(), 0, 120, 2), c = view(s2, candidatePose(), 60, 180, 3);
    Eigen::Affine3d T_fc_fq;
    EXPECT_FALSE(LoopClosureTestAccess::verify(lc, q, c, T_fc_fq));
}

// The candidate's landmarks are 0.6 m off its image (a bad triangulation, a wrong map): each direction finds a pose,
// but they disagree, so the loop is rejected
TEST(LoopVerificationTest, directionsThatDisagreeAreRejected) {
    LoopClosure lc(verifyOptions());
    const Scene s(1);
    const Record q = view(s, Eigen::Affine3d::Identity(), 0, 120, 2);
    const Record c = view(s, candidatePose(), 60, 180, 3, Eigen::Vector3d(0.6, 0, 0));
    Eigen::Affine3d T_fc_fq;
    EXPECT_FALSE(LoopClosureTestAccess::verify(lc, q, c, T_fc_fq));
}

} // namespace isae

