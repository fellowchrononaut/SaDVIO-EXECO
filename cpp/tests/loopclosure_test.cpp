#include <gtest/gtest.h>

#include <cmath>
#include <random>

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

} // namespace isae
