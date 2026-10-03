#include <gtest/gtest.h>

#include "utilities/geometry.h"
#include "utilities/timer.h"

#include <atomic>
#include <thread>

// Demonstrate some basic assertions.
TEST(BasicTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

// Left Jacobian of SO(3): Exp(w + d) ~ Exp(Jl(w) d) Exp(w). Must be finite at w = 0 (Issue 10).
TEST(GeometryTest, so3LeftJacobian) {
  using namespace isae;

  Eigen::Matrix3d J0 = geometry::so3_leftJacobian(Eigen::Vector3d::Zero());
  ASSERT_TRUE(J0.allFinite());
  EXPECT_NEAR((J0 - Eigen::Matrix3d::Identity()).norm(), 0, 1e-12);

  const std::vector<Eigen::Vector3d> ws = {Eigen::Vector3d(1e-9, 0, 0), Eigen::Vector3d(1e-6, -2e-6, 1e-6),
                                           Eigen::Vector3d(0.2, -0.3, 0.1), Eigen::Vector3d(1.5, 0.4, -0.9)};
  const double h = 1e-7;
  for (const auto &w : ws) {
    Eigen::Matrix3d Jl = geometry::so3_leftJacobian(w);
    ASSERT_TRUE(Jl.allFinite()) << w.transpose();

    // Numerical left Jacobian, column by column
    Eigen::Matrix3d J_num;
    for (int k = 0; k < 3; k++) {
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      d(k)              = h;
      J_num.col(k) = geometry::log_so3(geometry::exp_so3(w + d) * geometry::exp_so3(w).transpose()) / h;
    }
    EXPECT_NEAR((Jl - J_num).cwiseAbs().maxCoeff(), 0, 1e-6) << "w = " << w.transpose();

    // Jl(w) = Exp(w) Jr(w) (so3_rightJacobian returns I below 1e-5, hence the tolerance)
    EXPECT_NEAR((Jl - geometry::exp_so3(w) * geometry::so3_rightJacobian(w)).cwiseAbs().maxCoeff(), 0, 1e-5)
        << "w = " << w.transpose();
  }
}

// The timer stack is used from several threads (reader, front end, back end...): each thread must get
// its own, or concurrent tic/toc corrupt it (the cause of rare segfaults in the offline runs).
TEST(TimerTest, ticTocFromSeveralThreads) {
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; t++)
    threads.emplace_back([&bad]() {
      for (int i = 0; i < 200000; i++) {
        isae::timer::tic();
        isae::timer::tic();
        if (isae::timer::silentToc() < 0 || isae::timer::silentToc() < 0)
          bad++;
      }
    });
  for (auto &th : threads)
    th.join();
  EXPECT_EQ(bad.load(), 0);
}
