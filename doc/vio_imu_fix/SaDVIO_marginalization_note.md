# SaDVIO: two flaws in the marginalization prior (sparse mode)

To: the SaDVIO-EXECO maintainers · From: the SaDLIO port (SaDLIO-EXECO, milestone M2) · Date: 2026-10-06
Code: SaDVIO-EXECO `dense_devel` at `e035ccc`, `cpp/src/optimizers/marginalization.cpp` (`sparsifyVIO`,
`sqrtInformation`, `rankReveallingDecomposition`).

## Summary

- **Sparse mode (`sparsification: 1`) is affected.** When the dense prior has unobservable directions (in any VIO
  prior: position and yaw), the absolute inertial factor built by `sparsifyVIO` loses almost all its information:
  rotation, velocity and biases of the kept keyframe. On a synthetic prior with 4 kept landmarks, its largest
  information is 2.9 to 23 instead of 1e7. The sparse prior is then mostly the pose-to-landmark factors.
- **Dense mode (`sparsification: 0`, the default config) is not affected by the first flaw.** The second flaw (rank
  threshold) also touches it, probably mildly; this was not measured on a SaDVIO run.
- The existing test `inertialPriorWithoutKeptLandmarks` checks that the sparse factor does not have *too much*
  information; it does not catch *too little*.
- Both were found in SaDLIO, whose marginalization code is a copy of SaDVIO's. They are fixed there with tests. The
  fixes below are written against SaDVIO's code.

## Flaw 1: `sparsifyVIO` drops the precise directions of the absolute factor

The information of the absolute factor on the kept frame is computed through a covariance:

```cpp
// sparsifyVIO, e035ccc
Eigen::MatrixXd J = Eigen::MatrixXd::Zero(15, _n);
J.block(0, _map_frame_idx.at(_frame_to_keep), 15, 15) = absolutePriorJacobian(T_f_w);
Eigen::MatrixXd J_tilde = J * _U;
_map_frame_inf.emplace(_frame_to_keep, sqrtInformation(J_tilde * _Sigma.asDiagonal() * J_tilde.transpose(), _eps));
```

`_Sigma` is the inverse of the eigenvalues kept by `rankReveallingDecomposition`. The unobservable directions
(position, yaw) come out of the Schur complement with eigenvalues of about 1e-10. They are kept (flaw 2), so `_Sigma`
holds variances of about 1e10, and so does the 15×15 covariance of the absolute factor in those directions.
`sqrtInformation` then drops every eigenvalue below a tolerance *relative to the largest one*:

```cpp
// sqrtInformation, e035ccc
const double tol = 1e-12 * std::max(l.maxCoeff(), 0.0);   // ~1e-12 * 1e10 = 1e-2
for (int i = 0; i < l.size(); i++)
    if (l(i) > tol && 1 / l(i) > eps)
        s(i) = 1 / std::sqrt(l(i));
```

Every variance below about 1e-2 is treated as zero, and its direction gets no information. Those are the
well-measured directions: rotation (variances ~1e-6 rad²), velocity, biases. The covariance spans about 16 orders of
magnitude, more than this relative tolerance can handle.

## Flaw 2: `rankReveallingDecomposition` keeps rounding noise

```cpp
// rankReveallingDecomposition, e035ccc
Eigen::VectorXd d_full = (saes.eigenvalues().array() > _eps).select(...);   // _eps = 1e-12, absolute
```

The eigen solver's error is about 1e-16 of the largest eigenvalue, about 1e-9 for a prior whose largest eigenvalue
is 1e7. An eigenvalue of 1e-10 is rounding noise, but it passes the absolute threshold. Kept, its inverse feeds
flaw 1, and in the dense prior it enters the prior residual `_marginalization_residual = Sigma^1/2 U^T b`, amplified
by about 1e5 in a direction where the prior's Jacobian is about 1e-5. The KLD between two priors (`computeKLD`) is then
dominated by these directions and can come out as NaN.

## Evidence

A synthetic dense prior with exactly the sparse structure (an absolute factor plus 4 pose-to-landmark factors): the
absolute block has information from 0.1 to 1e7 on its 11 observable directions and 1e-10 on the 4 unobservable ones.
It is decomposed with `rankReveallingDecomposition`, then `sparsifyVIO` runs. Results with SaDVIO's code and with the
two fixes below (run in the SaDLIO copy, whose functions are identical to `e035ccc` apart from the fixes):

| Code | Rank kept | Largest information of the absolute factor (true: 1e7) | Relative error of the absolute factor | KLD dense → sparse |
|---|---|---|---|---|
| SaDVIO `e035ccc` | 26 / 27 | 22.9 | 1.0 (all lost) | NaN |
| SaDVIO, unobservable eigenvalues exactly 0 | 25 / 27 | 2.9 | 1.0 | 149 |
| with fix 2 (relative rank threshold) | 23 / 27 | 1.0004e7 | 0.0014 | 0.61 |

The second row shows that flaw 1 does not need the 1e-10 eigenvalues: with exact zeros, rounding still leaves two
noise eigenvalues above 1e-12.

On SaDLIO's real runs (LiDAR motion links, a prior without landmarks): KLD between the dense prior and its sparse
version 120 before the fixes, 0 after.

Not done: a SaDVIO run (e.g. EuRoC with `sparsification: 1`) before and after. That would show how much this changes
SaDVIO's trajectories.

## Proposed fixes

**Fix 2, rank threshold relative to the largest eigenvalue** (and the old absolute value as a floor):

```diff
 void Marginalization::rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d) {

     int n = A.rows();
     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(A);
-    Eigen::VectorXd d_full = Eigen::VectorXd((saes.eigenvalues().array() > _eps).select(saes.eigenvalues().array(), 0));
+    // Eigenvalues below 1e-12 of the largest are rounding noise
+    const double tol = n > 0 ? std::max(_eps, 1e-12 * saes.eigenvalues().maxCoeff()) : _eps;
+    Eigen::VectorXd d_full = Eigen::VectorXd((saes.eigenvalues().array() > tol).select(saes.eigenvalues().array(), 0));

     int j = 0;
     for (int i = 0; i < n; i++) {
-        if (d_full(i) > _eps)
+        if (d_full(i) > tol)
             j++;
     }
 ...
     for (int i = 0; i < n; i++) {
-        if (d_full(i) > _eps) {
+        if (d_full(i) > tol) {
             U.col(k) = saes.eigenvectors().col(i);
```

This alone restores the absolute factor in the landmark case (table row 3). It changes the rank of every prior with
such noise eigenvalues, so it also affects the dense mode.

**Fix 1, without kept landmarks: information form, exact.** When no landmark is kept (`_n == 15`), the absolute factor
is the only sparse factor and its Jacobian J (15×15) is invertible, so the information J^-T A J^-1 represents the dense
prior A exactly (KLD 0), with no covariance step:

```diff
     // For absolute frame factor
+    if (_n == 15) {
+        const Eigen::Matrix<double, 15, 15> J_inv = absolutePriorJacobian(T_f_w).inverse();
+        const Eigen::MatrixXd A                    = _U * _Lambda.asDiagonal() * _U.transpose();
+        Eigen::MatrixXd Om                         = J_inv.transpose() * A * J_inv;
+        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(0.5 * (Om + Om.transpose()));
+        const Eigen::VectorXd l = saes.eigenvalues().cwiseMax(0.0).cwiseSqrt();
+        _map_frame_inf[_frame_to_keep] = saes.eigenvectors() * l.asDiagonal() * saes.eigenvectors().transpose();
+        return true;
+    }
     Eigen::MatrixXd J                                     = Eigen::MatrixXd::Zero(15, _n);
```

(`operator[]` rather than `emplace`, so that a value left in `_map_frame_inf` cannot shadow the new one.)

**Still open, with kept landmarks:** after fix 2 the absolute factor keeps its information, but the sparse prior is not
exact on a rank-deficient prior that has the sparse structure (KLD 0.61, row 3). The per-factor covariance formula
assumes a full-rank prior. An information-form computation for the landmark case would be the next step; SaDLIO meets
this case again in its patch-landmark mode (M4).

## Reproduction tests

For `cpp/tests/marginalization_test.cpp` of SaDVIO, after `MarginalizationSparseKLDTest.sparsePriorIsTheKLDMinimum`
(inside `namespace isae`, using the file's `SparseVIOSetup`). Both fail on `e035ccc` and pass with the two fixes
(checked in the SaDLIO copy, which has the same helper with its own landmark class).

```cpp
// The absolute factor must keep the information of the observed directions when the prior also has unobservable
// ones (position and yaw, eigenvalues ~1e-10). e035ccc: largest information 22.9 instead of 1e7
TEST(MarginalizationSparseKLDTest, absoluteFactorKeepsTheObservedInformation) {
    std::srand(4321u);
    SparseVIOSetup setup(4);
    const int n = setup.marg._n;
    const std::vector<int> observed = {0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14}, gauge = {2, 3, 4, 5};
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(15, 11), G = Eigen::MatrixXd::Zero(15, 4);
    for (int k = 0; k < 11; k++)
        E(observed[k], k) = 1;
    for (int k = 0; k < 4; k++)
        G(gauge[k], k) = 1;
    const Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(Eigen::MatrixXd::Random(11, 11)).householderQ();
    Eigen::VectorXd scales(11);
    for (int k = 0; k < 11; k++)
        scales(k) = std::pow(10.0, -1 + 0.8 * k); // 0.1 ... 1e7
    Eigen::MatrixXd D     = Eigen::MatrixXd::Zero(n, n);
    D.block(0, 0, 15, 15) = E * Q * scales.asDiagonal() * Q.transpose() * E.transpose() + 1e-10 * G * G.transpose();
    for (int k = 0; k < 4; k++) {
        Eigen::Matrix3d B                     = Eigen::Matrix3d::Random();
        D.block(15 + 3 * k, 15 + 3 * k, 3, 3) = 100 * (B * B.transpose() + 0.5 * Eigen::Matrix3d::Identity());
    }
    setup.setDensePrior(setup.J.transpose() * D * setup.J);
    ASSERT_TRUE(setup.marg.sparsifyVIO());

    const Eigen::MatrixXd S       = setup.marg._map_frame_inf.at(setup.marg._frame_to_keep);
    const Eigen::MatrixXd D_frame = D.block(0, 0, 15, 15);
    const double largest          = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(S * S).eigenvalues().maxCoeff();
    EXPECT_GT(largest, 0.99e7) << "information of the best-observed direction of the kept frame";
    EXPECT_LT((S * S - D_frame).norm() / D_frame.norm(), 1e-2);
}

// Without kept landmarks the sparse prior represents the dense one exactly. e035ccc: rank 15 instead of 11 (the
// test stops there); with fix 2 only: relative error 1, KLD 160
TEST(MarginalizationSparseKLDTest, rankDeficientPriorWithoutLandmarksIsRecoveredExactly) {
    std::srand(4321u);
    SparseVIOSetup setup(0);
    const std::vector<int> observed = {0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14}, gauge = {2, 3, 4, 5};
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(15, 11), G = Eigen::MatrixXd::Zero(15, 4);
    for (int k = 0; k < 11; k++)
        E(observed[k], k) = 1;
    for (int k = 0; k < 4; k++)
        G(gauge[k], k) = 1;
    const Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(Eigen::MatrixXd::Random(11, 11)).householderQ();
    Eigen::VectorXd scales(11);
    for (int k = 0; k < 11; k++)
        scales(k) = std::pow(10.0, -1 + 0.8 * k);
    const Eigen::MatrixXd D = E * Q * scales.asDiagonal() * Q.transpose() * E.transpose() + 1e-10 * G * G.transpose();
    const Eigen::MatrixXd A_dense = setup.J.transpose() * D * setup.J;
    setup.setDensePrior(A_dense);
    ASSERT_EQ(setup.marg._n_full, 11) << "the 1e-10 eigenvalues are rounding noise next to 1e7: dropped";
    ASSERT_TRUE(setup.marg.sparsifyVIO());

    const Eigen::MatrixXd A_sparse = setup.sparseInformation();
    EXPECT_LT((A_sparse - A_dense).norm() / A_dense.norm(), 1e-8);
    EXPECT_NEAR(setup.marg.computeKLD(A_dense, A_sparse), 0, 1e-6);
}
```

## Where to look in SaDLIO

- Fixes: `cpp/src/optimizers/marginalization.cpp` (`rankReveallingDecomposition`, `sparsifyVIO`).
- Tests: `cpp/tests/marginalization_test.cpp` (`rankDeficientPriorWithoutLandmarksIsRecoveredExactly`) and
  `cpp/tests/lio_test.cpp` (`SLAMLIOTest.sparsePriorRecoversTheDensePriorOfTheLinks`, the same check on a prior
  built by a full run).
- Background and numbers: `doc/sadlio/BUILD_LEDGER.md`, section M2, "Inherited code fixed".
