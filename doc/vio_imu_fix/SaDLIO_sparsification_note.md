# SaDLIO: an exact sparse prior with kept landmarks (for the patch-landmark mode, M4)

To: the SaDLIO-EXECO maintainers · From: SaDVIO-EXECO `dense_devel` (the IMU fix work) · Date: 2026-10-06
Answers: `doc/sadlio/notes/SaDVIO_marginalization_note.md` (your note of 2026-10-06), section "Still open, with kept
landmarks". Code referred to: SaDLIO `af97f8c`, `cpp/src/optimizers/marginalization.cpp`.

## Summary

- Your two flaws are confirmed in SaDVIO. Your tests fail on SaDVIO `e035ccc` with your exact numbers (22.9 instead of
  1e7, KLD NaN; rank 15 instead of 11).
- The leftover KLD of 0.61 with kept landmarks has a cause and an exact fix. The pseudo-inverse of the rank-deficient
  prior is taken in state coordinates and then mapped through the factor Jacobian. It must be taken in the factors'
  own coordinates. With that change, a rank-deficient prior with the sparse structure is recovered exactly: KLD 2e-12
  instead of 0.61.
- One computation covers both modes: with no kept landmark it reduces to your fix 1 (`_n == 15`). It extends to the
  patch mode as long as each pose-to-plane factor's Jacobian with respect to its plane is invertible (section 3).
- SaDVIO now uses it (uncommitted on `dense_devel`). Its marginalization and IMU tests pass, including your two tests
  and a landmark exactness test.
- Two more items for you:
  - SaDLIO's Schur complement (`Amm_inv`, about line 248) still uses the absolute 1e-12 threshold. SaDVIO now applies
    the same relative threshold there as in `rankReveallingDecomposition`.
  - Patch mode will keep many more landmarks than SaDVIO, so cost matters. Section 2 is the cheap exact form SaDVIO
    runs: no second eigendecomposition and no inverse of J. Section 4 has its cost, and a form that looks cheaper but
    is not accurate.

## 1. Why the state-coordinate pseudo-inverse is wrong

The sparse topology is one absolute factor on the kept frame plus one pose-to-landmark factor per kept landmark (in
patch mode: pose-to-plane). Stack the factors' Jacobians, evaluated at the current state, into J, and write y = J x.
J is square, because each factor brings exactly one new variable block (the frame, then each landmark). It is also
block lower triangular, with diagonal blocks `absolutePriorJacobian` (15×15) and ∂(landmark in frame)/∂(landmark),
which is R for a point. So J is invertible.

For such a tree, the KLD-optimal information of factor i is the inverse of its block of the dense prior's covariance
**in y** (Mazuran et al., closed form). Your code computes that covariance as

  Σ_y = J A⁺ J^T   (A⁺ = U Λ⁻¹ U^T, the pseudo-inverse in x)

That is correct only when A has full rank. A VIO or LIO prior never does, because of the gauge (position, yaw), plus
any degenerate direction: a corridor, a flat floor. A pseudo-inverse does not carry over through a non-orthogonal J:

- A⁺ is zero on null(A), so J A⁺ J^T is zero on the subspace J^-T null(A).
- The prior in y, A_y = J^-T A J^-1, has null space J null(A), and its pseudo-inverse is zero there.
- The two subspaces differ in general unless J is orthogonal.

The absolute-prior Jacobian is not orthogonal: it has an `R·skew(R^T t)` block. So the state-coordinate version zeroes
the variance of the wrong subspace, and the factor covariance blocks it gives are wrong in some observed directions.
That is the 0.61. A check of this cause on the same synthetic prior: with an orthogonal J the state-coordinate form is
exact (relative error 6e-9); with the block-triangular J it is off by 1e-2. The fix is to take the pseudo-inverse
in y:

  Σ_y = (J^-T A J^-1)⁺,  information of factor i = ((Σ_y)_ii)⁺

On a full-rank prior this equals the current closed form, so the KLD-minimum property is kept (your test
`sparsePriorIsTheKLDMinimum` passes unchanged).

## 2. The change, against your `sparsifyVIO`

The direct way is to form A_y = J^-T A J^-1 and take its pseudo-inverse with a second eigendecomposition. It is exact,
but it costs as much as the rank revelation itself, and 3.4–3.7× more than the form below (section 4). SaDVIO now runs
an equivalent form that needs neither the eigendecomposition nor J^-1. Rank revelation (with your fix 2) has already
given A = U Λ U^T (rank r). Write

  W = J^-T U  (n × r, full column rank),  then  A_y = W Λ W^T  and  Σ_y = H H^T  with  H = W (W^T W)^-1 Λ^-1/2

Three properties make this form good:

- **Λ stays apart.** Each direction's variance is still 1/λ, so the well-observed directions keep their relative
  accuracy.
- **W^T W carries only J's conditioning,** not Λ's.
- **J^-T U is a block back-substitution:**
  - each landmark block first: D_i^T W_i = U_i, with D_i its 3×3 Jacobian block;
  - then the frame rows: J_f^T W_f = U_f − Σ_i F_i^T W_i, with F_i the landmark factor's block on the frame's first
    6 columns.

The cost is then that of W^T W, O(n r²). Below is SaDVIO's `sparsifyVIO`, without its branch for line landmarks
(which SaDLIO does not have). The blocks of J are kept as small matrices; J is never formed:

```cpp
bool Marginalization::sparsifyVIO() {

    if (_n == 0)
        return false;

    // Blocks of the factors' Jacobians: the frame block J_f (15 x 15) and, per landmark, a diagonal block D (sz x sz)
    // and a block F on the frame's first 6 columns
    struct FactorBlock {
        std::shared_ptr<ALandmark> lmk;
        int row, col, sz;
        Eigen::MatrixXd D, F;
    };
    Eigen::Affine3d T_f_w                  = _frame_to_keep->getWorld2FrameTransform();
    const int i_frame                      = _map_frame_idx.at(_frame_to_keep);
    const Eigen::Matrix<double, 15, 15> Jf = absolutePriorJacobian(T_f_w);
    std::vector<FactorBlock> blocks;
    int row = 15;
    for (auto tlmk : _lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            FactorBlock b;
            b.lmk = lmk;
            b.row = row;
            b.col = _map_lmk_idx.at(lmk);
            const Eigen::Matrix<double, 3, 9> J_lmk = poseToLandmarkJacobian(T_f_w, lmk->getPose().translation());
            b.sz = 3;
            b.D  = J_lmk.block(0, 6, 3, 3);   // patch mode: d(plane in frame)/d(plane), 3 x 3
            b.F  = J_lmk.block(0, 0, 3, 6);   // patch mode: d(plane in frame)/d(pose)
            row += b.sz;
            blocks.push_back(b);
        }
    }
    if (row != _n)
        return false; // a kept variable the sparse topology does not cover

    // W = J^-T U by block back-substitution: the landmark rows first, then the frame rows
    const int r = static_cast<int>(_U.cols());
    Eigen::MatrixXd W(_n, r);
    Eigen::MatrixXd rhs_f = _U.middleRows(i_frame, 15);
    for (const auto &b : blocks) {
        W.middleRows(b.row, b.sz) = b.D.transpose().partialPivLu().solve(_U.middleRows(b.col, b.sz));
        if (b.F.size() > 0)
            rhs_f.topRows(6) -= b.F.transpose() * W.middleRows(b.row, b.sz);
    }
    W.topRows(15) = Jf.transpose().partialPivLu().solve(rhs_f);

    // H^T = Lambda^-1/2 (W^T W)^-1 W^T; the covariance block of factor i in y is H_i H_i^T
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(r, r);
    G.selfadjointView<Eigen::Lower>().rankUpdate(W.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(G.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        return false;
    Eigen::MatrixXd Ht = llt.solve(W.transpose());
    Ht                 = _Lambda.cwiseSqrt().cwiseInverse().asDiagonal() * Ht;
    auto sigmaBlock    = [&](int i0, int sz) -> Eigen::MatrixXd {
        return Ht.middleCols(i0, sz).transpose() * Ht.middleCols(i0, sz);
    };

    for (const auto &b : blocks) {
        _map_lmk_inf[b.lmk]   = sqrtInformation(sigmaBlock(b.row, 3));
        _map_lmk_prior[b.lmk] = T_f_w * b.lmk->getPose().translation();
    }
    _map_frame_inf[_frame_to_keep] = sqrtInformation(sigmaBlock(0, 15));
    return true;
}
```

`sqrtInformation` takes the eigendecomposition of the small block and returns the symmetric square root of its
pseudo-inverse (eigenvalues below 1e-12 of the largest get no information). The `_n == 15` branch is this same code
with no landmark block, so you can keep it or drop it.

A kept variable that gets no sparse factor (in SaDVIO, lines: `addSparsePriorResiduals` uses points only) gets an
identity diagonal block, no F, and no factor. The other factors' blocks of Σ_y are then exactly those of the prior
with that variable marginalized out. That is correct, and better than failing or giving it a wrong factor. SaDVIO
tests this (`keptLineIsMarginalizedFromTheSparsePrior`). In patch mode, use the same trick for a plane whose 3×3
block is ill-conditioned (section 3).

## 3. What the patch-landmark mode needs for this to hold

1. **Each pose-to-plane factor's Jacobian with respect to its plane must be invertible (3×3).** That is the diagonal
   block of J that makes it invertible. With your minimal 3-parameter plane, check the parameterization's
   singularities. Closest-point π = n·d degenerates as d → 0 (the plane passes through the origin of the frame it is
   expressed in). Spherical (θ, φ, d) degenerates at the poles. For each kept plane, check the smallest singular value
   of that block (relative to its largest). If it is ill-conditioned, give the plane an identity diagonal block and no
   F (it is then marginalized from the sparse prior, as above) and do not give it a factor.
2. **Re-anchor before the prior is built.** A plane anchored to the keyframe being marginalized must be
   re-parameterized (re-anchored to a kept KF) before the factors that enter the dense prior are linearized. Otherwise
   the prior's variables are defined relative to a frame that no longer exists. If you re-anchor afterwards, transform
   the linearized prior exactly: A' = T^-T A T^-1, with T the re-anchoring Jacobian, and the same for b. A plane
   anchored to the kept frame itself makes its sparse factor a unary prior on the plane (identity block, nothing on
   the pose). That is still fine.
3. **Degenerate scenes are handled.** The pseudo-inverse in y treats any null direction the same way as the gauge: a
   corridor, a flat floor, all planes parallel. These are common in LiDAR, so this matters more for you than for
   SaDVIO. The form of section 2 keeps the rank decided in x (J invertible gives rank A_y = rank A), so a degenerate
   direction only has to be caught once, by the rank revelation of `_Ak`.
4. **Mixed units.** The relative threshold of the rank revelation covers rad, m, m/s, the biases and the plane
   parameters together. 1e-12 of the largest eigenvalue was fine on every SaDVIO case. If a weakly observed but real
   direction ever gets dropped, scale the state by a diagonal (for example the square root of diag(`_Ak`)) before the
   rank revelation and undo the scaling afterwards.

## 4. Cost

Measured in SaDVIO's test harness: the same gauge-deficient prior with P kept landmarks, `sparsifyVIO` pinned to 4
efficiency cores, single run. Both forms are exact; the error is the relative error of the recovered dense prior.

| Kept landmarks P | n | Eigendecomposition of A_y (first version) | Block back-substitution + W^T W (current) |
|---|---|---|---|
| 20 | 75 | 0.6 ms (error 5e-9) | 0.3 ms (error 4e-9) |
| 100 | 315 | 24 ms (6e-9) | 7 ms (6e-9) |
| 200 | 615 | 152 ms (7e-10) | 44 ms (7e-10) |
| 300 | 915 | 510 ms (2e-10) | 137 ms (2e-10) |

The rank revelation of `_Ak` (an n×n eigendecomposition) is still there and costs about as much as the first
version's column. At a few hundred planes per prior, that is the term to watch. If patch mode keeps that many, cap the
number kept for the sparse prior: limiting the kept set is the point of sparsification anyway.

**A form that looks cheaper but fails.** With N a basis of null(A) and Y an orthonormal basis of J N, the identity
pinv(A_y) = J (A + J^T Y Y^T J)^-1 J^T − Y Y^T is exact on paper and needs one Cholesky factorization. In double
precision it lost the well-observed directions: relative error 1e3 on the 4-landmark test. The Cholesky solve's
error is about eps × cond(M) × ‖M^-1‖ ≈ 1e-7 absolute, as large as the smallest variances (1e-7 at information 1e7),
and the subtraction cancels the rest. Any form must keep Λ apart, as the one above does.

## 5. Tests worth having for M4

Written on synthetic priors with hand-built Jacobian blocks (as SaDVIO's `SparseVIOSetup` does for points), so
they need no plane class and work before `PlaneLandmark` exists:

1. **Exact recovery.** A rank-deficient dense prior with the sparse structure, A = J^T D J, D block diagonal, with the
   frame block gauge-deficient at 1e-10 and 3×3 plane blocks. Check that the recovered J^T S^T S J equals A (relative
   error < 1e-6) and that the KLD is about 0. This is SaDVIO's `rankDeficientPriorWithLandmarksIsRecoveredExactly`
   with planes in place of points.
2. **KLD minimum.** A generic full-rank prior. Scaling any one factor's information by 0.95 or 1.05 must increase the
   KLD (your `sparsePriorIsTheKLDMinimum`).
3. **Degenerate scene.** All planes parallel, so one more null direction. Still exact.
4. **Ill-conditioned plane block.** A plane close to the parameterization's singularity. It gets no factor, and the
   remaining factors equal those of the prior with that plane marginalized by a Schur complement (as in SaDVIO's
   `keptLineIsMarginalizedFromTheSparsePrior`).
5. **Motion-link mode.** With no landmark block, the result equals your current `_n == 15` branch.

## 6. Results in SaDVIO

Synthetic, SaDVIO's `marginalization_test.cpp` with your setup (`SparseVIOSetup`):

| Case | `e035ccc` | Your fixes 1 + 2 | Pseudo-inverse in y, eigendecomposition (first version) | Pseudo-inverse in y, section 2 (current) |
|---|---|---|---|---|
| 4 kept landmarks: largest info of the absolute factor (true 1e7) | 22.9 | 1.0004e7 | 1e7 | 1e7 |
| 4 kept landmarks: relative error of the absolute factor | 1.0 | 1.4e-3 | 2e-9 | 4e-9 |
| 4 kept landmarks: KLD dense → sparse | NaN | 0.61 | 3e-11 | 2e-12 |
| No kept landmark: rank / KLD | 15 / test stops | 11 / 0 | 11 / 3e-14 | 11 / 3e-14 |

Real runs, `e035ccc` vs the fix, with `sparsification: 1`. Two runs per cell; each before/after pair ran at the same
time, pinned to two equal sets of 8 cores at nice 10:

- **Cost:** marginalization time per keyframe unchanged or slightly lower everywhere. EuRoC SaDVIO took 11–55 ms
  before and 10–51 ms after. The rank revelation of `_Ak` dominates, and the new sparsification is small next to it.
- **SaDVIO, EuRoC (11 sequences):** MH_01 0.079 / 0.098 → 0.060 / 0.058 m and MH_02 0.058 / 0.051 → 0.046 / 0.040 m.
  The rest is within the run-to-run spread (for example MH_04 0.07 / 0.13 before). The absolute factor was nearly empty
  before, yet the runs were already good: the sparse prior then rested on the pose-to-landmark factors, which keep
  most of the information when landmarks are kept.
- **room1 (TUM-VI):**
  - VIO with sparse prior and td: 0.049 / 0.061 → 0.049 / 0.044.
  - Stereo VO with sparse prior: 0.27 / 0.22 → 0.16 / 0.15. Stereo VO only gets the rank threshold (fix 2); its
    sparse path is the Chow-Liu one.
  - Mono VIO: 0.21 ± 0.05 → 0.24 ± 0.04 over 6 runs, which is noise.
- **Expect a larger effect in SaDLIO.** Your motion-link mode keeps no landmark, so there the absolute factor is the
  whole sparse prior, and your runs showed it (KLD 120 → 0).

## Where to look in SaDVIO

- `cpp/src/optimizers/marginalization.cpp`: `nullEigenvalueThreshold` (used by the Schur complement and by
  `rankReveallingDecomposition`), and `sparsifyVIO`.
- `cpp/tests/marginalization_test.cpp`:
  - `rankDeficientPriorWithoutLandmarksIsRecoveredExactly`;
  - `rankDeficientPriorWithLandmarksIsRecoveredExactly`;
  - `keptLineIsMarginalizedFromTheSparsePrior`.
- `cpp/tests/imu_test.cpp`: `inertialPriorWithoutKeptLandmarks` now also checks that the sparse factor is exact (the
  missing lower bound your note pointed out).
- `doc/vio_imu_fix/IMU_FIX_LEDGER.md`: the section "Sparse prior: pseudo-inverse in factor coordinates".
