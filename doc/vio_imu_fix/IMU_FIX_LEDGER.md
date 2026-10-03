# SaDVIO IMU fix loop — ledger

Started 2026-10-02. This file is the loop's state: read it first in every iteration, update it last.
Checklist source: `/home/deos/s.jois/EXECO/SaDVIO-Dense/SaDVIO-EXECO/doc/VIO_fix_+_LIO_Prospects.md`
(Part 2, Issues 1-20 + Minor). Tick items there as they are done.

## Rules (set by the user)

- Goal: every config option works correctly for any user, not just the checked-in config.
- Code: host checkout `/home/deos/s.jois/EXECO/SaDVIO-Dense/SaDVIO-EXECO`, branch `dense_devel`.
  **Do not stage or commit.** The user verifies everything at the end.
- Builds: `cpp/build/isaeslam` is the **frozen pre-fix binary** (baseline; do not rebuild until the
  end). Development binary and tests: `cpp/build_tests/` (FFS off, VDB-GPDF on); runs use
  `run_sadvio.py --bin /root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/cpp/build_tests/isaeslam`.
  At the end, rebuild `cpp/build` and the ROS colcon build with all fixes.
- Build/run: container `sad_vio_dense`, folder `/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel`
  (sync host → container by copying files). **Never touch `/root/SaDVIO-Dense/SaDVIO-EXECO` (simval).**
  Rebuild `cpp/build` and, if the ROS package is touched, its colcon build (it compiles its own copy).
- Layout (set by the user 2026-10-02): `Simulator_Validation/SaD_VIO_data/` (container:
  `/Simulator_Validation/SaD_VIO_data/`) holds **only what the container needs**: the datasets (`tumvi/`)
  and `scratch/`, the working folder of a running binary (removed when empty). Everything else lives in
  the repo under `doc/vio_imu_fix/`: this ledger, `tools/`, `configs/`, `runs/` (results; `run_sadvio.py`
  moves each run here when it ends, with the container's root-owned files handed back to the host user)
  and `logs/` (driver logs). `sync_to_container.sh` does not copy `doc/` into the container. EuRoC could not be downloaded (ETH server down, Research
  Collection rate-limits scripts); TUM-VI is used instead (see Data). RealSense data: leave alone
  (no camera-IMU calibration yet).
- Evaluation tooling and results live in `doc/vio_imu_fix/`; ground truth stays in the dataset folder and
  is read only by `tools/eval_traj.py`, never by SaDVIO code.
- Run the whole loop without stopping per phase; at the end, report a summary per fix.
- Quick checks (`tools/run_quick.sh <label>`) run from a snapshot of the dev binary + library
  (`/root/SaDVIO-Dense/bin_snapshots/<label>/`), so rebuilding during a check cannot leak into it.

## Acceptance rule

A fix is accepted when its unit/synthetic test passes (the test must fail before the fix where
possible). Trajectory metrics are tracked to catch regressions and measure overall effect; a correct
fix may make a metric worse (e.g. removing a fudge knob) — record that honestly, do not tune it away.

## Iteration procedure

1. Read this ledger; pick the next `todo` item in the phase order below.
2. Implement on the host. Keep each issue a self-contained change.
3. Add/extend tests in `cpp/tests/`; build and run them in the container.
4. Sync to the container, rebuild, run the quick evaluation (see Evaluation) where the issue affects
   trajectories.
5. Record: status, files changed, test evidence, metric deltas vs. baseline, notes. Tick the doc.
6. If blocked, mark `blocked` with the reason and move on; revisit at the end.

## Data

**TUM-VI `dataset-magistrale2_512_16`** at `SaD_VIO_data/tumvi/dataset-magistrale2_512_16/`
(verified 2026-10-02: MD5 matches the server, all files extracted, tar deleted).

- EuRoC folder format (`mav0/cam0`, `cam1`, `imu0`, `mocap0`); readable by SaDVIO's offline reader.
- Stereo 512×512 fisheye, 20 Hz, 10,810 pairs (540.5 s), identical left/right stamps. Model:
  equidistant (Kannala-Brandt), intrinsics/distortion in `dso/camchain.yaml` (SaDVIO `equidistant`
  also needs an `rmax`).
- IMU 200 Hz, 107,763 samples covering the whole run. Already axis-calibrated in this export.
- Calibration (Kalibr): `dso/camchain.yaml` gives `T_cam_imu`. For SaDVIO with the IMU as the body
  frame: camera `T_BS` = `T_imu_cam` = inverse(`T_cam_imu`); IMU `T_BS` = identity.
- Noise (`dso/imu_config.yaml`, inflated values recommended by TUM-VI): acc 0.0028 / 0.00086,
  gyro 0.00016 / 0.000022 (noise density / random walk).
- Ground truth: `mav0/mocap0/data.csv` = pose of the IMU frame in the mocap frame, **only at
  2.1-49.8 s and 490.2-540.5 s** (start and end inside the mocap room; no GT in between). Hard
  sequence: long walk through a building, varying lighting.
- No SaDVIO number exists for TUM-VI; published VIO results (ORB-SLAM3, Basalt, OpenVINS) on
  magistrale2 can serve as an order-of-magnitude reference.

**TUM-VI `dataset-room1_512_16`** at `SaD_VIO_data/tumvi/dataset-room1_512_16/` (verified
2026-10-02: MD5 matches the server, all files extracted, tar deleted).

- 2,821 stereo pairs at 20 Hz (141.0 s), identical left/right stamps; IMU 28,122 samples at
  200 Hz covering the whole run; all timestamps monotonic.
- Same rig: `dso/camchain.yaml` and `dso/imu_config.yaml` are byte-identical to magistrale2, so
  one SaDVIO dataset yaml serves both sequences.
- Ground truth for the **whole run** (mocap, 120 Hz), with 5 short dropouts totalling 2.5 s
  (longest 1.06 s).

## Evaluation

- Sequences: **room1** for the quick per-fix check (short, full GT); **magistrale2** for phase-end
  and final comparisons (long-range drift).
- room1 metrics: ATE RMSE over the whole run (SE3 align for stereo, Sim3 for mono with the scale
  reported) and RPE.
- Modes: `bimono` (VO reference), `bimonovio`, `monovio`. Several runs per cell (non-determinism).
- magistrale2 metrics, following the TUM-VI benchmark protocol: align on the GT segments (start +
  end, SE3 for stereo, Sim3 for mono with the scale reported) and report ATE RMSE over the GT
  segments; also report end drift after aligning on the start segment only. Plus completeness, resets/failures,
  NaNs, run time, and VIO logs (biases, velocity at the static start).
- Tools: `doc/vio_imu_fix/tools/` (moved from `SaD_VIO_data/tools/` on 2026-10-02).

## Phase order and status

| Phase | Item | Status | Notes |
|---|---|---|---|
| 0 | Setup: get test data into `SaD_VIO_data/` | done | EuRoC unreachable; TUM-VI magistrale2 and room1 downloaded by the user, verified and extracted 2026-10-02 |
| 0 | Setup: SaDVIO dataset yaml for TUM-VI (fisheye, Kalibr extrinsics, noise) | done | `configs/dataset/tumvi_512_ds.yaml` + `configs/base_config.yaml` |
| 0 | Setup: sync host → container workspace, build cpp + tests | done | `tools/sync_to_container.sh`; container code was already identical to host |
| 0 | Setup: evaluation tools (runner, ATE/RPE, aggregation) | done | `tools/run_sadvio.py`, `tools/eval_traj.py`, `tools/run_matrix.sh` |
| 0 | Setup: baseline runs (all modes, both sequences, before any fix) | done | label `baseline`, 3 runs per cell, frozen `cpp/build` binary |
| 1 | Issue 17 — tests that can fail | done | see per-fix log; 4 IMU tests now fail by design until Issues 3, 12, 13 |
| 1 | Reproduction logging (solver usability, NaNs, rejected IMU, durations, factor counts) | done | `log_slam/vio_diag.csv`, one row per KF optimization |
| 2 | Issue 10 — so3_leftJacobian zero guard | done | `GeometryTest.so3LeftJacobian` |
| 2 | Issue 11 — ESKF velocity gain rotation | done | test passes; room1 VIO metrics worse (exposes bias problems, see log) |
| 2 | Issue 1 — gyro random walk key | done | `ImuConfigTest.noiseKeysAreReadIntoTheRightFields` |
| 2 | Issue 7 — offline timing | done | reader rewritten on `ImuImageMerger`; room1 stereo VIO 0.82 → 0.22 m ATE |
| 2 | Issue 8 — ROS reader ordering | done | ROS reader uses the same merger; colcon build OK; not run on ROS data |
| 2 | Issue 6 — IMU T_BS (code part: offline reader applies rotation; T_s_f convention) | done (code) | config part waits for the RealSense camera-IMU calibration |
| 2 | Issue 18 — time gaps / durations | done | true step durations; no IMU factor across gaps |
| 2 | Issue 20 — config validation at load | done | `validateConfig()`; errors refuse to start |
| 3 | Issue 2 — VIInit bias application and prior | done | `viInitRecoversGyroBiasAndVelocities` |
| 3 | Issue 3 — IMUFactorInit residual + Jacobians | done | `viInitModelMatchesTiltedScaledWorld`, Jacobian test |
| 3 | Issue 9 — init handoff tests, mono `_6d_velocity` scale | done | mono motion model rescaled; handoff by reasoning + ordered readers |
| 3 | Issue 18 — init acceptance / stationarity / solver usability | done | `staticImuInitialization`, `inertialInitAccepted`; also a NaN at level start |
| — | Found in the loop: timer data race (the segfaults) | done | `timer` stack thread-local; `TimerTest.ticTocFromSeveralThreads` |
| 4 | Issue 12 — IMUFactor pose-j rotation Jacobian | done | reverted to the first-commit form; tests pass |
| 4 | Issue 4 — bias linearization bookkeeping | done | b_lin per preintegration, factor uses b_i + d − b_lin, re-integration |
| 5 | Issue 5 — low-parallax drop + marginalization factor check | done | preintegration bridged by re-integration; `imuFactorUsable` |
| 5 | Issue 16 — mono marginalization | done | single-view landmarks kept; mono back end marginalizes |
| 6 | Issue 13 — IMUPriordx whitening | done | |
| 6 | Issue 14 — sparse-prior recovery Jacobians | done | dense-vs-sparse KLD check not built |
| 6 | Issue 15 — sparse-prior gating (Analytic) | done | shared `addSparsePriorResiduals`; + 2 crashes fixed |
| 7 | Issue 19 — online time offset | done | end-to-end re-check passes: converges to ~2.5 ms from 0 and from +10 ms |
| 7 | Issue 20 — Numeric optimizer stubs | done (refused) | combinations rejected at load; not implemented |
| 7 | Minor items (Prior1D, parallax denominator, mono self-reference) | done | |
| 8 | Final: full option matrix, summary per fix | done | option matrix: no crash in any supported cell; `final2` matrix; main + colcon builds rebuilt |

## Setup notes

- **Camera model.** SaDVIO's `equidistant` model ignores distortion coefficients (pure `r = f·θ`),
  which would leave ~2 px error at the image edge with TUM-VI's Kannala-Brandt calibration. The
  dataset yaml uses the double-sphere model instead (SaDVIO implements Usenko's DS exactly), with
  Basalt's published TUM-VI 512 calibration (`configs/tumvi_512_ds_calib.basalt.json`), whose
  extrinsics agree with the dataset's Kalibr camchain to 0.03° / 2 mm.
- **Frames.** Camera `T_BS` = Basalt `T_imu_cam`; IMU `T_BS` = identity, so the SaDVIO frame is the
  IMU frame — the same frame as the mocap GT, so no hand-eye transform is needed in evaluation.
- **Base config.** Repo `ros/config/config.yaml` with `dataset_id: tumvi_512_ds`, `enable_visu: 0`,
  `estimate_td: 0` (hardware-synchronized rig). Everything else at repo defaults (`AngularAnalytic`,
  `multithreading: 0`, `marginalization: 0`, `sparsification: 0`, window 12, 150 FAST features).
- **Runs.** `tools/run_sadvio.py` runs the offline binary in the container per run folder
  (`runs/<label>/<seq>/<mode>/run_NN/`) and stops it after 30 s without output growth (the binary
  never exits at end of data). `EXECO_PERFRAME_LOG=1` is set (VO modes write per-frame poses).
- **Evaluation.** `tools/eval_traj.py` uses `log_slam/results.csv` (keyframe poses, last write per
  timestamp kept) in all modes.
- **Existing unit tests (pre-fix).** Run one by one (`--gtest_filter`, 120 s limit each): all pass
  except `LineFeatureTest.LineFeatureDetection` (unrelated to IMU). Running the whole suite in one
  process spins at 100 % CPU for >10 min, so some test interaction hangs; noted for Issue 17. Note
  the passes are weak evidence for the Jacobian tests (Issue 17).
- **Smoke run** room1 `bimono`: ATE 0.156 m, RPE 0.025 m/1 s, coverage 98 %, ~18 s.

## Baseline results

Label `baseline` (frozen pre-fix binary, base config, 3 runs per cell). ATE in m (SE3 align for
stereo, Sim3 for mono; magistrale2 over the start+end GT segments), RPE translation over 1 s in m,
rotation in deg, coverage = time span of the final trajectory / sequence span, wall time in s.

| seq | mode | runs | ate_rmse | rpe_t_1s | rpe_r_1s_deg | scale | end_drift | coverage | resets | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| magistrale2 | bimono | 3 | 6.737 ± 0.169 | 0.045 ± 0.004 | 1.517 ± 0.148 | 1.000 ± 0.000 | 14.463 ± 0.377 | 0.995 ± 0.000 | 0.0 ± 0.0 | 70.8 ± 0.9 |
| magistrale2 | bimonovio | 3 | 3.187 ± 4.002 | 1.033 ± 1.231 | 4.628 ± 1.506 | 1.000 ± 0.000 | — | 0.193 ± 0.108 | 4.3 ± 1.2 | 111.5 ± 7.4 |
| magistrale2 | mono | 3 | 0.252 ± 0.059 | 0.550 ± 0.156 | 52.604 ± 15.801 | 0.880 ± 0.172 | — | 0.005 ± 0.001 | 0.0 ± 0.0 | 68.1 ± 0.0 |
| magistrale2 | monovio | 3 (1 no traj) | 0.840 ± 0.075 | 0.628 ± 0.119 | 5.380 ± 1.217 | 0.177 ± 0.168 | — | 0.170 ± 0.121 | 0.0 ± 0.0 | 69.4 ± 0.9 |
| room1 | bimono | 3 | 0.219 ± 0.030 | 0.027 ± 0.002 | 1.025 ± 0.022 | 1.000 ± 0.000 | — | 0.982 ± 0.000 | 0.0 ± 0.0 | 18.1 ± 0.0 |
| room1 | bimonovio | 3 | 0.824 ± 0.245 | 0.137 ± 0.031 | 2.814 ± 0.346 | 1.000 ± 0.000 | — | 0.980 ± 0.001 | 0.0 ± 0.0 | 38.2 ± 4.3 |
| room1 | mono | 3 | 0.188 ± 0.054 | 0.211 ± 0.006 | 10.190 ± 3.201 | 0.866 ± 0.092 | — | 0.144 ± 0.138 | 0.0 ± 0.0 | 18.1 ± 0.0 |
| room1 | monovio | 3 (2 no traj) | 0.739 | 0.829 | 5.756 | 0.011 | — | 0.017 ± 0.024 | 0.0 ± 0.0 | 18.8 ± 0.9 |

**Reading:**
- Stereo VIO is worse than stereo VO on room1 (0.82 vs 0.22 m ATE, RPE 0.14 vs 0.03 m) — the
  reported symptom reproduces on a dataset with a trusted calibration.
- On magistrale2 stereo VO runs through (coverage 99.5 %, end drift 14.5 m), stereo VIO does not
  (4.3 resets on average; final trajectory covers 19 % of the sequence).
- Mono VO and mono VIO barely produce trajectories (coverage 0.5-17 %); their ATE values are over
  tiny fragments and not meaningful.
- **Coverage caveat:** SaDVIO's `profiling()` truncates `results.csv` when it (re)initializes, so
  after resets only the last segment is evaluated; `resets` counts them.


## Per-fix log

(one entry per item: what changed, files, test evidence before/after, metric deltas, caveats)

### Issue 17 — tests that can fail (done 2026-10-02)

**Changed (tests only, no library code):**
- New `cpp/tests/jacobian_check.h`: `JacobiansMatch` compares every analytic Jacobian column with a
  central finite difference (h = 1e-6, relative tolerance 1e-5) and reports block/column on
  failure; `JacobiansMatchAtRandomPoint` repeats it with all parameters offset by U(±0.05), which
  exposes bugs invisible at the zero increment; `ResidualNorm`.
- `imu_test.cpp`, `residual_test.cpp`: the 9 `GradientChecker` blocks (failure only logged, then
  analytic-vs-analytic asserts) replaced by real checks at zero and at a random point. The
  `IMUFactorInit` check now uses its own results (was asserting on the `IMUFactor` results).
- New IMU tests: `biasJacobiansMatchReintegration` (bias Jacobians vs re-integrating with a
  perturbed bias — independent of the Jacobian formulas), `imuFactorJacobiansAwayFromZero`
  (rotating, accelerating trajectory; Issue 12), `imuFactorInitJacobiansAtNonUnitScale` (λ = ln 3;
  Issue 3), `imuPriorJacobiansWithDenseWhitening` (Issue 13).
- Reproducibility: all fixtures seed `std::srand(12345u)` in `SetUp`; the three `srand(time(0))`
  and the `random_device` seed replaced by fixed seeds.

**Evidence:**
- Before: whole suite in one process spun >10 min at 100 % CPU (time-seeded randomness; one
  `while` sampling loop presumably never terminated); per-test runs all passed except
  `LineFeatureDetection`.
- After: whole suite runs in ~1 s, identical results on 3 consecutive runs.
- The new checks fail exactly where the review predicted, nowhere else in the IMU factors:
  `IMUFactor` block 1 (pose j) cols 0-2 away from zero (max abs err ~1.1e3 on columns of scale
  ~1.5e4; whitened); `IMUFactorInit` block 5 (scale) at λ ≠ 0 (err 39 on scale 58);
  `IMUPriordx` blocks 1-3 (velocity, biases) with dense whitening.
- `biasJacobiansMatchReintegration` **passes**: the preintegration bias Jacobians are correct.

**Known failing tests after this item (expected):** `ImuTest.predictionPositionVelocity`,
`imuFactorJacobiansAwayFromZero` (→ Issue 12), `imuFactorInitJacobiansAtNonUnitScale` (→ Issue 3),
`imuPriorJacobiansWithDenseWhitening` (→ Issue 13).

**Found, outside the IMU scope (not fixed; for the user):**
- `ResidualTest.angularTest`: `AngularErrCeres_pointxd_depth` anchor-pose Jacobian (block 1,
  rotation columns) is wrong even at zero (err ~0.02). The class is not used in `src/`.
- `ResidualTest.RelativePose6DResidual`: `Relative6DPose` translation columns wrong at zero
  (err ~0.04); used by the relative-pose code of both analytic optimizers
  (`BundleAdjustmentCERESAnalytic.cpp:959`, `AngularAdjustmentCERESAnalytic.cpp:1239`).
- `ResidualTest.ESKFLmkTest` (ESKF landmark refinement, 0.071 m vs 0.02 m tolerance),
  `LineFeatureTest.LineFeatureDetection` / `LineFeatureMatching`: fail deterministically with the
  fixed seed (they passed or failed by luck before).
- Build: `-DISAESLAM_WITH_VDBGPDF=OFF` fails to link (`MarginalDepthInjector` calls `VDBGPDFMap`
  without the option guard). Tests are built in `cpp/build_tests` with FFS off, VDB-GPDF on.

### Reproduction logging (done 2026-10-02)

**Changed (logging only; estimation unchanged):**
- `AOptimizer.h`: `VIOptimStats` (solver usable, termination, initial/final cost, iterations, IMU
  factors inserted), `recordVIStats()`, `getLastVIStats()`. Recorded after
  `localMapVIOptimization` and both `localMapVIOptimizationTd` variants.
- `AOptimizer::addIMUResiduals` returns the number of preintegration factors inserted (was 0).
- `IMU`: `_integrated_dt` (time integrated since the KF) and `_n_clamped_steps` (steps whose
  dt > 1 s was replaced by 1/rate), with getters.
- `SLAMCore::logVIODiag()` → `log_slam/vio_diag.csv`, one row per KF window optimization (and the
  post-init optimization): KF/last-KF stamps, factor dt vs integrated dt, clamped steps, rejected
  IMU samples (running total, counted where `processIMU()` returns false in both pipelines),
  factor count, solver outcome, non-finite window states, KF velocity and biases.

**First reading** (room1 `bimonovio`, current estimator code, dev binary): 1,423 window
optimizations; all usable; 0 non-finite states; 0 rejected / clamped IMU samples; integrated dt =
factor dt exactly. 120 windows end in NO_CONVERGENCE (20-iteration cap). 3-11 IMU factors per
12-KF window. Accelerometer bias after init (0.016, -0.028, **-0.224**) m/s² vs Basalt's
calibration (-0.001, -0.008, 0.000) — consistent with Issue 2 (Δba applied twice).

### Issue 10 — `so3_leftJacobian` zero guard (done 2026-10-02)

**Changed:** `cpp/include/utilities/geometry.h` — `so3_leftJacobian` rewritten in the same closed form
as `so3_rightJacobian`, `I + (1−cos θ)/θ²·[w]× + (θ−sin θ)/θ³·[w]×²` (mathematically identical to
the previous axis-angle form), with the first-order expansion `I + ½[w]×` below |w| = 1e-5.

**Evidence:** new `GeometryTest.so3LeftJacobian` (in `basic_test.cpp`): `Jl(0)` is finite and equal
to I (before: `w / |w|` = 0/0 → NaN, so the `allFinite` assertion fails by construction); `Jl`
matches finite differences of `Exp` (max err < 1e-6) at |w| = 1e-9, ~2e-6, 0.37 and 1.8 rad;
`Jl(w) = Exp(w)·Jr(w)`. Passes.

**Effect on trajectories:** none expected — the ESKF only reaches `Jl(0)` when the visual and IMU
rotations agree exactly (noise-free data); not evaluated on TUM-VI.

### Issue 11 — ESKF velocity gain rotation (done 2026-10-02)

**Changed:** `ESKFEstimator::updateVelocity()` (new static function, used by
`estimateTransformBetween`): `H = R_w_f1ᵀ`, `K = Hᵀ(HHᵀ + C)⁻¹` (identity prior covariance kept),
`v = v_pred + K·(dV − H(v_pred − v1 − g·dt))`. Before: `K = R_w_f1ᵀ(R_w_f1R_w_f1ᵀ + C)⁻¹`, i.e. the
body-frame innovation was rotated by the inverse rotation.

**Evidence:** new `ESKFVelocityTest.updateVelocityFrames`: with an accurate measurement the update
returns the true velocity (|err| < 1e-6) for a 90° yaw and two general rotations; with anisotropic
noise only the error along the poorly measured body axis remains, exactly `R·diag(c/(1+c))·Rᵀ·err`.
Passes. Same test logic with the old formula: errors of 0.72, 0.47, 0.52 m/s.

**Effect on room1 (dev binary, 3 runs; baseline in brackets):**
- `bimonovio` ATE 2.69 / 91.65 / 1.04 m [0.68 / 0.63 / 1.17], RPE 1 s 0.55 / 26.5 / 0.14 m
  [0.11 / 0.13 / 0.18] → **worse**, one run diverged.
- `monovio`: 1 of 3 runs without trajectory (baseline 2 of 3); Sim3 scale ~0.01 (meaningless,
  init still broken).
- Diagnosis (`vio_diag.csv`): the biases run away — diverged run ends with ba = (6.9, 3.4, 1.0)
  m/s², bg = (−0.34, 0.40, −0.11) rad/s; the "good" run ba = (−0.35, 1.29, −0.05), bg ~0.03 rad/s
  (Basalt calibration: |ba| < 0.01, |bg| < 1e-4). With a correct gain the front end now trusts the
  IMU velocity (prior covariance I ≫ C), so wrong biases propagate into the velocity seeds instead
  of being scrambled away. Expected to recover with Issues 1, 2 and 4 (bias noise, init bias,
  bias bookkeeping). Not tuned away; the fix is kept.

### Issue 1 — gyroscope random walk read from the accelerometer key (done 2026-10-02)

**Changed:** `adataprovider.cpp:85` reads `gyroscope_random_walk` into `bgyr_noise`. Every dataset
yaml in `ros/config/dataset/` with an `imu` block has that key (checked), so no shipped config breaks.

**Evidence:** new `ImuConfigTest.noiseKeysAreReadIntoTheRightFields` loads a yaml with four distinct
noise values and checks each field (before: `bgyr_noise` = 0.4 instead of 0.2). Passes. On TUM-VI the
gyro bias random walk drops from 8.6e-4 to 2.2e-5 (39× tighter).

**Effect on room1** (cumulative 10+11+1, label `issue01`): `bimonovio` ATE 21.7 ± 27.0 m — still
diverging (biases), no clear change vs `issue11`. Only the 3 `bimonovio` runs and `monovio` run_00
are valid: I rebuilt the dev binary for Issue 7 while the check ran, so `monovio` run_01/02 used
the Issue 7 code. (This is why quick checks now run from binary snapshots.)

### Issue 7 — offline timing (done 2026-10-02)

**Changed:** `EUROCGrabber` (offline reader) rewritten on a new shared class `ImuImageMerger`
(`adataprovider.h/.cpp`):
- hard-coded +15 ms IMU shift removed; `dt_imu_cam` now applied offline with the ROS convention
  (t_cam = t_imu − dt_imu_cam, documented on `imu_config::dt_imu_cam`);
- image frames keep the image stamp (were given the IMU stamp);
- IMU and images merged strictly in time order; each image frame carries an IMU measurement at the
  image time (the coinciding sample, else linear interpolation), so preintegration ends exactly at
  the image; images are no longer discarded for lacking an IMU sample within ±2.5 ms;
- stereo pairing uses `stereo_sync_tolerance_ms` (was a hard-coded 20 ms); an unmatched image no
  longer swallows an IMU sample (the old sync-error branches popped one into a list never queued);
- IMU stamps parsed as integers (`stod` rounded 19-digit ns stamps to ~256 ns); non-increasing IMU
  stamps dropped with a count; an unreadable image is skipped instead of ending the reader;
- VO modes ignore the IMU file (no more need for the synthetic zero-IMU rows);
- `ADataProvider::addFrameToTheQueue` takes integer ns (was `double`); `queueSize()` added.

**Evidence:** new `DataProviderTest` (synthetic stereo EuRoC folder, 2 ms clock offset, linear IMU
signal, one unmatched right image): frames strictly time-ordered, image frames keep image stamps,
interpolated IMU exact to 1e-9, all 13 IMU samples before the last image emitted, unmatched image
dropped; VO mode works without `imu0/`; out-of-order IMU / late images rejected by the merger.
All pass.

**Effect on room1** (cumulative 10+11+1+7, label `issue07`, binary snapshot):

| mode | ATE (m) | RPE 1 s (m) | coverage | note |
|---|---|---|---|---|
| bimonovio | 0.224 ± 0.077 [baseline 0.824] | 0.027 [0.137] | 0.81 | run_02 crashed (segfault) at 63 s |
| monovio | 0.378 ± 0.036 [no traj in 2/3] | 0.154 [0.829] | 0.975 [0.017] | Sim3 scale 0.86 |

The 15 ms shift was the dominant problem on this hardware-synchronized data: stereo VIO now matches
stereo VO (0.219 m) and mono VIO produces a full trajectory. Biases at the end of the stereo runs:
ba ≈ 0.03-0.25 m/s², bg ≈ 0.001-0.007 rad/s (still larger than Basalt's calibration).

**Open:** one segfault (exit 139) in 42 runs so far (`issue07/room1/bimonovio/run_02`, after 63 s).
Four reruns of the same binary under gdb did not reproduce it. Crashes are now counted in the
evaluation summary (`crashed`); to be investigated if it recurs.

### Issue 8 — ROS reader ordering (done 2026-10-02)

**Changed:**
- `ros/src/sensorSubscriber.h` rewritten: one template `SensorSubscriberT<ImageMsg>` for raw and
  compressed images (the two classes were copies). The sync thread moves IMU messages into an
  `ImuImageMerger` (body frame, camera clock, `dt_imu_cam` read per message), pairs stereo images
  by tolerance, and emits an image only once the IMU stream has passed its stamp, preceded by the
  IMU before it and with an interpolated IMU at its stamp. Integer ns stamps; every queue access
  under its mutex; IMU subscription depth 10 → 1000 (bag playback bursts). Compressed images keep
  their previous 1920×1080 resize and best-effort QoS.
- Both VIO pipelines: an IMU measurement that `processIMU()` rejects is detached from its frame
  (`Frame::clearIMU()`), so a KF can no longer keep an unprocessed IMU (zero covariance → NaN); a KF
  without IMU gets one synthesized from the last valid sample, as before.

**Evidence:** the merger logic is covered by `DataProviderTest.mergerRejectsOutOfOrderData`; the
ROS package builds (`colcon build` with FFS, VDB-GPDF, LAS2: OK). **Not run on ROS data** (the
RealSense bags are out of scope until calibrated).

### Issue 6 — IMU T_BS, code part (done 2026-10-02; config part waits for calibration)

**Changed:**
- `ADataProvider::createImuSensor` rotates acc and gyr by the IMU `T_BS` rotation (sensor → body),
  so every reader applies it; the ROS subscriber no longer rotates (it did; the offline reader did
  not, so the same yaml behaved differently offline and online).
- `loadIMUConfig`: comment that the IMU keeps `T_BS` (sensor → body) while cameras store
  `T_BS⁻¹`; warning when the `T_BS` rotation is not a proper rotation, and when it has a
  translation (ignored: no lever-arm model — make the IMU the body frame instead).

**Evidence:** new `ImuConfigTest.measurementsAreRotatedIntoTheBodyFrame` (R_x(90°): sensor
(0, 9.81, 0) → body (0, 0, 9.81), gyro likewise). Passes. No effect on TUM-VI runs (identity `T_BS`).

**Not done (needs data/calibration, out of scope for now):** RealSense yaml frames (inspect the
bags, Kalibr camera-IMU calibration, camera `T_BS` relative to the IMU).

### Issue 18, time-gap part (done 2026-10-02)

**Changed:**
- `IMU::processIMU` integrates every step over its true duration (a step > 1 s used to be replaced
  by 1/rate while the factor kept the real duration). Steps longer than `IMU::maxStepDt()` (10
  nominal periods, at least 50 ms) are counted as gap steps (`getGapSteps()`, also in
  `vio_diag.csv` as `imu_gap_steps`).
- `AOptimizer::addIMUResiduals` and `VIInit` build no preintegration factor across a KF interval
  that contains a gap step (missing IMU data); the bias random-walk factor is kept.
- Non-increasing IMU stamps are dropped by the readers (Issue 7/8); decreasing frame stamps were
  already rejected by `processIMU`.
- `IMU::_rate_hz` gets a default (the two-argument constructor left it uninitialized).

**Evidence:** new `ImuTest.imuDataGapsAreIntegratedAndNotTurnedIntoFactors`: a chain with a 2 s hole
integrates 40·dt + 2 s and counts 1 gap step; a window optimization over it inserts 0 IMU factors
(1 without the hole). Passes. No IMU gaps in room1 (0 in `vio_diag.csv`), so no trajectory check.

### Issue 20, config-validation part (done 2026-10-02)

**Changed:** new `isae::validateConfig(cfg, ncam, has_imu, warnings)` (`slamParameters.h/.cpp`),
called by `SLAMParameters` right after loading the dataset. Errors (printed, then
`std::runtime_error`): unknown `slam_mode` or `optimizer`; VIO mode without `imu` block; stereo
mode with < 2 cameras / mode without camera; `optimizer: Numeric` with `marginalization: 1` or with
VIO `estimate_td: 1` (Numeric inherits do-nothing `marginalize()` / `localMapVIOptimizationTd()`).
Warnings: `sparsification` without `marginalization`; `estimate_td` in VO modes; `monovio` with
`marginalization: 1` (Issue 16).

**Evidence:** new `ConfigTest.unsupportedOptionCombinationsAreReported` (valid stereo VIO, each error,
each warning). Passes.

**To revisit:** remove the Numeric errors if Issue 20 (rest) implements those paths; remove the
`monovio` warning when Issue 16 is done.

### Issue 2 — `VIInit` bias application and prior (done 2026-10-02)

**Changed (`AOptimizer::VIInit`):** the solved Δba and Δbg are applied once each (Δba was applied
twice, Δbg never), and every KF whose previous KF is in the window gets `biasDeltaCorrection(Δba,
Δbg)` (current in-place policy; revisited with Issue 4). The bias prior is replaced: it was
σ = √(dt in **ns**)·random_walk with a ×1000 information factor (effectively ~30× *weaker* than a
random-walk prior); it is now an explicit prior on the change from the static estimate, as in
ORB-SLAM3's inertial-only initialization: σ_Δbg = 0.1 rad/s (weak: the gyro bias is well observed
through rotation), σ_Δba = 0.01 m/s² (the accelerometer bias is barely observable over a short
window). Just converting `dt` to seconds while keeping ×1000 would have frozen the biases (σ ≈ 2e-6).

**Evidence:** new `ImuTest.viInitRecoversGyroBiasAndVelocities`: simulated rotating, accelerating
motion (IMU samples that integrate exactly to it, ZOH), 10 KFs at their true poses, true bg =
(0.02, −0.015, 0.01) rad/s, preintegration started with zero bias → after `VIInit` every KF has
|bg − bg_true| < 2e-3, |ba| < 0.02, velocities within 5 cm/s, and the corrected ΔR match the true
relative rotations within 2e-3 rad. Passes. (Old code: bg stays 0, 0.027 rad/s off.)

### Issue 3 — `IMUFactorInit` position residual and Jacobians (done 2026-10-02)

**Changed (`residuals.hpp`):** `r_p = R_fi_w·(s·Δp_w) − R_fi_w·R_w_i·(v_i·dt + ½g·dt²) − Δp` (this
repository's convention: velocities and gravity in the gravity-aligned frame, positions in the
visual world; the map is then moved with `T_f_I = (R_f_w R_w_i, s t_f_w)`); before, `R_w_i` also
rotated `s·Δp_w`. Gravity Jacobian of `r_p` now `R_fi_w R_w_i [v_i dt + ½g dt²]× Jr`; scale
Jacobian `exp(λ)·R_fi_w·Δp_w` (the `exp(λ)` was missing). `IMUFactorInitBis` (unused) documented as
not validated.

**Evidence:** `ImuTest.imuFactorInitJacobiansAtNonUnitScale` now passes (λ = ln 3, random point).
New model test `ImuTest.viInitModelMatchesTiltedScaledWorld`: simulated in the gravity-aligned
frame, KF poses expressed in a world tilted by (0.08, −0.05) rad and scaled by 1/2.5 → the factor
is ~0 (< 0.1 whitened) at the true tilt/scale/velocities/biases for every KF pair, and `VIInit`
with scale estimation recovers s = 2.5 (±1 %), the tilt (< 0.01 rad) and the true positions (< 1 cm).

### Issue 9 — initialization handoff (done 2026-10-02)

**Changed:** mono VIO scales the translational part of `_6d_velocity` by the scale returned by
`VIInit` (the motion model is a relative twist: the gravity rotation cancels, the old scale does not).

**Evidence / reasoning:** with ordered input both pipelines finish initialization right after
inserting the 10th KF, so `_last_IMU` is that KF's IMU, which `VIInit` transforms; `processIMU`
refreshes the previous sample's cached pose from its frame. Out-of-order input can no longer reach
the pipelines (Issues 7/8: the merger keeps the frame queue in time order, tested). No dedicated
pipeline-level test (would need a full SLAM run); covered by the room1 runs.

### Issue 18, initialization part (done 2026-10-02)

**Changed:**
- New `staticImuInitialization()` (`IMU.h/.cpp`), used by both VIO inits: gravity alignment with
  `Quaternion::FromTwoVectors` — the previous Rodrigues formula normalized the cross product of the
  measured and world up vectors, which is **NaN for a level start** (acc exactly along +z, e.g.
  simulated IMUs) or an upside-down one. Biases are guessed (bg = mean rate, ba = magnitude error
  along gravity) only if the first window is static (mean rate < 0.1 rad/s, gyro spread < 0.05
  rad/s, accel rms spread < 0.2 m/s²), otherwise left at zero for `VIInit`.
- `VIInit` returns −1 without applying anything when the solution is not usable.
- New `SLAMCore::inertialInitAccepted(scale)`: scale > 0 and finite, finite states, |ba| < 2 m/s²,
  |bg| < 0.5 rad/s; otherwise both pipelines reset and restart the initialization.

**Evidence:** new `ImuInitTest.staticInitializationAlignsGravityAndGuessesBiases` (level,
upside-down, tilted with an accelerometer bias, moving). Passes.

- `IMU::hasValidCovariance()` (finite, Cholesky succeeds): `addIMUResiduals` and `VIInit` build no
  preintegration factor on an unusable covariance (e.g. a never-processed IMU). Tested in
  `imuDataGapsAreIntegratedAndNotTurnedIntoFactors` (valid chain vs zero covariance).

**Not done:** no "enough motion" criterion for mono scale observability (would need a validated
excitation measure; not invented here).

### Found in the loop: timer data race — the cause of the segfaults (done 2026-10-02)

**Diagnosis:** the two crashes (exit 139) were general-protection faults, logged by the kernel at
`libisae_slam.so` + 0x35c40b = `isae::timer::tic()` (`issue07` snapshot) and inside libc (`phase2`).
`timer::tic/toc` pushed/popped one global `std::stack` from every thread; the offline reader thread
calls `createImageSensors`, which called `tic()` (never matched by a `toc`) while the SLAM thread
timed its own work. gdb never reproduced it (slower timing).

**Changed:** `src/utilities/timer.cpp`: the stack is `thread_local`; the unmatched `tic()` in
`ADataProvider::createImageSensors` removed (it also grew the reader's stack by one entry per frame
and, with the shared stack, corrupted the SLAM thread's timings).

**Evidence:** new `TimerTest.ticTocFromSeveralThreads` (4 threads × 200k nested tic/toc) passes 3/3;
the same loop compiled against the old shared stack crashes 3/3 (segfault, `double free or
corruption`, `free(): invalid size`).

### Phase 3 quick check (label `phase3`, room1, cumulative through the init fixes; before the timer fix)

| mode | ATE (m) | RPE 1 s (m) | coverage | crashes | scale |
|---|---|---|---|---|---|
| bimonovio | 0.265 ± 0.086 | 0.031 | 0.98 | 0 | — |
| monovio | 0.322 ± 0.078 | 0.123 | 0.975 | 0 | 0.893 ± 0.048 |

End-of-run biases: stereo ba ≈ 0.2-0.4 m/s² (still far from Basalt's < 0.01), bg ≈ 0.001-0.005
rad/s; mono ba ≈ 0.08-0.1 m/s², bg ≈ 0.002-0.003 rad/s.

### Issue 12 — `IMUFactor` pose-j rotation Jacobian (done 2026-10-02)

**Changed:** `residuals.hpp` — the pose-j rotation block is `−Jr(r)⁻¹·T_fj_w.rotation()·Jr(w)` again
(`T_fj_w` includes the increment: `R_fj0·Exp(w)`); commit `291a492` had changed it to
`R_fj0·Exp(w)ᵀ`, which is only right at w = 0.

**Evidence:** `ImuTest.imuFactorJacobiansAwayFromZero` and the `IMUFactor` part of
`predictionPositionVelocity` now pass (before: block 1 cols 0-2 off by ~8 % of the column scale).

### Issue 4 — bias linearization bookkeeping (done 2026-10-02)

**Policy:** each preintegration keeps the bias it was integrated with (`b_lin`): the bias of its KF
when it started, then copied along the IMU chain (the back end may update the KF bias meanwhile).
All terms of the recursion use `b_lin`. Factors correct the deltas to first order with
`δb = (bias of KF i) + increment − b_lin`; when that bias moves beyond the first-order range
(VINS-Mono thresholds 0.1 m/s², 0.01 rad/s) the preintegration is integrated again from its raw
measurements. No in-place correction any more.

**Changed:**
- `IMU`: preintegration recursion factored into one static `preintegrate()` on a `Preint` state,
  used by `processIMU` and by the new `repropagate(ba_lin, bg_lin)` (walks the `_last_IMU` chain back
  to the KF's IMU and integrates again). Starting at a KF = starting from the zero state (the old
  restart branch used the previous interval's ΔR in its noise input; harmless for isotropic noise,
  now exact). Before, increments used the chain bias but the right Jacobian and the acceleration
  coupling used the KF's *current* bias, mixing linearization points.
- `IMUFactor`, `IMUFactorInit`: `δb = b_i + d − b_lin(j)` (Jacobians unchanged).
- Removed: `IMU::biasDeltaCorrection` and its calls (after `localMapVIOptimization`, both
  `localMapVIOptimizationTd`, `VIInit`); `IMU::updateBiases` and its calls in both back ends.
- New `AOptimizer::repropagateIfNeeded()` after every window optimization and after `VIInit`.
- Prediction (dead reckoning) uses `b_lin` too; the time-offset path uses the same factor.

**Evidence:**
- New `ImuTest.imuFactorAccountsForBiasChangesAfterIntegration`: KF i's bias moved by (4, 3, 2) mg-ish
  after integration → the factor at zero increment matches a factor on a chain integrated at the new
  bias to < 2 % of the error made when the change is ignored (the old behaviour). Passes.
- New `ImuTest.repropagateEqualsFreshIntegration`: re-integration with a large bias change equals a
  fresh integration (deltas, covariance, Jacobians, duration) to 1e-12. Passes.
- `biasJacobiansMatchReintegration` still passes; `viInitRecoversGyroBiasAndVelocities` passes with
  re-integration instead of in-place correction.
- Thread safety: getters/setters lock; re-integration only touches KF IMUs of the window.

**Note (not changed):** every IMU keeps a strong pointer to the previous one, so the whole IMU history
stays in memory for the whole run (pre-existing; re-integration relies on the chain within a window).
Candidate improvement: cut the chain when a KF leaves the window.

### Phase 4 quick check (label `phase4`, through Issue 4; before Issue 5)

| mode | ATE (m) | RPE 1 s (m) | coverage | crashes | scale |
|---|---|---|---|---|---|
| bimonovio | 0.290 ± 0.019 | 0.037 | 0.98 | 0 | — |
| monovio | 0.483 ± 0.190 | 0.260 | 0.975 | 0 | 0.764 ± 0.203 |

### Issue 5 — low-parallax KF drop and the marginalization factor check (done 2026-10-02)

**Changed:**
- Both back ends: when the previous KF is dropped for low parallax, the new KF's preintegration is
  re-pointed to the KF before it and integrated again over the longer interval
  (`SLAMCore::bridgePreintegration` → `IMU::repropagate`; the measurement chain runs through the
  dropped KF, so no IMU information is lost). Before, the IMU link was cut (`setLastKF(nullptr)`).
- The front end hands the parallax to the back end with the KF (`_parallax_to_optim`), instead of
  the back end reading `_parallax` while the front end overwrites it.
- New `AOptimizer::imuFactorUsable(fi, fj)` (fj's preintegration starts at fi, ≤ 1 s, no gap, usable
  covariance) used by `addIMUResiduals`, `VIInit` and the 4 marginalization sites
  (`marginalize` / `marginalizeRelative` of both analytic optimizers), which built `IMUFactor(frame0,
  frame1)` without checking that frame1's preintegration started at frame0. The bias random-walk
  factor is kept in all cases.

**Evidence:** new `ImuTest.droppingAKeyframeKeepsItsImuInformation`: 3 KFs (0, 60, 120 samples),
the last re-pointed to the first and re-integrated = the preintegration of a 2-KF chain (0, 120) to
1e-12 (deltas, covariance, duration); `imuFactorUsable` accepts (KF0, KF2) and rejects the stale
(KF1, KF2). Passes.

### Issue 16 — mono VIO marginalization (done 2026-10-02)

**Changed:**
- `Marginalization::preMarginalize`: a landmark needs observations from two cameras of the
  marginalized frame only if that frame has two cameras; with one camera, landmarks seen by other
  frames are kept (rank-deficient information is handled by the pseudo-inverse in the Schur complement
  and the rank-revealing decomposition). Before, every mono landmark was discarded (`setMarg`).
- Mono VIO back end calls `marginalize()` when `marginalization: 1` (it never did); the
  `validateConfig` warning for `monovio` + marginalization removed.

**Evidence:** new `MarginalizationMonoTest.preMargKeepsSingleViewLandmarks` (2 shared landmarks kept,
the frame-0-only one marginalized; before: 0 kept). Stereo marginalization tests unchanged. End-to-end
check with `marginalization: 1`: see Phase 5 below.

**Not done:** the sparse VIO prior for mono (pose-to-landmark factors inverse a 3×3 block that may be
rank-deficient for single-view landmarks): not validated.

### Phase 5 quick check (label `phase5`, through Issues 5 and 16, default config)

| mode | ATE (m) | RPE 1 s (m) | coverage | crashes | scale |
|---|---|---|---|---|---|
| bimonovio | 0.275 ± 0.022 | 0.026 | 0.98 | 0 | — |
| monovio | 0.314 ± 0.011 | 0.110 | 0.962 | 0 | 0.903 ± 0.015 |

### Issue 13 — `IMUPriordx` velocity and bias Jacobians not whitened (done 2026-10-02)

**Changed:** `residuals.hpp` — the velocity, accelerometer-bias and gyro-bias Jacobian blocks are
multiplied by the (dense) square-root information like the residual and the pose block.

**Evidence:** `ImuTest.imuPriorJacobiansWithDenseWhitening` passes (before: blocks 1-3 wrong by ~80 %).

### Issue 14 — sparse-prior recovery Jacobians (done 2026-10-02)

**Changed (`marginalization.cpp`):** new `Marginalization::poseToLandmarkJacobian(T, p)` =
`[−R[p]×, R | R]` (used the frame translation instead of the landmark position) and
`absolutePriorJacobian(T)` = pose block `[R, 0; R[Rᵀt]×, R]`, identity for velocity and biases (was
`[R, R; 0, R]`), used by `sparsifyVIO`.

**Evidence:** new `MarginalizationJacobianTest.recoveryJacobiansMatchTheFactors`: both equal the
Jacobians that `PoseToLandmarkFactor` and `IMUPriordx` themselves return at the marginalization
point (1e-12). Passes.

**Not done:** dense-vs-sparse prior comparison (KLD) on a synthetic VIO problem.

### Issue 15 — sparse-prior gating, and two crashes in the same code (done 2026-10-02)

**Changed:** the sparse-prior part of `addMarginalizationResiduals` (duplicated in both analytic
optimizers) replaced by one `AOptimizer::addSparsePriorResiduals`:
- gating on the retained variables: kept point landmarks > 0; VIO case only with a frame to keep
  that has an IMU and is in the problem; VO case only with an anchor landmark. Before: `Analytic`
  required > 1 landmark *types* (a point-only map never got its sparse prior), `AngularAnalytic`
  returned early whenever the frame to keep was absent — i.e. **always in VO** (no VO sparse prior);
- landmark blocks missing from the problem are added before use (the Angular VO chain used
  `.at()` before adding them, and its "missing kp1" branch inserted `lmk_k` instead of `lmk_kp1`).
- **Crash fixed:** `BundleAdjustmentCERESAnalytic::marginalize` dereferenced `_frame_to_keep`
  (null in VO) when choosing sparsifyVIO/VO → `optimizer: Analytic` + `sparsification: 1` + any VO
  mode segfaulted at the first marginalization.

**Evidence:** new `MarginalizationTest.sparsePriorForPointOnlyMap` (Analytic, stereo VO, point-only
map: marginalize with sparsification, then the sparse prior adds residual blocks). It segfaulted
before the null check; passes now.

**Test suite now:** 55 pass; the 5 failures left are the pre-existing non-IMU ones
(`LineFeatureDetection`, `LineFeatureMatching`, `ESKFLmkTest`, `RelativePose6DResidual`,
`angularTest`).

### Issue 19 — online camera-IMU time offset (`estimate_td: 1`) (done 2026-10-02)

**Changed:**
- `AngularErrCeres_pointxd_td` (AngularAnalytic) and `ReprojectionErrCeres_pointxd_dx_td` (Analytic)
  rewritten on one model: the image was taken at t + td, `T_w_f(t+td) = T_w_f(t)·[Exp(ω·td) | v_f·td]`
  with the bias-corrected rate ω = gyr − bg and the velocity in the frame axes v_f = R_f_w·v. Before,
  the IMU velocity (world axes) was used as a frame-axes translation, and the raw gyro was used.
  Jacobians: chain through the point in the frame at t + td, `dy/dtd = −R_tdᵀ(ω × (x_f − v_f td) + v_f)`;
  the reprojection version uses the camera's landmark Jacobian for all blocks.
- Offline reader: raw IMU samples are moved to the merger only when needed, with the **current**
  `dt_imu_cam`, so the online estimate reaches the data read afterwards (before: shifted once at load
  time, and the reader had queued the whole sequence anyway). The reader now stays at most 1000
  frames (~5 s) ahead of the SLAM, instead of holding the whole dataset in memory.

**Evidence:**
- New `ResidualTest.timeOffsetResidualJacobians` (both residuals, non-identity rotation, at zero and at
  random points). Before: the td column was off by 0.3 % (angular) and 65 % (reprojection) of its
  scale. Passes.
- New `DataProviderTest.onlineTimeOffsetAppliesToLaterImu`: a change of `dt_imu_cam` while reading
  applies to the IMU read afterwards. Passes.

**Limitation (unchanged design):** each window estimate is accumulated into `dt_imu_cam`; KFs already
in the window keep the timing they were built with, so the estimate mixes timings until the offset
settles.

### Issue 20, rest — Numeric optimizer (2026-10-02)

Not implemented: `BundleAdjustmentCERESNumeric` still has no `marginalize` / `marginalizeRelative` /
`localMapVIOptimizationTd`. Those combinations (`optimizer: Numeric` with `marginalization: 1`, or with
VIO `estimate_td: 1`) are refused at load with a clear error (Issue 20, validation part), so they can
no longer silently skip work. Implementing them would mean numeric-Jacobian versions of the
marginalization blocks; out of proportion for this loop.

### Minor items (done 2026-10-02)

- `Prior1D` (`residuals.hpp`): the constructor parameter `_prior` shadowed the member, which stayed
  uninitialized → renamed and stored (used by the non-overlapping-FOV scale prior).
- `shouldInsertKeyframe` (`slamCore.cpp`): parallax averaged over `n_matches` (it already counts the
  landmark matches; landmark matches were counted twice in the denominator, underestimating parallax).
  This changes KF voting in all modes (VO included) and the low-parallax decision. The mono branch of the
  "few tracks" vote, `n_matches_lmk + n_matches < min_lmk_number`, double counts the same way; it is
  vision policy, left unchanged and reported.
- Mono VIO init: `_last_IMU = imu_kf` was set before `imu_kf->setLastIMU(_last_IMU)`, so the first KF's
  IMU pointed to itself (a reference cycle, never freed); now it points to the last measurement.

### Issue 19, end-to-end check — failed, then fixed (2026-10-02)

**First check** (`issue19_td0`, `issue19_td10`; room1, `estimate_td: 1`, 3 runs):

| start offset | mode | final `dt_imu_cam` (3 runs) | ATE | coverage | resets |
|---|---|---|---|---|---|
| 0 ms (true ≈ 0) | bimonovio | −32 ms, −32 ms, +3 ms | 0.213 ± 0.085 | **0.06** | 2.3 |
| 0 ms | monovio | +1.77 s, −56 ms, +5.38 s | (2/3 no traj) | 0.007 | 0 |
| +10 ms (wrong) | bimonovio | −30 ms, −63 ms, −1 ms | 0.511 ± 0.516 | 0.39 | 0.7 |
| +10 ms | monovio | +0.91 s, +1.09 s, +2.35 s | no traj | 0 | 0 |

So online estimation made VIO fail, including from the true offset.

**Cause:** each window optimization estimated an offset *increment* from 0 and added it to
`dt_imu_cam` (≈1,400 times per run). Consecutive windows share most of their KFs, built with the older
offset, so they all saw the same residual error and each added it again (overshoot past zero), and
the per-window noise accumulated as a random walk (mono: seconds).

**Fix (VINS-Mono style):**
- every frame stores the offset its IMU data was read with (`Frame::getTimeOffset()`, set by
  `ADataProvider::addFrameToTheQueue`);
- the time-offset residuals take the **absolute** offset as parameter and shift each frame by
  `td − td_built` (`AngularErrCeres_pointxd_td`, `ReprojectionErrCeres_pointxd_dx_td`, new
  `td_built` argument, passed by both `localMapVIOptimizationTd`);
- the window starts from the current `dt_imu_cam` and its estimate **replaces** it; a non-finite
  estimate or a jump > 50 ms in one window is rejected and reported.

**Evidence:** `ResidualTest.timeOffsetResidualJacobians` extended: with `td_built = a`, parameter
`a + x` gives exactly the residual of `td_built = 0`, parameter `x` (both residuals). Passes.
End-to-end re-check: labels `issue19b_td0`, `issue19b_td10` (running).

Note: only `estimate_td: 1` runs are affected by this fix; the `final` matrix (default config,
`estimate_td: 0`) remains valid. The option-matrix `td` variant ran on the pre-fix snapshot and will be
re-run.

**End-to-end re-check after the fix** (`issue19b_td0`, `issue19b_td10`; room1, `estimate_td: 1`, 3 runs):

| start offset | mode | offset over the run (3 runs) | rejected updates | ATE (m) | RPE 1 s (m) | scale |
|---|---|---|---|---|---|---|
| 0 ms | bimonovio | → 2.5 / 2.3 / 2.4 ms | 0 | 0.226 ± 0.059 | 0.019 | — |
| 0 ms | monovio | → 2.8 / 2.9 / 2.7 ms | 0 | 0.198 ± 0.005 | 0.063 | 0.950 |
| +10 ms | bimonovio | 10 → 2.2 / 2.4 / 2.4 ms | 0 | 0.131 ± 0.006 | 0.018 | — |
| +10 ms | monovio | 6 → 3.1 / 3.4 / 2.9 ms | 0 | 0.225 ± 0.044 | 0.068 | 0.945 |

The estimate now converges to the same value from both starts, in both modes (before: drift to −32 ms
or seconds), and VIO accuracy improves over `estimate_td: 0` (Phase 5: stereo 0.275 m / RPE 0.026,
mono 0.314 m / scale 0.90). The ~2.5 ms offset is not in Basalt's TUM-VI calibration
(`cam_time_offset_ns: 0`); it may be a real exposure-timing offset or a small model artifact — not
verified independently.

## Phase 8 — final matrix and option matrix

### Final matrix (label `final`, default config, all fixes through Issue 19; 3 runs per cell)

| seq | mode | ate_rmse | rpe_t_1s | rpe_r_1s_deg | scale | end_drift | coverage | resets |
|---|---|---|---|---|---|---|---|---|
| magistrale2 | bimono | 4.632 ± 3.171 | 0.040 ± 0.004 | 1.416 ± 0.080 | 1.000 | 14.644 ± 0.954 | 0.751 ± 0.345 | 0 |
| magistrale2 | bimonovio | 7.284 ± 0.144 | 0.028 ± 0.001 | 0.895 ± 0.019 | 1.000 | 15.967 ± 0.373 | 0.994 ± 0.000 | 0 |
| magistrale2 | mono | 0.284 ± 0.077 | 0.409 ± 0.061 | 15.046 ± 9.757 | 1.228 ± 0.851 | — | 0.007 ± 0.002 | 0 |
| magistrale2 | monovio | 0.943 ± 0.004 | 0.748 ± 0.002 | 1.289 ± 0.019 | 0.002 ± 0.001 | 84.276 ± 21.342 | 0.993 ± 0.000 | 0 |
| room1 | bimono | 0.154 ± 0.032 | 0.024 ± 0.000 | 0.924 ± 0.015 | 1.000 | — | 0.982 ± 0.000 | 0 |
| room1 | bimonovio | 0.273 ± 0.043 | 0.026 ± 0.001 | 0.896 ± 0.015 | 1.000 | — | 0.982 ± 0.000 | 0 |
| room1 | mono | 0.147 ± 0.078 | 0.211 ± 0.108 | 11.747 ± 0.182 | 0.592 ± 0.053 | — | 0.026 ± 0.000 | 0 |
| room1 | monovio | 0.253 ± 0.022 | 0.076 ± 0.008 | 0.964 ± 0.023 | 0.944 ± 0.009 | — | 0.976 ± 0.001 | 0 |

vs. baseline: room1 stereo VIO 0.824 → 0.273 m ATE, RPE 0.137 → 0.026; mono VIO from mostly no
trajectory to 0.253 m at 97.6 % coverage, scale 0.94. magistrale2 stereo VIO coverage 0.19 → 0.994,
resets 4.3 → 0, RPE 1.03 → 0.028 (better than VO, 0.040); its end drift (16.0 m) is close to VO's
(14.6 m). Mono VIO on magistrale2 runs through but its Sim3 alignment degenerates (scale 0.002, 84 m
end drift): not solved. One magistrale2 VO run re-initialized; the evaluator's `resets` counter only
counts VIO resets and SaDVIO truncates `results.csv` on re-init, hence VO coverage 0.75 ± 0.35.

### Option matrix, round 1 (label `opt`, room1, 1 run, one option changed from the default)

Failures found (all fixed below): `analytic` and `numeric` stereo VIO 22 m and 521 m ATE (no visual
factors, DS stub); `marg` stereo VIO, mono VO and `marg_sparse` mono VO / mono VIO crashed;
`analytic_marg_sparse` mono VIO crashed; `mt` crashed in both stereo modes; `td` ran on the pre-fix
Issue 19 code (re-run as `td_fixed`: stereo VIO 0.108 m, mono VIO 0.171 m).

### Fixes from the option matrix (2026-10-02)

- **Double-sphere projection with Jacobians** (`DoubleSphere.cpp`): was a stub returning `false`, so the
  `Analytic`/`Numeric` optimizers built no visual factor with this camera. Implemented;
  `ResidualTest.reprojTestDoubleSphere`.
- **Duplicate landmarks in a frame** (`frame.h`, `marginalization.cpp`): `Frame::addLandmark` and
  `preMarginalize` deduplicate (Ceres aborted on duplicate parameter blocks with stereo VIO + marg).
- **Mono VIO init KF inserted twice** (`slamMonoVIO.cpp`): init added the KF to the window and then sent
  it to the back end, which added it again; with marginalization `marginalize(f, f)` indexed outside the
  prior (`_map_frame_idx = {f → −15}`, gdb). The init KF is no longer re-sent; `LocalMap::addFrame` ignores
  a frame already in the window; both analytic `marginalize` refuse `frame0 == frame1`. Pre-existing in
  the first commit.
- **Frame-1 pose prior in the AngularAnalytic marginalization** (`AngularAdjustmentCERESAnalytic.cpp`):
  frame 1 stays in the window, which already applies its prior, so marginalizing it counted it twice; in
  VO frame 1 is not a marginalization variable at all → `_map_frame_idx.at` threw (mono VO after a
  re-init, whose first KF has a prior). Removed; `BundleAdjustmentCERESAnalytic` never had it.
- **`multithreading: 1`** (`slamCore.*`, all `slam*.cpp`): front end and back end modified landmarks
  concurrently (gdb: `ALandmark::removeExpiredFeatures` in the back end while the front end freed
  features → `free(): corrupted unsorted chunks`), and `_frame_to_optim` was handed over with a spin loop
  on a plain `shared_ptr`. Now a step mutex: the front end holds it for a whole frame, the back end for a
  whole KF; the front end releases it while waiting for data (`nextFrame()`) or for the back end
  (`waitBackEnd()`, condition variable); a re-init / reset first waits for the back end. Single-thread
  mode is unchanged (no lock). The two threads no longer overlap their map work, so the speed-up is
  limited to waiting for data.
- **`dt_imu_cam` atomic** (`IMU.h`): written by the back end (`estimate_td`), read by the data thread
  (offline reader and ROS callbacks) — a data race in both threading modes.

Debug evidence: `runs/dbg4_b` (mono VO + marg), `runs/dbg4_c` (mono VIO + marg + sparse) and
`runs/dbg5_*` (multithreading) ran to the end of room1 under gdb with the debug build
(`_GLIBCXX_ASSERTIONS`) after the fixes.

### Option matrix, round 2 (label `opt2`, snapshot `opt2`: all fixes above, before the ESKF change; room1, 1 run)

ATE (m) / RPE 1 s (m) / coverage; mono: Sim3 scale in brackets.

| option | bimono | bimonovio | mono | monovio |
|---|---|---|---|---|
| default | 0.174 / 0.025 / 0.98 | 0.281 / 0.027 / 0.98 | 0.755 / 0.752 / 0.79 (0.35) | 0.210 / 0.060 / 0.98 (0.97) |
| optimizer Analytic | 0.122 / 0.022 / 0.98 | 0.232 / 0.022 / 0.98 | 0.909 / 0.889 / 0.79 (0.24) | 0.135 / 0.051 / 0.94 (1.00) |
| optimizer Numeric | 0.127 / 0.023 / 0.98 | 0.196 / 0.023 / 0.98 | 0.221 / 0.204 / 0.46 (0.90) | 0.211 / 0.062 / 0.98 (0.97) |
| marginalization | 0.187 / 0.025 / 0.98 | 0.182 / 0.022 / 0.98 | 0.177 / 0.253 / 0.03 (0.58) | 0.322 / 0.103 / 0.98 (0.91) |
| marg + sparsification | 0.143 / 0.025 / 0.98 | 0.184 / 0.026 / 0.98 | 0.125 / 0.201 / 0.02 (0.76) | 0.302 / 0.099 / 0.95 (0.92) |
| Analytic + marg + sparse | — (late VO re-init, see below) | 0.149 / 0.020 / 0.98 | 0.030 / 0.100 / 0.01 (0.22) | 0.254 / 0.067 / 0.98 (0.95) |
| estimate_td | 0.181 / 0.023 / 0.98 | 0.188 / 0.018 / 0.98 | 0.563 / 0.432 / 0.79 (0.25) | 0.280 / 0.092 / 0.98 (0.92) |
| multithreading (1 run) | 0.127 / 0.033 / 0.98 | 0.248 / 0.025 / 0.98 | 0.066 / 0.105 / 0.03 (0.76) | 0.265 / 0.081 / 0.94 (0.94) |
| multithreading (3 more runs) | 0.120–0.137 / 0.025 | 0.238–0.277 / 0.025 | 2× 0.02 cov, 1× 0.79 | 0.217–0.583 / 0.06–0.38 (0.63–0.97) |
| Numeric + marginalization | refused at load (`CONFIG ERROR`) | refused | refused | refused |

- **No crash in any cell** (round 1: 8 crashes). All VIO cells run through the sequence (coverage ≥ 0.94).
- Analytic / Numeric stereo VIO: 22 m → 0.23 m and 521 m → 0.20 m ATE (DS projection).
- Multithreading: 4 runs per mode without a crash (round 1: both stereo modes crashed). One mono VIO
  run of four is clearly worse (scale 0.63, RPE 0.38): thread timing changes which frames are tracked
  while the back end runs; not investigated further.
- Mono VO coverage swings between 0.01 and 0.79 across cells and runs: mono VO re-initializes ~24 times
  on room1 and `results.csv` keeps only the segment after the last re-init. Checked that the new
  duplicate-frame guard never fires in mono VO (instrumented run: 0 hits). Not an IMU issue.
- **Analytic + marg + sparse, stereo VO:** "stall" with an empty evaluation — not a hang. The per-frame
  log covers all 2,819 frames; VO lost tracking and re-initialized 2.1 s before the end, so
  `results.csv` held only the last 9 KFs. Six reruns (release and debug, with and without gdb) ran
  through without a re-init. In round 1 this cell had no back-end visual factors at all (DS stub).

### Remaining checklist items (2026-10-02)

- **Issue 11 follow-up — ESKF covariance conventions** (`ESKFEstimator.*`): the IMU step's covariance
  (error of `dT = T_f1_f2`: `R Exp(δθ)`, `t + δt`) was reused as the covariance of the right perturbation
  of `T_cam2_cam1` in the visual step and returned as the covariance of `dT`. It is now mapped with
  adjoints (`imuToCameraErrorJacobian`, `cameraToFrameErrorJacobian`). The rotation update is one
  Gauss-Newton step of the MAP cost (`updateRotation`: `H = Jl⁻¹(e)`, gain `P Hᵀ(HPHᵀ + C)⁻¹`, posterior
  `(I − KH)P`); before, the gain had no `Hᵀ` and the noise was conjugated by `Jr⁻¹` from a different
  linearization. The finite-difference test also showed `T_cam2_cam1` was initialized as
  `T_cam1_f1 dT⁻¹ T_cam2_f2⁻¹`, right only when both extrinsics are equal (always true in the pipeline,
  which uses camera 0 for both); now `T_cam2_f2 dT⁻¹ T_cam1_f1⁻¹`. Heuristic tuning unchanged. `covdT` is
  only logged (`cov_mat.csv`); `LocalMap::computeRelativePose` uses it but is never called (and orders its
  Jacobian blocks differently — reported, not changed). Tests: `ESKFCovarianceTest.errorJacobiansMatch
  FiniteDifferences`, `ESKFCovarianceTest.rotationUpdateIsOneGaussNewtonStep`. This changes the ESKF
  weighting in VO too (its identity prior is now carried through the change of parameterization):
  re-checked by the `final2` matrix.
- **Issue 14 — dense vs sparse prior**: `MarginalizationSparseKLDTest.recoversADensePriorWithTheSparse
  Structure` (KLD ≈ 0) and `.sparsePriorIsTheKLDMinimum` (any factor's information × 0.95 or × 1.05
  raises the KLD). Mutation check: multiplying the absolute factor's information by 1.1 makes both fail
  (the second only after tightening the steps from ±20 % to ±5 %).
- **Issue 20 — smoke test per combination**: `ImuTest.everySupportedOptionCombinationInsertsItsFactors`
  builds a synthetic stereo/mono VIO window (simulated IMU chain + exact projections) and runs every
  supported optimizer × marginalization × sparsification × estimate_td: IMU factors = KFs − 1, > 100
  visual factors, prior factors > 0 after marginalizing, true states recovered (< 2 mm / 2 mrad),
  time offset ≈ 0. `VIOptimStats` now also counts prior and other factors (also in `vio_diag.csv`, last
  two columns). The test first caught its own scene bug (landmarks pre-marked in-map were never pushed to
  the window: 6 non-IMU factors), so the visual-factor bound was raised from > 0 to > 100.
- **Issue 17 — counterexamples as regression tests**: already present
  (`ESKFVelocityTest.updateVelocityFrames`, `ImuTest.imuFactorJacobiansAwayFromZero`,
  `MarginalizationJacobianTest.recoveryJacobiansMatchTheFactors`).
- **Issue 19 — consistency of the accumulated offset**: covered by the absolute-offset fix above.

Unit tests: 63 pass; the same 5 pre-existing non-IMU failures (`LineFeatureTest.LineFeatureDetection`,
`.LineFeatureMatching`, `ResidualTest.angularTest`, `.RelativePose6DResidual`, `.ESKFLmkTest`).

### Final matrix with every fix (label `final2`, snapshot `final2`, default config, 3 runs per cell)

| seq | mode | ate_rmse | rpe_t_1s | rpe_r_1s_deg | scale | end_drift | coverage | resets |
|---|---|---|---|---|---|---|---|---|
| magistrale2 | bimono | 6.960 ± 1.047 | 0.037 ± 0.003 | 1.394 ± 0.082 | 1.000 | 14.825 ± 2.212 | 0.995 | 0 |
| magistrale2 | bimonovio | 6.676 ± 0.702 | 0.031 ± 0.003 | 0.923 ± 0.022 | 1.000 | 14.641 ± 1.577 | 0.994 | 0 |
| magistrale2 | mono | 0.156 ± 0.015 | 0.394 ± 0.022 | 37.535 ± 4.083 | 0.681 ± 0.151 | — | 0.006 | 0 |
| magistrale2 | monovio | 0.706 ± 0.322 | 0.533 ± 0.299 | 1.213 ± 0.267 | 0.301 ± 0.425 | 106.4 ± 10.8 | 0.741 ± 0.357 | 0 |
| room1 | bimono | 0.145 ± 0.008 | 0.025 ± 0.000 | 0.911 ± 0.012 | 1.000 | — | 0.981 | 0 |
| room1 | bimonovio | 0.265 ± 0.015 | 0.026 ± 0.002 | 0.899 ± 0.015 | 1.000 | — | 0.982 | 0 |
| room1 | mono | 0.457 ± 0.276 | 0.416 ± 0.239 | 8.002 ± 3.402 | 0.416 ± 0.136 | — | 0.538 ± 0.361 | 0 |
| room1 | monovio | 0.333 ± 0.064 | 0.119 ± 0.038 | 0.946 ± 0.037 | 0.893 ± 0.043 | — | 0.963 ± 0.017 | 0 |

vs `final`: stereo VIO unchanged on room1 (0.265 vs 0.273) and slightly better on magistrale2 (end
drift 14.6 vs 16.0 m, ATE 6.7 vs 7.3; RPE 0.031 vs 0.028). Mono VIO room1 worse (0.333 vs 0.253,
RPE 0.119 vs 0.076). magistrale2 mono VIO is degenerate in both (scale ≈ 0.001, 64–117 m end drift); one
`final2` run re-initialized (coverage 0.24, the last segment only), hence 0.74 ± 0.36.

**ESKF ablation** (room1 mono VIO, 5 runs each): `final2` 0.309 ± 0.057 m / RPE 0.105 ± 0.027 / scale
0.910; same binary without the covariance conversion (`abl_noconv`, container-only edit, reverted)
0.253 ± 0.053 / 0.083 ± 0.027 / 0.935. About 1σ: not significant, but the same direction as `final2`.
The ESKF's P is a tuning heuristic (identity prior, inflated IMU covariance), so the conversion is the
consistent definition but not necessarily better tuned. Kept (acceptance rule: correct by test, metrics
recorded honestly). Reverting it is two lines in `ESKFEstimator::estimateTransformBetween`.

**Builds:** `cpp/build` rebuilt with all fixes (FFS on, VDB on); smoke run room1 stereo VIO 0.311 m /
RPE 0.026 (`main_build_check`). ROS colcon build rebuilt (FFS, VDB, LAS2 on): finished, no errors.

## Analysis — why stereo VIO is not better than stereo VO (2026-10-02)

Binary: snapshot `final2`. Tool: `tools/analyse_vio_vs_vo.py` (drift aligned on the first 10 s, yaw/tilt
split in the gravity-aligned mocap frame, VIO velocity and bias vs GT). room1, 3 runs per row.

| room1 config | ATE | drift @40 s | drift @120 s | yaw end | tilt end | Sim3 scale | VIO v err | VIO ‖ba‖ median / std |
|---|---|---|---|---|---|---|---|---|
| VO default | 0.145 | 0.130 | 0.237 | 13.0° | 3.6° | 0.985 | — | — |
| VIO default | 0.265 | 0.108 | 0.566 | 11.2° | 1.0° | 0.947 | 0.121 | 0.179 / 0.205 |
| VIO + marginalization | 0.193 | 0.163 | 0.493 | 8.4° | 0.4° | 0.972 | 0.116 | 0.045 / 0.024 |
| VIO + estimate_td | 0.178 | 0.122 | 0.339 | 9.8° | 1.2° | 0.975 | 0.109 | 0.169 / 0.178 |
| VIO + fixed dt_imu_cam 2.5 ms | 0.209 | 0.099 | 0.475 | 9.0° | 0.5° | 0.968 | 0.097 | 0.155 / 0.173 |
| VIO + marg + estimate_td | **0.071** | 0.047 | 0.134 | 6.3° | 0.6° | 1.002 | 0.096 | 0.049 / 0.037 |
| VIO + marg + fixed 2.5 ms | **0.075** | 0.030 | 0.151 | 4.6° | 0.3° | 1.001 | 0.085 | 0.047 / 0.037 |
| VIO + marg + sparse + estimate_td | **0.063** | 0.069 | 0.144 | 3.3° | 0.5° | 1.004 | 0.069 | 0.047 / 0.035 |
| mono VIO default / + marg + td | 0.333 / **0.175** | | | | | 0.89 / 0.97 | | |

**Findings**
1. The IMU does help where it can: tilt (gravity direction) error 1.0° vs VO 3.6°.
2. Default VIO's velocity (10 % error) and accelerometer bias (0.18 m/s² median, wandering by 0.2;
   the bias random-walk model allows ~0.01 over the run) are poorly estimated. Cause: with
   `marginalization: 0` a dropped KF takes its information with it, and only the oldest KF's *pose* is
   held fixed (`SetParameterBlockConstant` on poses only) — nothing anchors velocity and bias across
   windows. The window spans ~1.2 s (12 KFs at ~0.1 s), too short for the accelerometer bias, so each
   window re-estimates it almost freely and it absorbs errors → late drift (0.57 m at 120 s vs VO 0.24).
3. Marginalization alone pins the bias (median 0.045, std 0.024) but is not enough; with the ~2.5 ms
   camera-IMU offset also modelled (online or fixed — same result) VIO halves VO's ATE (0.07 vs 0.145),
   gives a correct global scale (1.00 vs 0.95) and RPE 1 s 0.014 vs 0.025. With a free-wandering bias the
   offset error was absorbed; once the bias is pinned, the offset has to be modelled.
4. Yaw drifts similarly in all configs (unobservable with an IMU); it dominates the remaining error and the
   magistrale2 end drift.
5. Cost: marginalization raises the back end from ~10 to ~38 ms per KF (room1 wall 25 → 57 s offline,
   still faster than real time); sparsification did not reduce it here.

**magistrale2 (stereo VIO)**: default ‖ba‖ median 0.25–1.3 m/s² (end tilt error 4–8°, VO 9–16°).
With marg + td: RPE 1 s 0.016–0.026 (default 0.031) but 1 of 3 runs diverged; with marg + fixed 2.5 ms
all 3 runs reset 1–2 times. Divergence onset at 342–345 s in 3 of 4 diverging runs (one at 111 s);
images (brightness, texture) and IMU (gyro ≤ 1 rad/s, normal walking accelerations) are unremarkable
there, and the window cost is flat until it jumps (×60) at the event — a sudden failure, not a slow
build-up of prior inconsistency. Not yet explained. Neither the VO nor the VIO window uses a robust loss,
so an outlier marginalized into the prior cannot be down-weighted afterwards (hypothesis; VO +
marginalization also failed late in 2 of ~7 room1 runs, at 135.9 s, a 0.68 m jump).

## Marginalization prior bugs (found while diagnosing the magistrale2 divergence, 2026-10-02)

Both affect every configuration with `marginalization: 1` (VO and VIO, dense and sparse prior). Neither
shows when the prior is built at a converged state, which is why the smoke test (optimize → marginalize →
optimize) passed: the pipeline marginalizes *before* optimizing the window, i.e. away from the optimum.

1. **Wrong sign of the prior residual** (`Marginalization::computeJacobiansAndResiduals`). `b` is assembled
   as `+Jᵀr` (`computeInformationAndGradient`), so the prior `½‖r_p + J_p dx‖²` must have
   `r_p = Λ^{-1/2} Uᵀ b_k`; the code used `−Λ^{-1/2} Uᵀ b_k` (its comment assumed `g = −Jᵀr`). Every prior
   pushed the kept states *away* from what the marginalized measurements say, by their full disagreement.
2. **No linearization point** (`MarginalizationFactor`, sparse `IMUPriordx`). The dense prior's residual
   `r0 + J dx` took `dx` = the window's parameter blocks, which are increments from the state at the start
   of each window, so a prior used again after the states moved (the next marginalization folds it in; a
   low-parallax KF drop re-uses it in a second window) measured them from the wrong origin and re-applied its
   pull. The sparse inertial prior was even centred on the current state each window
   (`IMUPriordx(T, T, v, v, …)`), so it never resisted drift. Now `Marginalization` records the linearization
   point of the kept variables (`storeLinearizationPoint`, called when the prior is computed, copied to
   `_marginalization_last`); `MarginalizationFactor` measures `dx` from it (right perturbations, chain-rule
   Jacobians `Jr⁻¹` / `R_linᵀR_now`); the sparse inertial prior is centred on it (VINS-Mono keeps the
   linearization values the same way).

**Test** `ImuTest.marginalizationPriorKeepsItsLinearizationPoint` (stereo and mono synthetic VIO window,
exact data): marginalize at the perturbed start, optimize; re-optimizing the converged window must not
move it (< 1 mm); a second marginalization must still end at the truth (< 3 mm, < 1 cm/s).
Before: moved 97 mm / 31 mm, ended 16 cm / 32 cm off. Sign fix only: moved 24 / 45 mm (mutation check).
Both fixes: passes; a diagnostic chain of three marginalizations from the perturbed start ends at 1e-7.
Suite: 64 pass, same 5 pre-existing failures.

Lesson for container-only mutations: `sync_to_container.sh` restores files with the host's (older) mtime,
so after a container-only edit + re-sync, `touch` the file *after* the sync or make keeps the mutated objects.

### After the two prior fixes: stereo VIO + dense marginalization broke (2026-10-02)

Re-run with the fixed prior (snapshot `margfix`, room1, 3 runs): mono VIO marg + td 0.175 → 0.132 m,
stereo VIO marg + sparse + td 0.063 → 0.078, VO marg 0.205 (unchanged), but **stereo VIO dense marg
0.19 → 3.2 m and marg + td 0.071 → 3.05 m**, all runs failing from 113–119 s (window cost > 20× median).
The old prior re-centred itself on whatever state it found every window, which hid two other problems:

3. **Front-end structure-only BA moved prior landmarks** (`AOptimizer::landmarkOptimization`, run at every
   new KF on all of its landmarks, poses fixed, prior ignored). Landmarks with `hasPrior()` are now held
   constant there; the window estimates them with their prior. Alone this did not fix the failure
   (`margfix2`: 5.6 m / 2.2 m).
4. **The window held the oldest pose fixed although the prior constrains it.** New per-window diagnostics
   (`vio_diag.csv`: `prior_cost0` and the kept states' offsets from the prior's linearization point) showed
   the offsets ≈ 0 but the prior's own cost growing steadily (median ~12 over 0–50 s → 500 → 1,200 → 3,000 →
   6,000) until the failure: the prior's pull on the fixed pose could never be satisfied and was folded
   into each following prior. With `fixed_frame_number: 0` the prior cost stayed at ~5 and the run was
   clean (ATE 0.187, RPE 0.027). Fix (VINS-Mono style): when a visual-inertial prior constrains a window
   frame, no frame is held fixed (`fixedFramesGivenPrior`); after the optimization the window is moved back
   by the rotation about gravity and the translation that restore the oldest frame's yaw and position
   (`restoreGauge`; these 4 directions are unobservable in VIO, roll/pitch stay optimized). VO is unchanged
   (its prior holds landmarks only). `ImuTest.marginalizationPriorKeepsItsLinearizationPoint` also passes with
   the gauge reset (its absolute-pose checks failed without it: 2.6 / 6.5 cm gauge drift).
   No synthetic reproduction of the cost growth: a pipeline-like synthetic loop (noisy features, add KF →
   marginalize → optimize, 80 steps) stays bounded with and without the fix, so no unit test guards it; the
   evidence is end to end (below).

**End to end with fixes 1–4** (snapshot `margfix3`, 3 runs): room1 stereo VIO marg + sparse + td **0.054 m**,
marg + td 0.080, mono VIO marg + td 0.177, VO marg 0.220; stereo VIO marg (no td) 2 runs 0.24–0.27, 1 run
failed at 81 s. The prior cost stays at ~5–12 for the whole of room1 (before: 12 → 6,000). But
**magistrale2 got worse: 5–18 resets per run** (marg + td and marg + fixed 2.5 ms; before fixes 3–4: 1–2).
Just before the first failure (run that failed at 47 s): a KF at every image (factor_dt 0.05 s, window
≈ 0.55 s), visual factors dropping (1,770 → 830), and the window's start cost and the prior cost both
blowing up within ~0.6 s (prior 13 → 9,000). Likely cause: neither the VIO window nor the marginalization
used a robust loss, so bad associations in hard segments were marginalized into the prior at full weight;
the old prior forgot them every window (accidental robustness), the corrected one keeps them.

5. **Robust visual loss in the VIO window and its marginalization.** Visual factors of the VIO windows
   (`localMapVIOptimization`, both `…Td`) use the front end's Huber loss (`AOptimizer::newVisualLoss`); IMU,
   bias and prior factors stay unrobustified (the `…Td` variants passed `loss_function` to the IMU and prior
   factors — now explicit `nullptr`). `MarginalizationBlockInfo` takes an optional loss and applies Triggs'
   correction (as Ceres and VINS-Mono), and VIO marginalizations use the window's loss for visual factors
   (VO unchanged: no loss in its window or marginalization). Test
   `MarginalizationRobustTest.robustBlockGivesTheRobustGradient` (gradient = ρ′Jᵀr; unchanged in the
   quadratic zone).
6. **One angular noise in AngularAnalytic** (`kAngularSigmaPx = 1.5`): the window used 1.5 px / f, the
   time-offset window and the marginalization 1 px / f, so marginalized information was 2.25× more confident
   than the window's. (The front-end single-frame step and the non-overlapping-FOV relative marginalization
   keep 1 px; they do not feed this prior.)

Note: fix 5 also changes default VIO (no marginalization); re-checked in the `rb_*` batch.

**End to end with fixes 1–6** (snapshot `robust1`, 3 runs): room1 stereo VIO default 0.266 (unchanged), mono
VIO default **0.210** (was 0.333), stereo VIO marg **0.202, no failure** (was 1 failure in 3), marg + td 0.087,
marg + sparse + td **0.052**, mono marg + td 0.180. magistrale2 stereo VIO default ATE 8.5 / end drift 18.6
(final2 6.7 / 14.6; 0 resets). magistrale2 stereo VIO marg + td / marg + fixed 2.5 ms: **4 of 6 runs clean**
(end drift 3.9, 6.3, 11.4, 4.1 m vs ~15 m for VO and default VIO; RPE 1 s 0.014–0.017 vs 0.031), 2 runs reset
(1 and 2 resets, from ~380–400 s).

Before those failures: kept landmarks 30–430 m from the prior's linearization point, prior cost exploding;
in one, many low-parallax KF drops in a row (factor span 0.9 → 1.75 s, IMU factor lost past 1 s) and the kept
velocity 2–5 m/s off its linearization point. Container-only instrumentation (`tempdiag`, 2 magistrale2 runs)
of every prior landmark offset > 5 m: median depth at linearization 25–40 m, **more than half behind the
camera at linearization** (163 / 259 in the clean run, 3,431 / 7,563 in the failing run), all inliers in the
map with 4–6 observations; in the failing run depths diverged up to ~10⁶ m.

7. **Only well-conditioned landmarks are kept in the prior** (`Marginalization::wellConditioned`, called in
   `preMarginalize` for landmarks to keep): every observation's predicted bearing within 2° of the measured
   one, and the observing rays spanning ≥ 1°. Otherwise frame 0's observations of it are dropped, and if it
   carries prior information it is marginalized out of the prior at this step (Schur complement; it stays a
   free window variable). Lonely landmarks (eliminated immediately) are not checked. Test
   `MarginalizationTest.wellConditionedLandmarks`. Suite: 66 pass, same 5 pre-existing failures.

**End to end with fixes 1–7** (snapshot `cond1`, 3 runs per cell) — every run clean, 0 resets:

| config | seq | ATE | RPE 1 s | end drift | resets |
|---|---|---|---|---|---|
| VO default (`final2`) | magistrale2 | 6.96 ± 1.05 | 0.037 | 14.8 | 0 |
| VIO default (`rb_default`) | magistrale2 | 8.46 ± 2.12 | 0.031 | 18.6 | 0 |
| **VIO marg + estimate_td** | magistrale2 | **1.47 ± 0.17** | **0.015** | **3.2 ± 0.4** | 0 |
| VIO marg + fixed 2.5 ms | magistrale2 | 2.24 ± 0.10 | 0.014 | 4.9 ± 0.2 | 0 |
| VO default (`final2`) | room1 | 0.145 | 0.025 | — | 0 |
| VIO default (`rb_default`) | room1 | 0.266 | 0.026 | — | 0 |
| VIO marg | room1 | 0.214 ± 0.002 | 0.021 | — | 0 |
| **VIO marg + estimate_td** | room1 | **0.069 ± 0.004** | 0.014 | — | 0 |
| **VIO marg + sparse + estimate_td** | room1 | **0.055 ± 0.003** | 0.012 | — | 0 |
| mono VIO marg + estimate_td | room1 | 0.194 ± 0.013 | 0.052 | — | 0 |

Prior cost at the window start: median 4–7, p99 8–49 (before fix 7 the failing runs reached 10⁵–10⁹). One
clean run still had one prior landmark 289 m off (max cost 10,961 once) without consequence. Option
matrix round 3 (`opt3`, snapshot `cond1`) running.

### Option matrix, round 3 (label `opt3`, snapshot `cond1`: fixes 1–7; room1, 1 run per cell)

**No crash and no reset in any of the 32 cells**; `Numeric` + marginalization refused at load as designed.
Compared with round 2 (`opt2`, before fixes 1–7), single-run differences are within the run-to-run spread
seen in the 3-run batches (e.g. stereo VO Analytic 0.122 vs 0.219 — a path untouched by these fixes; mono
VIO default 0.210 vs 0.344, whose 3-run spread is 0.21–0.33; mono VO coverage still swings 0.01–0.82 with its
re-inits). Notable: VO + marg 0.187 → 0.194, VO + marg + sparse 0.143 → 0.153 (unchanged within noise);
Analytic + marg + sparse stereo VO now runs through (round 2: late re-init); stereo VIO estimate_td 0.188 →
0.152; mono VIO estimate_td 0.280 → 0.152; mono VIO Analytic + marg + sparse 0.254 → 0.161.

## RealSense D455 trajectories (2026-10-02, user request: test without a camera-IMU calibration)

**Data.** The 9 real trajectories of RealSense_Captures_060526 (2026-05-06; 20–69 s each), from the existing
bag extraction `dsol_calibrated_synced_0726/png/<traj>/realsense_real/` (stereo IR 848×480 at 30 Hz, the
recorded `/camera/camera/imu` at 200 Hz: gyro 200 Hz, accelerometer 100 Hz copied into the gyro stream) and the
OptiTrack CSV. `tools/prepare_realsense.py` builds EuRoC trees in `SaD_VIO_data/realsense/<traj>/mav0/`
(symlinks to the PNGs, real IMU, GT). (The existing photoslam mav0 trees carry an all-zero IMU.)
**Config** `configs/dataset/realsense_d455_vio.yaml`: body = IMU frame; cameras: Kalibr stereo calibration;
camera-IMU extrinsic: nominal D455 factory values (not calibrated; the bags' extrinsics topics are empty);
IMU noise from `kalibr_work/imu.yaml`. **Evaluation**: GT body ≠ IMU frame (uncalibrated), so for `rs_*`
sequences `eval_traj.py` reports position ATE (SE3), Sim3 scale, relative translation error in the aligned world
frame, path ratio; no rotation metrics.

**First matrix** (snapshot `bridge1`, 3 runs, nominal config): stereo VO works on all 9 (median ATE 0.065 m,
0.03–0.17, scale 1.00); stereo VIO fails on 7–9 of 9 (median ATE 17.9 m default, 103 m marg + td; scale
collapsing, resets).

**Diagnosis (13_18_29 unless stated)** — tools `check_cam_imu_rotation.py`, `check_imu_vs_gt.py`,
`synth_imu_from_gt.py`:
- Camera-IMU rotation: 0.6–1.1° off nominal (VO vs gyro, and via GT); correcting it: no change.
- Lever-arm sign flipped: no change. Time offset: ~0 ± 5 ms (online estimate, gyro-VO correlation); synthetic
  IMU shifted by ±15/30 ms: no change.
- IMU vs GT: gyro matches GT angular velocity (corr 0.995–0.998, residual 0.034 rad/s); accelerometer: constant
  bias ~(0.12, 0.16, −0.09) m/s², no delay vs gyro; |a| at rest 9.655 ± 0.008.
- Gravity initialisation is correct (accelerometer mean even when not static).
- Real bug found: during handheld standstills every new KF has low parallax, so the previous KF was dropped and
  its preintegration bridged, past the 1 s factor limit (0.55 → 7.9 s): the newest KFs lost their IMU factor.
  **Fix 8**: a low-parallax KF is only dropped if the bridged factor stays usable (`SLAMCore::canBridge`,
  `AOptimizer::kMaxImuFactorDt`); otherwise the oldest KF is marginalized. Alone it did not make VIO work here.
  TUM-VI re-check (`br_margtd`, 3 runs): room1 0.074 (was 0.069), magistrale2 1.85 / end drift 4.1 (was 1.47 /
  3.2 — about 2σ worse; to watch).
- **IMU synthesised from GT (noise-free)**: VIO tracks the GT speed for ~45 s (e.g. 0.49 vs 0.50 m/s), then
  fails at the final standstill; with the real IMU it fails at ~7 s. Hybrids (real gyro + synthetic accel,
  synthetic gyro + real accel) both fail; interpolating the 100 Hz accelerometer instead of copying: no change.
  → the low-frequency content of both sensors is right; the high-frequency handheld vibration is what VIO
  cannot absorb with the configured noise model.
- Effective noise on quasi-still handheld segments: accel 0.015–0.08 m/s²/√Hz, gyro 0.002–0.016 rad/s/√Hz
  (configured 0.0039 / 0.0002, i.e. 4–80× too small). **Inflated densities make default VIO work**:
  noise4 (0.08 / 0.01): 13_18_29 0.070, 13_36_44 0.647, 13_45_18 0.219; noise5 (0.15 / 0.02): 13_36_44 0.248,
  13_45_18 0.037 (VO 0.054); noise6 (noise5 + 10× bias random walks): 13_36_44 0.147 (VO 0.174), 13_45_18
  0.098, 13_21_59 0.316 (VO 0.086). Marginalization (± td) remains unreliable on this data (0.049–4.6 m,
  scale sometimes collapsing). Second open failure: final standstill with a KF at every image (also with the
  synthetic IMU).
- Per-factor-type final costs added to `VIOptimStats` / `vio_diag.csv` (`cost_visual`, `cost_imu`,
  `cost_bias`, `cost_prior`).

**RealSense matrix with vibration-level IMU noise** (snapshot `diag2`, 3 runs per cell, median ATE in m; stereo;
"nominal" = `kalibr_work/imu.yaml` noise, n5 = densities accel 0.15 / gyro 0.02, n6 = n5 + 10× bias random walks):

| traj | VO | VIO nominal | VIO n5 | VIO n6 | VIO marg n6 | VIO marg+td n6 |
|---|---|---|---|---|---|---|
| 13_18_29 | 0.045 | 2.213 (4 resets) | 0.061 | 0.035 | 0.061 | 0.052 |
| 13_21_59 | 0.086 | 142.9 | 0.185 | 0.155 | 0.179 (1) | 0.108 |
| 13_30_22 | 0.048 | 0.775 | 0.060 | 0.046 | 0.060 | 0.070 |
| 13_31_26 | 0.077 | 17.87 (1) | 0.054 | 0.081 | 0.069 | 0.081 |
| 13_34_21 | 0.029 | 0.220 | 0.031 | 0.045 | 0.131 | 0.047 |
| 13_36_44 | 0.174 | 288.7 (2) | 0.158 | 0.187 | 0.878 | 1.648 |
| 13_41_28 | 0.065 | 77.98 | 0.097 | 0.100 | 2.576 | 2.387 (1) |
| 13_42_11 | 0.139 | 11.49 (4) | 0.207 | 0.137 | 0.599 (1) | 2.648 (1) |
| 13_45_18 | 0.054 | 100.1 | 0.056 | 0.104 | 1.506 | 0.127 |
| **median ATE** | **0.065** | 17.87 | **0.061** | 0.100 | 0.179 | 0.108 |
| median RPE 1 s | 0.034 | 7.36 | 0.031 | 0.035 | 0.071 | 0.033 |
| median Sim3 scale | 1.002 | 0.054 | 1.003 | 1.002 | 0.996 | 0.999 |
| resets / 27 runs | 0 | 11 | 0 | 0 | 2 | 2 |
| trajectories within 1.5× VO | 9 | 0 | 7 | 5 | 3 | 4 |

Reading: with the configured (datasheet-level) IMU noise, VIO fails on this handheld rig; with densities raised
to the measured vibration level, default VIO (n5) is on par with VO (median 0.061 vs 0.065, 0 resets) but not
better; marginalization still fails on 3–4 of 9 trajectories (13_36_44, 13_41_28, 13_42_11, partly 13_45_18).
Unlike TUM-VI, VIO does not beat VO here. Open: the marginalization failures (likely the bias / vibration model
under a confident prior), the final-standstill failure, and the uncalibrated camera-IMU extrinsic. A Kalibr
camera-IMU calibration (with an Allan-variance noise model of the mounted IMU) is the natural next step.

## TUM-VI rooms 2–6 and the VO back-end control (2026-10-03)

rooms 2–6 downloaded by the user (MD5 match with the TUM-VI server, extracted to `SaD_VIO_data/tumvi/`, tars
deleted; same 512_16 calibration as room1). All runs: snapshot `diag2` (fixes 1–8 + diagnostics), offline
binary, 3 runs per cell; VIO default room1 / magistrale2 from `rb_default`, marg + td room1 / magistrale2 from
`cd_*` (fixes 1–7). Question (user): is the gain from marginalization + sparsification itself rather than from
the IMU? → control: the same back-end options in VO (`vo_marg`, `vo_margsparse`; estimate_td is IMU-only).

Mean ATE (m), stereo:

| seq | VO | VO marg | VO marg+sparse | VIO | VIO marg+td | VIO marg+sparse+td |
|---|---|---|---|---|---|---|
| room1 | 0.142 | 0.196 | 0.205 | 0.266 | 0.069 | **0.055** |
| room2 | 0.158 | 0.168 | 0.171 | 0.428 | **0.093** | 0.094 |
| room3 | 0.175 | 0.143 | 0.155 | 0.154 | **0.087** | 0.093 |
| room4 | 0.118 | 0.095 | 0.121 | 0.144 | **0.066** | **0.066** |
| room5 | 0.198 | 0.180 | 0.178 | 0.333 | 0.113 | **0.062** |
| room6 | 0.083 | 0.097 | 0.082 | 0.072 | **0.047** | 0.057 |
| magistrale2 (end drift) | 15.9 (2 runs) | 15.8 (2 of 3) | 13.1 (2 of 3) | 18.6 | **3.2** | — |

Mean RPE 1 s (m): VO 0.016–0.032, VO marg 0.017–0.035, VO marg+sparse 0.016–0.031, VIO 0.013–0.034,
VIO marg+td 0.008–0.019, VIO marg+sparse+td 0.008–0.013 (magistrale2: VO 0.046, VO m+s 0.042, VIO 0.033,
VIO marg+td 0.015).

**Reading:** marginalization / sparsification leave VO within ~±20 % of plain VO on every room (RPE unchanged);
with the IMU and the time offset, ATE drops 30–70 % and RPE roughly halves on every sequence. The gain comes
from the IMU, which marginalization (bias / velocity memory) and the time offset make usable — not from the
back-end options alone. Default VIO is about level with VO.

New issues seen in the control (VO, magistrale2): `vo_marg` run_00 re-initialized near the end (coverage
0.008; the evaluator's reset counter only counts VIO re-inits) → its 59.8 m mean ATE is that artifact (other two
runs 6.4 / 8.4 m); `vo_margsparse` run_02 and **VO default** (`rooms_default` magistrale2 run_01) **crashed
(exit 139, segfault)** — 2 crashes in 9 VO runs on magistrale2 with snapshot `diag2`, whereas VO ran 3/3 there with
`final2`: a regression in code shared with VO since then, to debug (gdb, debug build). VO + marginalization
(+ sparsification) on long sequences also needs its own robustness pass (the robust loss of fix 5 is VIO-only).

## Open items 1–4 (user request 2026-10-03: VO crash, ROS check, standstill KF flood, VO + marginalization)

### Item 1 — VO segfault on magistrale2 (2 of 9 VO runs with `diag2`)
Reproduced: debug build under gdb 4/4 clean (timing), release `diag2` under gdb 1 crash in 6
(`runs/debug_vocrash/magistrale2_vo_r6_gdb.log`): `SLAMBiMono::init` (a VO re-initialization near the end
of magistrale2) → `SLAMCore::initLandmarks` → `Point3DLandmarkInitializer::initLandmark` → `AFeature::getRays`.
**Cause (original code):** stereo VO's `init()` did not clear `_matches_in_time` / `_matches_in_time_lmk`, which
still held the matches of the frames before the re-init; those frames had been cleaned, so the features' sensor
pointers had expired and `getRays()` dereferenced a null sensor (the stereo VIO init already cleared them; the
mono inits refill them via `trackFeatures`). `initLandmarks` also dereferenced `getLandmark().lock()` unchecked.
Not a regression: rare because it needs a VO re-init (the old "late re-init" artefacts). **Fix 9:** clear both
in `SLAMBiMono::init`; `initLandmarks` skips matches whose landmark is gone and features without a sensor.
Note: a segfault also loses the end of stdout (buffered), so crash logs are incomplete.

### Item 3 — KF flood at standstill
`log_slam/kf_votes.csv` added (per tracked frame: parallax, matches, decision and reason). The "standstill"
flood seen with the GT-synthesised IMU (13_18_29) is not a standstill effect: with that IMU, only 24–45 landmarks
were tracked during *motion* (15–35 s, 50–65 s) and the few-landmark rule fired at ~80 % of frames; with the real
IMU (n5) on the same images 55–124 landmarks are tracked, that rule fires 0–23 times per 5 s and the final
standstill triggers essentially nothing. The synthetic IMU (smoothed GT, no handheld jitter) predicted feature
motion badly → tracking losses → flood. On magistrale2 (real data) ~37 % of KFs come at every image (≈30 floods
of ≥ 10 KFs per run, up to 11 s) in runs that are clean and the best we have (cd_margtd: 0 resets, ATE
1.2–1.6 m): the rule reacts to segments with < 50 tracked landmarks. **No change** to the KF policy (shared with
VO) without evidence of harm; the vote log stays as a diagnostic.

**Item 1 validation** (snapshot `vofix` = fix 9 + VO robust loss): VO default magistrale2 6/6 without crash
(ATE 6.5–8.2 m, coverage 1.00; before 2 crashes in 9).

### Item 4 — VO + marginalization robustness
**Change:** with `marginalization: 1`, VO also uses the robust visual loss, in its window (`localMapBA`) and in the
marginalization of visual factors (`AOptimizer::setRobustVisualVO`, set from the config at optimizer creation);
the prior itself is never robustified (`localMapBA` passed the window's loss to it — harmless while it was null).
Default VO unchanged.
**Validation (`vofix`):** VO + marg magistrale2 3/3 clean (ATE 6.1–8.1, drift 13–17 m, as plain VO; before 1 of 3
late re-init); VO + marg + sparse room1 6/6 clean (0.13–0.23); VO + marg room1 5/6 clean (0.16–0.25), 1 run with a
single 7.5 m jump at 50.8 s; VO + marg + sparse magistrale2 1/3 clean, 2 with a single jump (228 s, 237 s). VIO
sanity (room1 marg + td): 0.079 / 0.080.
**The remaining failures are a front-end problem, not marginalization:** `kf_votes.csv` shows one frame with an
absurd pose (rotation-compensated "parallax" 33–89°), voted KF (parallax rule), which the window then cannot
pull back. In `SLAMCore::predict` the PnP plausibility check was skipped when the predicted translation since the
last KF was < 0.1 m, so a PnP solution with a flipped rotation passed during slow motion (garbage PnP poses of
10⁶–10⁹ m appear in many magistrale2 logs, where the check happened to apply).
**Fix 10:** `SLAMCore::plausibleUpdate` — the PnP pose is rejected if it deviates from the constant-velocity
prediction by more than 10 × the predicted motion with a floor of 0.1 (rotation [rad] + translation [m]), i.e.
always ≥ 1.0 allowed; the same gate on the ESKF update in the four pipelines (an implausible ESKF update keeps the
PnP pose, with a message). Validation pending (built after the ROS batch, which plays bags in real time).

### Item 2 — ROS playback
ROS package rebuilt (colcon, dense_devel workspace in `sad_vio_dense`, 2026-10-03, with fixes 1–9 and the VO robust
loss). New `tools/run_sadvio_ros.py`: same configs / scratch / results / evaluation as the offline runner; starts
`vio_ros <config>` and plays the original RealSense bag in real time (`ros2 bag play --rate 1`, camera + IMU topics
only, isolated ROS domain, localhost only). 12 runs (13_18_29, 13_36_44, 13_45_18 × VO / VIO n5 × 2): all
complete, playback in real time (bag duration + ~8 s start-up), 0 resets, same coverage as offline, and ATE / RPE
within the offline run-to-run spread (e.g. 13_18_29 VO ROS 0.039 / 0.044 vs offline 0.039–0.046; VIO n5 0.055 /
0.059 vs 0.053–0.062; RPE 1 s 0.017–0.060 vs 0.021–0.065). The ROS reader rewritten in Issue 8 works on real
data. Not covered: TUM-VI through ROS (no ROS 2 bag here), marginalization configs through ROS, multithreading in
ROS.

**Fix 10 validation** (snapshot `fix10`, 66 unit tests pass): no single-frame jumps left, every run at full
coverage, no crash:

| config | room1 ATE (m) | magistrale2 ATE (m) | before (`vofix`) |
|---|---|---|---|
| VO + marg | 6/6, 0.12–0.24 | 3/3, 6.8–7.7 | room1 1/6 with a 7.5 m jump |
| VO + marg + sparse | 3/3, 0.13–0.20 | 3/3, 5.4–7.7 | magistrale2 2/3 with a jump |
| VO default | 0.16–0.20 | 6.4–7.7 (no crash) | unchanged |
| VIO default | 0.26 / 0.32 | — | 0.265 ± 0.015 |
| VIO marg + td | 0.067 / 0.088 / 0.095 | 1.68 / 1.94 (drift 3.7 / 4.2) | 0.069–0.087; magistrale2 1.85 / 4.1 after fix 8 (1.47 / 3.2 before it) |

PnP plausibility rejections: 0–2 per run; the ESKF gate never fired (the bad poses came from PnP). VIO is unchanged
by fix 10; the magistrale2 VIO margin since fix 8 (1.47 → ~1.7–1.9 m) remains to be understood.

## Follow-up items (user request 2026-10-03: "do 1 3 4 and 5", item 2 = RealSense marginalization deferred)

### Item 1 — magistrale2 VIO "regression" since fix 8: none, run-to-run variance
A/B snapshots on top of `fix10`: `varA` (`canBridge` always true = the bridging of before fix 8), `varB`
(`kMaxImuFactorDt = 10` s instead of 1 s). magistrale2, stereo VIO, marginalization + `estimate_td`:

| variant | runs | ATE (m) | median | room1 ATE |
|---|---|---|---|---|
| A, always bridge (before fix 8) | 6 | 4.65 1.43 1.26 1.31 4.54 0.73 | 1.37 | 0.080 0.078 |
| B, 10 s bridges | 3 | 1.84 1.32 2.87 | 1.84 | 0.081 0.061 |
| current code (`mem`, `rss_mem`) | 4 | 0.98 1.23 1.50 4.66 | 1.36 | — |
| `fix10` (`f10_viomargtd`, `rss_fix10`) | 3 | 1.94 1.68 1.65 | 1.68 | 0.067–0.095 |
| before fix 8 (`cd_margtd`, `rb_margtd`) | 6 | 1.60 1.24 1.57 1.08 1.75 2.87 | 1.59 | — |

The 1.47 → 1.7–1.9 m difference was sampling noise: the same code spreads over 0.7–4.7 m. magistrale2 has ground
truth only at the start and the end (mocap room), so its ATE is essentially one end-to-end drift per run. Both the
old and the current bridging have the same failure mode, a run ending at ATE ≈ 4.6 m / drift ≈ 10 m (2 of 6 with
A, 1 of 4 current). **No change**; fix 8 stays. The ≈ 4.6 m mode (~1 run in 3–4) is a separate open item.

### Item 4 — smaller items
- **Unit tests (5 old failures):**
  - `LineFeatureDetection`: called `cv::imshow` in the headless container → commented out.
  - `angularTest`: the analytic Jacobian of `AngularErrCeres_pointxd_depth` w.r.t. the second pose was
    only exact at zero motion → fixed with `se3_RTtoVec6d(dTa)` (used by the angular optimizer with inverse depth).
  - `RelativePose6DResidual`: the rotation blocks of `Relative6DPose`'s Jacobians were missing the rotations of
    the linearization point → fixed (used in the relative-pose factor of the VO marginalization / NFR).
  - `ESKFLmkTest`: `ESKFEstimator::refineTriangulation` (not called by the pipelines) used noise values far looser
    than the test's precision → rewritten as an iterated MAP / Gauss-Newton update (5 iterations, 1.5 px noise,
    only features still attached to a sensor).
  - `LineFeatureMatching`: **not fixed, reported**. Its pass threshold sits at the median of a randomly shuffled
    detector, so it is a coin flip (8 of 20 seeds pass); fixed seeds in `SetUp` make it deterministic (fails).
  - Suite: 70 pass, 1 fails (`LineFeatureMatching`), in `build_tests` and in the rebuilt `build`.
- **`ISAESLAM_WITH_VDBGPDF=OFF` link error:** `VDBGPDFMap.cpp` had no definitions without VDB; added an `#else`
  branch with stubs (constructor, `integrate` and `mesh` throw). `cpp/build_novdb` configures, builds and links.
- **IMU-chain memory growth:** every `IMU` keeps a `shared_ptr` to the previous sample, so the whole run's IMU
  history stayed alive. `SLAMCore::releaseImuHistory(kf)` cuts the link of the oldest window KF before it is
  discarded (`slamBiMonoVIO`, `slamMonoVIO`); nothing walks the chain past the oldest window KF (repropagation
  stops at the last KF, bridging at the previous window KF). RSS trace (`run_rss.sh`, `rss.log`), magistrale2
  stereo VIO marg + td: `fix10` 364 MB at 1 min → 625 MB at 9.5 min (≈ 0.5 MB/s); with the fix flat at
  338–344 MB, peak 350 MB. Trajectory unaffected (see item 1, the `mem` runs).
- **Mono "few tracks" double count:** `n_matches` already includes the landmark matches, the mono rule used
  `n_matches + n_matches_lmk < 50` → now `n_matches < 50`. A/B on one build (`mem2` vs `mem2_dc` = the old count),
  TUM-VI rooms 1–6:
  - mono VO: re-inits 73 vs 144 in total, coverage room1 0.81 vs 0.17, room6 0.92 vs 0.40; rooms 2–5 mono VO fails
    with both (coverage ≤ 0.05, 8–63 re-inits) — pre-existing, not addressed.
  - mono VIO: neutral, mean ATE over the 6 rooms 0.36 vs 0.36 m (better on room2 0.68/0.78 and room4 0.17/0.20,
    worse on room3 0.41/0.30 and room6 0.19/0.13, equal on rooms 1 and 5); ~30 % more KFs; 0 resets in both.
  - **Kept** (it is the rule as documented, and it helps mono VO).
- **Re-init messages:** mono VO, mono VIO and stereo VO re-initialized silently (only stereo VIO printed
  "Reinitializing"), so the `resets` column was 0 for them; they now print the same message. Earlier mono / VO
  `resets` values in `RESULTS_ALL_RUNS.md` therefore under-count.

### Item 3 — ROS coverage: TUM-VI, marginalization, multithreading
- **TUM-VI through ROS:** new `tools/euroc_to_ros2bag.py` writes an EuRoC sequence as a ROS 2 Humble sqlite3 bag
  with the container's own rosbag2 (`/cam0/image_raw`, `/cam1/image_raw` mono16 as in the official TUM-VI bags,
  `/imu0`; dataset stamps as message and record times). room1 converted to
  `SaD_VIO_data/tumvi/dataset-room1_512_16/ros2bag` (2.8 GB). `run_sadvio_ros.py` takes the TUM-VI sequences that
  have a `ros2bag`. The node's cv_bridge mono16 → mono8 conversion (×255/65535) matches the offline `imread`
  (high byte) to ≤ 1 grey level.
- **Results (room1, 1 ROS run per config, final code, offline = same binary, 1–2 runs):**

| config | ROS ATE / RPE 1 s | offline ATE / RPE 1 s |
|---|---|---|
| stereo VIO default | 0.237 / 0.027 | 0.332, 0.269 / 0.026 |
| stereo VIO marg + td | 0.083 / 0.014 | 0.081, 0.062, 0.080 / 0.013–0.014 |
| stereo VIO multithreading | 0.274 / 0.025 | 0.301, 0.231 / 0.026–0.028 |
| stereo VIO multithreading + marg + td | 0.097 / 0.014; 0.095 after the race fix | see below |
| stereo VO marg + sparse | 0.151 / 0.023 | 0.216, 0.146 / 0.023–0.025 |
| stereo VO default | 0.198 / 0.026 | 0.153, 0.172 / 0.025–0.027 |
| mono VIO default | 0.310 / 0.104 | 0.241, 0.301 / 0.078–0.086 |
| mono VIO marg + td | 0.141 / 0.036 | 0.354, 0.124 / 0.036–0.115 |

  All complete in real time (bag + ~8 s), coverage 0.98, 0 resets, within the offline spread. The ROS node drops a
  few left images when the machine is loaded (0–7 of 2821 "Sync error", reliable QoS depth 10, 512 kB images; 0
  when idle): transport, not the stereo pairing logic.
- **Found: multithreading race (fix 11).** `multithreading: 1` + marginalization had never been run; offline it failed
  in 8 of 10 runs (marg: 4/4, ATE 0.71–2.06 m, RPE 1 s 0.12–0.22; marg + td: 4/6, ATE 0.43–0.57), with single-KF
  jumps that the window then keeps (e.g. 0.55 m in z at 51.55 s, cost 1677 → 40 onto the jumped pose).
  **Cause:** the front end and the back end share `_step_mutex`; the front end releases it only while loading the
  next frame. When the back end had not yet taken it, the front end processed the next frame while
  `getLastKF()` was still the KF *before* the pending `_frame_to_optim`: the frame's IMU preintegration was anchored
  to that older KF (`setLastKF`), and if it became a KF its IMU factor skipped a KF of the window (`vio_diag.csv`:
  factor_dt 0.20 s with the previous KF 0.05 s earlier). Without marginalization the effect was mild (the default
  config's multithreading runs looked fine). **Fix 11:** `SLAMCore::nextFrame()` returns the frame only once the
  back end has taken the pending KF (`waitBackEnd()` after relocking); the overlap that remains is the data
  loading, as before (the two never computed at the same time anyway). Single-thread unchanged (no front-end lock).
  New `tools/check_kf_skips.py` counts IMU factors that skip a KF other than low-parallax bridges.

| snapshot | config | runs | ATE (m) | race skips per run |
|---|---|---|---|---|
| before (`build` final3) | mt + marg | 4 | 0.78 0.82 2.06 0.71 | 15–29 |
| before | mt + marg + td | 6 | 0.48 0.10 0.06 0.45 0.43 0.57 | 0–21 |
| `mtfix` | mt + marg | 4 | 0.20–0.22 (single thread 0.18–0.24) | 0 |
| `mtfix` | mt + marg + td | 6 | 0.072–0.098 | 0 |
| `mtfix` | mt default / mono VIO marg + td / VO / mono VO | 3/3/2/2 | as single thread, 0 crashes | 0 |
| `build` final4 | mt + marg + td | 3 + ROS 1 | 0.081–0.100; ROS 0.095 | 0 |

  Single-thread runs and all ROS runs before the fix: 0 race skips (ROS pacing gives the back end time).

### Item 5 — builds
`cpp/build` (main, FFS + VDBGPDF) and the ROS colcon workspace (`ros/`, FFS + VDBGPDF + LAS2) rebuilt with the final
code (fixes 1–11, items 1–4) on 2026-10-03; unit tests in `cpp/build`: 70 pass, `LineFeatureMatching` fails (coin
flip, above). Sanity with the main binary (room1): stereo VIO 0.33 / marg + td 0.080–0.081, stereo VO 0.153 /
0.172, mono VIO 0.24 / 0.30, multithreading + marg + td 0.081–0.100; through ROS 0.095.

### Live view (2026-10-03, user request)
magistrale2 converted to a ROS 2 bag (`tumvi/dataset-magistrale2_512_16/ros2bag`, 10.6 GB, 540 s). New
`tools/live_ros_compare.py`: two `vio_ros` nodes in namespaces (`/vo` stereo VO default, `/vio` stereo VIO marg + td,
`enable_visu: 1`) on one real-time playback, one RViz window each on the xpra display `:20` (topics of
`ros/launch/isae_slam.rviz` prefixed by the namespace). Result (`runs/live_vo`, `runs/live_vio`): VO ATE 5.88 m, end
drift 12.5 m, RPE 1 s 0.035; VIO ATE 1.97 m, drift 4.3 m, RPE 1 s 0.015; both full coverage, 0 resets, real time
(554 s wall for 540 s of data) with both nodes and two RViz on the machine; 1–4 dropped left images.

### Run replay page (2026-10-03, user request)
New diagnostic log: `SLAMCore::logKfFeatures` writes `log_slam/kf_features.csv` (KF stamp, pixel u, v, landmark
status as in the ROS `image_kps` view) when `EXECO_KF_FEATURES_LOG` is set, called at the KF hand-off in the four
pipelines; `run_sadvio.py --kf-features` sets it. Runs `viewer_vo` (stereo VO default) / `viewer_vio` (stereo VIO
marg + td), final code, all 7 TUM-VI sequences; `tools/build_traj_viewer.py` packs the trajectories (two GT
alignments: all KFs / first 20 s), errors, metrics and keypoints; the camera videos (320 px, 20 fps, H.264) are
encoded separately. Published as the private artifact https://claude.ai/artifact/RdSgwnB2HmxsG5BFB71ias.
ATE VO / VIO: room1 0.139 / 0.089, room2 0.215 / 0.089, room3 0.154 / 0.104, room4 0.120 / 0.063,
room5 0.161 / 0.084, room6 0.071 / 0.059, magistrale2 6.94 / 2.81 m.
