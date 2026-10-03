# VIO fix + LIO prospects

Branch: `dense_devel` · Code: `f157380` · Written: 2026-10-01 · Revised: 2026-10-02 · Status: review
done, no fixes applied yet

This document records a review of the IMU integration in SaDVIO (stereo VO works, VIO fails in both
mono and stereo), what "Sparsify" and "Densify" mean in this project, and options for a separate
LiDAR-inertial project (SaDLIO) built on the same ideas. Actionable items are checklists.

**Revision 2026-10-02.** This version merges the corrections from
[`VIO_fix_+_LIO_Prospects_review.md`](VIO_fix_+_LIO_Prospects_review.md). Its new findings
(Issues 10-18 below, labelled R1-R9 in the review) were re-checked against the source, and R2, R3
and R5 were re-checked with independent finite-difference tests; all were confirmed. Issue 9 was
replaced, the bias-prior and lever-arm guidance were corrected, and the SaDLIO part was qualified.
Issues 19 (online time offset) and 20 (option combinations that silently do nothing) are new in
this revision. Scope: every config option must work, not just the checked-in configuration.

"Confirmed" means the defect is visible in the source (and numerically where stated). It does
**not** mean it has been shown to cause the reported failure: no failing recording was replayed and
no C++ build or test was run for this document.

Reference: Debeunne, Torres, Vivet, *SaDVIO: Sparsify and Densify VIO for UGV Traversability
Estimation*, IEEE RA-L 2024 ([`RAL_2024___SaDVIO.pdf`](../RAL_2024___SaDVIO.pdf)). The paper uses
Forster on-manifold preintegration for the inertial residual and ORB-SLAM3-style inertial
initialization.

---

## Part 1 — IMU background (beginner level)

- **What an IMU measures.** The gyroscope measures rotation rate (rad/s). The accelerometer measures
  *specific force*: acceleration minus gravity, in the sensor's own axes. At rest it reads about
  9.81 m/s² pointing up; in general `f_B = R_BW (a_W − g_W) + b_a`. It reads (0, 0, +9.81) only
  when a forward-left-up body is level.
- **Integration (dead reckoning).** Summing rotation rate gives orientation; summing acceleration
  (gravity removed) gives velocity, and summing velocity gives position. Errors grow quickly.
- **Bias.** Each sensor has a slowly drifting offset. A gyro bias of 0.01 rad/s becomes a 30°
  heading error within a minute, so the estimator estimates biases along with the rest of the state.
- **Preintegration.** The IMU samples between two keyframes (KF i and KF j) are summarised once as
  "rotated by ΔR, velocity changed by Δv, moved by Δp, all in KF i's axes". The optimizer uses that
  summary as a spring (the **IMU factor**) between the two KFs. This avoids re-integrating during
  ordinary optimization; re-integration may still be needed after a large bias change.
- **Bias correction.** The summary is computed with a bias guess. When the optimizer improves the
  bias, the summary is corrected to first order: summary + Jacobian × bias change. This only works
  if the code always knows **which bias the summary was computed with**.
- **Noise parameters.** `*_noise_density` sets how much the IMU is trusted relative to the camera;
  `*_random_walk` sets how fast the biases may drift. The code discretizes the noise density as
  σ²·rate, which assumes a nominal, regular sample rate.
- **Frames.** The IMU axes must be the same body axes the camera extrinsics (`T_BS`) are expressed
  in. The code assumes the frame pose *is* the IMU pose, including its origin. The cleanest setup
  makes the IMU the body frame and expresses the cameras relative to it (the form Kalibr outputs).
- **Initialization.** Before the IMU helps, the system needs the gravity direction, velocities, rough
  biases and (mono) the scale. SaDVIO runs vision only for 10 KFs, then solves for these (`VIInit`).

### What has been checked so far

- Preintegration recursion in [`IMU.cpp`](../cpp/src/data/sensors/IMU.cpp): the ΔR/Δv/Δp updates,
  the 9×9 covariance recursion and the five bias-Jacobian recursions follow Forster's structure,
  assuming one consistent bias and regular timing. Issues 4 and 18 show those assumptions are not
  always met.
- `IMUFactor` in [`residuals.hpp:145`](../cpp/include/isaeslam/optimizers/residuals.hpp#L145): the
  residuals match Forster. The Jacobians match **at zero increment only**; the pose-j rotation block
  is wrong at nonzero increments (Issue 12).
- The VIO sparse prior (`IMUPriordx` + `PoseToLandmarkFactor`) has the residual form of paper eqs.
  4 and 5, but its Jacobians and the information recovery are wrong (Issues 13 and 14).
- The existing Jacobian tests cannot fail on a Jacobian error (Issue 17), so "tests pass" is not
  evidence of correctness yet.

---

## Part 2 — VIO issues and fixes

### Goal and scope

The goal is that **every option exposed in the config works correctly**, for any user and any
combination: `slam_mode` (`mono`/`bimono` VIO), `optimizer` (`Analytic`, `Numeric`,
`AngularAnalytic`), `multithreading`, `estimate_td`, `marginalization`, `sparsification`, both
readers (offline EuRoC-format and ROS, raw and compressed) and any dataset yaml. No issue below is
out of scope because a particular config does not exercise it. The table says which options trigger
each issue, so each fix can be tested on the right combinations.

| Issue | Triggered by |
|---|---|
| 1, 2, 3, 11, 12, 18 | Every VIO run (mono and stereo, all optimizers) |
| 4 | Every VIO run (`updateBiases()`, `VIInit`); the front-end/back-end race only with `multithreading: 1` |
| 5 | Every VIO run once the window is full (low-parallax drop); the factor check in both analytic optimizers' `marginalize()` with `marginalization: 1` |
| 6 | Any dataset whose IMU axes differ from the body axes (RealSense configs today); the offline reader ignores IMU `T_BS` in all cases |
| 7 | Offline reader (`cpp/main.cpp`), any dataset |
| 8 | ROS reader, raw and compressed |
| 9 | VIO initialization; the `_6d_velocity` scale part only in mono |
| 10 | Every VIO run, when the visual and IMU rotations agree exactly |
| 13, 14 | `sparsification: 1` with `Analytic` or `AngularAnalytic` |
| 15 | `sparsification: 1` with `Analytic` |
| 16 | Mono VIO: oldest KF never marginalized regardless of `marginalization` |
| 17 | Test suite; affects confidence in every fix |
| 19 | `estimate_td: 1` (both pipelines); the offset is never applied by the offline reader |
| 20 | `optimizer: Numeric` with `marginalization: 1` and/or `estimate_td: 1` |

For reference, the checked-in [`config.yaml`](../ros/config/config.yaml) uses `airsim`, `mono`,
`AngularAnalytic`, `multithreading: 0`, `estimate_td: 1`, `marginalization: 0`, `sparsification: 0`.
When reproducing a specific failure, record the configuration actually used.

### Issue 1 — Gyro bias random walk read from the accelerometer key

[`adataprovider.cpp:85`](../cpp/src/dataproviders/adataprovider.cpp#L85) sets `bgyr_noise` from
`accelerometer_random_walk`. With the EuRoC values that is ~52× the intended standard deviation
(~2,700× the variance), so the optimizer uses the gyro bias as a fudge knob.

- [x] Read `gyroscope_random_walk` for `bgyr_noise`.

### Issue 2 — `VIInit` estimates biases but applies them wrongly

[`AOptimizer.cpp:477-483`](../cpp/src/optimizers/AOptimizer.cpp#L477-L483). Commit `291a492`
("improvement mono vio") freed the bias blocks that upstream held constant, which made these bugs
live. Estimating the biases is correct (ORB-SLAM3 does it); applying them is what is broken.

- [x] Call `setBa` once (it is called twice, so Δba is applied 2×).
- [x] Add the missing `setBg` (the gyro bias estimate is currently discarded).
- [x] Apply the bias change to the preintegrated deltas, consistent with the policy chosen in Issue 4.
- [x] Replace the bias prior ([`AOptimizer.cpp:451`](../cpp/src/optimizers/AOptimizer.cpp#L451))
      with an explicit, justified initial-bias uncertainty. **Do not just convert `dt` to seconds
      and keep the ×1000**: that shrinks the prior σ to ~2e-6 and effectively freezes the biases at
      their static-init values. Initial-bias uncertainty is not the same as random-walk drift over
      the window.
- [x] Test on synthetic data with a known nonzero bias that the init recovers it.

### Issue 3 — `IMUFactorInit` position residual and scale Jacobian

[`residuals.hpp:350-353`](../cpp/include/isaeslam/optimizers/residuals.hpp#L350-L353) computes
`R·Q·(s·Δp_W − v·dt − ½g·dt²)` with `R = R_fi_w`, `Q = R_w_i`, `s = exp(λ)`. After the solve, the code
transforms the map with `T_f_I = (R·Q, s·t)`, so positions become `p_I = Qᵀ·s·p_W` while velocities
are already in the aligned frame. The residual consistent with **this repository's convention** is:

```text
r_p = R (s Δp_W) − R Q (v_i dt + ½ g dt²) − Δp_corrected
```

ORB-SLAM3 uses a different parameterization (it also scales the velocities), so its equations
should not be copied verbatim. The scale Jacobian
([`residuals.hpp:418`](../cpp/include/isaeslam/optimizers/residuals.hpp#L418)) is missing the
`exp(λ)` factor, which matters for mono (the scale starts at an arbitrary 1/10).

- [x] Fix the position residual to the form above.
- [x] Update the matching gravity Jacobian block (the skew term must no longer contain `s·Δp_W`).
- [x] Multiply the scale Jacobian by `exp(λ)`.
- [x] Treat `IMUFactorInitBis` (unused today) as unvalidated: it scales the velocities, and its scale
      Jacobian lacks both `exp(λ)` and `inf_sqrt`.
- [x] Add numerical-vs-analytic Jacobian tests at λ ≠ 0 and nonzero gravity correction (see
      Issue 17), plus a model-consistency test: a finite-difference check can validate the
      derivative of a physically wrong residual.

### Issue 4 — Bias linearization point gets out of sync

The factor treats `d_b` as relative to the bias currently stored on KF i, but the deltas were
integrated with whatever bias was current at integration time
([`IMU.cpp:23`](../cpp/src/data/sensors/IMU.cpp#L23)). Even within one integration step, the
increments use the previous sample's bias ([`IMU.cpp:34-36`](../cpp/src/data/sensors/IMU.cpp#L34-L36))
while the right Jacobian and the acceleration coupling use the KF's bias
([`IMU.cpp:37`](../cpp/src/data/sensors/IMU.cpp#L37), [`IMU.cpp:81`](../cpp/src/data/sensors/IMU.cpp#L81)).
The two diverge because:

- with `multithreading: 1`, the back end optimizes KF j while the front end keeps integrating toward
  KF k with KF j's old bias; the post-optimization correction only touches frames already in the
  window;
- `updateBiases()` overwrites the previous KF's optimized bias with the newest KF's bias
  ([`slamBiMonoVIO.cpp:691`](../cpp/src/slamBiMonoVIO.cpp#L691),
  [`slamMonoVIO.cpp:628`](../cpp/src/slamMonoVIO.cpp#L628)), behind the optimizer's and the
  marginalization prior's back;
- `VIInit` (Issue 2).

- [x] Choose one policy and apply it everywhere: either the stored deltas stay at an immutable
      integration bias, or they are corrected in place together with their declared linearization
      point. Mixing the two applies a bias change twice.
- [x] Store the linearization biases (`b_a_lin`, `b_g_lin`) inside each preintegration, and use one
      bias for all terms of an integration step.
- [x] In `IMUFactor`, compute the bias change as `b_i + d_b − b_lin` (ORB-SLAM3 / VINS approach),
      and drop the in-place `biasDeltaCorrection` after optimization if that is the chosen policy.
- [x] Remove the `updateBiases()` calls (or make them no-ops) once the above is in.
- [x] Cover initialization, front-end propagation, prediction and the `estimate_td` path.
- [x] If re-integration is part of the policy (when `|b − b_lin|` is large), store the raw samples.
- [x] Test a bias change during an active interval; validate single-threaded first, then threaded.

### Issue 5 — Low-parallax KF removal cuts the IMU chain and corrupts marginalization

[`slamBiMonoVIO.cpp:646`](../cpp/src/slamBiMonoVIO.cpp#L646) and
[`slamMonoVIO.cpp:602`](../cpp/src/slamMonoVIO.cpp#L602) remove the second-newest KF without
marginalizing it and set the new KF's `lastKF` to null. The IMU information across the gap is lost,
precisely in slow or rotation-only motion. Later, `marginalize()` builds `IMUFactor(frame0, frame1)`
without checking that `frame1`'s preintegration starts at `frame0`, baking a wrong factor into the
prior: [`AngularAdjustmentCERESAnalytic.cpp:703`](../cpp/src/optimizers/AngularAdjustmentCERESAnalytic.cpp#L703)
and [`:1125`](../cpp/src/optimizers/AngularAdjustmentCERESAnalytic.cpp#L1125),
[`BundleAdjustmentCERESAnalytic.cpp:625`](../cpp/src/optimizers/BundleAdjustmentCERESAnalytic.cpp#L625)
and [`:857`](../cpp/src/optimizers/BundleAdjustmentCERESAnalytic.cpp#L857). The paper states that
mid-frame marginalization is not needed, so this branch deviates from the paper's design.

- [x] Remove the `_parallax < 0.5` branch in both pipelines, or replace it with merging the two
      preintegrations (ΔR = ΔR₁ΔR₂, etc., with covariance and Jacobian propagation).
- [x] In `marginalize()` and `marginalizeRelative()` (both optimizers), only add the IMU factor when
      `frame1->getIMU()->getLastKF() == frame0` (and dt ≤ 1 s, as in `addIMUResiduals`).
- [x] Note: `_parallax` is written by the front end and read by the back end without
      synchronization.

### Issue 6 — IMU frame in the RealSense configs

`sensor_config_realsense_d435i_sadvio.yaml` (in `Simulator_Validation`) and
`realsense_optitrack*.yaml` map the cameras from optical axes into a forward-left-up body, but set
the IMU `T_BS` to identity. RealSense IMU data uses optical-style axes (x right, y down,
z forward), so gyro rotations would not match camera rotations. The
[RealSense wrapper documentation](https://github.com/realsenseai/realsense-ros) also notes that
separate gyro and accel streams fill only their own fields; a combined IMU stream requires merging.
This diagnosis depends on the recording: confirm it before changing calibration.

- [ ] Inspect the actual bag: IMU `frame_id`, published transforms, which fields are filled, and
      whether gyro and accel were merged (and how).
- [ ] Make the IMU the body frame: express the camera `T_BS` relative to the IMU (Kalibr
      camera-IMU output) and set the IMU `T_BS` to identity. Adding a translation to the IMU
      `T_BS` does nothing: the code only uses its rotation and has no lever-arm model.
- [ ] If a separate body origin is kept, add an explicit lever-arm model instead.
- [x] Apply the IMU `T_BS` rotation in the offline EuRoC grabber too; today only the ROS path
      ([`sensorSubscriber.h:70`](../ros/src/sensorSubscriber.h#L70)) applies it.
- [x] Do not rely on the name `T_s_f`: it is inverted for cameras but not for the IMU in
      [`adataprovider.cpp`](../cpp/src/dataproviders/adataprovider.cpp).
- [ ] Sanity checks: at rest the rotated accelerometer magnitude is ≈ 9.81 m/s² and points along
      body "up" for the current tilt; a rotation about the body z-axis shows up on gyro z only.

### Issue 7 — Offline timing

- [x] Remove the hard-coded +15 ms IMU timestamp shift
      ([`adataprovider.cpp:381`](../cpp/src/dataproviders/adataprovider.cpp#L381)).
- [x] Apply `dt_imu_cam` from the yaml in the offline grabber (today it is ignored offline), with
      one sign convention shared with the ROS reader.
- [x] Keep the image timestamp for image frames; grouped frames currently take the IMU timestamp
      ([`adataprovider.cpp:448`](../cpp/src/dataproviders/adataprovider.cpp#L448),
      [`:556`](../cpp/src/dataproviders/adataprovider.cpp#L556)).
- [x] Stop discarding images that have no IMU sample within ±2.5 ms
      ([`adataprovider.cpp:405`](../cpp/src/dataproviders/adataprovider.cpp#L405)); at 100 Hz IMU
      (AirSim) many images are dropped. Instead, integrate to the image time with a documented
      interpolation or hold policy.
- [x] Define startup and end-of-stream handling.

### Issue 8 — ROS reader does not merge by timestamp

[`sensorSubscriber.h:76`](../ros/src/sensorSubscriber.h#L76) pops one image and one IMU message
per loop without ordering them by time. With normal image latency, frames come out of order and
`processIMU` silently rejects IMU samples. A failed IMU stays attached to its frame; if that frame
becomes a KF, an unprocessed IMU (uninitialized deltas, zero covariance) reaches the optimizer and
`cov.inverse()` gives inf/NaN.

- [x] Process whichever buffered message (image or IMU) has the older timestamp, and only once the
      other buffer has a newer message (standard VINS-style sync).
- [x] Apply the same fix in `SensorSubscriberCompressed`.
- [x] Guard: never let a KF keep an IMU whose `processIMU()` returned false; create a fresh KF IMU
      from the last valid sample instead.
- [x] Access both queues only under their mutexes (image `empty()` in the mono branch and IMU
      `empty()`/`front()` are read outside the lock).

### Issue 9 — Initialization handoff (replaces the earlier "stale IMU-only frames" item)

The earlier claim was wrong: with ordered input, initialization ends right after the 10th KF is
inserted, so `_last_IMU` belongs to a KF that `VIInit` does transform, and `processIMU` refreshes
the previous sample's cached pose from its frame. The earlier proposed fix (rotate and scale all of
`_6d_velocity`) was also wrong: it is a relative motion, so a global rotation cancels.

- [x] Test the handoff explicitly: ordered and out-of-order input, and image-only prediction that
      reads the cache before a fresh IMU propagation.
- [x] Mono: after `VIInit`, scale only the translational part of `_6d_velocity` (or recompute it from
      the corrected states).

### Issue 10 — Zero rotation gives NaN in the ESKF update (review R1)

[`so3_leftJacobian`](../cpp/include/utilities/geometry.h#L57) divides by `|w|` with no zero guard
(`so3_rightJacobian` has one). The ESKF calls it on the visual/IMU rotation mismatch
([`ESKFEstimator.cpp:144`](../cpp/src/estimator/ESKFEstimator.cpp#L144)), used by both pipelines
([mono](../cpp/src/slamMonoVIO.cpp#L485), [stereo](../cpp/src/slamBiMonoVIO.cpp#L480)).
`log_so3` returns an exact zero only for an exactly symmetric mismatch, so this is rare on real
data but possible with noise-free simulation.

- [x] Add the small-angle limit (identity plus Taylor terms).
- [x] Test exact agreement and small positive and negative perturbations; check the ESKF state and
      covariance stay finite.

### Issue 11 — ESKF velocity correction rotated the wrong way (review R2)

[`ESKFEstimator.cpp:121`](../cpp/src/estimator/ESKFEstimator.cpp#L121): the innovation is in KF i's
body axes and the measurement model is `R1ᵀ(v − v1 − g·dt)`, so `H = R1ᵀ` and the gain must map
back with `R1`. The code uses `Jv = R1`, so its gain applies `R1ᵀ`. For a 90° yaw and a unit body-x
innovation, the code moves the velocity by −0.99 along world y instead of +0.99 (independently
reproduced). This runs on every VIO frame that carries an IMU sample and seeds KF velocities and
the IMU dead-reckoning.

- [x] Use `H = R1ᵀ`, `K = Hᵀ (H Hᵀ + C)⁻¹`, `v = v_pred + K · innovation`.
- [x] Test several non-identity orientations and anisotropic covariances.
- [x] Audit the remaining ESKF covariance conventions separately. Found and fixed: the covariance of the
      IMU step (error of `dT`) was reused unchanged for the visual step (error of `T_cam2_cam1`) and
      returned as the covariance of `dT`; it is now carried through with adjoints both ways. The rotation
      update now is one Gauss-Newton step of the MAP cost (`H = Jl⁻¹(e)`, gain `P Hᵀ (H P Hᵀ + C)⁻¹`;
      before: no `Hᵀ`, and a noise Jacobian from another linearization). The `T_cam2_cam1` initialization
      was only right for equal camera extrinsics on both frames (always the case in the pipeline).
      Tests: `ESKFCovarianceTest.*`. The heuristic tuning (P = I, pixel noise 0.1, IMU covariance × 1000)
      is unchanged.

### Issue 12 — `IMUFactor` pose-j rotation Jacobian (review R3)

With `R_fj = R_fj0 · Exp(w)`, the correct block is `−Jr(r)⁻¹ · R_fj0 · Exp(w) · Jr(w)`.
[`residuals.hpp:207-209`](../cpp/include/isaeslam/optimizers/residuals.hpp#L207-L209) uses
`Exp(w)ᵀ`. Both agree at `w = 0`, which hides the error; at a nonzero increment the error was 0.39
(vs 1e-10 for the correct form) in an independent finite-difference test. The bug came in with
upstream commit `291a492` (2025-10-09); the first-commit version (`T_fj_w.rotation()`) was correct.

- [x] Revert this block to `T_fj_w.rotation()` (equivalently `R_fj0 · Exp(w)`).
- [x] Test every `IMUFactor` block at nonzero increments, including bias increments.

### Issue 13 — `IMUPriordx` velocity and bias Jacobians not whitened (review R4)

[`residuals.hpp:713-728`](../cpp/include/isaeslam/optimizers/residuals.hpp#L713-L728): the residual
and the pose Jacobian are multiplied by `_sqrt_inf`, but the velocity, accel-bias and gyro-bias
Jacobians are bare identity blocks. Only active with sparsification.

- [x] Multiply every block by `_sqrt_inf` (it is dense, so scaling diagonal entries is not enough).
- [x] Test with diagonal and dense non-identity square-root information.

### Issue 14 — Sparse-prior recovery uses the wrong Jacobians (review R5)

In `sparsifyVIO` ([`marginalization.cpp:349`](../cpp/src/optimizers/marginalization.cpp#L349)):

- the pose-to-landmark block uses `−R[t]×` (frame translation,
  [line 362](../cpp/src/optimizers/marginalization.cpp#L362)) where `PoseToLandmarkFactor` gives
  `−R[p]×` (landmark position);
- the absolute-pose block is built as `[R R; 0 R]`
  ([lines 380-384](../cpp/src/optimizers/marginalization.cpp#L380-L384)) where the `IMUPriordx`
  residual gives `[R 0; R[Rᵀt]× R]`.

Both were confirmed by independent finite differences (errors 4.8 and 1.6 vs ~1e-10). Only active
with sparsification.

- [x] Use the Jacobians of the actual unweighted residuals at marginalization time.
- [x] Compare them numerically before checking the recovered information matrices.
- [x] Compare dense vs sparse priors on a small synthetic problem, including gauge/rank handling
      (`MarginalizationSparseKLDTest.*`: a dense prior with the sparse structure is recovered exactly;
      for a generic dense prior, scaling any sparse factor's information by ±5 % increases the KLD;
      both tests fail on a 10 % information error).

### Issue 15 — Sparse prior gated on the number of landmark types (review R6)

[`BundleAdjustmentCERESAnalytic.cpp:538`](../cpp/src/optimizers/BundleAdjustmentCERESAnalytic.cpp#L538)
requires `_lmk_to_keep.size() > 1`, but that container maps landmark *type* to landmarks, so a
point-only map skips the whole sparse prior. The `AngularAnalytic` optimizer uses `empty()` instead.

- [x] Replace the type-count check with explicit checks on the retained variables and factors.
- [x] Test a point-only map, and a retained inertial state with no retained landmarks.

### Issue 16 — Mono VIO discards the oldest KF without marginalizing it (review R7)

[`slamMonoVIO.cpp:607-609`](../cpp/src/slamMonoVIO.cpp#L607-L609) moves the oldest KF to the global
map and discards it without calling `marginalize()`, whatever the `marginalization` flag says;
stereo calls it when configured. The shared `preMarginalize()` filters for two-camera
observations, so enabling mono marginalization needs deliberate rank handling.

- [x] Implement mono marginalization with landmark selection suited to monocular observations.
- [x] Test oldest-frame removal separately from the low-parallax policy (Issue 5).

### Issue 17 — The Jacobian tests cannot fail on a Jacobian error (review R8)

[`imu_test.cpp:453`](../cpp/tests/imu_test.cpp#L453): a failed `GradientChecker::Probe()` is only
logged. The assertions compare `local_jacobians` with `jacobians`, which are both analytic and
identical when no manifold is supplied. The init check at
[`imu_test.cpp:515-526`](../cpp/tests/imu_test.cpp#L515-L526) also asserts on `results` instead of
`results1`, and `λ = 0` hides Issue 3. `residual_test.cpp` has similar comparisons.

- [x] Assert that `Probe()` succeeds and print its error log on failure.
- [x] Compare analytic against `numeric_jacobians` with a max-abs norm (not a signed sum, which can
      cancel).
- [x] Probe at zero and nonzero increments, non-unit scale, nonzero gravity correction, nonzero
      bias increments, and non-identity whitening.
- [x] Add the finite-difference counterexamples from Issues 11, 12 and 14 as regression tests
      (`ESKFVelocityTest.updateVelocityFrames`, `ImuTest.imuFactorJacobiansAwayFromZero`,
      `MarginalizationJacobianTest.recoveryJacobiansMatchTheFactors`).

### Issue 18 — Validity checks and time gaps (review R9)

- `VIInit` and `localMapVIOptimization` apply solver results without checking
  `summary.IsSolutionUsable()` ([`AOptimizer.cpp:337`](../cpp/src/optimizers/AOptimizer.cpp#L337),
  [`:475`](../cpp/src/optimizers/AOptimizer.cpp#L475)).
- The init assumes the robot is static for the first samples and starts inertial alignment after
  10 KFs regardless of how much motion information they contain.
- [`IMU.cpp:29`](../cpp/src/data/sensors/IMU.cpp#L29) replaces any sample gap over 1 s by
  `1/rate`, while the factor still uses the real timestamps, so the integrated duration and the
  factor duration disagree.

- [x] Check stationarity before using the static bias estimate.
- [x] Accept init only on a usable solver result, finite states, plausible scale and biases, and
      enough motion; otherwise retry.
- [x] Define handling for duplicate, decreasing, missing and widely separated timestamps.
- [x] Track the integrated duration and compare it with the factor duration.
- [x] Check covariance finiteness and factorization success before building an IMU factor.

### Issue 19 — Online time-offset estimation (`estimate_td: 1`)

Not yet reviewed in detail. The estimated offset is accumulated into `dt_imu_cam`
([`slamBiMonoVIO.cpp:678`](../cpp/src/slamBiMonoVIO.cpp#L678), same in mono), which only the ROS
reader uses, so offline runs estimate an offset but never apply it.

- [x] Review `localMapVIOptimizationTd` in both analytic optimizers and the
      `AngularErrCeres_pointxd_td` residual (model and Jacobians, with the Issue 17 test method).
- [x] Make the offline reader apply the estimated offset (Issue 7 shares the sign convention).
- [x] Check that accumulating `td` into `dt_imu_cam` is consistent with states and preintegrations
      already built with the old offset. It was not: every window re-added the same correction. Each
      frame now stores the offset it was read with, the window estimates the absolute offset (residuals
      shift by `td − td_built`) and replaces `dt_imu_cam` (VINS-Mono style). `dt_imu_cam` is atomic
      (written by the back end, read by the data thread).
- [x] While debugging other issues, use `estimate_td: 0` with a known fixed offset, then validate
      `estimate_td: 1` on its own.

### Issue 20 — Option combinations that silently do nothing

`optimizer: Numeric` (`BundleAdjustmentCERESNumeric`) does not override `marginalize()`,
`marginalizeRelative()` or `localMapVIOptimizationTd()`, so it inherits the empty defaults in
[`AOptimizer.h:79-129`](../cpp/include/isaeslam/optimizers/AOptimizer.h#L79-L129):

- with `marginalization: 1`, the oldest KF is dropped with no prior (the flag is silently ignored);
- with `estimate_td: 1`, the back end calls `localMapVIOptimizationTd` whenever the KF rotation
  exceeds 0.05 rad, and that stub returns without optimizing anything, so those KFs get **no
  window optimization at all**.

- [x] Implement these for `Numeric`, or reject unsupported combinations at config load with a clear
      error (`slamParameters.cpp`).
- [x] Add a config-validation step that rejects or warns on every unsupported combination of
      `slam_mode`, `optimizer`, `marginalization`, `sparsification` and `estimate_td`.
- [x] Add a smoke test per supported combination (a short synthetic or EuRoC segment) that checks
      the expected factors are actually inserted
      (`ImuTest.everySupportedOptionCombinationInsertsItsFactors`: mono/stereo × 3 optimizers ×
      marginalization × sparsification × estimate_td on a synthetic window; checks the IMU, visual and
      prior factor counts and recovery of the true states). End to end: the option matrix on TUM-VI
      room1 (ledger, Phase 8).

### Minor

- [x] `Prior1D` never initializes its `_prior` member
      ([`residuals.hpp:745`](../cpp/include/isaeslam/optimizers/residuals.hpp#L745)); used only by
      the non-overlapping-FOV code.
- [x] `shouldInsertKeyframe` counts landmark matches twice in the parallax denominator
      ([`slamCore.cpp:386`](../cpp/src/slamCore.cpp#L386), [`:404`](../cpp/src/slamCore.cpp#L404)),
      so parallax is underestimated.
- [x] Mono init: `_last_IMU = imu_kf` is set before `imu_kf->setLastIMU(_last_IMU)`, so the KF IMU
      points to itself ([`slamMonoVIO.cpp:51`](../cpp/src/slamMonoVIO.cpp#L51)).

### Found while running the option matrix

Running every option one at a time (TUM-VI room1, all four modes) exposed failures that the default
config never reaches:

- [x] `optimizer: Analytic` / `Numeric` with a double-sphere camera: `DoubleSphere::project` with
      Jacobians was a stub returning `false`, so those optimizers had **no visual factors** (stereo VIO
      ATE 22 m and 521 m). Implemented (`ResidualTest.reprojTestDoubleSphere`).
- [x] `marginalization: 1` with stereo VIO crashed (Ceres: duplicate parameter blocks): the same
      landmark was added twice to a frame. `Frame::addLandmark` and `preMarginalize` now deduplicate.
- [x] Mono VIO + marginalization (+ sparsification): the init KF was added to the window twice (init
      and again by the back end), so `marginalize(f, f)` indexed outside the prior. The init KF is no
      longer re-sent; `LocalMap::addFrame` ignores a frame already in the window; `marginalize` refuses
      `frame0 == frame1`.
- [x] Mono VO + marginalization crashed (`_map_frame_idx.at`): the `AngularAnalytic` marginalization
      added frame 1's pose prior, but frame 1 is not a marginalization variable in VO, and in VIO the
      prior stayed in the window too (counted twice). Removed (the `Analytic` optimizer never had it).
- [x] `multithreading: 1` crashed (heap corruption): front end and back end modified the same landmarks
      without synchronization, and handed `_frame_to_optim` over with a spin loop on a plain
      `shared_ptr`. Each thread now holds a step mutex for a whole frame / KF step; the front end
      releases it while it waits for data or for the back end (condition variable).
- [x] `estimate_td: 1`: `dt_imu_cam` is written by the back end and read by the data thread; now atomic.
- [x] Marginalization prior residual had the wrong sign (`r_p = −Λ^{-1/2}Uᵀb` with `b = +Jᵀr`): every prior
      pushed the kept states away from what the marginalized measurements say. Found while diagnosing
      stereo VIO divergences on TUM-VI magistrale2 with `marginalization: 1`.
- [x] The marginalization prior had no linearization point: reused priors measured the kept states from
      the start of each window, and the sparse inertial prior was centred on the current state every
      window. The prior now records its linearization point (`ImuTest.marginalizationPriorKeepsIts
      LinearizationPoint`).
- [x] The front end's structure-only landmark refinement moved landmarks tied to the prior (ignoring it);
      they are now held constant there.
- [x] The VIO window held the oldest pose fixed although the prior constrains it: the prior's pull on it
      was never resolved and was folded into every following prior (prior cost 12 → 6,000 over room1, then
      failure). With a VIO prior no frame is fixed; the window is moved back afterwards so that the oldest
      frame keeps its yaw and position (VINS-Mono style gauge, `restoreGauge`).
- [x] No robust loss in the VIO window or its marginalization: bad associations entered the prior at full
      weight. Visual factors now use the front end's Huber loss in both (Triggs correction in the
      marginalization blocks, `MarginalizationRobustTest`). The AngularAnalytic marginalization also used
      1 px instead of the window's 1.5 px (2.25× overconfident prior): unified (`kAngularSigmaPx`).
- [x] Distant or badly triangulated landmarks (half of the runaway ones were behind the camera when
      linearized) were kept in the prior, whose linear model breaks down when their depth slides. Only
      well-conditioned landmarks are kept now (`Marginalization::wellConditioned`).
- [x] With `multithreading: 1`, the front end could process a frame before the back end had added the last
      voted KF to the window, anchoring its IMU preintegration to the KF before; as a KF its IMU factor skipped
      a window KF (single-KF jumps, 8 of 10 room1 runs with marginalization). The front end now waits for the
      back end before processing the next frame (`SLAMCore::nextFrame`).
- [x] Memory grew through the run (each IMU sample kept the previous one alive, ≈ 0.5 MB/s on magistrale2):
      the history is released when a KF leaves the window.

**Result (TUM-VI, 3 runs):** stereo VIO with `marginalization: 1` and `estimate_td: 1` — room1 ATE 0.069 m
(VO 0.145, default VIO 0.266), magistrale2 ATE 1.47 m / end drift 3.2 m (VO 6.96 / 14.8, default VIO
8.46 / 18.6), no resets. Details: `doc/vio_imu_fix/IMU_FIX_LEDGER.md`.

### Fix order and validation

- [x] 1. **Reproduce and make validation trustworthy.** Record the exact commit, config, reader,
      calibration, topics and failure onset. Fix the tests (Issue 17) and add the counterexamples.
      Log solver usability, non-finite states, rejected IMU samples, interval durations and inserted
      factor counts.
- [x] 2. **Direct numerical failures and inputs.** Issues 10, 11, 1, 7, 8, the verified part of 6,
      the time-gap part of 18, and the config validation of Issue 20. Use a fixed time offset
      while doing this (Issue 19).
- [x] 3. **Initialization as one coherent model.** Issues 2, 3, 9 and the init part of 18. Validate
      on synthetic stationary, biased, accelerating and rotating cases, including mono scale.
- [x] 4. **Normal factors and bias bookkeeping.** Issues 12 and 4. Single-threaded first.
- [x] 5. **Window information.** Issue 5 and Issue 16; validate the dense-prior path with
      sparsification off.
- [x] 6. **Sparsification separately.** Issues 13, 14 and 15; compare against the dense-prior
      baseline for accuracy and run time.
- [ ] 7. **Datasets.** EuRoC MH_01 / V1_01 first (paper Table I: SaDVIO 0.09 m / 0.06 m ATE as a
      comparison target, not a pass/fail threshold), then RealSense and ExECoSim. Evaluate mono and
      stereo separately; record completion, ATE/RPE, scale, bias and velocity behaviour, run time.
      Do not let scale-aligned scoring hide a metric-scale failure.
      *Status: EuRoC could not be downloaded (server down); TUM-VI room1 and magistrale2 were used
      instead. RealSense waits for its camera-IMU calibration (Issue 6); ExECoSim not run.*
- [x] 8. **Option matrix.** Fix Issue 19 and the rest of Issue 20, then run the same sequences
      across every supported combination: `mono`/`bimono` × `Analytic`/`Numeric`/`AngularAnalytic`
      × `marginalization` 0/1 × `sparsification` 0/1 × `estimate_td` 0/1 × `multithreading` 0/1,
      on both readers. Vary one option at a time from a validated baseline so each regression has
      one cause.
- [ ] 9. Once every supported combination passes, freeze SaDVIO (tag the commit) before starting
      SaDLIO.

---

## Part 3 — What "Sparsify" and "Densify" mean in this project

### S — Sparsify (back end)

SaDVIO optimizes a sliding window of the last N KFs and their landmarks. When the oldest KF leaves,
it is **marginalized**: its information is folded into a prior on the variables it was connected to.
That prior links every remaining variable to every other one, which makes the problem dense and
slow. **Sparsification** replaces it with a few simple factors that carry almost the same
information, chosen with a KL-divergence criterion
([`marginalization.cpp:331`](../cpp/src/optimizers/marginalization.cpp#L331)):

- **VO** (`sparsifyVO`): a Chow-Liu tree of landmarks. One landmark gets an absolute position factor
  (`Landmark3DPrior`); the others get relative factors to a neighbour (`LandmarkToLandmarkFactor`).
- **VIO** (`sparsifyVIO`, [`marginalization.cpp:349`](../cpp/src/optimizers/marginalization.cpp#L349)):
  one absolute factor on the kept KF's full state (pose, velocity, both biases; `IMUPriordx`, paper
  eq. 4) plus a pose-to-landmark factor per landmark (`PoseToLandmarkFactor`, eq. 5).

The sparse factors are used in the optimization, but the **dense prior is kept and used for the
next marginalization** (paper §III-A.3; computed in every case at
[`AngularAdjustmentCERESAnalytic.cpp:906`](../cpp/src/optimizers/AngularAdjustmentCERESAnalytic.cpp#L906)).
This is part of how information propagates, not an implementation detail. Matching the eq. 4-5
topology does not validate the implementation (Issues 13-15) or guarantee negligible information
loss on every dataset. Enabled with the `sparsification` flag.

### D — Densify (map)

- **Paper version** (mesh thread, [`mesher.cpp`](../cpp/src/data/mesh/mesher.cpp), `mesh3D` flag):
  2D Delaunay on the tracked landmarks in the left image → for each triangle, warp a 15×15 patch
  into the right image assuming a planar surface and keep it only if ZNCC is high enough (removes
  triangles over depth jumps) → ray-cast every pixel into the resulting triangle soup to get a
  dense point cloud, **without** running dense stereo.
- **This branch** (`MarginalDepthInjector`): when a KF leaves the window, run real dense stereo
  (SGBM, FFS or LAS2) and build a surface with ZNCC-Delaunay, the SLAMesh-style GP mesh (plus
  optional global GP map), primal-dual, or VDB-GPDF. The paper's "D" avoids dense stereo to save
  compute; this branch uses it with the aim of better maps (a design goal, not a measured result).
  The dense job is queued when the KF leaves the window, before and independently of the optional
  probabilistic marginalization, so "queued for dense" does not imply "marginalized".

---

## Part 4 — SaDLIO prospects (LiDAR-inertial, no visual information)

This part is a set of research hypotheses, separate from the verified VIO defect list above.

Neither idea is tied to cameras. Sparsify works for any sensor that produces keyframes plus
landmarks. Densify turns sparse 3D points into a surface; a LiDAR scan is sparse too (16-128 rings,
large gaps on the ground far from the robot). What changes is the landmark type and the check that
replaces ZNCC.

### Component mapping

| SaDVIO | SaDLIO equivalent |
|---|---|
| Visual point landmark | Small **plane/surfel landmark**, tracked across scans |
| Reprojection factor | **Point-to-plane factor** (KF ↔ plane landmark) |
| IMU preintegration factor | Same (after the fixes) |
| Scan deskewing | New: needs a within-scan trajectory (IMU propagation at each point's time), not just the KF-to-KF preintegrated delta |
| VIO init (vision-IMU alignment, scale) | LiDAR removes the scale ambiguity but not gravity/bias ambiguity: static init needs a stationary interval, dynamic init needs enough motion and geometry |
| Sparsified prior: absolute KF + pose-to-landmark | Absolute KF + **pose-to-plane** factors |
| Delaunay on image features | Delaunay on the **LiDAR range image** (rings × azimuth) |
| ZNCC left/right check | **Geometric or temporal** consistency check (see D1) |
| Ray casting from the left camera | Ray casting from the LiDAR at finer vertical resolution |

### Options for S

- **S1 — small plane/surfel landmarks (recommended to investigate first).** Voxels of roughly
  0.5-1 m (to be tuned), a plane per voxel, tracked as landmarks (VoxelMap / Voxel-SLAM / MSPA-LIO
  style). Marginalize with the SaDVIO VIO topology: absolute KF factor + pose-to-plane factors.
  This is **not** a drop-in change to `sparsifyVIO`: the plane parameterization, normal constraints,
  data association, residual dimensions, rank handling, covariance recovery and degeneracy all need
  designing, and the current code assumes point landmarks and stereo information. Rough terrain
  yields many small planes, which is where sparsification could pay off most. Large-plane systems
  such as π-LSAM were designed for indoor scenes; whether they suit rough terrain is untested here.
- **S2 — no landmarks in the state (BALM-style).** Plane parameters are solved analytically and
  removed. Eliminating a plane seen by several poses creates a joint constraint on all of them, not
  pairwise factors; turning that into a chain or tree of relative-pose factors is itself an
  approximation that needs its own information-loss analysis.
- **S3 — filter (FAST-LIO2 style).** Nothing to sparsify; baseline only.

### Options for D

- **D1 — range-image mesh with a consistency check (closest to the paper).** 2D Delaunay on the
  range image, keep triangles that pass a check, then ray-cast. Gingras et al. (cited by the paper)
  already triangulated LiDAR scans in spherical coordinates for rover planning; the new part is the
  SaD check plus ray casting. Checks:
  - geometric: maximum edge length, normal agreement, point-to-plane distance of the raw points
    inside the triangle (rejects triangles bridging a rock and the ground behind it);
  - temporal: project the triangle into another scan and check the ranges agree;
  - temporal reflectivity ZNCC: compare Ouster reflectivity patches between **two scans** through
    the triangle's plane. A single scan's reflectivity image has no second viewpoint, so it cannot
    play the role of the paper's left/right check. Needs visibility handling, motion compensation
    and a reflectivity model (reflectivity varies with incidence angle and range).
  - Ray-cast samples are inferred geometry, not measurements: evaluate uncertainty, unsupported
    gaps, occlusion boundaries and whether small obstacles survive, not just point count.
- **D2 — mesh from the plane landmarks** (if S1): stitch or clip the window's plane patches into a
  compact mesh.
- **D3 — implicit fields (TSDF, GP, VDB-GPDF), SLAMesh, ImMesh.** Already partly available here
  (VDB-GPDF `lidar` preset, best result in [`dense_vdb_gpdf.md`](dense_vdb_gpdf.md)). Use as
  baselines.

### Reuse from this codebase

- Reuse (after the fixes): IMU preintegration and factors, Ceres sliding-window machinery,
  `LocalMap`, marginalization (KL divergence, rank-revealing decomposition), `ALandmark` hierarchy
  (`Plane3D` next to `Line3D`).
- New: LiDAR sensor type and point-cloud input, per-point timestamps and deskewing, LiDAR-IMU
  extrinsics, voxel/plane extraction and tracking, point-to-plane residual and its sparsification
  Jacobians, range-image mesher.

### Plan

- [ ] Create a new repo for SaDLIO that forks or shares the core modules; keep SaDVIO frozen.
- [ ] Do a proper literature review on (a) sparsification of LiDAR plane-landmark windows and
      (b) consistency-checked range-image meshing with ray casting. A short search found neither,
      so these may be contributions, but novelty is **not** established. Existing work already
      covers plane tracking and LiDAR-IMU degeneracy (e.g. LIC-Fusion 2.0).
- [ ] Inspect the candidate datasets before choosing: the paper's Mars Terrain dataset has an
      Ouster OS128, and ExECoSim captures have LiDAR. Check which channels (reflectivity),
      per-point timestamps, calibrations and synchronized ground truth a given recording provides.
- [ ] v0: IMU + deskewed scans + S1 plane landmarks + dense marginalization. Compare against
      FAST-LIO2, LIO-SAM and Voxel-SLAM (ATE/RPE).
- [ ] v1: turn on sparsification (S1 topology); measure back-end time saved and accuracy lost, as in
      the paper.
- [ ] v2: D1 with the geometric check, then the temporal checks; compare against VDB-GPDF, SLAMesh
      and ImMesh (accuracy, completeness, small-obstacle preservation, traversability cost map, run
      time).
- [ ] Known pitfalls: per-point timestamps and deskewing, LiDAR-IMU extrinsic calibration, LiDAR
      returns on the robot's own body, degenerate scenes (flat ground, corridors).

### Sources

- [LIC-Fusion 2.0](https://arxiv.org/pdf/2008.07196) — sliding-window plane-feature tracking
- [Voxel-SLAM](https://arxiv.org/html/2410.08935v1)
- [MSPA-LIO](https://www.researchgate.net/publication/398060895_MSPA-LIO_LiDAR-Inertial_Odometry_with_Multi-Scale_Plane_Adjustment)
- [LMBAO](https://arxiv.org/pdf/2209.08810)
- [LiDAR-Inertial SLAM with Efficiently Extracted Planes](https://april.zju.edu.cn/wp-content/papercite-data/pdf/chen2023lidar.pdf)
- [π-LSAM](https://ieeexplore.ieee.org/document/9561933/)
- [BALM](https://arxiv.org/pdf/2010.08215)
- [LiLi-OM](https://arxiv.org/pdf/2010.13150)
- [Rough Terrain Reconstruction for Rover Motion Planning (Gingras et al.)](https://www.researchgate.net/publication/224143633_Rough_Terrain_Reconstruction_for_Rover_Motion_Planning)
- [Autonomous over-the-horizon navigation using LIDAR data](https://link.springer.com/article/10.1007/s10514-012-9309-9)
- [Traversable region detection with range images](https://doi.org/10.3390/s23135898)
- [ImMesh](https://arxiv.org/pdf/2301.05206)
- [SLAMesh](https://github.com/RuanJY/SLAMesh)
- [Forster et al., On-Manifold Preintegration](https://arxiv.org/abs/1512.02363)
- [ORB-SLAM3 initialization edges (`G2oTypes.cc`)](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/G2oTypes.cc)
- [RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros)
