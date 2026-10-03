# Review of “VIO fix + LIO prospects”

Date: 2026-10-02 · Branch: `dense_devel` · Reviewed code: `f157380`

Document reviewed: [VIO_fix_+_LIO_Prospects.md](VIO_fix_+_LIO_Prospects.md), dated 2026-10-01.
Primary reference: [SaDVIO paper supplied with this repository](../RAL_2024___SaDVIO.pdf).
Status: review only; no implementation fixes or edits to the original document were made.

The original document identifies several genuine defects, particularly in bias configuration,
initialization, input timing, and keyframe handling. Its conclusion that the main IMU-factor
Jacobians and sparse VIO prior need no further investigation is incorrect. Additional defects
exist in those implementations and in the ESKF front end used by both VIO pipelines.

This review distinguishes a defect visible in the source from a demonstrated cause of a failed
trajectory. No failing recording was replayed, and no full C++ build or test suite was run.
Independent NumPy calculations checked selected formulas by finite differences during the review.
Those checks are evidence about the equations, not execution of the C++ factor classes.
The supplied PDF was used directly; locating a separate Markdown copy was unnecessary.

## 1. Scope and applicability

The audit covered IMU propagation, initialization and normal inertial factors, the ESKF update,
mono/stereo VIO state handoffs, ROS and offline readers, marginalization, sparse-prior recovery,
and the relevant existing tests. It also compared the document's descriptions of sparsification,
densification, and prospective LiDAR reuse with the paper and inspected implementation.

The checked-in [configuration](../ros/config/config.yaml#L6) selects `airsim`, `slam_mode: mono`,
`AngularAnalytic`, `multithreading: 0`, `estimate_td: 1`, `marginalization: 0`, and
`sparsification: 0`. This is not evidence of the settings used in the reported failing VIO runs.
Sparse-prior defects cannot explain a run in which those factors are never enabled. Threading
concerns similarly depend on the runtime execution path.

“Confirmed” below means the defect or behavior is established by source inspection, with numerical
evidence where stated. It does not establish its contribution to the user's observed failure.
Source line references refer to `f157380` and may move after fixes.

## 2. Assessment of the original issue list

| Original item | Assessment | Evidence and qualification |
|---|---|---|
| Issue 1: gyro random-walk key | Confirmed | [adataprovider.cpp:85](../cpp/src/dataproviders/adataprovider.cpp#L85) reads `accelerometer_random_walk` into `bgyr_noise`. The EuRoC YAML values imply about 51.6× the intended standard deviation and 2,659× the variance. |
| Issue 2: initialization bias application | Confirmed | [AOptimizer.cpp:477](../cpp/src/optimizers/AOptimizer.cpp#L477) applies the accelerometer increment twice and never applies the gyro increment. `VIInit` also omits corresponding preintegration correction. The duration in the bias prior is in nanoseconds. See Section 4 before changing the prior strength. |
| Issue 3: initialization residual and scale derivative | Confirmed, with a convention clarification | [residuals.hpp:350](../cpp/include/isaeslam/optimizers/residuals.hpp#L350) rotates the visual displacement inconsistently with the map transformation applied afterward. The scale derivative at line 418 omits `exp(lambda)`. The correction must follow this implementation's velocity convention, rather than copying another estimator verbatim. |
| Issue 4: bias linearization mismatch | Confirmed | Propagation, optimization, and `updateBiases()` can leave biases and preintegrated quantities inconsistent. Explicit linearization biases are appropriate, but need a coherent correction/reintegration policy. |
| Issue 5: low-parallax keyframe removal | Confirmed | Both VIO back ends remove an intermediate keyframe and clear the new frame's IMU predecessor. Marginalization constructs factors without verifying the corresponding preintegration interval. Mono additionally lacks the normal marginalization call; see R7. |
| Issue 6: RealSense frames | Credible configuration concern; diagnosis needs recording verification | The inspected `realsense_optitrack*.yaml` files rotate cameras into forward-left-up axes and leave IMU rotation at identity. The proposed extrinsic-translation fix is incomplete; see Section 4. The separately named `sensor_config_realsense_d435i_sadvio.yaml` was not independently inspected in this repository. |
| Issue 7: offline timing | Confirmed | [adataprovider.cpp:381](../cpp/src/dataproviders/adataprovider.cpp#L381) adds 15 ms unconditionally. The offline path ignores configured `dt_imu_cam` and drops images outside its IMU grouping tolerance. Grouped images also inherit the IMU timestamp. |
| Issue 8: ROS ordering | Confirmed | [sensorSubscriber.h:76](../ros/src/sensorSubscriber.h#L76) processes image and IMU queues without a chronological merge; the compressed reader has the same structure. Failure to process an attached IMU does not remove it from the frame. |
| Issue 9: frames left behind after initialization | Not established as a general failure scenario | Under normal ordered input, initialization finishes when the tenth keyframe is inserted. Newer IMU-only frames do not necessarily exist. Cached-state refresh deserves a test, but rotating all of `_6d_velocity` is the wrong fix. |
| Minor: `Prior1D` | Confirmed | [residuals.hpp:745](../cpp/include/isaeslam/optimizers/residuals.hpp#L745) does not initialize its `_prior` member. This is not a primary explanation for the standard mono/stereo VIO paths. |
| Minor: parallax denominator | Confirmed | [slamCore.cpp:386](../cpp/src/slamCore.cpp#L386) includes landmark matches in `n_matches`, then counts them again in the denominator at line 404. |
| Minor: mono initial IMU self-reference | Confirmed | [slamMonoVIO.cpp:51](../cpp/src/slamMonoVIO.cpp#L51) assigns `_last_IMU = imu_kf` before setting `imu_kf`'s previous IMU. It creates a self-reference. |

Historical claims such as “also present upstream,” the exact effect of commit `291a492`, and why
the authors introduced the 15 ms shift were not independently established by this review.
They should remain separate from findings about the current checkout.

## 3. Important defects and gaps omitted from the document

### R1 — Zero rotation can produce NaNs in the ESKF update

**Applicability:** mono and stereo VIO when the ESKF IMU-update branch is entered.

[geometry.h:57](../cpp/include/utilities/geometry.h#L57) implements `so3_leftJacobian(w)` using
`w / ||w||`, `sin(||w||) / ||w||`, and related divisions, without handling zero. Its mathematical
limit at zero is the identity matrix; the implemented expression instead produces nonfinite values.

[ESKFEstimator.cpp:142](../cpp/src/estimator/ESKFEstimator.cpp#L142) evaluates this Jacobian on the
rotation disagreement between `dT` and the IMU delta. Exact agreement therefore creates an invalid
innovation covariance and can contaminate the pose update. Both pipelines call this estimator:
[mono](../cpp/src/slamMonoVIO.cpp#L478) and [stereo](../cpp/src/slamBiMonoVIO.cpp#L479).

The zero-input expression was reproduced numerically and was nonfinite. This is a concrete
failure case, particularly relevant to stationary or ideal simulated measurements; it is not a
claim that every stationary recording necessarily reaches exact zero in floating-point arithmetic.

- [ ] Implement the zero/small-angle limit, with an appropriate Taylor expansion.
- [ ] Test exact visual/IMU rotation agreement and small positive and negative perturbations.
- [ ] Check that the complete ESKF update preserves finite state and covariance in those cases.

### R2 — The ESKF velocity innovation is rotated in the wrong direction

**Applicability:** the same IMU-update branch as R1; error depends on orientation and innovation.

In [ESKFEstimator.cpp:118](../cpp/src/estimator/ESKFEstimator.cpp#L118), `R1` maps body coordinates
to world coordinates. The innovation at line 126 is in the first body's coordinates, but the gain
uses `Jv = R1` and then `Jv.transpose()`. The measurement Jacobian with respect to world velocity
is `R1.transpose()`, so the code's gain maps the innovation in the wrong direction.

Under the implementation's identity velocity-prior covariance, the consistent expression is:

```text
H = R1^T
K = H^T (H H^T + C_delta_v)^-1
v_updated = v_predicted + K * innovation_body
```

For a 90° world yaw, a unit body-x innovation, and `C_delta_v = 0.01 I`, the current expression
gives approximately `(0, -0.990099, 0)` in world coordinates. The consistent expression gives
`(0, +0.990099, 0)`.

- [ ] Correct the frame convention in the velocity measurement Jacobian and gain.
- [ ] Test several nonidentity orientations and anisotropic covariances.
- [ ] Audit the remaining ESKF covariance conventions separately; this correction alone does not
      establish that the entire ESKF is statistically consistent.

### R3 — `IMUFactor` has an incorrect pose-j rotation Jacobian

**Applicability:** ordinary IMU-factor optimization at nonzero pose-j rotation increments.

The factor uses `R_j = R_j_base Exp(w_j)`. Its rotation residual is
`r_R = Log(Delta_R_corrected^T R_i R_j^T)`. Under the implemented additive rotation-vector
parameterization, the derivative is:

```text
dr_R / dw_j = -Jr(r_R)^-1 R_j_base Exp(w_j) Jr(w_j)
```

However, [residuals.hpp:207](../cpp/include/isaeslam/optimizers/residuals.hpp#L207) uses
`Exp(w_j).transpose()` in that expression. The two forms agree at `w_j = 0`, concealing the error
in tests restricted to the initial parameter values.

Independent finite differences at a nonzero increment gave maximum absolute derivative error
`0.2522523784` for the current formula and approximately `4.04e-11` for the corrected formula.
The check concerned the unweighted rotation residual; a fixed nonsingular whitening transform
does not remove the mismatch.

- [ ] Correct this rotation block.
- [ ] Test every IMU-factor parameter block at nonzero increments, including bias increments.
- [ ] Remove the original document's assertion that every Jacobian block has been verified correct.

### R4 — `IMUPriordx` omits whitening from velocity and bias Jacobians

**Applicability:** sparse VIO priors, when inserted into the optimization problem.

[residuals.hpp:690](../cpp/include/isaeslam/optimizers/residuals.hpp#L690) multiplies the complete
15-dimensional residual by `_sqrt_inf`. The pose Jacobian is also multiplied by it, but the
velocity, accelerometer-bias, and gyro-bias Jacobians at lines 713–728 are left as unweighted
identity subblocks.

The Jacobians must be `_sqrt_inf` multiplied by their respective selection matrices. A dense
information square root may couple rows, so scaling just the three diagonal entries is insufficient.
A diagonal square root with entries 1 through 15 already demonstrates an absolute velocity
derivative error of 8.

- [ ] Apply the same whitening to all parameter-block Jacobians.
- [ ] Test with nonidentity diagonal and dense information square roots.

### R5 — Sparse-prior covariance recovery uses incompatible Jacobians

**Applicability:** `sparsifyVIO()` in either analytic optimizer.

There are two separate mismatches in [marginalization.cpp:349](../cpp/src/optimizers/marginalization.cpp#L349).
Let `T_f_w = (R, t)` and let `p` denote the landmark's world position.

For the pose-to-landmark residual `T_f_w p - prior`, right pose increments give, at zero increment:

```text
J_pose = [ -R [p]_x , R ]
J_landmark = R
```

The recovery code instead uses `-R [t]_x`, substituting the frame translation for the landmark
position. This disagrees with the implemented [PoseToLandmarkFactor](../cpp/include/isaeslam/optimizers/residuals.hpp#L615).

For the absolute pose residual used by `IMUPriordx`, at the prior state the expected pose block is:

```text
J_pose = [ R                  0 ]
         [ R [R^T t]_x        R ]
```

The recovery code at lines 380–384 constructs `[R, R; 0, R]` instead. It introduces translation
dependence into the rotation residual and omits the actual rotation-to-translation coupling.
Both errors feed directly into the recovered factor information matrices.

Finite differences confirmed these mismatches; results are recorded in Section 7.

- [ ] Make recovery use the Jacobians of the actual unweighted residuals at marginalization time.
- [ ] Compare those Jacobians numerically before checking recovered information matrices.
- [ ] Test dense versus sparse priors on the same small synthetic problem, including gauge/rank
      handling. The formula corrections do not alone validate the full sparsification algorithm.

### R6 — One optimizer gates sparse priors on the number of landmark types

**Applicability:** `BundleAdjustmentCERESAnalytic` with sparsification enabled and a point-only map.

[BundleAdjustmentCERESAnalytic.cpp:538](../cpp/src/optimizers/BundleAdjustmentCERESAnalytic.cpp#L538)
requires `_lmk_to_keep.size() > 1`. This container is an unordered map from a landmark-type label
to a vector of landmarks. Its size counts types, not points. A map containing many `pointxd`
landmarks can therefore skip the complete sparse prior.

The [AngularAnalytic version](../cpp/src/optimizers/AngularAdjustmentCERESAnalytic.cpp#L577) uses
an emptiness check instead. Do not attribute the type-count condition to that optimizer.

- [ ] Replace type-count gating with explicit checks for the retained variables and required factors.
- [ ] Verify factor insertion for a point-only map, and audit handling of a retained inertial state
      with no retained landmarks.

### R7 — Mono VIO discards the oldest keyframe without marginalizing it

**Applicability:** normal mono VIO window removal, independently of the low-parallax branch.

[slamMonoVIO.cpp:607](../cpp/src/slamMonoVIO.cpp#L607) adds the oldest frame to the global map and
calls `discardLastFrame()` without calling the optimizer's `marginalize()`. The corresponding
[stereo path](../cpp/src/slamBiMonoVIO.cpp#L655) calls marginalization when configured.
Moving a frame into a global container does not create a prior on the remaining local states.

Enabling mono marginalization is more than adding the missing call. The shared
[preMarginalize()](../cpp/src/optimizers/marginalization.cpp#L65) includes a two-camera-observation
filter intended to ensure full 3D landmark information. Mono observations have different rank
properties and must be handled deliberately.

- [ ] Implement and validate mono window information propagation.
- [ ] Review landmark selection and rank handling for monocular observations.
- [ ] Test removal of the oldest frame separately from any intermediate-frame removal policy.

### R8 — Existing Jacobian tests can report success without checking numerical agreement

**Applicability:** confidence in the current tests and in the document's mathematical assurances.

In [imu_test.cpp:449](../cpp/tests/imu_test.cpp#L449), a failed `GradientChecker::Probe()` is
logged but not asserted as a test failure. The following assertions compare `local_jacobians`
and `jacobians`: both are analytic derivatives and coincide here because no manifold is supplied.
The numerical results are stored in `numeric_jacobians` and `local_numeric_jacobians`, as documented
in [Ceres's GradientChecker definition](https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/include/ceres/gradient_checker.h).

The initialization check at lines 513–524 additionally tests the previous `results` object instead
of `results1`. Its initial `lambda = 0` also hides the missing exponential in the scale derivative.
Similar analytic-to-analytic comparisons occur in `residual_test.cpp`.

- [ ] Assert successful `Probe()` results and include their error logs in assertion output.
- [ ] If comparing arrays directly, compare against numerical derivatives with a suitable norm;
      do not sum signed errors, which may cancel.
- [ ] Exercise zero and nonzero increments, nonunit scale, nonzero gravity correction, nonzero
      bias increments, and nonidentity whitening.
- [ ] Add initialization model-consistency tests: numerical differentiation can validate the
      derivative of a physically incorrect residual, so derivative tests alone are insufficient.

### R9 — Initialization acceptance and time-gap handling need explicit validity policies

Both initializers estimate gravity/biases from initial measurements under a stationary assumption,
then trigger inertial alignment after ten keyframes. Neither keyframe count nor visual parallax
establishes that gravity, accelerometer bias, velocity, and mono scale are sufficiently constrained.

[VIInit](../cpp/src/optimizers/AOptimizer.cpp#L474) and
[localMapVIOptimization](../cpp/src/optimizers/AOptimizer.cpp#L336) apply the solver result without
first checking `summary.IsSolutionUsable()`. This is particularly problematic in the presence of
invalid covariances, nonfinite residuals, or weakly constrained initialization.

Separately, [IMU.cpp:27](../cpp/src/data/sensors/IMU.cpp#L27) replaces any sample interval greater
than one second with `1 / _rate_hz`, while factor durations still come from the original timestamps.
This can make the integrated measurement represent a different duration from the residual model.
The normal factor-admission limit does not make silently changing a propagation interval correct.

- [ ] Check stationarity before using a static bias estimate, or use an initialization method that
      explicitly accommodates motion.
- [ ] Gate initialization on usable solver output, finite states, plausible scale/biases, and
      sufficient motion information; retry rather than accepting a failed alignment.
- [ ] Define handling for duplicate, decreasing, missing, and excessively separated timestamps.
- [ ] Track integrated duration and valid interval endpoints, and compare them with factor duration.
- [ ] Validate covariance finiteness and factorization success before constructing an IMU factor.

## 4. Corrections needed in the proposed fixes

### Bias bookkeeping must remain consistent throughout propagation

The original explicit-linearization-bias proposal is appropriate, but it must be applied coherently.
In [IMU.cpp:34](../cpp/src/data/sensors/IMU.cpp#L34), increments use the previous IMU's bias;
the right Jacobian at line 37 and acceleration coupling at line 81 use the keyframe's bias.
If those biases differ, even the increments and their propagated Jacobians describe different
linearization points.

Choose whether stored deltas remain at an immutable integration bias, or are updated along with
their declared linearization point. Adding `b_i + delta_b - b_lin` to factors while retaining
unaccounted in-place delta corrections risks applying a bias change twice. Corrections also need
to cover initialization, ongoing front-end propagation, prediction, and any active time-offset
optimization path. Store/replay samples if reintegration is part of the policy.

### Fix bias-prior units together with its statistical meaning

At [AOptimizer.cpp:451](../cpp/src/optimizers/AOptimizer.cpp#L451), converting nanoseconds to
seconds while retaining the multiplier `1000` increases residual weighting by `sqrt(1e9)`, or
about 31,623×. The information weight increases by `1e9`.

The original “30× weaker / 1000× stronger” statement describes square-root-information weights,
not variances. More importantly, uncertainty in an initial bias estimate is not the same as bias
random-walk uncertainty accumulated over a window. An extremely tight prior on a zero bias
increment may prevent recovery from an incorrect stationary initialization. Use an explicit,
justified initialization-bias uncertainty and check its effect on synthetic biased measurements.

### State the initialization coordinate convention before changing equations

Let `Q = R_w_i` be the alignment matrix returned by `VIInit`, `R = R_fi_w`, `s = exp(lambda)`,
and `Delta_p_W = p_j_W - p_i_W`. The code applies `T_f_I = (R Q, s t)` to a frame whose old
world-to-frame transform was `(R, t)`. Thus the aligned position is `p_I = Q^T s p_W`.
With velocities represented in the aligned inertial world, the consistent position residual is:

```text
r_p = R (s Delta_p_W) - R Q (v_i_I dt + 0.5 g_I dt^2) - Delta_p_corrected
```

This establishes the error from this repository's own transformation convention. It is not enough
to say that ORB-SLAM3 “rotates only gravity”: its initialization factor uses a different velocity
parameterization, including a scale factor on velocity. See the actual
[ORB-SLAM3 initialization equations](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/G2oTypes.cc).
Update the matching gravity and scale Jacobians together with the residual.

`IMUFactorInitBis` also needs its own convention audit before use. It has scale-dependent
velocities and omits both exponential and whitening factors in its scale Jacobian; it should not
be treated as a validated alternative initialization path.

### IMU extrinsic rotation and translation are different problems

The [ROS reader](../ros/src/sensorSubscriber.h#L70) rotates the IMU vectors using the YAML
rotation. It does not compensate accelerations for an offset between the IMU origin and the
estimated body origin. The inertial factor explicitly assumes the state frame is the IMU frame.

Adding a translation to `T_BS` therefore does not fix a lever arm. Prefer an estimated state whose
origin is the IMU, with camera extrinsics expressed relative to it. Keeping a distinct body origin
requires an explicit lever-arm model. Confirm the direction and origin of every transform rather
than relying on the variable name `T_s_f`, which is used differently in the loading paths.

The [RealSense wrapper documentation](https://github.com/realsenseai/realsense-ros) supports the
optical-axis concern and explains that separate gyro/accelerometer messages populate only their
respective fields; a combined IMU stream requires merging. Verify the actual bag's `frame_id`,
transforms, fields, and merge method before prescribing a rotation.

At rest, specific force has magnitude approximately 9.81 m/s². Its body components are
approximately `(0, 0, +9.81)` only for a level forward-left-up body. The general relation is
`f_B = -R_BW g_W + b_a`, ignoring noise. Rotation about the body's z-axis is the appropriate
axis-isolation check; arbitrary world-yaw motion of a tilted sensor is not equivalent.

### Preserve timestamps and integrate to camera boundaries

The offline reader assigns grouped images the IMU timestamp:
[mono](../cpp/src/dataproviders/adataprovider.cpp#L448),
[stereo](../cpp/src/dataproviders/adataprovider.cpp#L556). Sorting queues alone does not correct
this change in effective exposure time.

Preserve image timestamps, define one offset sign convention shared by ROS and offline input,
and integrate to the requested camera boundary using a documented interpolation/hold policy.
Image-only emission is useful only if the IMU interval can subsequently be constructed correctly.
Specify startup and end-of-stream flushing as well as ordinary chronological merging. Lock queue
access consistently; IMU queue emptiness/front access also occurs outside the mutex in the ROS reader.

### Reframe Issue 9 as an initialization handoff test

Under ordered input, `step_init()` returns after inserting a keyframe and the outer initialization
loop stops at ten keyframes. `_last_IMU` is normally that frame's IMU, including when an IMU was
synthesized for it. Furthermore, [processIMU](../cpp/src/data/sensors/IMU.cpp#L16) refreshes the
previous sample's cached pose from its live frame before propagation.

This does not prove every handoff is safe: image-only prediction can read a cache before a fresh
IMU propagation, and out-of-order input changes the situation. Test these cases explicitly rather
than asserting that newer IMU-only frames always remain after initialization.

The `_6d_velocity` value is computed from `T_fi_w T_w_fj`, a relative transform. A common global
alignment rotation cancels from that product. Uniform map scale changes its translational part,
not its angular part. Recompute the motion model from the corrected states or update only the
appropriate components; do not rotate and scale the whole six-vector as originally proposed.

## 5. Review of “Sparsify” and “Densify”

The high-level account is broadly consistent with the paper: marginalization produces a dense
prior, the VIO approximation uses a retained inertial state connected to point landmarks, and the
paper's densification uses photometric triangle validation followed by ray casting.

Three distinctions should be added:

- The paper explicitly retains the **dense prior for subsequent marginalization**, even when
  sparse factors are used during optimization. The implementation does this at
  [AngularAdjustmentCERESAnalytic.cpp:906](../cpp/src/optimizers/AngularAdjustmentCERESAnalytic.cpp#L906).
  This is part of the algorithm's information propagation, not an expendable implementation detail.
- The topology's resemblance to equations 4–5 does not validate its Jacobians or recovered weights;
  R4–R6 are counterexamples. Nor does it guarantee negligible information loss on every dataset.
- In this branch, dense reconstruction is queued around keyframe removal and is separately
  configured. Do not equate that event with successful probabilistic marginalization: for example,
  stereo queues dense processing before the conditional marginalization call. Statements that
  dense stereo is “for quality” describe a design goal, not an accuracy result established here.

The paper's 15-pixel patch description and Table I values of 0.09 m for MH_01 and 0.06 m for V1_01
are correctly reported. They are useful comparison targets under a matched evaluation protocol,
not guaranteed pass/fail thresholds for this modified implementation or its mono mode.

## 6. Review of the SaDLIO prospects

The proposals are plausible research directions. They should remain a separate design discussion
from the verified VIO repair list, with the following qualifications.

| Proposal | What is reasonable | What is missing or overstated |
|---|---|---|
| Plane/surfel landmarks | A sensible representation to investigate for LiDAR constraints. | Reuse is not “only the landmark Jacobians change.” State parameterization, normal constraints, association, residual dimensions, rank, covariance recovery, and degeneracy must be designed. The present code hard-codes point geometry and stereo information assumptions. |
| BALM-style elimination | Removing explicit feature parameters is an established approach; see [BALM](https://arxiv.org/abs/2010.08215). | Eliminating a plane shared by multiple poses creates a joint pose constraint; it does not automatically produce pairwise chain/tree factors. Any such approximation needs its own information-loss analysis. “Less novel” is not established here. |
| Reflectivity ZNCC | Temporal reflectivity consistency may be worth testing. | One scan's aligned range/reflectivity images do not provide the paper's second geometric viewpoint. A depth-sensitive warp needs another observation, visibility handling, motion compensation, and a suitable reflectivity model. |
| Finer-resolution ray casting | Can create a usable surface representation. | Interpolated samples are inferred geometry, not new measurements. Evaluate uncertainty, unsupported gaps, occlusion boundaries, and preservation of small obstacles, not point count alone. |
| Reusing preintegration for deskewing | IMU propagation is useful. | A delta between keyframes alone does not provide poses at each LiDAR return time. Retain or reconstruct a within-scan trajectory and handle per-point timestamps and extrinsics. |
| Static or LiDAR-assisted initialization | LiDAR removes monocular scale ambiguity. | It does not remove gravity/bias ambiguity or geometric degeneracy. Static initialization requires a stationary interval; dynamic initialization needs suitable motion and geometry. |
| Novelty and baseline selection | A focused literature review and comparisons are appropriate. | Novelty, fixed voxel-size choices, and claims that particular plane systems cannot suit rough terrain were not demonstrated. Existing work already studies plane tracking and LiDAR–IMU degeneracy; see [LIC-Fusion 2.0](https://arxiv.org/abs/2008.07196). |

The supplied paper confirms an Ouster OS128 in the terrain dataset. That does not establish which
channels, raw per-point timestamps, calibrations, or synchronized ground truth are available in a
particular downloaded recording. Inspect those before committing to a reflectivity/deskewing study.
No comprehensive novelty search or LiDAR experiment was performed in this review.

## 7. Numerical evidence and limits

The following independent calculations were executed during the review. They translate selected
source formulas into NumPy; they do not call compiled C++ code. Central differences used
`h = 1e-6`. Errors below are maximum absolute matrix-entry differences, not trajectory errors.

| Check | Current formula | Corrected/reference formula |
|---|---:|---:|
| IMU-factor pose-j rotation derivative | 0.2522523784 | 4.04e-11 |
| Sparse pose-to-landmark rotation recovery derivative | 3.8501366408 | 3.55e-10 |
| Sparse absolute-pose recovery derivative | 0.9752903090 | 7.86e-11 |
| `IMUPriordx` velocity derivative, whitening diagonal 1…15 | 8.0 | Residual requires whitening of the selection matrix |
| `so3_leftJacobian(0)` finiteness | False | Correct limit is identity |
| ESKF world-y velocity correction, 90° yaw and body-x innovation | -0.990099 | +0.990099 |

For reproduction, rotation vectors are in radians and `Exp`/`Log` denote SO(3) exponential/logarithm:

```text
IMU rotation check:
  R_i = Exp([0.2, -0.3, 0.1])
  R_j_base = Exp([-0.1, 0.15, 0.25])
  Delta_R_corrected = Exp([0.07, -0.11, 0.03])
  w_j = [0.12, -0.08, 0.06]
  f(w_j) = Log(Delta_R_corrected^T R_i (R_j_base Exp(w_j))^T)

Sparse recovery checks:
  R = Exp([0.2, 0.1, -0.3]); t = [0.5, -0.4, 0.2]; p = [2, 1, 4]
  landmark function = R Exp(w) p + t, differentiated at w = 0
  R_updated = R Exp(w); t_updated = t + R u
  pose function = [Log(R_updated R^T), t_updated - R_updated R^T t]
  differentiate at [w, u] = 0

Whitening check:
  S = diag(1, 2, ..., 15)
  velocity residual = S [0_6, v, 0_6], differentiated at v = 0

Velocity check:
  R1 = Exp([0, 0, pi/2]); innovation = [1, 0, 0]; C = 0.01 I
  current = R1^T (R1 R1^T + C)^-1 innovation
  reference = R1 (R1^T R1 + C)^-1 innovation
```

The host's standard include locations did not contain the Ceres, Eigen, and OpenCV headers needed
for a direct build at review time. A configured project/container build may provide them, but was
not exercised. No C++ test-pass claim, performance measurement, or ATE improvement is made here.

## 8. Revised repair and validation order

The ordering below prioritizes interpretable failures and depends on the actual failing-run
configuration. It is a proposed implementation plan, not a list of completed fixes.

1. **Capture the reproduction and make validation trustworthy.** Record the exact commit,
   configuration, reader, calibration, topics, and failure onset. Repair gradient assertions and
   add the numerical counterexamples above. Log solver usability, nonfinite states, rejected IMU
   samples, interval endpoints/duration, and inserted factor counts.
2. **Repair direct numerical failures and input consistency.** Fix R1/R2, the wrong random-walk
   key, chronological input handling, timestamp preservation, and any verified axis mismatch.
   Eliminate silent time-gap substitution and invalid attached IMUs. Initially use a known fixed
   time offset to isolate propagation from online offset estimation.
3. **Repair initialization as a coherent model.** Fix bias application, residual/Jacobian
   conventions, initial-bias uncertainty, motion/solver acceptance, and the state handoff.
   Validate stationary, biased, accelerating, and rotating synthetic cases, including mono scale.
4. **Repair normal factors and bias bookkeeping.** Fix R3 and establish explicit linearization
   biases across propagation, correction, prediction, and optimization. Test bias changes during
   an active interval. Start with deterministic single-thread execution, then validate threading.
5. **Restore window information propagation.** Preserve the IMU chain, validate interval identity
   before factor insertion, implement mono marginalization with appropriate rank handling, and
   first validate the dense-prior path with sparsification disabled.
6. **Validate sparsification separately.** Fix R4–R6, compare residual derivatives and information
   recovery, then measure accuracy and runtime against the same dense-prior baseline.
7. **Run dataset comparisons and controlled ablations.** Use EuRoC MH_01/V1_01 first with correct
   timing/calibration, then the relevant RealSense and ExECoSim recordings. Evaluate mono and
   stereo separately. Record trajectory completion, ATE/RPE, scale, bias/velocity behavior, and
   runtime. State alignment and frame conventions; scale-aligned scoring must not conceal a
   metric-scale failure. Re-enable online time-offset estimation and multithreading individually.

## 9. Recommended edits to the original document

- [ ] Replace “What is correct (no need to dig here)” with a bounded statement about the intended
      Forster residual structure, followed by R1–R9 and remaining validation requirements.
- [ ] Keep Issues 1–5, 7, and 8, incorporating the qualifications and additional paths above.
- [ ] Mark the RealSense diagnosis as recording-dependent and correct the lever-arm guidance.
- [ ] Replace Issue 9 with explicit initialization-handoff tests and the correct relative-motion
      model transformation.
- [ ] Correct the introductory accelerometer explanation to specific force, and replace “samples
      are never re-integrated” with “preintegration avoids repeated integration during ordinary
      optimization; reintegration may be needed after sufficiently large bias changes.”
- [ ] Distinguish noise-density discretization under nominal sample-rate assumptions from a
      blanket validation of covariance propagation under irregular timing and changing biases.
- [ ] Add the retained dense-prior detail to the sparsification explanation.
- [ ] Separate the SaDLIO hypotheses and unverified novelty claims from the VIO defect inventory.
- [ ] Use the revised validation order and retain the distinction between confirmed defects and
      an experimentally established cause of the reported VIO failures.
