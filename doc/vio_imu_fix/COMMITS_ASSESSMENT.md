# Assessment of the VIO fixes in 6d8cb3c and a3fab2b

Reviewed 2026-10-04 against HEAD `a3fab2b`, with `f157380` as the pre-fix baseline.

The fixes substantially improve SaDVIO. The evidence is strongest for stereo VIO on TUM-VI with marginalization and camera–IMU time-offset estimation enabled. RealSense requires much looser IMU weighting to become reliable in the recorded experiments, and monocular metric scale remains a concern. Several correctness and validation gaps remain, including issues identified in this assessment.

This assessment changes no implementation or configuration. Trajectory numbers below come from the committed experiment records; I did not replay those datasets during this review. I independently inspected the code changes, reran the existing CTest suite in the development container, and checked two mathematical/data-flow edge cases described below.

## 1. What the two commits contain

| Commit | Role | Scope |
|---|---|---|
| `6d8cb3c` — “Not a SaD day for VIO corrections” | Implementation and regression tests | 42 files, 3,876 added lines and 1,387 removed lines. Includes IMU processing, initialization, optimization, marginalization, readers, threading, camera projection and tests. |
| `a3fab2b` — “Test docs” | Evidence and experiment tooling | 40 files, 6,675 added lines. Includes the issue list, historical review, experiment ledger, 1,027 per-run metric records, dataset configurations, runners and diagnostic tools. |

The second commit records work performed at several intermediate code snapshots. A run named `final` or `final2` is not necessarily a test of the final committed implementation. Later sections of the ledger supersede earlier status statements.

The production `ros/config/config.yaml` was not changed by these commits. Installing the fixed code does not automatically select the experimental settings that produced the strongest results.

## 2. Terms used in this assessment

- **VO:** estimating movement from cameras alone. **VIO:** adding the IMU.
- **Bias:** a persistent offset in an IMU reading. Even a small offset can accumulate into large position errors.
- **Preintegration:** compressing many IMU samples between two selected camera frames into one motion constraint.
- **Keyframe:** a selected camera frame kept in the optimizer.
- **Marginalization:** removing an old frame while keeping a mathematical summary of what it taught us. This summary is called a **prior**.
- **Sparsification:** replacing that summary with a simpler, cheaper approximation.
- **Jacobian:** the derivative that tells the optimizer how changing a variable changes an error. An incorrect Jacobian can send an otherwise correct model in the wrong direction.
- **ATE:** overall trajectory position error after alignment to ground truth. Lower is better, but completion and scale must also be checked.

## 3. Status of the original 20 issues

“Implemented” means the intended repair is present and has relevant validation. It does not mean every dataset, motion pattern or configuration is proved reliable.

| Issue | What changed, in plain language | Assessment |
|---|---|---|
| 1. Wrong gyro noise key | Gyroscope bias drift now reads the gyroscope YAML field, rather than the accelerometer field. | Implemented; configuration test added. Actual sensor noise values still need calibration. |
| 2. Initialization bias updates | Applies the accelerometer correction once and the gyroscope correction once. Replaces the initialization prior based on nanosecond timestamps with explicit bias uncertainty values. | Implemented. The prior strengths are chosen tuning values, not measured properties of every IMU. |
| 3. Initialization equations | Corrects the gravity/rotation relationship in the position equation and the scale derivative at non-unit scale. | Implemented in the active initializer. The unused `IMUFactorInitBis` is documented as problematic rather than repaired. |
| 4. Bias bookkeeping | Records the bias used when integrating samples. Factors compare the current bias with that recorded value. Large changes trigger reintegration. | Substantial correction. Reintegration thresholds are 0.1 m/s² for accelerometer bias and 0.01 rad/s for gyro bias. |
| 5. Removing a low-parallax frame | Rebuilds the IMU link across a removed keyframe instead of losing part of the motion history. Later changes limit bridges to one second. | Implemented, with synthetic tests. Longer intervals use a different window-removal decision instead of endlessly extending a bridge. |
| 6. RealSense coordinate frames | Both readers rotate measurements into the configured body frame. Checks warn about bad rotations and unsupported IMU displacement from the body origin. | Partial physical solution. Experimental configurations use the IMU as body origin, but camera–IMU calibration remains approximate. No general lever-arm acceleration model was added. |
| 7. Offline timing | Removes the hard-coded 15 ms adjustment. Uses a common chronological merger and supplies an IMU sample at the image time by interpolation. | Major improvement. The raw-data-gap edge case in section 7 remains. |
| 8. ROS ordering | Raw and compressed readers share the merger so image and IMU processing use consistent timing logic. | Implemented; ROS playback evidence exists. This does not establish zero transport loss under load or comprehensive validation of both transports. |
| 9. Initialization handoff | Makes gravity alignment safe at parallel/opposite directions, estimates static biases only when measurements look stationary, and rescales only the translational motion-model component. | Implemented. Correctly narrows the earlier claim that every cached IMU-only pose had to be transformed. |
| 10. Zero-rotation NaNs | Supplies the small-angle expression for the SO(3) left Jacobian. | Implemented and tested. This removes a direct numerical failure at zero rotation. |
| 11. ESKF frame errors | Corrects the velocity update direction, covariance changes between IMU/camera coordinates, rotation correction, and the general camera-extrinsic composition. | Implemented, with derivative tests. Some uncertainty weights remain heuristic; a documented mono ablation worsened, so correctness is not equivalent to a universal ATE gain. |
| 12. Ordinary IMU-factor derivative | Fixes the second pose's rotation derivative when the optimizer has already taken a nonzero step. | Implemented; tested away from zero increments. |
| 13. Prior weighting | Applies the same uncertainty weighting to the prior's velocity/bias derivatives as to its errors. | Implemented; tested with a nontrivial weighting matrix. |
| 14. Sparse-prior mathematics | Corrects the derivatives used to reconstruct the cheaper prior. Adds comparisons to a dense prior and tests of the information approximation. | Implemented for tested cases. These tests do not establish that sparsification always improves trajectories or runtime. |
| 15. Prior insertion | Stops deciding whether a prior exists by counting landmark types; shares insertion logic and repairs associated VO crashes. | Common point-landmark case repaired. An IMU-state-only prior can still be skipped when no landmarks remain. |
| 16. Mono marginalization | Monocular VIO now calls marginalization and can retain landmarks seen through one camera across frames. | Implemented and exercised in synthetic and option tests. Does not solve all mono scale/initialization failures. |
| 17. Ineffective tests | Tests now compare analytic derivatives with numerical changes and fail on discrepancies. Adds nonzero-motion, tilted-gravity, bias and option-combination cases. | Major improvement in confidence. A new derivative gap remains outside that coverage. |
| 18. Validity and data gaps | Uses actual elapsed time, tracks long steps, checks covariance, rejects unusable initialization solves and adds state checks/diagnostics. | Partially complete. No adequate-motion gate for mono initialization; normal VIO solves still apply unusable results; the pose check has a logic error; interpolated samples can conceal gaps. |
| 19. Online time offset | Uses an absolute offset and a stored per-frame reference instead of repeatedly accumulating the full estimate. Applies timing corrections in both analytic optimizers/readers. | Large correction; tests and trajectories support it. Convergence near 2.5 ms on TUM-VI is not independent proof of the physical clock offset. |
| 20. Unsupported options | Rejects Numeric optimizer combinations that do not implement marginalization or time-offset estimation; warns about ineffective options. | Useful and implemented. Rejection is the solution for those Numeric combinations, not an implementation of the missing features. |

Sources: [issue list](../VIO_fix_+_LIO_Prospects.md), [ledger](IMU_FIX_LEDGER.md), [IMU implementation](../../cpp/src/data/sensors/IMU.cpp), [optimizer](../../cpp/src/optimizers/AOptimizer.cpp), [configuration validation](../../cpp/src/slamParameters.cpp), and [tests](../../cpp/tests/imu_test.cpp).

## 4. Important fixes beyond the original list

### The optimizer's memory was repaired

These are among the most consequential changes:

1. **Wrong prior sign:** the stored prior could push against the direction justified by the measurements. Its sign is corrected.
2. **Missing reference state:** the prior previously followed the current estimate as that estimate moved. It now remembers the state at which it was built and measures changes from there.
3. **Landmarks moved behind the prior's back:** front-end landmark-only refinement now freezes landmarks tied to a prior, instead of moving them while ignoring that prior.
4. **Over-constrained old frame:** VIO no longer fixes the entire oldest pose when a retained prior should constrain it. After solving, a common yaw/translation adjustment preserves the arbitrary world reference without blocking gravity-related tilt correction.
5. **Bad visual matches retained too strongly:** a robust loss reduces their weight both in the active VIO window and when storing their information in a prior.
6. **Inconsistent visual weighting:** angular residual uncertainty is made consistent between the normal window, time-offset path and marginalization, using a 1.5-pixel assumption.
7. **Poor 3D points stored in the prior:** landmark checks reject large bearing disagreement and insufficient viewing-angle spread before retaining the point. This limits unstable distant or badly triangulated points entering the prior.

These changes explain why “the IMU integration is fixed” is too narrow a description. The camera measurements, IMU measurements and retained history must remain consistent together. The new dense prior still needs the derivative correction described below.

### Crashes, races and lifetime problems were repaired

- Deduplicates landmarks and avoids adding the same initialization keyframe twice, preventing duplicate optimizer parameter blocks.
- Includes the retained-frame prior in the angular marginalization path where it was missing.
- Uses a thread-local timer stack and removes an unmatched timer call.
- Serializes front-end/back-end modifications of shared map state.
- Waits for a pending keyframe to enter the back end before processing a later frame. Otherwise IMU links could skip that pending keyframe. Recorded skips dropped from as many as 15–29 per run to zero in the checked runs.
- Makes the time-offset value atomic between threads.
- Clears stale visual matches during reinitialization and avoids using detached features.
- Applies a plausibility check to visual/ESKF pose updates even for small translations. This is a heuristic jump filter, not a physical consistency proof.
- Cuts the old IMU history link when the oldest required frame leaves the window. The ledger reports memory changing from roughly 364→625 MB during a long run to roughly 338–344 MB after the repair.

The threading changes improve correctness. Most front-end/back-end computation is serialized by the shared lock, so they should not be advertised as evidence of a major parallel-compute speedup.

### Smaller functional changes matter too

- Implements the previously missing DoubleSphere projection derivative used by the pixel-based optimizer path.
- Initializes `Prior1D` correctly.
- Fixes double counting of matches in the parallax/mono keyframe decisions.
- Removes an IMU self-reference during mono initialization.
- Repairs additional angular inverse-depth and relative-pose derivatives.
- Reworks `refineTriangulation`; the ledger notes that the current pipelines do not call it, so it should not be credited for their trajectory gains.
- Adds definitions so a build without VDB-GPDF links; selecting the unavailable functionality still throws.
- Adds diagnostics for solver status, factors, biases, prior errors and frame-chain problems, plus runners, snapshots, ROS conversion and replay tools.
- Makes reinitializations visible in logs for modes that previously restarted silently.

The commits do not implement the proposed LiDAR-inertial system. The LIO section remains future work.

## 5. What the trajectory evidence supports

### TUM-VI stereo: convincing improvement in the tested configuration

The initial room1 record had stereo VO at **0.219 m ATE** and stereo VIO at **0.824 m**. Later final-build checks put ordinary stereo VIO around **0.27–0.33 m**, but stereo VIO with **marginalization plus time-offset estimation** around **0.08 m**. Stereo VO in those later checks was around **0.15–0.17 m**.

The final replay runs reported in the ledger give the following comparison. Each cell is one run, not a confidence interval:

| Sequence | Stereo VO ATE, metres | Stereo VIO + marginalization + offset ATE, metres |
|---|---:|---:|
| room1 | 0.139 | 0.089 |
| room2 | 0.215 | 0.089 |
| room3 | 0.154 | 0.104 |
| room4 | 0.120 | 0.063 |
| room5 | 0.161 | 0.084 |
| room6 | 0.071 | 0.059 |
| magistrale2 | 6.94 | 2.81 |

Earlier repeated room experiments also support the improvement. VO controls with marginalization/sparsification did not reproduce the VIO benefit. This supports the conclusion that the combined inertial information and timing correction matter, rather than the improvement simply coming from changing the visual back end.

Sparsification sometimes helps and sometimes hurts. It is not required for the central demonstrated gain.

For magistrale2, ground truth exists at the beginning and end, not throughout the long route. Its ATE therefore cannot certify the unobserved middle. The records also show considerable variation: roughly 0.7–4.7 m across relevant runs, with occasional runs near the worse end. The ledger's categorical “no regression” wording is stronger than the small A/B samples justify; “no clear systematic regression demonstrated” is safer.

### RealSense: useful stabilization, calibration still unresolved

Across nine recordings, the recorded median sequence ATEs were:

| Configuration | Median ATE, metres | Reported resets over 27 runs |
|---|---:|---:|
| Stereo VO | 0.065 | 0 |
| Stereo VIO, nominal IMU noise | 17.87 | 11 |
| Stereo VIO, noise5 | 0.061 | 0 |
| Stereo VIO, noise5 + marginalization | 0.179 | 2 |
| Stereo VIO, noise5 + marginalization + offset | 0.108 | 2 |

`noise5` uses accelerometer noise density 0.15 and gyro noise density 0.02, in the configuration's SI noise-density units. Larger noise tells the optimizer to trust these IMU readings less. This successfully stabilizes these recordings, but does not establish that the actual problem was only random sensor noise. Calibration error, vibration and other model mismatch can also be softened by lowering the IMU weight.

The camera–IMU transform remains approximate. The OptiTrack tracked-body-to-IMU transform is also uncalibrated. The evaluator says position metrics are independent of that transform, but this is only partly true: a displaced tracked origin moves relative to the IMU origin as the rig rotates. A single trajectory alignment cannot remove that changing lever-arm displacement. Thus **0.061 versus 0.065 m is evidence of comparable performance, not a convincing 4 mm advantage**.

RealSense marginalization is explicitly deferred in the later ledger. The TUM-VI settings should not be treated as a universal RealSense solution.

### Mono and ROS: progress with limits

Mono VIO is much more capable of completing the tested room sequences. Metric scale is not uniformly reliable, especially on the long sequence. Mono ATE is computed after allowing a global rescaling, so a visually good trajectory score can hide an incorrect distance scale. Check the reported scale as well as ATE.

Mono VO still fails badly on several room sequences despite the match-count correction. This is a separate remaining visual-pipeline limitation.

ROS playback has useful final-code checks, including marginalization and multithreading. Room1 stereo VIO with marginalization/offset is approximately 0.08–0.10 m ATE in those checks. Some images are dropped under load. The simultaneous live VO/VIO comparison is evidence that the tested setup kept up in that experiment, not a general performance guarantee for other computers or configurations.

Sources: [recorded results](RESULTS_ALL_RUNS.md), [per-run CSV](all_runs_metrics.csv), [ledger](IMU_FIX_LEDGER.md), [evaluation code](tools/eval_traj.py), [RealSense noise5 configuration](configs/dataset/realsense_d455_vio_noise5.yaml).

## 6. Validation independently checked in this review

I ran CTest in the existing development container from `cpp/build_tests`, its configured working directory: **70 passed, 1 failed out of 71**, matching the latest ledger. The failure is `LineFeatureTest.LineFeatureMatching`.

The ledger describes that failure as an unstable detector/matching threshold made repeatable by fixed random seeds. It remains a real failing test, though it is outside the IMU changes. This is not an entirely green suite.

An initial direct invocation from the repository root produced extra missing-file failures because these tests expect relative paths from the build directory. CTest resolved those path failures. The numerical checks and the CTest run did not require changes to the implementation. Relevant IMU, marginalization and test source hashes matched between this checkout and the development container; I used its existing compiled test binary rather than doing a fresh rebuild.

The committed CSV contains **1,027 run records**. This is substantial experimental work, but it mixes debugging runs, old snapshots, configuration sweeps and repeated runs. It is not 1,027 independent validations of HEAD. The final replay labels quoted in the ledger are absent from that CSV, so those particular numbers have ledger-level provenance in the committed material.

Further limits of the recorded metrics:

- “Coverage” measures the time between first and last estimated frames divided by sequence duration, not the percentage of frames tracked successfully.
- Trajectory logging can leave only the segment after reinitialization. Low ATE on a short surviving segment is not success.
- Early restart counts for mono/VO are understated because those modes did not yet print the message counted by the evaluator.
- The tests cover many useful option combinations, but a single room smoke test does not establish field reliability.
- No EuRoC reproduction of the paper's published accuracy was completed. TUM-VI evidence is useful, but is a different benchmark.

## 7. Remaining code findings

### A. New dense-prior rotation derivative is incomplete

In [MarginalizationFactor::Evaluate](../../cpp/include/isaeslam/optimizers/marginalization.hpp), the rotation error is now correctly measured from the stored reference, as `Log(A * Exp(x))`. However, its derivative uses only `Jr(Log(A * Exp(x)))^-1`.

For the ordinary additive parameter block used here, it also needs the right Jacobian of the current increment, `Jr(x)`. At zero increment that extra factor is identity, so checks confined to zero will miss this.

I checked the mathematical component using central finite differences, with reference rotation vector `(-0.1, 0.25, 0.15)` and increment `(0.2, -0.15, 0.1)` radians. Maximum element error was **0.1060** for the committed expression and **4.55e-10** after including the missing factor. This was an independent numerical check of the expression, not a newly compiled C++ regression test.

Consequence: after a nonzero optimization step, the new prior can still supply an incorrect direction/weight for rotation changes. Add a full-factor derivative test at nonzero increments, then correct the expression.

### B. Image interpolation can conceal missing raw IMU data

[ImuImageMerger::emitImageFrame](../../cpp/src/dataproviders/adataprovider.cpp) interpolates at each image timestamp without limiting the separation of the real IMU samples used. [IMU::processIMU](../../cpp/src/data/sensors/IMU.cpp) detects missing data by the intervals between the samples it receives, including those interpolated samples.

For a 200 Hz configured IMU, the gap threshold is 0.05 s. Consider a two-second raw IMU outage while images continue at 20 Hz: the merger can generate image-time samples every 0.05 s throughout that outage. The integrator then sees no step greater than its threshold. An arithmetic reconstruction of those emitted timestamps gives **zero flagged gaps** despite the two-second raw outage.

Consequence: factors may trust invented motion across missing measurements. The existing gap test exercises integration directly and does not cover this reader-plus-integrator interaction. Preserve raw-gap information or reject interpolation across an excessive raw interval, and test the complete path.

### C. Ordinary VIO still applies unusable solver results

Initialization now checks `IsSolutionUsable()`. However, [AOptimizer::localMapVIOptimization](../../cpp/src/optimizers/AOptimizer.cpp) and both analytic time-offset variants record the solver status and then apply pose, landmark, velocity and bias updates without an equivalent rejection gate.

Logging failure is helpful, but it does not prevent a bad update from entering the map. Define rejection/recovery behavior and validate the candidate state before changing the accepted state.

### D. Initialization accepts less than the checklist implies

In [SLAMCore::inertialInitAccepted](../../cpp/src/slamCore.cpp), a non-finite pose causes `continue`, not `return false`. That skips checking the frame rather than rejecting the initialization. Missing IMUs are skipped in the same branch.

The code checks positive finite scale and bias bounds, but does not establish that the motion provided enough information to determine mono scale and gravity reliably. The issue-18 checkbox claiming finite states and “enough motion” are handled should be marked partial.

Integrated duration is logged, but `imuFactorUsable` does not explicitly compare that duration with the factor's timestamp span. Likewise, the ledger's claim that bias random-walk constraints are kept in all cases needs narrowing: the early interval-over-one-second skip bypasses both the motion and bias factor in that loop.

### E. Priors without retained landmarks remain an edge case

[AOptimizer::addSparsePriorResiduals](../../cpp/src/optimizers/AOptimizer.cpp) returns immediately when the retained point list is empty, before inserting a possible retained IMU-state prior. Dense-prior insertion/reuse is also gated by the retained-landmark container in both analytic optimizers.

The previous landmark-type-count bug is repaired for ordinary point maps. The more general case of useful inertial state information surviving without retained landmarks still needs an explicit policy and regression test.

### F. Some uncertainty and calibration choices remain approximations

The integration covariance still includes a fixed extra position-noise term, and sample noise uses the configured nominal IMU rate. Interpolated samples are not independent new measurements, but that correlation is not represented in the current covariance model. ESKF weighting also retains heuristic constants.

These are model limitations rather than a demonstration that the successful trajectories are invalid. They do mean that passing derivative tests is not proof that every uncertainty value is statistically calibrated. Buffer behavior while online time offset changes also deserves a dedicated test: shifted queued samples and per-frame offset references must describe the same timing convention.

## 8. Recommended next work

1. Correct the dense-prior derivative and add a nonzero-increment regression test.
2. Reject unusable ordinary VIO updates and fix the initialization non-finite-pose branch.
3. Test missing raw IMU data through the real merger, not only through the integrator.
4. Add a mono initialization check for sufficient informative motion and evaluate mono VIO without allowing rescaling as well as with it.
5. Finish camera–IMU and tracked-body calibration for RealSense before treating noise inflation as the final tuning solution; revisit its marginalization after that.
6. Publish a clearly identified final-code configuration/results table, with completion, resets, scale and repeated runs. Include the remaining failed unit test and the long-sequence variation.
7. Reproduce the relevant EuRoC paper benchmark before claiming paper-level validation or freezing the core for LIO reuse.

For reproducing the strongest existing TUM-VI evidence, the relevant combination is stereo VIO, `AngularAnalytic`, `marginalization: 1`, and `estimate_td: 1`, with the matching TUM-VI camera calibration. Dense marginalization is an adequate starting point; sparsification is a separate tradeoff to measure. RealSense has a different documented outcome and should retain a separate configuration and acceptance target.
