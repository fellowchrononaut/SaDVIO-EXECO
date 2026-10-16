# nofovvio: two non-overlapping fisheye cameras + IMU (GUTWrap, Insta360 X5)

The request: `GUTWrap/GUTWrap_Discussion/SaDVIO_feature_request.md` (2026-10-16). Branch `dense_devel`. Reference
point before this work: tag `sadvio-pre-nofovvio` (`e209680`).

Rules for this work:
- **Other modes stay unaffected.** New code goes in new classes: the `nofovvio` mode and the KB4 sensor. Changes in
  shared code are opt-in: mask reading only when a mask folder exists, the ray limit only for the KB4 model,
  cross-lens loop closure only with two or more non-overlapping cameras.
- Before each commit: unit tests, plus a `doc/loop_closure/tools/run_final_matrix.sh` subset (all six existing
  configurations) compared with the tag.
- The data goes in `Simulator_Validation/SaD_VIO_data/` (container data only); notes, tools and results go here.
- Results are written in this ledger and in a reply note to GUTWrap.
- We work through the checklist one item at a time; the user decides each step.

## Checklist

### 0. Data (first)
- [x] 0.1 Ask the GUTWrap session what exists for VID_001 and VID_012: frames per lens, IMU, masks, calibration
  (KB4 and double sphere, rig, camera–IMU), COLMAP reference, a measured distance. Asked 2026-10-16.
- [x] 0.2 The user decides what is copied, and where (2026-10-16: the export request was sent to `gutwrap-50`).
- [x] 0.3 Convert to EuRoC layout if needed (`mav0/cam0`, `cam1`, `imu0`, `camN/mask`), and write a SaDVIO dataset
  YAML. Working resolution 960 px first, then 1920 px.

### 1. Baseline with what exists today
- [x] 1.1 Per-frame masks in the dataset reader (`mav0/camN/mask/<ts>.png`, white = keep), switched on by the
  config option `use_masks` (default 0), with a unit test.
- [x] 1.2 TUM-format pose export (per frame), plus a summary of the final td and biases (VIO modes).
- [ ] 1.3 Existing `nofov` (VO) with the double-sphere calibration on both captures. Scored against the COLMAP
  reference: ATE after Sim(3), and whether the run completes without re-initialization. Infeasible on this rig (log,
  2026-10-16): `nofov` never initializes. Replaced by 1.3b.
- [x] 1.3b Baseline B (the user's choice): `monovio`, front lens only, KB4 + masks + IMU, 960 px, 4 runs per
  capture, `doc/nofovvio/tools/eval_x5.py`. Results in the log (2026-10-16).
- [x] 1.4 Replay of baseline B in the presentation's SaDVIO "VO / VIO replays" view
  (`Presentations/EXECO-Oct-12-2026/presentation.html`, group "Insta360 X5"), `doc/nofovvio/tools/build_x5_replay.py`.

### 2. KB4 / OpenCV-fisheye camera model
- [x] 2.1 New sensor class: θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸), Newton unprojection, bearings valid
  beyond 90° (to the 95–100° image circle). Unit tests: round trip, beyond 90°, behind the image plane.
- [x] 2.2 Ray limit (about 80° from the axis) for the x/z-based estimators (PnP, ESKF, essential matrix, loop
  verification), for this model only. The angular window optimizer uses every ray.
- [ ] 2.3 `nofov` with KB4 against the double-sphere baseline.

### 3. `nofovvio` mode
- [x] 3.0 VIO scale on VID_012: `monovio` lengths were ×1.13 against GUTWrap's first box scale (1.37 m/unit). Time
  offset, IMU noise, time windows and rotation speed ruled out a VIO cause; GUTWrap's v2 measurement (1.525 m/unit,
  ±1.3 %) puts the VIO at ×1.000–1.031. Closed 2026-10-16: the error was in the reference scale.
- [x] 3.1 `SLAMNonOverlappingFovVIO` (`slam_mode: nofovvio`), derived from `SLAMMonoVIO` (not `SLAMBiMonoVIO`: its
  initialization needs stereo depth between overlapping cameras): `monovio`'s initialization on camera 0, then both
  cameras tracked in time on their own, one ESKF update over both cameras' landmarks, `monovio`'s back end. Mode
  validation in `slamParameters`. Details in the log (2026-10-16).
- [x] 3.2 Sanity check: EuRoC stereo run as non-overlapping, against `bimonovio` (log, 2026-10-16).
- [x] 3.3 VID_001 and VID_012 (results in the log, 2026-10-16):
  - ATE after Sim(3), and after gravity-only alignment;
  - completion without re-initialization;
  - scale against a measured distance; gravity within 1°;
  - runtime at full and half resolution.
- [x] 3.4 Outputs: per-frame TUM poses in a gravity-aligned metric world, final td and biases, window pose
  covariance (`log_pose_covariance`), and the camera–IMU extrinsics as used (`export_tum.py`, log 2026-10-16).

### 4. Loop closure with two lenses
- [ ] 4.1 Cross-lens candidates (the rear lens sees what the front lens saw on the way out), enabled only with two
  or more non-overlapping cameras; validated on the captures that return to their start.

### 5. Export for SfM
- [ ] 5.1 COLMAP text export (`cameras`, `images`, `points3D`, rig) from the final map, with KB4 written as
  `OPENCV_FISHEYE`.

### 6. Later, if the results call for it
- [ ] 6.1 Rolling-shutter compensation (needs the measured readout time): per-row time offset in the residuals,
  using the IMU velocity.
- [ ] 6.2 Online camera–IMU extrinsic refinement (window parameter blocks, VINS-style; observability check).
- [ ] 6.3 Line features on low-texture captures (VID_012: white walls); fix the flaky `LineFeatureMatching` test
  first.

### 7. Reporting
- [ ] 7.1 Reply note to GUTWrap (`GUTWrap_Discussion/`) after each milestone (1, 3, 4/5).

## Older open items (kept from before)

- [ ] RealSense calibration: waiting on the user. It also unblocks a dense run on large-scene data.
- [ ] MegaLoc: optional or not, and how to distribute the model.
- [ ] ORB vocabulary licence (GPLv3 data): ship it separately, or train our own.
- [ ] Rename the labels in the older ledger sections.
- [ ] The `hloc_simval` container that is still running.
- [ ] Threaded loop closure (`multithreading: 1`), never evaluated.
- [ ] Mono VO accuracy (about 0.31 m loop-closed, huge run-to-run spread; compare with ≥ 4–6 runs per sequence);
  mono VIO on MH_01, MH_02, V1_03 (0.36–1.09 m).
- [ ] magistrale2 (1.3 m loop-closed, even with VIO).
- [ ] Stereo VO back-end time (+65–140 %).
- [ ] ExECoSim validation.
- [ ] MegaLoc on CPU, too slow for live use.
- [ ] Dense: range weighting for the per-keyframe methods' fused-view chunks and `stereo_tsdf_fast`; far-coverage
  check on a large scene. The MeshCSLAM objectives themselves are handled by the MeshCSLAM developers.
- [ ] Scratch leftovers (`lc_kfsets`, `lc_desc`, `live_*`).

## Log

- 2026-10-16:
  - Tag `sadvio-pre-nofovvio` created; pushing it is left to the user (no credentials in the agent's shell).
  - Dense dumps and replays deleted from scratch (2.4 GB).
  - Asked the three gutwrap sessions about the data (item 0.1).
- 2026-10-16, item 0.1 answered by `gutwrap-50` (owner of the X5 data and SfM):
  - Source: `.insv` per capture under `Simulator_Validation/GS_Captures/Insta360x5/Insta360x5`; the two HEVC streams
    3840² at 29.97 fps plus the IMU (demuxed copies in `VID_0xx/dual_fisheye/{front,rear}.hevc`).
  - Frames: only 4 fps subsets as JPEG. VID_012 has 191 pairs with exact times (`selection.json`); VID_001 has 103
    pairs at approximate times. Not in EuRoC layout.
  - IMU: not exported; read through telemetry-parser. VID_012: 994 Hz, gyro in deg/s, on the video clock,
    `t_imu = t_video − 0.005 s` (estimated). VID_001 not extracted.
  - Masks: operator only, white = keep, for the subset frames only; the lens circle is not masked.
  - Calibration: KB4 per lens and per capture (focal lengths about 2 % apart between captures); no double-sphere
    fit, no YAML. Rig for VID_012 only (about 180° about y, translation in SfM units). Camera–IMU rotation for VID_012
    (`gravity_report.json`, `R_refcam_imu`: IMU → front camera, 0.04° residual); translation unknown.
  - Reference: VID_012 COLMAP rig BA (382 images, 1.13 px), with a gravity-aligned version; VID_001 Metashape (not
    gravity-aligned, no rig). Neither is metric, and no distance has been measured in either scene.
  - They offer: EuRoC export at any rate with masks, double-sphere fit, YAML in our schema, VID_001 IMU and gravity
    alignment. Requests go through the user.
  - Our proposal to the user: full-rate EuRoC export at 1920 px under `SaD_VIO_data/insta360x5/`; masks for every
    frame; a YAML with KB4 and `T_BS`; VID_001's IMU; exact reference frame times; a measured distance on site; KB4
    model before the `nofov` baseline (no double-sphere fit needed).
- 2026-10-16, user decisions: send the export request (sent to `gutwrap-50`). VID_012's scale: the cube on the floor
  is 1 m × 1 m × 0.5 m (measured by the user); the user will measure a distance for VID_001. KB4 comes before the
  `nofov` baseline.
- 2026-10-16, items 2.1 and 2.2 (code; no data needed):
  - `KannalaBrandt` sensor (`cpp/include/isaeslam/data/sensors/KannalaBrandt.h`), `projection_model: kb4`:
    - intrinsics `[fu, fv, cu, cv]`, `distortion_coefficients [k1..k4]`;
    - optional `max_theta_deg` (default 100, the valid angle from the axis);
    - unprojection by Newton, giving unit bearings that are valid beyond 90°;
    - analytic Jacobian of the projection.
  - An unknown `projection_model` is now a clear error; it used to leave the camera unset and crash later.
  - Ray limit for the x / z estimators (`geometry::planarRayMinCos`, off by default). It is set to
    cos(`planar_max_theta_deg`, default 80°) only when a KB4 camera is configured.
    - PnP and the ESKF pose update keep wider rays in track but leave them out of the estimate.
    - The ESKF landmark refinement drops their residual.
    - The essential matrix leaves them out and keeps them untested.
    - With the limit off, every estimator behaves exactly as before.
  - Tests (`cpp/tests/camera_test.cpp`, X5 front lens at 1920 px):
    - round trip to 99.5° (angle < 1e-9 rad; bearings behind the image plane beyond 90°);
    - invalid beyond 100° and straight behind;
    - same pixels as `cv::fisheye::projectPoints` (< 1e-6 px);
    - analytic Jacobian against numeric, near the axis, at 60° and at 95°;
    - the ray limit is opt-in.
  - Suite 100 / 101 (`LineFeatureMatching`).
- 2026-10-16, export delivered by `gutwrap-50` (`SaD_VIO_data/insta360x5/`, see its README):
  - Both captures at the full 29.97 fps, 1920 px gray: VID_001 774 frames per lens, VID_012 1430.
  - IMU at 994 Hz on the video clock, no offset applied; per-frame SAM2 operator masks; YAMLs with KB4.
  - `reference_poses_cam0.csv`: exact ns timestamps, gravity-aligned with down = +y, SfM units.
  - The scale cube's corners are pending: they asked the user which object it is.
  - The last ~10 s of VID_012 leave the arena.
- 2026-10-16, item 0.3:
  - Our dataset YAMLs are `doc/vio_imu_fix/configs/dataset/insta360x5_vid_{001,012}.yaml`. They are copies of
    GUTWrap's with cu, cv − 0.5 px (COLMAP pixel centres at +0.5; KB4 and OpenCV at integer pixels).
  - `dt_imu_cam` keeps the same sign: SaDVIO places an IMU sample stamped t_imu at camera time t_imu − dt, which is
    GUTWrap's t_imu = t_video + dt.
  - Sequences `x5_001` and `x5_012` registered in `run_sadvio.py`.
- 2026-10-16, item 1.1:
  - `use_masks` (slam config, default 0; the user asked for an explicit option): `EUROCGrabber` reads
    `camN/mask/<image file name>` and binarizes it (non-zero = keep).
  - A missing or mis-sized mask means the whole image, with one warning. `use_masks` set without a mask folder
    gives a warning and whole images.
  - Each mask goes through the same undistortion (pinhole) and downsampling as its image (nearest).
  - KB4 cameras: every mask is intersected with the lens' image circle (θ ≤ `max_theta_deg`), computed once per
    camera. That is the model's valid region, independent of `use_masks`.
  - Tests: `DataProviderTest.perImageMasksFollowTheImages` (default mask without the option even when files exist;
    masked half, binarized half, downsampled, missing file), `KannalaBrandtTest.imageCircleMask`. Suite 102 / 103.
- 2026-10-16, item 1.2:
  - The per-frame pose log (`EXECO_PERFRAME_LOG`, `results_perframe.csv`, unchanged format) moved from
    `slamBiMono.cpp` / `slamMono.cpp` into `SLAMCore::logPerFrame`, now called by every mode's front end (VIO and
    `nofov` included).
  - `doc/nofovvio/tools/export_tum.py <run> [--frame body|cam0]` writes:
    - `trajectory_tum.txt`: every frame, each taking the correction of its latest keyframe,
      T_final(kf) T_front(kf)^-1 T_front(frame), so it follows the window refinement and the loop closure;
    - `keyframes_tum.txt`;
    - `run_summary.json`: final `dt_imu_cam`, last biases, the T_BS used.
  - Check on EuRoC V1_01 (`bimonovio`, `estimate_td`): per-frame ATE 0.054 m over 2791 frames, keyframes 0.052 m (raw
    front end 0.056).
  - Also: `mask_image_circle` (default 1, KB4 only) makes the image-circle mask switchable; `nofov` added to
    `run_sadvio.py`'s modes.
- 2026-10-16, item 1.3 (first try, `x5_nofov_try`):
  - The rear camera's T_BS got a baseline, needed by `nofov`: VID_012's rig translation in SfM units, put into the
    body frame (about 1.9 cm, assuming SfM units ≈ metres), used for both captures. VID_001's own (Metashape units)
    would be 10.4 cm, which is not physical.
  - Crash: `recoverPose` on an essential matrix with stacked solutions. Fixed in `EpipolarPoseEstimator` (the
    two-camera estimator): no solution fails, several use the first.
  - `nofov` then never initializes. Its scale comes from a 1-point RANSAC over the other lens' tracks with the ~2 cm
    baseline, which is degenerate for back-to-back lenses on handheld motion: λ = 0, 59 init attempts, then it waits
    for frames past the end of the data (stall).
  - `nofov` also reads a hard-coded ground-truth file from a PhD folder
    (`/home/deos/ce.debeunne/.../chariot4.txt`), only to print a "gt scale".
  - Conclusion: on this rig the scale must come from the IMU (`nofovvio`). Proposed baseline instead: `monovio` on the
    front lens.
- 2026-10-16, `nofov` init: the hard-coded ground-truth read is removed from `SLAMNonOverlappingFov::init()`
  (`pose_from_line`, the `chariot4.txt` reads, the "gt scale" print and the commented rescale). Built.
- 2026-10-16, item 1.3b, baseline B (`doc/vio_imu_fix/runs/x5_B/<seq>/monovio/run_00..03`):
  - Setup: `monovio` (camera_0 of the two-lens YAML only), `use_masks=1`, `mask_image_circle=1`, `downsampling=0.5`
    (960 px), KB4, TUM-VI placeholder IMU noise, no `estimate_td`. Cores 8–15 / 16–23, nice 10.
  - Scoring: `eval_x5.py` on `trajectory_tum.txt` (`export_tum.py --frame cam0`) at the reference's 4 fps frames;
    Sim(3) on the positions. Errors in reference (SfM) units; the reference is not metric yet (cube pending).
  - All 8 runs complete, no re-initialization. VIO init takes ~2.6–3.1 s, hence the coverage below 1.
  - VID_012 (191 reference frames, path 30.7 units):

    | run | coverage | ATE | median | max | scale (units/m) | rot RMS | gravity |
    |---|---|---|---|---|---|---|---|
    | 00 | 0.948 | 0.561 | 0.096 | 6.24 | 0.636 | 1.30° | 0.39° |
    | 01 | 0.948 | 0.067 | 0.046 | 0.38 | 0.656 | 0.95° | 0.23° |
    | 02 | 0.906 | 0.058 | 0.035 | 0.34 | 0.650 | 0.70° | 0.28° |
    | 03 | 0.901 | 0.081 | 0.050 | 0.57 | 0.647 | 1.25° | 0.49° |

    - run_00's ATE comes from its last 0.7 s: at the arena exit tracking is lost and the IMU-only propagation
      diverges (6.2 units at 47.65 s). Before 46.9 s (`--until 46.9`): ATE 0.050–0.057 for all 4 runs, gravity
      0.23–0.46°.
  - VID_001 (103 reference frames, path 39.5 units):

    | run | coverage | ATE | median | max | scale (units/m) | rot RMS | gravity |
    |---|---|---|---|---|---|---|---|
    | 00 | 0.883 | 0.122 | 0.113 | 0.27 | 2.836 | 1.82° | 0.46° |
    | 01 | 0.883 | 0.145 | 0.131 | 0.29 | 2.877 | 1.71° | 0.21° |
    | 02 | 0.893 | 0.156 | 0.151 | 0.33 | 2.860 | 2.30° | 0.34° |
    | 03 | 0.883 | 0.205 | 0.186 | 0.55 | 2.914 | 2.13° | 0.32° |

  - Relative to path length: VID_012 ≈ 0.2 %, VID_001 ≈ 0.4 %. Gravity agrees with the reference to < 0.5°.
  - Scale: the fitted scale is steady within each capture (VID_012 0.636–0.656, VID_001 2.84–2.91 units per VIO
    metre). The accelerometer reads |a| ≈ 9.87 m/s² (metric), so if the VIO scale is right, VID_012's SfM unit is
    ~1.55 m, not ~1 m. Consequence for `nofovvio`: the rig baseline would then be ~2.9 cm (VID_012) / ~3.6 cm
    (VID_001's Metashape value), not the 1.9 cm now in the rear T_BS. The cube must settle this before 3.3.
- 2026-10-16, item 1.4, replay in the presentation:
  - One more baseline-B run per capture with the keypoint log (`doc/vio_imu_fix/runs/x5_B_viewer/<seq>/monovio/run_00`):
    VID_012 ATE 0.048 units (0.039 before 46.9 s), scale 0.648, gravity 0.05°; VID_001 ATE 0.139 units, scale
    2.894, gravity 0.81°. Both are inside the 4-run spread, and neither re-initialized.
  - `build_x5_replay.py` writes the viewer's format: the reference rotated to z up and divided by the run's Sim(3)
    scale (VIO metres), per-KF errors against the linearly interpolated reference, keypoints at 960 px, and the front
    lens video at 384 px and 29.97 fps. KF ATE in VIO metres: VID_012 0.065, VID_001 0.050.
  - Viewer (`sources/sadvio-replay/index.html`): it draws only the runs a sequence has (the VO panel is hidden for
    X5), adds the "Insta360 X5" group, and shows the per-sequence reference label, note and metrics. The deck was
    rebuilt with `tools/build_presentation.py` (+5.4 MiB). The previous files are in `backups/20261016_x5_replay/`.
  - Checked: the viewer standalone in headless Firefox (VID_012, VID_001, and V1_01 unchanged), and the new deck holds
    the X5 assets and the new viewer. The full 1.3 GB deck stalls at "Opening embedded viewer" in headless Firefox,
    and the previous deck stalls the same way, so the deck itself is still to be checked in a desktop browser.
- 2026-10-16, VID_012 scale from GUTWrap (`gutwrap-50`, `insta360x5/VID_012/cube_corners_VID_012.csv`, README):
  - 1.37 m per SfM unit (1.31–1.45, about ±5 %), from the two 1 × 1 × 0.5 m boxes Shreyank identified in the arena:
    box heights 0.381 / 0.346 units (1.31 / 1.45 m per unit), from the trained splat's dense depth in rear frame 25
    (the sparse SfM has almost no points on the dark box tops). The 1 m widths give 1.29–1.45. The floor planes are
    the noisier part; box edges soft by a few cm. VID_001: still needs the office measurement.
  - Rig baseline: 0.01864 units × 1.37 = 2.55 cm (2.4–2.7), not the 1.9 cm assumed. Rear T_BS translation in both
    X5 YAMLs (`doc/vio_imu_fix/configs/dataset/insta360x5_vid_0xx.yaml`) set to [0.022615, 0.00026, 0.011845] m,
    with the source in the header.
  - `eval_x5.py`: `M_PER_UNIT` (VID_012: 1.37, 1.31–1.45) or `--m-per-unit`; adds `ate_m`, `vio_scale_ratio` and
    its range. Baseline B on VID_012, rescored:

    | run | ATE (m, Sim3) | ATE before 46.9 s (m) | VIO scale (×) | over the scale range |
    |---|---|---|---|---|
    | x5_B 00 | 0.768 | 0.072 | 1.147 | 1.08–1.20 |
    | x5_B 01 | 0.092 | 0.075 | 1.113 | 1.05–1.16 |
    | x5_B 02 | 0.080 | 0.069 | 1.124 | 1.06–1.18 |
    | x5_B 03 | 0.111 | 0.079 | 1.127 | 1.07–1.18 |
    | x5_B_viewer 00 | 0.066 | 0.054 | 1.126 | 1.06–1.18 |

    VIO lengths ×1.13 too long: item 3.0.
  - Replay: VID_012's reference is now drawn in metres (1.37 m per unit), with the VIO aligned without scale, so the
    scale error shows (SE3 KF ATE 0.663 m; Sim3 ATE 0.066 m on the metrics card). VID_001 unchanged (scaled to the
    VIO). Deck rebuilt; the README says so.
- 2026-10-16, GUTWrap (`gutwrap-50`) on item 3.0: X5 rolling-shutter readout 21.244 ms per frame (INSV metadata
  and Insta360's lens profile, both captures; direction not stated); exposure 1/60 s (VID_012). IMU white noise
  from the recordings' 150–450 Hz PSD: gyro 3.8–6.1e-5 rad/s/√Hz, accel 2.8e-4–1.3e-3 m/s²/√Hz (lower bounds if
  the IMU filters internally); suggested VIO values ×5: gyro 2.5e-4, accel 3.5e-3 (ours: 1.6e-4 / 2.8e-3, TUM-VI).
  Random walks need ~1 h static. Their box scale is less certain than ±5 %: box A's top came out 11 % non-square;
  they offer a multi-view measurement (triangulated corners or mesh edges). Also: Insta360's factory dual-lens
  calibration (`offset_v3`) is in the metadata, not decoded.
- 2026-10-16, item 3.0 tests (VID_012, `monovio`, 2 runs each):
  - `estimate_td=1` (`x5_s30_td`): td converges to −4.97 / −4.86 ms (the YAML has −5 ms); VIO scale ×1.133 /
    ×1.116. Not the cause.
  - GUTWrap's noise values (`x5_s30_gwimu`, test YAML `insta360x5_vid_012_gwimu.yaml`): ×1.130 / ×1.127 (before
    46.9 s; run_01 diverges in the arena-exit tail). ATE before 46.9 s 0.078 / 0.085 m (baseline 0.054–0.079). Not
    the cause.
  - Local scale in 10 s windows (Sim3 per window, 9 runs, 3 configurations): ×1.10–1.15 in every window of every
    run, mean ×1.114–1.139 per window, already ×1.12 in the first window (3–13 s); no growth over time, no link to
    rotation speed (window mean |ω| 0.40–0.56 rad/s; correlation −0.4 over 7 overlapping windows, not significant).
  - Reading: a VIO scale error from noise, td, initialization or rolling shutter would vary between runs and over
    time; this factor is constant across runs, settings and time. The IMU is metric (|a| 9.87 m/s², gyro–camera
    rotation residual 0.04°, td consistent), so the factor most likely sits in the reference's 1.37 m/unit, which
    the VIO puts at ~1.55. Next: GUTWrap's multi-view box measurement (user's go needed); rolling shutter (6.1) on
    hold until then.
- 2026-10-16, user decisions: GUTWrap is doing the multi-view box measurement. GUTWrap's IMU white-noise values
  (gyro 2.5e-4, accel 3.5e-3) are now in both X5 YAMLs; the random walks stay TUM-VI placeholders. The test YAML
  `insta360x5_vid_012_gwimu.yaml` is deleted. Milestone-1 note drafted in `doc/nofovvio/notes/SaDVIO_reply_1.md`,
  waiting on the user's review before it goes to `GUTWrap_Discussion/`. Next: 3.1.
- 2026-10-16, revised VID_012 scale from GUTWrap (`gutwrap-50`, `cube_corners_VID_012.csv` v2, README): 1.525 m per
  SfM unit (1.507–1.545, ±1.3 %), from the SfM points on the boxes' printed side walls (track ≥ 3, wall plane = median
  of its points, ~1 cm scatter): widths A 0.6560 × 0.6563, B 0.6634 × 0.6474 units, mean 0.6558 ± 0.0056 units per
  metre. v1 (1.37) segmented the box tops in the splat's rendered depth, where soft edges made them look larger.
  - Applied (user's go): `eval_x5.M_PER_UNIT` VID_012 = (1.525, 1.507, 1.545); rear T_BS translation in both X5
    YAMLs = [0.025173, 0.00029, 0.013185] m (2.84 cm; 2.81–2.88); replay and deck rebuilt (VID_012 KF ATE SE3
    0.093 m, Sim3 0.074 m, VIO scale ×1.011).
  - Rescored VID_012 (m, Sim3):

    | run | ATE | ATE before 46.9 s | VIO scale (×) | over the scale range |
    |---|---|---|---|---|
    | x5_B 00 | 0.855 | 0.080 | 1.031 | 1.018–1.043 |
    | x5_B 01 | 0.102 | 0.084 | 1.000 | 0.987–1.012 |
    | x5_B 02 | 0.088 | 0.077 | 1.009 | 0.996–1.022 |
    | x5_B 03 | 0.123 | 0.087 | 1.013 | 1.000–1.025 |
    | x5_B_viewer 00 | 0.074 | 0.060 | 1.012 | 0.999–1.024 |
    | x5_s30_td 00 | 0.091 | 0.086 | 1.018 | 1.004–1.030 |
    | x5_s30_td 01 | 0.111 | 0.110 | 1.002 | 0.989–1.014 |
    | x5_s30_gwimu 00 | 0.131 | 0.087 | 1.016 | 1.002–1.028 |
    | x5_s30_gwimu 01 | 1.808 | 0.094 | 1.110 (tail; 1.012 before 46.9 s) | — |

  - Item 3.0 closed; rolling shutter (6.1) is not needed for the scale. Earlier tables in this log at 1.37 m/unit are
    superseded (metres there are ×0.898 of the true values).
- 2026-10-16, item 3.1, `nofovvio` (`cpp/src/slamNonOverlappingFovVIO.cpp`, class in `slamCore.h`):
  - Initialization: `SLAMMonoVIO::init()` on camera 0 (visual-only map, then the inertial initialization fixes scale
    and gravity); camera 1 detects on the last KF once the map is metric, so its landmarks, triangulated over time
    through its extrinsics, are metric from the start. Camera 1 is unused during the ~3 s of initialization.
  - Front end (as `monovio`'s, for both cameras): KLT per camera, essential / PnP RANSAC per camera (outliers),
    epipolar filtering per camera, one ESKF update over both cameras' landmarks
    (`ESKFEstimator::estimateTransformBetweenCameras`: body-frame state, each observation through its own camera's
    extrinsics; the single-camera version uses the first match's camera for all), IMU-only pose when both cameras
    together have ≤ 5 tracks, KF vote on both cameras (parallax with each camera's rotation compensated, mono rule),
    landmark init and map-landmark recovery on both cameras. Back end: `monovio`'s (the window optimizer takes each
    observation through its own camera).
  - Shared code, behaviour unchanged for the other modes: `ESKFEstimator`'s IMU step moved into a helper
    (`imuUpdate`) used by both updates; `SLAMMonoVIO::_last_IMU` protected; `nofovvio` added where the modes are
    listed (config validation: VIO, needs 2 cameras, loop_graph_dof 4 allowed; IMU loading; loop-closure window
    options as `monovio`; mesher on both cameras as `nofov`; `max_lost_frames` default 10 as `monovio`; both
    `main.cpp`; `run_sadvio.py`).
  - Tests (`residual_test.cpp`): `ESKFMultiCameraTest.oneCameraMatchesTheSingleCameraUpdate` (one camera with a 3 cm
    offset and a half turn: same estimate and covariance as the single-camera update to second order, ~1e-6 for a
    ~0.04 correction; the corrections are [Exp(θ), ρ], which change error frames exactly only to first order) and
    `twoCamerasRecoverTheMotion` (back-to-back cameras, exact observations). Suite: 104 / 105 (the flaky
    `LineFeatureMatching`).
  - Smoke runs (`nfv_smoke`): VID_012 (masks, 960 px) completes, no re-initialization and no "Not enough tracked"
    frame (`monovio` lost tracking at the arena exit): Sim3 ATE 0.086 m (0.079 before 46.9 s), VIO scale ×1.010,
    gravity 0.24°, rotation RMS 0.69°, max error 0.18 m, coverage 0.953. EuRoC V1_01: no reset, ATE 0.073 m.
- 2026-10-16, item 3.2 (`nfv_euroc`, default config, 2 runs per mode, `eval_traj.py`):

  | | `bimonovio` | `monovio` | `nofovvio` |
  |---|---|---|---|
  | V1_01 ATE (m) | 0.052 / 0.065 | 0.237 / 0.207 | 0.120 / 0.133 |
  | MH_01 ATE (m) | 0.116 / 0.119 | 1.044 / 0.212 | 0.146 / 0.107 |
  | resets | 0 | 0 | 0 |
  | RPE 1 s (m), V1_01 / MH_01 | 0.052 / 0.016 | 0.055 / 0.04–0.13 | 0.051 / 0.017 |
  | wall (s), V1_01 / MH_01 | 16 / 20 | 16 / 20 | 26 / 34 |

  `nofovvio` runs on a pinhole pair with no reset, much better than one camera, as good as stereo VIO on MH_01 and
  between the two on V1_01: it never matches points between the cameras (by design), and on EuRoC's overlapping
  pair that matching is what gives stereo VIO its depth. ~1.6× the time (two cameras tracked). The rotation RMS
  (~5° on V1_01 for every mode) is the evaluation's GT-orientation convention, common to all modes.
- 2026-10-16, item 3.3 (in progress), `nofovvio` on the X5:
  - Bug found and fixed: `SLAMCore::shouldInsertKeyframe` (used by the `monovio` initialization that `nofovvio`
    reuses) applied the mono landmark rule only for `mono` / `monovio`; under `nofovvio` the initialization voted KFs
    much more often, the 10 initialization KFs spanned too little motion, and the inertial initialization was rejected
    (scale undetermined, sigma log scale 0.11–0.36 > 0.1) until 21.7 s of VID_001's 25.8 s (coverage 0.13–0.19).
    `nofovvio` now takes the mono rule there. `monovio` with GUTWrap's noise values initializes at once on VID_001
    (sigma 0.036–0.044), so the noise change was not the cause. EuRoC (3.2) ran before the fix: to redo.
  - VID_012 (5 runs, 960 px; `nfv_smoke`, `nfv_x5`, `nfv_x5b`): all complete, no re-initialization, no IMU-only
    frame; Sim3 ATE 0.058–0.086 m (before 46.9 s: 0.057–0.080), VIO scale ×1.005–1.017, gravity 0.24–0.28°.
    1920 px (`nfv_x5_full`, 1 run): 0.082 m (0.047 before 46.9 s), scale ×1.001; 96 s for 47.7 s of video.
  - VID_001: with VID_012's rig (current YAML) `nofovvio` is behind the front lens alone. Diagnostics:
    - Rear lens alone (`monovio` on a lens-swapped view, `scratch/x5_rear/` symlinks, `*_rear.yaml` variants): VID_001
      0.154 / 0.123 units (as good as the front lens); VID_012 0.21 / 0.56 units, tracking lost 3 times (white walls).
    - Lens time offsets (`estimate_td` per lens): VID_001 front −2.0 ms, rear −2.4 ms; VID_012 front −4.9 ms, rear
      −5.2 ms. No offset between the lenses. (VID_001's YAML has 0; the estimate is ~−2 ms.)
    - Rig translation (6 runs each, ATE in units): VID_012's rig 0.218 0.211 0.211 0.119 0.208 0.161 (mean 0.188);
      VID_001's own Metashape rig direction scaled to 2.84 cm, body [0.028166, −0.000781, 0.003557] m: 0.167 0.139
      0.178 0.165 0.130 0.149 (mean 0.155); rotation-only rig (2 runs) 0.133 / 0.255; `monovio` front (7 runs) mean
      0.149. The two SfM rigs disagree by ~22° in direction (VID_012: 13 mm sideways; VID_001: 3.6 mm, almost along
      the optical axis); a 2–3 cm baseline is weakly determined in either reconstruction.
  - Runtime: `nofovvio` and `monovio` take the same wall time on the X5 (76 s for VID_012, 42 s for VID_001 at
    960 px), so the runs are bound by reading and decoding two 1920 px PNGs per frame, not by the SLAM.
- 2026-10-16, item 3.3 results and decisions:
  - Rig: VID_001's rear T_BS translation is now its own Metashape rig direction at the 2.84 cm measured on VID_012
    ([0.028166, −0.000781, 0.003557] m); VID_012 keeps its COLMAP rig. With VID_001's direction, VID_012 was the same
    before the arena exit (mean 0.068 vs 0.067 m over 5 runs each) but 2 of 5 runs diverged in the last second (0 of 5
    with its own rig): each capture's rig estimate goes with its own intrinsics. Diagnostic YAMLs, the lens-swapped
    view and its `run_sadvio.py` entries are removed.
  - Final results, 960 px, masks, GUTWrap's IMU white noise:

    | | VID_012 (metres, 1.525 m/unit) | VID_001 (SfM units) |
    |---|---|---|
    | `nofovvio` runs | 5 | 7 |
    | complete, no re-initialization | 5 / 5 | 7 / 7 |
    | Sim3 ATE | 0.058–0.086 m | 0.130–0.229 (mean 0.165) |
    | ATE before 46.9 s | 0.057–0.080 m | — |
    | gravity-only ATE | 0.090–0.126 m | — |
    | VIO scale | ×1.005–1.017 | — |
    | gravity vs reference | 0.24–0.28° | 0.1–0.5° |
    | frames on the IMU alone ("Not enough tracked") | 0 | 0 |
    | `monovio` (front lens) for comparison | Sim3 0.074–0.123 m, 1 of 5 diverged at the exit | mean 0.149 (7 runs) |

    1920 px, VID_012 (1 run): Sim3 0.082 m (0.047 before 46.9 s), scale ×1.001, 96 s for 47.7 s of video (960 px:
    76 s, the same as `monovio`: bound by reading two 1920 px PNGs per frame).
  - Reading: on VID_012 two lenses keep tracking through the arena exit, where the front lens alone loses it; the ATE
    is otherwise the same. On VID_001 the front lens alone is slightly better; the rig translation, weakly determined
    in both reconstructions, is the limiting input.
  - EuRoC (3.2) after the KF-rule fix (`nfv_euroc2`): V1_01 0.084 / 0.097 m (was 0.120 / 0.133), MH_01 0.107 /
    0.124 m; no reset.
- 2026-10-16, item 3.4: a `nofovvio` run with `log_pose_covariance: 1` (`nfv_out`) writes `cov_window.csv` (window
  pose covariance, newest / previous / oldest KF, 18×18) and `cov_front.csv` (per-frame ESKF covariance) besides the
  usual logs; `export_tum.py` gives the per-frame TUM poses (cam0 or body), the biases and the T_BS used, and now the
  dataset's `dt_imu_cam` with `dt_imu_cam_estimated: false` when `estimate_td` is off.
- 2026-10-16, `run_sadvio.py`: two lanes with the same label could race on removing the shared empty scratch folders
  (one lane crashed after its first run); the cleanup now tolerates it.
- 2026-10-16, regression subset before the `nofovvio` commit: `doc/loop_closure/tools/run_final_matrix.sh` (new env
  options PREFIX and SEQS) with snapshot `nf2` (current code), 6 shipped configurations × room1, MH_01, V1_01,
  V2_03 × 2 runs, labels `reg_nf2_*`, against `fin_*` (snapshot `fin1`, 2026-10-10). No crash; VIO configurations
  within run-to-run noise on window and loop-closed ATE. Two cells differ from `fin1`: `vo_mono` V2_03 (resets 17 / 15
  vs 5 / 6, ATE better: 0.37 / 0.39 vs 1.34 / 1.18 m) and `vio_noprior` V2_03 (loop ATE 0.24 / 0.28 vs 0.16 / 0.21 m).
  The pre-`nofovvio` snapshot `nf1` gives the same (`reg_nf1_*`: resets 18 / 19, ATE 0.34 / 1.10; loop 0.31 / 0.26):
  today's changes are neutral, and the shift predates them (commits after `fin1`). `vo_mono` is known to be noisy
  (older open item); the V2_03 reset count is worth a look later.
- 2026-10-16, replay: each X5 capture also shows a `nofovvio` run (purple, `nfv_viewer` runs: VID_012 Sim3 0.062 m,
  VID_001 0.162 units), with the rear lens's video and keypoints in its panel. `logKfFeatures` writes camera 1's
  features to `log_slam/kf_features_cam1.csv` in `nofovvio` (EXECO_KF_FEATURES_LOG only). Viewer: third run slot
  `nfv`, per-run video and camera label; `build_x5_replay.py --nfv`. Deck rebuilt.
- 2026-10-16, loop closure in the `nofov` modes (before 4.1): the loop closure is attached to every mode (the local
  map's discard callback), `nofov` and `nofovvio` included, on camera 0 only (`LoopClosure::addKeyframe`), and its
  verification drops rays more than ~78° off the axis (fisheye-safe). Test (`nfv_lc`, `nofovvio`, `loop_closure: 1`,
  bag of words, 1 run per capture): runs complete; VID_012 88 candidates, VID_001 6, none accepted. Descriptor matches
  between the candidate's landmarks and the query's keypoints: median 1, max 7 (verification needs 12 PnP inliers each
  way), 0 inliers. ATE with the loop-closed output: VID_012 0.113 m, VID_001 0.168 units (no correction applied).
  So on the X5 the matching itself fails, before any cross-lens question: the first thing to look at in 4.1.

