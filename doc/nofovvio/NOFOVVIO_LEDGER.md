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
- [ ] 3.0 VIO scale on VID_012: `monovio` lengths are ×1.13 the true ones (×1.11–1.15 over 5 runs; 1.05–1.20 over
  the boxes' scale range), beyond the ±5 % of the measurement. Find the cause before 3.3: IMU noise (TUM-VI
  placeholders), `estimate_td`, rolling shutter (6.1), accelerometer scale (|a| 9.87 m/s², +0.6 % only), VIO init.
- [ ] 3.1 `SLAMNonOverlappingFovVIO`, derived from `SLAMBiMonoVIO`: the `nofov` front end (per-camera tracking and
  landmarks, no stereo matching between the lenses) on the VIO back end (IMU initialization, pre-integration, bias
  prior, marginalization, `estimate_td`). Mode validation in `slamParameters`.
- [ ] 3.2 Sanity check: EuRoC stereo run as non-overlapping, against `bimonovio`.
- [ ] 3.3 VID_001 and VID_012:
  - ATE after Sim(3), and after gravity-only alignment;
  - completion without re-initialization;
  - scale against a measured distance; gravity within 1°;
  - runtime at full and half resolution.
- [ ] 3.4 Outputs: per-frame TUM poses in a gravity-aligned metric world, final td and biases, window pose
  covariance (`log_pose_covariance`), and the camera–IMU extrinsics as used.

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

