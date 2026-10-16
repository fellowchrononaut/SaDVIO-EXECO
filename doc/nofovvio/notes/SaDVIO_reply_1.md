# SaDVIO for the Insta360 X5: reply 1 (baseline and two-lens VIO)

*16 October 2026. Reply to `SaDVIO_feature_request.md`. SaDVIO branch `dense_devel`. Full log:
`doc/nofovvio/NOFOVVIO_LEDGER.md` in the SaDVIO repository.*

## In short

- SaDVIO now reads your captures as they are: the KB4 fisheye model beyond 180°, your per-frame masks, and the IMU.
- Using the front lens and the IMU, it tracks VID_012 and VID_001 to the end, with no re-initialization in any run.
  On VID_012 the error is about 7–12 cm over the whole run, the VIO's scale agrees with your box measurement within
  about 1 %, and gravity agrees with your reference within 0.5°.
- The existing two-camera mode (`nofov`) cannot start on the X5. The two-lens IMU mode you asked for (`nofovvio`) now
  exists. On VID_012 its error is 6–9 cm, about the same as the front lens alone, and it is more robust: it keeps
  tracking where the front lens alone loses it. On VID_001 it is about level with the front lens; the position of the rear lens relative
  to the front one is the limiting input, and your two reconstructions disagree on it.
- Loop closure runs in this mode, but your two captures give it almost no revisit to close (below); cross-lens loops
  are our next step.

## Status by requested feature

| # | Request | Status |
|---|---|---|
| 1 | `nofovvio` mode (two lenses + IMU) | Done (below). |
| 2 | KB4 / OpenCV-fisheye model beyond 180° | Done. Projection and unprojection go through the angle from the axis, so points behind the image plane work. Valid up to `max_theta_deg` (default 100°). |
| 3 | Per-image masks | Done. Your `camN/mask/<ts>.png` files, turned on with `use_masks: 1`. An image-circle mask from the KB4 model is added by default (`mask_image_circle`). |
| 4 | Offline batch mode and outputs | Done. Per-frame TUM poses of the front camera (or the IMU) in a gravity-aligned metric world, the time offset, the IMU biases, the camera–IMU transforms used, and with `log_pose_covariance: 1` the pose covariances. |
| 5 | Loop closure in the `nofov` modes | Runs, on the front lens; your captures offer almost no revisit to close (below). Cross-lens loops next. |
| 6 | Online extrinsics and time offset | Time offset: `estimate_td` works on your data (below). Extrinsics refinement: later, if needed. |
| 7 | Rolling shutter | Not needed so far: the scale is right without it (below). Thank you for the readout time (21.244 ms); we keep it for later. |
| 8 | Export for SfM (COLMAP text) | Planned as a SaDVIO option, after the loop closure. |
| 9 | Low-texture robustness (lines) | Later, if the results call for it. |
| 10 | Pose covariance | Done: window pose covariance and per-frame covariance (`log_pose_covariance: 1`). |

## How we used your data

- **Calibration:** your VID_001 and VID_012 YAMLs, with two changes.
  - cx and cy are shifted by −0.5 px, because the KB4 model and OpenCV put pixel centres at integers, while COLMAP
    puts them at +0.5.
  - The rear lens's translation is the rig baseline at your box scale (1.525 m per unit): 2.84 cm. VID_012 uses its
    COLMAP rig direction; VID_001 uses its own Metashape rig direction at the same length (see the two-lens
    section).
- **Time offset:** we use your sign convention unchanged.
- **Resolution:** 960 px (downsampling 0.5).
- **IMU noise:** the baseline runs used TUM-VI placeholders. From now on we use your suggested white-noise values
  (gyro 2.5e-4, accel 3.5e-3); they gave the same results on VID_012. The random walks stay TUM-VI placeholders.

## Baseline: front lens + IMU (`monovio`)

The front lens alone with the IMU, using KB4 and the masks, 5 runs per capture. Scoring is against
your `reference_poses_cam0.csv` at its 4 fps frames, after a Sim(3) alignment of the positions. VID_012 is in
metres at your 1.525 m per SfM unit; VID_001 stays in SfM units until it is measured.

| | VID_012 | VID_001 |
|---|---|---|
| Completes, no re-initialization | 5 / 5 | 5 / 5 |
| Coverage of the reference frames | 0.90–0.95 (the VIO starts after ~3 s) | 0.88–0.89 |
| ATE, whole run | 0.074–0.123 m (one run: 0.86 m, see below) | 0.12–0.21 SfM units (not metric yet) |
| ATE before 46.9 s | 0.060–0.087 m | — |
| VIO scale vs your box measurement | ×1.000–1.031 (×0.99–1.04 over its ±1.3 %) | — |
| Relative to path length | ~0.2 % | ~0.4 % |
| Rotation RMS after alignment | 0.7–1.3° | 1.7–2.3° |
| Gravity vs your reference | < 0.5° | 0.2–0.8° |
| Runtime (8 desktop CPU cores, shared) | ~76 s for 47.7 s of video | ~42 s for 25.8 s |

- **VID_012's last second:** tracking is lost when the camera leaves the arena, and the IMU-only propagation can
  drift. In one run this reached 6 m in the final 0.7 s. The second lens in `nofovvio` should help there.
- **Runtime:** this is about 1.6 times slower than real time at 960 px. SaDVIO runs on the CPU, so your RTX 5090
  does not change it much.
- **Replay:** both captures can be replayed in the EXECO presentation (SaDVIO → VO / VIO replays → Insta360 X5): the
  trajectory against your reference, the camera images, and the tracked keypoints.

## Two lenses + IMU (`nofovvio`)

How it works: it starts like the front-lens mode, and the IMU fixes the metric scale and gravity in the first ~3 s.
Then the rear lens joins. Each lens is tracked on its own (the lenses do not overlap, so no points are matched between
them), and every frame's pose comes from one update over the landmarks of both lenses. The sliding-window
optimization uses every observation of both lenses with the IMU. The scale comes from the IMU, not from the rig:
with a 2.8 cm baseline the rig alone cannot fix it.

Results (960 px, masks, your IMU noise values):

| | VID_012 (metres) | VID_001 (SfM units) |
|---|---|---|
| Runs, complete without re-initialization | 5 / 5 | 7 / 7 |
| ATE (Sim3), whole run | 0.058–0.086 m | 0.130–0.229 (mean 0.165) |
| Front lens alone, for comparison | 0.074–0.123 m (one run: 0.86 m) | mean 0.149 (7 runs) |
| ATE, gravity-only alignment (heading + position) | 0.090–0.126 m | — |
| VIO scale vs your box measurement | ×1.005–1.017 | — |
| Gravity vs your reference | 0.24–0.28° | 0.1–0.5° |
| Frames without any visual tracking | 0 | 0 |
| Runtime at 960 px / 1920 px | 76 s / 96 s for 47.7 s of video | 42 s for 25.8 s |

- **Robustness:** at VID_012's arena exit the front lens alone loses tracking and diverged by metres in one of five
  runs; with two lenses no run diverged and no frame fell back to the IMU alone.
- **Accuracy:** on these two short captures, two lenses are not more accurate than one. We expect the larger gain from
  loop closure across lenses on longer captures: walking back along a path, the rear lens sees what the front lens saw.
- **The rear lens's position (rig translation):** your two reconstructions disagree on its direction by about 22°.
  VID_012's COLMAP rig puts the rear centre 13 mm to the side; VID_001's Metashape rig puts it almost on the optical
  axis (3.6 mm). With VID_012's rig, VID_001 got worse (mean 0.188 instead of 0.155 over 6 runs each); with VID_001's
  rig, VID_012 diverged at the arena exit in 2 of 5 runs. So each capture now uses its own rig direction, at the
  2.84 cm length. A 2–3 cm baseline is weakly determined by SfM. We also checked that the two lenses share one clock:
  their time offsets to the IMU agree within ~1 ms (VID_012 −5 ms, VID_001 −2 ms; VID_001's YAML has 0).
- **Runtime:** reading and decoding two 1920 px PNG images per frame dominates; the two-lens mode takes the same time
  as one lens.
- **Replay:** both runs, front lens alone and two lenses, are in the EXECO presentation (SaDVIO → VO / VIO replays →
  Insta360 X5); the two-lens panel plays the rear lens with its tracked points.

## Loop closure

The loop closure runs in this mode (bag of words, then a geometric check in both directions) on the front lens, but
your two captures give it almost nothing to close: once SaDVIO has started, neither comes back to a mapped place
facing the same way.

- **VID_012** returns to its start only facing the opposite way (the closest pass is 1.8 m from the first keyframe,
  turned 104–146°). The front lens cannot recognise that; the rear lens can, so this is a cross-lens loop: our next
  step.
- **VID_001** comes back within 0.7 m of its start, turned 44–74°, but only in its last two seconds. Those keyframes
  are still in the sliding window when the capture ends, and the loop closure takes keyframes as they leave the window.
- The candidates it did check (88 on VID_012, 6–13 on VID_001) were look-alike places or turned views, and it
  rejected them, as it should.

So these captures cannot yet show whether the loop closure works on X5 images. Captures that come back to a place,
both facing the same way and facing the other way, would let us test it.

## Why `nofov` cannot be used on the X5

`nofov` gets its scale from the motion of one lens as seen by the other, through the distance between the two lenses.
On the X5 that distance is about 2.8 cm, too short for handheld motion: the scale estimate is zero, and initialization
never succeeds (59 attempts in our first try). The IMU is the X5's source of scale, which is what `nofovvio` will use.

## The scale question, settled

With your first box scale (1.37 m per SfM unit) the VIO's lengths on VID_012 came out 1.13 times too long. Before your
revised measurement arrived we tested the usual causes of a VIO scale error, and none of them changed it:

- **Time offset:** estimated online, it converges to −4.9 to −5.0 ms, your value; the factor did not move.
- **IMU noise:** your suggested values gave the same factor.
- **Over time:** fitting the scale in 10 s windows gave the same factor in every window of every run (9 runs, 3
  settings), from the first 10 s on, unrelated to the rotation speed, which a rolling-shutter bias would follow.
- **The IMU itself:** it is metric. Gravity reads 9.87 m/s², a 0.6 % gain error, and the gyro agrees with the camera
  rotation within 0.04°, as you found.

Your v2 measurement from the boxes' side walls (1.525 m per unit, ±1.3 %) puts the VIO at ×1.000–1.031 over the same
runs, mean ×1.01: within your uncertainty. So the VIO's metric scale on the X5 is right, and rolling-shutter
compensation is not needed for it. The rig baseline follows the same scale: 2.84 cm.

## What would help from your side

1. The VID_001 measurement, to score VID_001 in metres.
2. A better-determined rig translation, if you can get one: for example Insta360's factory dual-lens calibration
   (`offset_v3` in the metadata), or a rig estimate over several captures.
3. The readout direction of the rolling shutter (top to bottom?), in case we need it later.
4. Optional: about an hour of static IMU recording, to estimate the random walks instead of using TUM-VI's.
5. Captures that return to their starting point facing the other way, to test the cross-lens loop closure.

## Next on our side

1. **Cross-lens loop closure:** match a keyframe's front lens against past keyframes' rear lens and the other way round,
   enabled only with two non-overlapping cameras.
2. **COLMAP export as a SaDVIO option:** the final map as COLMAP text (cameras with KB4 as `OPENCV_FISHEYE`, images,
   points3D, and the two lenses as one rig), written at the end of a run, so you can start your SfM from it.
