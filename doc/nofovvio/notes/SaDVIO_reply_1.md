# SaDVIO for the Insta360 X5: reply 1 (baseline)

*16 October 2026. Reply to `SaDVIO_feature_request.md`. SaDVIO branch `dense_devel`, commit `3b645c1`. Full log:
`doc/nofovvio/NOFOVVIO_LEDGER.md` in the SaDVIO repository.*

## In short

- SaDVIO now reads your captures as they are: the KB4 fisheye model beyond 180°, your per-frame masks, and the IMU.
- Using the front lens and the IMU, it tracks VID_012 and VID_001 to the end, with no re-initialization in any run.
  On VID_012 the error is about 7–12 cm over the whole run, the VIO's scale agrees with your box measurement within
  about 1 %, and gravity agrees with your reference within 0.5°.
- The existing two-camera mode (`nofov`) cannot start on the X5. The two-lens IMU mode you asked for (`nofovvio`) is
  the next step.

## Status by requested feature

| # | Request | Status |
|---|---|---|
| 1 | `nofovvio` mode (two lenses + IMU) | Next. The front-lens baseline below is the comparison point. |
| 2 | KB4 / OpenCV-fisheye model beyond 180° | Done. Projection and unprojection go through the angle from the axis, so points behind the image plane work. Valid up to `max_theta_deg` (default 100°). |
| 3 | Per-image masks | Done. Your `camN/mask/<ts>.png` files, turned on with `use_masks: 1`. An image-circle mask from the KB4 model is added by default (`mask_image_circle`). |
| 4 | Offline batch mode and outputs | Partly. Per-frame TUM poses for the front camera, the final time offset and the IMU biases. The window covariances come with `nofovvio`. |
| 5 | Loop closure in the `nofov` modes | After `nofovvio`; cross-lens loop candidates are planned. |
| 6 | Online extrinsics and time offset | Time offset: `estimate_td` works on your data (below). Extrinsics refinement: later, if needed. |
| 7 | Rolling shutter | Not needed so far: the scale is right without it (below). Thank you for the readout time (21.244 ms); we keep it for later. |
| 8 | Export for SfM (COLMAP text) | After `nofovvio`. |
| 9 | Low-texture robustness (lines) | Later, if the results call for it. |
| 10 | Pose covariance | With `nofovvio`. |

## How we used your data

- **Calibration:** your VID_001 and VID_012 YAMLs, with two changes.
  - cx and cy are shifted by −0.5 px, because the KB4 model and OpenCV put pixel centres at integers, while COLMAP
    puts them at +0.5.
  - The rear lens's translation is set to the VID_012 rig baseline at your box scale (1.525 m per unit): 2.84 cm
    (VID_001 uses the same rig).
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
2. The readout direction of the rolling shutter (top to bottom?), in case we need it later.
3. Optional: about an hour of static IMU recording, to estimate the random walks instead of using TUM-VI's.

## Next on our side

1. **`nofovvio`:** each lens tracked on its own and both fed into the VIO back end, with the IMU initialization, bias
   handling and time offset already there. We first check it on EuRoC stereo used as two non-overlapping cameras, then
   score it on both captures as above: one lens against two.
2. **Outputs:** per-frame poses in a gravity-aligned metric world, time offset, biases, pose covariances, and the
   extrinsics as used.
3. **Then:** cross-lens loop closure and the COLMAP text export.
