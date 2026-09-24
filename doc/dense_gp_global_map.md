# Development document: dense GP global map (SLAMesh-style fusion and registration)

Branch: `dense_devel` · Started: 2026-09-23 · Status: implemented; synthetic test passing; first bag runs done (§6)

## 1. Goal

The dense stereo pipeline (`dense_depth: true`, `dense_mesh_method: "gp"`) meshes each marginalised
keyframe with a SLAMesh-style Gaussian-process (GP) mesher and publishes that keyframe's mesh on
`/dense_mesh`. Each message replaces the previous one; nothing is accumulated or written to disk.
Overlapping keyframes therefore never form one surface, and a naive merge of all keyframe meshes
gives stacked, slightly offset layers.

This development adds a **global GP map** that fuses keyframes into one world mesh the way SLAMesh
builds its LiDAR map, so that stereo and LiDAR meshes can be compared on equal terms:

- **A. Map update** (always on when the global map is on): SLAMesh's per-cell, per-vertex
  inverse-variance fusion.
- **B. Frame-to-model registration** (optional): SLAMesh's point-to-mesh registration of each new
  keyframe against the map before fusing, starting from the SaDVIO pose.

Poses come only from SaDVIO. No ground truth enters this code.

## 2. How SLAMesh does it

Source: `github.com/RuanJY/SLAMesh`, `src/map.cpp`, `src/cell.cpp` (read in the `dsol_simval`
container, commit `50b4356`).

Per scan, `slamesher_node.cpp` runs `processNewScan` (bucket the scan on the world grid at the pose
guess and GP-mesh each cell) → `registerToMap` → `updateMap`.

**Cells.** The world is a voxel grid (`grid`). A cell with enough points gets a PCA-based choice of
one to three prediction axes (`Cell::reconstructSurfaces`). Per axis, a GP predicts the axis
coordinate over a fixed `num_test × num_test` lattice of the other two coordinates, with a
variance per vertex. Because the lattice positions are fixed per cell and axis, vertex *i* of any
scan and vertex *i* of the map lie on the same line along the prediction axis.

**B. Registration** (`Map::registerToMap`, map.cpp:1068). Up to `register_times` iterations:
1. `findMatchPointToMesh`: for each scan lattice vertex, take the map vertex with the same lattice
   index from the same cell, or from cells shifted along the prediction axis by up to
   `cross_cell_overlap_length` (0 in the last iteration). The closest along the axis wins. The
   same-cell candidate is accepted regardless of variance; the others only if their variance is
   below `variance_register`.
2. The normal at the matched map vertex is the normalised mean of cross products with its lattice
   neighbours (neighbours farther than one cell are ignored); without neighbours it falls back to
   the scan-to-map direction.
3. A pair enters the solve only if both variances are below `variance_register` and the normal is
   valid.
4. `computeTPointToMesh`: one Ceres solve of `n · (R s + t − m)` over all pairs, weight 1,
   `HuberLoss(0.1)`, `DENSE_QR`, 10 iterations.
5. Apply the increment to the scan points and **re-mesh the scan** before the next iteration. Stop
   when `5‖R − I‖ + ‖t‖ < converge_thr` (1e-5) after the first iteration.

Registration residuals are not variance-weighted; variance only gates which vertices qualify.

**A. Map update** (`Map::updateMap`, map.cpp:1147; `Cell::updateVertices`, cell.cpp:282).
- Existing surface cell: for each axis present in both map and scan, fuse per vertex
  `h = (h_map σ²_scan + h_scan σ²_map) / (σ²_map + σ²_scan)`,
  `σ² = σ²_map σ²_scan / (σ²_map + σ²_scan)` when both variances are ≤ `variance_map_update`,
  otherwise keep the lower-variance vertex (ties keep the map). An axis the map cell lacks is not
  added.
- Map cell that is not a surface yet, scan cell that is: append the scan's raw points and re-run
  the GP on the union.
- Unseen cell: insert.

SLAMesh has no loop closure.

## 3. What was built

| File | Change |
|---|---|
| `cpp/include/isaeslam/stereo/GPGlobalMap.h`, `cpp/src/stereo/GPGlobalMap.cpp` | **New.** The global map: cells, registration (B), fusion (A), mesh extraction, PLY output. |
| `cpp/include/isaeslam/stereo/GPMeshDetail.h` | **New.** Declares the per-cell GP building blocks so both mesher and map use one implementation. |
| `cpp/src/stereo/GPMeshEstimator.cpp` | Anonymous namespace renamed to `gp_detail`; three structs moved to the header. No behaviour change. |
| `cpp/include/isaeslam/stereo/MarginalDepthInjector.h`, `.cpp` | Global-map path in the dense worker; optional lossless keyframe queue; registration log; periodic PLY save. |
| `cpp/include/isaeslam/slamParameters.h`, `cpp/src/slamParameters.cpp` | 18 new config keys. |
| `cpp/src/slamBiMono.cpp`, `cpp/src/slamBiMonoVIO.cpp` | Copy the new keys into `MarginalDepthConfig`. |
| `ros/config/config.yaml` | New keys, all off by default. |

With the defaults (`dense_gp_global_map: false`, `dense_keep_all_keyframes: false`) the pipeline
behaves exactly as before.

### 3a. Related fixes made along the way

| File | Change |
|---|---|
| `cpp/src/slamBiMono.cpp`, `cpp/src/slamBiMonoVIO.cpp` | **Bug fix: double undistortion in the dense path.** For a `radial-tangential` camera the data provider undistorts the images on load and stores the *new* camera matrix in `K`, but keeps the *original* coefficients in `d`. The dense injector received both and undistorted the already-undistorted images again. It now gets zero distortion for cameras with `undistort == true`. Configs with zero distortion (e.g. `airsim`, `realsense_optitrack`) are unaffected. |
| `ros/config/dataset/realsense_optitrack_kalibr0716.yaml` | **New.** Real D455 bags with the Kalibr calibration of `kalibr_work/rosbag2_2026_09_03-08_07_16` (reprojection ±0.073 / 0.076 px): per-camera intrinsics, radtan distortion and the measured cam0→cam1 extrinsic (95.12 mm, 0.22° toe-in). The SDK's `image_rect_raw` assumes one principal point for both IR cameras (`realsense_optitrack.yaml`: cx 425.969 for both); Kalibr measures cx₀ − cx₁ = 0.883 px, which SaDVIO's own rectification now takes into account. Background: `Simulator_Validation/RealSense_Captures_060526/scale_attribution_20260906/KALIBR_HANDOFF.md`. |

### Data flow

```
keyframe marginalised (slamBiMono / slamBiMonoVIO)
  └─ MarginalDepthInjector::queueFrame      images + T_w_rectcam (SaDVIO pose of rectified left cam)
       └─ worker: SGBM → disparity
            └─ integrateGlobalMap
                 ├─ points in the rectified camera frame (same selection as GPMeshEstimator)
                 ├─ GPGlobalMap::integrate(points_cam, T_w_rectcam)
                 │    T_guess = carried correction · T_vo
                 │    bucket + GP at T_guess
                 │    [B] registerToMap → correction C (accepted or reverted)
                 │    [A] updateMap with the keyframe at C · T_guess
                 ├─ /dense_point_cloud moved by the same correction
                 ├─ /dense_mesh = whole global mesh
                 ├─ log_slam/dense_gp_registration.csv row
                 └─ every dense_gp_save_every keyframes: log_slam/dense_gp_global_mesh.ply
```

Registration corrections stay inside the map (and the carried offset); they are not fed back into
SaDVIO's state or published poses.

## 4. Configuration

All keys live in `ros/config/config.yaml` next to the existing `dense_gp_*` keys.

| Key | Default | Meaning |
|---|---|---|
| `dense_keep_all_keyframes` | `false` | `true`: FIFO queue, no keyframe is dropped. Needed for a complete map. |
| `dense_gp_global_map` | `false` | A: fuse keyframes into one GP map. |
| `dense_gp_variance_map_update` | `0.5` | SLAMesh `variance_map_update` (its `online.yaml` value). |
| `dense_gp_max_raw_points_per_cell` | `2000` | Raw points kept per non-surface cell. |
| `dense_gp_register` | `false` | B: registration before fusing. Needs `dense_gp_global_map`. |
| `dense_gp_register_times` | `5` | SLAMesh `register_times`. |
| `dense_gp_variance_register` | `0.1` | SLAMesh `variance_register` (its code default; see §6). |
| `dense_gp_cross_cell_overlap_length` | `1` | SLAMesh `cross_cell_overlap_length`. |
| `dense_gp_register_converge_thr` | `1e-5` | SLAMesh `converge_thr`. |
| `dense_gp_register_huber` | `0.1` | Huber delta. |
| `dense_gp_register_min_matches` | `30` | Below this many pairs, keep the SaDVIO pose. |
| `dense_gp_register_max_translation` | `0.5` | m. Larger corrections are rejected. |
| `dense_gp_register_max_rotation_deg` | `10` | deg. Larger corrections are rejected. |
| `dense_gp_register_carry_correction` | `true` | Start each keyframe from the last accepted correction. |
| `dense_gp_register_depth_weighting` | `false` | Weight residuals by `min(1, (ref/z)²)`. |
| `dense_gp_register_depth_ref` | `2.0` | m, for the above. |
| `dense_gp_global_mesh_path` | `log_slam/dense_gp_global_mesh.ply` | Output mesh; the CSV goes in the same folder. |
| `dense_gp_save_every` | `5` | Keyframes between saves; `0` never saves. |

To run A only: `dense_depth: true`, `dense_mesh_method: "gp"`, `slam_mode: "bimono"` (or
`bimonovio`), `dense_gp_global_map: true`, `dense_keep_all_keyframes: true`. For A + B, add
`dense_gp_register: true`.

## 5. Deviations from SLAMesh

Each is marked `DEVIATION` in `GPGlobalMap.cpp`.

1. **Initial guess.** SLAMesh predicts the next pose from its own trajectory; here the guess is the
   SaDVIO keyframe pose, optionally with the last accepted correction carried forward.
2. **Too few pairs.** SLAMesh always solves. A stereo keyframe sees a narrow frustum, so early or
   off-map keyframes can have almost no overlap; below `min_matches` the SaDVIO pose is kept.
3. **Plausibility gate.** Corrections beyond `max_correction_translation` /
   `max_correction_rotation_deg` are rejected.
4. **Raw-point cap.** SLAMesh keeps all raw points of a cell; dense stereo would grow without
   bound, so non-surface cells keep at most `max_raw_points_per_cell` (uniform stride).
5. **Solver parameterisation.** SLAMesh optimises quaternion + translation with an analytic
   Jacobian and its own SE(3) parameterisation; here the same residual is auto-differentiated over
   angle-axis + translation (Ceres 2.2 removed `LocalParameterization`). Same cost, same loss.
6. **Optional depth weighting** (off by default): stereo depth error grows with z², which SLAMesh's
   uniform weighting ignores.
7. **Not ported:** registration time budget, viewed-direction bookkeeping and old-cell margining,
   TSDF visualisation, SLAMesh's display variance threshold (`variance_map_show`). The global mesh
   uses SaDVIO's existing face filter (`dense_gp_max_variance`) and seam stitching.

## 6. Verification

### Build

Built in the container workspace `/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel` (see §8): standalone
`cpp/build` and the ROS 2 package. The new files compile without warnings under `-Wall`.

### Synthetic registration test

A room corner (floor `z = 0`, walls `x = 4` and `y = 4`, 5 mm noise) seen by two cameras with the
same frustum limits as a keyframe. Keyframe 1 is fused at its true pose; keyframe 2 is handed in
with a known error of 7.8 cm and 2.0°. The table gives keyframe 2's remaining pose error.

| Run | `variance_register` | Iterations | Remaining error |
|---|---|---|---|
| A only | – | – | 7.8 cm, 2.00° (unchanged, as intended) |
| A + B | 0.6 | 5 | 2.86 cm, 0.51° |
| A + B, **started at the true pose** | 0.6 | 5 | 2.82 cm, 0.50° |
| A + B | 0.6 | 20 | 3.06 cm, 0.51° |
| A + B | 0.4 | 5 | 1.89 cm, 0.35° |
| A + B | 0.2 | 5 | 1.02 cm, 0.19° |
| A + B | **0.1** | 5 | **0.14 cm, 0.024°** |
| A + B | 0.05 | 5 | 0.09 cm, 0.012° |
| A + B | 0.1 | 20 (converged at 14) | 0.04 cm, 0.006° |

**Finding.** With SLAMesh's `online.yaml` gate (`variance_register: 0.6`) registration settles
about 3 cm from the truth, even when it starts there. Keyframes cover their cells only partly, so
many lattice vertices are GP extrapolations with variance in the 0.1–0.6 range and biased heights;
the loose gate lets them into the solve. At SLAMesh's code default (`0.1`) the bias disappears. The
default here is therefore 0.1. A 360° LiDAR scan covers its cells fully, which is presumably why
0.6 works for SLAMesh's own demo.

`variance_map_update` (0.5) is left at the `online.yaml` value. It is untested whether the same
effect degrades fusion: fused variances shrink with every update, so biased extrapolated vertices
may gain weight over many keyframes (see §7).

The test program is not part of the repository.

### Bag runs: ExECoSim `13_36_44` (simulated D455 stereo, aviary SfM scene)

`bimono`, `dataset_id: airsim`, bag played at 0.5×, `dense_keep_all_keyframes: true`,
`dense_gp_save_every: 1`. Evaluation is outside this repository (DeTest-EXECO
`execosim_data/scripts/eval_sadvio_mesh.py`): keyframe poses are rigidly aligned to ground truth
(Umeyama on positions; the orientation-based alignment agrees to 0.6°) and the mesh is compared
with the aviary SfM mesh the simulator renders.

| | A (fusion) | A + B (fusion + registration) |
|---|---|---|
| Keyframes integrated / dropped | 246 / 0 | 247 / 0 |
| Registration | – | 243 ok, 3 rejected (`correction_too_large`), 1 `empty_map`; always 5 iterations |
| Dense time per keyframe | ≈ 45 ms | ≈ 200 ms |
| Map | 5202 cells, 319 k vertices, 270 k faces | 5265 cells, 325 k vertices, 270 k faces |
| SaDVIO keyframe ATE (VO poses) | 5.75 cm | 9.52 cm |
| ATE of the poses the map used | 5.75 cm | **94.9 cm** |
| Correction relative to VO (median / max) | – | 1.46 m / 1.91 m |
| Mesh vertices 2–4 m from a keyframe: median distance to SfM | 24.8 cm | 35.8 cm |
| Same, GP variance < 0.1 | 14.7 cm | 25.3 cm |

For reference, raw SGBM (SaDVIO's settings) back-projected with ground-truth poses lies a median
3.5 cm (2–3 m depth), 4.7 cm (3–4 m) and 8.8 cm (4–6 m) from the SfM surface.

**Findings.**
1. **Registration makes the sim map worse.** Each keyframe's own correction passes the 0.5 m /
   10° gate, but with `carry_correction` the corrections accumulate into a 1.5 m offset from the
   VO trajectory. The model the keyframes register against is itself 15–25 cm off near the
   cameras, so registration pulls poses toward the model's errors and the carry lets that
   compound. SLAMesh does not show this with LiDAR because its per-scan geometry is centimetre
   accurate.
2. **The GP mesh is much less accurate than the SGBM points it is built from** (≈ 25 cm against
   ≈ 4 cm at 2–4 m). Suspects, not yet separated: lattice extrapolation and clamping in 1 m
   cells that the frustum covers only partly; fusion accepting vertices up to variance 0.5;
   points out to `stereo_depth_max_depth: 20` m, where disparity is ≈ 2 px. 57 % of vertices lie
   more than 10 m from every keyframe because each sparse far cell still gets a full lattice.
3. The scene seen by the cameras extends past the SfM mesh into the Blocks level (ground plane,
   cubes), so far geometry is not expected to match the SfM surface; only near-range figures are
   meaningful.

### Bag run: real RealSense `13_36_44` with the Kalibr calibration

A + B with `dataset_id: realsense_optitrack_kalibr0716` (§3a), bag
`RealSense_Captures_060526/13_36_44/rosbag2_2026_05_06-13_36_44` (infra1/infra2 only) at 0.5×.
Ground truth: the OptiTrack CSV, converted as in `capture_airsim_images.interp_ned_pose` and moved
95 mm to infra1; the pivot-to-camera lever arm is approximate, which contributes to ATE.

| | Kalibr `08_07_16`, A + B |
|---|---|
| Keyframes integrated / dropped | 525 / 0 |
| Registration | 508 ok, 16 `correction_too_large`, 1 `empty_map` |
| Dense time per keyframe (end of run) | ≈ 460 ms (23 357-cell map) |
| SaDVIO keyframe ATE (VO poses, rigid) | 19.5 cm |
| **Fitted Sim(3) scale GT / SaDVIO (VO poses)** | **0.9973** |
| ATE / scale of the poses the map used | 152 cm / 0.925 |
| Correction relative to VO (median / max) | 2.83 m / 3.71 m |
| Mesh vertices 2–4 m from a keyframe: median distance to SfM | 52.5 cm |

**Findings.**
1. **The Kalibr calibration removes the real-arm scale error in SaDVIO's VO.** The study's fitted
   scale for this trajectory was 0.9292 (`KALIBR_HANDOFF.md`); with the per-camera principal
   points it is 0.9973. This is one run of one trajectory without a same-code SDK-config
   baseline, so it supports the disparity-offset hypothesis rather than settling it.
2. Registration drifts on real data as on sim data, by more (2.8 m median carried correction).
3. The real scene extends far beyond the aviary mesh (84 % of vertices lie more than 10 m from
   every keyframe), so the SfM comparison is only indicative near the cameras.

## 7. Known limitations and open items

- **Last keyframes.** Keyframes still in the sliding window when the input ends are never
  marginalised, so never meshed.
- **Shutdown.** `ros/src/main.cpp` returns from `rclcpp::spin` with detached threads and a joinable
  `sync_thread`, which aborts without running destructors. The mesh is therefore saved every
  `dense_gp_save_every` keyframes (atomically); the save in the destructor is best effort.
- **Real-time drops.** With `dense_keep_all_keyframes: false` the single-slot queue drops keyframes
  when the worker is busy (the `dropped=` count in the `[DenseMesh]` log line). Offline runs should
  set it to `true` and/or play bags slower.
- **Map growth.** The whole mesh is rebuilt and published after every keyframe; fine for rooms,
  may need throttling for long runs.
- **No loop closure**, as in SLAMesh. Drift not removed by registration stays in the map.
- **Fused-variance bias** (open): see §6; test `variance_map_update` 0.1 vs 0.5 on real data.
- **Axes are never added to an existing cell** (as in SLAMesh), so a cell first seen from one side
  keeps only the axes chosen then.
- **Registration drift** (§6): with an inaccurate stereo model, carried corrections accumulate.
  Candidates: bound the *total* correction relative to VO rather than each increment, or turn
  `dense_gp_register_carry_correction` off; first improve the model (finding 2).
- **GP mesh accuracy** (§6, finding 2): try `stereo_depth_max_depth` ≈ 6 m,
  `dense_gp_variance_map_update: 0.1`, and dropping extrapolated lattice vertices before fusion.
- **Running headless:** `ros2 run` wraps `vio_ros` in a Python process; stopping the wrapper can
  leave `vio_ros` running and still subscribed. Start the binary directly
  (`ros/install/isae_slam_ros/lib/isae_slam_ros/vio_ros <config_dir>`) so its PID is the one to stop.

## 8. Workspace and build

Development happens on the host checkout `~/EXECO/SaDVIO-Dense/SaDVIO-EXECO`, branch
`dense_devel`. The `sad_vio_dense` container gets a separate copy at
`/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel`; the container's original
`/root/SaDVIO-Dense/SaDVIO-EXECO` (branch `simval`) is left untouched.

```bash
# host → container (changed and new files only)
cd ~/EXECO/SaDVIO-Dense/SaDVIO-EXECO
git ls-files -mo --exclude-standard | tar -cf - -T - | \
  docker exec -i sad_vio_dense tar -C /root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel --no-same-owner -xf -

# standalone build (container)
cd /root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/cpp/build && cmake .. && make -j20

# ROS 2 build (container)
cd /root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/ros
source /opt/ros/humble/setup.bash && colcon build --symlink-install
source install/setup.bash    # use this workspace, not the simval one
```

## 9. Changelog

- **2026-09-23** — Global GP map (A) and optional registration (B) implemented; `variance_register`
  default set to 0.1 after the synthetic test; lossless keyframe queue option; registration CSV and
  periodic atomic PLY save.
- **2026-09-23** — First bag runs (ExECoSim `13_36_44`, A and A + B). Fixed double undistortion in
  the dense path; added the Kalibr `08_07_16` RealSense dataset config; real RealSense
  `13_36_44` run (A + B) with it: VO scale 0.9973.
