# Dense maps that follow the loop closure

Plan agreed on 2026-10-10. The loop closure corrects keyframe poses after the dense map has already used them. This
document is about getting every dense mesh method to follow those corrections. It also covers what that does to mesh
quality and to the MeshCSLAM interface.

Status:
- Done (2026-10-10): steps 1 and 2, for every method. Options `dense_anchored`, `dense_submap_kfs`,
  `dense_anchored_mesh_path`, `dense_anchored_save_every`, `dense_fused_view`, `dense_fused_mesh_path`. Results:
  `doc/loop_closure/LOOP_CLOSURE_LEDGER.md`.
- Later: step 3.
- The first version (vdbgpdf only, `dense_submap_kfs > 0`, `log_slam/dense_submaps/`) is replaced: `dense_anchored`
  switches the anchored maps on, and `dense_submap_kfs` (default 10) only sets the submap size.

## Why the dense map does not follow corrections today

`MarginalDepthInjector` queues each keyframe when it leaves the window. Its depth is turned into geometry in world
coordinates, at the odometry pose of that moment. Loops are found later: `LoopClosure` only sees keyframes that have
left the window. Each method keeps the geometry in one of two ways:

| method | what it builds | what a correction can do to it |
|---|---|---|
| `zncc`, `pd`, `gp` per keyframe (`dense_gp_global_map: false`), `none` (point cloud) | one mesh or cloud per keyframe, in world coordinates; only the latest is kept | nothing: there is no accumulated map |
| `gp` global map (SLAMesh cells, `dense_gp_global_map: true`) | one fused set of GP cells | nothing: cells keep no per-keyframe record |
| `vdbgpdf`, `fusion: gpdf` or `fusion: tsdf` (`stereo_tsdf` preset) | one fused VDB volume | nothing, unless `dense_submap_kfs > 0` |

Fusion is one-way. A voxel or GP cell holds a weighted average of everything that reached it, and it cannot be split
back into its keyframes.

## Why not deform the fused mesh

The obvious idea is to move parts of one fused mesh by the strength of the loop (ElasticFusion's deformation graph).
The pose graph already provides that strength: each anchor's correction is its compromise between odometry and
loop uncertainty. The problem is timing. By the time a loop is found, the revisit has already been fused at its
drifted pose:
- **Drift below the TSDF truncation (~10 cm):** the two passes were averaged into one thick, blurred surface.
  Deformation moves the blur but cannot sharpen it.
- **Larger drift:** the two passes are two surfaces in the same volume, and a smooth deformation cannot send them to
  different places.

ElasticFusion avoids this by closing loops at every frame, before it fuses the revisit. Here the loop closure runs
one window late. Deformation alone is therefore good for display at most, not for the map.

## The plan

### 1. Anchored meshes for every method

A single store in the injector holds anchored meshes. Each entry has:
- an anchor keyframe timestamp and segment;
- the odometry pose it was built at;
- its geometry in the anchor's frame (a rectified left camera).

The global mesh, the PLYs, `anchors.csv` and the `dense_mesh` topic are assembled from it. Each entry is placed at
`keyframeCorrection(anchor) * T_w_anchor`.

- **Per-keyframe methods** (`zncc`, `pd`, per-keyframe `gp`, `none`): each keyframe is its own anchor, so every
  keyframe gets its own correction. The estimators build in the keyframe's camera frame instead of in world
  coordinates. Their meshes are kept instead of replaced by the next one.
- **Fused methods** (`gp` global map, `vdbgpdf` gpdf / tsdf): submaps of `dense_submap_kfs` keyframes. Each is fused
  in its first keyframe's frame and placed by it.
  - A new submap starts at every re-initialization. At a window correction (the world moved by C) the open submap
    continues with its anchor re-expressed as C·T_anchor. Until 2026-10-13 it was closed instead: with the dense
    prior, room1 then broke into submaps of a few keyframes (450 window moves).
  - `dense_gp_register` registers each keyframe against the current submap only.
- Files: `log_slam/dense_anchors/anchor_NNNN.ply` (geometry in the anchor frame) and `anchors.csv` (anchor timestamp,
  segment, keyframes, closed, odometry pose, corrected pose). These are what the MeshCSLAM map server takes: submap
  meshes and their poses.
- Overlapping entries stay separate layers. Placing them is exact and can be redone after any later loop.
- Within a submap the geometry moves rigidly. Blending neighbouring anchors' corrections per vertex would remove the
  seams; that can be added later, for display.

Built as `AnchoredMaps` (`cpp/include/isaeslam/stereo/AnchoredMaps.h`). The injector feeds it:
- Per-keyframe methods: the estimators still build at the keyframe's pose, and their result is moved into the camera
  frame (exact). For `none`, the point cloud is kept in 5 cm voxels.
- Fused methods: each keyframe is integrated into the open submap's volume (`VDBGPDFMap` or `GPGlobalMap`).

The global mesh and `anchors.csv` are rewritten every `dense_anchored_save_every` keyframes.

### 2. Fused global view (option, on top of 1)

`dense_fused_view` merges the placed submap volumes into one global volume by weighted voxel merging (c-blox /
Voxgraph style). Revisits are averaged, and the layers and ghosts disappear. The view can always be rebuilt from the
submaps, so later loops never leave it inconsistent.
- `0`: off (anchored meshes only).
- `1` (default when submaps are on): at every map save (`dense_*_mesh_every` keyframes), plus a full rebuild at the
  end.
  - A save re-projects only the submaps whose correction changed by more than a threshold (about 2 cm / 0.5°) since
    their last projection.
  - Each such submap's old contribution is removed and it is added again at its new pose. For a weighted-average
    TSDF the removal is exact.
  - This runs in the dense worker thread, which is low priority.
- `2`: at the end only (a full rebuild when the map is written), for weak CPUs.
- Not on every loop: during a revisit, loops close at nearly every keyframe (room2: 429 loops) with small changes.
- Needs the closed submaps' grids kept. Today a closed submap keeps only its mesh. A sparse VDB TSDF costs a few MB
  per submap.
- Per method (as built): the fused view is a VDB volume for every method.
  - `vdbgpdf` (gpdf or tsdf fusion) merges its own submap volumes, which closed submaps keep for this.
  - The other methods (zncc, pd, per-keyframe gp, gp global map, none) first integrate their geometry into TSDF
    chunks of `dense_submap_kfs` keyframes (5 cm voxels, 10 cm truncation, no space carving).
    - Per-keyframe methods integrate each keyframe's vertices from its own camera.
    - The GP map integrates each submap's mesh from the submap's first camera.
  - The chunks are merged the same way. This replaces "the GP map refits": refitting the GP cells cannot be undone
    for one submap, while a TSDF chunk can be.
  - So the fused view needs the VDB-GPDF build (`-DISAESLAM_WITH_VDBGPDF=ON`) for every method.
- Merge (`VDBGPDFMap::accumulateInto`): each voxel of a placed volume is splatted trilinearly onto the fused lattice
  as a weighted running average, so a rotation leaves no holes. The same splat with a negative weight removes it.
  A voxel whose weight drops under 0.01 (of one observation) is cleared, which absorbs float32 cancellation (weights
  reach hundreds).
- The open submap or chunk is not merged yet. It is added to the fused mesh as its own layer until it closes.
- Files: `dense_fused_mesh_path`. With `dense_fused_view: 1` it is also what the `dense_mesh` topic shows.
- End of the run: the SLAM threads never return. `main` (SIGTERM / SIGINT, as batch runs end with `pkill`) and the ROS
  node (after `rclcpp::spin`) call `MarginalDepthInjector::finish()`, which processes the queued keyframes, closes the
  last submap, writes the anchors and rebuilds the fused view, then the process exits. Before this, no final dense
  file was ever written in a batch run.

### 3. Alternative, later: de-integration and re-integration (BundleFusion style)

- Store each keyframe's depth (subsampled points, roughly 0.5–1 MB per keyframe).
- After a correction, subtract a keyframe's contribution at its old pose and add it at its new one.
- This gives one truly fused volume, consistent with the corrected trajectory: the best mesh.
- Cost: memory, and re-integrating the keyframes a loop moved. At 284 ms per keyframe for `stereo_tsdf`, a large loop
  can only be absorbed over time, within a budget per frame.
- Exact for TSDF only. VDB-GPDF's GP fusion and SLAMesh's GP cells cannot be de-integrated exactly.
- Worth building if the fused view of step 2 is not good enough.

## Measured so far (`vdbgpdf` submaps)

From `doc/loop_closure/LOOP_CLOSURE_LEDGER.md`, 2026-10-10:
- The anchors move toward the ground truth in all 12 runs.
- On the EuRoC Vicon rooms with decimetre drift (no prior, V1_03 and V2_02), the mesh moves 3–4 cm closer to the
  Leica scan in median, and 4–6 points more of it lies within 10 cm.
- On easy sequences the SGBM depth (6–17 cm median) hides the correction.

## Measured with FFS depth (2026-10-10, all methods)

From `doc/loop_closure/LOOP_CLOSURE_LEDGER.md`: EuRoC V1_03 and V2_02, VIO no prior with loop closure, Leica scan.
- Anchors at corrected poses beat anchors at odometry poses for every method on both sequences: 2–15 points more of
  the mesh within 10 cm of the scan.
- The fused view beats the global map it replaces (or ties: vdbgpdf gpdf on V1_03). It loses to the anchored layers
  at this pose accuracy (~5–9 cm ATE after loops, near the 10 cm TSDF truncation).
- So MeshCSLAM should take the anchored meshes. The fused view is for viewing and local use.
- Checked with better poses (VIO dense / sparse prior): the layers improve, the fused view does not. Every
  long-horizon volume (old global map, one whole-run submap, fused view) stays at 14–20 cm median against 4.5–8 cm
  for 10-keyframe submaps, so the limit is long-horizon TSDF fusion of FFS depth on this data, not the merge (3 cm
  resampling blur). Step 3 would inherit it and stays unbuilt until that is understood (ledger, "Fused view with
  better poses").
