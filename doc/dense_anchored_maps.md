# Dense maps that follow the loop closure

Plan agreed on 2026-10-10. The loop closure corrects keyframe poses after the dense map has already used them. This
document is about getting every dense mesh method to follow those corrections. It also covers what that does to mesh
quality and to the MeshCSLAM interface.

Status:
- Done: submaps for `vdbgpdf` (`dense_submap_kfs`, `doc/loop_closure/LOOP_CLOSURE_LEDGER.md`, 2026-10-10).
- Next: steps 1 and 2.
- Later: step 3.

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
  - A new submap starts at every re-initialization and at every window correction, as `vdbgpdf` does now.
  - `dense_gp_register` registers each keyframe against the current submap only.
- Files: `log_slam/dense_anchors/anchor_NNNN.ply` (geometry in the anchor frame) and `anchors.csv` (anchor timestamp,
  segment, keyframes, closed, odometry pose, corrected pose). These are what the MeshCSLAM map server takes: submap
  meshes and their poses.
- Overlapping entries stay separate layers. Placing them is exact and can be redone after any later loop.
- Within a submap the geometry moves rigidly. Blending neighbouring anchors' corrections per vertex would remove the
  seams; that can be added later, for display.

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
- Per method:
  - TSDF and VDB-GPDF merge their VDB grids (OpenVDB resamples a transformed grid).
  - The GP global map refits from the placed submaps' points.
  - Per-keyframe methods have no volume to merge. Their fused view would integrate their placed per-keyframe geometry
    into a TSDF, which is to be decided when step 2 is built.

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
