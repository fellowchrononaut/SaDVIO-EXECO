# VDB-GPDF as a dense mapping backend

Development document for `dense_mesh_method: "vdbgpdf"`, an alternative to the SLAMesh-style GP
meshing (`"gp"`, `doc/dense_gp_global_map.md`) and the primal-dual mesh (`"pd"`). It fuses the dense
stereo points of every keyframe into **VDB-GPDF** (Wu, Le Gentil, Vidal-Calleja, *VDB-GPDF: Online
Gaussian Process Distance Field with VDB Structure*, IEEE RA-L 10, 2025, arXiv:2407.09649) and
extracts the mesh with marching cubes.

## 1. Why

The GP-mesh diagnosis on ExECoSim 13_36_44 (DeTest-EXECO `execosim_data/scripts/gp_harness/`,
ground-truth-posed frames) found that the SLAMesh representation, not the port or its parameters,
causes most of the visible artefacts: every 1-1.2 m cell stores up to three axis-aligned height-field
layers, a surface tilted more than ~18° from an axis gets two overlapping layers, and 25-29 % of the
mesh area is more than 45° off the true surface even on LiDAR (73 % of it from layers predicted along
the wrong axis). Stereo noise (FFS 3.9 cm, SGBM 6.9 cm median, against 1.3 cm for LiDAR) is not
averaged within a frame. VDB-GPDF keeps a GP model with uncertainty but represents the surface as the
zero level set of a fused signed distance field, so orientation-dependent layers and cell seams do not
exist, and every voxel averages many observations.

## 2. What VDB-GPDF does

Per integrated scan (`VDBVolume::Integrate`, upstream):

1. The points go into a local VDB grid (`voxel_size_local`); every occupied leaf (8³ voxels) becomes a
   small local GP ("L-GPDF", reverting-function GPDF of Le Gentil et al. by default,
   `distance_method: 0`, or Log-GPIS, `1`) trained on its voxel centres, down-sampled by
   `voxel_downsample`.
2. Query points are generated along each point's normal (PCL normals, `surface_normal_num`
   neighbours), `query_trunc_in`/`query_trunc_out` steps of `query_iterval` in front of and behind the
   surface, plus free-space points along the ray to the sensor (`freespace_iterval`).
3. Each query point takes the distance (and variance) inferred by its nearest local GPs.
4. The values are fused into the global VDB grid (`voxel_size_global`) as a weighted running average
   (`variance_method` 0: constant weight; 1-3: variance-based weights).
5. A global GPDF (G-GPDF) is prepared from the fused grid for distance/gradient queries.

`ExtractTriangleMesh(fill_holes, recon_min_weight)` runs marching cubes on the global grid.
The upstream code also keeps a projective TSDF path (`Integrate` without GP outputs) that its mapper
uses "for vdbfusion as comparison"; SaDVIO exposes it as `fusion: tsdf`.

## 3. What was built

| Piece | File |
|---|---|
| Vendored core (GPL-3.0, like SaDVIO; origin, changes) | `cpp/thirdparty/vdb_gpdf/` (+ `README.md`) |
| Wrapper, preset loader | `cpp/include/isaeslam/stereo/VDBGPDFMap.h`, `cpp/src/stereo/VDBGPDFMap.cpp` |
| Build option | `cpp/cmake/ISAESLAM_VDBGPDF.cmake`, included by `cpp/` and `ros/` CMakeLists (`-DISAESLAM_WITH_VDBGPDF=ON`, default OFF) |
| Injector wiring | `MarginalDepthInjector::integrateVDBGPDF` (`mesh_method == "vdbgpdf"`) |
| Parameters | `slamParameters.{h,cpp}`, `slamBiMono.cpp`, `slamBiMonoVIO.cpp`, `ros/config/config.yaml` |
| Presets | `ros/config/vdbgpdf/{rgbd,lidar,stereo}.yaml` |
| Shared PLY writer | `writeDenseMeshPly` in `DenseMesh.{h,cpp}` (`GPGlobalMap::savePly` now uses it) |

Vendored rather than linked because the core is small (~1.9 k lines), not packaged, and tied to ROS
only through its constructor; the changes are listed in `cpp/thirdparty/vdb_gpdf/README.md`.

### Per keyframe

`integrateVDBGPDF` back-projects the keyframe's disparity (every `dense_vdbgpdf_stride`-th pixel, default 2,
depth ≤ `stereo_depth_max_depth`) with SaDVIO's VO pose (no ground truth), integrates the world-frame
points with the camera centre as sensor origin, and every `dense_vdbgpdf_mesh_every` keyframes runs
marching cubes, writes `dense_vdbgpdf_mesh_path` and republishes the mesh (the last mesh is shown in
between). The final mesh is written when the injector is destroyed. As with the GP map, use
`dense_keep_all_keyframes: true` so no keyframe is dropped.

## 4. Configuration

```yaml
dense_mesh_method:        "vdbgpdf"
dense_vdbgpdf_preset:     "stereo"   # config/vdbgpdf/<preset>.yaml
dense_vdbgpdf_stride:     1
dense_vdbgpdf_mesh_every: 5
dense_vdbgpdf_mesh_path:  "log_slam/dense_vdbgpdf_mesh.ply"
```

Presets use the upstream parameter names (flat, or the upstream ROS 2
`vdb_gpdf_mapping_node: ros__parameters:` layout) plus `fusion: gpdf | tsdf`:

* `rgbd.yaml` — upstream `vdb_gpdf_mapping_realsense_d455.yaml` (main branch): 2 cm local / 5 cm global
  voxels, λ = 900 (length scale 3.3 cm), range 0.1-4 m.
* `lidar.yaml` — upstream `vdb_gpdf_mapping_newer_college.yaml`: 10 cm voxels, λ = 100, range 1-50 m.
* `stereo.yaml` — the D455 preset with `max_scan_range: 8.0` (4 m in upstream; 8 m raises F@10 from
  38 to 57 on FFS, 20 m adds nothing), `fusion: gpdf`.
* `stereo_tsdf.yaml` — `stereo.yaml` with `fusion: tsdf` (5 cm voxels, `sdf_trunc` 0.1 m, space
  carving): the best stereo mesh in section 5.

Each keyframe's fused pose is logged to `dense_vdbgpdf_keyframes.csv` next to the mesh
(timestamp, points, integration time, position, quaternion).

A build without `ISAESLAM_WITH_VDBGPDF` throws at start-up for `dense_mesh_method: "vdbgpdf"`.

## 5. Verification

### Ground-truth-posed harness (evaluation only)

DeTest-EXECO `execosim_data/scripts/gp_harness/`: `make_frames.py` writes world-frame frames
(LiDAR scans; FFS / SGBM stereo, stride 2, ≤ 20 m) with ground-truth poses, `vdbgpdf_harness.cpp` feeds
them to `VDBGPDFMap`, `eval_harness.py` scores the mesh against the VID012 SfM mesh in the aviary crop
(accuracy = mesh → GT median, F@10 cm, wrong-orient = area with normal > 45° off GT, open edge =
boundary length per m² after 1 cm welding). One process on an idle 24-core CPU.

| Input | Mesher | acc. | F@10 | compl. < 10 cm | wrong-orient | open edge | ms/frame |
|---|---|---|---|---|---|---|---|
| LiDAR | SLAMesh (real run, own poses) | 3.5 cm | 80.7 | 77 % | 29 % | 73 | — |
| LiDAR | SaDVIO GP map, SLAMesh settings | 2.5 cm | 81.5 | 75 % | 25 % | 42 | ~5 |
| LiDAR | **tsdf, `lidar` preset (10 cm)** | **1.3 cm** | **90.2** | **82 %** | **10 %** | **26** | 82 |
| LiDAR | gpdf, `lidar` preset (10 cm) | 6.4 cm | 64.3 | 56 % | 23 % | 24 | 51 |
| LiDAR | gpdf, `stereo` (D455) preset | 3.3 cm | 79.8 | 67 % | 16 % | 361 | 65 |
| FFS | SaDVIO GP map, SLAMesh settings | 6.5 cm | 66.1 | 64 % | 32 % | 39 | ~6 |
| FFS | **tsdf, `stereo_tsdf` (5 cm, ≤ 8 m)** | **4.6 cm** | **75.3** | **69 %** | **18 %** | 94 | 420-480 |
| FFS | gpdf, D455 preset as upstream (≤ 4 m) | 5.3 cm | 37.7 | 26 % | 20 % | 174 | 31 |
| FFS | gpdf, `stereo` (D455, ≤ 8 m) | 6.7 cm | 56.9 | 51 % | 24 % | 163 | 174 |
| FFS | gpdf, D455, ≤ 20 m | 6.5 cm | 56.9 | 51 % | 22 % | 134 | 203 |
| FFS | gpdf, `lidar` preset, ≤ 20 m | 8.6 cm | 47.3 | 41 % | 24 % | 14 | 48 |
| SGBM | gpdf, `stereo` | 9.3 cm | 57.5 | 63 % | 52 % | 289 | 216 |

Controls: the same FFS frames rigidly moved into the first camera frame (SaDVIO's world convention)
give the same results (gpdf 57.6, tsdf 75.5), so neither backend depends on the world axes; frames
taken at SaDVIO's keyframe timestamps instead of every 7th image give gpdf 57.7, tsdf 75.0.

### Pipeline runs (FFS depth, SaDVIO VO poses, keyframe ATE 8.2-8.7 cm)

`run_dense_visu.sh ... ffs "" "" vdbgpdf <preset>` on the 13_36_44 sim bag, scored the same way after a
rigid alignment of the VO keyframes to ground truth:

| Backend | acc. | F@10 | compl. | wrong-orient | open edge | ms/keyframe |
|---|---|---|---|---|---|---|
| GP map A (`gp`) | 8.9 cm | 57.0 | 60 % | 39 % | 75 | 16-24 |
| `vdbgpdf`, `stereo` (gpdf) | 4.9 cm | 72.6 | 66 % | 32 % | 604 | 145-190 |
| `vdbgpdf`, `stereo_tsdf` | 5.9 cm | 70.8 | 64 % | 23 % | 339 | 440 |

Both backends improve clearly on the GP map. The gpdf pipeline score is higher than its
ground-truth-pose score (57.7), which does not mean it prefers VO poses: a horizontal slice shows
doubled, thickened walls from VO drift that raise completeness, and adding 3 cm / 0.5° per-frame noise
to the ground-truth poses already raises gpdf to 62.0 (tsdf stays at 74.8). With consistent poses
gpdf drops surface patches (fewer covered wall sections, more open edges), and in RViz the tsdf mesh
is visibly more complete with fewer gaps. **For meshing, use `stereo_tsdf`**; `stereo` (gpdf) is the
option when the distance field and its variance are wanted.

### Runtime

Upstream reports 100-650 ms per frame for VDB-GPDF (Fig. 4, i5-1245U laptop, including local GPs,
fusion and global GPDF). With FFS input (~100 k points per keyframe) SaDVIO's integration takes
31-203 ms (gpdf) and 420-480 ms (tsdf) on a 24-core desktop CPU, constant over the run. Upstream
hard-codes 12 OpenMP threads (`omp_set_num_threads(12)`, 12-thread PCL normals, nested parallel loops),
so several concurrent processes oversubscribe the CPU.

### Bugs found and fixed while integrating

* `CreateLocalDisantceField` reads `colors[i]` for every point: an empty colour vector crashed it
  (wrapper passes zeros).
* NaN PCL normals reached the KD-tree and aborted the process (vendored fix, see thirdparty README).
* `Integrate` appends to its output vectors; the first wrapper reused them across scans, so every
  scan re-queried and re-fused all previous scans (per-frame time grew to 5-23 s and fusion was
  wrong). The upstream mapper declares them per scan; the wrapper now does the same. Results above are
  from the fixed wrapper.

## 6. Dependencies (sad_vio_dense)

OpenVDB 9.1.1 built from <https://github.com/nachovizzo/openvdb> `nacho/vdbfusion` (commit c110123)
into `/usr/local` (`-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_ZLIB=OFF -DOPENVDB_BUILD_BINARIES=OFF`);
Ubuntu's `libopenvdb-dev` 8.1 fails to compile against the installed oneTBB 2021 and was removed. PCL
1.12, glog, gflags and OpenMP come from the ROS Humble image.

## 7. Changelog

* 2026-09-24 — vendored VDB-GPDF (ros2 @ b2faf51), wrapper, build option, injector backend, presets,
  shared PLY writer, harness evaluation.
