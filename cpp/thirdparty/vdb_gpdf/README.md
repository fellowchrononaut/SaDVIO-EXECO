# Vendored VDB-GPDF

Core library of **VDB-GPDF: Online Gaussian Process Distance Field with VDB Structure**
(Lan Wu, Cédric Le Gentil, Teresa Vidal-Calleja, IEEE RA-L vol. 10, 2025, arXiv:2407.09649),
copied from <https://github.com/UTS-RI/VDB_GPDF>, branch `ros2`, commit `b2faf51`.
Licensed under GPL-3.0 (`LICENSE`, unchanged), the same licence as SaDVIO. Copyright notices in the
files are kept.

Used by `isae::VDBGPDFMap` (`cpp/include/isaeslam/stereo/VDBGPDFMap.h`) for
`dense_mesh_method: "vdbgpdf"`; see `doc/dense_vdb_gpdf.md`. Built only with
`-DISAESLAM_WITH_VDBGPDF=ON` (`cpp/cmake/ISAESLAM_VDBGPDF.cmake`).

## Files

| File | Upstream path | Changed |
|---|---|---|
| `include/VDBVolume.h` | `vdb_gpdf/include/VDBVolume.h` | yes, see below |
| `src/VDBVolume.cpp` | `vdb_gpdf/src/VDBVolume.cpp` | yes, see below |
| `include/MarchingCubesConst.h` | `vdb_gpdf/include/MarchingCubesConst.h` | no |
| `include/utils.h` | `vdb_gpdf/include/utils.h` | no |
| `include/timer.h` | `vdb_gpdf/include/timer.h` | no |
| `src/MarchingCubes.cpp` | `vdb_gpdf/src/MarchingCubes.cpp` | no |

The ROS 2 node (`vdb_gpdf_mapping/`), its messages and the `3dparty/` submodules are not vendored.

## Modifications (all marked `SaDVIO:` in the code)

1. `#include <rclcpp/rclcpp.hpp>` removed from `VDBVolume.h`.
2. New `struct VDBVolumeParams` holding the parameters the upstream constructor declared on its ROS node,
   with the same defaults; `debug_print` and `use_color`, which the upstream mapper node declared,
   default to `false`.
3. `VDBVolume(std::shared_ptr<rclcpp::Node>)` replaced by `explicit VDBVolume(const VDBVolumeParams&)`;
   the `node_` member and the three `RCLCPP_INFO` log lines are removed.
4. `GenerateVoxelsToUpdate`: a NaN PCL normal (degenerate neighbourhood) is replaced by the viewing
   direction, the upstream `surface_normal_method: 1` normal. Without this, NaN query points reach the
   PCL KD-tree and abort the process (seen with SaDVIO stereo points and the RGB-D preset).

Otherwise the algorithm is unchanged. Known upstream behaviour kept as is: `CreateLocalDisantceField` reads one
colour per input point even when colour is disabled, so callers pass a colour vector of the same size
(`VDBGPDFMap` passes zeros); `WeightFunction::no_constant_weight` (unused) has no return on one path.

## Dependencies

OpenVDB ≥ 9 (the Ubuntu 22.04 `libopenvdb-dev` 8.1 does not compile with oneTBB 2021); upstream uses
<https://github.com/nachovizzo/openvdb> branch `nacho/vdbfusion` (9.1.1), built with
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_ZLIB=OFF`. Also PCL (common, search, kdtree, features),
glog, OpenMP and Eigen.
