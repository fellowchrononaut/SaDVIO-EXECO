# Lite Any Stereo V2 as a CPU dense stereo backend

Development document for the `stereo_depth_matcher: "las2"` option of the dense stereo pipeline
(`MarginalDepthInjector`). It runs Lite Any Stereo V2 (LAS2; Jing, Luo, Mao, Mikolajczyk, *Lite Any
Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching*, arXiv:2606.24457) on the CPU
through OpenVINO, as a GPU-free alternative to Fast-FoundationStereo (`doc/dense_ffs_stereo.md`).
Everything downstream of the disparity map is unchanged, as for FFS.

## 1. Why LAS2 and not FFS on the CPU

Measured on the host (Core Ultra 9 285K, 24 cores, AVX2/VNNI, no AVX-512/AMX), under heavy load
(load average 29-37), one 480×864 pair unless stated:

| Model | ONNX Runtime | OpenVINO f32 | OpenVINO f16 |
|---|---|---|---|
| FFS 23-36-37 it8 (the TensorRT model) | ~11 s | 8.5 s | 6.3-7.6 s |
| FFS 20-30-48 it4, 256×448 | 2.0 s | 1.06 s | 0.86 s |
| LAS2-S | 0.76 s | 0.56 s | 0.40 s |
| LAS2-M | 1.18 s | 0.72 s | 0.58 s |
| LAS2-H | 2.7 s | 2.5 s | 1.6 s |
| LAS2-S, 256×448 | | 0.17 s | 0.11 s |
| LAS2-M, 256×448 | | 0.23 s | 0.15 s |

PyTorch CPU runs FFS in ~20 s. The ONNX Runtime profile of FFS is dominated by its 3D transposed
convolutions (24 %, one layer 2.1 s) and the `GridSample` lookups of the GRU refinement (16 %); no
runtime closes that gap. LAS2 aggregates the cost volume with 2D convolutions only (FasterNet
backbone, correlation volume at 1/4 resolution, convex upsampling) and has no refinement loop
except in the H model (4 ConvGRU iterations). The paper reports zero-shot KITTI 2015 D1 of 3.31 %
for LAS2-H against 3.66 % for FFS.

## 2. What was built

| Piece | File | Notes |
|---|---|---|
| LAS2 backend | `cpp/include/isaeslam/stereo/LAS2StereoMatcher.h`, `cpp/src/stereo/LAS2StereoMatcher.cpp` | OpenVINO runtime only; compiled to nothing without `ISAESLAM_WITH_LAS2` |
| Build option | `cpp/cmake/ISAESLAM_LAS2.cmake`, included by `cpp/CMakeLists.txt` and `ros/CMakeLists.txt` | `-DISAESLAM_WITH_LAS2=ON -DISAESLAM_OPENVINO_DIR=<dir with OpenVINOConfig.cmake>` (default OFF) |
| Factory / config | `StereoMatcher.{h,cpp}`, `MarginalDepthInjector.{h,cpp}` | `"las2"` next to `"sgbm"` and `"ffs"`; timing line prints `LAS2=` |
| Parameters | `slamParameters.{h,cpp}`, `slamBiMono.cpp`, `slamBiMonoVIO.cpp`, `ros/config/config.yaml` | eight keys, defaults keep SGBM |

No LAS source is copied into SaDVIO; it loads an ONNX file exported by the upstream repository
(MIT code, CC-BY 4.0 release). OpenVINO is linked from the `openvino` pip wheel, which ships the
C++ SDK (`openvino/{cmake,include,libs}`, including its own oneTBB); the build adds the wheel's
`libs/` to the library's build and install RPATH.

### Pre/post-processing

Same as the FFS backend, following upstream `demo.py`/`run_onnx.py`: grey replicated to 3 channels
(BGR converted to RGB), values left at 0-255 (the exported network normalises internally), uniform
downscale if the image is larger than the model input (never upscaled), centred replicate padding
(upstream `InputPadder`), disparity cropped back and rescaled (nearest neighbour, × width ratio),
`d <= 0` and `u - d < 0` set to -1, optional mirrored left-right check. The model's `max_disp` is
fixed at 192 by the weights (the disparity axis is baked into the aggregation channels), so a
smaller input resolution is the only speed/accuracy knob besides the model size.

The model I/O is checked at load time (`left`, `right`: static 1×3×H×W float32; `disparity`: H×W).
One warm-up inference runs in the constructor. Compiling a model takes under 1 s, so there is no
model cache.

## 3. Configuration

```yaml
stereo_depth_matcher:         "las2"
stereo_depth_las2_model_dir:  "/root/SaDVIO-Dense/las2/models"  # las2_<size>_<HxW>.onnx
stereo_depth_las2_size:       "m"        # s | m | l | h
stereo_depth_las2_resolution: "480x864"  # 480x864 | 320x576 | 256x448
stereo_depth_las2_onnx:       ""         # explicit path, overrides the three keys above
stereo_depth_las2_device:     "CPU"      # OpenVINO device
stereo_depth_las2_precision:  "f32"      # f32 | f16 | bf16
stereo_depth_las2_threads:    8          # 0 = all cores
stereo_depth_las2_lr_check:   0.0        # px; 0 = off
```

`stereo_depth_matcher: "las2"` in a build without `ISAESLAM_WITH_LAS2` throws at start-up, as does a
missing model file. The startup line `[LAS2] model ... precision ..., N threads` reports what
OpenVINO actually uses. `stereo_depth_las2_threads` defaults to 8 so the dense worker does not take
the cores the sparse front-end needs (the front-end is timing dependent, `doc/dense_ffs_stereo.md`
§4). `run_dense_visu.sh` (DeTest-EXECO) takes the LAS2 keys as arguments 14-18.

## 4. Preparing models and the SDK

Host, in the LAS repository clone (`git clone https://github.com/TomTomTommi/LiteAnyStereo`,
checkpoints from `download_checkpoint.py`, Hugging Face `tomtomtommi/LiteAnyStereoV2`), with the
FFS venv (torch 2.11, timm):

```bash
for s in s m h; do for r in 480x864 320x576 256x448; do
  CUDA_VISIBLE_DEVICES= python export_onnx.py --version las2 --model_size $s \
      --restore_ckpt checkpoints/LAS2_${s^^}.pth --height ${r%x*} --width ${r#*x} --max_disp 192 \
      --output_name las2_models/las2_${s}_${r}.onnx
done; done
```

The export runs on the CPU. LAS2-L was not downloaded; its models follow the same naming.

Container (`sad_vio_dense`):

* `/root/SaDVIO-Dense/las2/models/` — the ONNX files above.
* `/root/SaDVIO-Dense/las2/openvino_sdk/openvino/{cmake,include,libs}` — copied from
  `pip install openvino` (2026.4.0). The directory layout must be kept: the CMake package resolves
  the libraries relative to itself.
* Build: `colcon build --cmake-args -DISAESLAM_WITH_FFS=ON -DISAESLAM_WITH_VDBGPDF=ON
  -DISAESLAM_WITH_LAS2=ON -DISAESLAM_OPENVINO_DIR=/root/SaDVIO-Dense/las2/openvino_sdk/openvino/cmake`

## 5. Verification

### Matcher unit check (one ExECoSim frame, 13_36_44 bag, image 400 — the FFS check frame)

Scratch tool in the container (`/root/SaDVIO-Dense/las2/check/las2_check.cpp`, links
`libisae_slam_ros.so`), 8 OpenVINO threads, host at load average ~34, time per pair including
pre/post-processing (median of 5):

| Setting | time | valid px |
|---|---|---|
| M 480×864 f32 | 243 ms | 98.3 % |
| M 480×864 f16 | 187 ms | 98.3 % |
| M 480×864 bf16 | 132 ms | 98.3 % |
| M 480×864 f32 + LR check 1 px | 366 ms | 98.0 % |
| S 480×864 f32 / f16 | 136 / 129 ms | 98.2 % |
| M 320×576 f32 | 86 ms | 98.3 % |
| M 256×448 f32 | 49 ms | 98.3 % |
| S 256×448 f16 | 32 ms | 98.3 % |
| H 480×864 f32 | 851 ms (9 s start-up) | 98.3 % |

These are 2-3× below the Python OpenVINO numbers of section 1, which were taken at a higher load
with 12 threads; compare rows within one table only.

Correctness, pixels valid in both (`compare.py` in the same folder):

| | median | p99 | max | > 1 px |
|---|---|---|---|---|
| C++ M f32 vs ONNX Runtime fp32 (Python, same padding) | 0.0000 px | 0.000 | 0.00 | 0 % |
| M f16 vs ONNX Runtime fp32 | 0.004 px | 0.024 | 0.06 | 0 % |
| M bf16 vs ONNX Runtime fp32 | 0.027 px | 0.162 | 0.35 | 0 % |
| LAS2-H 480×864 vs FFS fp32 engine | 0.057 px | 0.52 | 4.4 | 0.24 % |
| LAS2-M 480×864 vs FFS | 0.073 px | 0.65 | 4.1 | 0.39 % |
| LAS2-S 480×864 vs FFS | 0.081 px | 0.72 | 5.2 | 0.48 % |
| LAS2-M 320×576 vs FFS | 0.113 px | 0.91 | 5.8 | 0.87 % |
| LAS2-M 256×448 vs FFS | 0.156 px | 1.09 | 6.1 | 1.21 % |
| LAS2-S 256×448 f16 vs FFS | 0.177 px | 1.24 | 5.3 | 1.79 % |
| SGBM (block 9, 64 disparities) vs FFS | 0.183 px | 1.70 | 5.8 | 2.56 % |

FFS is not ground truth, but on this frame every LAS2 setting sits closer to FFS than SGBM does.
The raw-depth evaluation against the SfM mesh (`eval_raw_disparity.py`) and the mesh comparison of
`doc/dense_ffs_stereo.md` §4 have not been run for LAS2 yet.

### Pipeline smoke run

`/root/SaDVIO-Dense/las2/check/smoke_las2.sh`: `vio_ros` bimono, LAS2-M 480×864 f32 8 threads,
`dense_mesh_method: vdbgpdf` (preset `stereo`), `dense_keep_all_keyframes`, 90 s of the 13_36_44
bag at 0.5×. 85 keyframes through LAS2 (216-228 ms each) and into the VDB-GPDF map, none dropped.
The process ends with `terminate called without an active exception` on SIGINT, as the SGBM
vdbgpdf runs already did (`runs/13_36_44_sgbm_vdbgpdf`), so it is not caused by this backend.

## 6. Limitations and open points

* Only one frame and one short run checked; mesh quality against SGBM/FFS is still to be measured.
* **oneTBB swap.** The wheel's `libtbb.so.12` comes first in the RUNPATH, so in a
  `ISAESLAM_WITH_LAS2` build OpenVDB (built against the system oneTBB 2021.5) also loads the
  wheel's newer oneTBB. oneTBB keeps backward ABI compatibility and the smoke run's VDB-GPDF map
  integrated normally, but a VDB-GPDF-only comparison should keep in mind that the TBB differs.
* As for FFS, no per-pixel confidence; the LR check is the only filter, and the GP sensor variance
  is still the SGBM-tuned constant.
* The dense worker shares the CPU with the sparse front-end; with more threads LAS2 gets faster but
  the front-end timing changes. `stereo_depth_las2_threads` is the knob.
* The matcher is created again after every front-end re-initialisation (model compile < 1 s, H
  model ~9 s).
* The pre/post-processing duplicates `FFSStereoMatcher.cpp`; the two could share a helper.
* Intel iGPU (`stereo_depth_las2_device: "GPU"`) is untested and needs the Intel compute runtime in
  the container.

## 7. Changelog

* 2026-09-29 — LAS2 OpenVINO backend, build option, eight config keys, CPU timing comparison with FFS.
