# Fast-FoundationStereo as a dense stereo backend

Development document for the `stereo_depth_matcher: "ffs"` option of the dense stereo pipeline
(`MarginalDepthInjector`). It replaces OpenCV SGBM with Fast-FoundationStereo (FFS; Wen, Dewan,
Birchfield, *Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching*, CVPR 2026,
arXiv:2512.11130) run through TensorRT. Everything downstream of the disparity map (depth, point
cloud, per-keyframe GP mesh, global GP map A/B of `doc/dense_gp_global_map.md`) is unchanged, so
the matcher is the only variable when meshes are compared.

## 1. What was built

| Piece | File | Notes |
|---|---|---|
| Matcher interface + SGBM backend + factory | `cpp/include/isaeslam/stereo/StereoMatcher.h`, `cpp/src/stereo/StereoMatcher.cpp` | SGBM code moved out of `MarginalDepthInjector` unchanged (same parameters, `MODE_SGBM_3WAY`, /16 fixed point) |
| FFS backend | `cpp/include/isaeslam/stereo/FFSStereoMatcher.h`, `cpp/src/stereo/FFSStereoMatcher.cpp` | TensorRT runtime only; compiled to nothing without `ISAESLAM_WITH_FFS` |
| Build option | `cpp/cmake/ISAESLAM_FFS.cmake`, included by `cpp/CMakeLists.txt` and `ros/CMakeLists.txt` | `-DISAESLAM_WITH_FFS=ON` (default OFF) |
| Injector wiring | `MarginalDepthInjector.{h,cpp}` | `_sgbm` replaced by `std::unique_ptr<StereoMatcher>`; timing line prints `SGBM=` or `FFS=` |
| Parameters | `slamParameters.{h,cpp}`, `slamBiMono.cpp`, `slamBiMonoVIO.cpp`, `ros/config/config.yaml` | three keys, defaults keep SGBM |

No Fast-FoundationStereo source is copied into SaDVIO. The FFS repository is under the NVIDIA
license, whose derivative-work definition excludes works that "remain separable from, or merely
link (or bind by name) to the interfaces of, the Work"; SaDVIO only loads a TensorRT engine file
that the user builds from the upstream export.

### Pre/post-processing (follows upstream `scripts/run_demo.py`)

1. SaDVIO images arrive as `mono8` (cv_bridge); grey is replicated to 3 channels (upstream notes
   that FFS works on monochrome/IR stereo). BGR input is converted to RGB.
2. ImageNet normalisation on 0-255 values, mean (123.675, 116.28, 103.53), std (58.395, 57.12,
   57.375) — the single-ONNX export strips it from the network.
3. The rectified image is placed into the engine input with replicate padding split evenly on both
   sides (upstream `InputPadder`, "sintel" mode). An image larger than the engine is first
   downscaled uniformly (never upscaled) and the disparity is rescaled back (nearest neighbour,
   × width ratio). For the 848×480 RealSense/ExECoSim images and a 864×480 engine this is pure
   padding (8 px left and right).
4. Output disparity is cropped back; pixels with `d <= 0` or whose match falls left of the right
   image (`u - d < 0`, upstream `remove_invisible`) are set to -1 (invalid, like SGBM).
5. Optional left-right check (`stereo_depth_ffs_lr_check > 0`): a second inference on the mirrored,
   swapped pair (flip(R), flip(L)) gives the right-image disparity; pixels with
   `|dL(u) - dR(u - dL(u))|` above the threshold are dropped. Doubles the inference cost.

The engine's I/O is checked at load time (`left_image`, `right_image`: 1×3×H×W float32;
`disparity`: float32); H and W are read from the engine. One warm-up inference runs in the
constructor so the first keyframe does not pay the lazy CUDA module loading.

## 2. Configuration

```yaml
stereo_depth_matcher:    "sgbm"   # "sgbm" or "ffs"
stereo_depth_ffs_engine: ""       # TensorRT engine (see section 3)
stereo_depth_ffs_lr_check: 0.0    # px; 0 = off
```

`stereo_depth_matcher: "ffs"` in a build without `ISAESLAM_WITH_FFS`, or an unknown matcher name,
throws at start-up instead of silently falling back to SGBM. The SGBM keys
(`stereo_depth_num_disp`, `_block_size`, ...) are ignored by FFS; `stereo_depth_scale`,
`stereo_depth_stride` and `stereo_depth_max_depth` still apply. FFS searches up to the export's
`max_disp` (192 px at engine resolution) versus SGBM's `stereo_depth_num_disp` (64), so FFS also
returns depth closer than f·b/64 and in the left image band SGBM cannot match.

## 3. Preparing an engine

Engines are specific to the TensorRT version and GPU. In `sad_vio_dense` (RTX 5090, sm_120):

* CUDA runtime 12.9 (`cuda-cudart-dev-12-9`, `cuda-crt-12-9`) and TensorRT 10.16.1.11+cuda12.9
  (`libnvinfer*`, `libnvonnxparsers*`, `libnvinfer-bin` for `trtexec`) from the NVIDIA apt repo,
  all pinned to the `+cuda12.9` build and held with `apt-mark hold` (apt otherwise mixes in
  `+cuda13.x` dependencies). No nvcc is needed.
* ONNX export on the host (`~/EXECO/Fast-FoundationStereo/.venv_ffs`, a venv on top of the
  `execosim` conda env's torch 2.11+cu128 with timm/einops/omegaconf/onnx added):

  ```bash
  cd ~/EXECO/Fast-FoundationStereo
  .venv_ffs/bin/python ~/EXECO/DeTest-EXECO/execosim_data/scripts/ffs/export_ffs_onnx.py . [--amp] \
      --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
      --save_path weights/onnx_execo/23_36_37_480x864_it8[_amp] \
      --height 480 --width 864 --valid_iters 8 --max_disp 192
  ```

  The wrapper runs upstream `scripts/make_single_onnx.py` unchanged except that it forces the
  legacy TorchScript exporter (`dynamo=False`): torch ≥ 2.9 defaults to the dynamo exporter, which
  cannot translate `aten.adaptive_max_pool2d` used by the backbone (upstream pins torch 2.6/2.9).
  `--amp` re-enables the model's own fp16 autocast regions during export (see below).
* Engines (in the container, `/root/SaDVIO-Dense/ffs/`):

  | Engine | Build | GPU compute (trtexec, 480×864) | vs. PyTorch fp32 (one sim frame) |
  |---|---|---|---|
  | `23_36_37_480x864_it8/fast_foundationstereo_fp32.engine` | `trtexec --onnx=... ` | 25.5 ms | median 0.0005 px, p99 0.004 px, max 0.10 px |
  | `23_36_37_480x864_it8_amp/fast_foundationstereo_amp.engine` | `--amp` export, `trtexec --stronglyTyped` | 13.0 ms | median 0.0025 px, p99 0.014 px, max 0.13 px |
  | `23_36_37_480x864_it8/fast_foundationstereo_fp16.engine` | `trtexec --fp16` | 11.4 ms | **broken: no valid pixel** |

  **FP16 finding.** A plain `--fp16` build of the fp32 graph returns no valid disparity on
  TensorRT 10.16 / Blackwell — also for NVIDIA's released
  `onnx/23_36_37/576x960/23_36_37_iters_8_res_576x960.onnx`, so it is not caused by our export.
  `--fp16` lets TensorRT run the L2 normalisations (`ReduceL2`, in the cross-covariance attention
  and the group-wise-correlation volume), LayerNorms and softmaxes in fp16, which overflows; PyTorch
  autocast keeps those in fp32. `--bf16` fails to build (no bf16 ConvTranspose tactic). The working
  fast path exports under the model's own autocast (`--amp`), so the ONNX carries explicit
  fp16/fp32 casts (274 fp16 and 217 fp32 initialisers), and builds with `--stronglyTyped` so
  TensorRT keeps them. `onnxconverter-common`'s fp16 converter with an fp32 op block list crashed on
  this graph (`remove_unnecessary_cast_node`).

## 4. Verification

### Matcher unit check (one ExECoSim frame, 13_36_44 bag, image 400)

Scratch tool in the container (`/root/SaDVIO-Dense/ffs/check/ffs_check.cpp`, links
`libisae_slam.so`), compared on the host with the PyTorch model (fp32, same padding, 8 iterations,
max_disp 192):

| | valid pixels | time per pair (C++ incl. pre/post, RTX 5090) |
|---|---|---|
| SGBM (block 9, 64 disparities) | 91.2 % | 8.5 ms (CPU) |
| FFS fp32 engine | 98.3 % | 28.7 ms |
| FFS fp32 + LR check 1 px | 98.1 % | 58.3 ms |
| FFS AMP engine | 98.3 % | 15.6 ms |
| FFS AMP + LR check 1 px | 98.1 % | 32.2 ms |

On pixels valid in both, SGBM and FFS differ by a median 0.18 px, and 2.6 % of pixels differ by
more than 1 px. Visually, SGBM has speckle holes and a 64 px invalid band on the left; FFS is smooth
with sharp object boundaries.

### Raw depth before meshing (ExECoSim, 11 frames, evaluation script in DeTest-EXECO)

`execosim_data/scripts/eval_raw_disparity.py`: every 150th pair, stride-4 pixels ≤ 20 m,
back-projected with the ground-truth pose (evaluation only), distance to the placed VID012 SfM mesh
inside the aviary crop.

| | valid px | median | p90 | < 5 cm | 1-2 m | 2-3 m | 3-5 m | 5-8 m |
|---|---|---|---|---|---|---|---|---|
| SGBM | 90.1 % | 6.7 cm | 23.7 cm | 40 % | 3.1 | 3.5 | 6.0 | 11.1 cm |
| FFS | 98.7 % | 3.4 cm | 11.8 cm | 63 % | 2.7 | 2.2 | 3.0 | 5.1 cm |
| FFS + LR 1 px | 97.4 % | 3.4 cm | 11.8 cm | 63 % | 2.7 | 2.2 | 3.0 | 5.1 cm |

The SfM reference itself is only good to a few cm (SLAMesh's LiDAR mesh scores 3.5 cm against it),
so the FFS near-range numbers are close to the reference's floor. The LR check removes 1.2 % of
pixels and changes nothing measurable on the simulator.

### Pipeline runs (13_36_44, same GP / global-map settings, SaDVIO VO poses)

Run with `DeTest-EXECO/execosim_data/scripts/sadvio_container/run_dense_visu.sh` (matcher as 8th
argument; `dense_keep_all_keyframes`, `dense_gp_global_map` on; `dense_gp_register` false = A,
true = B), FFS with the fp32 engine, no LR check. Meshes compared by
`execosim_data/scripts/compare_meshes.py` in the aviary crop (accuracy = mesh → GT, completeness =
GT → mesh, F-score at 5/10 cm):

| Run | faces | area m² | acc. median | acc. p90 | compl. < 10 cm | F@5 | F@10 |
|---|---|---|---|---|---|---|---|
| sim A, SGBM | 65556 | 1139 | 20.2 cm | 81.2 cm | 58.0 % | 20.9 | 39.0 |
| **sim A, FFS** | 39486 | 645 | **8.9 cm** | **27.9 cm** | 59.9 % | **32.9** | **57.0** |
| sim B, SGBM | 52768 | 916 | 30.2 cm | 95.3 cm | 36.3 % | 13.9 | 27.1 |
| sim B, FFS | 35652 | 583 | 10.1 cm | 30.5 cm | 50.9 % | 27.8 | 50.2 |
| RealSense A, SGBM | 88444 | 1539 | 38.8 cm | 115.7 cm | 48.1 % | 13.2 | 25.1 |
| **RealSense A, FFS** | 51906 | 853 | **18.7 cm** | **53.8 cm** | 47.9 % | **19.5** | **37.0** |
| RealSense B, SGBM | 70768 | 1253 | 46.4 cm | 121.2 cm | 33.5 % | 10.1 | 19.6 |
| RealSense B, FFS | 40076 | 659 | 24.5 cm | 73.8 cm | 27.6 % | 12.9 | 24.7 |
| SLAMesh, sim VLP-16 (reference) | 23786 | 830 | 3.5 cm | 12.3 cm | 76.8 % | 60.9 | 80.8 |

Accuracy roughly halves at unchanged completeness; the smaller area is SGBM's spurious surface
disappearing (on the uncropped sim A mesh, 57 % of SGBM vertices lie 10-40 m from any keyframe
against 16.5 % for FFS, and the map has 1055 instead of 5202 cells). Per-keyframe cost in the
pipeline: FFS 29-31 ms (fp32 engine) + GP 16-24 ms, versus SGBM 9-14 ms + GP 31-35 ms (the GP step
gets cheaper because there are fewer cells).

Registration B: FFS depth reduces the drift of the carried SLAMesh correction on the simulator
(median correction 146 → 27 cm, ATE of the used poses 95 → 16 cm, VO 6.5 cm), but B is still worse
than A. On RealSense it still drifts by ~2 m (43 of 532 keyframes rejected as
`correction_too_large`); the B limitations of `doc/dense_gp_global_map.md` are unchanged.

On RealSense the remaining error is dominated by the VO trajectory (keyframe ATE 17.5-19.7 cm
against OptiTrack), not by depth.

Repeatability: a second sim A FFS run gives the same mesh quality (within 4 m and GP variance < 0.1:
6.4 cm vs 7.2 cm median) while the VO ATE varies between runs (6.3 / 8.7 cm) because the sparse
front-end is timing dependent.

## 5. Limitations and open points

* The front-end still sees `mono8`; FFS gets replicated grey even when the bag has colour.
* FFS returns a disparity for almost every pixel, including textureless and occluded areas where
  it hallucinates plausible surfaces; the LR check is the only confidence filter. FFS has no
  per-pixel uncertainty output, so the GP sensor variance (`dense_gp_variance_sensor`) is still
  the SGBM-tuned constant.
* The CUDA context and TensorRT engine are created in `SLAMBiMono::init()` / `SLAMBiMonoVIO::init()`
  together with the injector, i.e. again after every front-end re-initialisation.
* **Intermittent crash, not reproduced.** The first sim A FFS run aborted at keyframe 47 with
  `malloc(): unsorted double linked list corrupted` (no tracking failure before it). Two further sim
  A runs, two B runs and the RealSense A run with the same binary completed, and an AddressSanitizer
  build (`colcon build --build-base build_asan --install-base install_asan`, CMAKE_CXX_FLAGS
  `-fsanitize=address`, run with `ASAN_OPTIONS=protect_shadow_gap=0` for CUDA) at 0.1× bag rate
  integrated all 250 keyframes without a report, so it is most likely a race in the timing-dependent
  sparse front-end rather than an out-of-bounds access in the dense code.
* **Pre-existing front-end bug found by ASan.** At 0.5× rate the ASan build is too slow, PnP fails,
  the front-end re-initialises, and `SLAMCore::initLandmarks` (`cpp/src/slamCore.cpp`, first loop)
  dereferences an expired landmark: when `getLandmark().lock()` is null it falls through to
  `getLandmark().lock()->getFeatures()`. Not fixed here (sparse VIO core, unrelated to the matcher).
* The fp16/AMP engine was verified on one frame only; the pipeline results above use the fp32 engine.

## 6. Changelog

* 2026-09-24 — StereoMatcher interface, SGBM moved behind it, FFS TensorRT backend, build option,
  three config keys, engine preparation and FP16 finding, matcher unit check, raw depth and mesh
  comparison against SGBM (sim A/B, RealSense A/B).
