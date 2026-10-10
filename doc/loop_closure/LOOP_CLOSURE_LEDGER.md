# SaDVIO loop closure — ledger

Started 2026-10-07 at the user's request: "can we build both options and test. also anyloc with cpu should be tested
as well". This file is the work's state: read it first, update it last.

## Rules (carried over from the IMU fix work)

- Branch `dense_devel`; do not stage or commit, the user does.
- Build and run in container `sad_vio_dense`, folder `/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel`; never touch the
  simval checkout `/root/SaDVIO-Dense/SaDVIO-EXECO`.
- Ground truth only in the evaluation tools (`tools/lc_eval.py`, the IMU work's `eval_traj.py`); the describe tools and
  SaDVIO never see it.
- `Simulator_Validation/SaD_VIO_data/` holds only what containers need (datasets, `scratch/`); tools, results and notes
  live here under `doc/loop_closure/`.
- Runs share the machine: pin to the efficiency cores 8–15 / 16–23 at nice 10, at most two at a time, cores 0–7 free.

## Plan

| Phase | What | Status |
|---|---|---|
| 0 | Offline loop-detection benchmark on keyframe images (no C++ changes): ORB bag-of-words (DBoW3 + ORB-SLAM3 vocabulary; 150 / 500 FAST+ORB, 1000 pyramid ORB), AnyLoc (DINOv2 ViT-G/B/S), MegaLoc, proximity gate; recall, precision, cost on GPU and on 8 CPU cores | running |
| 1 | Shared C++: keyframe database (descriptors + 3D points, no images), verification (descriptor matching + PnP RANSAC), pose graph (4-DoF VIO, 6-DoF stereo VO, Sim3 mono VO), loop thread, output files | |
| 2 | Option 1: proximity gate + bag-of-words → separate pose graph (window untouched, drift correction on the output) | |
| 3 | Option 2: learned retrieval (ONNX: TensorRT on GPU, OpenVINO on CPU) → relocalization factors in the window + pose graph | |
| 4 | Evaluation matrix: TUM-VI rooms + magistrale2, EuRoC, all modes, with and without loop closure; ATE and cost | |

## Phase 0: offline loop-detection benchmark

Keyframes: those of a reference run per sequence (`tools/make_kf_sets.py`): EuRoC `sp2_fix_sadvio` (SaDVIO, marg +
sparse), TUM-VI rooms `viewer8_vio` (stereo VIO). 17 sequences, 11 728 keyframes with ground truth.

Labels (`tools/lc_eval.py`), for a query and an earlier keyframe at least 20 s before it: positive within 1 m and
30°, negative beyond 3 m or 60°, ambiguous in between (neither hit nor false positive).

Tools:
- `tools/lc_describe.py`: learned methods in the `hloc_simval` container (AnyLoc and DINOv2 / MegaLoc there). AnyLoc
  ViT-G uses its published indoor vocabulary; ViT-B and ViT-S have none, so each dataset gets a vocabulary fitted on
  the other dataset (EuRoC queries: TUM-VI vocabulary, and the reverse).
- `tools/bow_bench.cpp`: DBoW3 (BSD) in `/root/SaDVIO-Dense/lc_bench/` of the SaDVIO container, with ORB-SLAM3's
  `ORBvoc.txt` (converted once to DBoW3's binary format there).
- `tools/lc_eval.py`: the metrics.
- Bug found and fixed: TUM-VI images are 16-bit (`_512_16`); PIL's `convert("RGB")` saturated them to white, so the
  first learned-descriptor pass on the rooms (and the ViT-B/S vocabularies fitted on them) was invalid and was
  redone. OpenCV (bag-of-words) scales 16-bit images correctly.

### Phase 0 results: bag-of-words (DBoW3, ORB-SLAM3 vocabulary), 17 sequences, query-weighted

| setting | R@1 | R@5 | R@100P | R@95P | ms per KF (features + transform) | ms per query |
|---|---|---|---|---|---|---|
| fast150: the front end's own points (~60–107 kept on the grid) | 0.47 | 0.69 | 0.06 | 0.19 | 0.9–2.0 | < 0.3 |
| fast500: VINS-style extra corners (~130–330) | 0.52 | 0.73 | 0.14 | 0.27 | 1.0–2.6 | < 0.6 |
| orb1000: pyramid ORB, as the vocabulary was trained | **0.74** | **0.88** | **0.35** | **0.54** | 4.3–7.7 | < 1.8 |

- The vocabulary was trained on pyramid ORB; single-scale FAST matches it poorly. Option 1 extracts 1000 pyramid ORB
  per KF in the loop module (not the tracker's points).
- Proximity gate (drift of these reference runs is small): a true loop KF is inside a 1 m gate 93–100 % of the time,
  inside 2 m always, but the 2 m gate holds 11–571 candidates (hundreds in the rooms); ranking by bag-of-words inside
  the gate barely changes R@1.

CPU cost of the learned methods (PyTorch fp32, 8 efficiency cores, TUM-VI 512 x 512, ms per image): MegaLoc 204,
AnyLoc ViT-S 214, AnyLoc ViT-G (as published) 5 584. GPU (RTX 5090, single image): MegaLoc 36–46, AnyLoc ViT-S 18–25,
ViT-B ~45, ViT-G 196. Recall: see below once the rerun with the 16-bit fix ends.

## Phase 1–2: C++ loop closure, option 1 (2026-10-07)

Code (`dense_devel`, not staged):
- `cpp/thirdparty/DBoW3/` (BSD, rmsalinas/DBow3 `1cc587b`, with its LICENSE), compiled into the library like ELSED;
  `cpp/CMakeLists.txt` and `ros/CMakeLists.txt`.
- `cpp/include/isaeslam/loopclosure/PoseGraph.h`, `cpp/src/loopclosure/PoseGraph.cpp`: 4-DoF (VIO, VINS-Mono's
  residual) or 6-DoF pose graph, odometry edges to the 4 previous KFs of the segment, loop edges; segments are joined
  by loops (gauge: the earliest node of each connected group). Loops enter without a robust loss (with metres of
  drift a correct loop starts far beyond any robust scale and would be ignored, which is what the first version did);
  the worst loop above chi2 20 is disabled and the graph solved again (at most 5 solves).
- `cpp/include/isaeslam/loopclosure/LoopClosure.h`, `cpp/src/loopclosure/LoopClosure.cpp`: per KF leaving the window
  (callback in `LocalMap::discardLastFrame`, before its sensors are cleaned; the window's KFs before a reset via
  `logWindowBeforeReset`): 1000 pyramid ORB as rays (any camera model) + BoW vector; camera 0's landmarks in camera 0
  with oriented ORB computed at their pixels. Candidates (bow / proximity / proximity_bow), verification by PnP RANSAC
  in both directions (each KF's landmarks against the other's landmarks and ORB), the two relative poses must agree
  (0.3 m, 5 deg). Output: `log_slam/loops.csv`, `loop_attempts.csv`, `results_loop.csv` (final), 
  `results_loop_online.csv` (as corrected at each KF); profiler lines.
- Config: `loop_closure`, `loop_detector`, `loop_vocabulary`, `loop_min_gap`, `loop_min_inliers`, `loop_gate_radius`
  (validated; mono VO refused until a Sim3 graph exists). Vocabulary: `SaD_VIO_data/vocabulary/ORBvoc.dbow3` (ORB-SLAM3's
  ORBvoc.txt converted by DBoW3; data, not committed: ORB-SLAM3 is GPLv3).
- Tests: `PoseGraphTest.loopCorrectsTheDrift` (4-DoF and 6-DoF), `wrongLoopIsRejected`, `loopJoinsTwoSegments`.

Found while building it:
- The KF's sensors and landmarks are cleaned by `discardLastFrame` before its result row is written, so the first
  hook (in `writeResultRow`) received empty frames: 0 KFs processed.
- The front end's ORB descriptors are computed on FAST points without orientation: matched against oriented ORB they
  are at chance level (mean Hamming 121 of 256; recomputed with the intensity-centroid angle: 8). The loop module
  computes oriented ORB at the landmark pixels itself (OpenCV keeps a provided angle: checked, Hamming 0).
- Matching the query's landmarks against the candidate's ORB only gave 14 matches (median) on true revisits; adding
  the candidate's landmarks as targets: 22.

First result, room1 stereo VIO (default config), bow, 25 inliers: odometry ATE 0.270 m; loop-closed final 0.040 m,
online 0.082 m; 57 loops, all true (ground truth within 1.5 m / 45 deg); 10.7 ms per KF, 24 ms per pose graph solve.

### Phase 0 complete: learned methods (after the 16-bit fix), 17 sequences, query-weighted

| method | R@1 | R@5 | R@100P | R@95P | GPU ms/img (PyTorch) | CPU ms/img (8 E-cores) |
|---|---|---|---|---|---|---|
| MegaLoc (DINOv2-B + SALAD-type head, 8448-D) | **0.87** | **0.95** | **0.82** | **0.84** | 36–46 | 204 |
| AnyLoc ViT-G/14 (as published, indoor vocabulary) | 0.85 | 0.94 | 0.73 | 0.82 | 142–363 | 5 584 |
| AnyLoc ViT-B/14 (cross-dataset vocabulary) | 0.84 | 0.94 | 0.78 | 0.82 | ~45 | – |
| AnyLoc ViT-S/14 (cross-dataset vocabulary) | 0.85 | 0.94 | 0.78 | 0.82 | 18–25 | 214 |
| bag-of-words, 1000 pyramid ORB | 0.74 | 0.88 | 0.35 | 0.54 | – | 4–8 |

Full table: `results/phase0_retrieval.md`. AnyLoc ViT-G is no better than ViT-S at 26x the CPU cost. MegaLoc is the
learned detector of option 2.

## Option 1 tuning (2026-10-07/08)

`results/tune_option1.md`: 4 sequences x {bow 25/15/12 inliers, proximity 15, proximity_bow 15}, stereo VIO. 12
inliers is best (room4 final ATE 0.147 -> 0.049 m) and every accepted loop has a correct relative pose (its measured
distance between the two KFs within 0.15 m of the ground truth's: 31/31, 40/40, 525/525, 101/101): the two-direction
agreement keeps precision. Default: bow, 12 inliers.

First option 1 matrix (`lc1_*`, lane A only, under heavy load from another session: costs not usable) showed:
- TUM-VI rooms: final ATE 0.16–0.40 m -> 0.013–0.029 m (stereo VIO), 0.10–0.14 -> 0.022–0.042 (stereo VO),
  0.27–0.41 -> 0.039–0.068 (mono VIO); EuRoC roughly halved.
- Stereo VO V2_03 (17–25 re-initializations): online ATE 0.63–0.72 m against 0.18–0.26 odometry. All its loops join
  different segments; a joined segment's later rows were written in another segment's frame under its own label, so
  the evaluation aligned rows of two frames together. Fixed: results_loop.csv keeps the segment column and adds a
  group column (segments joined by loops; `lc_runs_eval.py` also reports the ATE with joined segments aligned as one);
  the online file labels a segment anew each time its frame changes. Rerun: online 0.207 against 0.220.
- magistrale2: 7.0 -> 3.0 m, not lower. Two reasons: the candidate list was filled by the end's own recent KFs
  (fixed: at most one candidate per 20 s window, and up to 2 loops per KF from different times); and the end passes
  the start position facing the other way (ground truth: 179 deg, no end sample shares a view with the start), so the
  forward camera cannot close that loop. Rerun: 5.5 -> 2.65 m.

## Option 2: learned detector + window correction (2026-10-08)

- MegaLoc exported to ONNX (`tools/export_megaloc_onnx.py`; 322x322 for TUM-VI, 322x504 for EuRoC; max |onnx -
  torch| 2e-7), in `SaD_VIO_data/models/` (data, not committed). TensorRT engines built with trtexec. **fp16 breaks
  MegaLoc** (room1: 118 loops against 448; its optimal-transport aggregation runs in log space): fp32 engines only.
- `cpp/include/isaeslam/loopclosure/GlobalDescriptor.h` / `.cpp`: OpenVINO (CPU) and TensorRT (GPU) backends, each
  behind its build option (`cpp/cmake/ISAESLAM_VPR.cmake`: `ISAESLAM_WITH_VPR_OPENVINO`, `ISAESLAM_WITH_VPR_TENSORRT`).
  Config `loop_detector: learned`, `loop_model`, `loop_model_device` (CPU|GPU), `loop_model_threads`.
- Cost on room1 (idle machine, 8 E-cores): descriptor 3.1 ms on the GPU (TensorRT fp32), 164 ms on the CPU (OpenVINO,
  8 threads); whole loop module 24 / 184 ms per KF. Room1 stereo VIO: 0.305 -> 0.025 m (GPU), 0.263 -> 0.023 (CPU).
- Window correction (`loop_correct_window: 1`): after a loop the back end moves its window (KF poses, velocities,
  landmarks, the KF being inserted) by the pose graph's correction, at the start of its next step
  (`SLAMCore::applyLoopCorrection`); the marginalization prior, linearized in the old frame, is dropped then
  (`resetMarginalization`); the pose graph re-anchors the segment's odometry (`PoseGraph::reanchor`). Room1 stereo VIO:
  window output 0.044 m (0.032 with marginalization + sparse prior), loop-closed final 0.026 (0.019).
- Tests: `PoseGraphTest.*` (4) pass; `ConfigTest` covers the new options.

Running (2026-10-08, binary snapshot `lc3`): the full matrix `tools/run_lc_matrix.sh` with variants opt1 (bow,
output only), opt2 (MegaLoc GPU + window), bowwin (bow + window), cpu (MegaLoc on the CPU, 5 sequences), on 18
sequences x {stereo VIO, stereo VO, mono VIO} + EuRoC SaDVIO (marginalization + sparse prior).

## Matrix results (2026-10-08, binary `lc3`; `results/matrix_summary.md`, per run: `results/matrix_runs.json`)

343 runs: opt1 (bag-of-words, output corrected) and opt2 (MegaLoc on the GPU, window corrected) 2 runs per cell,
bowwin (bag-of-words, window corrected) 1, cpu (MegaLoc on the CPU, window corrected) 1 on 5 sequences; 18 sequences x
{stereo VIO, stereo VO, mono VIO} + EuRoC SaDVIO (marginalization + sparse prior). No run was cut short (coverage
within 0.9 of the opt1 window everywhere). Without loop closure = opt1's window (untouched by option 1).

Median ATE (m) over the sequences:

| config | without | opt1 final / online | opt2 window / final | bowwin window / final |
|---|---|---|---|---|
| stereo VIO (18) | 0.169 | **0.050** / 0.059 | 0.087 / 0.046 | 0.071 / 0.050 |
| stereo VO (18) | 0.152 | **0.046** / 0.084 | 0.083 / 0.051 | 0.081 / 0.050 |
| mono VIO (18) | 0.402 | 0.133 / 0.255 | 0.191 / **0.116** | **0.161** / 0.133 |
| SaDVIO, EuRoC (11) | 0.058 | **0.039** / 0.064 | 0.064 / 0.049 | 0.075 / 0.056 |

- RPE over 1 s is unchanged (loop closure removes drift, not local error); over 5 s it improves where drift builds
  quickly (rooms: stereo VIO room5 0.113 -> 0.045, mono VIO room2 0.994 -> 0.091; magistrale2).
- Loops accepted: 17 024 (opt1), 14 648 (opt2), 8 550 (bowwin), 1 507 (cpu); correct by the ground truth 99.6–99.7 %
  in every variant; the pose graph absorbs the rest.
- Window correction matters for mono VIO: its odometry fails on MH_01 / MH_02 / MH_05 (1.0–1.5 m) and correcting the
  output only cannot repair that (MH_01 1.03 -> 1.05), correcting the window can (bowwin 0.12, opt2 0.52; MH_05 1.15 ->
  0.23 / 0.26). With marginalization (SaDVIO) it costs accuracy (0.039 -> 0.049–0.056): the prior is dropped at each
  correction.
- Detector: MegaLoc and bag-of-words give about the same final ATE once the window is corrected (stereo VIO 0.046 /
  0.050, mono VIO 0.116 / 0.133, SaDVIO 0.049 / 0.056).
- Cost (idle machine, room1, 8 E-cores): loop module ~11 ms per KF with bag-of-words, 24 ms with MegaLoc on the GPU
  (descriptor 3.1 ms), 184 ms with MegaLoc on the CPU (descriptor 164 ms); pose graph ~11 ms per solve (matrix
  median). The matrix's own timings ran next to another session's unpinned jobs (CPU descriptor 317 ms there).
- magistrale2: 7.5 -> 3.5–3.7 m (stereo VIO), 7.0 -> 1.3–1.7 m (stereo VO); the closing loop is not observable (the
  end faces the opposite way).

**Not caused by loop closure — open item:** default stereo VIO (no marginalization) on V2_03 fails: 23–51 m in every
matrix run. The same configuration with loop closure off and with the `e035ccc` build also fails (11.6 / 1.4 m;
9.2 m); earlier builds gave 0.12–1.8 m, so it is a flaky case of the default configuration around V2_03's visual
dropout. With marginalization (SaDVIO) V2_03 is stable (0.10–0.19 m).

**Recommended configuration:**
- Stereo VIO, stereo VO, SaDVIO: `loop_closure: 1`, `loop_detector: bow`, `loop_correct_window: 0` (option 1):
  median ATE / 3 (stereo) and / 1.5 (SaDVIO), ~11 ms per KF, no GPU, the window and its prior untouched.
- Mono VIO: `loop_correct_window: 1` (the window correction repairs failing mono VIO runs), bag-of-words.
- MegaLoc (`loop_detector: learned`): available, not the default: no accuracy gain on these datasets, needs a GPU
  (on the CPU 164 ms per KF). Its strength (viewpoint and lighting changes, R@1 0.87 against 0.74 offline) matters
  more for cross-robot recognition (MeshCSLAM).

## magistrale2: VIO configuration and pose graph DoF (2026-10-08)

Question from the user: why is stereo VIO about the same as stereo VO on magistrale2, and VO better with loop closure?
- The matrix used default VIO (no marginalization): median 7.2 m on magistrale2 (12 earlier runs) against 1.8 m with
  marginalization + td (42 runs). Without the prior, the 12-KF window forgets the IMU's information and drifts like VO.
- With an IMU the pose graph was 4-DoF (roll, pitch kept from the odometry). Default VIO ends magistrale2 4.8 deg off
  in tilt and the 4-DoF graph cannot fix it (4.3 deg after); VO starts at 12.7 deg and its 6-DoF graph halves it
  (5.5 deg). VO also finds ~35 % more loops there (267–270 against 196–203 per run).
- New option `loop_graph_dof` (0: 4 with an IMU, 6 without; 4; 6), validated (4 needs an IMU).

Runs (`tools/run_lc_dof.sh`, binary `lc4`, bag-of-words, output corrected, ATE m, 2 runs each):

| configuration | graph | odometry | loop-closed final |
|---|---|---|---|
| default VIO (matrix) | 4-DoF | 7.63 / 7.28 | 3.53 / 3.85 |
| default VIO | 6-DoF | 6.03 / 5.62 | 1.49 / 1.83 |
| VIO marginalization + td | 4-DoF | 1.35 / 2.52 | 0.53 / 1.33 |
| VIO marginalization + td | 6-DoF | 1.29 / 0.82 | 0.76 / 0.65 |
| SaDVIO (marg + sparse + td) | 4-DoF | 1.85 / 1.51 | 0.69 / 0.59 |
| stereo VO (matrix) | 6-DoF | 7.27 / 6.67 | 1.95 / 1.45 |

6-DoF repairs default VIO (3.5–3.8 -> 1.5–1.8 m); with marginalization 4- and 6-DoF are within the run spread.
Pose graph solve over ~3 200 KFs: 51–53 ms (4-DoF), 65–78 ms (6-DoF). Candidate default for loop_graph_dof 0:
4-DoF with an IMU and marginalization, 6-DoF otherwise (to check on more sequences: the rooms were fine with 4-DoF).

## Live view in RViz (2026-10-08)

- `LoopClosure::display()`: a snapshot (loop-closed trajectory as segments per segment, accepted loops, the latest
  loop); `SLAMCore::getLoopClosure()` (atomic: created by the back end, read by the viewer).
- `ros/src/rosVisualizer.h`: topics `loop_traj` (green), `loop_edges` (red), `loop_new` (yellow), at most 5 Hz;
  `ros/launch/isae_slam.rviz` shows them. The ROS package is built with the learned backends too
  (`--cmake-args -DISAESLAM_WITH_VPR_OPENVINO=ON -DISAESLAM_WITH_VPR_TENSORRT=ON`).
- Checked on a room1 live run (real time, ROS node): 547 loops, 18 ms per KF; the marker topics carry data (the
  visualizer needs `enable_visu: 1`, which `live_ros_compare.py` sets; the evaluation configs have it off).
- Live demo: `tools/live_ros_compare.py --seq magistrale2 --display :20 --label live_lc --node vo:bimono:loop_closure=1
  --node vio:bimonovio:loop_closure=1,marginalization=1,estimate_td=1` (in `doc/vio_imu_fix/tools`).

## magistrale2: the end is recognised as the start room, but cannot be verified (2026-10-09)

Place recognition does propose the start for the end: queries of the last 90 s had keyframes of the first 60 s as
candidates 725 times (bag-of-words) and 654 times (MegaLoc) in a run, about as often as recent keyframes of the end.
Verification rejects all of them: 5 matches median (max 15–17), at most 0–6 PnP inliers, against 7–8 (max 42) and
32–35 inliers for end-to-end candidates (96–145 accepted). Ground truth: the end passes within 3 m of the start
positions throughout but facing ~170 deg the other way; no end sample has a start sample within 3 m and 60 deg. The
retrieval recognises the room's overall look; the verification needs the same points seen from both, and the camera
faces the opposite walls. Closing this loop would need a view of the start's side (a turn in the room), a rear
camera, or structure-based matching (MeshVLPR-style), not a looser verification.

## Live run by the user (2026-10-09): room1, stereo VIO, bag-of-words, output corrected, ROS node at real time

`live_ros_compare.py --seq room1 --display :20 --label live_lc --node lc:bimonovio:loop_closure=1`, 541 loops
(530 / 531 correct), loop closure 18.5 ms per KF, pose graph 18 ms per solve:

| trajectory | ATE (APE trans.) | APE trans. mean / max | APE rot. RMSE | RPE 1 s | RPE 5 s | RPE rot. 1 s |
|---|---|---|---|---|---|---|
| without loop closure | 0.290 m | 0.263 / 0.673 m | 4.07 deg | 0.027 m | 0.084 m | 0.93 deg |
| with, final | 0.024 m | 0.021 / 0.074 m | 1.36 deg | 0.023 m | 0.038 m | 0.88 deg |
| with, online | 0.036 m | 0.032 / 0.091 m | 1.42 deg | 0.036 m | 0.053 m | 0.94 deg |

## Pose graph DoF on TUM-VI rooms + EuRoC (task 2, 2026-10-09)

`tools/run_lc_dof_all.sh` (binary `lc4`, bag-of-words, output corrected, 2 runs per cell): 6-DoF runs paired with
the matrix's 4-DoF ones, plus SaDVIO 4-DoF on the rooms. Failed odometry (default stereo VIO on V2_03, tens of m)
excluded.

| configuration | sequences | median final ATE 4-DoF -> 6-DoF | 6-DoF better in | median final / odometry, 4 -> 6 |
|---|---|---|---|---|
| stereo VIO, default | 16 | 0.046 -> 0.038 m | 12 / 16 | 0.47 -> 0.43 |
| mono VIO, default | 17 | 0.123 -> 0.125 m | 9 / 17 | 0.62 -> 0.51 |
| SaDVIO (marg + sparse) | 17 | 0.035 -> 0.032 m | 14 / 17 | 0.56 -> 0.53 |

With magistrale2 (6-DoF much better for default VIO, equal with marginalization), 6-DoF is as good or better in every
configuration: the IMU's roll and pitch are not exact enough to be held fixed. The mono VIO odometry differed a lot
between the two batches (MH_01 1.03 against 0.26 m), so its ratio to the same run's odometry is the fairer measure.
Cost: 6-DoF solves are slower (magistrale2, ~3 200 KFs: 65–78 ms against 51–53 ms).

Decision (task 2): `loop_graph_dof: 0` (automatic) is now 6-DoF in every mode; 4-DoF stays available as an explicit
option for IMU modes. Suite 83 / 84 (`LineFeatureMatching`, coin flip).

## Clean timing pass (task 3, 2026-10-09; SaDLIO paused, machine idle)

`tools/run_lc_timing.sh` (binary `lc5`, automatic graph = 6-DoF): stereo VIO, output corrected, one run at a time on
cores 16–23; means of 2 runs (magistrale2: 1). Times in ms.

| sequence (KFs) | variant | back end / KF | loop module / KF | descriptor / KF | pose graph / solve | loops | wall |
|---|---|---|---|---|---|---|---|
| room1 (~1 230) | none | 11.7 | – | – | – | – | 29 s |
| | bag-of-words | 37.2 | 24.6 | – | 31 | 566 | 61 s |
| | MegaLoc GPU | 34.1 | 21.3 | 3.0 | 24.5 | 456 | 57 s |
| | MegaLoc CPU | 197.9 | 182.7 | 163.3 | 27 | 451 | 260 s |
| MH_01 (~390) | none | 9.3 | – | – | – | – | 20 s |
| | bag-of-words | 23.2 | 12.3 | – | 8.3 | 24 | 23 s |
| | MegaLoc GPU | 28.4 | 13.7 | 4.0 | 7.8 | 29 | 25 s |
| | MegaLoc CPU | 328.9 | 305.6 | 295.6 | 8.9 | 24 | 143 s |
| magistrale2 (~3 170) | none | 9.2 | – | – | – | – | 88 s |
| | bag-of-words | 27.0 | 16.6 | – | 69 | 273 | 140 s |

- Front end unchanged (3.8–4.5 ms per frame; 4.7–4.8 with MegaLoc on the CPU sharing the cores).
- With `multithreading: 0` the loop closure runs in the back end's step (back end ~10 -> 23–37 ms per KF); with
  `multithreading: 1` in its own thread.
- Pose graph solve grows with the KFs: 8 ms (~390), 25–31 ms (~1 230), 69 ms (~3 200, 6-DoF).
- Real time: room1 (highest KF rate, ~8.7 KF/s) costs 0.21 s of CPU per second of data with bag-of-words; all
  bag-of-words / GPU runs stay well under real time (room1: 61 s of processing for 141 s of data).
- MegaLoc on the CPU is not real-time capable here: 163 ms (322 x 322) / 295 ms (322 x 504) per KF; room1 took 260 s
  for 141 s of data. Live it would need its own thread and a subset of KFs.

## Unit tests of the verification (task 4, 2026-10-09)

`LoopVerificationTest` (`cpp/tests/loopclosure_test.cpp`, access through `LoopClosureTestAccess`, a friend of
`LoopClosure`): synthetic scene of random points with random 256-bit descriptors, views differing by 6 bits.
- `revisitIsAcceptedWithItsRelativePose`: 0.5 m / 10 deg apart, accepted, relative pose within 5 cm / 0.5 deg.
- `differentPlaceIsRejected`: another scene, rejected.
- `directionsThatDisagreeAreRejected`: the candidate's landmarks 0.6 m off its image; both PnP directions succeed (61
  and 120 inliers) but disagree by 0.60 m (> 0.3): rejected by the agreement check, as intended.
Suite: 86 / 87 (`LineFeatureMatching`, coin flip).

## The marginalization prior follows the window correction (task 5, mandatory, 2026-10-09/10)

Before: `loop_correct_window: 1` dropped the prior at each correction (`resetMarginalization`), costing SaDVIO
accuracy (EuRoC 0.039 -> 0.049–0.056 m).

First attempt (discarded): re-expressing the prior's Jacobian in the corrected world (J <- J M^-1). The invariance test
showed it is only first-order exact: the prior's translation coordinate is the world-to-frame translation, which
depends on the rotation increment through the world origin (dt' = R_C dt - R_C (Exp(dr) - I) R_C^T t_C); with a 3.6 m
correction the residual was off by 5–9 %.

Exact design (kept): the prior stays expressed in the world it was linearized in and records the change,
`Marginalization::_W_prior` (prior's world <- current world; `transformWorld(C)`: W <- W C^-1; reset to identity by
`storeLinearizationPoint`, i.e. by the next marginalization, which linearizes in the current world).
- Dense factor (`MarginalizationFactor`): the current states are mapped into the prior's world before the offsets
  (T_f_w W^-1, R_W v, W T_l), and so are the increments: y_r = R_W x_r, y_t = R_W x_t - R_W (Exp(x_r) - I) R_W^T t_W,
  y_v = R_W x_v; Jacobians by the chain rule (the rotation columns gain the translation's term
  A R_W Exp(x_r) [R_W^T t_W]x Jr(x_r)). With W = identity it is the previous code.
- Sparse factors (`AOptimizer::addSparsePriorResiduals`) take their values in the current world:
  `linPoseCurrent()` (T_lin W), `linVelocityCurrent()` (R_W^T v_lin), velocity columns of the absolute factor's
  information times R_W; the VO landmark prior W^-1 p with information S R_W, the landmark-to-landmark differences
  R_W^T d with S R_W. The pose-to-landmark factors (in the frame) do not change.
- `AOptimizer::transformPriors(C)` applies it to both prior slots; `SLAMCore::applyLoopCorrection` calls it instead
  of `resetMarginalization`; the slots' copy carries `_W_prior`.
- Points only translate (no orientation), in `applyLoopCorrection` and in `AOptimizer::restoreGauge`. The gauge
  restore rotated point poses by its yaw, after which the reprojection factors (increments `T_w_l * x`) and the
  sparse prior's landmark factors (increments along world axes) disagreed by that yaw: a small inconsistency in
  SaDVIO's sparse mode, fixed with it.

Tests (`cpp/tests/imu_test.cpp`):
- `priorFollowsTheWindowWhenTheWorldMoves`: a real stereo VIO prior with kept landmarks, states perturbed (0.02),
  the world moved by a yaw + 3.6 m (and a full rotation): the transformed prior gives the moved states the residual
  the old states had (within 1e-4); the prior left as it was does not (> 100 times that); its Jacobians with the world
  change match numerical ones. Failed with the first attempt (5–9 %), passes with the exact design.
- `sparsePriorFollowsTheWindowWhenTheWorldMoves`: the absolute factor's residual is unchanged (1e-9).
- Suite 88 / 89 (`LineFeatureMatching`, coin flip). ROS package rebuilt.

Running: `tools/run_lc_prior.sh`: drop (binary `lc5`) against transform (`lc6`), window correction, SaDVIO on the
rooms + EuRoC and VIO marginalization + td on the rooms + magistrale2, 2 runs per cell.

### First real-data comparison: the transformed prior fought gravity

`tools/run_lc_prior.sh` (drop `lc5` against transform `lc6`, window correction, 2 runs per cell): transform was worse,
median final SaDVIO 0.038 against 0.030 m, and VIO with marginalization + td worse in every cell, room2 diverging
(129 m). Cause: the pose graph is 6-DoF now, so its correction contains a tilt, and a window moved by a tilt
contradicts gravity (fixed along the world z axis by the IMU factors). Dropping the prior hid it (the IMU levels the
window again); a kept prior, which carries the old tilt, fights the IMU factors. The invariance tests have no
gravity factors and could not see it.

Fix: with an IMU (`LoopClosure::Options::window_gravity`, set for monovio / bimonovio) the window moves by the
correction's yaw and translation only, the translation chosen so that the segment's latest KF lands where the full
correction puts it; the pose graph re-anchors by what was applied and keeps the rest (the tilt) in its output.
Test `LoopWindowCorrectionTest.imuWindowIsMovedWithoutTilt`: a correction with a tilt gives a pure rotation about z,
the latest KF lands exactly, the graph keeps the rest.

Quick check (`tools/run_lc_prior_check.sh`, binary `lc7`) on the worst cells, final ATE (m), drop / transform 6-DoF
window / transform gravity-consistent window: VIO marg + td room2 0.021 / 129.6 / 0.016, room4 0.032 / 0.125 / 0.048,
magistrale2 0.82 / 3.54 / 0.67; SaDVIO room2 0.024 / 0.057 / 0.040, MH_01 0.027 / 0.050 / 0.035, V2_03 0.144 /
0.179 / 0.101. The divergence is gone; against dropping, mixed. Full comparison: `tools/run_lc_prior_grav.sh`.

### Full comparison (binary `lc7`, `tools/run_lc_prior_grav.sh`, 2 runs per cell)

Final / window ATE medians (m):

| configuration | output only (final) | drop prior: window / final | transform, 6-DoF window | transform, gravity-consistent window |
|---|---|---|---|---|
| VIO marg + td (7: rooms + magistrale2) | – (magistrale2 0.705) | 0.041 / 0.021 | 0.125 / 0.051 (room2: 129 m) | **0.035 / 0.017** (final better than drop in 5 / 7) |
| SaDVIO (17: rooms + EuRoC) | 0.032 | 0.050 / 0.030 | 0.057 / 0.038 | 0.057 / 0.035 (better than drop in 11 / 17, 2 ties) |

- No divergence with the gravity-consistent window. With a dense prior, keeping it is better than dropping it (window
  and final); with SaDVIO it wins in most sequences (V2_03 0.144 -> 0.101, MH_05 0.062 -> 0.053), the median being
  slightly higher from how the per-sequence values fall; all three within a few mm of output-only correction.
- Recommendation: SaDVIO keeps output-only correction as the simplest default; window correction with marginalization
  is now safe when the live window must run in the corrected frame (MeshCSLAM). VIO with a dense prior: window
  correction is now the better choice.
- Task 5 done. Suite 89 / 90 (`LineFeatureMatching`, coin flip).

## Defaults (2026-10-10, decided by the user)

Shipped config `ros/config/config.yaml`: `loop_closure: 1`, bag-of-words, automatic graph (6-DoF), and the new
automatic window correction `loop_correct_window: -1` (also the code default): the window is moved for VIO with a
dense prior (better than output-only and than dropping the prior) and for mono VIO (repairs failing runs); elsewhere
(VIO with no prior or with a sparse prior, stereo VO) the output is corrected and the window left as it is.
Mono VO: loop closure is skipped with a warning (no Sim3 graph yet) instead of refusing the config, so the shipped
config (mono) stays valid. The evaluation base config (`doc/vio_imu_fix/configs/base_config.yaml`) keeps
`loop_closure: 0`, so odometry runs stay comparable with the earlier ones. `ConfigTest` covers the loop options
(automatic window value, ranges, 4-DoF without IMU, unreadable vocabulary, the mono warning). Suite 89 / 90
(`LineFeatureMatching`, coin flip).

Naming from here on: "VIO no prior" (`marginalization: 0`), "VIO dense prior" (`marginalization: 1`), "VIO sparse
prior" (`marginalization: 1`, `sparsification: 1`, the SaDVIO paper's configuration, called "SaDVIO" in the earlier
sections and run labels).
