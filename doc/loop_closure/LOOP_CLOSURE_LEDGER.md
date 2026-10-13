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

## 2026-10-10 — Mono VO: Sim3 pose graph

Mono VO's scale is unknown, drifts, and every segment (re-initialization) starts with a scale of its own. Its loop
closure therefore needs a Sim3 graph, not a rigid one. Until now mono VO skipped loop closure with a warning.

Code:
- `PoseGraph(four_dof, neighbors, sim3)`: each node also has a log scale. `Sim3Error` (7 residuals) adds the log
  scale ratio to the 6-DoF residual and divides the translation by node i's scale.
  - Sigmas: odometry 0.05 and loops 0.05 in log scale. 0.02 for the odometry held the scale too stiffly: in the unit
    test, 4 good loops were rejected.
  - The gauge (earliest node of each group) also fixes its scale.
  - `correction(segment)` is a similarity. `LoopClosure::correct` returns the corrected rigid pose (the similarity
    moves the position; the scale is not part of the pose).
- Verification (`Options::sim3`) is the same two-way PnP. The scale ratio of the two maps comes from the inliers that
  hit the other keyframe's landmarks: the depth of the matched landmark (candidate units) over the depth of the query
  landmark brought into the candidate's camera (query units), and the reverse.
  - It takes the median over both directions, with at least 5 ratios in all. When both directions have at least 3,
    their medians must agree within 15%.
  - The two directions must then agree once the query's translation is scaled: within 10% of the candidate's median
    landmark depth (0.3 m at 3 m, as the metric threshold), and within 5°.
  - `loops.csv` gains a last column, the scale ratio.
- Config: `loop_graph_dof` 7 (Sim3, mono only). Automatic (0) means 7 in mono, else 6. Mono VO corrects the output
  only: `loop_correct_window: 1` in mono is an error, since the window cannot follow a similarity.
- `lc_runs_eval.py`: in mono, the measured loop distance (in the match KF's map units) is scaled by its segment's
  Sim3 alignment to the ground truth before the 0.15 m check.
- Tests: `PoseGraphTest.sim3LoopCorrectsScaleDrift` uses two laps with 0.3% shorter steps each KF (45% scale drift).
  Sim3 brings the last KF to under 15% of the odometry error and under half of the 6-DoF graph's error.
  `LoopVerificationTest.sim3MeasuresTheScaleRatio`: a candidate map 2.5× larger is accepted with a ratio of 2.5 ± 0.05
  and the right pose; the rigid verification rejects it. ConfigTest has the mono cases.

Runs: labels `lc_mono7` (Sim3) and `lc_mono6` (6-DoF graph, scale ratio ignored), bow detector, fallback off.
Snapshot `lcm1`, 18 sequences × 2 runs, `tools/run_lc_mono.sh`. Table: `results/lc_mono.md`. Medians of 36 runs (m):

| | odometry | loop final | loop online | segments joined by loops | correct loops |
|---|---|---|---|---|---|
| Sim3 | 0.438 | **0.255** | 0.365 | 0.288 | 2899 / 3035 (95.5%) |
| 6-DoF | 0.497 | 0.492 | 0.494 | 0.583 | 2166 / 2619 (83%) |

Sim3 improves the final trajectory by more than 10% in 22 of 36 runs and never makes it worse:
- room3 0.31 → 0.07 and 0.33 → 0.09;
- room1 0.37 → 0.10;
- MH_03 0.65 → 0.18;
- V1_01 0.43 → 0.10.

The 6-DoF graph, which cannot absorb scale drift, improves the final trajectory in 17 runs, by little; joining
segments at their own scales makes it worse (0.58). Both rows have runs with no loop or no gain: V1_02, V1_03, V2_03
(mono VO breaks into many short segments there).

Correct loops are 95.5% against 99.6% in stereo. The check scales a whole segment by one factor, which is approximate
under scale drift, so part of the gap is the evaluation.

Cost: loop closure 8–28 ms per KF (the rooms, with hundreds of loops, are the most), pose graph 4–90 ms per solve.
Back end unchanged (3–4 ms per KF).

Defaults: `loop_graph_dof: 0` gives Sim3 in mono. These runs had the mono VO essential-matrix fallback off. It was
then made the default, and reverted after the final matrix: with loop closure it costs mono VO 0.19 → 0.52 m
(`doc/vio_imu_fix/IMU_FIX_LEDGER.md`, 2026-10-10).

## 2026-10-10 — Dense submaps placed by the loop closure (MeshCSLAM)

MeshCSLAM needs, from each robot, "KF/submap-anchored meshes that can be moved when poses are optimized"
(`doc/SaDLIO_novelty_survey.md` §0). Until now the dense map (`MarginalDepthInjector`) fused every keyframe in world
coordinates at its odometry pose. Its geometry could not follow a loop correction: GP cells and VDB voxels keep no
per-keyframe record and cannot be un-fused.

Code (VDB-GPDF path only, `dense_submap_kfs`, default 0 = one global map as before):
- Every `dense_submap_kfs` keyframes the injector starts a new `VDBGPDFMap`, fused in the frame of its first keyframe
  (the anchor: its rectified left camera, `T_w_anchor`, in the odometry world at fusion). A submap never spans two
  odometry worlds. A new one also starts at a re-initialization (`queueFrame(frame, mesh, segment)`) and when the loop
  closure moves the window (`worldMoved()`, called by `applyLoopCorrection`).
- A closed submap runs marching cubes once, writes `log_slam/dense_submaps/submap_NNNN.ply` (anchor frame), and
  frees its grid. Marching cubes only runs on the open submap, every `dense_vdbgpdf_mesh_every` keyframes.
- `dense_submaps/submaps.csv` holds, per submap: anchor timestamp, segment, keyframes, closed, the anchor's odometry
  pose, and its corrected pose. The global mesh (`dense_vdbgpdf_mesh_path`, the `dense_mesh` topic) is the submaps
  placed at their corrected anchor poses, rebuilt at every mesh save.
- `LoopClosure::keyframeCorrection(ts, C)` gives corrected world ← the world a keyframe was handed over in: its graph
  pose times the inverse of its hand-over pose. Window corrections made since then are included (`T_w_f_handover`
  is kept apart from `T_w_f`, which follows them). The dense keyframe is queued at marginalization with the same
  pose the loop closure receives.
- `SLAMCore` sets the correction callback when its loop closure starts. The lambda holds a `shared_ptr`, so the
  injector's final write in its destructor stays valid. With submaps, the injector is kept across
  re-initializations.

Runs (`tools/run_dense_submaps.sh`, snapshot `dsm1`): SGBM, `stereo_tsdf`, submaps of 10 KFs, all keyframes kept,
loop closure on. VIO no prior (output correction) and VIO dense prior (window correction), one run each.
`tools/dsm_eval.py` compares the anchors with the ground truth. `tools/dsm_mesh_eval.py` measures the distance of the
closed submaps' vertices (1500 per submap) to the Leica scan of the Vicon room (`mav0/pointcloud0`, 1 cm subsampled).
Each placement is aligned by an SE3 fit of its anchors.

| run | submaps | anchor ATE odometry → corrected | KF ATE odometry → loop | mesh to scan, median (odometry → corrected) |
|---|---|---|---|---|
| no prior MH_01 | 38 | 0.086 → 0.053 | 0.082 → 0.047 | (no scan) |
| no prior V1_01 | 35 | 0.098 → 0.092 | 0.061 → 0.042 | 7.0 → 6.7 cm |
| no prior V2_01 | 30 | 0.062 → 0.056 | 0.044 → 0.029 | 17.1 → 17.4 cm |
| dense prior MH_01 | 55 | 0.106 → 0.047 | 0.106 → 0.058 | (no scan) |
| dense prior V1_01 | 42 | 0.096 → 0.086 | 0.052 → 0.039 | 6.8 → 6.3 cm |
| dense prior V2_01 | 39 | 0.060 → 0.053 | 0.035 → 0.025 | 13.7 → 14.1 cm |

- The anchors move toward the ground truth in all six runs. The anchor ATE includes the rectified camera's lever arm
  to the body, which rotates with the camera, so it reads higher than the KF ATE.
- With the window correction (VIO dense prior), the window moved 9–22 times per run and each move started a new
  submap (39–55 submaps against 30–38).
- On these easy sequences the mesh accuracy is set by the SGBM depth (median 6–17 cm). The drift is 3–6 cm, so the
  correction hardly shows (±0.5 cm). Next: V1_02, V1_03, V2_02 (drift up to 0.3 m without a prior).
- Cost: integration 284 ms per KF (stereo_tsdf, ~75 k points), marching cubes of the open submap 8–17 ms, no keyframe
  dropped. Meshes: 1.9–4.7 M faces in all. Submaps overlap where they see the same surface: the union carries one
  layer per submap, as in other submap systems. Fusing them is left to the map server.

Harder Vicon sequences (`SEQS="V1_02 V1_03 V2_02" tools/run_dense_submaps.sh`, same snapshot, so VIO no prior still
without the bias prior):

| run | submaps | anchor ATE odometry → corrected | KF ATE odometry → loop | mesh to scan, median | mesh within 10 cm |
|---|---|---|---|---|---|
| no prior V1_02 | 51 | 0.100 → 0.066 | 0.073 → 0.037 | 10.1 → 9.8 cm | 50 → 50 % |
| no prior V1_03 | 89 | 0.272 → 0.103 | 0.267 → 0.092 | 22.8 → 19.7 cm | 28 → 32 % |
| no prior V2_02 | 71 | 0.277 → 0.146 | 0.280 → 0.148 | 24.2 → 20.3 cm | 25 → 31 % |
| dense prior V1_02 | 67 | 0.077 → 0.059 | 0.059 → 0.030 | 9.1 → 8.5 cm | 52 → 54 % |
| dense prior V1_03 | 98 | 0.091 → 0.066 | 0.085 → 0.055 | 18.1 → 17.5 cm | 34 → 36 % |
| dense prior V2_02 | 91 | 0.072 → 0.067 | 0.059 → 0.051 | 17.0 → 16.7 cm | 36 → 37 % |

Where the odometry drifts by decimetres (no prior, V1_03 and V2_02), the anchors move by 0.13–0.17 m toward the
ground truth. The mesh improves too: median 3–4 cm closer to the scan, and 4–6 points more of it within 10 cm. Every
run improves, and the gain follows the drift that was removed. The SGBM depth (median ~9 cm at its best here) and the
stereo TSDF still dominate the mesh error. A sharper matcher (FFS, `doc/dense_ffs_stereo.md`) would show more of the
correction.

Not done: overlapping submaps are not fused into one surface (each keeps its layer), and a submap is placed by its
anchor only (a loop inside a submap's 10 KFs does not bend it). Both are map-server work in MeshCSLAM, which receives
the submaps and their anchor poses.

Next (agreed 2026-10-10, plan in `doc/dense_anchored_maps.md`):
1. Anchored meshes for every dense method: per-keyframe anchors for zncc, pd, per-keyframe gp and point clouds;
   submaps for the gp global map and vdbgpdf (gpdf, tsdf).
2. Optional fused global view (`dense_fused_view`): submap volumes merged into one, refreshed at map saves for the
   submaps whose correction changed, plus a full rebuild at the end.
3. Later, as an alternative: per-keyframe depth de-integrated and re-integrated (TSDF).

## 2026-10-10 — Anchored maps for every mesh method, fused view (`doc/dense_anchored_maps.md`)

Steps 1 and 2 of the plan are built (`AnchoredMaps`, options `dense_anchored`, `dense_submap_kfs`,
`dense_anchored_*`, `dense_fused_view`, `dense_fused_mesh_*`), and they replace the vdbgpdf-only submaps above.
`main` and the ROS node now stop cleanly on SIGTERM / SIGINT or ROS shutdown (`MarginalDepthInjector::finish()`), so
the final dense files are written. Before this, batch runs ended with `pkill` and no final dense file was ever written.

Tests (`mesh_test.cpp`):
- anchors placed at corrected poses;
- submaps start when full, at a new segment and at a new epoch;
- merge: one wall merged at an off-lattice pose lies on the wall; merged twice it is averaged, not layered; removed
  twice it leaves nothing;
- a synthetic room in two submaps (space carving, rotated cameras): the fused mesh lies on the room as closely as the
  placed submaps.

Suite 95 / 96 (`LineFeatureMatching`).

Bugs found on the way:
- The merge first used OpenVDB's voxel centres (`indexToWorld`). The vendored code puts them half a voxel further
  (`GetVoxelCenter`, marching cubes), so merged submaps were shifted by a rotated half-voxel (up to ~4 cm). Fixed.
- TSDF chunks of the GP methods' geometry took stereo points up to 20 m and grew to 60–78 M voxels of noise, and
  per-keyframe gp was killed (exit 137). Chunks are now capped at 8 m (as the `stereo_tsdf` preset) and thinned to
  2.5 cm.
- Weights count observations and reach hundreds, so float cancellation left voxels after a removal. A voxel under
  0.01 of one observation is now cleared.

Runs: `tools/run_dense_anchored.sh` (snapshot `dam3`), VIO no prior with loop closure, all keyframes kept, fused view
1. **FFS depth** (fp32 engine, as `doc/dense_vdb_gpdf.md`), V1_03 and V2_02, the Vicon rooms with laser scans and the
most drift. Labels `ffs_<method>`; baselines `ffsglob_<method>` are the global maps without anchors, i.e. the map as
it was before. `tools/dense_anchor_eval.py` measures the distance of the mesh vertices to the Leica scan, capped at
20 cm, and completeness (scan points within 10 cm of the mesh). Each mesh is aligned by an SE3 fit of the keyframe
trajectory: `results.csv` for odometry and global, `results_loop.csv` for corrected and fused. Raw output:
`results/dense_anchored_ffs.txt`.

Share of vertices within 5 cm / 10 cm of the scan (median distance):

| method | seq | anchors at odometry poses | anchors at corrected poses | fused view | global map (before) |
|---|---|---|---|---|---|
| zncc | V1_03 | 30 / 59 % (8.4 cm) | **37 / 68 % (6.9)** | 27 / 50 % (9.9) | |
| zncc | V2_02 | 44 / 70 % (5.9) | **51 / 76 % (4.9)** | 28 / 48 % (10.6) | |
| pd | V1_03 | 25 / 49 % (10.3) | **37 / 64 % (7.1)** | 16 / 30 % | |
| pd | V2_02 | 32 / 57 % (8.4) | **39 / 62 % (6.9)** | 16 / 28 % | |
| gp per keyframe | V1_03 | 18 / 34 % | **25 / 45 %** | 9 / 17 % | |
| gp per keyframe | V2_02 | 22 / 38 % | **23 / 40 %** | 9 / 17 % | |
| gp global map | V1_03 | 15 / 28 % | **19 / 33 %** | 11 / 20 % | 7 / 12 % |
| gp global map | V2_02 | 16 / 30 % | **20 / 34 %** | 10 / 20 % | 8 / 15 % |
| vdbgpdf tsdf | V1_03 | 29 / 52 % (9.4) | **38 / 67 % (6.6)** | 20 / 36 % | 14 / 28 % |
| vdbgpdf tsdf | V2_02 | 28 / 50 % (10.0) | **35 / 59 % (7.8)** | 17 / 28 % | 13 / 25 % |
| vdbgpdf gpdf | V1_03 | 24 / 45 % (11.4) | **36 / 64 % (7.2)** | 18 / 34 % | 18 / 35 % |
| vdbgpdf gpdf | V2_02 | 32 / 57 % (8.3) | **37 / 63 % (7.3)** | 19 / 34 % | 13 / 26 % |

- **Step 1 works for every method.** In all 12 runs the anchors at their loop-corrected poses are closer to the scan
  than at their odometry poses: 6–15 points more within 10 cm on V1_03 (the larger drift), 2–9 on V2_02. Completeness
  rises too (V1_03 tsdf 68 → 78 %).
- **The fused view is better than the global map it replaces, but less accurate than the anchored layers.** Fused vs
  global: tsdf 36 vs 28 % and 28 vs 25 % within 10 cm, gpdf 34 vs 35 % and 34 vs 26 %, gp map 20 vs 12 % and 20 vs
  15 %. It follows the loop closure, and the global map cannot.
  - Against the layers it loses about half of the surface within 5 cm, so averaging the volumes costs accuracy here.
  - Likely cause: the poses. VIO no prior is still at ~5–9 cm ATE after loops, close to the TSDF truncation (10 cm).
    Overlapping submaps then disagree by about their truncation band, so the average blurs, and one submap's space
    carving erodes another's surface.
  - The global map shows the same loss (median 15–20 cm against 6.6–7.8 cm for the layers). Not yet tested: a run
    with better poses (VIO dense or sparse prior, ~3–5 cm), where the fused view should come closer to the layers.
- For the per-keyframe methods, the fused view is a TSDF of their mesh vertices, a sparser input. It only gains
  completeness (zncc 85 % against 77 % on V2_02) at the cost of accuracy.
- With FFS, zncc at corrected anchors is the most accurate mesh here (V2_02: 4.9 cm median, 51 % within 5 cm).

Products:
- **For MeshCSLAM: the anchored meshes** (`dense_anchors/`, anchor PLYs plus `anchors.csv`). They are the most
  accurate and stay exact under later loops, and the map server fuses them with its own optimized poses.
- **For viewing and local use: the fused view.** One surface, better than the old global map, consistent with the
  loop closure.

The first pass was run by mistake with SGBM, the configs' default (`results/dense_anchored_sgbm_first.txt`, before the
half-voxel fix and the chunk cap). It showed the same ordering for odometry vs corrected anchors (zncc V1_03 43 → 57 %
within 10 cm), with every mesh 3–8 cm worse than with FFS.

### Fused view with better poses, and why it stays behind the layers (2026-10-10)

Same runs with VIO dense prior and sparse prior (FFS, VDB TSDF and VDB-GPDF, V1_03 / V2_02; labels `ffsdense_*`,
`ffssparse_*`). Loop-closed ATE is 3.2–6.3 cm, against ~9 cm without a prior. Raw output:
`results/dense_fused_checks.txt`.

| VDB TSDF, V1_03 | ATE loop | anchors at corrected poses, within 5 / 10 cm | fused view |
|---|---|---|---|
| no prior | ~0.09 | 38 / 67 % (6.6 cm) | 20 / 36 % |
| dense prior | 0.056 | **54 / 80 % (4.5 cm)** | 27 / 39 % |
| sparse prior | 0.046 | **53 / 78 % (4.7 cm)** | 26 / 39 % |

- With better poses the anchored layers improve a lot; the fused view barely does. So pose accuracy was not the main
  cause, and the explanation given above (poses close to the truncation band) is withdrawn.
- **Merge only the surface band** (`accumulateInto` skips voxels with |d| at the truncation, i.e. the free space
  written by space carving). This is kept, but it only raises completeness by 3–5 points (TSDF V2_02 68 → 71 %,
  VDB-GPDF 84 → 89 %); accuracy is unchanged. Labels `band_*`, snapshot `dam4`.
- **One submap for the whole sequence** (`dense_submap_kfs` 100000, `dbg_onesub`): the anchored mesh of that single
  volume is already as poor as the fused view (18.5 cm median against ~6 cm for 10-keyframe submaps), with no merge
  involved. Its resampled copy (the fused view of one volume) is a median 3 cm from it: this is the resampling blur.
- **Space carving off** (`stereo_tsdf` with `space_carving: false`, `dbg_onesub_nocarve`, `nocarve_tsdf`): more
  complete (+14 points) but less accurate (outlier fragments survive). Carving is not the cause; the experimental
  preset was removed.
- **Per surface area** (`dense_anchor_eval.py --area`, one vertex per 5 cm cell, so a surface repeated by
  overlapping anchors counts once): the scores hardly change. Vertex weighting is not the cause either.
- Where the poor surface is (`dbg_onesub`, fused): 49 % of the vertices are 20 cm or more from the scan.
  - 74 % of those lie more than 1 m from the camera path, against 8 % of the accurate ones.
  - So a long-lived volume builds up long-range stereo surfaces that a 10-keyframe submap does not.

**Conclusion.** Any long-horizon volume on this data has the same 14–20 cm median: the old global map, a single
whole-run submap, and the fused view. Short volumes have 4.5–8 cm. The cause is long-horizon TSDF fusion of FFS depth
here, not the anchored maps or their merge, which adds about 3 cm of resampling blur.
- Step 3 (one volume re-integrated at corrected poses) would inherit this, so it stays unbuilt until the
  long-horizon issue is understood.
- Next suspects, not tested: the 8 m range (try 4–5 m for long volumes), the reconstruction weight threshold
  (`recon_min_weight` 0.1: one ray suffices), no weight cap or decay, and FFS depth error that grows with range.
- The ExECoSim result in `doc/dense_vdb_gpdf.md` (4.6 cm, one volume) was simulated images with ground-truth poses.
- Products unchanged: anchored meshes for MeshCSLAM; the fused view (default `dense_fused_view: 1`) for viewing. The
  fused view is better than the old global map and follows the loop closure.

## 2026-10-10 — Final regression matrix (every shipped default, loop closure on)

`tools/run_final_matrix.sh` (snapshot `fin1`): six configurations × 18 sequences × 2 runs, labels `fin_<config>`.
Loop closure is on with bag-of-words, the automatic graph and window rule, the VIO bias prior 0.3 and Sim3 in mono.
The mono VO fallback was on in this matrix; it has since been reverted. Table: `results/final_matrix.md`
(`tools/final_summary.py`). No run crashed. Medians over the sequences (ATE m, window → loop-closed):

| configuration | window | loop-closed | earlier reference |
|---|---|---|---|
| VIO no prior | 0.137 | **0.049** | matrix: 0.169 → 0.050. V2_03 is now 0.39 → 0.19 (was 29.7 m: bias prior); magistrale2 4.7 → 1.3 (was 7.5 → 3.7) |
| VIO dense prior | 0.062 | **0.039** | window correction automatic (on) |
| VIO sparse prior | 0.111 | **0.033** | ~0.032 |
| stereo VO | 0.163 | **0.050** | V2_03: 37 resets, 0.27 m |
| mono VIO | 0.166 | **0.103** | ~0.12; MH_01/MH_02/V1_03 still 0.36–1.09 m |
| mono VO | 0.714 | 0.534 | fallback on in this matrix. Without it, pooled over 4–8 runs per sequence: 0.514 → **0.306** (`IMU_FIX_LEDGER.md`, correction of 2026-10-13); fallback off again |

The other configurations are as good as or better than before. The bias prior gives VIO no prior its first V2_03 run
without a failure.

## 2026-10-10 — ROS map interface (`ros_map_interface`, `doc/loop_closure/ros_map_interface.md`)

What the node publishes for a map server (MeshCSLAM), with standard messages, also when `enable_visu` is 0:
- TF `map -> world` (the loop closure's correction of the odometry frame) and `vo_pose_corrected`;
- `keyframes_corrected` (`nav_msgs/Path`, latched);
- `dense_anchor_mesh` (each closed anchor's mesh once, in its anchor frame, marker pose = anchor pose);
- `dense_anchor_poses` (`nav_msgs/Path`, latched, refreshed at every map save).

Supporting code:
- `LoopClosure::keyframes`, the `AnchoredMaps` outbox (off until a consumer enables it), `SLAMCore::segment`;
- the loop closure markers and the anchored `dense_mesh` are in `map` when the interface is on.

Live check (`run_sadvio_ros.py --record-interface`, `record_map_interface.py`): room1 ROS 2 bag, stereo VIO with the
dense prior, loop closure, zncc anchored meshes (SGBM: this checks the plumbing, not the meshes), `enable_visu: 0`.
Result: `results/ros_map_interface_room1.json`.
- 2703 frames, each with a corrected pose and a `map -> world`. The transform changed 323 times; it stays under 2 cm
  because the dense prior's window correction moves the odometry itself.
- `keyframes_corrected` was sent 654 times and ends with all 1228 keyframes.
- 1228 anchor meshes, each id once (329 k triangles).
- The last anchor pose list holds 1225 anchors, all matching a mesh. The last 3 closed after the last save; they are
  in `anchors.csv` only, since the final save happens as the process exits.
- After their meshes were sent, anchors moved by up to 8.4 cm (mean 1.7 cm) in the pose list, so re-placement is
  exercised.

Also fixed on the way:
- `run_sadvio_ros.py` now pins the node (`--cpus`, `--nice`), as the batch runner does, and waits up to 600 s for the
  node's final dense files.
- The recorder opts out of rclpy's signal handling, which swallowed SIGINT and SIGTERM, and writes its summary every
  second.

Follow-ups:
- With the dense prior, room1's loop closure moved the window 450 times, and every move starts a new submap. The
  fused methods then get submaps of a few keyframes (515 volumes in a first TSDF attempt). Better: on a move by C,
  re-anchor the open submap (`T_w_anchor := C T_w_anchor`) and continue it.
- Mono VO: the essential-matrix fallback as a scale break in the Sim3 graph (`IMU_FIX_LEDGER.md`).
- The open question of long-horizon TSDF fusion (fused view, above).

## 2026-10-13 — Anchored submaps continue across window corrections

A window correction moves the odometry world by a known C. The open submap no longer closes there:
- `MarginalDepthInjector::worldMoved(C)` passes C to `AnchoredMaps::worldMoved`, which keeps the cumulative transform
  of every epoch.
- A keyframe of a later epoch is put into the open submap through the anchor re-expressed in that epoch:
  `T_a_c = (move(epoch_anchor -> epoch_kf) T_w_anchor)^-1 T_w_c`. The fused view's TSDF chunks work the same way.
- Only a re-initialization (an unrelated world) still closes a submap.
- Placement does not change: the anchor keeps its hand-over pose, and `keyframeCorrection` is relative to it.

Test: `AnchoredMapsTest.submapsSpanWindowCorrectionsButNotReinitializations` checks that the relative pose across a
correction is exact and that a re-initialization still splits.

Runs (snapshot `sb1`, labels `reanc_*`), VIO dense prior, FFS:

| run | window moves | anchors before → now | corrected anchors, median per area, before → now |
|---|---|---|---|
| VDB TSDF V1_03 | 11 | 98 → 89 | 5.1 → 5.5 cm |
| VDB TSDF V2_02 | 26 | 90 → 72 | 6.8 → 7.1 cm |
| VDB-GPDF V1_03 | 14 | 98 → 91 | 6.4 → 6.1 cm |
| VDB-GPDF V2_02 | 29 | 97 → 71 | 6.5 → 6.8 cm |
| VDB TSDF room1 (SGBM, count only) | 438 | 515 → 122 | – |

- Submap counts are back to about one per `dense_submap_kfs` keyframes. Accuracy is unchanged within run-to-run noise
  (±0.4 cm; these are separate runs).
- The TSDF fused view on V2_02 gains completeness (65 → 72 %).
- For MeshCSLAM this means about 4× fewer anchor messages on room1-like runs.

**Mono VO numbers: correction (2026-10-13).** Mono VO varies hugely from run to run (MH_01: 0.25 m and 2.80 m with the
same binary). The Sim3 section's medians (0.438 → 0.255) come from one batch of 2 runs per sequence. Pooled over 4–8
runs per sequence, fallback off: odometry 0.514 → loop-closed 0.306 (medians of per-sequence medians). The paired
statement holds: within each run, the loop-closed trajectory beats its own odometry. For mono VO, compare
configurations with at least 4–6 runs per sequence.

## 2026-10-13 — Long-horizon TSDF fusion: cause found (range noise and constant weights)

Tools:
- `EXECO_DENSE_DUMP=<dir>` dumps every dense keyframe's points and poses (`MarginalDepthInjector::dumpKeyframe`).
- `cpp/tools/dense_replay.cpp` re-integrates a dump offline at chosen poses: odometry, loop-corrected, or EuRoC
  ground truth (analysis only, never in the SLAM). It builds one volume or chunks, layered or merged, with any TSDF
  setting. `tools/run_dense_replay.sh` runs it pinned in the container.
- `tools/dense_replay_eval.py` scores a whole mesh against the Leica scan.
- `tools/depth_error_by_range.py` gives the stereo error by range.
- Dumps: V1_03 and V2_02, VIO sparse prior, FFS, `stereo_tsdf` (label `dump_tsdf`).
- Results: `results/dense_long_horizon_replays.txt`.

**Scoring correction.** `dense_anchor_eval.py` samples the same number of vertices per anchor. That under-weights
large anchors (the far, noisy surface) and scored the anchored layers about 5 cm too well: 12.0 cm instead of 7.3 cm
on the same mesh. `dense_replay_eval.py` counts all of a mesh's surface once (5 cm cells), completeness is measured
on the scan within 4 m of the ground-truth path, and both are split by distance from the path. The comparisons of
layers vs fused view above that used `dense_anchor_eval.py` overstate the layers.

**Findings, V2_02** (one volume with ground-truth poses unless stated; median, within 5 / 10 cm, complete):
- **Not the odometry.** Ground-truth poses give the same as odometry for 10-keyframe chunks (12.0 vs 12.1 cm). One
  ground-truth volume for the whole run is still worse (17.2 cm, 55 %) than the chunks (12.0 cm, 76 %).
- **Not a stereo bias.** The signed error along the ray has a median within ±0.8 cm at every range. Its spread grows
  steeply with range: median |error| 1.8 cm at 1–2 m, 5.5 at 3–4 m, 10.5 at 5–6 m, 11 cm at 6–8 m. More than half the
  points are beyond 3 m.
- **Range is the lever.** A 4 m cap gives 4.0 cm, 57 / 78 %, but coverage of surfaces more than 3 m from the path
  falls from 29 % to 7 %. A cap is not acceptable: drones would have to fly closer, and coverage feeds MeshVLPR and
  MeshCSLAM.
- **Not space carving.** Without carving: 20 cm, more complete (73 %); with a 4 m cap, carving clearly helps (4.0 vs
  10.4 cm). Not the meshing weight threshold either (3 observations: 16.3 cm).
- **Truncation** 0.05 m is worse (18.4 cm, 35 %), 0.2 m slightly better (16.6 cm, 64 %).
- **Range weighting** (vendored TSDF `Integrate` takes per-point weights; `VDBGPDFMapConfig::range_weight_ref` and
  `range_weight_exp`, weight min(1, (ref / range)^exp), default off):
  - (2 m / r)² gives 16.7 cm;
  - (2 m / r)⁴, the 1/σ² of stereo, gives 12.0 cm, 31 / 46 %, 63 %;
  - with 0.2 m truncation: 7.1 cm, 42 / 57 %, 67 %, better than the default in every band (near 2.3 cm / 88 %
    against 5.0 / 79 %; 1.5–3 m 10.3 / 61 % against 20 / 47 %; beyond 3 m 19.1 / 40 % against 20 / 29 %).
  - Lowering the meshing threshold (0.01) brings the far junk back.

**Fused view and layers with the weighting** (loop-corrected SLAM poses, chunks of 10 keyframes):

| | V2_02 | V1_03 |
|---|---|---|
| fused view (merged chunks), default | 20 cm, 31 % within 10 cm, 60 % complete | (in-SLAM, earlier scorer: 14–16 cm) |
| fused view, weighted (2 m / r)⁴ + 0.2 m truncation | **6.4 cm, 59 %, 67 %** | **7.0 cm, 59 %, 62 %** |
| layers, default (ground truth) | 12.0 cm, 44 %, 76 % | 9.8 cm, 51 %, 71 % |
| layers, weighted (ground truth) | 10.5 cm, 48 %, 75 % | 10.0 cm, 50 %, 71 % |
| one volume, weighted: loop poses / ground truth | 6.9 / 7.1 cm | 7.5 / 8.7 cm |

- The weighting turns the fused view from the worst product into the most accurate, ahead of the layers.
- The layers keep more coverage (76 / 71 % against 67 / 62 %), and the most far surface (beyond 3 m from the path:
  64–73 % against 47–51 %).
- Far coverage under the weighting is mixed. V2_02 gains (29 → 40 % for one volume, 38 → 47 % for the fused view).
  V1_03's single volume loses it (56 → 41 %).
- SLAM poses do as well as ground truth.

Proposed, pending the user's decision: `stereo_tsdf` with `range_weight_ref: 2`, `range_weight_exp: 4`,
`sdf_trunc: 0.2`; MeshCSLAM keeps the anchored layers; check far coverage on a large scene (ExECoSim / RealSense).

V1_03 far coverage, the fused view (loop poses, merged 10-keyframe chunks) without and with the weighting:
- Overall: 14.7 cm, 42 % within 10 cm, 62 % complete → 7.0 cm, 59 %, 62 %.
- Near surfaces (≤ 1.5 m from the path): 4.4 cm / 83 % → 3.9 / 84 %.
- 1.5–3 m: 10.6 / 59 % → 6.0 / 59 %.
- Beyond 3 m: 20 cm / 63 % → 20 cm / 51 %.
- Total coverage is unchanged. The weighting drops 12 points of far surface that is at least 20 cm off the scan in
  both versions (median capped at 20 cm). On V2_02 far coverage rose instead (38 → 47 %).

**Decided (user, 2026-10-13): `stereo_tsdf` gets range weighting and `sdf_trunc: 0.2`; tuned below (2026-10-14) to `range_weight_ref: 1.5`, `range_weight_exp: 4`.**
- `stereo_tsdf_fast` (10 cm voxels) is unchanged: the weighting is untested at that voxel size.
- The fused view's TSDF chunks for the per-keyframe methods (zncc, pd, gp: `MarginalDepthInjector`'s `view_cfg`)
  are not weighted yet.
- Far coverage is still to be checked on a large scene (ExECoSim / RealSense); the Vicon rooms are small.

**Tuning (2026-10-14).** Fused view (loop poses, merged 10-keyframe chunks) on both dumps. Reference 1.5 / 2 / 3 m ×
exponent 3 / 4 / 6 at 0.2 m truncation, then truncation 0.15 / 0.3 m for the chosen pair (`tune_*` replays,
`results/dense_long_horizon_replays.txt`). Median / within 10 cm / complete / coverage beyond 3 m from the path:

| weighting | V2_02 | V1_03 |
|---|---|---|
| none | 20 cm / 31 % / 60 % / 38 % | 14.7 / 42 % / 62 % / 63 % |
| 3 m ^4 | 11.5 / 48 / 68 / 49 | 9.0 / 52 / 62 / 52 |
| 2 m ^4 (first choice) | 6.4 / 59 / 67 / 47 | 7.0 / 59 / 62 / 51 |
| 3 m ^6 | 6.5 / 58 / 68 / 49 | 7.3 / 57 / 62 / 51 |
| **1.5 m ^4 (chosen)** | **4.8 / 70 / 63 / 45** | **5.9 / 67 / 61 / 50** |
| 2 m ^6 | 4.1 / 76 / 60 / 40 | 5.6 / 71 / 61 / 49 |
| 1.5 m ^6 | 3.6 / 82 / 55 / 21 | 4.9 / 80 / 59 / 39 |
| 1.5 m ^4, truncation 0.15 m | 5.2 / 67 / 63 / 40 | 6.5 / 62 / 62 / 53 |
| 1.5 m ^4, truncation 0.3 m | 4.6 / 72 / 62 / 46 (near 3.4 cm) | 5.4 / 73 / 61 / 49 (near 4.2 cm) |

1.5 m ^4 with 0.2 m truncation is better than the first choice on both sequences: 1.1–1.6 cm more accurate, 8–11
points more within 10 cm. It costs 4 / 1 points of coverage and 1–2 of far coverage, and it fits the measured noise
(flat about 1.7 cm below ~1.5 m, steep beyond).
- Steeper settings trade coverage away: 1.5 m ^6 halves the far surface on V2_02.
- 0.3 m truncation helps the overall median but costs accuracy on near surfaces.

**`stereo_tsdf`: `range_weight_ref: 1.5`, `range_weight_exp: 4`, `sdf_trunc: 0.2`.**
