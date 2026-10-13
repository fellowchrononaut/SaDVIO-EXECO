#!/usr/bin/env bash
# One dense_replay (cpp/tools/dense_replay.cpp) in the container, pinned: run_dense_replay.sh <lane> <seq> <name>
# <dense_replay options...>. Reads the dump of scratch/dense_dump/<seq> (EXECO_DENSE_DUMP), writes
# scratch/dense_replay/<seq>/<name>.ply. Poses: "gt" (EuRoC ground truth: analysis only), "loop" (the dump run's
# results_loop.csv, copied next to the replays) or "odometry".
lane=$1; seq=$2; name=$3; shift 3
S=/Simulator_Validation/SaD_VIO_data
declare -A DIR=([V1_01]=V1_01_easy [V1_02]=V1_02_medium [V1_03]=V1_03_difficult [V2_01]=V2_01_easy [V2_02]=V2_02_medium [V2_03]=V2_03_difficult)
args=()
for a in "$@"; do
  case $a in
    gt)   args+=("gt:$S/euroc/${DIR[$seq]}/mav0/state_groundtruth_estimate0/data.csv") ;;
    loop) args+=("loop:$S/scratch/dense_replay/$seq/results_loop.csv") ;;
    *)    args+=("$a") ;;
  esac
done
docker exec sad_vio_dense bash -c "LD_LIBRARY_PATH=/root/SaDVIO-Dense/bin_snapshots/${REPLAY_SNAP:-dump2} nice -n 10 taskset -c $lane \
  /root/SaDVIO-Dense/bin_snapshots/${REPLAY_SNAP:-dump2}/dense_replay $S/scratch/dense_dump/$seq $S/scratch/dense_replay/$seq/$name.ply \
  --preset /root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/ros/config/vdbgpdf/stereo_tsdf.yaml ${args[*]}"
