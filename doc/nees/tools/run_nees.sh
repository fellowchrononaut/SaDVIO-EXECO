#!/usr/bin/env bash
# NEES runs: stereo VIO (bimonovio) with the dense marginalization prior (best window ATE in the final matrix),
# loop closure on, pose covariance logging on. EuRoC (11) + TUM-VI rooms 1-6 (magistrale2 has GT only at its ends).
# Usage: run_nees.sh <lane cpus> <part> <parts> [label]   (cores 8-15 or 16-23, nice 10)
lane=$1; part=$2; parts=$3; label=${4:-nees_vio_dense}
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/${SNAP:-nees1}/run.sh
SEQS=(MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03 room1 room2 room3 room4 room5 room6)
for ((i = part; i < ${#SEQS[@]}; i += parts)); do
  q=${SEQS[$i]}
  python3 run_sadvio.py --label $label --seq $q --mode bimonovio --runs 1 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 --set marginalization=1 --set sparsification=0 --set log_pose_covariance=1 > /dev/null 2>&1
  echo "done $q"
done
echo ALL_DONE
