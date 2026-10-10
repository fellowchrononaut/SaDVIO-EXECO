#!/usr/bin/env bash
# Mono VO loop closure: Sim3 pose graph (loop_graph_dof 7, the automatic choice in mono) against a 6-DoF graph (the
# loops' scale ratio ignored), bow detector, output correction. The odometry (results.csv) of the same runs is the
# reference without loop closure. Usage: run_lc_mono.sh <lane> <part> <parts> (cores 8-15 or 16-23, nice 10)
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/lcm1/run.sh
jobs=()
for q in room1 room2 room3 room4 room5 room6 magistrale2 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do
  for d in 7 6; do jobs+=("$q|$d"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r q d <<< "${jobs[$i]}"
  python3 run_sadvio.py --label lc_mono$d --seq $q --mode mono --runs 2 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 --set loop_detector=\"bow\" --set loop_graph_dof=$d > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $q $d"
done
echo ALL_DONE
