#!/usr/bin/env bash
# Pose graph DoF rule (loop_graph_dof) on TUM-VI rooms + EuRoC: 6-DoF runs to pair with the matrix's 4-DoF ones
# (lc_opt1_*), plus SaDVIO 4-DoF on the rooms (the matrix had SaDVIO on EuRoC only). Option 1 (bag-of-words, output
# corrected), 2 runs per cell. Usage: run_lc_dof_all.sh <lane> <part> <parts>
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/lc4/run.sh
ROOMS="room1 room2 room3 room4 room5 room6"
EUROC="MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03"
SAD="--set marginalization=1 --set sparsification=1"
jobs=()
for q in $ROOMS $EUROC; do
  jobs+=("lc_dof6_bimonovio|$q|bimonovio|--set loop_graph_dof=6")
  jobs+=("lc_dof6_monovio|$q|monovio|--set loop_graph_dof=6")
  jobs+=("lc_dof6_sadvio|$q|bimonovio|$SAD --set loop_graph_dof=6")
done
for q in $ROOMS; do jobs+=("lc_opt1_sadvio|$q|bimonovio|$SAD --set loop_graph_dof=4"); done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r label q m extra <<< "${jobs[$i]}"
  python3 run_sadvio.py --label $label --seq $q --mode $m --runs 2 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 --set loop_detector='"bow"' --set loop_correct_window=0 $extra > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $label $q $m"
done
echo ALL_DONE
