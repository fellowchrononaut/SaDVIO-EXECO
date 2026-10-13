#!/usr/bin/env bash
# Mono VO with Sim3 loop closure: the essential-matrix fallback flagged as scale breaks (mono_essential_fallback 1,
# labels sb1_vo_mono) against no fallback (0, sb0_vo_mono), same binary (snapshot sb1), shipped defaults otherwise,
# 18 sequences x 2 runs. Usage: run_mono_scale_break.sh <lane> <part> <parts> (cores 8-15 or 16-23, nice 10)
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/${SNAP:-sb1}/run.sh
jobs=()
for q in room1 room2 room3 room4 room5 room6 magistrale2 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do
  for v in 1 0; do jobs+=("$q|$v"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r q v <<< "${jobs[$i]}"
  python3 run_sadvio.py --label sb${v}_vo_mono --seq $q --mode mono --runs 2 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 --set mono_essential_fallback=$v > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $q $v"
done
echo ALL_DONE
