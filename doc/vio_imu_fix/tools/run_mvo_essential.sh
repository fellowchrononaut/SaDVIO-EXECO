#!/usr/bin/env bash
# Mono VO A/B: the essential-matrix fallback when PnP fails (mono_essential_fallback 1) against without (0), same
# binary (snapshot mvo3), TUM-VI rooms + EuRoC, 2 runs per cell. Usage: run_mvo_essential.sh <lane> <part> <parts>
# [snapshot] [values] [label suffix], e.g. "mvo4 1 b" reruns only the fallback with snapshot mvo4 as mvo_ess1b
lane=$1; part=$2; parts=$3; snap=${4:-mvo3}; vals=${5:-"0 1"}; suffix=${6:-}
cd "$(dirname "$0")"
jobs=()
for q in room1 room2 room3 room4 room5 room6 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do
  for v in $vals; do jobs+=("$q|$v"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r q v <<< "${jobs[$i]}"
  python3 run_sadvio.py --label mvo_ess$v$suffix --seq $q --mode mono --runs 2 --cpus $lane --nice 10 \
    --bin /root/SaDVIO-Dense/bin_snapshots/$snap/run.sh --set mono_essential_fallback=$v > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $q $v"
done
echo ALL_DONE
