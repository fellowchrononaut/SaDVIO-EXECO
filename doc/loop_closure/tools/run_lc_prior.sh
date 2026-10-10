#!/usr/bin/env bash
# Window correction with a marginalization prior: dropping the prior (binary lc5) against transforming it into the
# corrected world (lc6, Marginalization::transformWorld). Bag-of-words, loop_correct_window 1, automatic graph (6-DoF),
# 2 runs per cell. Usage: run_lc_prior.sh <lane> <part> <parts>
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
S=/root/SaDVIO-Dense/bin_snapshots
ROOMS="room1 room2 room3 room4 room5 room6"
EUROC="MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03"
jobs=()
for b in lc5:drop lc6:transform; do
  bin=${b%%:*}; name=${b#*:}
  for q in $ROOMS $EUROC; do
    jobs+=("lc_prior_${name}_sadvio|$q|$S/$bin/run.sh|--set marginalization=1 --set sparsification=1")
  done
  for q in $ROOMS magistrale2; do
    jobs+=("lc_prior_${name}_margtd|$q|$S/$bin/run.sh|--set marginalization=1 --set estimate_td=1")
  done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r label q bin extra <<< "${jobs[$i]}"
  python3 run_sadvio.py --label $label --seq $q --mode bimonovio --runs 2 --cpus $lane --nice 10 --bin $bin \
    --set loop_closure=1 --set loop_detector='"bow"' --set loop_correct_window=1 $extra > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $label $q"
done
echo ALL_DONE
