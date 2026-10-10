#!/usr/bin/env bash
# Stereo VIO without a prior (default config): a weak zero-mean prior on the accelerometer bias (vio_bias_prior_acc,
# m/s^2), snapshot vbp1, TUM-VI rooms + magistrale2 + EuRoC, 2 runs per cell; the reference without it is vab0 (same
# code path, snapshot vab1). Usage: run_vio_bias_prior.sh <lane> <part> <parts>
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")"
jobs=()
for q in room1 room2 room3 room4 room5 room6 magistrale2 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do
  for v in 0.3 0.1; do jobs+=("$q|$v"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r q v <<< "${jobs[$i]}"
  python3 run_sadvio.py --label vbp$v --seq $q --mode bimonovio --runs 2 --cpus $lane --nice 10 \
    --bin /root/SaDVIO-Dense/bin_snapshots/vbp1/run.sh --set vio_bias_prior_acc=$v > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $q $v"
done
echo ALL_DONE
