#!/usr/bin/env bash
# Final regression matrix with every shipped default (loop closure on, bag-of-words, automatic graph and window rule,
# VIO bias prior 0.3, mono VO essential-matrix fallback, Sim3 in mono): six configurations x 18 sequences x 2 runs.
# Labels fin_<config>. Usage: run_final_matrix.sh <lane> <part> <parts> (cores 8-15 or 16-23, nice 10)
# Regression subsets: PREFIX (label prefix, default fin) and SEQS (space-separated) from the environment
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/${SNAP:-fin1}/run.sh
SEQS="${SEQS:-room1 room2 room3 room4 room5 room6 magistrale2 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03}"
PREFIX="${PREFIX:-fin}"
declare -A C=(
  [vio_noprior]="bimonovio|--set marginalization=0"
  [vio_dense]="bimonovio|--set marginalization=1 --set sparsification=0"
  [vio_sparse]="bimonovio|--set marginalization=1 --set sparsification=1"
  [vo_stereo]="bimono|"
  [vio_mono]="monovio|"
  [vo_mono]="mono|"
)
jobs=()
for q in $SEQS; do
  for c in vio_noprior vio_dense vio_sparse vo_stereo vio_mono vo_mono; do jobs+=("$c|$q"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r c q <<< "${jobs[$i]}"
  IFS='|' read -r mode extra <<< "${C[$c]}"
  eval python3 run_sadvio.py --label ${PREFIX}_$c --seq $q --mode $mode --runs 2 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 $extra > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $c $q"
done
echo ALL_DONE
