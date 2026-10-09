#!/usr/bin/env bash
# magistrale2 stereo VIO with loop closure (bag-of-words, output corrected): 4-DoF against 6-DoF pose graph, and the
# configurations where VIO is good there (marginalization + td). Usage: run_lc_dof.sh <lane> <A|B|C|D ...>
lane=$1; shift
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/lc4/run.sh
for c in "$@"; do
  case $c in
    A) label=lc_dof_margtd_4;   extra="--set marginalization=1 --set estimate_td=1 --set loop_graph_dof=4" ;;
    B) label=lc_dof_margtd_6;   extra="--set marginalization=1 --set estimate_td=1 --set loop_graph_dof=6" ;;
    C) label=lc_dof_default_6;  extra="--set loop_graph_dof=6" ;;
    D) label=lc_dof_sadviotd_4; extra="--set marginalization=1 --set sparsification=1 --set estimate_td=1 --set loop_graph_dof=4" ;;
  esac
  python3 run_sadvio.py --label $label --seq magistrale2 --mode bimonovio --runs 2 --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 --set loop_correct_window=0 $extra > /dev/null 2>&1
  echo "done $c $label"
done
echo ALL_DONE
