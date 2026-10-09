#!/usr/bin/env bash
# Loop closure evaluation matrix. Each run gives the window's trajectory (results.csv) and the loop-closed ones
# (results_loop*.csv). Usage: run_lc_matrix.sh <variants> <lane> <part> <parts>: runs every <parts>-th job starting at
# <part>, pinned to cores <lane> (8-15 or 16-23) at nice 10. Variants (comma-separated):
#   opt1     bow detector, output corrected only (window untouched): option 1
#   opt2     learned detector (MegaLoc, TensorRT fp32 on the GPU) and the window corrected: option 2
#   bowwin   bow detector and the window corrected (separates the detector from the correction)
#   cpu      learned detector on the CPU (OpenVINO, 8 threads), window corrected: cost of option 2 without a GPU
#            (a subset of sequences)
variants=$1; lane=$2; part=$3; parts=$4
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/lc3/run.sh
M=/Simulator_Validation/SaD_VIO_data/models
SEQS="room1 room2 room3 room4 room5 room6 magistrale2 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03"
EUROC="MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03"
model() { case $1 in room*|magistrale2) echo 322x322 ;; *) echo 322x504 ;; esac; }
jobs=()
for v in ${variants//,/ }; do
  case $v in
    opt1)   base="--set loop_detector=\"bow\" --set loop_correct_window=0"; runs=2; seqs="$SEQS" ;;
    bowwin) base="--set loop_detector=\"bow\" --set loop_correct_window=1"; runs=1; seqs="$SEQS" ;;
    opt2)   base="--set loop_detector=\"learned\" --set loop_model_device=\"GPU\" --set loop_correct_window=1"; runs=2; seqs="$SEQS" ;;
    cpu)    base="--set loop_detector=\"learned\" --set loop_model_device=\"CPU\" --set loop_correct_window=1"; runs=1; seqs="room1 room4 MH_01 V1_02 V2_02" ;;
  esac
  for q in $seqs; do
    case $v in opt2) mdl="--set loop_model=\"$M/megaloc_$(model $q)_fp32.engine\"" ;; cpu) mdl="--set loop_model=\"$M/megaloc_$(model $q).onnx\"" ;; *) mdl="" ;; esac
    for m in bimonovio bimono monovio; do jobs+=("lc_${v}_$m|$q|$m|$runs|$base $mdl"); done
    case " $EUROC " in *" $q "*) jobs+=("lc_${v}_sadvio|$q|bimonovio|$runs|$base $mdl --set marginalization=1 --set sparsification=1") ;; esac
  done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r label q m runs extra <<< "${jobs[$i]}"
  eval python3 run_sadvio.py --label $label --seq $q --mode $m --runs $runs --cpus $lane --nice 10 --bin $BIN \
    --set loop_closure=1 $extra > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $label $q $m"
done
echo ALL_DONE
