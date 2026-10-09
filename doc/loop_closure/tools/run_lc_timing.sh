#!/usr/bin/env bash
# Clean timing pass (task 3): one run at a time on cores 16-23, idle machine (the other sessions paused). Stereo VIO,
# output corrected, binary lc5. Variants: none (loop closure off), bow, gpu (MegaLoc TensorRT fp32), cpu (MegaLoc
# OpenVINO, 8 threads). Runs go to doc/vio_imu_fix/runs/lc_time_<variant>/.
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/lc5/run.sh
M=/Simulator_Validation/SaD_VIO_data/models
model() { case $1 in room*|magistrale2) echo 322x322 ;; *) echo 322x504 ;; esac; }
run() {  # label seq runs extra...
  local label=$1 q=$2 n=$3; shift 3
  python3 run_sadvio.py --label $label --seq $q --mode bimonovio --runs $n --cpus 16-23 --nice 10 --bin $BIN "$@" \
    > /dev/null 2>&1
  echo "done $label $q"
}
for q in room1 MH_01; do
  run lc_time_none $q 2 --set loop_closure=0
  run lc_time_bow $q 2 --set loop_closure=1 --set loop_detector='"bow"'
  run lc_time_gpu $q 2 --set loop_closure=1 --set loop_detector='"learned"' --set loop_model_device='"GPU"' \
    --set loop_model="\"$M/megaloc_$(model $q)_fp32.engine\""
  run lc_time_cpu $q 2 --set loop_closure=1 --set loop_detector='"learned"' --set loop_model_device='"CPU"' \
    --set loop_model="\"$M/megaloc_$(model $q).onnx\""
done
run lc_time_none magistrale2 1 --set loop_closure=0
run lc_time_bow magistrale2 1 --set loop_closure=1 --set loop_detector='"bow"'
echo ALL_DONE
