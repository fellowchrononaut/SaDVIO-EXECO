#!/usr/bin/env bash
# Window correction with the prior transformed and the IMU window moved by yaw + translation only (binary lc7), on the
# worst cells of run_lc_prior.sh. Usage: run_lc_prior_check.sh <lane> <cells...> (cell = config:seq)
lane=$1; shift
cd "$(dirname "$0")/../../vio_imu_fix/tools"
for c in "$@"; do
  cfg=${c%%:*}; q=${c#*:}
  case $cfg in
    sadvio) extra="--set marginalization=1 --set sparsification=1" ;;
    margtd) extra="--set marginalization=1 --set estimate_td=1" ;;
  esac
  python3 run_sadvio.py --label lc_prior_grav_$cfg --seq $q --mode bimonovio --runs 2 --cpus $lane --nice 10 \
    --bin /root/SaDVIO-Dense/bin_snapshots/lc7/run.sh --set loop_closure=1 --set loop_detector='"bow"' \
    --set loop_correct_window=1 $extra > /dev/null 2>&1
  echo "done $c"
done
echo ALL_DONE
