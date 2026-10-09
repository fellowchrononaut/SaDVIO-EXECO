#!/usr/bin/env bash
# Loop closure tuning: detectors and inlier thresholds on a few sequences, stereo VIO (default config), one run at a
# time on cores 16-23 at nice 10 (the other lane may be busy). Runs go to doc/vio_imu_fix/runs/lc_tune_<variant>/.
cd "$(dirname "$0")/../../vio_imu_fix/tools"
for q in room1 room4 MH_01 V1_02; do
  for v in "bow25|loop_detector='\"bow\"' loop_min_inliers=25" \
           "bow15|loop_detector='\"bow\"' loop_min_inliers=15" \
           "bow12|loop_detector='\"bow\"' loop_min_inliers=12" \
           "prox15|loop_detector='\"proximity\"' loop_min_inliers=15" \
           "proxbow15|loop_detector='\"proximity_bow\"' loop_min_inliers=15"; do
    name=${v%%|*}; sets=""
    for kv in ${v#*|}; do sets="$sets --set $(eval echo $kv)"; done
    python3 run_sadvio.py --label lc_tune_$name --seq $q --mode bimonovio --runs 1 --cpus 16-23 --nice 10 \
      --set loop_closure=1 $sets > /dev/null 2>&1
    echo "done $q $name"
  done
done
echo ALL_DONE
