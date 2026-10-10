#!/usr/bin/env bash
# Dense submaps placed by the loop closure (dense_submap_kfs): SGBM depth, VDB TSDF (stereo_tsdf), submaps of 10 KFs,
# loop closure on. VIO no prior (output correction) and VIO dense prior (window correction: submaps restart when the
# window moves), EuRoC (SEQS to choose others). Usage: [SEQS="..."] run_dense_submaps.sh <lane> <part> <parts>
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/dsm1/run.sh
DENSE="--set dense_depth=true --set dense_mesh_method=\"vdbgpdf\" --set dense_vdbgpdf_preset=\"stereo_tsdf\" \
--set dense_keep_all_keyframes=true --set dense_submap_kfs=10 --set loop_closure=1"
jobs=()
for q in ${SEQS:-V1_01 V2_01 MH_01}; do
  jobs+=("dsm_noprior|$q|--set marginalization=0")
  jobs+=("dsm_dense|$q|--set marginalization=1 --set sparsification=0")
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r label q extra <<< "${jobs[$i]}"
  eval python3 run_sadvio.py --label $label --seq $q --mode bimonovio --runs 1 --cpus $lane --nice 10 --bin $BIN \
    $DENSE $extra > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $label $q"
done
echo ALL_DONE
