#!/usr/bin/env bash
# Anchored dense maps for every mesh method (doc/dense_anchored_maps.md): dense_anchored on, fused view 1, loop
# closure on, SGBM depth, all keyframes kept; VIO no prior on the Vicon rooms with laser scans and the most drift.
# Labels <PREFIX><method> (default dam_). Usage: [SEQS="..."] [METHODS="..."] [SNAP=dam1] [PREFIX=dam_]
# [ANCHORED=false: the global maps, as a baseline] [DEPTH=ffs: Fast-FoundationStereo fp32 engine instead of SGBM, as in
# doc/dense_vdb_gpdf.md and the MeshCSLAM drone path] [PRIOR=none|dense|sparse: VIO without a prior (default), with the
# dense prior, or with the sparse prior (the SaDVIO paper config)]
# run_dense_anchored.sh <lane> <part> <parts> (cores 8-15 or 16-23, nice 10)
lane=$1; part=$2; parts=$3
cd "$(dirname "$0")/../../vio_imu_fix/tools"
BIN=/root/SaDVIO-Dense/bin_snapshots/${SNAP:-dam1}/run.sh
FFS="--set stereo_depth_matcher=\"ffs\" \
--set stereo_depth_ffs_engine=\"/root/SaDVIO-Dense/ffs/23_36_37_480x864_it8/fast_foundationstereo_fp32.engine\""
[ "${DEPTH:-sgbm}" = ffs ] || FFS=""
BASE="--set dense_depth=true --set dense_keep_all_keyframes=true --set dense_anchored=${ANCHORED:-true} --set dense_fused_view=1 \
--set loop_closure=1"
case ${PRIOR:-none} in
  none)   BASE="$BASE --set marginalization=0" ;;
  dense)  BASE="$BASE --set marginalization=1 --set sparsification=0" ;;
  sparse) BASE="$BASE --set marginalization=1 --set sparsification=1" ;;
esac
declare -A M=(
  [zncc]="--set dense_mesh_method=\"zncc\""
  [pd]="--set dense_mesh_method=\"pd\""
  [gpkf]="--set dense_mesh_method=\"gp\" --set dense_gp_global_map=false"
  [gpmap]="--set dense_mesh_method=\"gp\" --set dense_gp_global_map=true"
  [tsdf]="--set dense_mesh_method=\"vdbgpdf\" --set dense_vdbgpdf_preset=\"stereo_tsdf\""
  [gpdf]="--set dense_mesh_method=\"vdbgpdf\" --set dense_vdbgpdf_preset=\"stereo\""
)
jobs=()
for q in ${SEQS:-V1_03 V2_02}; do
  for m in ${METHODS:-zncc pd gpkf gpmap tsdf gpdf}; do jobs+=("$m|$q"); done
done
for ((i = part; i < ${#jobs[@]}; i += parts)); do
  IFS='|' read -r m q <<< "${jobs[$i]}"
  eval python3 run_sadvio.py --label ${PREFIX:-dam_}$m --seq $q --mode bimonovio --runs 1 --cpus $lane --nice 10 --bin $BIN \
    --stall 60 $BASE $FFS ${M[$m]} > /dev/null 2>&1
  echo "done $i/${#jobs[@]} $m $q"
done
echo ALL_DONE
