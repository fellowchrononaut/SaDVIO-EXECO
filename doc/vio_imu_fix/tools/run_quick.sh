#!/usr/bin/env bash
# Quick per-fix check with the development binary: room1, VIO modes, 3 runs each.
# Usage: run_quick.sh <label> [modes...] [-- --set k=v ...]
set -u
cd "$(dirname "$0")/.."
label=$1; shift
modes=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do modes+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
[ ${#modes[@]} -eq 0 ] && modes=(bimonovio monovio)
# Snapshot the development binary and library so later rebuilds cannot leak into this evaluation
BUILD=/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/cpp/build_tests
SNAP=/root/SaDVIO-Dense/bin_snapshots/$label
docker exec sad_vio_dense bash -c "rm -rf $SNAP && mkdir -p $SNAP && cp $BUILD/isaeslam $BUILD/libisae_slam.so $SNAP/"
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%s\nexec %s/isaeslam "$@"\n' "$SNAP" "$SNAP" \
  | docker exec -i sad_vio_dense bash -c "cat > $SNAP/run.sh && chmod +x $SNAP/run.sh"
docker exec sad_vio_dense bash -c "LD_LIBRARY_PATH=$SNAP ldd $SNAP/isaeslam | grep isae_slam"
DEV=$SNAP/run.sh
for mode in "${modes[@]}"; do
  python3 -u tools/run_sadvio.py --label "$label" --seq room1 --mode "$mode" --runs 3 --bin "$DEV" "$@"
  python3 tools/eval_traj.py runs/"$label"/room1/"$mode"/run_* > /dev/null
done
python3 tools/eval_traj.py --summary runs/"$label" | tee runs/"$label"/summary.md
