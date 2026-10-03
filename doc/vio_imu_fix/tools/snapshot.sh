#!/usr/bin/env bash
# Snapshot the development binary + library under a label; prints the wrapper path to use with --bin / BIN=
set -eu
label=$1
BUILD=/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel/cpp/build_tests
SNAP=/root/SaDVIO-Dense/bin_snapshots/$label
docker exec sad_vio_dense bash -c "rm -rf $SNAP && mkdir -p $SNAP && cp $BUILD/isaeslam $BUILD/libisae_slam.so $SNAP/"
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%s\nexec %s/isaeslam "$@"\n' "$SNAP" "$SNAP" \
  | docker exec -i sad_vio_dense bash -c "cat > $SNAP/run.sh && chmod +x $SNAP/run.sh"
echo "$SNAP/run.sh"
