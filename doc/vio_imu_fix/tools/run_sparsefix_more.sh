#!/usr/bin/env bash
# More before/after pairs for the two cells that looked worse (same lanes and priorities as run_sparsefix.sh)
cd "$(dirname "$0")"
S=/root/SaDVIO-Dense/bin_snapshots
i=0
for c in "sadvo_more|--seq V1_03 --mode bimono --set marginalization=1 --set sparsification=1" \
         "monoviomargsp_more|--seq room1 --mode monovio --set marginalization=1 --set sparsification=1"; do
  name=${c%%|*}; args=${c#*|}
  for k in 1 2; do
    if [ $((i % 2)) -eq 0 ]; then cb=8-15; cf=16-23; else cb=16-23; cf=8-15; fi
    python3 run_sadvio.py --label sp2_base_${name}$k $args --runs 2 --bin $S/base_e035/run.sh --cpus $cb --nice 10 > /dev/null 2>&1 &
    python3 run_sadvio.py --label sp2_fix_${name}$k $args --runs 2 --bin $S/sparsefix2/run.sh --cpus $cf --nice 10 > /dev/null 2>&1 &
    wait
    echo "done $name $k"
    i=$((i + 1))
  done
done
echo ALL_DONE
