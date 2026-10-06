#!/usr/bin/env bash
# Sparse-prior configurations, before (e035ccc) and after (sparsefix2: factor-coordinate sparsification), run with
# care for the shared machine: two lanes, each pinned to its own 8 cores at nice 10 (cores 0-7 stay free). For every
# cell, the before and after runs go at the same time, one per lane, and the lanes swap from one cell to the next,
# so both builds see the same conditions.
cd "$(dirname "$0")"
S=/root/SaDVIO-Dense/bin_snapshots
cells=()
for q in MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do
  cells+=("sadvio|--seq $q --mode bimonovio --set marginalization=1 --set sparsification=1")
  cells+=("sadvo|--seq $q --mode bimono --set marginalization=1 --set sparsification=1")
done
cells+=("margsparsetd|--seq room1 --mode bimonovio --set estimate_td=1 --set marginalization=1 --set sparsification=1")
cells+=("vomargsp|--seq room1 --mode bimono --set marginalization=1 --set sparsification=1")
cells+=("monoviomargsp|--seq room1 --mode monovio --set marginalization=1 --set sparsification=1")
i=0
for c in "${cells[@]}"; do
  name=${c%%|*}; args=${c#*|}
  if [ $((i % 2)) -eq 0 ]; then cb=8-15; cf=16-23; else cb=16-23; cf=8-15; fi
  python3 run_sadvio.py --label sp2_base_$name $args --runs 2 --bin $S/base_e035/run.sh --cpus $cb --nice 10 > /dev/null 2>&1 &
  python3 run_sadvio.py --label sp2_fix_$name $args --runs 2 --bin $S/sparsefix2/run.sh --cpus $cf --nice 10 > /dev/null 2>&1 &
  wait
  echo "done $((i + 1))/${#cells[@]}: $name $args"
  i=$((i + 1))
done
echo ALL_DONE
