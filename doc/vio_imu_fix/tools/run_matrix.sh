#!/usr/bin/env bash
# Run all modes on the given sequences and evaluate.  Usage: run_matrix.sh <label> <runs> <seq...> [-- --set k=v ...]
set -u
cd "$(dirname "$0")/.."
label=$1; runs=$2; shift 2
seqs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do seqs+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
# Optional binary (e.g. a snapshot wrapper): BIN=... run_matrix.sh ...
BINARG=(); [ -n "${BIN:-}" ] && BINARG=(--bin "$BIN")
for seq in "${seqs[@]}"; do for mode in bimono bimonovio mono monovio; do
  python3 -u tools/run_sadvio.py --label "$label" --seq "$seq" --mode "$mode" --runs "$runs" "${BINARG[@]}" "$@"
  python3 tools/eval_traj.py runs/"$label"/"$seq"/"$mode"/run_* > /dev/null
done; done
python3 tools/eval_traj.py --summary runs/"$label" | tee runs/"$label"/summary.md
