#!/usr/bin/env bash
# Re-run the estimate_td option variant (all modes, room1) with a given binary
set -u
cd "$(dirname "$0")/.."
for mode in bimono bimonovio mono monovio; do
  python3 -u tools/run_sadvio.py --label "$1" --seq room1 --mode "$mode" --runs 1 --bin "$BIN" --set estimate_td=1
  python3 tools/eval_traj.py runs/"$1"/room1/"$mode"/run_* > /dev/null 2>&1
done
