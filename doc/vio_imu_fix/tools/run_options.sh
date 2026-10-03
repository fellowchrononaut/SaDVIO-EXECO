#!/usr/bin/env bash
# Option matrix on room1: one option changed at a time from the default, all modes, 1 run each.
# Usage: BIN=<wrapper> run_options.sh <label_prefix>
set -u
cd "$(dirname "$0")/.."
prefix=$1
declare -A V
V[default]=""
V[analytic]='--set optimizer="Analytic"'
V[numeric]='--set optimizer="Numeric"'
V[marg]='--set marginalization=1'
V[marg_sparse]='--set marginalization=1 --set sparsification=1'
V[analytic_marg_sparse]='--set optimizer="Analytic" --set marginalization=1 --set sparsification=1'
V[td]='--set estimate_td=1'
V[mt]='--set multithreading=1'
V[numeric_marg]='--set optimizer="Numeric" --set marginalization=1'
for name in default analytic numeric marg marg_sparse analytic_marg_sparse td mt numeric_marg; do
  eval "args=(${V[$name]})"
  for mode in bimono bimonovio mono monovio; do
    python3 -u tools/run_sadvio.py --label "${prefix}_$name" --seq room1 --mode "$mode" --runs 1 --bin "$BIN" --stall 30 "${args[@]}"
    python3 tools/eval_traj.py runs/"${prefix}_$name"/room1/"$mode"/run_* > /dev/null 2>&1
  done
done
