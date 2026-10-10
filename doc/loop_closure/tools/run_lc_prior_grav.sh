#!/usr/bin/env bash
# Full comparison for the gravity-consistent window correction with the prior transformed (binary lc7): the cells of
# run_lc_prior.sh. Usage: run_lc_prior_grav.sh <lane> <part> <parts>
lane=$1; part=$2; parts=$3
T="$(dirname "$0")"
cells=()
for q in room1 room2 room3 room4 room5 room6 MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02 V2_03; do cells+=("sadvio:$q"); done
for q in room1 room2 room3 room4 room5 room6 magistrale2; do cells+=("margtd:$q"); done
sel=()
for ((i = part; i < ${#cells[@]}; i += parts)); do
  # the quick check already has these
  case ${cells[$i]} in margtd:room2|margtd:room4|margtd:magistrale2|sadvio:room2|sadvio:MH_01|sadvio:V2_03) continue ;; esac
  sel+=("${cells[$i]}")
done
"$T/run_lc_prior_check.sh" "$lane" "${sel[@]}"
