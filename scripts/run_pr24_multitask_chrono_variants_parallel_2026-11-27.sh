#!/usr/bin/env bash
set -euo pipefail

cd /output/PR24-MULTITASK-EXTREMA
mkdir -p logs/pr24_multitask_chrono

for variant in unc w030; do
  for cycle in SC20 SC21 SC24; do
    for seed in 2019 2020 2021 2022 2023; do
      printf '%s %s %s\n' "$cycle" "$seed" "$variant"
    done
  done
done | xargs -n 3 -P 4 bash scripts/run_pr24_multitask_chrono_variant_job.sh

for spec in "unc:analysis/pr24_multitask_cycle_holdout_unc_v2" "w030:analysis/pr24_multitask_cycle_holdout_w030_v2"; do
  variant="${spec%%:*}"
  base="${spec#*:}"
  python3 scripts/summarize_cycle_holdout.py \
    --base "$base" \
    --methods p24_update,p24_recursive_multitask \
    --output-prefix "pr24_multitask_${variant}"
done
