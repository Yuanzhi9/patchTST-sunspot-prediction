#!/usr/bin/env bash
set -euo pipefail

cd /output/PR24-MULTITASK-EXTREMA
mkdir -p logs/pr24_multitask_chrono

for cycle in SC20 SC21 SC24; do
  for seed in 2019 2020 2021 2022 2023; do
    printf '%s %s\n' "$cycle" "$seed"
  done
done | xargs -n 2 -P 4 bash scripts/run_pr24_multitask_chrono_job.sh

python3 scripts/summarize_cycle_holdout.py \
  --base analysis/pr24_multitask_cycle_holdout_v2 \
  --methods p24_update,p24_recursive_multitask \
  --output-prefix pr24_multitask_v2
