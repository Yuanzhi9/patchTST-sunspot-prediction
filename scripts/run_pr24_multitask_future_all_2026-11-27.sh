#!/usr/bin/env bash
set -euo pipefail

cd /output/PR24-MULTITASK-EXTREMA
export PYTHONPATH="$PWD/PatchTST_supervised"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs/pr24_multitask_future

for track in raw smooth13; do
  for seed in 2019 2020 2021 2022 2023; do
    printf '%s %s\n' "$track" "$seed"
  done
done | xargs -n 2 -P 4 bash -c '
  track="$1"
  seed="$2"
  cd /output/PR24-MULTITASK-EXTREMA
  PYTHONPATH="$PWD/PatchTST_supervised" python3 -u \
    scripts/train_forecast_pr24_multitask_final.py \
    --track "$track" \
    --seed "$seed" \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --horizon 168 \
    --block_size 24 \
    >>"logs/pr24_multitask_future/${track}_seed${seed}.log" 2>&1
' _
