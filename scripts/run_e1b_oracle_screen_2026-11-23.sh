#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/PatchTST_supervised"
export OMP_NUM_THREADS=4

declare -A CONFIG=(
  [SC20]="configs/chrono_cycle20_holdout_2026-11.json"
  [SC21]="configs/chrono_cycle21_holdout_2026-11.json"
  [SC24]="configs/chrono_cycle24_holdout_2026-11.json"
)

for cycle in SC20 SC21 SC24; do
  for mode in none amp time both; do
    des="B26M-E1B-${cycle}-${mode}-S2021"
    echo "[$(date '+%F %T')] ${des}"
    python3 -u run_oracle_cond.py \
      --cycle "$cycle" \
      --cond_mode "$mode" \
      --cv_file "${CONFIG[$cycle]}" \
      --cv_fold 1 \
      --seq_len 132 \
      --label_len 48 \
      --pred_len 24 \
      --patch_len 36 \
      --stride 18 \
      --d_model 128 \
      --n_heads 8 \
      --e_layers 2 \
      --d_ff 2048 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0.0 \
      --train_epochs 50 \
      --patience 100 \
      --batch_size 32 \
      --learning_rate 3e-4 \
      --random_seed 2021 \
      --des "$des" \
      > "logs/${des}.log" 2>&1
  done
done

python3 -u scripts/summarize_e1b_oracle.py 2>&1 | tee logs/e1b_oracle_screen_summary.log
printf 'E1b seed2021 screening complete\n'
