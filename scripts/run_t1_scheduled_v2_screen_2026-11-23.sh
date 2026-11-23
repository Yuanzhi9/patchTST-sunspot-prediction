#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/PatchTST_supervised:$ROOT/scripts"
export OMP_NUM_THREADS=4

declare -A CONFIG=(
  [SC20]="configs/chrono_cycle20_holdout_2026-11.json"
  [SC21]="configs/chrono_cycle21_holdout_2026-11.json"
  [SC24]="configs/chrono_cycle24_holdout_2026-11.json"
)

for cycle in SC20 SC21 SC24; do
  for mode in teacher scheduled; do
    des="B26M-T1V2-${cycle}-${mode}-S2021"
    metrics="analysis/t1_scheduled_sampling_v2/screen/${cycle}/${mode}/seed2021/metrics.json"
    if [[ -f "$metrics" ]]; then
      echo "[$(date '+%F %T')] skip existing ${des}"
      continue
    fi
    echo "[$(date '+%F %T')] ${des}"
    python3 -u run_scheduled_sampling.py \
      --cycle "$cycle" \
      --mode "$mode" \
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
      --scheduled_warmup_epochs 10 \
      --min_checkpoint_epoch 25 \
      --scheduled_max_prob 0.3 \
      --second_block_weight 0.5 \
      --random_seed 2021 \
      --des "$des" \
      --analysis_root ./analysis/t1_scheduled_sampling_v2 \
      > "logs/${des}.log" 2>&1
  done
done

python3 -u scripts/summarize_t1_scheduled.py \
  > logs/t1_scheduled_screen_summary.log 2>&1
printf 'T1 seed2021 screening complete\n'
