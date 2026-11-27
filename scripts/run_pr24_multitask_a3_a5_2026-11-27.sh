#!/usr/bin/env bash
set -euo pipefail

cd /output/PR24-MULTITASK-EXTREMA
export PYTHONPATH="$PWD/PatchTST_supervised"

mkdir -p logs/pr24_multitask

COMMON=(
  --seq_len 132
  --pred_len 24
  --patch_len 36
  --stride 18
  --d_model 128
  --n_heads 8
  --e_layers 2
  --d_ff 2048
  --dropout 0.2
  --fc_dropout 0.2
  --head_dropout 0.0
  --batch_size 32
  --learning_rate 0.0003
  --train_epochs 50
  --patience 100
  --random_seed 2021
  --use_gpu 1
  --mt_aux_weight 0.1
)

for spec in "peak_time:B26M-MT-PTIME-A3" "trough_time:B26M-MT-TTIME-A4" "all:B26M-MT-ALL-A5"; do
  mode="${spec%%:*}"
  des="${spec#*:}"
  for fold in 1 2 3 4; do
    log="logs/pr24_multitask/${des}_F${fold}.log"
    python3 -u run_cv5_multitask.py \
      "${COMMON[@]}" \
      --cv_fold "$fold" \
      --mt_mode "$mode" \
      --des "$des" \
      >>"$log" 2>&1
  done
  python3 scripts/eval_cv5.py \
    --des "$des" \
    --seed 2021 \
    --out_dir "analysis/pr24_multitask_extrema/${des}_seed2021"
done
