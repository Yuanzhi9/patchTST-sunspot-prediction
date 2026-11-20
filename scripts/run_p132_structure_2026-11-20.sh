#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/PatchTST_supervised"
export OMP_NUM_THREADS=4

run_config() {
  local des="$1"
  local seq_len="$2"
  local patch_len="$3"
  local stride="$4"
  local fold

  for fold in 1 2 3 4 5; do
    echo "[$(date '+%F %T')] ${des} F${fold} seq=${seq_len} patch=${patch_len}/${stride}"
    python3 -u run_base_cv5.py \
      --cv_file configs/cv5_folds_p132_2026-10.json \
      --cv_fold "${fold}" \
      --seq_len "${seq_len}" \
      --pred_len 132 \
      --patch_len "${patch_len}" \
      --stride "${stride}" \
      --d_model 128 \
      --n_heads 8 \
      --e_layers 2 \
      --d_ff 2048 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --train_epochs 50 \
      --patience 100 \
      --batch_size 32 \
      --learning_rate 0.0003 \
      --random_seed 2021 \
      --des "${des}" \
      --use_gpu 1 2>&1 | tee "logs/${des}-F${fold}.log"
  done
}

run_config B26M-P132-S132-P16 132 16 8
run_config B26M-P132-S132-P24 132 24 12
run_config B26M-P132-S132-P36 132 36 18
run_config B26M-P132-S336-P16 336 16 8
run_config B26M-P132-S336-P24 336 24 12
run_config B26M-P132-S336-P36 336 36 18
