#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/PatchTST_supervised"
export OMP_NUM_THREADS=4

run_config() {
  local des="$1"
  local lr="$2"
  local bs="$3"
  local seed="$4"
  local fold

  for fold in 1 2 3 4 5; do
    echo "[$(date '+%F %T')] ${des} F${fold} lr=${lr} bs=${bs} seed=${seed}"
    python3 -u run_base_cv5.py \
      --cv_file configs/cv5_folds_2026-10.json \
      --cv_fold "${fold}" \
      --seq_len 132 \
      --pred_len 24 \
      --patch_len 36 \
      --stride 18 \
      --d_model 128 \
      --n_heads 8 \
      --e_layers 2 \
      --d_ff 2048 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --train_epochs 50 \
      --patience 100 \
      --batch_size "${bs}" \
      --learning_rate "${lr}" \
      --random_seed "${seed}" \
      --des "${des}" \
      --use_gpu 1 2>&1 | tee "logs/${des}-F${fold}.log"
  done
}

run_config B26M-LV2-P36-1E3-BS32-SEED2019 0.001 32 2019
run_config B26M-LV2-P36-1E3-BS32-SEED2020 0.001 32 2020
run_config B26M-LV2-P36-1E3-BS32-SEED2021 0.001 32 2021
run_config B26M-F-P36-SEED2022 0.0003 32 2022
run_config B26M-F-P36-SEED2023 0.0003 32 2023
