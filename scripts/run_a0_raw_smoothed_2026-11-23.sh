#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/PatchTST_supervised"
export OMP_NUM_THREADS=4

run_config() {
  local des="$1"
  local data_path="$2"
  local cv_file="$3"
  local seed
  local fold

  for seed in ${SEEDS:-2019 2020 2021 2022 2023}; do
    for fold in 1 2 3 4 5; do
      echo "[$(date '+%F %T')] ${des} seed=${seed} F${fold} data=${data_path}"
      python3 -u run_base_cv5.py \
        --cv_file "${cv_file}" \
        --cv_fold "${fold}" \
        --data_path "${data_path}" \
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
        --batch_size 32 \
        --learning_rate 0.0003 \
        --random_seed "${seed}" \
        --des "${des}" \
        --use_gpu 1 2>&1 | tee "logs/${des}-F${fold}-S${seed}.log"
    done
  done
}

run_config \
  B26M-A0RAW-1749-2026-04 \
  sunspot_a0_raw_1749-07_2026-04.csv \
  configs/cv5_folds_a0_raw_2026-04.json

run_config \
  B26M-A0SMOOTH13-1749-2026-04 \
  sunspot_a0_smoothed13_1749-07_2026-04.csv \
  configs/cv5_folds_a0_smoothed13_2026-04.json
