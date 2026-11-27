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
  --use_gpu 1
)

run_if_missing() {
  local des="$1"
  local seed="$2"
  local fold="$3"
  local mode="$4"
  local weight="$5"
  local expected
  expected=$(find results -maxdepth 1 -type d \
    -name "*_${des}_F${fold}_seed${seed}" -print -quit)
  if [ -n "$expected" ] && [ -f "$expected/pred.npy" ] && [ -f "$expected/true.npy" ]; then
    return
  fi
  local log="logs/pr24_multitask/${des}_F${fold}_seed${seed}.log"
  python3 -u run_cv5_multitask.py \
    "${COMMON[@]}" \
    --cv_fold "$fold" \
    --random_seed "$seed" \
    --mt_mode "$mode" \
    --mt_aux_weight "$weight" \
    --des "$des" \
    >>"$log" 2>&1
}

run_if_missing B26M-MT-BASE-A1 2021 5 none 0
python3 scripts/eval_cv5.py \
  --des B26M-MT-BASE-A1 \
  --seed 2021 \
  --out_dir analysis/pr24_multitask_extrema/A1_baseline_seed2021

for seed in 2019 2020 2021 2022 2023; do
  for fold in 1 2 3 4 5; do
    run_if_missing B26M-MT-ALL-A5 "$seed" "$fold" all 0.1
  done
  python3 scripts/eval_cv5.py \
    --des B26M-MT-ALL-A5 \
    --seed "$seed" \
    --out_dir "analysis/pr24_multitask_extrema/A5_all_seed${seed}"
  python3 scripts/eval_pr24_multitask_aux.py \
    --des B26M-MT-ALL-A5 \
    --seed "$seed" \
    --out_dir "analysis/pr24_multitask_extrema/A5_all_seed${seed}"
done
