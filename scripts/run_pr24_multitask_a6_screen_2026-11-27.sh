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
  --mt_mode all
)

for spec in \
  "fixed:0.03:B26M-MT-A6-W003" \
  "fixed:0.3:B26M-MT-A6-W030" \
  "uncertainty:1.0:B26M-MT-A6-UNC"; do
  weighting="${spec%%:*}"
  rest="${spec#*:}"
  weight="${rest%%:*}"
  des="${rest#*:}"
  for fold in 1 2 3 4; do
    log="logs/pr24_multitask/${des}_F${fold}.log"
    python3 -u run_cv5_multitask.py \
      "${COMMON[@]}" \
      --cv_fold "$fold" \
      --mt_aux_weight "$weight" \
      --mt_weighting "$weighting" \
      --des "$des" \
      >>"$log" 2>&1
  done
  python3 scripts/eval_cv5.py \
    --des "$des" \
    --seed 2021 \
    --out_dir "analysis/pr24_multitask_extrema/${des}_seed2021"
  python3 scripts/eval_pr24_multitask_aux.py \
    --des "$des" \
    --seed 2021 \
    --out_dir "analysis/pr24_multitask_extrema/${des}_seed2021"
done
