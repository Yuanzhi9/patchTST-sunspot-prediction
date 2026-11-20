#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 9 ]; then
  echo "usage: $0 DES SEQ_LEN PATCH STRIDE LR BATCH DROPOUT HEADS SEED [FOLD...]" >&2
  exit 2
fi

des="$1"
seq_len="$2"
patch_len="$3"
stride="$4"
lr="$5"
batch_size="$6"
dropout="$7"
heads="$8"
seed="$9"
shift 9

if [ "$#" -eq 0 ]; then
  set -- 1 2 3 4 5
fi

export PYTHONPATH="$(pwd)/PatchTST_supervised"
export OMP_NUM_THREADS=4

for fold in "$@"; do
  echo "[$(date '+%F %T')] ${des} F${fold} seq=${seq_len} patch=${patch_len}/${stride} lr=${lr} bs=${batch_size} do=${dropout} nh=${heads} seed=${seed}"
  python3 -u run_base_cv5.py \
    --cv_file configs/cv5_folds_p132_2026-10.json \
    --cv_fold "${fold}" \
    --seq_len "${seq_len}" \
    --pred_len 132 \
    --patch_len "${patch_len}" \
    --stride "${stride}" \
    --d_model 128 \
    --n_heads "${heads}" \
    --e_layers 2 \
    --d_ff 2048 \
    --dropout "${dropout}" \
    --fc_dropout "${dropout}" \
    --train_epochs 50 \
    --patience 100 \
    --batch_size "${batch_size}" \
    --learning_rate "${lr}" \
    --random_seed "${seed}" \
    --des "${des}" \
    --use_gpu 1 2>&1 | tee "logs/${des}-F${fold}.log"
done
