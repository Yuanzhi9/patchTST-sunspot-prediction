#!/usr/bin/env bash
set -euo pipefail

cd /output/PR24-MULTITASK-EXTREMA
export PYTHONPATH="$PWD/PatchTST_supervised"

mkdir -p logs/pr24_multitask_chrono
BASE="analysis/pr24_multitask_cycle_holdout"

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
  --mt_mode all
  --mt_aux_weight 0.1
)

for cycle in SC20 SC21 SC24; do
  cycle_number="${cycle#SC}"
  cv_file="configs/chrono_cycle${cycle_number}_holdout_2026-11.json"
  labels="analysis/pr24_multitask_extrema/chrono_labels/${cycle}/event_labels_all_windows.csv"
  des="B26M-MT-CHRONO-${cycle}"
  mkdir -p "${BASE}/${cycle}"

  for seed in 2019 2020 2021 2022 2023; do
    marker="${des}_F1_seed${seed}"
    result_dir="$(find results -maxdepth 1 -type d -name "*_${marker}" -print -quit)"
    if [ -z "$result_dir" ] || [ ! -f "$result_dir/pred.npy" ]; then
      log="logs/pr24_multitask_chrono/${des}_seed${seed}.log"
      python3 -u run_cv5_multitask.py \
        "${COMMON[@]}" \
        --cv_file "$cv_file" \
        --event_labels "$labels" \
        --cv_fold 1 \
        --random_seed "$seed" \
        --des "$des" \
        >>"$log" 2>&1
    fi
    result_dir="$(find results -maxdepth 1 -type d -name "*_${marker}" -print -quit)"
    if [ -z "$result_dir" ]; then
      printf 'missing result for %s\n' "$marker" >&2
      exit 1
    fi

    update_dir="${BASE}/${cycle}/p24_update_seed${seed}"
    python3 scripts/eval_cv5.py \
      --des "$des" \
      --seed "$seed" \
      --out_dir "$update_dir" \
      >"logs/pr24_multitask_chrono/${des}_seed${seed}_update_eval.log" 2>&1
    python3 scripts/stitch_p24_update_curve.py \
      --result_dir "$result_dir" \
      --out_csv "${update_dir}/p24_update_curve.csv" \
      >"logs/pr24_multitask_chrono/${des}_seed${seed}_update_stitch.log" 2>&1
    python3 scripts/eval_cycle_extrema.py \
      --curve "${update_dir}/p24_update_curve.csv" \
      --out "${BASE}/${cycle}/p24_update_seed${seed}_extrema.json" \
      >"logs/pr24_multitask_chrono/${des}_seed${seed}_update_extrema.log" 2>&1

    python3 scripts/rollout_p24_multitask_cycle.py \
      --result_dir "$result_dir" \
      --horizon 132 \
      --block_size 24 \
      --out_csv "${update_dir}/p24_recursive_multitask_curve.csv" \
      --blocks_csv "${update_dir}/p24_recursive_multitask_blocks.csv" \
      >"logs/pr24_multitask_chrono/${des}_seed${seed}_recursive.log" 2>&1
    python3 scripts/eval_cycle_extrema.py \
      --curve "${update_dir}/p24_recursive_multitask_curve.csv" \
      --out "${BASE}/${cycle}/p24_recursive_multitask_seed${seed}_extrema.json" \
      >"logs/pr24_multitask_chrono/${des}_seed${seed}_recursive_extrema.log" 2>&1

    python3 scripts/eval_pr24_multitask_aux.py \
      --des "$des" \
      --seed "$seed" \
      --out_dir "${update_dir}"
  done
done

python3 scripts/summarize_cycle_holdout.py \
  --base "$BASE" \
  --methods p24_update,p24_recursive_multitask \
  --output-prefix pr24_multitask
