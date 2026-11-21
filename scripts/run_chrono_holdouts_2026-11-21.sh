#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/PatchTST_supervised:/openbayes/home/python-packages"
export OMP_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0

P24_D_MODEL="${P24_D_MODEL:-128}"
P24_N_HEADS="${P24_N_HEADS:-8}"
P24_E_LAYERS="${P24_E_LAYERS:-2}"
P24_D_FF="${P24_D_FF:-2048}"

cycles=(SC20 SC21 SC24)

p24_config() {
  case "$1" in
    SC20) printf '%s\n' "configs/chrono_cycle20_holdout_2026-11.json" ;;
    SC21) printf '%s\n' "configs/chrono_cycle21_holdout_2026-11.json" ;;
    SC24) printf '%s\n' "configs/chrono_cycle24_holdout_2026-11.json" ;;
    *) return 1 ;;
  esac
}

p132_config() {
  case "$1" in
    SC20) printf '%s\n' "configs/chrono_cycle20_holdout_p132_2026-11.json" ;;
    SC21) printf '%s\n' "configs/chrono_cycle21_holdout_p132_2026-11.json" ;;
    SC24) printf '%s\n' "configs/chrono_cycle24_holdout_p132_2026-11.json" ;;
    *) return 1 ;;
  esac
}

result_dir() {
  local marker="$1"
  local found
  found="$(find results -maxdepth 1 -type d -name "*_${marker}" -print)"
  if [[ -z "$found" || "$(printf '%s\n' "$found" | wc -l)" -ne 1 ]]; then
    printf 'expected exactly one result for %s, got:\n%s\n' "$marker" "$found" >&2
    return 1
  fi
  printf '%s\n' "$found"
}

for cycle in "${cycles[@]}"; do
  mkdir -p "analysis/cycle_holdout/${cycle}"
  for seed in 2019 2020 2021 2022 2023; do
    p24_des="B26M-CHRONO-${cycle}"
    p24_marker="${p24_des}_F1_seed${seed}"
    p24_dir="$(find results -maxdepth 1 -type d -name "*_${p24_marker}" -print 2>/dev/null || true)"
    if [[ -z "$p24_dir" ]]; then
      python3 -u run_base_cv5.py \
        --cv_file "$(p24_config "$cycle")" --cv_fold 1 \
        --seq_len 132 --label_len 48 --pred_len 24 --patch_len 36 --stride 18 \
        --d_model "$P24_D_MODEL" --n_heads "$P24_N_HEADS" \
        --e_layers "$P24_E_LAYERS" --d_ff "$P24_D_FF" \
        --dropout 0.2 --fc_dropout 0.2 --head_dropout 0.0 \
        --train_epochs 50 --patience 100 --batch_size 32 --learning_rate 3e-4 \
        --random_seed "$seed" --des "$p24_des" --use_gpu 1 \
        > "logs/${p24_des}-S${seed}-F1.log" 2>&1
    fi
    p24_dir="$(result_dir "$p24_marker")"
    p24_out="analysis/cycle_holdout/${cycle}/p24_update_seed${seed}"
    python3 scripts/eval_cv5.py --des "$p24_des" --seed "$seed" --out_dir "$p24_out" \
      > "logs/${p24_des}-S${seed}-update-eval.log" 2>&1

    update_csv="${p24_out}/p24_update_curve.csv"
    if [[ ! -f "$update_csv" ]]; then
      python3 scripts/stitch_p24_update_curve.py \
        --result_dir "$p24_dir" --out_csv "$update_csv" \
        > "logs/${p24_des}-S${seed}-update-stitch.log" 2>&1
    fi

    recursive_csv="${p24_out}/p24_recursive_curve.csv"
    if [[ ! -f "$recursive_csv" ]]; then
      python3 scripts/rollout_p24_cycle.py \
        --result_dir "$p24_dir" --horizon 132 --out_csv "$recursive_csv" \
        > "logs/${p24_des}-S${seed}-recursive.log" 2>&1
    fi

    python3 scripts/eval_cycle_extrema.py \
      --curve "$update_csv" \
      --out "analysis/cycle_holdout/${cycle}/p24_update_seed${seed}_extrema.json" \
      > "logs/${p24_des}-S${seed}-update-extrema.log" 2>&1
    python3 scripts/eval_cycle_extrema.py \
      --curve "$recursive_csv" \
      --out "analysis/cycle_holdout/${cycle}/p24_recursive_seed${seed}_extrema.json" \
      > "logs/${p24_des}-S${seed}-recursive-extrema.log" 2>&1

    for block_size in 6 12; do
      sensitivity_csv="${p24_out}/p24_recursive_b${block_size}_curve.csv"
      if [[ ! -f "$sensitivity_csv" ]]; then
        python3 scripts/rollout_p24_cycle.py \
          --result_dir "$p24_dir" --horizon 132 --block_size "$block_size" \
          --out_csv "$sensitivity_csv" \
          > "logs/${p24_des}-S${seed}-recursive-b${block_size}.log" 2>&1
      fi
      python3 scripts/eval_cycle_extrema.py \
        --curve "$sensitivity_csv" \
        --out "analysis/cycle_holdout/${cycle}/p24_recursive_b${block_size}_seed${seed}_extrema.json" \
        > "logs/${p24_des}-S${seed}-recursive-b${block_size}-extrema.log" 2>&1
    done

    p132_des="B26M-P132-CHRONO-${cycle}"
    p132_marker="${p132_des}_F1_seed${seed}"
    p132_dir="$(find results -maxdepth 1 -type d -name "*_${p132_marker}" -print 2>/dev/null || true)"
    if [[ -z "$p132_dir" ]]; then
      python3 -u run_base_cv5.py \
        --cv_file "$(p132_config "$cycle")" --cv_fold 1 \
        --seq_len 132 --label_len 48 --pred_len 132 --patch_len 36 --stride 18 \
        --d_model 128 --n_heads 2 --e_layers 2 --d_ff 2048 \
        --dropout 0.4 --fc_dropout 0.3 --head_dropout 0.0 \
        --train_epochs 50 --patience 100 --batch_size 32 --learning_rate 2.9802410473297515e-3 \
        --random_seed "$seed" --des "$p132_des" --use_gpu 1 \
        > "logs/${p132_des}-S${seed}-F1.log" 2>&1
    fi
    p132_dir="$(result_dir "$p132_marker")"
    p132_out="analysis/cycle_holdout/${cycle}/p132_direct_seed${seed}"
    python3 scripts/eval_p132_cv5.py --des "$p132_des" --seed "$seed" --out_dir "$p132_out" \
      > "logs/${p132_des}-S${seed}-eval.log" 2>&1
    python3 scripts/eval_cycle_extrema.py \
      --curve "${p132_dir}/p132_curve.csv" \
      --out "analysis/cycle_holdout/${cycle}/p132_direct_seed${seed}_extrema.json" \
      > "logs/${p132_des}-S${seed}-extrema.log" 2>&1

    printf 'completed %s seed%s\n' "$cycle" "$seed"
  done
done

python3 scripts/summarize_cycle_holdout.py
python3 scripts/plot_cycle_holdout.py
printf 'chronological holdout suite complete\n'
