#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  printf 'usage: %s <cycle> <seed> <unc|w030>\n' "$0" >&2
  exit 2
fi

cycle="$1"
seed="$2"
variant="$3"
cd /output/PR24-MULTITASK-EXTREMA
export PYTHONPATH="$PWD/PatchTST_supervised"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

case "$variant" in
  unc)
    weighting="uncertainty"
    weight="1.0"
    des="B26M-MT-CHRONO-UNC-${cycle}-V2"
    base="analysis/pr24_multitask_cycle_holdout_unc_v2"
    ;;
  w030)
    weighting="fixed"
    weight="0.3"
    des="B26M-MT-CHRONO-W030-${cycle}-V2"
    base="analysis/pr24_multitask_cycle_holdout_w030_v2"
    ;;
  *)
    printf 'unknown variant: %s\n' "$variant" >&2
    exit 2
    ;;
esac

cycle_number="${cycle#SC}"
cv_file="configs/chrono_cycle${cycle_number}_holdout_2026-11.json"
labels="analysis/pr24_multitask_extrema/chrono_labels/${cycle}/event_labels_all_windows.csv"
marker="${des}_F1_seed${seed}"
log="logs/pr24_multitask_chrono/${des}_seed${seed}.log"
mkdir -p logs/pr24_multitask_chrono "${base}/${cycle}"

result_dir="$(find results -maxdepth 1 -type d -name "*_${marker}" -print -quit)"
if [ -z "$result_dir" ] || [ ! -f "$result_dir/pred.npy" ]; then
  python3 -u run_cv5_multitask.py \
    --cv_file "$cv_file" \
    --event_labels "$labels" \
    --cv_fold 1 \
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
    --head_dropout 0.0 \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --train_epochs 50 \
    --patience 100 \
    --use_gpu 1 \
    --mt_mode all \
    --mt_aux_weight "$weight" \
    --mt_weighting "$weighting" \
    --random_seed "$seed" \
    --des "$des" \
    >>"$log" 2>&1
fi

result_dir="$(find results -maxdepth 1 -type d -name "*_${marker}" -print -quit)"
if [ -z "$result_dir" ]; then
  printf 'missing result for %s\n' "$marker" >&2
  exit 1
fi
python3 - "$result_dir/cv_window_meta.json" "$cv_file" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
spec = json.load(open(sys.argv[2], encoding="utf-8"))
observed = meta["fold_spec"]["outer_val"]
expected = spec["folds"][0]["outer_val"]
for key in ("start", "end", "start_date", "end_date"):
    if observed[key] != expected[key]:
        raise SystemExit(
            f"outer_val {key} mismatch: result={observed[key]} expected={expected[key]}"
        )
PY

update_dir="${base}/${cycle}/p24_update_seed${seed}"
mkdir -p "$update_dir"
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
  --out "${base}/${cycle}/p24_update_seed${seed}_extrema.json" \
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
  --out "${base}/${cycle}/p24_recursive_multitask_seed${seed}_extrema.json" \
  >"logs/pr24_multitask_chrono/${des}_seed${seed}_recursive_extrema.log" 2>&1
python3 scripts/eval_pr24_multitask_aux.py \
  --des "$des" \
  --seed "$seed" \
  --out_dir "$update_dir" \
  >"logs/pr24_multitask_chrono/${des}_seed${seed}_aux_eval.log" 2>&1
