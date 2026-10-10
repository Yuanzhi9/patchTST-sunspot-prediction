#!/bin/bash
# EXP-27~30 驱动：四个实验**串行**执行（W2/W3 × do0.05/0.2），跑完即评估
# 串行铁律：一次只跑一个训练进程（PID 锁在 launch_experiment.sh 内）
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/PatchTST_supervised"
export NVIDIA_TF32_OVERRIDE=0

COMMON="--model_id sunspot --model PatchTST --data custom --root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv --features MS --target ssn --enc_in 3 --seq_len 336 --label_len 48 --pred_len 24 --d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 --revin 1 --affine 0 --head_dropout 0.0 --individual 0 --train_epochs 50 --patience 100 --batch_size 16 --learning_rate 0.0001 --loss mse --activation gelu --random_seed 2021 --num_workers 0 --itr 1 --use_gpu 1"

run_one() {
  local EID="$1" DROP="$2" TS="$3" TE="$4" NT="$5"
  local ARGS="$COMMON --test_start $TS --test_end $TE --dropout $DROP --fc_dropout $DROP"
  echo "==================== $EID  dropout=$DROP  window $TS~$TE  num_train=$NT ===================="
  python3 save_config.py "$EID" --reason "dropout 口径分裂跨窗口验证（$EID: window $TS~$TE, dropout $DROP）" \
      --num_train "$NT" $ARGS >/dev/null || { echo "[$EID] save_config 失败"; return 1; }
  ./scripts/launch_experiment.sh "$EID" bash -c \
      "python3 -u run_longExp.py --is_training 1 $ARGS --des $EID && python3 -u run_longExp.py --is_training 0 $ARGS --des $EID" \
      || { echo "[$EID] launch 失败"; return 1; }
  while kill -0 "$(cat logs/$EID.pid 2>/dev/null)" 2>/dev/null; do sleep 5; done
  echo "[$EID] 训练结束"
  echo "--- 物理四指标 ---"
  python3 scripts/eval_metrics.py --config "configs/${EID}_2026-10-10.json" 2>&1 | tail -14
  echo "--- 滚动（整段测试期） ---"
  local DIR; DIR="$(ls -d checkpoints/*"$EID"_0 2>/dev/null | head -1)"
  python3 roll_eval.py --checkpoint "$DIR/full_checkpoint.pth" --num_train "$NT" --test_end "$TE" 2>&1 \
      | grep -E "滚动段|滚动 MAE|前期|后期|真实峰值"
  echo
}

run_one EXP-27 0.05 2008-12 2019-11 2986
run_one EXP-28 0.2  2008-12 2019-11 2986
run_one EXP-29 0.05 2019-12 2025-10 3118
run_one EXP-30 0.2  2019-12 2025-10 3118
echo "=== 四个实验全部完成 ==="
