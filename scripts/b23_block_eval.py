"""
b23_block_eval.py — block 预测评估（EXP-23）

block 预测策略：模型一次输出 pred_len 步，全部回填输入，隔 block_size 个月再预测下一块
（对比：滚动策略每次只用 step0，逐月回填）。回填频率低 = 误差自回归累积机会少。

自检（正确性验证）：block_size=1 时本脚本结果必须与 roll_eval.py 的滚动结果逐位一致
（此时两者数学上等价）——不一致 = 回填逻辑有 bug，禁止用于正式实验。

用法:
  python3 scripts/b23_block_eval.py --checkpoint PATH --num_train 2838 --num_val 132 \
    --n_roll 148 --block_size 24 [--overlap 12] [--target_transform sqrt]

输出: 全段 MAE / 前期后期 / 峰值对照。与 roll_eval 同口径可直接对比。
"""

import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, INV_TARGET_TRANSFORMS

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")


def inverse_phy(scaler, z_row):
    """z 行 → 物理 SSN（无 target_transform 时恒等）。"""
    return float(scaler.inverse_transform(z_row.reshape(1, -1))[0, 2])


def block_predict(model, scaler, data_z, test_start_idx, seq_len, n_roll, block_size, overlap):
    """block 预测。overlap=0 无重叠；overlap>0 时相邻块重叠 overlap 个月，接缝取平均。"""
    window = data_z[test_start_idx - seq_len : test_start_idx].copy()
    preds = [None] * n_roll
    counts = np.zeros(n_roll, dtype=int)

    step = 0
    while step < n_roll:
        x = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        # [修复] 只取前 block_size 步（block=1 时仅 step0，与滚动等价）
        z_block = out[0, :block_size, 2].cpu().numpy()          # (block_size,) z 值
        actual = min(block_size, n_roll - step)

        # 记录预测（重叠月平均）
        for j in range(actual):
            z_val = float(z_block[j])
            row_z = np.array([[window[-1, 0], window[-1, 1], z_val]])
            phy = inverse_phy(scaler, row_z)
            if preds[step + j] is None:
                preds[step + j] = phy
            else:
                preds[step + j] = (preds[step + j] * counts[step + j] + phy) / (counts[step + j] + 1)
            counts[step + j] += 1

        # 回填：整块预测值进入窗口（sin/cos 用真实值，同 roll_eval 约定）
        for j in range(actual):
            t = step + j
            true_z_month = data_z[test_start_idx + t]
            new_row = np.array([[window[-1, 0], window[-1, 1], float(z_block[j])]])
            new_row[0, 0] = true_z_month[0]
            new_row[0, 1] = true_z_month[1]
            window = np.vstack([window[1:], new_row])

        advance = block_size - overlap if overlap > 0 else block_size
        step += max(advance, 1)

    return np.array([p for p in preds if p is not None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_train", type=int, required=True)
    p.add_argument("--num_val", type=int, default=132)
    p.add_argument("--n_roll", type=int, default=148)
    p.add_argument("--block_size", type=int, default=24)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--target_transform", default="")
    args = p.parse_args()

    model, ckpt_args = load_model(args.checkpoint)
    seq_len = getattr(ckpt_args, "seq_len", 96)
    _, scaler, data_z, t0 = load_data_enc3(
        DATA_CSV, args.num_train, args.num_val, 'standard', args.target_transform)

    # 真值（物理空间，注意 target_transform 逆变换）
    inv = INV_TARGET_TRANSFORMS.get(args.target_transform, lambda y: y)
    trues = []
    for t in range(args.n_roll):
        z_row = data_z[t0 + t]
        trues.append(float(inv(inverse_phy(scaler, z_row))))
    trues = np.array(trues)

    preds = block_predict(model, scaler, data_z, t0, seq_len, args.n_roll,
                          args.block_size, args.overlap)
    if args.target_transform:
        preds = inv(preds)

    err = np.abs(preds - trues)
    mae = float(np.mean(err))
    print(f"block 预测 (block={args.block_size}, overlap={args.overlap}, 全 {len(preds)} 月): MAE = {mae:.2f}")
    third = max(len(preds) // 3, 1)
    first, last = float(np.mean(err[:third])), float(np.mean(err[-third:]))
    print(f"  前期 MAE (前 {third} 月): {first:.2f}  后期 (后 {third} 月): {last:.2f}  比值: {last/max(first,1e-9):.1f}x")
    ti, pi = int(np.argmax(trues)), int(np.argmax(preds))
    print(f"  真实峰: {trues[ti]:.1f}  预测峰: {preds[pi]:.1f}  误差: {trues[ti]-preds[pi]:+.1f}")


if __name__ == "__main__":
    main()
