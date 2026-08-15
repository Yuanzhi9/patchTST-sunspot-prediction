"""
b3_m4_comparison.py — M4 同口径对拍（线 B-3）

背景（2026-08-15 读学长代码发现的三个口径差异）：
  1. 目标变量：M4 用 13 月平滑 SSN（Cycle 25 平滑峰值 159.2）；我们用月均值（峰值 216）
  2. 验证窗口：M4 是 future_only——前 36 个月观测用于拟合，验证其后 35 个月（2022-12~2025-10）
  3. 预测方式：M4 是参数化 Waldmeier 曲线外推；我们是逐月滚动

本脚本把我们的 W3 滚动预测拉到 M4 的口径：
  滚动月均值预测 → 13 月平滑 → 取 obs 36 个月后的 35 个月 → 算 MAE/RMSE/峰值误差
与 M4 的 MAE=6.144（cycle24_cycle25_validation_metrics.csv, M4 36m, cycle 25）对比。
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
CKPT = os.path.join(
    ROOT,
    "checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-19-3_0/full_checkpoint.pth",
)
NUM_TRAIN, NUM_VAL, N_ROLL = 3118, 132, 71


def smooth13(x):
    """13 月滑动平均（SILSO 惯例：t-6..t+6，端点用可用部分）。"""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - 6)
        hi = min(n, i + 7)
        out[i] = np.mean(x[lo:hi])
    return out


def main():
    print("加载模型 + W3 滚动推理（71 月，2019-12 ~ 2025-10）...")
    model, ckpt_args = load_model(CKPT)
    seq_len = getattr(ckpt_args, "seq_len", 96)
    _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, NUM_TRAIN, NUM_VAL)
    preds, trues = run_rolling_enc3(model, scaler, data_z, t0, seq_len, N_ROLL)

    print("13 月平滑（SILSO 惯例）...")
    preds_s = smooth13(preds)
    trues_s = smooth13(trues)

    # M4 口径：obs 36 个月，验证其后 35 个月（idx 36..70）
    lo = 36
    p, t = preds_s[lo:], trues_s[lo:]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r2 = 1 - float(np.sum((p - t) ** 2) / np.sum((t - np.mean(t)) ** 2))

    pi = int(np.argmax(p))
    ti = int(np.argmax(t))
    print(f"\n=== 我们的 W3 拉到 M4 口径（平滑空间, obs36m, 验证 35 个月） ===")
    print(f"  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.3f}")
    print(f"  预测峰 = {p[pi]:.1f} (平滑) @ 第{lo+pi}月")
    print(f"  真实峰 = {t[ti]:.1f} (平滑) @ 第{lo+ti}月")
    print(f"  峰值幅度误差 = {p[pi]-t[ti]:+.1f}")

    print(f"\n=== M4 (36m) 同口径对照（学长代码输出） ===")
    print("  M4 Waldmeier (36m) MAE = 6.144, RMSE = 7.344, R² = 0.710, Bias = +0.031")
    print("  M4 峰值误差 = -15.86 (143.4 vs 159.2)")

    print(f"\n=== 对比结论 ===")
    print(f"  MAE:  我们 {mae:.2f} vs M4 6.14  → 差 {mae/6.144:.1f} 倍")
    print(f"  注意: 完整口径差异说明见 phys_params_survey.md §口径对拍节")


if __name__ == "__main__":
    main()
