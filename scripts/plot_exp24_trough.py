"""
plot_exp24_trough.py — EXP-24 谷值放大图（W1 终点谷 2008-08，唯一完整谷结构）

复用 plot_exploration.py 已验证的滚动/block 函数，对谷点 ±12 月窗口放大画图。
对拍门禁：轨迹 MAE 必须等于已记值（滚动 50.41、block 21.29）。

输出: plots/fig_W1_trough_2008-08_2026-08-20.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3
from scripts.b23_block_eval import block_predict
from scripts.plot_exploration import trues_array

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
PLOT_DIR = os.path.join(ROOT, "plots")
CKPT = os.path.join(ROOT, "checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-18-5b_0/full_checkpoint.pth")
NUM_TRAIN, N_ROLL, TEST_START = 2838, 148, "1996-08"
EXP_ROLL, EXP_BLOCK = 50.41, 21.29
TROUGH_YM = "2008-08"


def main():
    model, ckpt_args = load_model(CKPT)
    seq_len = getattr(ckpt_args, "seq_len", 96)
    _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, NUM_TRAIN, 132)
    trues = trues_array(scaler, data_z, t0, N_ROLL)
    roll_preds, _ = run_rolling_enc3(model, scaler, data_z, t0, seq_len, N_ROLL)
    block_preds = block_predict(model, scaler, data_z, t0, seq_len, N_ROLL, 24, 0)

    mae_r = float(np.mean(np.abs(roll_preds - trues)))
    mae_b = float(np.mean(np.abs(block_preds - trues)))
    ok = abs(mae_r - EXP_ROLL) < 0.5 and abs(mae_b - EXP_BLOCK) < 0.5
    print(f"对拍: 滚动={mae_r:.2f}(记{EXP_ROLL}) block={mae_b:.2f}(记{EXP_BLOCK}) {'PASS' if ok else 'FAIL'}")
    assert ok, "对拍失败，不出图"

    x = pd.date_range(start=TEST_START, periods=N_ROLL, freq="MS")
    ym = np.array([d.strftime("%Y-%m") for d in x])
    ti = int(np.where(ym == TROUGH_YM)[0][0])
    lo, hi = max(0, ti - 12), min(N_ROLL, ti + 13)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x[lo:hi], trues[lo:hi], color="#111111", lw=1.5, label="Observation")
    ax.plot(x[lo:hi], roll_preds[lo:hi], color="#1f5fbf", lw=1.2, ls="--",
            label="Rolling (trough collapse: min -194.2)")
    ax.plot(x[lo:hi], block_preds[lo:hi], color="#e08000", lw=1.2, ls="--",
            label="Block 24 (min 15.3)")

    tmin_i = int(lo + np.argmin(trues[lo:hi]))
    ax.plot(x[tmin_i], trues[tmin_i], "o", color="#111111", ms=6, zorder=5)
    ax.annotate(f"True trough {trues[tmin_i]:.1f} ({ym[tmin_i]})",
                xy=(x[tmin_i], trues[tmin_i]), xytext=(6, -14),
                textcoords="offset points", fontsize=9, color="#333333")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("SSN", fontsize=11)
    ax.set_title("W1 trough zoom (2007-08 ~ 2008-11) — rolling collapses, block stays bounded", fontsize=11.5)
    ax.grid(axis="y", color="#cccccc", alpha=0.35, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig_W1_trough_2008-08_2026-08-20.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
