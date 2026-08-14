"""
plot_rolling.py — 滚动预测曲线 vs 真实观测（阶段1验证三窗口，汇报用图）

复用 roll_eval.py 的推理逻辑（load_model/load_data_enc3/run_rolling_enc3），
滚动推理后画图：真实月值(黑实线) + 滚动预测(蓝虚线) + 峰值点标注。

用法:
  python3 scripts/plot_rolling.py
输出:
  plots/phase1_W1_rolling_YYYY-MM-DD.png
  plots/phase1_W2_rolling_YYYY-MM-DD.png
  plots/phase1_W3_rolling_YYYY-MM-DD.png

自检: 每个窗口打印滚动 MAE，必须与已记录值一致（W1=50.41, W2=66.80, W3=42.05），
      偏差 >0.5 说明画图数据与正式评估不一致，需排查。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
PLOT_DIR = os.path.join(ROOT, "plots")
TODAY = date.today().isoformat()

WINDOWS = [
    {
        "name": "W1",
        "label": "Cycle 23 (1996-08 to 2008-11)",
        "checkpoint": os.path.join(
            ROOT, "checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-18-5b_0/full_checkpoint.pth"),
        "num_train": 2838, "num_val": 132, "n_roll": 148,
        "test_start": "1996-08", "expected_mae": 50.41,
    },
    {
        "name": "W2",
        "label": "Cycle 24 (2008-12 to 2019-11)",
        "checkpoint": os.path.join(
            ROOT, "checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-19-2_0/full_checkpoint.pth"),
        "num_train": 2986, "num_val": 132, "n_roll": 132,
        "test_start": "2008-12", "expected_mae": 66.80,
    },
    {
        "name": "W3",
        "label": "Cycle 25 partial (2019-12 to 2025-10)",
        "checkpoint": os.path.join(
            ROOT, "checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-19-3_0/full_checkpoint.pth"),
        "num_train": 3118, "num_val": 132, "n_roll": 71,
        "test_start": "2019-12", "expected_mae": 42.05,
    },
]


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    for w in WINDOWS:
        print(f"=== {w['name']} ===")
        if not os.path.exists(w["checkpoint"]):
            print(f"  SKIP: checkpoint 不存在 {w['checkpoint']}")
            continue

        model, ckpt_args = load_model(w["checkpoint"])
        seq_len = getattr(ckpt_args, "seq_len", 96)
        enc_in = getattr(ckpt_args, "enc_in", 3)
        assert enc_in > 1, "本脚本只支持 enc_in=3 管线"

        df, scaler, data_z, t0 = load_data_enc3(DATA_CSV, w["num_train"], w["num_val"])
        n_roll = min(w["n_roll"], len(data_z) - t0)
        preds, trues = run_rolling_enc3(model, scaler, data_z, t0, seq_len, n_roll)

        # --- 自检：MAE 与已记录值一致 ---
        mae = float(np.mean(np.abs(preds - trues)))
        ok = abs(mae - w["expected_mae"]) < 0.5
        print(f"  滚动 MAE = {mae:.2f} (期望 {w['expected_mae']}, 自检 {'PASS' if ok else 'FAIL'})")
        if not ok:
            print("  ⚠️ MAE 与正式评估不一致，中止出图，先排查！")
            continue

        # --- 时间轴 ---
        x_axis = pd.date_range(start=w["test_start"], periods=n_roll, freq="MS")

        # --- 画图（学术风格，英文标注） ---
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(x_axis, trues, color="#111111", lw=1.4, label="Observation (monthly mean)")
        ax.plot(x_axis, preds, color="#1f5fbf", lw=1.2, ls="--", label="Rolling prediction (1-month ahead)")

        peak_idx = int(np.argmax(trues))
        pred_peak_idx = int(np.argmax(preds))
        err = preds[peak_idx] - trues[peak_idx]
        ax.plot(x_axis[peak_idx], trues[peak_idx], "o", color="#c0392b", ms=6, zorder=5)
        ax.plot(x_axis[pred_peak_idx], preds[pred_peak_idx], "o", color="#1f5fbf", ms=6, zorder=5)
        ax.annotate(
            f"True peak: {trues[peak_idx]:.1f}\nPred peak: {preds[peak_idx]:.1f}\nError: {err:+.1f} ({err/trues[peak_idx]*100:+.0f}%)",
            xy=(x_axis[peak_idx], trues[peak_idx]),
            xytext=(0.03, 0.72), textcoords="axes fraction",
            fontsize=9.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
        )

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("SSN", fontsize=11)
        ax.set_title(f"{w['name']} {w['label']} — rolling prediction vs. observation", fontsize=12)
        ax.grid(axis="y", color="#cccccc", alpha=0.35, lw=0.5)
        ax.grid(axis="x", visible=False)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", fontsize=9.5, frameon=False)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"phase1_{w['name']}_rolling_{TODAY}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"  已保存: {out_path}")


if __name__ == "__main__":
    main()
