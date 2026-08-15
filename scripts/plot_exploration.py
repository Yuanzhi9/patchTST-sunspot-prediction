"""
plot_exploration.py — 探索期对比图（滚动 vs block vs oracle step0 + 真实值）

oracle step0 线：每个窗口用完全真实输入，只取 step0（预测下1月），
回填真实值——"有上帝视角的逐月预测"，展示无回填污染的模型单步能力。

自检：图内滚动 MAE 必须与 result.txt 已记值一致（不一致=bug，先修再出图）。

输出：
  plots/fig_W1_roll_vs_block_YYYY-MM-DD.png
  plots/fig_W2_roll_vs_block_YYYY-MM-DD.png
  plots/fig_W3_roll_vs_block_YYYY-MM-DD.png
  plots/fig_W1_multimodel_YYYY-MM-DD.png（基线/wmse_th/pow23 滚动对比）
"""

import os
import sys
from datetime import date

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

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
TODAY = date.today().isoformat()
PLOT_DIR = os.path.join(ROOT, "plots")

# (窗口名, checkpoint名, num_train, n_roll, test_start, 已记滚动MAE, 已记block MAE)
WINDOWS = [
    ("W1", "EXP-18-5b", 2838, 148, "1996-08", 50.41, 21.29),
    ("W2", "EXP-19-2", 2986, 132, "2008-12", 66.80, 27.28),
    ("W3", "EXP-19-3", 3118, 71, "2019-12", 42.05, 23.75),
]


def oracle_step0(model, scaler, data_z, t0, seq_len, n_roll):
    """oracle：真实输入，只取 step0，回填真实值（无误差累积）。"""
    import torch
    window = data_z[t0 - seq_len:t0].copy()
    preds = []
    for t in range(n_roll):
        x = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        z_val = out[0, 0, 2].item()
        row_z = np.array([[window[-1, 0], window[-1, 1], z_val]])
        preds.append(float(scaler.inverse_transform(row_z)[0, 2]))
        true_z = data_z[t0 + t]
        new_row = np.array([[window[-1, 0], window[-1, 1], float(true_z[2])]])
        new_row[0, 0] = true_z[0]
        new_row[0, 1] = true_z[1]
        window = np.vstack([window[1:], new_row])
    return np.array(preds)


def trues_array(scaler, data_z, t0, n_roll):
    out = []
    for t in range(n_roll):
        out.append(float(scaler.inverse_transform(data_z[t0 + t].reshape(1, -1))[0, 2]))
    return np.array(out)


def find_ckpt(name):
    import glob
    hits = glob.glob(os.path.join(ROOT, f"checkpoints/*{name}_0/full_checkpoint.pth"))
    assert hits, f"checkpoint not found: {name}"
    return hits[0]


def plot_window(wname, ckpt_name, num_train, n_roll, test_start, exp_roll, exp_block):
    ckpt = find_ckpt(ckpt_name)
    model, ckpt_args = load_model(ckpt)
    seq_len = getattr(ckpt_args, "seq_len", 96)
    _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, num_train, 132)

    trues = trues_array(scaler, data_z, t0, n_roll)
    oracle = oracle_step0(model, scaler, data_z, t0, seq_len, n_roll)
    roll_preds, _ = run_rolling_enc3(model, scaler, data_z, t0, seq_len, n_roll)
    block_preds = block_predict(model, scaler, data_z, t0, seq_len, n_roll, 24, 0)

    # 自检
    mae_roll = float(np.mean(np.abs(roll_preds - trues)))
    mae_block = float(np.mean(np.abs(block_preds - trues)))
    mae_oracle = float(np.mean(np.abs(oracle - trues)))
    ok = abs(mae_roll - exp_roll) < 0.5 and abs(mae_block - exp_block) < 0.5
    print(f"{wname}: 滚动MAE={mae_roll:.2f}(记{exp_roll}) blockMAE={mae_block:.2f}(记{exp_block}) oracleMAE={mae_oracle:.2f} 自检{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  ⚠️ 自检失败，不出图"); return

    x = pd.date_range(start=test_start, periods=n_roll, freq="MS")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, trues, color="#111111", lw=1.5, label="Observation")
    ax.plot(x, oracle, color="#999999", lw=1.1, ls=":", label="Oracle step0 (true-input)")
    ax.plot(x, roll_preds, color="#1f5fbf", lw=1.2, ls="--", label="Rolling (1-month feedback)")
    ax.plot(x, block_preds, color="#e08000", lw=1.2, ls="--", label="Block 24 (24-month feedback)")

    ti = int(np.argmax(trues))
    ax.plot(x[ti], trues[ti], "o", color="#111111", ms=5, zorder=5)
    ax.annotate(f"True peak {trues[ti]:.0f}",
                xy=(x[ti], trues[ti]), xytext=(8, 6), textcoords="offset points",
                fontsize=9, color="#333333")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("SSN", fontsize=11)
    ax.set_title(f"{wname} — rolling vs block vs oracle step0 (best model)", fontsize=12)
    ax.grid(axis="y", color="#cccccc", alpha=0.35, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"fig_{wname}_roll_vs_block_{TODAY}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  已保存 {out}")


def plot_multimodel():
    """W1 多模型滚动对比：基线 mse / wmse_th / pow23。"""
    # (checkpoint名, 颜色, 标签, 已记MAE, target_transform)
    models = [
        ("EXP-18-5b", "#1f5fbf", "mse (baseline)", 50.41, ""),
        ("EXP-22-3", "#2e8b57", "wmse_th (threshold)", 33.92, ""),
        ("EXP-21-2", "#8b008b", "pow23 transform", 22.13, "pow23"),
    ]
    _, scaler_ref, data_z_ref, t0_ref = load_data_enc3(DATA_CSV, 2838, 132)
    trues = trues_array(scaler_ref, data_z_ref, t0_ref, 148)
    x = pd.date_range(start="1996-08", periods=148, freq="MS")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, trues, color="#111111", lw=1.5, label="Observation")
    ok_all = True
    for name, color, label, exp_mae, tt in models:
        ckpt = find_ckpt(name)
        model, ckpt_args = load_model(ckpt)
        seq_len = getattr(ckpt_args, "seq_len", 96)
        # 变换模型需要各自的变换空间数据（scaler fit 在变换空间）
        _, scaler_m, data_z_m, t0_m = load_data_enc3(DATA_CSV, 2838, 132, 'standard', tt)
        preds, _ = run_rolling_enc3(model, scaler_m, data_z_m, t0_m, seq_len, 148, tt)
        mae = float(np.mean(np.abs(preds - trues)))
        ok = abs(mae - exp_mae) < 0.5
        ok_all = ok_all and ok
        print(f"  {label}: MAE={mae:.2f}(记{exp_mae}) {'PASS' if ok else 'FAIL'}")
        ax.plot(x, preds, color=color, lw=1.1, ls="--", label=f"{label} (rolling MAE {mae:.1f})")

    ti = int(np.argmax(trues))
    ax.plot(x[ti], trues[ti], "o", color="#111111", ms=5, zorder=5)
    ax.annotate(f"True peak {trues[ti]:.0f}", xy=(x[ti], trues[ti]),
                xytext=(8, 6), textcoords="offset points", fontsize=9, color="#333333")
    ax.set_xlabel("Date", fontsize=11); ax.set_ylabel("SSN", fontsize=11)
    ax.set_title("W1 — rolling trajectories of three models", fontsize=12)
    ax.grid(axis="y", color="#cccccc", alpha=0.35, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"fig_W1_multimodel_{TODAY}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"多模型图自检{'PASS' if ok_all else 'FAIL'} → {out}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    for w in WINDOWS:
        plot_window(*w)
    plot_multimodel()
