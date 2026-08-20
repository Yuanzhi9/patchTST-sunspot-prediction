"""
plot_figures_e1e4.py — 探索期核心图 E1-E4（2026-08-20）

铁律执行：数据源=各实验 checkpoint 重新推理（滚动/block）+ npy 分层重算；
全部与 result.txt 已记值对拍，逐位一致才出图；变换模型带同一 target_transform；
色盲友好色板（蓝/橙/灰）；single seed 标注；caption 自包含。

输出（plots/）：
  fig_E1_perf_scatter_2026-08-20.png   性能空间散点（跨系列+权衡线）
  fig_E2_transform_trend_2026-08-20.png  变换规律 2-panel 折线
  fig_E3_loss_tradeoff_2026-08-20.png   loss 权衡散点
  fig_E4_block_compare_2026-08-20.png   block 三窗口分组柱
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3, INV_TARGET_TRANSFORMS
from scripts.b23_block_eval import block_predict

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
PLOT_DIR = os.path.join(ROOT, "plots")
TODAY = "2026-08-20"
NUM_TRAIN_W1 = 2838

# (checkpoint名, target_transform, 记值: 滚动MAE, 滚动峰误差, 0-50分层, 标签, 系列)
MODELS = [
    ("EXP-18-5b", "", 50.41, -26.7, 12.3, "mse baseline", "baseline"),
    ("EXP-20-4", "sqrt", 27.80, -64.6, 9.1, "sqrt", "transform"),
    ("EXP-21-1", "pow07", 22.50, -34.2, 11.4, "pow07", "transform"),
    ("EXP-21-2", "pow23", 22.13, -36.7, 11.4, "pow23", "transform"),
    ("EXP-21-3", "log1p", 22.63, -47.8, 8.7, "log1p", "transform"),
    ("EXP-20-3", "", 59.79, -2.2, 15.8, "wmse a=1.0", "loss"),
    ("EXP-22-2", "", 38.87, -9.7, 14.4, "wmse a=0.5", "loss"),
    ("EXP-22-3", "", 33.92, -6.2, 13.7, "wmse_th", "loss"),
    ("EXP-22-1", "", 124.02, -30.4, 11.0, "mae", "loss"),
    ("EXP-22-4", "", 102.42, +25.5, 18.7, "asym", "loss"),
]

W2W3 = [
    ("EXP-19-2", 2986, 132, 66.80, 27.28, 56.76),
    ("EXP-19-3", 3118, 71, 42.05, 23.75, 56.03),
]


def find_ckpt(name):
    hits = glob.glob(os.path.join(ROOT, f"checkpoints/*{name}_0/full_checkpoint.pth"))
    assert hits, f"ckpt not found {name}"
    return hits[0]


def rolling_mae_peak(ckpt_name, tt, nt, nr, seq_len_hint=None):
    """滚动推理 → (MAE, 峰误差 pred-true)（与 result.txt 记值一致）。"""
    model, ckpt_args = load_model(find_ckpt(ckpt_name))
    seq_len = seq_len_hint or getattr(ckpt_args, "seq_len", 96)
    _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, nt, 132, 'standard', tt)
    preds, trues = run_rolling_enc3(model, scaler, data_z, t0, seq_len, nr, tt)
    mae = float(np.mean(np.abs(preds - trues)))
    i = int(np.argmax(trues))
    peak_err = float(preds[i] - trues[i])
    return mae, peak_err


def bin005_mae(ckpt_name, tt, nt):
    """从 npy 重算 0-50 分层 MAE（物理空间）。"""
    from sklearn.preprocessing import StandardScaler
    rdir = glob.glob(os.path.join(ROOT, f"results/*{ckpt_name}_0"))[0]
    pred_z = np.load(os.path.join(rdir, "pred.npy"))
    true_z = np.load(os.path.join(rdir, "true.npy"))
    df = pd.read_csv(DATA_CSV)
    ssn_train = df["ssn"].values[:nt].reshape(-1, 1)
    if tt:
        fn = {"sqrt": np.sqrt, "pow07": lambda x: np.power(x, 0.7),
              "pow23": lambda x: np.power(x, 2.0 / 3.0), "log1p": np.log1p}[tt]
        ssn_train = fn(np.clip(ssn_train, 0, None))
    scaler = StandardScaler()
    scaler.fit(ssn_train)
    ssn_col = pred_z.shape[-1] - 1
    pred_phy = scaler.inverse_transform(pred_z[:, :, ssn_col].reshape(-1, 1)).reshape(pred_z.shape[:2])
    true_phy = scaler.inverse_transform(true_z[:, :, ssn_col].reshape(-1, 1)).reshape(true_z.shape[:2])
    if tt:
        inv = INV_TARGET_TRANSFORMS[tt]
        pred_phy, true_phy = inv(pred_phy), inv(true_phy)
    mask = (true_phy >= 0) & (true_phy < 50)
    return float(np.mean(np.abs(pred_phy[mask] - true_phy[mask]))) if mask.any() else 0.0


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    # ---- 数据重推理 + 对拍 ----
    R = []  # (label, series, roll_mae, peak_err, bin005)
    for ckpt_name, tt, exp_mae, exp_peak, exp_005, label, series in MODELS:
        mae, peak = rolling_mae_peak(ckpt_name, tt, NUM_TRAIN_W1, 148)
        b005 = bin005_mae(ckpt_name, tt, NUM_TRAIN_W1)
        ok = abs(mae - exp_mae) < 0.5 and abs(peak - exp_peak) < 0.5 and abs(b005 - exp_005) < 0.5
        print(f"{label}: MAE={mae:.2f}(记{exp_mae}) 峰={peak:+.1f}(记{exp_peak}) 0-50={b005:.1f}(记{exp_005}) {'PASS' if ok else 'FAIL'}")
        assert ok, f"{label} 对拍失败"
        R.append((label, series, mae, peak, b005))

    base_mae, base_peak, base_005 = R[0][2], R[0][3], R[0][4]

    # ---- E1 性能空间散点 ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"transform": "#1f5fbf", "loss": "#e08000", "baseline": "#444444"}
    markers = {"transform": "o", "loss": "^", "baseline": "*"}
    for label, series, mae, peak, b005 in R:
        x = (base_peak - peak) / abs(base_peak) * 100 if base_peak != 0 else 0  # 峰改善%
        y = (base_mae - mae) / base_mae * 100  # 滚动改善%
        ax.scatter(x, y, c=colors[series], marker=markers[series], s=70,
                   edgecolors="white", linewidths=0.5, zorder=5)
        dx, dy = 1.2, 1.2
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.set_xlabel("Rolling peak error improvement vs baseline (%)", fontsize=9.5)
    ax.set_ylabel("Rolling MAE improvement vs baseline (%)", fontsize=9.5)
    ax.set_title("E1: All experiments in the performance plane (W1, single seed n=1)", fontsize=10.5)
    ax.grid(color="#cccccc", alpha=0.3, lw=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"fig_E1_perf_scatter_{TODAY}.png"), dpi=300)
    plt.close(fig)
    print("E1 saved")

    # ---- E2 变换规律 2-panel ----
    t_labels = ["none", "sqrt", "pow23", "pow07", "log1p"]
    t_data = {l: (m, p) for l, s, m, p, b in R if s in ("baseline", "transform")}
    t_data = {k: t_data[k] for k in t_labels if k in t_data}
    xpos = np.arange(len(t_data))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(xpos, [t_data[k][0] for k in t_data], "o-", color="#1f5fbf", lw=1.2, ms=5)
    ax1.axhline(base_mae, color="#444444", lw=0.7, ls="--", label=f"baseline {base_mae:.1f}")
    ax1.set_xticks(xpos); ax1.set_xticklabels(list(t_data), fontsize=8, rotation=20)
    ax1.set_ylabel("Rolling MAE", fontsize=9); ax1.set_title("(a) Rolling MAE", fontsize=10)
    ax2.plot(xpos, [t_data[k][1] for k in t_data], "s-", color="#e08000", lw=1.2, ms=5)
    ax2.axhline(base_peak, color="#444444", lw=0.7, ls="--", label=f"baseline {base_peak:+.1f}")
    ax2.set_xticks(xpos); ax2.set_xticklabels(list(t_data), fontsize=8, rotation=20)
    ax2.set_ylabel("Rolling peak error (true-pred)", fontsize=9); ax2.set_title("(b) Rolling peak error", fontsize=10)
    for a in (ax1, ax2):
        a.grid(color="#cccccc", alpha=0.3, lw=0.4)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        a.legend(fontsize=8, frameon=False)
    fig.suptitle("E2: Target transform effect (W1, single seed n=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"fig_E2_transform_trend_{TODAY}.png"), dpi=300)
    plt.close(fig)
    print("E2 saved")

    # ---- E3 loss 权衡散点 ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, series, mae, peak, b005 in R:
        if series not in ("loss", "baseline"):
            continue
        x = (base_peak - peak) / abs(base_peak) * 100
        y = (b005 - base_005) / base_005 * 100  # 低值损伤% (正=恶化)
        c = "#444444" if series == "baseline" else "#e08000"
        ax.scatter(x, y, c=c, s=70, edgecolors="white", linewidths=0.5, zorder=5)
        ax.annotate(label, (x, y), xytext=(1.2, 1.2), textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color="#999999", lw=0.7); ax.axvline(0, color="#999999", lw=0.7)
    ax.set_xlabel("Rolling peak error improvement vs baseline (%)", fontsize=9.5)
    ax.set_ylabel("Low-value (SSN 0-50) MAE worsening vs baseline (%)", fontsize=9.5)
    ax.set_title("E3: Loss trade-off — peak gain vs low-value cost (W1, single seed n=1)", fontsize=10.5)
    ax.grid(color="#cccccc", alpha=0.3, lw=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"fig_E3_loss_tradeoff_{TODAY}.png"), dpi=300)
    plt.close(fig)
    print("E3 saved")

    # ---- E4 block 三窗口分组柱 ----
    w1_roll, w1_block, w1_overlap = base_mae, 21.29, 58.23
    groups = []
    for ckpt_name, nt, nr, exp_roll, exp_block, exp_over in W2W3:
        mae, _ = rolling_mae_peak(ckpt_name, "", nt, nr)
        model, ckpt_args = load_model(find_ckpt(ckpt_name))
        seq_len = getattr(ckpt_args, "seq_len", 96)
        _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, nt, 132)
        bp = block_predict(model, scaler, data_z, t0, seq_len, nr, 24, 0)
        trues = []
        for t in range(nr):
            trues.append(float(scaler.inverse_transform(data_z[t0 + t].reshape(1, -1))[0, 2]))
        trues = np.array(trues)
        b_mae = float(np.mean(np.abs(bp - trues)))
        bo = block_predict(model, scaler, data_z, t0, seq_len, nr, 24, 12)
        bo_mae = float(np.mean(np.abs(bo - trues)))
        ok = abs(mae - exp_roll) < 0.5 and abs(b_mae - exp_block) < 0.5 and abs(bo_mae - exp_over) < 0.5
        print(f"{ckpt_name}: roll={mae:.2f} block={b_mae:.2f} overlap={bo_mae:.2f} 记{exp_roll}/{exp_block}/{exp_over} {'PASS' if ok else 'FAIL'}")
        assert ok
        groups.append((ckpt_name, mae, b_mae, bo_mae))

    win_names = ["W1", "W2", "W3"]
    roll_vals = [w1_roll] + [g[1] for g in groups]
    block_vals = [w1_block] + [g[2] for g in groups]
    over_vals = [w1_overlap] + [g[3] for g in groups]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - w, roll_vals, w, color="#444444", label="Rolling")
    ax.bar(x, block_vals, w, color="#1f5fbf", label="Block 24 (no overlap)")
    ax.bar(x + w, over_vals, w, color="#e08000", label="Block 24 (overlap 12)")
    for xi, v in zip(x - w, roll_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x, block_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w, over_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(win_names, fontsize=10)
    ax.set_ylabel("Trajectory MAE", fontsize=9.5)
    ax.set_title("E4: Rolling vs block extrapolation (single seed n=1)", fontsize=10.5)
    ax.grid(axis="y", color="#cccccc", alpha=0.3, lw=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"fig_E4_block_compare_{TODAY}.png"), dpi=300)
    plt.close(fig)
    print("E4 saved")


if __name__ == "__main__":
    main()
