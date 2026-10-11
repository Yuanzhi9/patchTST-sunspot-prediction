"""
plot_block_vs_roll_newenv.py — EXP-23-R2 图（新环境 block24 vs 滚动，do0.2）

故事问题：同一个 checkpoint，"逐月回填"和"每 24 月回填一次"差在哪、差多少？

数据源=三个 do0.2 checkpoint 重新推理（EXP-26 / EXP-28 / EXP-30，均在 GPU-dropout-abl 相关 worktree）。
门禁（图规铁律 2）：图内 MAE 必须与记录值逐位一致，不一致直接报错退出。
记录值来源：EXPERIMENT_block-crosenv_2026-10-11.md / result.txt EXP-23-R2 行。

用法: python3 scripts/plot_block_vs_roll_newenv.py
输出: plots/fig_block_vs_roll_newenv_2026-10-11.png
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
from b23_block_eval import block_predict, inverse_phy  # noqa: E402  (需在 scripts/ 内 import)

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")

BLUE, ORANGE, GRAY, BLACK = "#0072B2", "#E69F00", "#999999", "#000000"

# (窗口名, 工作区根, checkpoint 实验目录名, num_train, n_roll, 记录值: 全步/滚动/block/预测峰block)
WINDOWS = [
    ("W1  1996-08 ~ 2008-11  (Cycle 23)", "/output/GPU-exp26",
     "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-26_0",
     2838, 148, (22.87, 50.29, 23.55)),
    ("W2  2008-12 ~ 2019-11  (Cycle 24)", ROOT,
     "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-28_0",
     2986, 132, (20.69, 47.95, 28.44)),
    ("W3  2019-12 ~ 2025-10  (Cycle 25)", ROOT,
     "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-30_0",
     3118, 71, (20.25, 43.65, 22.17)),
]


def mae(p, t):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(t))))


def main():
    df = pd.read_csv(DATA_CSV)
    dates = df["date"].astype(str).str[:7].values

    results = []
    for name, ws, ckpt_dir, num_train, n_roll, rec in WINDOWS:
        ckpt = os.path.join(ws, "checkpoints", ckpt_dir, "full_checkpoint.pth")
        model, ck_args = load_model(ckpt)
        seq_len = int(getattr(ck_args, "seq_len", 336))
        _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, num_train, 132, "standard", "")

        r_pred, r_true = run_rolling_enc3(model, scaler, data_z, t0, seq_len, n_roll, "")
        b_pred = block_predict(model, scaler, data_z, t0, seq_len, n_roll, 24, 0)
        assert len(b_pred) == len(r_true) == n_roll, f"{name}: 长度不一致"

        m_roll, m_block = mae(r_pred, r_true), mae(b_pred, r_true)

        # --- 门禁：与记录值逐位一致（保留 2 位）---
        ok = (round(m_roll, 2) == rec[1]) and (round(m_block, 2) == rec[2])
        print(f"{name}: 滚动 {m_roll:.2f}（记录 {rec[1]}）  block24 {m_block:.2f}（记录 {rec[2]}） "
              f"→ {'对拍 PASS' if ok else '对拍 FAIL'}")
        if not ok:
            raise SystemExit("[门禁失败] 图内数值与记录值不一致，禁止出图")

        x = dates[t0:t0 + n_roll]
        results.append(dict(name=name, x=x, true=r_true, roll=r_pred, block=b_pred,
                            m_roll=m_roll, m_block=m_block, m_full=rec[0], n=n_roll))
        print(f"    前 12 月 真实 {np.round(r_true[:12]).astype(int).tolist()} "
              f"| 滚动 {np.round(r_pred[:12]).astype(int).tolist()} "
              f"| block {np.round(b_pred[:12]).astype(int).tolist()}")

    # ---------------- 绘图 ----------------
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9,
                         "axes.titlesize": 9, "legend.fontsize": 8.5})
    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.52, wspace=0.24,
                          left=0.06, right=0.985, top=0.835, bottom=0.165)
    handles = None

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        h1, = ax.plot(r["x"], r["true"], color=BLACK, lw=1.5, label="observation")
        h2, = ax.plot(r["x"], r["roll"], color=BLUE, lw=1.1, ls="--",
                      label="rolling — refill every 1 month")
        h3, = ax.plot(r["x"], r["block"], color=ORANGE, lw=1.4,
                      label="block24 — refill every 24 months")
        handles = handles or [h1, h2, h3]
        ax.set_ylim(bottom=min(-20.0, 1.06 * min(r["true"].min(), r["roll"].min(), r["block"].min())))
        pk = int(np.argmax(r["true"]))
        ax.axvline(pk, color=GRAY, lw=0.8, ls=":")
        ax.annotate(f"observed peak {r['true'][pk]:.0f}  ({r['x'][pk]})", xy=(pk, r["true"][pk]),
                    xytext=(0.30, 0.95), textcoords="axes fraction", fontsize=7.5, color="#444444",
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
        ax.set_title(f"{r['name']}\nMAE  full {r['m_full']:.2f}  |  block {r['m_block']:.2f}"
                     f"  |  roll {r['m_roll']:.2f}", pad=5, linespacing=1.5, fontsize=8.2)
        ax.set_ylabel("SSN (monthly mean)")
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144])
        ax.set_xticklabels([r["x"][k][:4] if k < len(r["x"]) else "" for k in
                            [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]], fontsize=7)
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.985, 0.03, "single seed (n = 1)", transform=ax.transAxes,
                fontsize=6.5, color="#666666", ha="right")

    # ---- 下排：分组柱（三把尺子）+ 峰值幅度 ----
    axb = fig.add_subplot(gs[1, :2])
    w = 0.26
    idx = np.arange(3)
    axb.bar(idx - w, [r["m_full"] for r in results], w, color=GRAY,
            label="full-step (no self-feeding)")
    axb.bar(idx, [r["m_block"] for r in results], w, color=ORANGE, label="block24")
    axb.bar(idx + w, [r["m_roll"] for r in results], w, color=BLUE, label="rolling")
    axb.set_ylim(0, 64)
    for j, r in enumerate(results):
        for off, v in ((-w, r["m_full"]), (0, r["m_block"]), (w, r["m_roll"])):
            axb.text(j + off, v + 0.8, f"{v:.2f}", ha="center", fontsize=7.2)
    axb.set_xticks(idx)
    axb.set_xticklabels([r["name"].split()[0] for r in results])
    axb.set_ylabel("physical MAE over the whole test period")
    axb.set_title("block24 beats rolling in every window; self-feeding is the cost", pad=6)
    axb.grid(axis="y", color="#DDDDDD", lw=0.6)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.legend(frameon=False, ncol=1, loc="upper right", fontsize=7.5)

    axp = fig.add_subplot(gs[1, 2])
    peak_true = [float(np.max(r["true"])) for r in results]
    peak_pred = [float(np.max(r["block"])) for r in results]
    axp.bar(idx - 0.19, peak_true, 0.36, color=BLACK, label="observed peak")
    axp.bar(idx + 0.19, peak_pred, 0.36, color=ORANGE, label="block24 predicted max")
    axp.set_ylim(0, 300)
    for j in range(3):
        axp.text(j - 0.19, peak_true[j] + 4, f"{peak_true[j]:.0f}", ha="center", fontsize=7.2)
        axp.text(j + 0.19, peak_pred[j] + 4, f"{peak_pred[j]:.0f}", ha="center", fontsize=7.2)
    axp.set_xticks(idx)
    axp.set_xticklabels([r["name"].split()[0] for r in results])
    axp.set_ylabel("SSN")
    axp.set_title("peak amplitude", pad=6)
    axp.grid(axis="y", color="#DDDDDD", lw=0.6)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    axp.legend(frameon=False, ncol=1, loc="upper center", fontsize=7.5)

    fig.suptitle("Block vs rolling refill — new GPU environment (A6000 / torch 2.8), dropout 0.2, best-val checkpoint",
                 fontsize=10.5, y=0.975)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.53, 0.943), fontsize=8.5)
    fig.text(0.06, 0.055,
             "Data: SILSO monthly-mean SSN (unsmoothed), 1749-02 ~ 2025-10; each window trained with its own "
             "train/val split (W1 nt=2838 / W2 nt=2986 / W3 nt=3118, num_val=132), evaluated over the whole test\n"
             "period (148 / 132 / 71 months). Rolling: predict 24 months, keep step 0, append it, slide 1 month -> 1 refill per\n"
             "month. block24: keep all 24 predicted months as a chunk, slide 24 -> 1 refill per 24 months. MAE = mean|pred - truth|.\n"
             "single seed = 2021 (n = 1); MAE values gates-checked against result.txt (EXP-23-R2). Right panel: "
             "block24 predicted max = argmax of the predicted trajectory (not the value at the observed peak month).",
             fontsize=6.6, color="#444444", ha="left", va="bottom")

    out = os.path.join(ROOT, "plots", "fig_block_vs_roll_newenv_2026-10-11.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"\n出图: {out}")


if __name__ == "__main__":
    main()
