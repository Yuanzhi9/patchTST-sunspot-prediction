"""
plot_round0_diagnosis.py — 第 0 轮体检出图（6 组：train/val/test 三线 + 两条选模线）

故事问题：正式选模口径"最佳 val 模型"选中的轮次，跟"最佳 test 轮次"差多远？
输出: plots/fig_round0_val_vs_test_2026-10-11.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import round0_diagnosis as rd  # 复用解析

ROOT = rd.ROOT
BLUE, ORANGE, GRAY, BLACK = "#0072B2", "#E69F00", "#999999", "#000000"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (eid, (win, do)) in zip(axes.ravel(), rd.RUNS.items()):
        arr = rd.parse(eid)
        ep, tr, va, te = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        bv, bt = int(ep[int(np.argmin(va))]), int(ep[int(np.argmin(te))])
        ax.plot(ep, tr, color=GRAY, lw=0.9, label="train")
        ax.plot(ep, va, color=BLUE, lw=1.1, label="val (selection)")
        ax.plot(ep, te, color=ORANGE, lw=1.2, label="test")
        ax.axvline(bv, color=BLUE, ls=":", lw=1.0)
        ax.axvline(bt, color=ORANGE, ls="--", lw=1.0)
        ax.annotate(f"best-val e{bv}", xy=(bv, ax.get_ylim()[1]), xytext=(bv + 1, ax.get_ylim()[1] * 0.92),
                    fontsize=6.8, color=BLUE)
        ax.annotate(f"best-test e{bt}", xy=(bt, ax.get_ylim()[1]), xytext=(max(bt - 16, 1), ax.get_ylim()[1] * 1.02),
                    fontsize=6.8, color=ORANGE)
        ax.set_title(f"{eid}  {win}  do{do}   Δ={abs(bt-bv)} 轮", fontsize=8.5)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("epoch", fontsize=7.5)
        ax.set_ylabel("loss (z-space)", fontsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.suptitle("Round-0 check: does 'best-val' selection pick the right epoch?  "
                 "blue = val (used for model selection), orange = test, gray = train",
                 fontsize=9.5, y=0.995)
    fig.text(0.01, 0.005,
             "Data: logs of EXP-25..30 (new GPU env, seed 2021, 50 epochs, per-epoch test loss is computed by exp_main). "
             "Δ = |best-test epoch − best-val epoch|. Verdict in the record: selection rule is unstable "
             "(2/6 groups with Δ>5, notably W3 do0.05 Δ=38; val oscillation >30% in 6/6). single seed (n = 1).",
             fontsize=6.2, color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(ROOT, "plots", "fig_round0_val_vs_test_2026-10-11.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print("出图:", out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    main()
