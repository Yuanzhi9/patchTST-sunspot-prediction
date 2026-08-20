"""
plot_figure_e7.py — E7 机制-对策概念框架图（2026-08-20）

期刊风格简洁框图：左侧 4 个峰值压制机制，右侧对应对策，标注验证状态。
无数据依赖。色盲友好（蓝/橙/灰）。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT, "plots")
TODAY = "2026-08-20"

MECHANISMS = [
    ("M1  MSE loss ledger:\nhigh-value months are\nfew, model trades peak\nfor low-value accuracy", "#444444"),
    ("M2  Right-skewed SSN:\nStandardScaler anchors\ncompress peak salience", "#444444"),
    ("M3  RevIN denorm:\nhypothesized peak\nsuppressor", "#444444"),
    ("M4  Skew amplifies\nrolling error\naccumulation", "#444444"),
]
SOLUTIONS = [
    ("S1  Weighted loss\n(wmse_th)\npeak err -6.2 (77%)\n[validated]", "#1f5fbf"),
    ("S2  Target transform\n(pow23)\nrolling 22.1 (56%)\n[validated]", "#1f5fbf"),
    ("S3  RevIN diagnosis\nrevin=0 collapses\n[refuted: stability pillar]", "#e08000"),
    ("S4  Block strategy\n(24-month feedback)\n44-59% gain\n[validated]", "#1f5fbf"),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    for i, ((mtext, mc), (stext, sc)) in enumerate(zip(MECHANISMS, SOLUTIONS)):
        y = 5.0 - i * 1.45
        # 机制框（左）
        mb = FancyBboxPatch((0.35, y - 0.55), 3.7, 1.1,
                            boxstyle="round,pad=0.08", fc="white", ec=mc, lw=1.3)
        ax.add_patch(mb)
        ax.text(2.2, y, mtext, ha="center", va="center", fontsize=8.3, color="#222222")
        # 对策框（右）
        sb = FancyBboxPatch((6.0, y - 0.55), 3.7, 1.1,
                            boxstyle="round,pad=0.08", fc=sc, ec=sc, lw=1.3, alpha=0.12)
        ax.add_patch(sb)
        ax.text(7.85, y, stext, ha="center", va="center", fontsize=8.3, color="#222222")
        # 箭头
        ar = FancyArrowPatch((4.15, y), (5.9, y), arrowstyle="-|>",
                             mutation_scale=13, color="#666666", lw=1.1)
        ax.add_patch(ar)

    ax.text(2.2, 6.05, "Peak suppression\nmechanisms (M1-M4)", ha="center",
            fontsize=10, weight="bold", color="#333333")
    ax.text(7.85, 6.05, "Countermeasures\n(status 2026-08)", ha="center",
            fontsize=10, weight="bold", color="#333333")
    ax.text(5.0, 0.12, "E7: Mechanisms and countermeasures of peak suppression "
                       "(all experiments single seed n=1; M3 refuted as cause, "
                       "kept as rolling-stability pillar)",
            ha="center", fontsize=8.5, color="#555555")

    out = os.path.join(PLOT_DIR, f"fig_E7_mechanism_map_{TODAY}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"E7 saved: {out}")


if __name__ == "__main__":
    main()
