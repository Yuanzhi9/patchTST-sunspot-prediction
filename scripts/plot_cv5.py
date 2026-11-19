"""Plot five outer-fold prediction curves and annotated extrema."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
BLACK = "#111111"
GRAY = "#B7B7B7"


def annotate_extrema(
    ax,
    records: list[dict],
    color: str,
    marker: str,
    y_levels: list[float],
    slot_offset: int,
) -> None:
    for i, rec in enumerate(records[:3]):
        x = pd.Timestamp(rec["date"])
        y = rec["true"]
        ax.scatter([x], [y], s=20, marker=marker, color=color, zorder=6)
        label = f"{rec['kind']} {rec['date']}\ntrue {rec['true']:.1f}, pred {rec['pred']:.1f}"
        ax.annotate(
            label,
            (x, y),
            xytext=(x, y_levels[(slot_offset + i) % len(y_levels)]),
            textcoords="data",
            fontsize=5.5,
            color=color,
            ha="left" if i % 2 == 0 else "right",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "lw": 0.4, "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BASE-2026-11-M CV5 curves")
    parser.add_argument("--des", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_glob", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pattern = args.results_glob or f"results/*_{args.des}_F*_seed{args.seed}/"
    dirs = [Path(p) for p in sorted(glob.glob(pattern)) if Path(p, "cv_metrics.json").exists()]
    if not dirs:
        raise SystemExit("no evaluated CV5 result directories; run scripts/eval_cv5.py first")

    rows = []
    for d in dirs:
        metrics = json.loads((d / "cv_metrics.json").read_text(encoding="utf-8"))
        curve = pd.read_csv(d / "cv_block_curve.csv", parse_dates=["date"])
        rows.append((metrics, curve))
    rows.sort(key=lambda item: item[0]["fold"])

    fig, axes = plt.subplots(
        len(rows), 1, figsize=(12, 2.65 * len(rows)), dpi=300, squeeze=False
    )
    for ax, (metrics, curve) in zip(axes[:, 0], rows):
        y_min = min(curve["true"].min(), curve["pred"].min())
        y_max = max(curve["true"].max(), curve["pred"].max())
        y_span = max(1.0, float(y_max - y_min))
        label_levels = [
            float(y_max + 0.16 * y_span),
            float(y_max + 0.30 * y_span),
            float(y_max + 0.44 * y_span),
        ]
        ax.set_ylim(float(y_min - 0.06 * y_span), float(y_max + 0.52 * y_span))
        ax.fill_between(
            curve["date"], curve["true"], curve["pred"], color=ORANGE, alpha=0.13, lw=0
        )
        ax.plot(curve["date"], curve["true"], color=BLACK, lw=1.25, label="Observed SSN")
        ax.plot(curve["date"], curve["pred"], color=BLUE, lw=1.15, ls="--", label="Predicted SSN")
        annotate_extrema(ax, metrics["peaks"], ORANGE, "^", label_levels, 0)
        annotate_extrema(ax, metrics["troughs"], BLUE, "v", label_levels, 1)
        ax.set_title(
            f"Fold {metrics['fold']} outer validation "
            f"{metrics['outer_val']['start_date']}–{metrics['outer_val']['end_date']} | "
            f"block MAE {metrics['block24_metrics']['mae']:.2f} SSN",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.suptitle(
        f"{args.des} | sequential blocked 5-fold | dataset through 2026-10 | seed {args.seed}",
        fontsize=10,
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        "Prediction curve uses non-overlapping 24-month model calls. Faint band is prediction error. "
        "Peak/trough labels use 13-month centered smoothing for display only.",
        fontsize=6,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    out = Path(args.out or f"plots/{args.des}_cv5_curves_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
