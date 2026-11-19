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


def mark_extrema(ax, records: list[dict], color: str, marker: str) -> None:
    for rec in records[:2]:
        x = pd.Timestamp(rec["date"])
        y = rec["true"]
        ax.scatter([x], [y], s=20, marker=marker, color=color, zorder=6)


def add_extrema_box(ax, peaks: list[dict], troughs: list[dict]) -> None:
    lines = ["Major extrema (true / pred)"]
    for rec in peaks[:2]:
        lines.append(
            f"P {rec['date']}  {rec['true']:.1f} / {rec['pred']:.1f}"
        )
    for rec in troughs[:2]:
        lines.append(
            f"T {rec['date']}  {rec['true']:.1f} / {rec['pred']:.1f}"
        )
    ax.text(
        1.012,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=5.8,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#BBBBBB", "lw": 0.5},
        clip_on=False,
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
        ax.set_ylim(float(y_min - 0.07 * y_span), float(y_max + 0.10 * y_span))
        ax.fill_between(
            curve["date"], curve["true"], curve["pred"], color=ORANGE, alpha=0.13, lw=0
        )
        ax.plot(curve["date"], curve["true"], color=BLACK, lw=1.25, label="Observed SSN")
        ax.plot(curve["date"], curve["pred"], color=BLUE, lw=1.15, ls="--", label="Predicted SSN")
        mark_extrema(ax, metrics["peaks"], ORANGE, "^")
        mark_extrema(ax, metrics["troughs"], BLUE, "v")
        add_extrema_box(ax, metrics["peaks"], metrics["troughs"])
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
    fig.subplots_adjust(left=0.055, right=0.80, top=0.95, bottom=0.035, hspace=0.48)

    out = Path(args.out or f"plots/{args.des}_cv5_curves_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
