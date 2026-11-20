"""Plot p132 direct-cycle curves, lead profile, and window-error distribution."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
BLACK = "#111111"
GRAY = "#777777"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot p132 CV5 outputs")
    parser.add_argument("--des", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_glob", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stability_out", default=None)
    args = parser.parse_args()

    pattern = args.results_glob or f"results/*_{args.des}_F*_seed{args.seed}/"
    dirs = [
        Path(path)
        for path in sorted(glob.glob(pattern))
        if Path(path, "p132_metrics.json").exists() and Path(path, "p132_curve.csv").exists()
    ]
    if not dirs:
        raise SystemExit("no evaluated p132 result directories")

    rows = []
    for path in dirs:
        metrics = json.loads((path / "p132_metrics.json").read_text(encoding="utf-8"))
        curve = pd.read_csv(path / "p132_curve.csv", parse_dates=["date"])
        rows.append((metrics, curve))
    rows.sort(key=lambda item: item[0]["fold"])

    fig, axes = plt.subplots(
        len(rows), 1, figsize=(13, 3.0 * len(rows)), dpi=300, squeeze=False
    )
    for ax, (metrics, curve) in zip(axes[:, 0], rows):
        y_min = min(curve["true"].min(), curve["pred"].min())
        y_max = max(curve["true"].max(), curve["pred"].max())
        span = max(1.0, float(y_max - y_min))
        ax.set_ylim(y_min - 0.07 * span, y_max + 0.10 * span)
        ax.fill_between(
            curve["date"], curve["true"], curve["pred"], color=ORANGE, alpha=0.13, lw=0
        )
        ax.plot(curve["date"], curve["true"], color=BLACK, lw=1.2, label="Observed SSN")
        ax.plot(curve["date"], curve["pred"], color=BLUE, lw=1.15, ls="--", label="Predicted SSN")
        ax.set_title(
            f"Fold {metrics['fold']} {metrics['outer_val']['start_date']}–"
            f"{metrics['outer_val']['end_date']} | direct-cycle MAE "
            f"{metrics['direct_cycle_metrics']['mae']:.2f} SSN",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

        peak = metrics["peak_summary"]
        trough = metrics["trough_summary"]
        text = (
            f"direct cycles: {metrics['n_direct_cycle_windows']}\n"
            f"all-step MAE: {metrics['full_window_metrics']['mae']:.2f}\n"
            f"peak |amp|: {peak['abs_amplitude_error_mean']:.1f}\n"
            f"peak |time|: {peak['abs_time_error_months_mean']:.1f} mo\n"
            f"trough |amp|: {trough['abs_amplitude_error_mean']:.1f}\n"
            f"trough |time|: {trough['abs_time_error_months_mean']:.1f} mo"
        )
        ax.text(
            1.012,
            0.96,
            text,
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="top",
            linespacing=1.45,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "lw": 0.5},
            clip_on=False,
        )

    fig.suptitle(
        f"{args.des} | direct 132-month predictions | dataset through 2026-10 | seed {args.seed}",
        fontsize=10,
        y=0.995,
    )
    fig.text(
        0.01,
        0.004,
        "Curve uses non-overlapping 132-month model calls. Peak/trough timing uses a 13-month "
        "centered smoother for display/evaluation only.",
        fontsize=6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.80, top=0.955, bottom=0.035, hspace=0.44)

    out = Path(args.out or f"plots/{args.des}_p132_direct_cycles_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

    lead = np.mean(np.asarray([metrics["lead_mae"] for metrics, _ in rows]), axis=0)
    window_errors = []
    labels = []
    for metrics, curve in rows:
        window_errors.append(curve.groupby("window_index").apply(
            lambda group: float(np.mean(np.abs(group["pred"] - group["true"]))),
            include_groups=False,
        ).to_numpy())
        labels.append(f"F{metrics['fold']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)
    x = np.arange(1, 133)
    ax1.plot(x, lead, color=BLUE, lw=1.3)
    ax1.set_xlabel("Forecast lead (months)")
    ax1.set_ylabel("Mean absolute error (SSN)")
    ax1.set_title("Lead-time error profile", loc="left", fontsize=9)
    ax1.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.boxplot(window_errors, tick_labels=labels, showfliers=False, patch_artist=True)
    ax2.set_xlabel("Outer fold")
    ax2.set_ylabel("Window MAE (SSN)")
    ax2.set_title("Non-overlapping direct-cycle window errors", loc="left", fontsize=9)
    ax2.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.suptitle(f"{args.des} | p132 stability | seed {args.seed}", fontsize=10)
    fig.tight_layout()

    stability_out = Path(
        args.stability_out or f"plots/{args.des}_p132_stability_seed{args.seed}.png"
    )
    fig.savefig(stability_out, facecolor="white", bbox_inches="tight")
    fig.savefig(stability_out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(f"saved: {stability_out}")


if __name__ == "__main__":
    main()
