"""Plot five-seed mean peak/trough amplitude and timing errors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/cycle_holdout"
METHODS = ("p24_update", "p24_recursive", "p132_direct")
LABELS = {
    "p24_update": "PR24 update24",
    "p24_recursive": "PR24 recursive24",
    "p132_direct": "PR132 direct",
}
COLORS = {
    "p24_update": "#009E73",
    "p24_recursive": "#E69F00",
    "p132_direct": "#0072B2",
}


def main() -> None:
    frame = pd.read_csv(BASE / "cycle_holdout_aggregate.csv")
    cycles = ["SC20", "SC21", "SC24"]
    panels = [
        ("peak_amplitude_error_mean", "Peak amplitude error (SSN)", True),
        ("peak_time_error_months_mean", "Peak timing error (months)", False),
        ("trough_amplitude_error_mean", "Trough amplitude error (SSN)", True),
        ("trough_time_error_months_mean", "Trough timing error (months)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=230)
    width = 0.24
    x = np.arange(len(cycles))
    for ax, (column, title, show_std) in zip(axes.ravel(), panels):
        for offset, method in zip((-width, 0.0, width), METHODS):
            subset = frame[frame["method"] == method].set_index("cycle").reindex(cycles)
            values = subset[column].to_numpy(dtype=float)
            std_column = column.replace("_mean", "_std")
            errors = (
                subset[std_column].fillna(0).to_numpy(dtype=float)
                if show_std and std_column in subset
                else np.zeros_like(values)
            )
            ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=COLORS[method],
                alpha=0.85,
                label=LABELS[method],
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, cycles)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Chronological one-cycle holdouts | five-seed mean; error bars = seed std",
        fontsize=10,
    )
    fig.tight_layout()
    out = BASE / "cycle_holdout_extrema_errors_2026-11-21"
    fig.savefig(out.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
