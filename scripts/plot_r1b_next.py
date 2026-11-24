"""Plot symmetric R1B-NEXT W0/Ridge/RF comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/r1b_next"
OUT = INPUT

ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
LABELS = {
    "A0": "Climatology median",
    "A1": "Ridge · log1p",
    "A2": "Ridge · raw",
    "A3": "Ridge · ratio",
    "A4": "RF · log1p",
    "A5": "RF · raw",
    "A6": "RF · ratio",
}
STYLES = {
    "A0": ("#222222", "--", 2.4),
    "A1": ("#0072B2", "-", 2.0),
    "A2": ("#56B4E9", "-", 2.0),
    "A3": ("#009E73", "-", 2.0),
    "A4": ("#D55E00", "-", 2.0),
    "A5": ("#C63D2F", "-", 2.0),
    "A6": ("#7A5195", "-", 2.0),
}


def main() -> None:
    predictions = pd.read_csv(INPUT / "r1b_next_predictions.csv")
    per_cycle = pd.read_csv(INPUT / "r1b_next_per_cycle.csv")
    summary = pd.read_csv(INPUT / "target_representation_summary.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9.2))

    for column, target_kind in enumerate(("raw", "smooth")):
        ax = axes[0, column]
        subset = predictions.loc[predictions["target_kind"] == target_kind]
        truth = (
            subset.groupby("target_cycle")["true"].first().sort_index()
        )
        ax.plot(
            truth.index,
            truth.to_numpy(),
            color="black",
            marker="o",
            linewidth=2.5,
            markersize=6,
            label="Observed",
            zorder=10,
        )
        for cycle, value in truth.items():
            ax.annotate(
                f"{value:.1f}",
                (cycle, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                zorder=12,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
            )
        for candidate_id in ORDER:
            group = subset.loc[subset["candidate_id"] == candidate_id]
            if group.empty:
                continue
            values = group.groupby("target_cycle")["predicted"].median()
            color, linestyle, linewidth = STYLES[candidate_id]
            ax.plot(
                values.index,
                values.to_numpy(),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker="o",
                markersize=3.5,
                label=LABELS[candidate_id],
            )
        title_kind = "Raw monthly peak" if target_kind == "raw" else "13m smoothed peak"
        ax.set_title(f"{'A' if column == 0 else 'B'}. {title_kind}")
        ax.set_xlabel("Target solar cycle")
        ax.set_ylabel("Peak amplitude (SSN)")
        ax.set_xticks(sorted(truth.index))
        ax.set_ylim(
            min(subset["predicted"].min(), truth.min()) - 25,
            max(subset["predicted"].max(), truth.max()) + 25,
        )
        ax.legend(frameon=False, ncol=2, fontsize=7, loc="best")

        ax = axes[1, column]
        matrix = (
            per_cycle.loc[per_cycle["target_kind"] == target_kind]
            .pivot(
                index="candidate_id",
                columns="target_cycle",
                values="median_absolute_error",
            )
            .reindex(ORDER)
        )
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
        )
        ax.set_xticks(
            np.arange(len(matrix.columns)),
            [str(int(value)) for value in matrix.columns],
        )
        ax.set_yticks(
            np.arange(len(matrix.index)),
            [LABELS[candidate] for candidate in matrix.index],
        )
        for row in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row, column_index]
                ax.text(
                    column_index,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if value < np.nanmax(matrix.to_numpy()) * 0.65 else "white",
                )
        ax.set_title(
            f"{'C' if column == 0 else 'D'}. Per-cycle absolute error"
        )
        ax.set_xlabel("Target solar cycle")
        ax.grid(False)
        colorbar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("MAE (SSN)")

    best = (
        summary.sort_values(["target_kind", "mae"])
        .groupby("target_kind", as_index=False)
        .first()
    )
    caption = " | ".join(
        f"{row.target_kind}: lowest descriptive MAE {LABELS[row.candidate_id]} "
        f"{row.mae:.2f}"
        for row in best.itertuples(index=False)
    )
    figure.suptitle(
        "R1B-NEXT W0 comparison — retrospective design, no preferred candidate",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.01,
        (
            "All A0–A6 candidates are reported symmetrically. "
            "Lowest MAE is descriptive, not a model-selection decision. "
            + caption
        ),
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.95))
    figure.savefig(OUT / "r1b_next_comparison.png", dpi=220)
    figure.savefig(OUT / "r1b_next_comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
