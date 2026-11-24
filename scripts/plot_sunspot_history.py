"""Plot the complete observed sunspot series and cycle-aligned contours."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
OUT = ROOT / "analysis/sunspot_history"


def load_data():
    frame = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    frame["smooth13"] = frame["ssn"].rolling(
        13, center=True, min_periods=13
    ).mean()
    boundaries = pd.read_csv(
        BOUNDARIES,
        parse_dates=["min_date", "peak_date", "next_min_date"],
    )
    return frame, boundaries


def plot_full_history(frame: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 9.2),
        gridspec_kw={"height_ratios": [1.5, 1]},
    )
    for index, cycle in enumerate(boundaries.itertuples(index=False)):
        if index % 2 == 0:
            end = (
                cycle.next_min_date
                if pd.notna(cycle.next_min_date)
                else frame["date"].max()
            )
            axes[0].axvspan(
                cycle.min_date, end, color="#F3F5F7", zorder=0
            )

    axes[0].plot(
        frame["date"],
        frame["ssn"],
        color="#6A9BC3",
        linewidth=0.75,
        alpha=0.55,
        label="Monthly raw SSN",
    )
    axes[0].plot(
        frame["date"],
        frame["smooth13"],
        color="#111111",
        linewidth=1.45,
        label="13-month centered mean",
    )
    axes[0].scatter(
        boundaries["peak_date"],
        boundaries["peak_smooth"],
        color="#D55E00",
        s=18,
        zorder=5,
        label="Cycle peak",
    )
    for cycle in boundaries.itertuples(index=False):
        axes[0].text(
            cycle.peak_date,
            cycle.peak_smooth + 16,
            f"{int(cycle.cycle)}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#7A3E00",
        )
    axes[0].set_xlim(frame["date"].min(), frame["date"].max())
    axes[0].set_ylim(0, 430)
    axes[0].set_ylabel("Sunspot number")
    axes[0].set_title(
        "Complete observed monthly sunspot record, 1749–2026"
    )
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    recent_start = pd.Timestamp("1954-01-01")
    recent = frame.loc[frame["date"] >= recent_start]
    recent_boundaries = boundaries.loc[
        boundaries["next_min_date"].fillna(pd.Timestamp("2100-01-01"))
        >= recent_start
    ]
    axes[1].plot(
        recent["date"],
        recent["ssn"],
        color="#6A9BC3",
        linewidth=0.85,
        alpha=0.55,
        label="Monthly raw",
    )
    axes[1].plot(
        recent["date"],
        recent["smooth13"],
        color="#111111",
        linewidth=1.6,
        label="13-month centered mean",
    )
    for cycle in recent_boundaries.itertuples(index=False):
        axes[1].axvline(
            cycle.min_date,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        axes[1].text(
            cycle.min_date,
            370,
            f"SC{int(cycle.cycle)}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
        )
        axes[1].scatter(
            cycle.peak_date,
            cycle.peak_smooth,
            color="#D55E00",
            s=28,
            zorder=5,
        )
    axes[1].set_xlim(recent_start, frame["date"].max())
    axes[1].set_ylim(0, 390)
    axes[1].set_ylabel("Sunspot number")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Recent cycles SC19–SC25: monthly contour")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    for axis in axes:
        axis.xaxis.set_major_locator(mdates.YearLocator(20))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(OUT / "sunspot_full_history.png", dpi=220)
    figure.savefig(OUT / "sunspot_full_history.pdf")
    plt.close(figure)


def cycle_slice(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return frame.loc[(frame["date"] >= start) & (frame["date"] < end)].copy()


def plot_cycle_contours(
    frame: pd.DataFrame, boundaries: pd.DataFrame
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=1, vmax=int(boundaries["cycle"].max()))

    for column, field in enumerate(("ssn", "smooth13")):
        axis = axes[column]
        for cycle in boundaries.itertuples(index=False):
            end = (
                cycle.next_min_date
                if pd.notna(cycle.next_min_date)
                else frame["date"].max() + pd.DateOffset(months=1)
            )
            segment = cycle_slice(frame, cycle.min_date, end)
            if segment.empty:
                continue
            months = np.arange(len(segment))
            highlight = int(cycle.cycle) >= 19
            axis.plot(
                months,
                segment[field],
                color=cmap(norm(cycle.cycle)),
                linewidth=2.0 if highlight else 0.9,
                alpha=0.95 if highlight else 0.5,
            )
            if highlight:
                axis.text(
                    months[-1] + 1,
                    segment[field].iloc[-1],
                    f"SC{int(cycle.cycle)}",
                    fontsize=7,
                    color=cmap(norm(cycle.cycle)),
                    va="center",
                )
        axis.set_title(
            "Raw monthly contour by cycle"
            if field == "ssn"
            else "13-month smoothed contour by cycle"
        )
        axis.set_xlabel("Months since cycle minimum")
        axis.set_ylabel("Sunspot number")
        axis.set_xlim(0, 160)
        axis.set_ylim(0, 410 if field == "ssn" else 330)

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Solar cycle number")
    figure.suptitle(
        "Cycle-aligned sunspot contours, SC1–SC25",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.98, 0.95))
    figure.savefig(OUT / "sunspot_cycle_contours.png", dpi=220)
    figure.savefig(OUT / "sunspot_cycle_contours.pdf")
    plt.close(figure)


def plot_recent_cycle_contours(
    frame: pd.DataFrame, boundaries: pd.DataFrame
) -> None:
    recent = boundaries.loc[boundaries["cycle"] >= 19].copy()
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    for column, field in enumerate(("ssn", "smooth13")):
        axis = axes[column]
        for position, cycle in enumerate(recent.itertuples(index=False)):
            end = (
                cycle.next_min_date
                if pd.notna(cycle.next_min_date)
                else frame["date"].max() + pd.DateOffset(months=1)
            )
            segment = cycle_slice(frame, cycle.min_date, end)
            months = np.arange(len(segment))
            color = colors(position % 10)
            axis.plot(
                months,
                segment[field],
                color=color,
                linewidth=2.1 if field == "smooth13" else 1.15,
                alpha=0.95 if field == "smooth13" else 0.42,
                label=f"SC{int(cycle.cycle)}",
            )
            peak_offset = int(
                (pd.Timestamp(cycle.peak_date) - pd.Timestamp(cycle.min_date))
                .days
                / 30.4375
            )
            peak_value = (
                cycle.peak_raw if field == "ssn" else cycle.peak_smooth
            )
            axis.text(
                peak_offset,
                peak_value + 8,
                f"SC{int(cycle.cycle)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
        axis.set_title(
            "Recent raw monthly contours"
            if field == "ssn"
            else "Recent 13-month smoothed contours"
        )
        axis.set_xlabel("Months since cycle minimum")
        axis.set_ylabel("Sunspot number")
        axis.set_xlim(0, 155)
        axis.set_ylim(0, 400 if field == "ssn" else 320)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle(
        "Recent cycle contours, SC19–SC25",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    figure.savefig(OUT / "sunspot_recent_cycle_contours.png", dpi=220)
    figure.savefig(OUT / "sunspot_recent_cycle_contours.pdf")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, boundaries = load_data()
    plot_full_history(frame, boundaries)
    plot_cycle_contours(frame, boundaries)
    plot_recent_cycle_contours(frame, boundaries)


if __name__ == "__main__":
    main()
