"""Plot separate per-cycle PR24 baseline and multitask curve comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_BASE = ROOT / "analysis/cycle_holdout"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)


def load_seed_curves(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    truth = None
    for seed, path in zip(SEEDS, paths):
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame[["date", "pred", "true"]].sort_values("date").reset_index(drop=True)
        pred_frames.append(
            frame[["date", "pred"]].rename(columns={"pred": f"seed{seed}"})
        )
        current_truth = frame[["date", "true"]]
        if truth is None:
            truth = current_truth
        elif not np.allclose(
            truth["true"].to_numpy(), current_truth["true"].to_numpy()
        ):
            raise RuntimeError(f"seed truths differ in {path}")
    wide = pred_frames[0]
    for frame in pred_frames[1:]:
        wide = wide.merge(frame, on="date", how="inner")
    pred_columns = [column for column in wide.columns if column.startswith("seed")]
    values = wide[pred_columns].to_numpy(dtype=float)
    summary = pd.DataFrame(
        {
            "date": wide["date"],
            "median": np.median(values, axis=1),
            "low": np.min(values, axis=1),
            "high": np.max(values, axis=1),
        }
    )
    return summary, truth, wide


def add_extrema(ax: plt.Axes, dates: pd.Series, values: pd.Series, color: str) -> None:
    smoothed = values.rolling(13, center=True, min_periods=13).mean()
    if not smoothed.notna().any():
        return
    peak = int(smoothed.idxmax())
    trough = int(smoothed.idxmin())
    for index, label in ((peak, "P"), (trough, "T")):
        ax.scatter(dates.iloc[index], values.iloc[index], color=color, s=22, zorder=5)
        ax.annotate(
            f"{label} {dates.iloc[index]:%Y-%m}",
            (dates.iloc[index], values.iloc[index]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def plot_one(
    cycle: str,
    method: str,
    multitask_base: Path,
    output_dir: Path,
    variant_label: str,
) -> None:
    if method == "recursive":
        baseline_files = [
            BASELINE_BASE / cycle / f"p24_update_seed{seed}/p24_recursive_curve.csv"
            for seed in SEEDS
        ]
        multitask_files = [
            multitask_base
            / cycle
            / f"p24_update_seed{seed}/p24_recursive_multitask_curve.csv"
            for seed in SEEDS
        ]
        method_label = "Recursive block24"
    elif method == "update":
        baseline_files = [
            BASELINE_BASE / cycle / f"p24_update_seed{seed}/p24_update_curve.csv"
            for seed in SEEDS
        ]
        multitask_files = [
            multitask_base / cycle / f"p24_update_seed{seed}/p24_update_curve.csv"
            for seed in SEEDS
        ]
        method_label = "Observed update"
    else:
        raise ValueError(method)

    baseline, truth, _ = load_seed_curves(baseline_files)
    multitask, mt_truth, _ = load_seed_curves(multitask_files)
    if not np.allclose(truth["true"].to_numpy(), mt_truth["true"].to_numpy()):
        raise RuntimeError(f"{cycle} {method}: baseline and multitask truth differ")

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=230)
    ax.plot(truth["date"], truth["true"], color="black", lw=1.45, label="Observed")
    for label, band, color in (
        ("PR24 baseline", baseline, "#0072B2"),
        (f"PR24 multitask {variant_label}", multitask, "#D55E00"),
    ):
        ax.fill_between(
            band["date"],
            band["low"],
            band["high"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(band["date"], band["median"], color=color, lw=1.55, label=f"{label} median")
        add_extrema(ax, band["date"], band["median"], color)

    ax.set_title(
        f"{cycle} | {method_label} | five-seed median; band = seed range",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel("Monthly SSN")
    ax.set_xlabel("Date")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{cycle}_{method}"
    fig.savefig(stem.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multitask_base",
        type=Path,
        default=ROOT / "analysis/pr24_multitask_cycle_holdout",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "analysis/pr24_multitask_figures/fixed_w01",
    )
    parser.add_argument("--variant_label", default="w=0.1")
    args = parser.parse_args()

    for cycle in CYCLES:
        for method in ("recursive", "update"):
            plot_one(
                cycle,
                method,
                args.multitask_base,
                args.output_dir,
                args.variant_label,
            )


if __name__ == "__main__":
    main()
