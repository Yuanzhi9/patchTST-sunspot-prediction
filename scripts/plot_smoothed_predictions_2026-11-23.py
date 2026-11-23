"""Compare raw and 13-month smoothed PR24/R1-A trajectories."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

from eval_cycle_extrema import metric_block

ROOT = Path(__file__).resolve().parents[1]
R1A = ROOT / "analysis/r1a_constrained_peak"
FUTURE = ROOT / "analysis/pr24_future_forecast"
OUT = ROOT / "analysis/smoothed_comparison"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)


def smooth(values: pd.Series, window: int = 13) -> pd.Series:
    return values.rolling(window, center=True, min_periods=window).mean()


def historical_comparison() -> None:
    metric_rows = []
    profiles: dict[tuple[str, int], pd.DataFrame] = {}
    for cycle in CYCLES:
        for seed in SEEDS:
            frame = pd.read_csv(
                R1A / cycle / f"constrained_peak_seed{seed}.csv",
                parse_dates=["date"],
            )
            frame["true_smooth13"] = smooth(frame["true"])
            frame["baseline_smooth13"] = smooth(frame["baseline_pred"])
            frame["r1a_smooth13"] = smooth(frame["local_h24"])
            profiles[(cycle, seed)] = frame
            valid = frame[
                ["true_smooth13", "baseline_smooth13", "r1a_smooth13"]
            ].notna().all(axis=1)
            for method, column in (
                ("pr24", "baseline_smooth13"),
                ("r1a_h24", "r1a_smooth13"),
            ):
                metrics = metric_block(
                    frame.loc[valid, column].to_numpy(),
                    frame.loc[valid, "true_smooth13"].to_numpy(),
                )
                metric_rows.append(
                    {"cycle": cycle, "seed": seed, "method": method, **metrics}
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "r1a_smoothed_seed_metrics.csv", index=False)
    aggregate = (
        metrics.groupby(["cycle", "method"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    aggregate.to_csv(OUT / "r1a_smoothed_aggregate.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    for ax, cycle in zip(axes, CYCLES):
        selected = {
            seed: frame for (name, seed), frame in profiles.items() if name == cycle
        }
        reference = next(iter(selected.values()))
        dates = pd.to_datetime(reference["date"])
        base = np.vstack([frame["baseline_smooth13"].to_numpy() for frame in selected.values()])
        corrected = np.vstack([frame["r1a_smooth13"].to_numpy() for frame in selected.values()])
        ax.plot(dates, reference["true"], color="#AAAAAA", lw=0.8, alpha=0.65, label="Observed raw")
        ax.plot(dates, reference["true_smooth13"], color="black", lw=1.5, label="Observed 13m")
        ax.plot(dates, np.median(base, axis=0), color="#0072B2", lw=1.4, label="PR24 13m")
        ax.fill_between(
            dates,
            np.nanmin(base, axis=0),
            np.nanmax(base, axis=0),
            color="#0072B2",
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(dates, np.median(corrected, axis=0), color="#D55E00", lw=1.4, label="R1-A Oracle 13m")
        ax.fill_between(
            dates,
            np.nanmin(corrected, axis=0),
            np.nanmax(corrected, axis=0),
            color="#D55E00",
            alpha=0.12,
            linewidth=0,
        )
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("13-month smoothed SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "R1-A smoothed comparison | Oracle local amplitude correction | five-seed median and range",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1a_smoothed_profiles{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(aggregate.to_string(index=False))


def future_comparison() -> None:
    data = pd.read_csv(
        ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv",
        parse_dates=["date"],
    )
    origin = pd.Timestamp("2026-10-01")
    history = data.loc[data["date"] <= origin, ["date", "ssn"]].copy()
    observed_smoothed = history.copy()
    observed_smoothed["smooth13"] = smooth(observed_smoothed["ssn"])

    seed_frames = {}
    for seed in SEEDS:
        forecast = pd.read_csv(
            FUTURE / f"forecast_seed{seed}.csv", parse_dates=["date"]
        )
        combined = pd.concat(
            [
                history.rename(columns={"ssn": "pred"}),
                forecast[["date", "pred"]],
            ],
            ignore_index=True,
        ).sort_values("date")
        combined["smooth13"] = smooth(combined["pred"])
        seed_frames[seed] = combined.loc[combined["date"] > origin].copy()

    dates = seed_frames[SEEDS[0]]["date"]
    stack = np.vstack([seed_frames[seed]["smooth13"].to_numpy() for seed in SEEDS])
    median = np.nanmedian(stack, axis=0)
    q10 = np.nanquantile(stack, 0.1, axis=0)
    q90 = np.nanquantile(stack, 0.9, axis=0)

    extrema = pd.read_csv(FUTURE / "forecast_extrema_by_seed.csv")
    trough_month = extrema["cycle25_trough_date"].mode().iloc[0]
    peak_month = extrema["cycle26_peak_date"].mode().iloc[0]
    summary = {
        "origin": "2026-10",
        "smoothing": "centered 13-month on observed history plus each forecast path",
        "trough_date": trough_month,
        "trough_value": float(extrema["cycle25_trough_smoothed"].median()),
        "peak_date": peak_month,
        "peak_value": float(extrema["cycle26_peak_smoothed"].median()),
        "seed_peak_range": [
            float(extrema["cycle26_peak_smoothed"].min()),
            float(extrema["cycle26_peak_smoothed"].max()),
        ],
    }
    (OUT / "pr24_future_smoothed_extrema.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    observed_plot = observed_smoothed.loc[
        observed_smoothed["date"] >= "2010-01-01"
    ]
    ax.plot(
        observed_plot["date"],
        observed_plot["smooth13"],
        color="black",
        lw=1.6,
        label="Observed 13-month",
    )
    ax.plot(dates, median, color="#D55E00", lw=1.6, label="PR24 forecast 13-month median")
    ax.fill_between(
        dates,
        q10,
        q90,
        color="#D55E00",
        alpha=0.16,
        linewidth=0,
        label="Five-seed 10–90% range",
    )
    ax.axvline(origin, color="#0072B2", lw=0.8, ls="--", label="Forecast origin")
    for _, row in extrema.iterrows():
        ax.scatter(
            pd.Timestamp(f"{row['cycle25_trough_date']}-01"),
            row["cycle25_trough_smoothed"],
            color="#0072B2",
            s=15,
            alpha=0.65,
            zorder=3,
        )
        ax.scatter(
            pd.Timestamp(f"{row['cycle26_peak_date']}-01"),
            row["cycle26_peak_smoothed"],
            color="#009E73",
            s=15,
            alpha=0.65,
            zorder=3,
        )
    ax.text(
        0.55,
        0.20,
        (
            f"Median seed extrema\n"
            f"Trough: {summary['trough_date']}, {summary['trough_value']:.1f}\n"
            f"Peak: {summary['peak_date']}, {summary['peak_value']:.1f}\n"
            f"Peak range: {summary['seed_peak_range'][0]:.1f}–"
            f"{summary['seed_peak_range'][1]:.1f}"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="#333333",
    )
    ax.set_ylabel("13-month smoothed SSN")
    ax.set_xlabel("Date")
    ax.set_title(
        "PR24 forecast shown on the 13-month smoothed scale",
        loc="left",
        fontsize=11,
    )
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUT / f"pr24_future_smoothed_profile{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    historical_comparison()
    future_comparison()


if __name__ == "__main__":
    main()
