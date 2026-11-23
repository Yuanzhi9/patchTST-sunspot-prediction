"""Audit raw versus 13-month-smoothed cycle peaks for R1 experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/r1b_constrained_peak_predicted"
OUT = ROOT / "analysis/peak_definition_comparison"
CYCLES = ("SC20", "SC21", "SC24")
METHODS = ("true", "pr24", "oracle", "ridge", "random_forest")
SEEDS = (2019, 2020, 2021, 2022, 2023)


def peak_record(dates: pd.Series, values: pd.Series) -> tuple[pd.Timestamp, float]:
    index = int(values.idxmax())
    return dates.iloc[index], float(values.iloc[index])


def month_delta(pred: pd.Timestamp, true: pd.Timestamp) -> int:
    return int(pred.to_period("M").ordinal - true.to_period("M").ordinal)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cycle in CYCLES:
        for seed in SEEDS:
            frame = pd.read_csv(
                SOURCE / f"{cycle}_seed{seed}.csv", parse_dates=["date"]
            )
            true_raw_date, true_raw = peak_record(frame["date"], frame["true"])
            true_smooth = frame["true"].rolling(13, center=True, min_periods=13).mean()
            true_smooth_date, true_smooth_value = peak_record(frame["date"], true_smooth)
            for method, column in (
                ("pr24", "pred_pr24"),
                ("oracle", "pred_oracle"),
                ("ridge", "pred_ridge"),
                ("random_forest", "pred_random_forest"),
            ):
                values = frame[column]
                raw_date, raw_value = peak_record(frame["date"], values)
                smooth = values.rolling(13, center=True, min_periods=13).mean()
                smooth_date, smooth_value = peak_record(frame["date"], smooth)
                rows.append(
                    {
                        "cycle": cycle,
                        "seed": seed,
                        "method": method,
                        "true_raw_date": true_raw_date.strftime("%Y-%m"),
                        "true_raw_peak": true_raw,
                        "raw_peak_date": raw_date.strftime("%Y-%m"),
                        "raw_peak": raw_value,
                        "raw_peak_error": raw_value - true_raw,
                        "raw_peak_time_error_months": month_delta(raw_date, true_raw_date),
                        "true_smooth_date": true_smooth_date.strftime("%Y-%m"),
                        "true_smooth_peak": true_smooth_value,
                        "smooth_peak_date": smooth_date.strftime("%Y-%m"),
                        "smooth_peak": smooth_value,
                        "smooth_peak_error": smooth_value - true_smooth_value,
                        "smooth_peak_time_error_months": month_delta(
                            smooth_date, true_smooth_date
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "peak_definition_seed_detail.csv", index=False)
    numeric = [
        "true_raw_peak",
        "raw_peak",
        "raw_peak_error",
        "raw_peak_time_error_months",
        "true_smooth_peak",
        "smooth_peak",
        "smooth_peak_error",
        "smooth_peak_time_error_months",
    ]
    aggregate = (
        detail.groupby(["cycle", "method"], as_index=False)
        .agg(
            **{f"{column}_mean": (column, "mean") for column in numeric},
            **{f"{column}_std": (column, "std") for column in numeric},
            seeds=("seed", "nunique"),
        )
        .sort_values(["cycle", "method"])
    )
    aggregate.to_csv(OUT / "peak_definition_aggregate.csv", index=False)
    (OUT / "peak_definition_summary.json").write_text(
        json.dumps(
            {
                "conclusion": (
                    "Raw monthly maximum and 13-month centered maximum are both "
                    "formal targets. Ridge is close to SC24 only under the raw "
                    "definition; the same prediction is far above truth under the "
                    "13-month definition. The two tracks must be reported separately."
                ),
                "reporting_contract": (
                    "dual track: raw monthly maximum and 13-month centered maximum"
                ),
                "raw_definition": "raw monthly maximum",
                "smooth_definition": "13-month centered smoothed maximum",
                "aggregate": aggregate.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), dpi=300, sharex=True)
    colors = {
        "true": "#555555",
        "pr24": "#0072B2",
        "oracle": "#D55E00",
        "ridge": "#009E73",
        "random_forest": "#CC79A7",
    }
    x = np.arange(4)
    shown_methods = ("pr24", "oracle", "ridge", "random_forest")
    method_labels = (
        "PR24\nbaseline",
        "R1-A Oracle\n13m target",
        "Ridge\npred. peak",
        "RF\npred. peak",
    )
    for col, cycle in enumerate(CYCLES):
        subset = aggregate.loc[aggregate["cycle"] == cycle].set_index("method")
        true_raw = float(subset.loc["pr24", "true_raw_peak_mean"])
        true_smooth = float(subset.loc["pr24", "true_smooth_peak_mean"])
        for row, (prefix, true_value) in enumerate(
            (("raw", true_raw), ("smooth", true_smooth))
        ):
            ax = axes[row, col]
            values = [float(subset.loc[m, f"{prefix}_peak_mean"]) for m in shown_methods]
            errors = [float(subset.loc[m, f"{prefix}_peak_std"]) for m in shown_methods]
            ax.bar(
                x,
                values,
                color=[colors[m] for m in shown_methods],
                width=0.7,
            )
            ax.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=3, lw=0.8)
            ax.axhline(true_value, color="black", lw=1.0, ls="--")
            ax.text(
                0.98,
                0.95,
                f"truth = {true_value:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            for position, value, error in zip(x, values, errors):
                ax.text(
                    position,
                    value + error + 3.0,
                    f"{value:.1f}\nΔ {value - true_value:+.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_ylim(0, max([true_value, *values]) * 1.28)
            ax.set_xticks(x, method_labels, rotation=0, fontsize=7)
            target_label = (
                "raw monthly maximum"
                if prefix == "raw"
                else "13-month centered maximum"
            )
            ax.set_title(
                f"{cycle} | {target_label}",
                loc="left",
                fontsize=9,
            )
            ax.grid(axis="y", color="#DDDDDD", lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel("Peak SSN")
    fig.suptitle(
        "Peak definition audit | bars = predicted value, dashed = truth, Δ = prediction - truth",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUT / f"peak_definition_comparison{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
