"""Summarize and gate the E1b seed2021 oracle-conditioning screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/e1b_oracle_conditioning"
CYCLES = ("SC20", "SC21", "SC24")
MODES = ("none", "amp", "time", "both")


def flatten(metrics: dict, mode: str) -> dict:
    clip = metrics["clip0"]
    return {
        "cycle": metrics["cycle"],
        "mode": mode,
        "seed": metrics["seed"],
        "setting": metrics["setting"],
        "mae": clip["curve"]["mae"],
        "rmse": clip["curve"]["rmse"],
        "r2": clip["curve"]["r2"],
        "negative_months": clip["curve"]["negative_months"],
        "peak_amplitude_error": clip["extrema"]["peak"]["amplitude_error"],
        "peak_time_error_months": clip["extrema"]["peak"]["time_error_months"],
        "trough_amplitude_error": clip["extrema"]["trough"]["amplitude_error"],
        "trough_time_error_months": clip["extrema"]["trough"]["time_error_months"],
        "condition_sensitivity_pass": metrics["condition_sensitivity"]["pass"],
        "amp_peak_delta": metrics["condition_sensitivity"]["amp_peak_delta_for_two_std"],
        "time_peak_index_delta": metrics["condition_sensitivity"]["time_peak_index_delta_for_two_std"],
    }


def main() -> None:
    rows = []
    for cycle in CYCLES:
        for mode in MODES:
            path = ANALYSIS / "screen" / cycle / mode / "seed2021/metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(flatten(json.loads(path.read_text()), mode))
    detail = pd.DataFrame(rows)
    detail.to_csv(ANALYSIS / "e1b_screen_detail.csv", index=False)

    baseline = detail.loc[detail["mode"] == "none"].set_index("cycle")
    gains = []
    for mode in MODES[1:]:
        candidate = detail.loc[detail["mode"] == mode].set_index("cycle")
        for cycle in CYCLES:
            gains.append(
                {
                    "cycle": cycle,
                    "mode": mode,
                    "mae_baseline": baseline.loc[cycle, "mae"],
                    "mae_candidate": candidate.loc[cycle, "mae"],
                    "mae_improvement_pct": (
                        (baseline.loc[cycle, "mae"] - candidate.loc[cycle, "mae"])
                        / baseline.loc[cycle, "mae"]
                        * 100.0
                    ),
                    "peak_time_abs_baseline": abs(
                        baseline.loc[cycle, "peak_time_error_months"]
                    ),
                    "peak_time_abs_candidate": abs(
                        candidate.loc[cycle, "peak_time_error_months"]
                    ),
                }
            )
    gain_table = pd.DataFrame(gains)
    summary = (
        gain_table.groupby("mode", as_index=False)
        .agg(
            cycles_mae_improved_5pct=(
                "mae_improvement_pct",
                lambda series: int((series >= 5.0).sum()),
            ),
            mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
        )
        .sort_values("mean_mae_improvement_pct", ascending=False)
    )
    # The count above is replaced with an explicit pairwise guard below.
    peak_not_worse = (
        gain_table.assign(
            not_worse=(
                gain_table["peak_time_abs_candidate"]
                <= gain_table["peak_time_abs_baseline"]
            )
        )
        .groupby("mode")["not_worse"]
        .sum()
    )
    summary["cycles_peak_time_not_worse"] = summary["mode"].map(peak_not_worse)
    sensitivity_pass = (
        detail.loc[detail["mode"] != "none"]
        .groupby("mode")["condition_sensitivity_pass"]
        .all()
    )
    summary["condition_sensitivity_pass"] = summary["mode"].map(sensitivity_pass)
    max_degradation = (
        gain_table.groupby("mode")["mae_improvement_pct"].min().mul(-1.0)
    )
    summary["max_cycle_degradation_pct"] = summary["mode"].map(max_degradation)
    summary["screen_rules_pass"] = (
        (summary["cycles_mae_improved_5pct"] >= 2)
        & (summary["cycles_peak_time_not_worse"] >= 2)
        & (summary["condition_sensitivity_pass"].fillna(False))
    )
    # Parent PR24 acceptance rule: no tested cycle may degrade by more than 10%.
    summary["safety_pass"] = summary["max_cycle_degradation_pct"] <= 10.0
    summary["screen_pass"] = summary["screen_rules_pass"] & summary["safety_pass"]
    gain_table.to_csv(ANALYSIS / "e1b_screen_gain_by_cycle.csv", index=False)
    summary.to_csv(ANALYSIS / "e1b_screen_summary.csv", index=False)

    passed = summary.loc[summary["screen_pass"]].copy()
    decision = {
        "baseline": "none",
        "screen_seed": 2021,
        "gate": {
            "cycles_mae_improved_5pct_min": 2,
            "cycles_peak_time_not_worse_min": 2,
            "max_single_cycle_degradation_pct": 10.0,
        },
        "preregistration_note": (
            "The original E1b screen gate omitted the parent-plan per-cycle "
            "10% degradation guard. safety_pass enforces that parent rule."
        ),
        "passed_modes": passed["mode"].tolist(),
        "best_mode": passed["mode"].iloc[0] if len(passed) else None,
        "proceed_to_five_seed": bool(len(passed)),
        "summary": summary.to_dict(orient="records"),
    }
    (ANALYSIS / "e1b_screen_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plot = detail.copy()
    plot["peak_time_abs"] = plot["peak_time_error_months"].abs()
    plot["trough_abs"] = plot["trough_amplitude_error"].abs()
    panels = [
        ("mae", "Curve MAE", "#0072B2"),
        ("peak_time_abs", "Peak timing error (months)", "#009E73"),
        ("peak_amplitude_error", "Peak amplitude error (SSN)", "#D55E00"),
        ("trough_abs", "Absolute trough error (SSN)", "#CC79A7"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    width = 0.20
    x = np.arange(len(CYCLES))
    colors = dict(zip(MODES, ("#555555", "#0072B2", "#009E73", "#D55E00")))
    for ax, (column, title, _) in zip(axes.flat, panels):
        for offset, mode in enumerate(MODES):
            values = [
                float(plot.loc[(plot.cycle == cycle) & (plot["mode"] == mode), column].iloc[0])
                for cycle in CYCLES
            ]
            ax.bar(
                x + (offset - 1.5) * width,
                values,
                width,
                label=mode,
                color=colors[mode],
            )
        ax.set_xticks(x, CYCLES)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "E1b Oracle conditioning screen | seed2021 | no mode passed the safety gate",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            ANALYSIS / f"e1b_oracle_screen{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    print(detail.to_string(index=False))
    print(gain_table.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
