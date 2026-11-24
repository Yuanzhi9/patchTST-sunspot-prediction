"""Plot R1B-NEXT leakage-audit evidence for reviewer inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/r1b_next"
OUT = INPUT
TARGET_CONTEXT_CYCLES = tuple(range(18, 24))


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.to_period("M") - start.to_period("M")).n)


def main() -> None:
    baselines = pd.read_csv(INPUT / "baseline_predictions.csv")
    audit = pd.read_csv(INPUT / "leakage_audit.csv")
    audit["passed"] = audit["passed"].astype(bool)
    audit["applicable"] = audit["applicable"].astype(bool)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    colors = {"raw": "#D55E00", "smooth": "#0072B2"}

    ax = axes[0, 0]
    labels = []
    x_positions = []
    offset = 0.0
    for target_kind in ("raw", "smooth"):
        for model_name in ("ridge", "random_forest"):
            subset = baselines.loc[
                (baselines["target_kind"] == target_kind)
                & (baselines["model"] == model_name)
            ]
            mae = float(
                subset.groupby("seed")["absolute_error"].mean().mean()
            )
            ax.bar(
                offset,
                mae,
                width=0.72,
                color=colors[target_kind],
                alpha=0.82,
                label=target_kind if model_name == "ridge" else None,
            )
            ax.text(
                offset,
                mae + 1.0,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            labels.append(
                f"{'Raw' if target_kind == 'raw' else '13m'}\n"
                f"{'Ridge' if model_name == 'ridge' else 'RF'}"
            )
            x_positions.append(offset)
            offset += 1.0
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Expanding-window MAE (SSN)")
    ax.set_title("A. Four registered baselines reproduced")
    ax.legend(frameon=False)
    ax.set_ylim(0, 58)

    ax = axes[0, 1]
    check_labels = {
        "truncation": "Truncation",
        "future_poisoning": "Future\npoisoning",
        "target_blindness": "Target\nblindness",
    }
    checks = list(check_labels)
    applicable = audit.loc[audit["applicable"]]
    passes = [
        int(applicable.loc[applicable["check"] == check, "passed"].sum())
        for check in checks
    ]
    failures = [
        int((~applicable.loc[applicable["check"] == check, "passed"]).sum())
        for check in checks
    ]
    x = np.arange(len(checks))
    ax.bar(x, passes, color="#2A9D8F", label="pass")
    ax.bar(x, failures, bottom=passes, color="#C63D2F", label="flagged")
    for index, (passed, failed) in enumerate(zip(passes, failures)):
        ax.text(
            index,
            passed + failed + 6,
            f"pass {passed}\nflag {failed}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, [check_labels[check] for check in checks])
    ax.set_ylabel("Applicable checks")
    ax.set_title("B. Automated gate evidence (pending review)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(np.array(passes) + np.array(failures)) * 1.3)

    ax = axes[1, 0]
    future = audit.loc[
        (audit["check"] == "future_poisoning")
        & audit["applicable"]
        & (~audit["passed"])
        & (~audit["item"].isin(["prediction", "prediction_same_origin"]))
    ]
    failed_by_cycle = (
        future.groupby("cycle")["item"].nunique().reindex(
            TARGET_CONTEXT_CYCLES, fill_value=0
        )
    )
    target_cycles = [cycle + 1 for cycle in TARGET_CONTEXT_CYCLES]
    ax.bar(
        [str(cycle) for cycle in target_cycles],
        failed_by_cycle.to_numpy(),
        color="#E9C46A",
        edgecolor="#8A6D1D",
    )
    ax.set_xlabel("Target cycle")
    ax.set_ylabel("Changed features per origin")
    ax.set_title("C. Future poisoning changes context features")
    ax.set_ylim(0, max(7.6, float(failed_by_cycle.max()) + 1.5))

    ax = axes[1, 1]
    origin_rows = audit.loc[
        (audit["check"] == "future_poisoning")
        & (audit["item"] == "origin_date")
        & audit["cycle"].isin(TARGET_CONTEXT_CYCLES)
    ].copy()
    origin_rows["reference"] = pd.to_datetime(origin_rows["reference"])
    origin_rows["candidate"] = pd.to_datetime(origin_rows["candidate"])
    origin_rows["shift_months"] = [
        months_between(reference, candidate)
        for reference, candidate in zip(
            origin_rows["reference"], origin_rows["candidate"]
        )
    ]
    origin_rows = origin_rows.sort_values("cycle")
    ax.bar(
        [str(int(cycle) + 1) for cycle in origin_rows["cycle"]],
        origin_rows["shift_months"],
        color="#7A5195",
    )
    for x_index, shift in enumerate(origin_rows["shift_months"]):
        ax.text(
            x_index,
            shift + 0.6,
            f"+{int(shift)}m",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Target cycle")
    ax.set_ylabel("Forecast-origin shift (months)")
    ax.set_title("D. Poisoning moves the nominal forecast origin")
    ax.set_ylim(0, max(36, int(origin_rows["shift_months"].max()) + 5))

    figure.suptitle(
        "R1B-NEXT Leakage Audit — evidence for 景修 review, not a final failure ruling",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.01,
        (
            "Baseline reproduced exactly. Truncation and target blindness pass. "
            "Future poisoning changes cycle peak/origin construction; "
            "the blocking decision remains pending reviewer approval."
        ),
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.01, 0.03, 0.99, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "r1b_next_audit.png", dpi=220)
    figure.savefig(OUT / "r1b_next_audit.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
