"""Summarize and gate the T1 scheduled-sampling screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/t1_scheduled_sampling_v2"
CYCLES = ("SC20", "SC21", "SC24")
MODES = ("teacher", "scheduled")


def row(cycle: str, mode: str, payload: dict) -> dict:
    curve = payload["clip0"]["curve"]
    ext = payload["clip0"]["extrema"]
    return {
        "cycle": cycle,
        "mode": mode,
        "seed": payload["seed"],
        "setting": payload["setting"],
        "mae": curve["mae"],
        "rmse": curve["rmse"],
        "r2": curve["r2"],
        "negative_months": curve["negative_months"],
        "peak_amplitude_error": ext["peak"]["amplitude_error"],
        "peak_time_error_months": ext["peak"]["time_error_months"],
        "trough_amplitude_error": ext["trough"]["amplitude_error"],
        "trough_time_error_months": ext["trough"]["time_error_months"],
    }


def main() -> None:
    rows = []
    for cycle in CYCLES:
        for mode in MODES:
            path = BASE / "screen" / cycle / mode / "seed2021/metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(row(cycle, mode, json.loads(path.read_text())))
    detail = pd.DataFrame(rows)
    detail.to_csv(BASE / "t1_screen_detail.csv", index=False)
    teacher = detail.loc[detail["mode"] == "teacher"].set_index("cycle")
    scheduled = detail.loc[detail["mode"] == "scheduled"].set_index("cycle")
    gains = pd.DataFrame(
        {
            "cycle": CYCLES,
            "teacher_mae": [teacher.loc[c, "mae"] for c in CYCLES],
            "scheduled_mae": [scheduled.loc[c, "mae"] for c in CYCLES],
            "mae_improvement_pct": [
                (teacher.loc[c, "mae"] - scheduled.loc[c, "mae"])
                / teacher.loc[c, "mae"]
                * 100.0
                for c in CYCLES
            ],
            "peak_abs_before": [abs(teacher.loc[c, "peak_amplitude_error"]) for c in CYCLES],
            "peak_abs_after": [abs(scheduled.loc[c, "peak_amplitude_error"]) for c in CYCLES],
            "peak_time_abs_before": [abs(teacher.loc[c, "peak_time_error_months"]) for c in CYCLES],
            "peak_time_abs_after": [abs(scheduled.loc[c, "peak_time_error_months"]) for c in CYCLES],
        }
    )
    gains.to_csv(BASE / "t1_screen_gain.csv", index=False)
    gate = {
        "two_cycles_mae_improve_5pct": int((gains["mae_improvement_pct"] >= 5).sum()) >= 2,
        "no_cycle_degrade_over_10pct": bool((gains["mae_improvement_pct"] >= -10).all()),
        "two_cycles_peak_not_double_worse": int(
            (
                (gains["peak_abs_after"] <= gains["peak_abs_before"])
                | (gains["peak_time_abs_after"] <= gains["peak_time_abs_before"])
            ).sum()
        )
        >= 2,
    }
    decision = {
        "baseline": "teacher",
        "candidate": "scheduled",
        "seed": 2021,
        "gate": gate,
        "passed": all(gate.values()),
        "decision": "expand_to_five_seeds" if all(gate.values()) else "stop_before_T2",
        "gains": gains.to_dict(orient="records"),
    }
    (BASE / "t1_screen_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    x = np.arange(len(CYCLES))
    width = 0.32
    for offset, mode, color in ((-width / 2, "teacher", "#555555"), (width / 2, "scheduled", "#D55E00")):
        subset = detail.loc[detail["mode"] == mode].set_index("cycle")
        axes[0].bar(
            x + offset,
            [subset.loc[c, "mae"] for c in CYCLES],
            width,
            color=color,
            label=mode,
        )
        axes[1].bar(
            x + offset,
            [abs(subset.loc[c, "peak_amplitude_error"]) for c in CYCLES],
            width,
            color=color,
            label=mode,
        )
    for ax, title in zip(axes, ("Curve MAE", "Absolute peak amplitude error")):
        ax.set_xticks(x, CYCLES)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("T1 scheduled sampling screen | seed2021", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(BASE / f"t1_screen{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(detail.to_string(index=False))
    print(gains.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
