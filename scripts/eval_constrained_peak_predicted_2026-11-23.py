"""Apply R1-A local correction using cycle-level predicted peak amplitudes."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "r1a_constrained_peak", SCRIPT_DIR / "eval_constrained_peak_2026-11-23.py"
)
assert spec is not None and spec.loader is not None
r1a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1a)
assess = r1a.assess
calibrate_to_peak = r1a.calibrate_to_peak
raised_cosine_kernel = r1a.raised_cosine_kernel
smooth13 = r1a.smooth13

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/cycle_holdout"
OUT = ROOT / "analysis/r1b_constrained_peak_predicted"
SCREEN = ROOT / "analysis/cycle_context/model_screening"
CYCLES = {"SC20": 20, "SC21": 21, "SC24": 23}
SEEDS = (2019, 2020, 2021, 2022, 2023)
HALF_WIDTH = 24


def load_predictions(model: str) -> dict[str, float]:
    path = SCREEN / f"loocv_target_peak_smooth_reduced_{model}.csv"
    frame = pd.read_csv(path)
    by_cycle = frame.set_index("cycle")
    return {
        project_cycle: float(by_cycle.loc[context_cycle, "prediction"])
        for project_cycle, context_cycle in CYCLES.items()
    }


def flatten(cycle: str, seed: int, method: str, result: dict) -> dict:
    curve = result["curve"]
    ext = result["extrema"]
    return {
        "cycle": cycle,
        "seed": seed,
        "method": method,
        "mae": curve["mae"],
        "rmse": curve["rmse"],
        "bias": curve["bias"],
        "r2": curve["r2"],
        "negative_months": curve["negative_months"],
        "peak_amplitude_error": ext["peak"]["amplitude_error"],
        "peak_time_error_months": ext["peak"]["time_error_months"],
        "trough_amplitude_error": ext["trough"]["amplitude_error"],
        "trough_time_error_months": ext["trough"]["time_error_months"],
        "amplitude_ratio": ext["amplitude_ratio"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {
        "ridge": load_predictions("ridge"),
        "random_forest": load_predictions("random_forest"),
    }
    rows = []
    profiles = {}

    for cycle in CYCLES:
        for seed in SEEDS:
            path = INPUT / cycle / f"p24_update_seed{seed}" / "p24_recursive_curve.csv"
            frame = pd.read_csv(path, parse_dates=["date"])
            baseline = np.clip(frame["pred"].to_numpy(dtype=float), 0.0, None)
            smooth_base = smooth13(baseline)
            valid_base = np.flatnonzero(~np.isnan(smooth_base))
            center = int(valid_base[np.argmax(smooth_base[valid_base])])
            kernel = raised_cosine_kernel(len(frame), center, HALF_WIDTH)

            smooth_true = smooth13(frame["true"].to_numpy(dtype=float))
            valid_true = np.flatnonzero(~np.isnan(smooth_true))
            target_peak = float(np.max(smooth_true[valid_true]))
            sources = {
                "pr24": None,
                "oracle": target_peak,
                "ridge": predictions["ridge"][cycle],
                "random_forest": predictions["random_forest"][cycle],
            }
            saved = frame.copy()
            saved["baseline_pred"] = baseline
            corrections = {"pr24": baseline}
            for method, peak in sources.items():
                pred = baseline if peak is None else calibrate_to_peak(baseline, kernel, peak)[0]
                rows.append(flatten(cycle, seed, method, assess(frame, pred)))
                saved[f"pred_{method}"] = pred
                corrections[method] = pred
            saved.to_csv(OUT / f"{cycle}_seed{seed}.csv", index=False)
            profiles[(cycle, seed)] = saved

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "seed_detail.csv", index=False)
    numeric = [
        "mae",
        "rmse",
        "bias",
        "r2",
        "negative_months",
        "peak_amplitude_error",
        "peak_time_error_months",
        "trough_amplitude_error",
        "trough_time_error_months",
        "amplitude_ratio",
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
    aggregate.to_csv(OUT / "aggregate.csv", index=False)

    baseline = aggregate.loc[aggregate["method"] == "pr24"].set_index("cycle")
    decisions = {}
    seed_pivot = detail.pivot_table(index=["cycle", "seed"], columns="method", values="mae")
    for method in ("oracle", "ridge", "random_forest"):
        candidate = aggregate.loc[aggregate["method"] == method].set_index("cycle")
        improvements = {
            cycle: float(
                (baseline.loc[cycle, "mae_mean"] - candidate.loc[cycle, "mae_mean"])
                / baseline.loc[cycle, "mae_mean"]
                * 100.0
            )
            for cycle in CYCLES
        }
        peak_improved = sum(
            abs(candidate.loc[cycle, "peak_amplitude_error_mean"])
            < abs(baseline.loc[cycle, "peak_amplitude_error_mean"])
            for cycle in CYCLES
        )
        improved_seeds = {
            cycle: int((seed_pivot.loc[cycle, method] < seed_pivot.loc[cycle, "pr24"]).sum())
            for cycle in CYCLES
        }
        changing_cycles = [cycle for cycle, value in improvements.items() if abs(value) > 1e-9]
        gate = {
            "two_cycles_mae_improve_5pct": int(
                sum(value >= 5.0 for value in improvements.values())
            )
            >= 2,
            "no_cycle_degrade_over_10pct": all(value >= -10.0 for value in improvements.values()),
            "peak_improved_two_cycles": peak_improved >= 2,
            "trough_unchanged": all(
                abs(
                    candidate.loc[cycle, "trough_amplitude_error_mean"]
                    - baseline.loc[cycle, "trough_amplitude_error_mean"]
                )
                < 1e-8
                for cycle in CYCLES
            ),
            "at_least_two_cycles_with_4of5_seed_improvement": int(
                sum(improved_seeds[cycle] >= 4 for cycle in changing_cycles)
            )
            >= 2,
        }
        decisions[method] = {
            "mae_improvement_pct": improvements,
            "peak_improved_cycles": int(peak_improved),
            "improved_seeds": improved_seeds,
            "gate": gate,
            "passed": all(gate.values()),
        }
    deployable_passed = any(
        decisions[model]["passed"] for model in ("ridge", "random_forest")
    )
    decision = {
        "input_peak_predictions": {
            model: {cycle: predictions[model][cycle] for cycle in CYCLES}
            for model in predictions
        },
        "cycles_used": CYCLES,
        "half_width_months": HALF_WIDTH,
        "methods": decisions,
        "oracle_upper_bound_passed": decisions["oracle"]["passed"],
        "deployable_passed": deployable_passed,
        "passed": deployable_passed,
        "decision": "proceed_to_deployable_constrained_peak"
        if deployable_passed
        else "stop_predicted_peak_reconstruction",
    }
    (OUT / "decision.json").write_text(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    colors = {
        "pr24": "#555555",
        "oracle": "#D55E00",
        "ridge": "#0072B2",
        "random_forest": "#009E73",
    }
    for ax, cycle in zip(axes, CYCLES):
        selected = {
            seed: frame for (name, seed), frame in profiles.items() if name == cycle
        }
        ref = next(iter(selected.values()))
        dates = pd.to_datetime(ref["date"])
        ax.plot(dates, ref["true"], color="black", lw=1.2, label="Observed")
        for method in ("pr24", "oracle", "ridge", "random_forest"):
            values = np.median(
                np.vstack([frame[f"pred_{method}"].to_numpy() for frame in selected.values()]),
                axis=0,
            )
            ax.plot(dates, values, color=colors[method], lw=1.25, label=method)
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly mean SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("R1-B local peak correction: Oracle vs predicted cycle amplitude", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1b_profiles{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300, sharey=True)
    methods = ("pr24", "oracle", "ridge", "random_forest")
    width = 0.2
    x = np.arange(len(methods))
    for ax, cycle in zip(axes, CYCLES):
        subset = aggregate.loc[aggregate["cycle"] == cycle].set_index("method")
        values = [float(subset.loc[m, "mae_mean"]) for m in methods]
        errors = [float(subset.loc[m, "mae_std"]) for m in methods]
        ax.bar(x, values, color=[colors[m] for m in methods], width=0.7)
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=3, lw=0.8)
        ax.set_xticks(x, methods, rotation=20)
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Curve MAE")
    fig.suptitle("R1-B deployability loss: predicted peak amplitude", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1b_mae{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(aggregate.to_string(index=False))
    print(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
    )


if __name__ == "__main__":
    main()
