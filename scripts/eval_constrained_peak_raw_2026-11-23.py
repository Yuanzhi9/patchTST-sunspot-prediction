"""R1-B raw-peak target audit with local constrained reconstruction."""

from __future__ import annotations

import importlib.util
import json
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
raised_cosine_kernel = r1a.raised_cosine_kernel

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/r1b_constrained_peak_predicted"
SCREEN = ROOT / "analysis/cycle_context/model_screening"
OUT = ROOT / "analysis/r1b_raw_peak"
CYCLES = {"SC20": 20, "SC21": 21, "SC24": 23}
SEEDS = (2019, 2020, 2021, 2022, 2023)
HALF_WIDTH = 24


def raw_peak(values: np.ndarray) -> tuple[int, float]:
    index = int(np.argmax(values))
    return index, float(values[index])


def calibrate_raw_peak(
    baseline: np.ndarray,
    kernel: np.ndarray,
    target_peak: float,
    iterations: int = 20,
) -> np.ndarray:
    delta = 0.0
    for _ in range(iterations):
        corrected = np.clip(baseline + delta * kernel, 0.0, None)
        index, peak = raw_peak(corrected)
        error = target_peak - peak
        if abs(error) < 1e-8:
            return corrected
        sensitivity = float(kernel[index])
        if sensitivity <= 1e-9:
            raise RuntimeError("kernel has no sensitivity at raw peak")
        delta += error / sensitivity
    raise RuntimeError("raw peak calibration did not converge")


def load_raw_predictions(model: str) -> dict[str, float]:
    frame = pd.read_csv(
        SCREEN / f"loocv_target_peak_raw_reduced_{model}.csv"
    ).set_index("cycle")
    return {
        cycle: float(frame.loc[context_cycle, "prediction"])
        for cycle, context_cycle in CYCLES.items()
    }


def evaluate(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    raw_index, raw_value = raw_peak(pred)
    true_raw_index, true_raw_value = raw_peak(frame["true"].to_numpy(dtype=float))
    base = assess(frame, pred)
    smooth_pred_peak = base["extrema"]["peak"]
    return {
        "curve": base["curve"],
        "raw_peak_value": raw_value,
        "raw_peak_error": raw_value - true_raw_value,
        "raw_peak_date": frame["date"].iloc[raw_index].strftime("%Y-%m"),
        "raw_peak_time_error_months": int(
            frame["date"].iloc[raw_index].to_period("M").ordinal
            - frame["date"].iloc[true_raw_index].to_period("M").ordinal
        ),
        "smooth_peak_value": float(smooth_pred_peak["pred_value"]),
        "smooth_peak_error": float(smooth_pred_peak["amplitude_error"]),
        "smooth_peak_date": smooth_pred_peak["pred_date"],
        "smooth_peak_time_error_months": int(smooth_pred_peak["time_error_months"]),
        "trough_amplitude_error": float(base["extrema"]["trough"]["amplitude_error"]),
        "trough_time_error_months": int(base["extrema"]["trough"]["time_error_months"]),
    }


def flatten(cycle: str, seed: int, method: str, result: dict) -> dict:
    return {
        "cycle": cycle,
        "seed": seed,
        "method": method,
        **{f"curve_{key}": value for key, value in result["curve"].items()},
        **{key: value for key, value in result.items() if key != "curve"},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_predictions = {
        "ridge_raw": load_raw_predictions("ridge"),
        "random_forest_raw": load_raw_predictions("random_forest"),
    }
    rows = []
    profiles = {}
    for cycle in CYCLES:
        for seed in SEEDS:
            frame = pd.read_csv(INPUT / f"{cycle}_seed{seed}.csv", parse_dates=["date"])
            baseline = np.clip(frame["pred_pr24"].to_numpy(dtype=float), 0.0, None)
            true_raw_index, true_raw_peak = raw_peak(
                frame["true"].to_numpy(dtype=float)
            )
            raw_base_index, _ = raw_peak(baseline)
            kernel = raised_cosine_kernel(len(frame), raw_base_index, HALF_WIDTH)
            sources = {
                "pr24": None,
                "oracle_raw": true_raw_peak,
                "ridge_raw": raw_predictions["ridge_raw"][cycle],
                "random_forest_raw": raw_predictions["random_forest_raw"][cycle],
            }
            saved = frame.copy()
            saved["baseline_pred"] = baseline
            for method, peak in sources.items():
                pred = (
                    baseline
                    if peak is None
                    else calibrate_raw_peak(baseline, kernel, peak)
                )
                rows.append(flatten(cycle, seed, method, evaluate(frame, pred)))
                saved[f"pred_{method}"] = pred
            saved.to_csv(OUT / f"{cycle}_seed{seed}.csv", index=False)
            profiles[(cycle, seed)] = saved

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "raw_peak_seed_detail.csv", index=False)
    numeric = [
        "curve_mae",
        "curve_rmse",
        "curve_r2",
        "raw_peak_value",
        "raw_peak_error",
        "raw_peak_time_error_months",
        "smooth_peak_value",
        "smooth_peak_error",
        "smooth_peak_time_error_months",
        "trough_amplitude_error",
        "trough_time_error_months",
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
    aggregate.to_csv(OUT / "raw_peak_aggregate.csv", index=False)

    baseline = aggregate.loc[aggregate["method"] == "pr24"].set_index("cycle")
    seed_pivot = detail.pivot_table(
        index=["cycle", "seed"], columns="method", values="raw_peak_error"
    )
    decisions = {}
    for method in ("oracle_raw", "ridge_raw", "random_forest_raw"):
        candidate = aggregate.loc[aggregate["method"] == method].set_index("cycle")
        raw_peak_improved = {
            cycle: int(
                abs(candidate.loc[cycle, "raw_peak_error_mean"])
                < abs(baseline.loc[cycle, "raw_peak_error_mean"])
            )
            for cycle in CYCLES
        }
        mae_change = {
            cycle: float(
                (candidate.loc[cycle, "curve_mae_mean"] - baseline.loc[cycle, "curve_mae_mean"])
                / baseline.loc[cycle, "curve_mae_mean"]
                * 100.0
            )
            for cycle in CYCLES
        }
        time_not_worse = {
            cycle: int(
                abs(candidate.loc[cycle, "raw_peak_time_error_months_mean"])
                <= abs(baseline.loc[cycle, "raw_peak_time_error_months_mean"])
            )
            for cycle in CYCLES
        }
        improved_seeds = {
            cycle: int((seed_pivot.loc[cycle, method].abs() < seed_pivot.loc[cycle, "pr24"].abs()).sum())
            for cycle in CYCLES
        }
        gate = {
            "raw_peak_improved_two_cycles": sum(raw_peak_improved.values()) >= 2,
            "curve_mae_no_cycle_worse_10pct": all(value <= 10.0 for value in mae_change.values()),
            "raw_peak_time_not_worse_two_cycles": sum(time_not_worse.values()) >= 2,
            "two_cycles_4of5_seed_peak_improved": sum(
                improved_seeds[cycle] >= 4 for cycle in CYCLES
            )
            >= 2,
        }
        decisions[method] = {
            "raw_peak_improved": raw_peak_improved,
            "curve_mae_change_pct": mae_change,
            "raw_peak_time_not_worse": time_not_worse,
            "improved_seeds": improved_seeds,
            "gate": gate,
            "passed": all(gate.values()),
        }
    deployable_pass = any(
        decisions[method]["passed"] for method in ("ridge_raw", "random_forest_raw")
    )
    decision = {
        "primary_definition": "raw monthly maximum",
        "secondary_definition": "13-month centered smoothed maximum",
        "half_width_months": HALF_WIDTH,
        "methods": decisions,
        "oracle_passed": decisions["oracle_raw"]["passed"],
        "deployable_passed": deployable_pass,
        "decision": "continue_r1b_raw_series" if deployable_pass else "r1b_raw_needs_new_amplitude_model",
    }
    (OUT / "raw_peak_decision.json").write_text(
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
        "oracle_raw": "#D55E00",
        "ridge_raw": "#0072B2",
        "random_forest_raw": "#009E73",
    }
    labels = {
        "pr24": "PR24",
        "oracle_raw": "Oracle raw peak",
        "ridge_raw": "Ridge raw",
        "random_forest_raw": "RF raw",
    }
    for ax, cycle in zip(axes, CYCLES):
        selected = {
            seed: frame for (name, seed), frame in profiles.items() if name == cycle
        }
        reference = next(iter(selected.values()))
        dates = pd.to_datetime(reference["date"])
        ax.plot(dates, reference["true"], color="black", lw=1.2, label="Observed raw")
        for method in colors:
            values = np.median(
                np.vstack([frame[f"pred_{method}"].to_numpy() for frame in selected.values()]),
                axis=0,
            )
            ax.plot(dates, values, color=colors[method], lw=1.3, label=labels[method])
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly mean SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("R1B-RAW | local correction targeting raw monthly peak", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1b_raw_profiles{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    methods = ("pr24", "oracle_raw", "ridge_raw", "random_forest_raw")
    x = np.arange(len(CYCLES))
    width = 0.20
    for ax, metric, title in (
        (axes[0], "raw_peak_error_mean", "Raw peak amplitude error"),
        (axes[1], "curve_mae_mean", "Curve MAE"),
    ):
        for offset, method in enumerate(methods):
            subset = aggregate.loc[aggregate["method"] == method].set_index("cycle")
            values = [float(subset.loc[cycle, metric]) for cycle in CYCLES]
            ax.bar(x + (offset - 1.5) * width, values, width, color=colors[method], label=labels[method])
        ax.axhline(0, color="black", lw=0.8) if metric == "raw_peak_error_mean" else None
        ax.set_xticks(x, CYCLES)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("R1B-RAW target audit", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1b_raw_errors{suffix}", bbox_inches="tight", facecolor="white")
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
