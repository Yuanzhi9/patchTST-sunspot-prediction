"""Oracle peak-local constrained reconstruction on PR24 cycle holdouts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_cycle_extrema import extrema, metric_block

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/cycle_holdout"
OUT = ROOT / "analysis/r1a_constrained_peak"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
HALF_WIDTHS = (12, 24, 36)
PRIMARY_WIDTH = 24


def smooth13(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(13, center=True, min_periods=13).mean().to_numpy()


def raised_cosine_kernel(n: int, center: int, half_width: int) -> np.ndarray:
    distance = np.abs(np.arange(n) - center)
    kernel = np.zeros(n, dtype=float)
    inside = distance <= half_width
    kernel[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside] / half_width))
    return kernel


def calibrate_to_peak(
    baseline: np.ndarray,
    kernel: np.ndarray,
    target_peak: float,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float]:
    smooth_baseline = smooth13(baseline)
    valid = np.flatnonzero(~np.isnan(smooth_baseline))
    if not len(valid):
        raise RuntimeError("no valid smoothed baseline")
    center = int(valid[np.argmax(smooth_baseline[valid])])
    kernel = kernel.copy()
    if kernel[center] <= 0:
        raise RuntimeError("kernel must be positive at the current peak")

    delta = 0.0
    for _ in range(max_iterations):
        corrected = np.clip(baseline + delta * kernel, 0.0, None)
        smooth_corrected = smooth13(corrected)
        valid = np.flatnonzero(~np.isnan(smooth_corrected))
        peak_index = int(valid[np.argmax(smooth_corrected[valid])])
        error = target_peak - float(smooth_corrected[peak_index])
        if abs(error) < tolerance:
            return corrected, delta
        smooth_kernel = smooth13(kernel)
        sensitivity = float(smooth_kernel[peak_index])
        if not np.isfinite(sensitivity) or sensitivity <= 1e-10:
            raise RuntimeError("kernel has no sensitivity at the current peak")
        delta += error / sensitivity
    raise RuntimeError("peak calibration did not converge")


def assess(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    eval_frame = frame[["date", "true"]].copy()
    eval_frame["pred"] = pred
    ext = extrema(eval_frame, 13, "peak_then_trough")
    return {
        "curve": metric_block(pred, frame["true"].to_numpy(dtype=float)),
        "extrema": ext,
    }


def extract_row(cycle: str, seed: int, width: int | str, result: dict) -> dict:
    curve = result["curve"]
    ext = result["extrema"]
    return {
        "cycle": cycle,
        "seed": seed,
        "width": width,
        "mae": curve["mae"],
        "rmse": curve["rmse"],
        "bias": curve["bias"],
        "r2": curve["r2"],
        "negative_months": curve["negative_months"],
        "peak_amplitude_error": ext["peak"]["amplitude_error"],
        "peak_time_error_months": ext["peak"]["time_error_months"],
        "peak_pred_date": ext["peak"]["pred_date"],
        "trough_amplitude_error": ext["trough"]["amplitude_error"],
        "trough_time_error_months": ext["trough"]["time_error_months"],
        "trough_pred_date": ext["trough"]["pred_date"],
        "amplitude_ratio": ext["amplitude_ratio"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    profiles: dict[tuple[str, int], pd.DataFrame] = {}

    for cycle in CYCLES:
        cycle_out = OUT / cycle
        cycle_out.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            path = INPUT / cycle / f"p24_update_seed{seed}" / "p24_recursive_curve.csv"
            frame = pd.read_csv(path, parse_dates=["date"])
            baseline = np.clip(frame["pred"].to_numpy(dtype=float), 0.0, None)
            baseline_result = assess(frame, baseline)
            rows.append(extract_row(cycle, seed, "baseline", baseline_result))

            smooth_true = smooth13(frame["true"].to_numpy(dtype=float))
            true_valid = np.flatnonzero(~np.isnan(smooth_true))
            target_peak = float(np.max(smooth_true[true_valid]))
            smooth_base = smooth13(baseline)
            base_valid = np.flatnonzero(~np.isnan(smooth_base))
            center = int(base_valid[np.argmax(smooth_base[base_valid])])

            saved = frame.copy()
            saved["baseline_pred"] = baseline
            for width in HALF_WIDTHS:
                kernel = raised_cosine_kernel(len(frame), center, width)
                corrected, delta = calibrate_to_peak(baseline, kernel, target_peak)
                result = assess(frame, corrected)
                rows.append(extract_row(cycle, seed, width, result))
                saved[f"local_h{width}"] = corrected
                saved[f"delta_h{width}"] = delta
            saved.to_csv(cycle_out / f"constrained_peak_seed{seed}.csv", index=False)
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
        detail.groupby(["cycle", "width"], as_index=False)
        .agg(
            **{f"{column}_mean": (column, "mean") for column in numeric},
            **{f"{column}_std": (column, "std") for column in numeric},
            seeds=("seed", "nunique"),
        )
        .sort_values(["cycle", "width"])
    )
    aggregate.to_csv(OUT / "aggregate.csv", index=False)

    baseline = aggregate.loc[aggregate["width"] == "baseline"].set_index("cycle")
    primary = aggregate.loc[aggregate["width"] == PRIMARY_WIDTH].set_index("cycle")
    improvements = {
        cycle: {
            "mae_pct": float(
                (baseline.loc[cycle, "mae_mean"] - primary.loc[cycle, "mae_mean"])
                / baseline.loc[cycle, "mae_mean"]
                * 100.0
            ),
            "peak_abs_error_before": abs(
                float(baseline.loc[cycle, "peak_amplitude_error_mean"])
            ),
            "peak_abs_error_after": abs(
                float(primary.loc[cycle, "peak_amplitude_error_mean"])
            ),
            "trough_abs_error_before": abs(
                float(baseline.loc[cycle, "trough_amplitude_error_mean"])
            ),
            "trough_abs_error_after": abs(
                float(primary.loc[cycle, "trough_amplitude_error_mean"])
            ),
        }
        for cycle in CYCLES
    }
    seed_pivot = detail.pivot_table(
        index=["cycle", "seed"], columns="width", values="mae"
    )
    seed_improved = {
        cycle: int(
            (
                seed_pivot.loc[cycle, PRIMARY_WIDTH]
                < seed_pivot.loc[cycle, "baseline"]
            ).sum()
        )
        for cycle in CYCLES
    }
    width_direction = (
        detail.loc[detail["width"].isin(HALF_WIDTHS)]
        .pivot_table(index=["cycle", "seed"], columns="width", values="mae")
        .lt(seed_pivot["baseline"], axis=0)
        .groupby("cycle")
        .sum()
    )
    # Written gate: at least two kernel widths must improve at least 4/5 seeds
    # in at least 2/3 cycles.
    widths_direction_consistent = bool(
        ((width_direction >= 4).sum(axis=0) >= 2).sum() >= 2
    )
    gate = {
        "sc20_sc21_mae_improve_5pct": bool(
            improvements["SC20"]["mae_pct"] >= 5
            and improvements["SC21"]["mae_pct"] >= 5
        ),
        "sc24_mae_degradation_within_10pct": bool(
            improvements["SC24"]["mae_pct"] >= -10
        ),
        "trough_abs_degradation_within_10pct_all": bool(
            all(
                (
                    item["trough_abs_error_after"]
                    <= 1.10 * item["trough_abs_error_before"]
                )
                for item in improvements.values()
            )
        ),
        "peak_improved_two_cycles": int(
            sum(
                item["peak_abs_error_after"] < item["peak_abs_error_before"]
                for item in improvements.values()
            )
        )
        >= 2,
        "primary_not_single_seed": bool(
            all(seed_improved[cycle] >= 4 for cycle in ("SC20", "SC21"))
        ),
        "width_sensitivity_consistent": widths_direction_consistent,
    }
    passed = all(gate.values())
    decision = {
        "primary_half_width_months": PRIMARY_WIDTH,
        "sensitivity_half_widths_months": list(HALF_WIDTHS),
        "gate": gate,
        "improvements": improvements,
        "seed_improved_primary": seed_improved,
        "width_direction_consistent_count": width_direction.to_dict(orient="index"),
        "passed": passed,
        "decision": "proceed_to_predicted_peak_r1b" if passed else "stop_local_peak",
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    for ax, cycle in zip(axes, CYCLES):
        selected = {
            seed: frame for (name, seed), frame in profiles.items() if name == cycle
        }
        reference = next(iter(selected.values()))
        dates = pd.to_datetime(reference["date"])
        baseline_stack = np.vstack([f["baseline_pred"].to_numpy() for f in selected.values()])
        corrected_stack = np.vstack(
            [f[f"local_h{PRIMARY_WIDTH}"].to_numpy() for f in selected.values()]
        )
        ax.plot(dates, reference["true"], color="black", lw=1.3, label="Observed")
        ax.plot(dates, np.median(baseline_stack, axis=0), color="#0072B2", lw=1.4, label="PR24 clip0")
        ax.fill_between(
            dates,
            np.min(baseline_stack, axis=0),
            np.max(baseline_stack, axis=0),
            alpha=0.10,
            color="#0072B2",
            linewidth=0,
        )
        ax.plot(dates, np.median(corrected_stack, axis=0), color="#D55E00", lw=1.4, label="Local peak, H=24")
        ax.fill_between(
            dates,
            np.min(corrected_stack, axis=0),
            np.max(corrected_stack, axis=0),
            alpha=0.12,
            color="#D55E00",
            linewidth=0,
        )
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly mean SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "R1-A Oracle local peak correction | H=24 months | non-deployable upper bound",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1a_profiles{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300, sharey=True)
    widths = ("baseline", *HALF_WIDTHS)
    colors = dict(zip(widths, ("#555555", "#0072B2", "#009E73", "#D55E00")))
    for ax, cycle in zip(axes, CYCLES):
        subset = aggregate.loc[aggregate["cycle"] == cycle].set_index("width")
        values = [float(subset.loc[width, "mae_mean"]) for width in widths]
        errors = [float(subset.loc[width, "mae_std"]) for width in widths]
        ax.bar(np.arange(len(widths)), values, color=[colors[w] for w in widths])
        ax.errorbar(np.arange(len(widths)), values, yerr=errors, fmt="none", color="black", capsize=3, lw=0.8)
        ax.set_xticks(np.arange(len(widths)), [str(w) for w in widths])
        ax.set_title(cycle, loc="left", fontsize=10)
        ax.set_xlabel("Half width (months)")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Curve MAE")
    fig.suptitle("R1-A kernel-width sensitivity", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"r1a_kernel_sensitivity{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(aggregate.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
