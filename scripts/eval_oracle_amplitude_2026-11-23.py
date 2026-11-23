"""Zero-training oracle-amplitude diagnostic for PR24 cycle holdouts."""

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
OUT = ROOT / "analysis/e1_oracle_amplitude"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
SMOOTH_WINDOW = 13


def evaluate_curve(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    eval_frame = frame[["date", "true"]].copy()
    eval_frame["pred"] = pred
    ext = extrema(eval_frame, SMOOTH_WINDOW, "peak_then_trough")
    return {
        "curve": metric_block(pred, frame["true"].to_numpy(dtype=float)),
        "peak_true_value": ext["peak"]["true_value"],
        "peak_pred_value": ext["peak"]["pred_value"],
        "peak_amplitude_error": ext["peak"]["amplitude_error"],
        "peak_time_error_months": ext["peak"]["time_error_months"],
        "peak_pred_date": ext["peak"]["pred_date"],
        "trough_true_value": ext["trough"]["true_value"],
        "trough_pred_value": ext["trough"]["pred_value"],
        "trough_amplitude_error": ext["trough"]["amplitude_error"],
        "trough_time_error_months": ext["trough"]["time_error_months"],
        "trough_pred_date": ext["trough"]["pred_date"],
        "amplitude_ratio": ext["amplitude_ratio"],
    }


def row_from_result(cycle: str, seed: int, method: str, result: dict) -> dict:
    return {
        "cycle": cycle,
        "seed": seed,
        "method": method,
        **{f"curve_{key}": value for key, value in result["curve"].items()},
        **{key: value for key, value in result.items() if key != "curve"},
    }


def plot_profiles(frames: dict[tuple[str, int], pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300, sharey=False)
    for ax, cycle in zip(axes, CYCLES):
        selected = {
            seed: frame for (name, seed), frame in frames.items() if name == cycle
        }
        reference = next(iter(selected.values()))
        dates = pd.to_datetime(reference["date"])
        baseline = np.vstack([frame["baseline_pred"].to_numpy() for frame in selected.values()])
        corrected = np.vstack([frame["oracle_pred"].to_numpy() for frame in selected.values()])
        ax.plot(dates, reference["true"], color="black", lw=1.3, label="Observed")
        ax.plot(
            dates,
            np.median(baseline, axis=0),
            color="#0072B2",
            lw=1.4,
            label="PR24 clip0",
        )
        ax.fill_between(
            dates,
            np.min(baseline, axis=0),
            np.max(baseline, axis=0),
            color="#0072B2",
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            dates,
            np.median(corrected, axis=0),
            color="#D55E00",
            lw=1.4,
            label="Oracle amplitude",
        )
        ax.fill_between(
            dates,
            np.min(corrected, axis=0),
            np.max(corrected, axis=0),
            color="#D55E00",
            alpha=0.12,
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
        "E1a non-deployable diagnostic: force each seed's 13-month peak to truth",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"e1a_oracle_amplitude{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    frames: dict[tuple[str, int], pd.DataFrame] = {}

    for cycle in CYCLES:
        cycle_dir = OUT / cycle
        cycle_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            curve_path = INPUT / cycle / f"p24_update_seed{seed}" / "p24_recursive_curve.csv"
            frame = pd.read_csv(curve_path, parse_dates=["date"])
            if frame[["pred", "true"]].isna().any().any():
                raise ValueError(f"NaN values in {curve_path}")

            baseline = np.clip(frame["pred"].to_numpy(dtype=float), 0.0, None)
            baseline_result = evaluate_curve(frame, baseline)
            true_peak = float(baseline_result["peak_true_value"])
            predicted_peak = float(baseline_result["peak_pred_value"])
            if predicted_peak <= 0:
                raise ValueError(f"non-positive predicted peak in {curve_path}")
            scale = true_peak / predicted_peak
            oracle = np.clip(baseline * scale, 0.0, None)
            oracle_result = evaluate_curve(frame, oracle)
            if abs(oracle_result["peak_amplitude_error"]) > 1e-7:
                raise RuntimeError(
                    f"oracle peak calibration failed for {cycle} seed{seed}: "
                    f"{oracle_result['peak_amplitude_error']}"
                )

            rows.append(row_from_result(cycle, seed, "pr24_clip0", baseline_result))
            rows.append(row_from_result(cycle, seed, "oracle_amplitude", oracle_result))
            saved = frame.copy()
            saved["baseline_pred"] = baseline
            saved["oracle_pred"] = oracle
            saved["scale"] = scale
            saved.to_csv(cycle_dir / f"oracle_amp_seed{seed}.csv", index=False)
            frames[(cycle, seed)] = saved

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "oracle_amplitude_seed_detail.csv", index=False)

    metric_columns = [
        "curve_mae",
        "curve_rmse",
        "curve_bias",
        "curve_r2",
        "curve_negative_months",
        "peak_amplitude_error",
        "peak_time_error_months",
        "trough_amplitude_error",
        "trough_time_error_months",
        "amplitude_ratio",
    ]
    aggregate = (
        detail.groupby(["cycle", "method"], as_index=False)
        .agg(
            **{f"{column}_mean": (column, "mean") for column in metric_columns},
            **{f"{column}_std": (column, "std") for column in metric_columns},
            seeds=("seed", "nunique"),
        )
        .sort_values(["cycle", "method"])
    )
    aggregate.to_csv(OUT / "oracle_amplitude_aggregate.csv", index=False)

    mae = aggregate.pivot(index="cycle", columns="method", values="curve_mae_mean")
    mae["improvement_pct"] = (
        (mae["pr24_clip0"] - mae["oracle_amplitude"]) / mae["pr24_clip0"] * 100.0
    )
    seed_pivot = detail.pivot_table(
        index=["cycle", "seed"], columns="method", values="curve_mae"
    )
    seed_pivot["improved"] = seed_pivot["oracle_amplitude"] < seed_pivot["pr24_clip0"]
    seed_gate = (
        seed_pivot.reset_index()
        .groupby("cycle")["improved"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "improved_seeds", "count": "n_seeds"})
    )
    trough = aggregate.pivot(
        index="cycle", columns="method", values="trough_amplitude_error_mean"
    )
    trough_abs = trough.abs()
    trough_worse_pct = (
        (trough_abs["oracle_amplitude"] - trough_abs["pr24_clip0"])
        / trough_abs["pr24_clip0"]
        * 100.0
    )

    cycles_improved = int((mae["improvement_pct"] > 0).sum())
    mean_improvement = float(mae["improvement_pct"].mean())
    seed_gate_pass = bool(
        all(
            seed_gate.loc[cycle, "improved_seeds"] >= 4
            for cycle in mae.index
            if mae.loc[cycle, "improvement_pct"] > 0
        )
    )
    trough_fail_count = int((trough_worse_pct > 25.0).sum())
    passed = bool(
        cycles_improved >= 2
        and mean_improvement >= 5.0
        and seed_gate_pass
        and trough_fail_count <= 1
    )

    decision = {
        "gate_definition": {
            "cycles_improved_min": 2,
            "mean_improvement_pct_min": 5.0,
            "improved_cycle_seeds_min": 4,
            "max_cycles_trough_abs_worse_gt_25pct": 1,
        },
        "cycles_improved": cycles_improved,
        "mean_improvement_pct": mean_improvement,
        "seed_gate_pass": seed_gate_pass,
        "trough_fail_count": trough_fail_count,
        "passed": passed,
        "decision": "proceed_to_e1b" if passed else "do_not_proceed_to_amplitude_fusion",
        "mae_by_cycle": {
            cycle: {
                "baseline": float(mae.loc[cycle, "pr24_clip0"]),
                "oracle_amplitude": float(mae.loc[cycle, "oracle_amplitude"]),
                "improvement_pct": float(mae.loc[cycle, "improvement_pct"]),
            }
            for cycle in mae.index
        },
        "seed_gate": seed_gate.to_dict(orient="index"),
        "trough_abs_worse_pct": trough_worse_pct.to_dict(),
    }
    (OUT / "oracle_amplitude_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_profiles(frames)

    print(mae.to_string())
    print(seed_gate.to_string())
    print(trough_worse_pct.to_string())
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
