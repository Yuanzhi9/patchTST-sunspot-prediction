"""Summarize date-matched raw versus official-smoothed A0 CV5 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/a0_raw_vs_smoothed"
DESIGNS = {
    "raw": "B26M-A0RAW-1749-2026-04",
    "smoothed13": "B26M-A0SMOOTH13-1749-2026-04",
}
SEEDS = [2019, 2020, 2021, 2022, 2023]


def physical(series: np.ndarray, meta: dict) -> np.ndarray:
    mean = float(meta["scaler_mean"][-1])
    scale = float(meta["scaler_scale"][-1])
    return series * scale + mean


def normalized_metrics(result_dir: Path) -> tuple[float, float]:
    pred = physical(np.load(result_dir / "pred.npy")[:, :, -1], json.loads((result_dir / "cv_window_meta.json").read_text()))
    true = physical(np.load(result_dir / "true.npy")[:, :, -1], json.loads((result_dir / "cv_window_meta.json").read_text()))
    scale = float(np.std(true, ddof=0))
    return (
        float(np.mean(np.abs(pred - true)) / scale),
        float(np.sqrt(np.mean((pred - true) ** 2)) / scale),
    )


def main() -> None:
    rows = []
    for variant, experiment in DESIGNS.items():
        for seed in SEEDS:
            aggregate_path = ROOT / f"analysis/{experiment}_seed{seed}/cv5_aggregate.json"
            if not aggregate_path.exists():
                raise FileNotFoundError(aggregate_path)
            aggregate = json.loads(aggregate_path.read_text())
            result_dirs = sorted(
                (ROOT / "results").glob(f"*_{experiment}_F*_seed{seed}")
            )
            if len(result_dirs) != 5:
                raise RuntimeError(f"{experiment} seed {seed}: expected 5 folds, got {len(result_dirs)}")
            normalized = [normalized_metrics(path) for path in result_dirs]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "full_MAE": aggregate["full_MAE_mean"],
                    "block24_MAE": aggregate["block24_MAE_mean"],
                    "normalized_MAE": np.mean([value[0] for value in normalized]),
                    "normalized_RMSE": np.mean([value[1] for value in normalized]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("variant", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            full_MAE_mean=("full_MAE", "mean"),
            full_MAE_std=("full_MAE", "std"),
            block24_MAE_mean=("block24_MAE", "mean"),
            block24_MAE_std=("block24_MAE", "std"),
            normalized_MAE_mean=("normalized_MAE", "mean"),
            normalized_RMSE_mean=("normalized_RMSE", "mean"),
        )
        .sort_values("variant")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "a0_raw_smoothed_seed_detail.csv", index=False)
    summary.to_csv(OUT / "a0_raw_smoothed_summary.csv", index=False)

    order = ["raw", "smoothed13"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=220)
    positions = np.arange(2)
    width = 0.34
    for offset, metric, label, color in (
        (-width / 2, "full_MAE", "Full", "#0072B2"),
        (width / 2, "block24_MAE", "Block24", "#D55E00"),
    ):
        values = [
            float(summary.loc[summary["variant"] == key, f"{metric}_mean"].iloc[0])
            for key in order
        ]
        errors = [
            float(summary.loc[summary["variant"] == key, f"{metric}_std"].iloc[0])
            for key in order
        ]
        axes[0].bar(positions + offset, values, width, yerr=errors, capsize=3, label=label, color=color)
    axes[0].set_xticks(positions, ["Raw target", "13m smoothed target"])
    axes[0].set_ylabel("Physical SSN MAE")
    axes[0].set_title("A. Metrics on each target's own scale", loc="left", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)

    for offset, metric, label, color in (
        (-width / 2, "normalized_MAE", "Normalized MAE", "#009E73"),
        (width / 2, "normalized_RMSE", "Normalized RMSE", "#CC79A7"),
    ):
        values = [
            float(summary.loc[summary["variant"] == key, f"{metric}_mean"].iloc[0])
            for key in order
        ]
        axes[1].bar(positions + offset, values, width, label=label, color=color)
    axes[1].set_xticks(positions, ["Raw target", "13m smoothed target"])
    axes[1].set_ylabel("Error / truth standard deviation")
    axes[1].set_title("B. Scale-normalized comparison", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "A0 date-matched baseline, 1749-07 to 2026-04, seeds 2019-2023",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"a0_raw_vs_smoothed{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
