"""Quantify negative predictions and post-hoc smoothing on cycle-holdout curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/cycle_holdout"
OUT = ROOT / "analysis/negative_smoothing"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
METHODS = ("p24_update", "p24_recursive", "p24_recursive_b12", "p24_recursive_b6", "p132_direct")
GAUSS_SIGMA = 3.6
GAUSS_24_FWHM_SIGMA = 24.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def load_curve(cycle: str, method: str, seed: int) -> pd.DataFrame:
    if method == "p132_direct":
        meta = json.loads(
            (BASE / cycle / f"p132_direct_seed{seed}_extrema.json").read_text(encoding="utf-8")
        )
        path = Path(meta["curve"])
    else:
        path = BASE / cycle / f"p24_update_seed{seed}" / f"{method}_curve.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame[["date", "pred", "true"]].sort_values("date").reset_index(drop=True)


def ma13(series: pd.Series) -> pd.Series:
    return series.rolling(13, center=True, min_periods=13).mean()


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    diff = pred - true
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    negative_rows = []
    smoothing_rows = []
    plot_curves: dict[tuple[str, int], pd.DataFrame] = {}

    for cycle in CYCLES:
        for method in METHODS:
            for seed in SEEDS:
                frame = load_curve(cycle, method, seed)
                pred = frame["pred"].to_numpy(dtype=float)
                true = frame["true"].to_numpy(dtype=float)
                negative = pred < 0
                raw_mae, raw_rmse = metrics(pred, true)
                clip_mae, clip_rmse = metrics(np.clip(pred, 0, None), true)
                negative_rows.append(
                    {
                        "cycle": cycle,
                        "method": method,
                        "seed": seed,
                        "negative_months": int(negative.sum()),
                        "first_negative": frame.loc[negative, "date"].min().strftime("%Y-%m")
                        if negative.any()
                        else "",
                        "last_negative": frame.loc[negative, "date"].max().strftime("%Y-%m")
                        if negative.any()
                        else "",
                        "min_pred": float(np.min(pred)),
                        "min_pred_date": frame.loc[int(np.argmin(pred)), "date"].strftime("%Y-%m"),
                        "raw_MAE": raw_mae,
                        "clip0_MAE": clip_mae,
                        "MAE_delta_after_clip": clip_mae - raw_mae,
                        "raw_RMSE": raw_rmse,
                        "clip0_RMSE": clip_rmse,
                    }
                )

                smooth_pred_ma = ma13(frame["pred"])
                smooth_true_ma = ma13(frame["true"])
                valid_ma = smooth_pred_ma.notna() & smooth_true_ma.notna()
                ma_mae, ma_rmse = metrics(
                    smooth_pred_ma[valid_ma].to_numpy(),
                    smooth_true_ma[valid_ma].to_numpy(),
                )
                pred_g3 = gaussian_filter1d(pred, sigma=GAUSS_SIGMA, mode="nearest")
                true_g3 = gaussian_filter1d(true, sigma=GAUSS_SIGMA, mode="nearest")
                g3_mae, g3_rmse = metrics(pred_g3, true_g3)
                pred_g24 = gaussian_filter1d(pred, sigma=GAUSS_24_FWHM_SIGMA, mode="nearest")
                true_g24 = gaussian_filter1d(true, sigma=GAUSS_24_FWHM_SIGMA, mode="nearest")
                g24_mae, g24_rmse = metrics(pred_g24, true_g24)
                smoothing_rows.append(
                    {
                        "cycle": cycle,
                        "method": method,
                        "seed": seed,
                        "raw_MAE": raw_mae,
                        "ma13_MAE": ma_mae,
                        "gauss_sigma3.6_MAE": g3_mae,
                        "gauss_24mo_fwhm_MAE": g24_mae,
                        "raw_RMSE": raw_rmse,
                        "ma13_RMSE": ma_rmse,
                        "gauss_sigma3.6_RMSE": g3_rmse,
                        "gauss_24mo_fwhm_RMSE": g24_rmse,
                    }
                )
                if method == "p24_recursive" and seed == 2021:
                    plotted = frame.copy()
                    plotted["pred_ma13"] = ma13(frame["pred"])
                    plotted["true_ma13"] = ma13(frame["true"])
                    plot_curves[(cycle, seed)] = plotted

    negative = pd.DataFrame(negative_rows)
    smoothing = pd.DataFrame(smoothing_rows)
    negative.to_csv(OUT / "negative_detail.csv", index=False)
    smoothing.to_csv(OUT / "smoothing_detail.csv", index=False)

    neg_agg = (
        negative.groupby(["cycle", "method"], as_index=False)
        .agg(
            negative_months_mean=("negative_months", "mean"),
            negative_months_max=("negative_months", "max"),
            min_pred=("min_pred", "min"),
            raw_MAE=("raw_MAE", "mean"),
            clip0_MAE=("clip0_MAE", "mean"),
        )
    )
    neg_agg["MAE_delta_after_clip"] = neg_agg["clip0_MAE"] - neg_agg["raw_MAE"]
    neg_agg.to_csv(OUT / "negative_summary.csv", index=False)

    smooth_agg = (
        smoothing.groupby(["cycle", "method"], as_index=False)
        .agg(
            raw_MAE=("raw_MAE", "mean"),
            ma13_MAE=("ma13_MAE", "mean"),
            gauss_sigma3_6_MAE=("gauss_sigma3.6_MAE", "mean"),
            gauss_24mo_fwhm_MAE=("gauss_24mo_fwhm_MAE", "mean"),
        )
    )
    smooth_agg.to_csv(OUT / "smoothing_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=220)
    labels = []
    counts = []
    mins = []
    for (cycle, method), group in negative.groupby(["cycle", "method"]):
        labels.append(f"{cycle}\n{method}")
        counts.append(float(group["negative_months"].mean()))
        mins.append(float(group["min_pred"].min()))
    x = np.arange(len(labels))
    ax.bar(x, counts, color="#0072B2", alpha=0.85)
    ax.set_ylabel("Mean negative months per seed")
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    ax.set_title("Physical negative SSN predictions and their trough location", loc="left")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "negative_predictions_2026-11-21.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "negative_predictions_2026-11-21.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    detail = pd.read_csv(BASE / "cycle_holdout_seed_detail.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220)
    method_colors = {
        "p24_update": "#009E73",
        "p24_recursive": "#E69F00",
        "p132_direct": "#0072B2",
    }
    method_labels = {
        "p24_update": "PR24 observed-update",
        "p24_recursive": "PR24 recursive24",
        "p132_direct": "PR132 direct",
    }
    for method, color in method_colors.items():
        subset = detail[detail["method"] == method]
        ax.scatter(
            subset["peak_true_value"],
            subset["peak_pred_value"],
            s=28,
            alpha=0.75,
            color=color,
            label=f"{method_labels[method]} (r={subset['peak_true_value'].corr(subset['peak_pred_value']):.2f})",
        )
    limits = [80, 260]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--", label="Perfect amplitude")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True 13-month-smoothed peak SSN")
    ax.set_ylabel("Predicted peak SSN")
    ax.set_title("Cycle-strength response across SC20/SC21/SC24, five seeds", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "peak_amplitude_response_2026-11-21.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "peak_amplitude_response_2026-11-21.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=220)
    for ax, cycle in zip(axes, CYCLES):
        frame = plot_curves[(cycle, 2021)]
        ax.plot(frame["date"], frame["true"], color="black", lw=1.1, label="Observed raw")
        ax.plot(frame["date"], frame["true_ma13"], color="#999999", lw=1.7, label="Observed 13m MA")
        ax.plot(frame["date"], frame["pred"], color="#E69F00", lw=1.0, label="Predicted raw")
        ax.plot(frame["date"], frame["pred_ma13"], color="#0072B2", lw=1.6, label="Predicted 13m MA")
        ax.set_title(f"{cycle} | post-hoc smoothing of PR24 recursive block24 | seed2021", loc="left", fontsize=8.5)
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if cycle == "SC20":
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "posthoc_smoothing_profiles_2026-11-21.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "posthoc_smoothing_profiles_2026-11-21.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(neg_agg.to_string(index=False))
    print(smooth_agg.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
