"""Evaluate the frozen PR24 baseline with a physical non-negative output gate.

The recursive block24 curve is the deployable forecast path because it never
uses test-period observations. Observed-update block24 is retained as a
reference for what real observations would buy.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/cycle_holdout"
OUT = ROOT / "analysis/pr24_nonnegative_holdout"
DATA_CSV = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
METHODS = ("p24_update", "p24_recursive")
VARIANTS = ("raw", "clip0")
SMOOTH_WINDOW = 13

METHOD_LABELS = {
    "p24_update": "PR24 observed update",
    "p24_recursive": "PR24 recursive block24",
}
COLORS = {
    "truth": "#111111",
    "raw": "#777777",
    "clip0": "#D55E00",
    "persistence": "#0072B2",
    "update": "#009E73",
    "recursive": "#E69F00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curve(cycle: str, method: str, seed: int) -> pd.DataFrame:
    path = BASE / cycle / f"p24_update_seed{seed}" / f"{method}_curve.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "pred", "true", "lead", "block_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame.sort_values("date").reset_index(drop=True)


def smooth13(values: pd.Series) -> pd.Series:
    return values.rolling(
        SMOOTH_WINDOW,
        center=True,
        min_periods=SMOOTH_WINDOW,
    ).mean()


def month_delta(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return int(
        (later.to_period("M") - earlier.to_period("M")).n
    )


def metric_block(pred: np.ndarray, true: np.ndarray) -> dict:
    diff = pred - true
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    pearson_r = float(pearsonr(pred, true).statistic) if np.std(pred) > 0 else np.nan
    spearman_r = (
        float(spearmanr(pred, true).statistic) if np.std(pred) > 0 else np.nan
    )
    true_std = float(np.std(true, ddof=0))
    valid_mape = np.abs(true) >= 5.0
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "nrmse_by_truth_std": float(np.sqrt(np.mean(diff**2)) / true_std)
        if true_std > 0
        else None,
        "mape_true_ge5_pct": float(
            100.0
            * np.mean(
                np.abs(diff[valid_mape]) / np.abs(true[valid_mape])
            )
        )
        if valid_mape.any()
        else None,
        "negative_months": int(np.sum(pred < 0)),
        "min_pred": float(np.min(pred)),
        "max_pred": float(np.max(pred)),
    }


def extrema_block(pred: np.ndarray, true: np.ndarray, dates: pd.Series) -> dict:
    smooth_true = smooth13(pd.Series(true))
    smooth_pred = smooth13(pd.Series(pred))
    valid = smooth_true.notna() & smooth_pred.notna()
    if not valid.any():
        raise RuntimeError("no valid values for 13-month extrema")

    idx_true_peak = int(smooth_true[valid].idxmax())
    idx_pred_peak = int(smooth_pred[valid].idxmax())
    after_true = valid & (pd.Series(np.arange(len(true))) > idx_true_peak)
    after_pred = valid & (pd.Series(np.arange(len(true))) > idx_pred_peak)
    if not after_true.any() or not after_pred.any():
        raise RuntimeError("cannot locate trough after peak")
    idx_true_trough = int(smooth_true[after_true].idxmin())
    idx_pred_trough = int(smooth_pred[after_pred].idxmin())

    def record(kind: str, pred_idx: int, true_idx: int) -> dict:
        pred_value = float(smooth_pred.iloc[pred_idx])
        true_value = float(smooth_true.iloc[true_idx])
        amplitude_error = pred_value - true_value
        time_error = month_delta(dates.iloc[pred_idx], dates.iloc[true_idx])
        return {
            f"{kind}_true_date": dates.iloc[true_idx].strftime("%Y-%m"),
            f"{kind}_true_value": true_value,
            f"{kind}_pred_date": dates.iloc[pred_idx].strftime("%Y-%m"),
            f"{kind}_pred_value": pred_value,
            f"{kind}_amplitude_error": amplitude_error,
            f"{kind}_relative_amplitude_error_pct": (
                100.0 * amplitude_error / true_value if true_value != 0 else None
            ),
            f"{kind}_time_error_months": time_error,
            f"{kind}_abs_time_error_months": abs(time_error),
            f"{kind}_hit_3m": abs(time_error) <= 3,
            f"{kind}_hit_6m": abs(time_error) <= 6,
            f"{kind}_hit_12m": abs(time_error) <= 12,
        }

    output = {
        **record("peak", idx_pred_peak, idx_true_peak),
        **record("trough", idx_pred_trough, idx_true_trough),
    }
    output["amplitude_ratio_raw"] = float(
        (np.max(pred) - np.min(pred)) / (np.max(true) - np.min(true))
    )
    output["amplitude_ratio_smoothed"] = float(
        (smooth_pred.max() - smooth_pred.min())
        / (smooth_true.max() - smooth_true.min())
    )
    return output


def persistence_prediction(
    frame: pd.DataFrame,
    raw_by_date: dict[pd.Timestamp, float],
) -> np.ndarray:
    output = pd.Series(index=frame.index, dtype=float)
    for block_index, group in frame.groupby("block_index", sort=False):
        start = group["date"].min()
        previous = start - pd.DateOffset(months=1)
        if previous not in raw_by_date:
            raise KeyError(f"missing persistence observation at {previous:%Y-%m}")
        output.loc[group.index] = raw_by_date[previous]
    return output.to_numpy(dtype=float)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.nan, np.nan
    draws = rng.choice(clean, size=(10000, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate_metrics(
    detail: pd.DataFrame,
    value_columns: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    group_columns = ["method", "variant", "cycle"]
    for keys, group in detail.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys))
        record["n_seeds"] = int(group["seed"].nunique())
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, rng)
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_std"] = float(np.std(values, ddof=0))
            record[f"{column}_median"] = float(np.median(values))
            record[f"{column}_min"] = float(np.min(values))
            record[f"{column}_max"] = float(np.max(values))
            record[f"{column}_mean_ci95_low"] = low
            record[f"{column}_mean_ci95_high"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def ensemble_metrics(
    curves: dict[tuple[str, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bands = []
    for cycle in CYCLES:
        for method in METHODS:
            for variant in VARIANTS:
                merged = None
                truth = None
                for seed in SEEDS:
                    frame = curves[(cycle, method, seed)]
                    column = "pred" if variant == "raw" else "pred_clip0"
                    current = frame[["date", "true", column]].rename(
                        columns={column: f"seed{seed}"}
                    )
                    if merged is None:
                        merged = current
                        truth = frame["true"].to_numpy(dtype=float)
                    else:
                        merged = merged.merge(
                            current[["date", f"seed{seed}"]],
                            on="date",
                            how="inner",
                        )
                values = merged[
                    [column for column in merged.columns if column.startswith("seed")]
                ].to_numpy(dtype=float)
                median = np.median(values, axis=1)
                low90 = np.quantile(values, 0.10, axis=1)
                high90 = np.quantile(values, 0.90, axis=1)
                low_all = np.min(values, axis=1)
                high_all = np.max(values, axis=1)
                metrics = metric_block(median, truth)
                metrics.update(extrema_block(median, truth, merged["date"]))
                metrics.update(
                    {
                        "cycle": cycle,
                        "method": method,
                        "variant": variant,
                        "coverage_80pct_interval": float(
                            np.mean((truth >= low90) & (truth <= high90))
                        ),
                        "coverage_seed_range": float(
                            np.mean((truth >= low_all) & (truth <= high_all))
                        ),
                    }
                )
                rows.append(metrics)
                bands.append(
                    pd.DataFrame(
                        {
                            "date": merged["date"],
                            "cycle": cycle,
                            "method": method,
                            "variant": variant,
                            "truth": truth,
                            "median": median,
                            "q10": low90,
                            "q90": high90,
                            "min_seed": low_all,
                            "max_seed": high_all,
                        }
                    )
                )
    return pd.DataFrame(rows), pd.concat(bands, ignore_index=True)


def leadtime_metrics(
    curves: dict[tuple[str, str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    bins = (
        ("01-24", 1, 24),
        ("25-48", 25, 48),
        ("49-72", 49, 72),
        ("73-96", 73, 96),
        ("97-132", 97, 132),
    )
    for cycle in CYCLES:
        for method in METHODS:
            for seed in SEEDS:
                frame = curves[(cycle, method, seed)].copy()
                frame["horizon"] = np.arange(1, len(frame) + 1)
                truth = frame["true"].to_numpy(dtype=float)
                raw = frame["pred"].to_numpy(dtype=float)
                clip = frame["pred_clip0"].to_numpy(dtype=float)
                persistence = frame["persistence"].to_numpy(dtype=float)
                for label, start, end in bins:
                    mask = (frame["horizon"] >= start) & (frame["horizon"] <= end)
                    if not mask.any():
                        continue
                    rows.append(
                        {
                            "cycle": cycle,
                            "method": method,
                            "seed": seed,
                            "horizon_bin": label,
                            "n_months": int(mask.sum()),
                            "raw_MAE": float(np.mean(np.abs(raw[mask] - truth[mask]))),
                            "clip0_MAE": float(np.mean(np.abs(clip[mask] - truth[mask]))),
                            "persistence_MAE": float(
                                np.mean(np.abs(persistence[mask] - truth[mask]))
                            ),
                        }
                    )
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby(
            ["method", "cycle", "horizon_bin"],
            as_index=False,
        )
        .agg(
            n_months=("n_months", "first"),
            raw_MAE_mean=("raw_MAE", "mean"),
            raw_MAE_std=("raw_MAE", "std"),
            clip0_MAE_mean=("clip0_MAE", "mean"),
            clip0_MAE_std=("clip0_MAE", "std"),
            persistence_MAE_mean=("persistence_MAE", "mean"),
            persistence_MAE_std=("persistence_MAE", "std"),
        )
        .fillna(0.0)
    )
    return detail, aggregate


def plot_profiles(
    bands: pd.DataFrame,
    curves: dict[tuple[str, str, int], pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), dpi=240)
    for ax, cycle in zip(axes, CYCLES):
        subset = bands[
            (bands["cycle"] == cycle)
            & (bands["method"] == "p24_recursive")
            & (bands["variant"] == "clip0")
        ].sort_values("date")
        raw_subset = bands[
            (bands["cycle"] == cycle)
            & (bands["method"] == "p24_recursive")
            & (bands["variant"] == "raw")
        ].sort_values("date")
        ax.fill_between(
            subset["date"],
            subset["q10"],
            subset["q90"],
            color=COLORS["clip0"],
            alpha=0.18,
            linewidth=0,
            label="Five-seed 10-90% range",
        )
        ax.plot(
            subset["date"],
            subset["truth"],
            color=COLORS["truth"],
            lw=1.5,
            label="Observed monthly SSN",
        )
        ax.plot(
            subset["date"],
            subset["median"],
            color=COLORS["clip0"],
            lw=1.7,
            label="PR24 recursive24, clip0 median",
        )
        ax.plot(
            raw_subset["date"],
            raw_subset["median"],
            color=COLORS["raw"],
            lw=0.9,
            ls="--",
            label="Raw median before non-negative gate",
        )
        ax.axhline(0, color="#BBBBBB", lw=0.8)
        smoothed_true = smooth13(subset["truth"].reset_index(drop=True))
        smoothed_pred = smooth13(subset["median"].reset_index(drop=True))
        for series, source, color in (
            (smoothed_true, "truth", COLORS["truth"]),
            (smoothed_pred, "pred", COLORS["clip0"]),
        ):
            peak = int(series.idxmax())
            trough = int(series.idxmin())
            for idx, label in ((peak, "P"), (trough, "T")):
                date = subset["date"].iloc[idx]
                y = (
                    subset["truth"].iloc[idx]
                    if source == "truth"
                    else subset["median"].iloc[idx]
                )
                ax.scatter(date, y, color=color, s=25, zorder=5)
                ax.annotate(
                    f"{label} {date:%Y-%m}",
                    (date, y),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                )
        ax.set_title(
            f"{cycle} | independent training before holdout | five seeds",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if cycle == "SC20":
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = OUT / "pr24_nonnegative_profiles_2026-11-21"
    fig.savefig(path.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(seed_metrics: pd.DataFrame) -> None:
    recursive = seed_metrics[seed_metrics["method"] == "p24_recursive"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=240)
    for ax, metric, title in zip(
        axes,
        ("mae", "pearson_r", "skill_vs_persistence"),
        ("MAE", "Pearson correlation", "Skill vs persistence"),
    ):
        data = []
        labels = []
        positions = []
        pos = 1
        for cycle in CYCLES:
            for variant in VARIANTS:
                subset = recursive[
                    (recursive["cycle"] == cycle)
                    & (recursive["variant"] == variant)
                ]
                data.append(subset[metric].to_numpy(dtype=float))
                labels.append(f"{cycle}\n{variant}")
                positions.append(pos)
                pos += 1
            pos += 0.8
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(
                COLORS["clip0"] if "clip0" in labels[index] else COLORS["raw"]
            )
            patch.set_alpha(0.55)
        for index, values in enumerate(data):
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(
                positions[index] + jitter,
                values,
                s=13,
                color="black",
                alpha=0.65,
                zorder=4,
            )
        ax.axhline(0 if metric != "skill_vs_persistence" else 0, color="#888888", lw=0.8)
        ax.set_xticks(positions, labels, fontsize=7)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("PR24 recursive block24: five-seed metric distributions", fontsize=10)
    fig.tight_layout()
    path = OUT / "pr24_nonnegative_metric_distributions_2026-11-21"
    fig.savefig(path.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_extrema(seed_extrema: pd.DataFrame) -> None:
    recursive = seed_extrema[seed_extrema["method"] == "p24_recursive"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=240)
    x = np.arange(len(CYCLES))
    width = 0.34
    panels = (
        ("peak_amplitude_error", "Peak amplitude error (SSN)"),
        ("peak_time_error_months", "Peak timing error (months)"),
        ("trough_amplitude_error", "Trough amplitude error (SSN)"),
        ("trough_time_error_months", "Trough timing error (months)"),
    )
    for ax, (column, title) in zip(axes.ravel(), panels):
        for offset, variant in zip((-width / 2, width / 2), VARIANTS):
            means = []
            stds = []
            for cycle in CYCLES:
                values = recursive[
                    (recursive["cycle"] == cycle)
                    & (recursive["variant"] == variant)
                ][column].to_numpy(dtype=float)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=0)))
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=2,
                color=COLORS[variant],
                alpha=0.72,
                label=variant,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, CYCLES)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "PR24 recursive block24 extrema | error bars = five-seed std",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUT / "pr24_nonnegative_extrema_errors_2026-11-21"
    fig.savefig(path.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_leadtime(lead_aggregate: pd.DataFrame) -> None:
    recursive = lead_aggregate[lead_aggregate["method"] == "p24_recursive"]
    bins = ("01-24", "25-48", "49-72", "73-96", "97-132")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=240, sharey=True)
    for ax, cycle in zip(axes, CYCLES):
        subset = recursive[recursive["cycle"] == cycle].set_index("horizon_bin").reindex(bins)
        x = np.arange(len(bins))
        ax.plot(
            x,
            subset["clip0_MAE_mean"],
            marker="o",
            color=COLORS["clip0"],
            label="clip0",
        )
        ax.fill_between(
            x,
            subset["clip0_MAE_mean"] - subset["clip0_MAE_std"].fillna(0),
            subset["clip0_MAE_mean"] + subset["clip0_MAE_std"].fillna(0),
            color=COLORS["clip0"],
            alpha=0.15,
        )
        ax.plot(
            x,
            subset["persistence_MAE_mean"],
            marker="s",
            color=COLORS["persistence"],
            label="persistence",
        )
        ax.set_xticks(x, bins, rotation=25, ha="right", fontsize=7)
        ax.set_title(cycle, loc="left", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("MAE")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "PR24 recursive block24 error by months since forecast origin",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUT / "pr24_nonnegative_leadtime_mae_2026-11-21"
    fig.savefig(path.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_peak_response(seed_extrema: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.3), dpi=240)
    markers = {"raw": "o", "clip0": "s"}
    for method, color in (
        ("p24_update", COLORS["update"]),
        ("p24_recursive", COLORS["recursive"]),
    ):
        for variant in VARIANTS:
            subset = seed_extrema[
                (seed_extrema["method"] == method)
                & (seed_extrema["variant"] == variant)
            ]
            ax.scatter(
                subset["peak_true_value"],
                subset["peak_pred_value"],
                marker=markers[variant],
                color=color,
                edgecolor="black",
                linewidth=0.3,
                s=30,
                alpha=0.7,
                label=f"{METHOD_LABELS[method]} {variant}",
            )
    limits = (70, 260)
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True 13-month-smoothed peak SSN")
    ax.set_ylabel("Predicted 13-month-smoothed peak SSN")
    ax.set_title("Cycle-strength response across three held-out cycles", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    fig.tight_layout()
    path = OUT / "pr24_nonnegative_peak_response_2026-11-21"
    fig.savefig(path.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves_dir = OUT / "curves"
    curves_dir.mkdir(exist_ok=True)
    raw_data = pd.read_csv(DATA_CSV, parse_dates=["date"])
    raw_by_date = dict(zip(raw_data["date"], raw_data["ssn"].astype(float)))

    curves: dict[tuple[str, str, int], pd.DataFrame] = {}
    seed_metric_rows = []
    seed_extrema_rows = []
    metric_columns = (
        "mae",
        "rmse",
        "bias",
        "r2",
        "pearson_r",
        "spearman_r",
        "nrmse_by_truth_std",
        "mape_true_ge5_pct",
        "negative_months",
        "min_pred",
        "max_pred",
        "skill_vs_persistence",
    )
    extrema_columns = (
        "peak_amplitude_error",
        "peak_relative_amplitude_error_pct",
        "peak_time_error_months",
        "peak_abs_time_error_months",
        "trough_amplitude_error",
        "trough_relative_amplitude_error_pct",
        "trough_time_error_months",
        "trough_abs_time_error_months",
        "amplitude_ratio_raw",
        "amplitude_ratio_smoothed",
    )

    for cycle in CYCLES:
        cycle_dir = curves_dir / cycle
        cycle_dir.mkdir(exist_ok=True)
        for method in METHODS:
            for seed in SEEDS:
                frame = load_curve(cycle, method, seed)
                frame["pred_clip0"] = np.maximum(frame["pred"], 0.0)
                frame["persistence"] = persistence_prediction(frame, raw_by_date)
                frame.to_csv(cycle_dir / f"{method}_seed{seed}_clip0.csv", index=False)
                curves[(cycle, method, seed)] = frame

                truth = frame["true"].to_numpy(dtype=float)
                persistence = frame["persistence"].to_numpy(dtype=float)
                persistence_mae = float(np.mean(np.abs(persistence - truth)))
                for variant, column in (("raw", "pred"), ("clip0", "pred_clip0")):
                    pred = frame[column].to_numpy(dtype=float)
                    metrics = metric_block(pred, truth)
                    metrics["skill_vs_persistence"] = (
                        1.0 - metrics["mae"] / persistence_mae
                        if persistence_mae > 0
                        else None
                    )
                    metrics.update(
                        {
                            "cycle": cycle,
                            "method": method,
                            "seed": seed,
                            "variant": variant,
                            "persistence_MAE": persistence_mae,
                        }
                    )
                    seed_metric_rows.append(metrics)

                    extrema = extrema_block(pred, truth, frame["date"])
                    extrema.update(
                        {
                            "cycle": cycle,
                            "method": method,
                            "seed": seed,
                            "variant": variant,
                        }
                    )
                    seed_extrema_rows.append(extrema)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_extrema = pd.DataFrame(seed_extrema_rows)
    seed_metrics.to_csv(OUT / "pr24_nonnegative_seed_metrics.csv", index=False)
    seed_extrema.to_csv(OUT / "pr24_nonnegative_seed_extrema.csv", index=False)

    rng = np.random.default_rng(20211121)
    aggregate_metrics_df = aggregate_metrics(
        seed_metrics,
        list(metric_columns),
        rng,
    )
    aggregate_extrema = aggregate_metrics(
        seed_extrema,
        list(extrema_columns),
        rng,
    )
    hit_columns = [
        "peak_hit_3m",
        "peak_hit_6m",
        "peak_hit_12m",
        "trough_hit_3m",
        "trough_hit_6m",
        "trough_hit_12m",
    ]
    hits = (
        seed_extrema.groupby(
            ["method", "variant", "cycle"],
            as_index=False,
        )[hit_columns]
        .mean()
    )
    aggregate_extrema = aggregate_extrema.merge(
        hits,
        on=["method", "variant", "cycle"],
        how="left",
    )
    aggregate_metrics_df.to_csv(
        OUT / "pr24_nonnegative_metric_aggregate.csv",
        index=False,
    )
    aggregate_extrema.to_csv(
        OUT / "pr24_nonnegative_extrema_aggregate.csv",
        index=False,
    )

    ensemble, bands = ensemble_metrics(curves)
    ensemble.to_csv(OUT / "pr24_nonnegative_ensemble_metrics.csv", index=False)
    bands.to_csv(OUT / "pr24_nonnegative_seed_bands.csv", index=False)
    lead_detail, lead_aggregate = leadtime_metrics(curves)
    lead_detail.to_csv(OUT / "pr24_nonnegative_leadtime_detail.csv", index=False)
    lead_aggregate.to_csv(OUT / "pr24_nonnegative_leadtime_aggregate.csv", index=False)

    recursive_clip = aggregate_metrics_df[
        (aggregate_metrics_df["method"] == "p24_recursive")
        & (aggregate_metrics_df["variant"] == "clip0")
    ]
    recursive_extrema_clip = aggregate_extrema[
        (aggregate_extrema["method"] == "p24_recursive")
        & (aggregate_extrema["variant"] == "clip0")
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "B26M-PR24-NONNEG-HOLDOUT",
        "frozen_model": "PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32",
        "data_csv": str(DATA_CSV.relative_to(ROOT)),
        "data_sha256": sha256(DATA_CSV),
        "cycles": CYCLES,
        "seeds": SEEDS,
        "formal_output_gate": "clip0 = max(pred, 0) after physical inverse transform",
        "recursive_is_deployable_path": True,
        "observed_update_uses_test_truth": True,
        "recursive_metrics": recursive_clip.to_dict(orient="records"),
        "recursive_extrema": recursive_extrema_clip.to_dict(orient="records"),
        "ensemble_metrics": ensemble[
            (ensemble["method"] == "p24_recursive")
            & (ensemble["variant"] == "clip0")
        ].to_dict(orient="records"),
    }
    (OUT / "pr24_nonnegative_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    plot_profiles(bands, curves)
    plot_metric_distributions(seed_metrics)
    plot_extrema(seed_extrema)
    plot_leadtime(lead_aggregate)
    plot_peak_response(seed_extrema)
    print(recursive_clip.to_string(index=False))
    print(recursive_extrema_clip.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
