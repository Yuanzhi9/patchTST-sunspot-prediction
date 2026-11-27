"""Assemble real forecasts and apply R1A local peak corrections."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
SMOOTH_DATA = (
    ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_smoothed13_2026-04.csv"
)
OLD_FUTURE = ROOT / "analysis/pr24_future_forecast"
MT_FUTURE = ROOT / "analysis/pr24_multitask_future_forecast"
OUT = ROOT / "analysis/pr24_multitask_future_correction"
SEEDS = (2019, 2020, 2021, 2022, 2023)
WIDTHS = (12, 24, 36)


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1A = load_module("r1a_constrained_peak", "scripts/eval_constrained_peak_2026-11-23.py")
R1B_RAW = load_module("r1b_raw_peak", "scripts/eval_constrained_peak_raw_2026-11-23.py")


def load_old_raw(curves: dict[int, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = None
    for seed in SEEDS:
        frame = pd.read_csv(curves[seed], parse_dates=["date"])
        frame = frame.loc[frame["mode"] == "clip0", ["date", "pred"]].copy()
        frame = frame.rename(columns={"pred": f"seed{seed}"})
        wide = frame if wide is None else wide.merge(frame, on="date", how="inner")
    values = wide[[f"seed{seed}" for seed in SEEDS]].to_numpy(dtype=float)
    raw = pd.DataFrame(
        {
            "date": wide["date"],
            "median": np.median(values, axis=1),
            "q10": np.quantile(values, 0.1, axis=1),
            "q90": np.quantile(values, 0.9, axis=1),
        }
    )
    raw.to_csv(out_dir / "old_pr24_raw_profile.csv", index=False)
    return raw


def derive_old_smooth13(
    raw_data: pd.DataFrame,
    curves: dict[int, pd.DataFrame],
    out_dir: Path,
) -> pd.DataFrame:
    origin = raw_data["date"].iloc[-1]
    history = raw_data[
        raw_data["date"] >= origin - pd.DateOffset(months=180)
    ][["date", "ssn"]]
    series = {}
    for seed in SEEDS:
        frame = pd.read_csv(curves[seed], parse_dates=["date"])
        frame = frame.loc[frame["mode"] == "clip0", ["date", "pred"]].copy()
        combined = pd.concat(
            [history, frame.rename(columns={"pred": "ssn"})], ignore_index=True
        )
        combined["smooth13"] = (
            combined["ssn"].rolling(13, center=True, min_periods=13).mean()
        )
        future = combined.loc[combined["date"] > origin, ["date", "smooth13"]]
        series[seed] = future.rename(columns={"smooth13": f"seed{seed}"})
    wide = series[SEEDS[0]]
    for seed in SEEDS[1:]:
        wide = wide.merge(series[seed], on="date", how="inner")
    values = wide[[f"seed{seed}" for seed in SEEDS]].to_numpy(dtype=float)
    value_frame = pd.DataFrame(values)
    smooth = pd.DataFrame(
        {
            "date": wide["date"],
            "median": value_frame.median(axis=1, skipna=True),
            "q10": value_frame.quantile(0.1, axis=1),
            "q90": value_frame.quantile(0.9, axis=1),
        }
    )
    smooth = smooth.loc[smooth["median"].notna()].reset_index(drop=True)
    smooth.to_csv(out_dir / "old_pr24_smooth13_profile.csv", index=False)
    return smooth


def load_mt_profile(track: str) -> pd.DataFrame:
    source = MT_FUTURE / f"{track}_target"
    frames = []
    for seed in SEEDS:
        path = source / f"forecast_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame.loc[frame["mode"] == "clip0", ["date", "pred"]]
        frames.append(frame.rename(columns={"pred": f"seed{seed}"}))
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="date", how="inner")
    values = wide[[f"seed{seed}" for seed in SEEDS]].to_numpy(dtype=float)
    profile = pd.DataFrame(
        {
            "date": wide["date"],
            "median": np.median(values, axis=1),
            "q10": np.quantile(values, 0.1, axis=1),
            "q90": np.quantile(values, 0.9, axis=1),
            "min": np.min(values, axis=1),
            "max": np.max(values, axis=1),
        }
    )
    profile.to_csv(source / "forecast_seed_band.csv", index=False)
    return profile


def future_extrema_by_seed(tracks: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for track in tracks:
        source = MT_FUTURE / f"{track}_target"
        observed_path = RAW_DATA if track == "raw" else SMOOTH_DATA
        observed = pd.read_csv(observed_path, parse_dates=["date"])
        origin = observed["date"].iloc[-1]
        history = observed[
            observed["date"] >= origin - pd.DateOffset(months=180)
        ][["date", "ssn"]]
        for seed in SEEDS:
            frame = pd.read_csv(source / f"forecast_seed{seed}.csv", parse_dates=["date"])
            frame = frame.loc[frame["mode"] == "clip0"].reset_index(drop=True)
            combined = pd.concat(
                [history, frame[["date", "pred"]].rename(columns={"pred": "ssn"})],
                ignore_index=True,
            )
            combined["smooth13"] = (
                combined["ssn"].rolling(13, center=True, min_periods=13).mean()
            )
            origin_index = int(combined.index[combined["date"] == origin][0])
            trough_candidates = combined.iloc[
                origin_index + 1 : origin_index + 85
            ].dropna(subset=["smooth13"])
            trough_index = int(trough_candidates["smooth13"].idxmin())
            peak_candidates = combined.iloc[
                trough_index + 24 : min(origin_index + 169, len(combined))
            ].dropna(subset=["smooth13"])
            peak_index = int(peak_candidates["smooth13"].idxmax())
            rows.append(
                {
                    "track": track,
                    "seed": seed,
                    "origin_date": origin.strftime("%Y-%m"),
                    "trough_date": combined.loc[trough_index, "date"].strftime("%Y-%m"),
                    "trough_value": float(combined.loc[trough_index, "ssn"]),
                    "raw_peak_value": (
                        float(combined.loc[peak_index, "ssn"])
                        if track == "raw"
                        else np.nan
                    ),
                    "smooth13_peak_value": float(
                        combined.loc[peak_index, "smooth13"]
                    ),
                    "peak_date": combined.loc[peak_index, "date"].strftime("%Y-%m"),
                }
            )
    detail = pd.DataFrame(rows)
    aggregate_rows = []
    for track_value, group in detail.groupby("track"):
        aggregate_rows.append(
            {
                "track": track_value,
                "origin_date": group["origin_date"].iloc[0],
                "trough_date_median": group["trough_date"].mode().iloc[0],
                "trough_value_median": float(group["trough_value"].median()),
                "trough_value_min": float(group["trough_value"].min()),
                "trough_value_max": float(group["trough_value"].max()),
                "peak_date_median": pd.Timestamp(
                    int(
                        pd.to_datetime(group["peak_date"] + "-01")
                        .astype("int64")
                        .median()
                    )
                ).strftime("%Y-%m"),
                "raw_peak_value_median": (
                    float(group["raw_peak_value"].dropna().median())
                    if group["raw_peak_value"].notna().any()
                    else np.nan
                ),
                "smooth13_peak_value_median": float(
                    group["smooth13_peak_value"].median()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    detail.to_csv(OUT / "mt_future_extrema_by_seed.csv", index=False)
    aggregate.to_csv(OUT / "mt_future_extrema_aggregate.csv", index=False)
    return detail, aggregate


def peak_comparison_table(
    old_raw_corr: pd.DataFrame,
    old_smooth_corr: pd.DataFrame,
    mt_raw_corr: pd.DataFrame,
    mt_smooth_corr: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(curve: str, origin: str, data: pd.DataFrame, column: str) -> None:
        valid = data[["date", column]].dropna()
        index = int(valid[column].idxmax())
        rows.append(
            {
                "curve": curve,
                "origin": origin,
                "peak_date": valid.loc[index, "date"].strftime("%Y-%m"),
                "peak_value": float(valid.loc[index, column]),
            }
        )

    add("Old PR24 raw", "2026-10", old_raw_corr, "median")
    add("Old PR24 raw + R1A/R1B-Ridge H24", "2026-10", old_raw_corr, "R1B-Ridge_H24")
    add("Old PR24 raw + R1A/R1B-RF H24", "2026-10", old_raw_corr, "R1B-RF_H24")
    add("Old PR24 13m profile", "2026-10", old_smooth_corr, "median")
    add("Old PR24 13m + R1A/R1B-Ridge H24", "2026-10", old_smooth_corr, "R1B-Ridge_H24")
    add("Old PR24 13m + R1A/R1B-RF H24", "2026-10", old_smooth_corr, "R1B-RF_H24")
    add("Multitask raw", "2026-10", mt_raw_corr, "median")
    add("Multitask raw + R1A/R1B-Ridge H24", "2026-10", mt_raw_corr, "R1B-Ridge_H24")
    add("Multitask raw + R1A/R1B-RF H24", "2026-10", mt_raw_corr, "R1B-RF_H24")
    add("Multitask 13m", "2026-04", mt_smooth_corr, "median")
    add("Multitask 13m + R1A/R1B-Ridge H24", "2026-04", mt_smooth_corr, "R1B-Ridge_H24")
    add("Multitask 13m + R1A/R1B-RF H24", "2026-04", mt_smooth_corr, "R1B-RF_H24")
    return pd.DataFrame(rows)


def calibrate(curve: np.ndarray, target_peak: float, width: int) -> tuple[np.ndarray, dict]:
    baseline = np.clip(curve.astype(float), 0.0, None)
    peak_index = int(np.argmax(baseline))
    kernel = R1A.raised_cosine_kernel(len(baseline), peak_index, width)
    corrected = R1B_RAW.calibrate_raw_peak(baseline, kernel, target_peak)
    corrected_peak_index = int(np.argmax(corrected))
    return corrected, {
        "width": width,
        "baseline_peak": float(baseline[peak_index]),
        "target_peak": float(target_peak),
        "corrected_peak": float(corrected[corrected_peak_index]),
        "peak_shift_months": corrected_peak_index - peak_index,
    }


def correction_methods(target_kind: str) -> dict[str, float]:
    if target_kind == "raw":
        return {
            "R1B-Ridge": 205.84797732120055,
            "R1B-RF": 190.03846165291333,
        }
    return {
        "R1B-Ridge": 139.83702400729265,
        "R1B-RF": 126.99576536771495,
    }


def build_corrections(
    profile: pd.DataFrame,
    target_kind: str,
    source_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = profile.loc[profile["median"].notna()].copy().reset_index(drop=True)
    data = valid[["date", "median"]].copy()
    rows = []
    for source, target in correction_methods(target_kind).items():
        for width in WIDTHS:
            key = f"{source}_H{width}"
            try:
                corrected, meta = calibrate(
                    valid["median"].to_numpy(dtype=float), target, width
                )
                data[key] = corrected
                rows.append(
                    {
                        "curve_source": source_label,
                        "target_kind": target_kind,
                        "peak_source": source,
                        "half_width_months": width,
                        "status": "ok",
                        **meta,
                    }
                )
            except RuntimeError as error:
                rows.append(
                    {
                        "curve_source": source_label,
                        "target_kind": target_kind,
                        "peak_source": source,
                        "half_width_months": width,
                        "status": "failed",
                        "error": str(error),
                        "target_peak": float(target),
                    }
                )
    return data, pd.DataFrame(rows)


def configure_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_corrections(
    raw_data: pd.DataFrame,
    smooth_data: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    colors = {
        "R1B-Ridge": {12: "#56B4E9", 24: "#0072B2", 36: "#004C7A"},
        "R1B-RF": {12: "#E69F00", 24: "#D55E00", 36: "#8C3B00"},
    }
    line_styles = {12: ":", 24: "-", 36: "--"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=260)
    for ax, data, target_kind, ylabel in (
        (axes[0], raw_data, "raw", "Monthly raw SSN"),
        (axes[1], smooth_data, "13m target", "13-month smoothed SSN"),
    ):
        ax.plot(
            data["date"],
            data["median"],
            color="black",
            lw=1.8,
            label="Uncorrected profile",
        )
        for source in ("R1B-Ridge", "R1B-RF"):
            for width in WIDTHS:
                key = f"{source}_H{width}"
                if key not in data.columns:
                    continue
                ax.plot(
                    data["date"],
                    data[key],
                    color=colors[source][width],
                    ls=line_styles[width],
                    lw=1.15,
                    label=f"R1A {source} H={width}",
                )
        configure_axis(ax, ylabel)
        ax.set_title(target_kind, loc="left", fontsize=10, fontweight="bold")
        missing_h12 = not {
            "R1B-Ridge_H12",
            "R1B-RF_H12",
        }.issubset(data.columns)
        if missing_h12:
            ax.text(
                0.02,
                0.96,
                "H=12 unavailable for downward flat-peak correction",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                color="#555555",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_real_forecast(
    raw_data: pd.DataFrame,
    raw_profile: pd.DataFrame,
    smooth_data: pd.DataFrame,
    smooth_profile: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=260)
    for ax, observed, years, profile, color, label, origin in (
        (
            axes[0],
            raw_data,
            15,
            raw_profile,
            "#D55E00",
            "Raw-target multitask forecast",
            "2026-10",
        ),
        (
            axes[1],
            smooth_data,
            15,
            smooth_profile,
            "#0072B2",
            "13m-target multitask forecast",
            "2026-04",
        ),
    ):
        recent = observed.loc[
            observed["date"] >= observed["date"].iloc[-1] - pd.DateOffset(years=years)
        ]
        ax.plot(recent["date"], recent["ssn"], color="black", lw=1.25, label="Observed")
        ax.fill_between(
            profile["date"],
            profile["q10"],
            profile["q90"],
            color=color,
            alpha=0.16,
            linewidth=0,
            label="Five-seed 10–90% range",
        )
        ax.plot(profile["date"], profile["median"], color=color, lw=1.8, label=label)
        peak_date = profile.loc[profile["median"].idxmax(), "date"]
        peak_value = float(profile["median"].max())
        ax.scatter(peak_date, peak_value, color=color, s=28, zorder=5)
        ax.annotate(
            f"SC26 peak\n{peak_date:%Y-%m}, {peak_value:.1f}",
            (peak_date, peak_value),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
        configure_axis(ax, "SSN")
        ax.set_title(
            f"{label} | origin {origin}",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(
        "Best multitask structure: real forecasts on separate raw and 13m tracks",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_data = pd.read_csv(RAW_DATA, parse_dates=["date"])
    smooth_data = pd.read_csv(SMOOTH_DATA, parse_dates=["date"])
    old_paths = {
        seed: OLD_FUTURE / f"forecast_seed{seed}.csv" for seed in SEEDS
    }
    old_raw = load_old_raw(old_paths, OUT)
    old_smooth = derive_old_smooth13(raw_data, old_paths, OUT)
    mt_raw = load_mt_profile("raw")
    mt_smooth = load_mt_profile("smooth13")
    future_extrema_by_seed(("raw", "smooth13"))

    plot_real_forecast(
        raw_data,
        mt_raw,
        smooth_data,
        mt_smooth,
        OUT / "mt_real_forecast_raw_and_13m",
    )

    old_raw_corr, old_raw_meta = build_corrections(old_raw, "raw", "old_PR24_raw")
    old_smooth_corr, old_smooth_meta = build_corrections(
        old_smooth, "smooth", "old_PR24_smooth13"
    )
    old_raw_corr.to_csv(OUT / "old_raw_with_R1A_R1B.csv", index=False)
    old_smooth_corr.to_csv(OUT / "old_smooth13_with_R1A_R1B.csv", index=False)
    pd.concat([old_raw_meta, old_smooth_meta], ignore_index=True).to_csv(
        OUT / "old_R1A_R1B_metadata.csv", index=False
    )
    plot_corrections(
        old_raw_corr,
        old_smooth_corr,
        OUT / "old_profiles_with_R1A_R1B",
        "Old PR24 future profiles with R1A local peak correction and R1B amplitudes",
    )

    mt_raw_corr, mt_raw_meta = build_corrections(
        mt_raw, "raw", "multitask_raw"
    )
    mt_smooth_corr, mt_smooth_meta = build_corrections(
        mt_smooth, "smooth", "multitask_smooth13"
    )
    mt_raw_corr.to_csv(OUT / "mt_raw_with_R1A_R1B.csv", index=False)
    mt_smooth_corr.to_csv(OUT / "mt_smooth13_with_R1A_R1B.csv", index=False)
    pd.concat([mt_raw_meta, mt_smooth_meta], ignore_index=True).to_csv(
        OUT / "mt_R1A_R1B_metadata.csv", index=False
    )
    plot_corrections(
        mt_raw_corr,
        mt_smooth_corr,
        OUT / "mt_profiles_with_R1A_R1B",
        "Multitask future profiles with R1A local peak correction and R1B amplitudes",
    )
    peak_comparison_table(
        old_raw_corr, old_smooth_corr, mt_raw_corr, mt_smooth_corr
    ).to_csv(OUT / "future_peak_comparison.csv", index=False)

    summary = {
        "status": "complete",
        "raw_origin": raw_data["date"].iloc[-1].strftime("%Y-%m"),
        "smooth13_origin": smooth_data["date"].iloc[-1].strftime("%Y-%m"),
        "r1b_targets": {
            "raw": correction_methods("raw"),
            "smooth13": correction_methods("smooth"),
        },
        "r1a_widths_months": list(WIDTHS),
        "note": "R1A Oracle is not available for the future because SC26 truth is unknown.",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
