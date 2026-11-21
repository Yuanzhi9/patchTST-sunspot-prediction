"""Plot true and seed-median curves for chronological one-cycle holdouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/cycle_holdout"
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
COLORS = {
    "p24_update": "#009E73",
    "p24_recursive": "#E69F00",
    "p132_direct": "#0072B2",
}
LABELS = {
    "p24_update": "PR24 observed-update block24",
    "p24_recursive": "PR24 recursive block24",
    "p132_direct": "PR132 direct",
}


def find_result(marker: str) -> Path:
    matches = list((ROOT / "results").glob(f"*_{marker}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result for {marker}, got {len(matches)}")
    return matches[0]


def load_curves(cycle: str) -> dict[str, dict[int, pd.DataFrame]]:
    curves: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in COLORS}
    for seed in SEEDS:
        p24_marker = f"B26M-CHRONO-{cycle}_F1_seed{seed}"
        update_path = BASE / cycle / f"p24_update_seed{seed}/p24_update_curve.csv"
        recursive_path = BASE / cycle / f"p24_update_seed{seed}/p24_recursive_curve.csv"
        p132_path = find_result(f"B26M-P132-CHRONO-{cycle}_F1_seed{seed}") / "p132_curve.csv"
        for method, path in (
            ("p24_update", update_path),
            ("p24_recursive", recursive_path),
            ("p132_direct", p132_path),
        ):
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, parse_dates=["date"])
            frame = frame[["date", "pred", "true"]].sort_values("date").reset_index(drop=True)
            curves[method][seed] = frame
    return curves


def median_band(curves: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = []
    for seed, frame in curves.items():
        merged.append(frame[["date", "pred"]].rename(columns={"pred": f"seed{seed}"}))
    wide = merged[0]
    for frame in merged[1:]:
        wide = wide.merge(frame, on="date", how="inner")
    pred_cols = [column for column in wide.columns if column.startswith("seed")]
    values = wide[pred_cols].to_numpy(dtype=float)
    output = pd.DataFrame({"date": wide["date"]})
    output["median"] = np.median(values, axis=1)
    output["low"] = np.min(values, axis=1)
    output["high"] = np.max(values, axis=1)
    truth = curves[next(iter(curves))][["date", "true"]]
    return output, truth, wide


def smooth(values: pd.Series) -> pd.Series:
    return values.rolling(13, center=True, min_periods=13).mean()


def add_extrema(ax: plt.Axes, dates: pd.Series, values: pd.Series, color: str) -> None:
    smoothed = smooth(values)
    if not smoothed.notna().any():
        return
    peak = int(smoothed.idxmax())
    trough = int(smoothed.idxmin())
    for idx, label in ((peak, "P"), (trough, "T")):
        ax.scatter(dates.iloc[idx], values.iloc[idx], color=color, s=22, zorder=5)
        ax.annotate(
            f"{label} {dates.iloc[idx]:%Y-%m}",
            (dates.iloc[idx], values.iloc[idx]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6.5,
            color=color,
        )


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=230, squeeze=False)
    for ax, cycle in zip(axes[:, 0], CYCLES):
        curves = load_curves(cycle)
        truth = curves["p24_update"][2021][["date", "true"]]
        ax.plot(truth["date"], truth["true"], color="black", lw=1.3, label="Observed")
        for method, seed_curves in curves.items():
            band, _, _ = median_band(seed_curves)
            ax.fill_between(
                band["date"],
                band["low"],
                band["high"],
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                band["date"],
                band["median"],
                color=COLORS[method],
                lw=1.5,
                label=f"{LABELS[method]} median",
            )
            add_extrema(ax, band["date"], band["median"], COLORS[method])
        ax.set_title(
            f"{cycle} | chronological holdout | five-seed median; band=seed range",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Monthly SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if cycle == "SC20":
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    out = BASE / "cycle_holdout_profiles_2026-11-21"
    fig.savefig(out.with_suffix(".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
