"""Audit stability of historical double-peaked solar cycle labels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
OUT = ROOT / "analysis/double_peak_audit"
WINDOWS = (0, 13, 17, 21)
THRESHOLDS = (3.0, 5.0, 10.0)


def smooth(values: pd.Series, window: int) -> pd.Series:
    if window == 0:
        return values
    return values.rolling(window, center=True, min_periods=1).mean()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["date"])
    boundaries = pd.read_csv(
        BOUNDARIES,
        parse_dates=["min_date", "peak_date", "next_min_date"],
    )
    boundaries = boundaries.loc[boundaries["next_min_date"].notna()].copy()

    detail_rows = []
    cycle_rows = []
    configs = [(window, threshold) for window in WINDOWS for threshold in THRESHOLDS]
    for _, cycle in boundaries.iterrows():
        segment = data.loc[
            (data["date"] >= cycle["min_date"])
            & (data["date"] < cycle["next_min_date"])
        ].copy()
        if len(segment) < 24:
            continue
        segment = segment.set_index("date")
        flags = []
        for window, threshold in configs:
            smoothed = smooth(segment["ssn"], window)
            peaks, properties = find_peaks(
                smoothed.to_numpy(),
                distance=6,
                prominence=threshold,
            )
            peaks = np.asarray(peaks, dtype=int)
            prominences = np.asarray(properties.get("prominences", []), dtype=float)
            order = np.argsort(smoothed.to_numpy()[peaks])[::-1] if len(peaks) else []
            top = peaks[order[:2]]
            top_values = smoothed.to_numpy()[top] if len(top) else np.array([])
            top_dates = [smoothed.index[i].strftime("%Y-%m") for i in top]
            is_double = bool(len(peaks) >= 2)
            flags.append(is_double)
            detail_rows.append(
                {
                    "cycle": int(cycle["cycle"]),
                    "window": window,
                    "threshold": threshold,
                    "n_peaks": int(len(peaks)),
                    "is_double": is_double,
                    "peak1_date": top_dates[0] if len(top_dates) else None,
                    "peak1_value": float(top_values[0]) if len(top_values) else None,
                    "peak2_date": top_dates[1] if len(top_dates) > 1 else None,
                    "peak2_value": float(top_values[1]) if len(top_values) > 1 else None,
                    "max_prominence": float(prominences.max()) if len(prominences) else None,
                }
            )
        support = float(np.mean(flags))
        if support >= 0.75:
            stability = "stable_double"
        elif support <= 0.25:
            stability = "stable_single"
        else:
            stability = "ambiguous"
        cycle_rows.append(
            {
                "cycle": int(cycle["cycle"]),
                "min_date": cycle["min_date"].strftime("%Y-%m"),
                "peak_date": cycle["peak_date"].strftime("%Y-%m"),
                "next_min_date": cycle["next_min_date"].strftime("%Y-%m"),
                "double_support": support,
                "stability": stability,
            }
        )

    detail = pd.DataFrame(detail_rows)
    stability = pd.DataFrame(cycle_rows)
    detail.to_csv(OUT / "peak_config_detail.csv", index=False)
    stability.to_csv(OUT / "cycle_stability.csv", index=False)
    robust = stability.loc[stability["stability"] == "stable_double"].copy()
    robust.to_csv(OUT / "robust_double_cycles.csv", index=False)
    summary = {
        "cycles_audited": int(len(stability)),
        "stable_double": int((stability["stability"] == "stable_double").sum()),
        "stable_single": int((stability["stability"] == "stable_single").sum()),
        "ambiguous": int((stability["stability"] == "ambiguous").sum()),
        "stable_double_cycles": robust["cycle"].astype(int).tolist(),
        "all_config_consistent_cycles": int(
            detail.groupby("cycle")["is_double"].nunique().eq(1).sum()
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    matrix = (
        detail.assign(config=lambda frame: frame["window"].astype(str) + "m-p" + frame["threshold"].astype(str))
        .pivot(index="cycle", columns="config", values="is_double")
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index, fontsize=7)
    ax.set_ylabel("Solar cycle")
    ax.set_title("Double-peak label stability across smoothing and prominence settings", loc="left")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"double_peak_stability{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(stability.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
