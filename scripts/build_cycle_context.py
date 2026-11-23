"""Build cycle-level context features from the existing SSN series.

Each row describes one solar cycle at a fixed lead of 24 months after its
peak. Features use data available at that origin. Targets describe the next
cycle's trough, peak amplitude, and peak timing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
OUT = ROOT / "analysis/cycle_context"


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.to_period("M") - start.to_period("M")).n)


def main() -> None:
    frame = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    frame = frame.set_index("date")
    raw = frame["ssn"].astype(float)
    smooth = raw.rolling(13, center=True, min_periods=13).mean()
    causal = raw.rolling(13, min_periods=13).mean()

    trough_indices, _ = find_peaks(
        -smooth.to_numpy(),
        distance=96,
        prominence=10,
    )
    trough_dates = smooth.index[trough_indices]
    cycles = []

    for cycle_number, start in enumerate(trough_dates):
        end = (
            trough_dates[cycle_number + 1]
            if cycle_number + 1 < len(trough_dates)
            else smooth.index[-1]
        )
        smooth_interval = smooth.loc[start:end]
        raw_interval = raw.loc[start:end]
        peak_date = smooth_interval.idxmax()
        peak_smooth = float(smooth_interval.max())
        peak_raw = float(raw_interval.max())
        cycles.append(
            {
                "cycle": cycle_number + 1,
                "min_date": start,
                "min_smooth": float(smooth.loc[start]),
                "min_raw": float(raw.loc[start]),
                "peak_date": peak_date,
                "peak_smooth": peak_smooth,
                "peak_raw": peak_raw,
                "next_min_date": (
                    trough_dates[cycle_number + 1]
                    if cycle_number + 1 < len(trough_dates)
                    else pd.NaT
                ),
            }
        )

    rows = []
    for index, cycle in enumerate(cycles):
        peak_index = smooth.index.get_loc(cycle["peak_date"])
        origin_index = min(peak_index + 24, len(smooth) - 1)
        origin_date = smooth.index[origin_index]
        if origin_date < cycle["peak_date"]:
            continue

        post_peak_raw = raw.iloc[peak_index + 1 : origin_index + 1]
        if post_peak_raw.empty:
            continue
        smooth_at_origin = causal.iloc[origin_index]
        next_cycle = cycles[index + 1] if index + 1 < len(cycles) else None
        target_peak_date = next_cycle["peak_date"] if next_cycle else pd.NaT
        target_peak_smooth = next_cycle["peak_smooth"] if next_cycle else np.nan
        target_peak_raw = next_cycle["peak_raw"] if next_cycle else np.nan
        target_min_date = cycle["next_min_date"]

        rows.append(
            {
                "cycle": cycle["cycle"],
                "min_date": cycle["min_date"],
                "peak_date": cycle["peak_date"],
                "origin_date": origin_date,
                "elapsed_since_min": months_between(
                    cycle["min_date"], origin_date
                ),
                "rise_months": months_between(
                    cycle["min_date"], cycle["peak_date"]
                ),
                "months_since_peak": months_between(
                    cycle["peak_date"], origin_date
                ),
                "current_peak_smooth": cycle["peak_smooth"],
                "current_peak_raw": cycle["peak_raw"],
                "current_min_smooth": cycle["min_smooth"],
                "current_min_raw": cycle["min_raw"],
                "smooth_at_origin": float(smooth_at_origin),
                "decline_24m": cycle["peak_smooth"] - float(smooth_at_origin),
                "decline_pct_24m": (
                    cycle["peak_smooth"] - float(smooth_at_origin)
                )
                / cycle["peak_smooth"],
                "postpeak_volatility_24m": float(post_peak_raw.std()),
                "postpeak_abs_slope_24m": float(
                    np.mean(np.abs(np.diff(post_peak_raw)))
                ),
                "previous_peak_smooth": (
                    cycles[index - 1]["peak_smooth"] if index else np.nan
                ),
                "previous_peak_raw": (
                    cycles[index - 1]["peak_raw"] if index else np.nan
                ),
                "previous_cycle_length": (
                    months_between(
                        cycles[index - 1]["min_date"], cycle["min_date"]
                    )
                    if index
                    else np.nan
                ),
                "cycle_parity": cycle["cycle"] % 2,
                "target_min_date": target_min_date,
                "target_peak_date": target_peak_date,
                "target_peak_smooth": target_peak_smooth,
                "target_peak_raw": target_peak_raw,
                "target_months_to_min": (
                    months_between(origin_date, target_min_date)
                    if pd.notna(target_min_date)
                    else np.nan
                ),
                "target_months_to_peak": (
                    months_between(origin_date, target_peak_date)
                    if pd.notna(target_peak_date)
                    else np.nan
                ),
            }
        )

    context = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUT / "cycle_context_2026-10.csv", index=False)
    cycles_df = pd.DataFrame(cycles)
    cycles_df.to_csv(OUT / "cycle_boundaries_2026-10.csv", index=False)
    print(context.to_string(index=False))


if __name__ == "__main__":
    main()
