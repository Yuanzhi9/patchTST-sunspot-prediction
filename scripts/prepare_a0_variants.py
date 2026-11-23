"""Prepare date-matched raw and official-smoothed SSN A0 variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "PatchTST_supervised/dataset"
RAW = DATASET / "sunspot_with_cycle_monthly_2026-10.csv"
SMOOTHED = DATASET / "sunspot_with_cycle_smoothed13_2026-04.csv"
START = pd.Timestamp("1749-07-01")
END = pd.Timestamp("2026-04-01")


def prepare(path: Path, output: Path) -> None:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.loc[frame["date"].between(START, END)].copy()
    if frame.empty:
        raise RuntimeError(f"{path} has no rows in the requested window")
    frame.to_csv(output, index=False)
    print(f"{output}: {len(frame)} rows, {frame.date.min():%Y-%m}..{frame.date.max():%Y-%m}")


def main() -> None:
    prepare(RAW, DATASET / "sunspot_a0_raw_1749-07_2026-04.csv")
    prepare(
        SMOOTHED,
        DATASET / "sunspot_a0_smoothed13_1749-07_2026-04.csv",
    )


if __name__ == "__main__":
    main()
