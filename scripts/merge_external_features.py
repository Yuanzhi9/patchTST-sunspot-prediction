"""Merge external monthly indices onto the canonical monthly SSN table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "PatchTST_supervised/dataset"
DEFAULT_BASE = DATASET / "sunspot_with_cycle_monthly_2026-10.csv"
DEFAULT_F107 = DATASET / "external/processed/f107_monthly_1947_2026.csv"
DEFAULT_POLAR = DATASET / "external/processed/wso_polar_monthly_1976_2026.csv"
DEFAULT_OUTPUT = DATASET / "external/processed/external_features_monthly_2026-10.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--f107", type=Path, default=DEFAULT_F107)
    parser.add_argument("--polar", type=Path, default=DEFAULT_POLAR)
    parser.add_argument("--aa", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = pd.read_csv(args.base, parse_dates=["date"])
    f107 = pd.read_csv(args.f107, parse_dates=["date"])
    polar = pd.read_csv(args.polar, parse_dates=["date"])

    frame = base.merge(
        f107[
            [
                "date",
                "f107_observed",
                "f107_adjusted",
                "f107_absolute",
                "coverage",
                "missing",
            ]
        ].rename(
            columns={
                "coverage": "f107_coverage",
                "missing": "f107_missing",
            }
        ),
        on="date",
        how="left",
    )
    frame = frame.merge(
        polar[
            [
                "date",
                "polar_north_filtered",
                "polar_south_filtered",
                "polar_average_filtered",
                "coverage",
                "missing",
            ]
        ].rename(
            columns={
                "coverage": "polar_coverage",
                "missing": "polar_missing",
            }
        ),
        on="date",
        how="left",
    )

    if args.aa:
        aa = pd.read_csv(args.aa, parse_dates=["date"])
        frame = frame.merge(aa, on="date", how="left")

    for column in ("f107_missing", "polar_missing"):
        frame[column] = frame[column].fillna(1).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    summary = {
        "rows": int(len(frame)),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "f107_valid": int(frame["f107_observed"].notna().sum()),
        "f107_first": str(
            frame.loc[frame["f107_observed"].notna(), "date"].min().date()
        ),
        "f107_last": str(
            frame.loc[frame["f107_observed"].notna(), "date"].max().date()
        ),
        "f107_missing_after_1947": int(
            frame.loc[frame["date"] >= "1947-02-01", "f107_missing"].sum()
        ),
        "polar_valid": int(frame["polar_average_filtered"].notna().sum()),
        "polar_first": str(
            frame.loc[
                frame["polar_average_filtered"].notna(), "date"
            ].min().date()
        ),
        "polar_last": str(
            frame.loc[
                frame["polar_average_filtered"].notna(), "date"
            ].max().date()
        ),
        "polar_missing_after_1976": int(
            frame.loc[frame["date"] >= "1976-05-01", "polar_missing"].sum()
        ),
        "aa_present": args.aa is not None,
    }
    (args.output.parent / "external_features_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
