"""Summarize chronological one-cycle holdout extrema across seeds and methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/cycle_holdout"
CYCLES = ("SC20", "SC21", "SC24")
METHODS = (
    "p24_update",
    "p24_recursive",
    "p24_recursive_b12",
    "p24_recursive_b6",
    "p132_direct",
)
SEEDS = (2019, 2020, 2021, 2022, 2023)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="comma-separated method prefixes",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="comma-separated integer seeds",
    )
    parser.add_argument(
        "--output-prefix",
        default="cycle_holdout",
        help="prefix for detail/aggregate output files",
    )
    args = parser.parse_args()
    base = Path(args.base)
    methods = tuple(value for value in args.methods.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)

    rows = []
    for cycle in CYCLES:
        for method in methods:
            for seed in seeds:
                path = base / cycle / f"{method}_seed{seed}_extrema.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                peak = payload["smoothed"]["peak"]
                trough = payload["smoothed"]["trough"]
                raw = payload["raw_metrics"]
                rows.append(
                    {
                        "cycle": cycle,
                        "method": method,
                        "seed": seed,
                        "raw_MAE": raw["mae"],
                        "raw_RMSE": raw["rmse"],
                        "raw_bias": raw["bias"],
                        "raw_R2": raw["r2"],
                        "negative_months": raw["negative_months"],
                        "peak_true_date": peak["true_date"],
                        "peak_true_value": peak["true_value"],
                        "peak_pred_date": peak["pred_date"],
                        "peak_pred_value": peak["pred_value"],
                        "peak_amplitude_error": peak["amplitude_error"],
                        "peak_time_error_months": peak["time_error_months"],
                        "peak_abs_time_error_months": abs(peak["time_error_months"]),
                        "trough_true_date": trough["true_date"],
                        "trough_true_value": trough["true_value"],
                        "trough_pred_date": trough["pred_date"],
                        "trough_pred_value": trough["pred_value"],
                        "trough_amplitude_error": trough["amplitude_error"],
                        "trough_time_error_months": trough["time_error_months"],
                        "trough_abs_time_error_months": abs(trough["time_error_months"]),
                        "amplitude_ratio": payload["smoothed"]["amplitude_ratio"],
                    }
                )
    if not rows:
        raise SystemExit("no cycle holdout extrema results found")
    detail = pd.DataFrame(rows).sort_values(["cycle", "method", "seed"])
    detail.to_csv(base / f"{args.output_prefix}_seed_detail.csv", index=False)

    numeric = [
        "raw_MAE",
        "raw_RMSE",
        "raw_bias",
        "raw_R2",
        "negative_months",
        "peak_amplitude_error",
        "peak_time_error_months",
        "peak_abs_time_error_months",
        "trough_amplitude_error",
        "trough_time_error_months",
        "trough_abs_time_error_months",
        "amplitude_ratio",
    ]
    grouped = detail.groupby(["cycle", "method"], sort=True)
    aggregate = grouped[numeric].agg(["mean", "std"]).reset_index()
    aggregate.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]
    aggregate.to_csv(base / f"{args.output_prefix}_aggregate.csv", index=False)
    (base / f"{args.output_prefix}_aggregate.json").write_text(
        aggregate.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(base / f"{args.output_prefix}_aggregate.csv")


if __name__ == "__main__":
    main()
