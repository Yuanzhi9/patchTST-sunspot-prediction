"""Summarize the intermediate d_model audit with paired seed comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/p24_capacity_refine"
BASELINE_CSV = ROOT / "analysis/p24_dual_summary/candidate_five_seed_detail.csv"


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV).rename(
        columns={
            "full_MAE": "baseline_full",
            "block24_MAE": "baseline_block24",
            "full_worst": "baseline_full_worst",
            "block24_worst": "baseline_block24_worst",
        }
    )
    rows = []
    for path in sorted(BASE.glob("B26M-CAPR-D*_seed*/cv5_aggregate.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "config": payload["des"],
                "d_model": int(payload["des"].split("-D")[-1]),
                "seed": int(payload["seed"]),
                "full_MAE": float(payload["full_MAE_mean"]),
                "full_std_folds": float(payload["full_MAE_std"]),
                "full_worst_fold": float(payload["full_MAE_worst"]),
                "block24_MAE": float(payload["block24_MAE_mean"]),
                "block24_std_folds": float(payload["block24_MAE_std"]),
                "block24_worst_fold": float(payload["block24_MAE_worst"]),
            }
        )
    detail = pd.DataFrame(rows).merge(
        baseline[["seed", "baseline_full", "baseline_block24", "baseline_full_worst", "baseline_block24_worst"]],
        on="seed",
        how="left",
    )
    detail["full_delta"] = detail["full_MAE"] - detail["baseline_full"]
    detail["block24_delta"] = detail["block24_MAE"] - detail["baseline_block24"]
    detail["full_pct_delta"] = 100.0 * detail["full_delta"] / detail["baseline_full"]
    detail["block24_pct_delta"] = (
        100.0 * detail["block24_delta"] / detail["baseline_block24"]
    )
    detail["both_better"] = (detail["full_delta"] < 0) & (detail["block24_delta"] < 0)
    detail.to_csv(BASE / "capacity_refine_seed_paired.csv", index=False)

    aggregate = (
        detail.groupby(["config", "d_model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            full_MAE_mean=("full_MAE", "mean"),
            full_MAE_std_seeds=("full_MAE", "std"),
            full_worst_fold=("full_worst_fold", "max"),
            block24_MAE_mean=("block24_MAE", "mean"),
            block24_MAE_std_seeds=("block24_MAE", "std"),
            block24_worst_fold=("block24_worst_fold", "max"),
            paired_full_pct_delta_mean=("full_pct_delta", "mean"),
            paired_block24_pct_delta_mean=("block24_pct_delta", "mean"),
            both_better_seeds=("both_better", "sum"),
        )
        .fillna({"full_MAE_std_seeds": 0.0, "block24_MAE_std_seeds": 0.0})
    )
    aggregate["full_win_seeds"] = (
        detail.assign(win=detail["full_delta"] < 0)
        .groupby("config")["win"]
        .sum()
        .reindex(aggregate["config"])
        .to_numpy()
    )
    aggregate["block24_win_seeds"] = (
        detail.assign(win=detail["block24_delta"] < 0)
        .groupby("config")["win"]
        .sum()
        .reindex(aggregate["config"])
        .to_numpy()
    )
    aggregate["five_seed_complete"] = aggregate["n_seeds"].eq(5)
    aggregate["eligible_replace"] = (
        aggregate["five_seed_complete"]
        & (aggregate["both_better_seeds"] >= 4)
        & (aggregate["paired_full_pct_delta_mean"] <= -3.0)
        & (aggregate["paired_block24_pct_delta_mean"] <= -3.0)
    )
    aggregate = aggregate.sort_values(
        ["five_seed_complete", "block24_MAE_mean", "full_MAE_mean"],
        ascending=[False, True, True],
    )
    aggregate.to_csv(BASE / "capacity_refine_summary.csv", index=False)
    (BASE / "capacity_refine_summary.json").write_text(
        aggregate.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
