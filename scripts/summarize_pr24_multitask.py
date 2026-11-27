"""Summarize PR24 multitask CV5 and chronological holdout evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cv_rows = []
    for path in sorted((ROOT / "analysis/pr24_multitask_extrema").glob("A5_all_seed*/cv5_aggregate.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cv_rows.append(
            {
                "seed": payload["seed"],
                "full_MAE": payload["full_MAE_mean"],
                "block24_MAE": payload["block24_MAE_mean"],
            }
        )
    cv = pd.DataFrame(cv_rows).drop_duplicates("seed").sort_values("seed")
    baseline = []
    for seed in (2019, 2020, 2022, 2023):
        path = ROOT / f"analysis/B26M-F-P36-SEED{seed}_seed{seed}/cv5_aggregate.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline.append(
            {
                "seed": seed,
                "full_MAE": payload["full_MAE_mean"],
                "block24_MAE": payload["block24_MAE_mean"],
            }
        )
    seed2021 = json.loads(
        (ROOT / "analysis/pr24_multitask_extrema/A1_baseline_seed2021/cv5_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    baseline.append(
        {
            "seed": 2021,
            "full_MAE": seed2021["full_MAE_mean"],
            "block24_MAE": seed2021["block24_MAE_mean"],
        }
    )
    baseline = pd.DataFrame(baseline).sort_values("seed")
    comparison = baseline.merge(cv, on="seed", suffixes=("_base", "_mt"))
    comparison["full_delta_pct"] = (
        comparison["full_MAE_mt"] / comparison["full_MAE_base"] - 1.0
    ) * 100.0
    comparison["block24_delta_pct"] = (
        comparison["block24_MAE_mt"] / comparison["block24_MAE_base"] - 1.0
    ) * 100.0

    out = ROOT / "analysis/pr24_multitask_extrema"
    comparison.to_csv(out / "cv5_five_seed_comparison.csv", index=False)
    payload = {
        "baseline_full_mean": float(baseline["full_MAE"].mean()),
        "baseline_block24_mean": float(baseline["block24_MAE"].mean()),
        "multitask_full_mean": float(cv["full_MAE"].mean()),
        "multitask_block24_mean": float(cv["block24_MAE"].mean()),
        "multitask_full_seed_std": float(cv["full_MAE"].std(ddof=0)),
        "multitask_block24_seed_std": float(cv["block24_MAE"].std(ddof=0)),
        "seeds_improved_full": int((comparison["full_delta_pct"] < 0).sum()),
        "seeds_improved_block24": int((comparison["block24_delta_pct"] < 0).sum()),
        "decision": "no_stable_net_gain_keep_research_branch",
    }
    (out / "cv5_five_seed_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
