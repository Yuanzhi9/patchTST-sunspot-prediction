"""Summarize evaluated p132 CV5 configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize p132 analysis directories")
    parser.add_argument("--analysis-root", default="analysis_p132")
    parser.add_argument("--prefix", default="B26M-P132")
    parser.add_argument("--out-dir", default="analysis_p132/summary")
    args = parser.parse_args()

    root = Path(args.analysis_root)
    rows = []
    for path in sorted(root.glob("*/cv5_aggregate.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dir_name = path.parent.name
        if args.prefix not in dir_name:
            continue
        config_id, seed_text = dir_name.rsplit("_seed", 1)
        rows.append(
            {
                "config": config_id,
                "seed": int(seed_text),
                "cycle_MAE_mean": payload["cycle_MAE"]["mean"],
                "cycle_MAE_std": payload["cycle_MAE"]["std"],
                "cycle_MAE_worst": payload["cycle_MAE"]["worst"],
                "full_MAE_mean": payload["full_MAE"]["mean"],
                "peak_abs_amp_mean": payload["peak_abs_amp"]["mean"],
                "peak_abs_time_mean": payload["peak_abs_time"]["mean"],
                "trough_abs_amp_mean": payload["trough_abs_amp"]["mean"],
                "trough_abs_time_mean": payload["trough_abs_time"]["mean"],
            }
        )
    if not rows:
        raise SystemExit("no p132 aggregate files found")

    frame = pd.DataFrame(rows).sort_values(
        ["seed", "cycle_MAE_mean", "cycle_MAE_worst"]
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "p132_search_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    for seed, group in frame.groupby("seed"):
        ax.scatter(
            group["cycle_MAE_mean"],
            group["cycle_MAE_worst"],
            s=38,
            label=f"seed {seed}",
        )
        for _, row in group.iterrows():
            label = row["config"].replace("B26M-P132-", "")
            ax.annotate(label, (row["cycle_MAE_mean"], row["cycle_MAE_worst"]), fontsize=6)
    ax.set_xlabel("Direct-cycle MAE mean (SSN)")
    ax.set_ylabel("Direct-cycle MAE worst fold (SSN)")
    ax.set_title("p132 parameter search: average vs worst fold", loc="left", fontsize=10)
    ax.grid(color="#DDDDDD", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out_png = out_dir / "p132_search_tradeoff.png"
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
