"""Summarize p24 lr x batch completion and final-candidate seed robustness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LR_BATCH = {
    ("3e-4", "16"): {
        2019: "B26M-LV-P36-3E4-SEED2019",
        2020: "B26M-LV-P36-3E4-SEED2020",
        2021: "B26M-L-P36-3E4",
    },
    ("3e-4", "32"): {
        2019: "B26M-F-P36-SEED2019",
        2020: "B26M-F-P36-SEED2020",
        2021: "B26M-B-P36-BS32",
    },
    ("1e-3", "16"): {
        2019: "B26M-LV-P36-1E3-SEED2019",
        2020: "B26M-LV-P36-1E3-SEED2020",
        2021: "B26M-L-P36-1E3",
    },
    ("1e-3", "32"): {
        2019: "B26M-LV2-P36-1E3-BS32-SEED2019",
        2020: "B26M-LV2-P36-1E3-BS32-SEED2020",
        2021: "B26M-LV2-P36-1E3-BS32-SEED2021",
    },
}

FINAL_CANDIDATE = {
    2019: "B26M-F-P36-SEED2019",
    2020: "B26M-F-P36-SEED2020",
    2021: "B26M-B-P36-BS32",
    2022: "B26M-F-P36-SEED2022",
    2023: "B26M-F-P36-SEED2023",
}


def load_aggregate(des: str, seed: int) -> dict:
    path = Path(f"analysis/{des}_seed{seed}/cv5_aggregate.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for (lr, batch), seed_map in LR_BATCH.items():
        for seed, des in seed_map.items():
            aggregate = load_aggregate(des, seed)
            rows.append(
                {
                    "lr": lr,
                    "batch": batch,
                    "seed": seed,
                    "experiment": des,
                    "full_MAE": aggregate["full_MAE_mean"],
                    "full_worst": aggregate["full_MAE_worst"],
                    "block24_MAE": aggregate["block24_MAE_mean"],
                    "block24_worst": aggregate["block24_MAE_worst"],
                }
            )
    detail = pd.DataFrame(rows)

    seed_rows = []
    for seed, des in FINAL_CANDIDATE.items():
        aggregate = load_aggregate(des, seed)
        seed_rows.append(
            {
                "seed": seed,
                "experiment": des,
                "full_MAE": aggregate["full_MAE_mean"],
                "block24_MAE": aggregate["block24_MAE_mean"],
                "full_worst": aggregate["full_MAE_worst"],
                "block24_worst": aggregate["block24_MAE_worst"],
            }
        )
    candidate = pd.DataFrame(seed_rows)

    out_dir = Path("analysis/p24_dual_summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "lr_batch_seed_detail.csv", index=False)
    candidate.to_csv(out_dir / "candidate_five_seed_detail.csv", index=False)
    aggregate_table = (
        detail.groupby(["lr", "batch"], sort=False)
        .agg(
            full_MAE_mean=("full_MAE", "mean"),
            full_MAE_seed_std=("full_MAE", "std"),
            block24_MAE_mean=("block24_MAE", "mean"),
            block24_MAE_seed_std=("block24_MAE", "std"),
        )
        .reset_index()
    )
    aggregate_table.to_csv(out_dir / "lr_batch_three_seed_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=300)
    batches = ["16", "32"]
    lrs = ["3e-4", "1e-3"]
    for ax, metric, label in (
        (axes[0], "full_MAE_mean", "Full-window MAE"),
        (axes[1], "block24_MAE_mean", "Block24 MAE"),
    ):
        matrix = np.asarray(
            [
                [
                    aggregate_table.loc[
                        (aggregate_table["lr"] == lr)
                        & (aggregate_table["batch"] == batch),
                        metric,
                    ].iloc[0]
                    for batch in batches
                ]
                for lr in lrs
            ]
        )
        image = ax.imshow(matrix, cmap="Blues_r")
        ax.set_xticks(range(len(batches)), labels=batches)
        ax.set_yticks(range(len(lrs)), labels=lrs)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{label} (three-seed mean)", loc="left", fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.suptitle("p24 P36 lr x batch interaction", fontsize=10)
    fig.tight_layout()
    out_png = out_dir / "p24_lr_batch_heatmap.png"
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
    x = np.arange(len(candidate))
    ax.plot(
        x,
        candidate["full_MAE"],
        marker="o",
        lw=1.4,
        color="#0072B2",
        label="Full MAE",
    )
    ax.plot(
        x,
        candidate["block24_MAE"],
        marker="s",
        lw=1.4,
        color="#E69F00",
        label="Block24 MAE",
    )
    ax.set_xticks(x, labels=[f"seed {seed}" for seed in candidate["seed"]])
    ax.set_ylabel("MAE (SSN)")
    ax.set_title("p24 final candidate: five-seed stability", loc="left", fontsize=10)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    seed_png = out_dir / "p24_candidate_five_seed.png"
    fig.savefig(seed_png, facecolor="white", bbox_inches="tight")
    fig.savefig(seed_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
