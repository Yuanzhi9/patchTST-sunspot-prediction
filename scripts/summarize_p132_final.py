"""Summarize the five-seed p132 final candidate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FINAL = {
    2019: "B26M-P132-S132-P36-LR1E3-DO03-SEED2019",
    2020: "B26M-P132-S132-P36-LR1E3-DO03-SEED2020",
    2021: "B26M-P132-S132-P36-LR1E3-DO03",
    2022: "B26M-P132-S132-P36-LR1E3-DO03-SEED2022",
    2023: "B26M-P132-S132-P36-LR1E3-DO03-SEED2023",
}


def main() -> None:
    rows = []
    for seed, des in FINAL.items():
        path = Path(f"analysis_p132/{des}_seed{seed}/cv5_aggregate.json")
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "experiment": des,
                "cycle_MAE": payload["cycle_MAE"]["mean"],
                "cycle_MAE_worst": payload["cycle_MAE"]["worst"],
                "full_MAE": payload["full_MAE"]["mean"],
                "peak_abs_amp": payload["peak_abs_amp"]["mean"],
                "trough_abs_amp": payload["trough_abs_amp"]["mean"],
                "peak_abs_time": payload["peak_abs_time"]["mean"],
                "trough_abs_time": payload["trough_abs_time"]["mean"],
            }
        )
    frame = pd.DataFrame(rows)
    out_dir = Path("analysis_p132/final_summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "candidate_five_seed_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=300)
    x = frame["seed"].to_numpy()
    ax.plot(x, frame["cycle_MAE"], marker="o", lw=1.4, color="#0072B2", label="Direct-cycle MAE")
    ax.plot(x, frame["full_MAE"], marker="s", lw=1.4, color="#E69F00", label="All-step MAE")
    ax.plot(
        x,
        frame["peak_abs_amp"],
        marker="^",
        lw=1.2,
        color="#009E73",
        label="Peak |amplitude error|",
    )
    ax.set_xlabel("Seed")
    ax.set_ylabel("SSN error")
    ax.set_title("p132 final candidate: five-seed stability", loc="left", fontsize=10)
    ax.set_xticks(x)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png = out_dir / "p132_candidate_five_seed.png"
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
