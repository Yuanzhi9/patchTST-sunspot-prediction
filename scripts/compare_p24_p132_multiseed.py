"""Compare five-seed p24 and p132 baselines on common fold/date intersections."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_p24_p132 import extrema, find_dirs, fold_number, metrics


P24 = {
    2019: "B26M-F-P36-SEED2019",
    2020: "B26M-F-P36-SEED2020",
    2021: "B26M-B-P36-BS32",
    2022: "B26M-F-P36-SEED2022",
    2023: "B26M-F-P36-SEED2023",
}

P132 = {
    2019: "B26M-P132-S132-P36-LR1E3-DO03-SEED2019",
    2020: "B26M-P132-S132-P36-LR1E3-DO03-SEED2020",
    2021: "B26M-P132-S132-P36-LR1E3-DO03",
    2022: "B26M-P132-S132-P36-LR1E3-DO03-SEED2022",
    2023: "B26M-P132-S132-P36-LR1E3-DO03-SEED2023",
}


def load_fold_curves(
    p24_results_root: str, p132_results_root: str, seed: int, fold: int
) -> pd.DataFrame:
    p24_root = Path(p24_results_root)
    p132_root = Path(p132_results_root)
    p24_dir = next(
        path
        for path in find_dirs(
            str(p24_root), P24[seed], seed, "cv_block_curve.csv"
        )
        if fold_number(path) == fold
    )
    p132_dir = next(
        path
        for path in find_dirs(
            str(p132_root), P132[seed], seed, "p132_curve.csv"
        )
        if fold_number(path) == fold
    )
    p24_curve = pd.read_csv(p24_dir / "cv_block_curve.csv", parse_dates=["date"])
    p132_curve = pd.read_csv(p132_dir / "p132_curve.csv", parse_dates=["date"])
    return (
        p24_curve[["date", "true", "pred"]]
        .rename(columns={"pred": "p24"})
        .merge(
            p132_curve[["date", "pred"]].rename(columns={"pred": "p132"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def main() -> None:
    p24_results_root = "/output/BASE-2026-11-M/results"
    p132_results_root = "results"
    out_dir = Path("analysis_p132/p24_vs_p132_five_seed")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    fold_curve_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for seed in sorted(P24):
        fold_rows = []
        for fold in range(1, 6):
            merged = load_fold_curves(p24_results_root, p132_results_root, seed, fold)
            fold_curve_cache[(seed, fold)] = merged
            true = merged["true"].to_numpy()
            dates = pd.to_datetime(merged["date"]).to_list()
            p24_metrics = metrics(merged["p24"].to_numpy(), true)
            p132_metrics = metrics(merged["p132"].to_numpy(), true)
            p24_ext = extrema(merged["p24"].to_numpy(), true, dates)
            p132_ext = extrema(merged["p132"].to_numpy(), true, dates)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_common_months": len(merged),
                    "p24_MAE": p24_metrics["mae"],
                    "p24_RMSE": p24_metrics["rmse"],
                    "p132_MAE": p132_metrics["mae"],
                    "p132_RMSE": p132_metrics["rmse"],
                    "p24_peak_abs_amp": p24_ext["peak"]["abs_amplitude_error_mean"],
                    "p132_peak_abs_amp": p132_ext["peak"]["abs_amplitude_error_mean"],
                    "p24_trough_abs_amp": p24_ext["trough"]["abs_amplitude_error_mean"],
                    "p132_trough_abs_amp": p132_ext["trough"]["abs_amplitude_error_mean"],
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        seed_rows.append(
            {
                "seed": seed,
                **{
                    column: float(fold_frame[column].mean())
                    for column in fold_frame.columns
                    if column not in {"seed", "fold"}
                },
            }
        )

    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(out_dir / "per_seed_summary.csv", index=False)
    aggregate = {}
    for column in seed_frame.columns:
        if column == "seed":
            continue
        aggregate[column] = {
            "mean": float(seed_frame[column].mean()),
            "std": float(seed_frame[column].std(ddof=1)),
            "min": float(seed_frame[column].min()),
            "max": float(seed_frame[column].max()),
        }
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=300)
    x = seed_frame["seed"].to_numpy()
    ax.plot(x, seed_frame["p24_MAE"], marker="o", color="#E69F00", label="p24 block24")
    ax.plot(x, seed_frame["p132_MAE"], marker="s", color="#0072B2", label="p132 direct")
    ax.set_xticks(x)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Common-date MAE (SSN)")
    ax.set_title("p24 vs p132: five-seed common-date curve error", loc="left", fontsize=10)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png = out_dir / "p24_vs_p132_five_seed.png"
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
