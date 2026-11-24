"""Full-history leave-one-cycle-out curves for R1B-NEXT W0 candidates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "analysis/cycle_context/cycle_context_2026-10.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
OUT = ROOT / "analysis/r1b_next/full_cycles"
ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
LABELS = {
    "A0": "Climatology median",
    "A1": "Ridge · log1p",
    "A2": "Ridge · raw",
    "A3": "Ridge · ratio",
    "A4": "RF · log1p",
    "A5": "RF · raw",
    "A6": "RF · ratio",
}
STYLES = {
    "A0": ("#222222", "--", 2.2),
    "A1": ("#0072B2", "-", 1.8),
    "A2": ("#56B4E9", "-", 1.8),
    "A3": ("#009E73", "-", 1.8),
    "A4": ("#D55E00", "-", 1.8),
    "A5": ("#C63D2F", "-", 1.8),
    "A6": ("#7A5195", "-", 1.8),
}


def load_evaluator():
    path = ROOT / "scripts/evaluate_r1b_next.py"
    spec = importlib.util.spec_from_file_location("r1b_next_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_curves(
    predictions: pd.DataFrame, boundaries: pd.DataFrame, output: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 9.4), sharex=True)
    kind_labels = {
        "raw": "Raw monthly peak",
        "smooth": "13-month centered smoothed peak",
    }
    for axis, target_kind in zip(axes, ("raw", "smooth")):
        truth = boundaries.set_index("cycle").loc[
            lambda frame: frame.index <= 24, f"peak_{target_kind}"
        ]
        axis.plot(
            truth.index,
            truth.to_numpy(),
            color="black",
            marker="o",
            linewidth=2.7,
            markersize=5.5,
            label="Observed peak",
            zorder=20,
        )
        subset = predictions.loc[predictions["target_kind"] == target_kind]
        for candidate_id in ORDER:
            group = subset.loc[subset["candidate_id"] == candidate_id]
            if group.empty:
                continue
            values = group.groupby("target_cycle")["predicted"].median()
            color, linestyle, linewidth = STYLES[candidate_id]
            axis.plot(
                values.index,
                values.to_numpy(),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker="o",
                markersize=2.8,
                label=LABELS[candidate_id],
                alpha=0.92,
            )
        axis.axvspan(0.5, 2.5, color="#EEEEEE", alpha=0.65, zorder=0)
        axis.text(
            1.5,
            axis.get_ylim()[1] * 0.96,
            "SC1–2: no previous-cycle\nfeature, no prediction",
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
        axis.set_title(kind_labels[target_kind])
        axis.set_ylabel("Cycle peak amplitude (SSN)")
        axis.set_xticks(range(1, 25))
        axis.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper right")

    axes[1].set_xlabel("Solar cycle")
    figure.suptitle(
        "R1B-NEXT full-history peak curves — leave-one-cycle-out, retrospective",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.01,
        (
            "All A0–A6 candidates shown symmetrically. RF lines are five-seed "
            "prediction medians. LOOCV trains on later cycles as well, so this "
            "is not a future-blind forecast."
        ),
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.95))
    figure.savefig(output, dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    context = pd.read_csv(CONTEXT)
    boundaries = pd.read_csv(BOUNDARIES)
    rows = []
    for target_kind in ("raw", "smooth"):
        target = f"target_peak_{target_kind}"
        denominator = f"current_peak_{target_kind}"
        features = evaluator.w0_features(target_kind)
        eligible = context.dropna(subset=[target, *features]).copy()
        eligible = eligible.loc[eligible["cycle"].between(2, 23)].copy()
        for target_cycle in sorted(eligible["cycle"].astype(int)):
            test = eligible.loc[eligible["cycle"] == target_cycle]
            train = eligible.loc[eligible["cycle"] != target_cycle]
            observed = float(test[target].iloc[0])
            for candidate in (
                ("A0", "median", "physical"),
                ("A1", "ridge", "log1p"),
                ("A2", "ridge", "raw"),
                ("A3", "ridge", "ratio"),
                ("A4", "random_forest", "log1p"),
                ("A5", "random_forest", "raw"),
                ("A6", "random_forest", "ratio"),
            ):
                candidate_id, model_name, representation = candidate
                seeds = (
                    evaluator.SEEDS
                    if model_name == "random_forest"
                    else (2021,)
                )
                for seed in seeds:
                    predicted = float(
                        evaluator.fit_predict(
                            train,
                            test,
                            target,
                            denominator,
                            features,
                            model_name,
                            representation,
                            seed,
                        )[0]
                    )
                    rows.append(
                        {
                            "target_kind": target_kind,
                            "candidate_id": candidate_id,
                            "model": model_name,
                            "representation": representation,
                            "seed": seed,
                            "context_cycle": target_cycle,
                            "target_cycle": target_cycle + 1,
                            "true": observed,
                            "predicted": predicted,
                            "signed_error": predicted - observed,
                            "absolute_error": abs(predicted - observed),
                        }
                    )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUT / "full_cycle_predictions.csv", index=False)
    summary = (
        predictions.groupby(
            ["target_kind", "candidate_id", "model", "representation", "seed"],
            as_index=False,
        )
        .agg(
            mae=("absolute_error", "mean"),
            rmse=("signed_error", lambda values: float(np.sqrt(np.mean(values**2)))),
            bias=("signed_error", "mean"),
        )
    )
    summary.to_csv(OUT / "full_cycle_seed_summary.csv", index=False)
    aggregate = (
        summary.groupby(
            ["target_kind", "candidate_id", "model", "representation"],
            as_index=False,
        )
        .agg(
            mae=("mae", "mean"),
            mae_seed_std=("mae", "std"),
            bias=("bias", "mean"),
        )
        .sort_values(["target_kind", "mae"])
    )
    aggregate.to_csv(OUT / "full_cycle_summary.csv", index=False)
    plot_curves(predictions, boundaries, OUT / "r1b_next_full_cycles.png")
    metadata = {
        "status": "descriptive_loocv_completed",
        "target_cycles_with_predictions": sorted(
            predictions["target_cycle"].unique().tolist()
        ),
        "note": (
            "LOOCV is retrospective and uses later cycles in training. "
            "No candidate is designated as primary."
        ),
        "summary": aggregate.to_dict(orient="records"),
    }
    (OUT / "full_cycle_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
