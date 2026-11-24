"""Evaluate R1B-NEXT W0 peak-amplitude candidates without a preferred model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "analysis/cycle_context/cycle_context_2026-10.csv"
OUT = ROOT / "analysis/r1b_next"
SEEDS = (2019, 2020, 2021, 2022, 2023)
CONTEXT_CYCLES = tuple(range(18, 24))
REPRESENTATIONS = ("log1p", "raw", "ratio")
MODELS = ("ridge", "random_forest")


def w0_features(target_kind: str) -> list[str]:
    return [
        f"current_peak_{target_kind}",
        f"previous_peak_{target_kind}",
        "decline_pct_24m",
        "cycle_parity",
        "previous_cycle_length",
    ]


def make_model(model_name: str, seed: int):
    if model_name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def encoded_target(
    frame: pd.DataFrame, target: str, denominator: str, representation: str
) -> np.ndarray:
    values = frame[target].to_numpy(dtype=float)
    if representation == "log1p":
        return np.log1p(values)
    if representation == "raw":
        return values
    if representation == "ratio":
        return values / frame[denominator].to_numpy(dtype=float)
    raise ValueError(representation)


def decode_prediction(
    values: np.ndarray, denominator: np.ndarray, representation: str
) -> np.ndarray:
    if representation == "log1p":
        return np.expm1(values)
    if representation == "raw":
        return values
    if representation == "ratio":
        return values * denominator
    raise ValueError(representation)


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    denominator: str,
    features: list[str],
    model_name: str,
    representation: str,
    seed: int,
) -> np.ndarray:
    if model_name == "median":
        return np.repeat(float(train[target].median()), len(test))
    model = make_model(model_name, seed)
    y_train = encoded_target(
        train, target, denominator, representation
    )
    model.fit(train[features], y_train)
    prediction = model.predict(test[features])
    return decode_prediction(
        prediction,
        test[denominator].to_numpy(dtype=float),
        representation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    context = pd.read_csv(CONTEXT)
    prediction_rows = []
    candidate_rows = [
        {
            "candidate_id": "A0",
            "model": "median",
            "representation": "physical",
        }
    ]
    index = 1
    for model_name in MODELS:
        for representation in REPRESENTATIONS:
            candidate_rows.append(
                {
                    "candidate_id": f"A{index}",
                    "model": model_name,
                    "representation": representation,
                }
            )
            index += 1
    candidates = pd.DataFrame(candidate_rows)

    for target_kind in ("raw", "smooth"):
        target = f"target_peak_{target_kind}"
        denominator = f"current_peak_{target_kind}"
        features = w0_features(target_kind)
        for cycle in CONTEXT_CYCLES:
            train = context.loc[context["cycle"] < cycle].dropna(
                subset=[target, *features]
            )
            test = context.loc[context["cycle"] == cycle].dropna(
                subset=[target, *features]
            )
            if test.empty:
                raise RuntimeError(
                    f"missing test row: target={target_kind}, context={cycle}"
                )
            observed = float(test[target].iloc[0])
            for candidate in candidates.itertuples(index=False):
                seeds = (
                    SEEDS
                    if candidate.model == "random_forest"
                    else (2021,)
                )
                for seed in seeds:
                    predicted = float(
                        fit_predict(
                            train,
                            test,
                            target,
                            denominator,
                            features,
                            candidate.model,
                            candidate.representation,
                            seed,
                        )[0]
                    )
                    prediction_rows.append(
                        {
                            "target_kind": target_kind,
                            "candidate_id": candidate.candidate_id,
                            "model": candidate.model,
                            "representation": candidate.representation,
                            "seed": seed,
                            "context_cycle": cycle,
                            "target_cycle": cycle + 1,
                            "true": observed,
                            "predicted": predicted,
                            "absolute_error": abs(predicted - observed),
                            "signed_error": predicted - observed,
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.out / "r1b_next_predictions.csv", index=False)

    seed_summary = (
        predictions.loc[predictions["candidate_id"] != "A0"]
        .groupby(
            [
                "target_kind",
                "candidate_id",
                "model",
                "representation",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            mae=("absolute_error", "mean"),
            rmse=("signed_error", lambda values: float(np.sqrt(np.mean(values**2)))),
            bias=("signed_error", "mean"),
        )
    )
    seed_summary.to_csv(args.out / "r1b_next_seed_summary.csv", index=False)

    median_errors = (
        predictions.loc[predictions["candidate_id"] == "A0"]
        .set_index(["target_kind", "target_cycle"])["absolute_error"]
    )
    summary_rows = []
    per_cycle_rows = []
    for target_kind in ("raw", "smooth"):
        target_predictions = predictions.loc[
            predictions["target_kind"] == target_kind
        ]
        for candidate_id, group in target_predictions.groupby("candidate_id"):
            metadata = group.iloc[0]
            seed_mae = (
                group.groupby("seed")["absolute_error"]
                .mean()
                .to_numpy(dtype=float)
            )
            reference = np.array(
                [
                    float(
                        median_errors.loc[(target_kind, target_cycle)]
                    )
                    for target_cycle in sorted(group["target_cycle"].unique())
                ]
            )
            improved_cycles = 0
            majority_improved = 0
            worst_degradation = -np.inf
            for target_cycle, cycle_group in group.groupby("target_cycle"):
                errors = cycle_group["absolute_error"].to_numpy(dtype=float)
                candidate_error = float(np.median(errors))
                baseline_error = float(
                    median_errors.loc[(target_kind, target_cycle)]
                )
                relative = (
                    (candidate_error - baseline_error) / baseline_error
                    if baseline_error != 0
                    else np.nan
                )
                worst_degradation = max(worst_degradation, relative)
                improved_cycles += bool(candidate_error < baseline_error)
                majority_improved += bool(
                    np.mean(errors < baseline_error) >= 0.6
                )
                per_cycle_rows.append(
                    {
                        "target_kind": target_kind,
                        "candidate_id": candidate_id,
                        "model": metadata["model"],
                        "representation": metadata["representation"],
                        "target_cycle": int(target_cycle),
                        "true": float(cycle_group["true"].iloc[0]),
                        "predicted_value": float(
                            cycle_group["predicted"].median()
                        ),
                        "median_absolute_error": candidate_error,
                        "climatology_absolute_error": baseline_error,
                        "relative_error_vs_climatology": relative,
                    }
                )
            candidate_mae = float(np.mean(seed_mae))
            baseline_mae = float(np.mean(reference))
            summary_rows.append(
                {
                    "target_kind": target_kind,
                    "candidate_id": candidate_id,
                    "model": metadata["model"],
                    "representation": metadata["representation"],
                    "n_target_cycles": 6,
                    "mae": candidate_mae,
                    "mae_seed_std": float(np.std(seed_mae)),
                    "mae_reference_climatology": baseline_mae,
                    "improvement_vs_climatology": (
                        (baseline_mae - candidate_mae) / baseline_mae
                    ),
                    "cycles_improved": improved_cycles,
                    "cycles_majority_seeds_improved": majority_improved,
                    "worst_cycle_relative_degradation": worst_degradation,
                    "bias": float(
                        group.groupby("seed")["signed_error"]
                        .mean()
                        .mean()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_kind", "mae"]
    )
    summary.to_csv(
        args.out / "target_representation_summary.csv", index=False
    )
    pd.DataFrame(per_cycle_rows).to_csv(
        args.out / "r1b_next_per_cycle.csv", index=False
    )
    decision = {
        "status": "exploratory_completed_pending_review",
        "review_notice": (
            "No candidate is designated as primary. Results are symmetric "
            "descriptive comparisons under the retrospective origin design."
        ),
        "target_cycles": [cycle + 1 for cycle in CONTEXT_CYCLES],
        "seeds": list(SEEDS),
        "candidates": candidates.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (args.out / "r1b_next_performance.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
