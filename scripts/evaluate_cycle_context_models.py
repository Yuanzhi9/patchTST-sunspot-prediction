"""Exploratory leave-one-cycle-out checks for cycle-level extrema prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/cycle_context/cycle_context_2026-10.csv"
OUT = ROOT / "analysis/cycle_context/model_screening"

FEATURES_REDUCED = [
    "current_peak_smooth",
    "decline_pct_24m",
    "postpeak_volatility_24m",
    "rise_months",
    "previous_peak_smooth",
    "cycle_parity",
]
FEATURES_FULL = [
    "elapsed_since_min",
    "rise_months",
    "months_since_peak",
    "current_peak_smooth",
    "current_peak_raw",
    "current_min_smooth",
    "smooth_at_origin",
    "decline_24m",
    "decline_pct_24m",
    "postpeak_volatility_24m",
    "postpeak_abs_slope_24m",
    "previous_peak_smooth",
    "previous_peak_raw",
    "previous_cycle_length",
    "cycle_parity",
]


def model_factory(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            random_state=20261123,
            n_jobs=1,
        )
    raise ValueError(name)


def evaluate(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    data = frame.dropna(subset=[target, *features]).reset_index(drop=True)
    predictions = np.full(len(data), np.nan)
    for test_index in range(len(data)):
        train = data.drop(index=test_index)
        test = data.iloc[[test_index]]
        model = model_factory(model_name)
        model.fit(train[features], train[target])
        predictions[test_index] = float(model.predict(test[features])[0])
    return data, predictions


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "R2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def add_rows(
    rows: list[dict],
    frame: pd.DataFrame,
    target: str,
    feature_set: str,
    features: list[str],
    model_name: str,
) -> None:
    data, predicted = evaluate(frame, target, features, model_name)
    actual = data[target].to_numpy(float)
    result = metrics(actual, predicted)
    rows.append(
        {
            "target": target,
            "feature_set": feature_set,
            "model": model_name,
            **result,
        }
    )
    predictions = data[
        ["cycle", "origin_date", "target_min_date", "target_peak_date", target]
    ].copy()
    predictions["prediction"] = predicted
    predictions.to_csv(
        OUT / f"loocv_{target}_{feature_set}_{model_name}.csv",
        index=False,
    )


def main() -> None:
    frame = pd.read_csv(DATA, parse_dates=["origin_date", "target_min_date", "target_peak_date"])
    complete = frame.loc[frame["target_peak_smooth"].notna()].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for target in (
        "target_peak_smooth",
        "target_peak_raw",
        "target_months_to_min",
        "target_months_to_peak",
    ):
        data = complete.dropna(subset=[target]).reset_index(drop=True)
        median = float(data[target].median())
        rows.append(
            {
                "target": target,
                "feature_set": "climatology_median",
                "model": "constant",
                "MAE": float(np.mean(np.abs(data[target] - median))),
                "RMSE": float(np.sqrt(np.mean((data[target] - median) ** 2))),
                "R2": float(r2_score(data[target], np.repeat(median, len(data)))),
                "bias": 0.0,
            }
        )
        for feature_set, features in (
            ("reduced", FEATURES_REDUCED),
            ("full", FEATURES_FULL),
        ):
            for model_name in ("ridge", "random_forest"):
                add_rows(rows, complete, target, feature_set, features, model_name)

    summary = pd.DataFrame(rows).sort_values(["target", "MAE", "model"])
    summary.to_csv(OUT / "cycle_model_screening_summary.csv", index=False)

    current = frame.loc[frame["cycle"] == 25].iloc[0]
    current_rows = []
    for target in (
        "target_peak_smooth",
        "target_peak_raw",
        "target_months_to_min",
        "target_months_to_peak",
    ):
        for feature_set, features in (
            ("reduced", FEATURES_REDUCED),
            ("full", FEATURES_FULL),
        ):
            for model_name in ("ridge", "random_forest"):
                train = complete.dropna(subset=[target, *features])
                model = model_factory(model_name)
                model.fit(train[features], train[target])
                prediction = float(model.predict(current[features].to_frame().T)[0])
                current_rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "model": model_name,
                        "prediction": prediction,
                    }
                )
    current_predictions = pd.DataFrame(current_rows)
    current_predictions.to_csv(
        OUT / "current_cycle25_predictions.csv", index=False
    )
    print(summary.to_string(index=False))
    print("\nCurrent Cycle 25 context predictions")
    print(current_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
