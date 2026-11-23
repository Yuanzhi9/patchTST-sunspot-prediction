"""Evaluate raw/13m peak-amplitude models over historical solar cycles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", message="R\\^2 score is not well-defined")

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "analysis/cycle_context/cycle_context_2026-10.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
AA = ROOT / "PatchTST_supervised/dataset/external/processed/aa_monthly_1869_2026.csv"
OUT = ROOT / "analysis/r1b_multicycle"
SEEDS = (2019, 2020, 2021, 2022, 2023)


def add_aa_features(context: pd.DataFrame, boundaries: pd.DataFrame, aa: pd.DataFrame) -> pd.DataFrame:
    aa = aa.set_index("date").sort_index()
    boundary_by_cycle = boundaries.set_index("cycle")
    rows = []
    for _, row in context.iterrows():
        cycle = int(row["cycle"])
        current_min = pd.Timestamp(boundary_by_cycle.loc[cycle, "min_date"])
        previous_min = (
            pd.Timestamp(boundary_by_cycle.loc[cycle - 1, "min_date"])
            if cycle > 1
            else pd.NaT
        )
        origin = pd.Timestamp(row["origin_date"])
        prev_slice = (
            aa.loc[(aa.index >= previous_min) & (aa.index < current_min), "aa_mean"]
            if pd.notna(previous_min)
            else pd.Series(dtype=float)
        )
        current_slice = aa.loc[
            (aa.index >= current_min) & (aa.index <= origin), "aa_mean"
        ]
        origin_value = aa.loc[aa.index == origin, "aa_mean"]
        rows.append(
            {
                "current_cycle": cycle,
                "aa_prev_cycle_max": float(prev_slice.max()) if prev_slice.notna().any() else np.nan,
                "aa_since_min_mean": float(current_slice.mean()) if current_slice.notna().any() else np.nan,
                "aa_since_min_max": float(current_slice.max()) if current_slice.notna().any() else np.nan,
                "aa_at_origin": float(origin_value.iloc[0]) if len(origin_value) else np.nan,
            }
        )
    return context.merge(
        pd.DataFrame(rows).rename(columns={"current_cycle": "cycle"}),
        on="cycle",
        how="left",
    )


def target_features(target_kind: str) -> tuple[list[str], list[str], list[str]]:
    peak = f"current_peak_{target_kind}"
    previous = f"previous_peak_{target_kind}"
    w0 = [peak, previous, "decline_pct_24m", "cycle_parity", "previous_cycle_length"]
    w1 = w0 + [
        "rise_months",
        "postpeak_abs_slope_24m",
        "postpeak_volatility_24m",
        "elapsed_since_min",
    ]
    aa_features = [
        "aa_prev_cycle_max",
        "aa_since_min_mean",
        "aa_since_min_max",
        "aa_at_origin",
    ]
    return w0, w1, aa_features


def make_model(kind: str, seed: int):
    if kind == "median":
        return None
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    if kind == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            ),
        )
    raise ValueError(kind)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], model_name: str, seed: int) -> np.ndarray:
    train = train.dropna(subset=[target, *features])
    if train.empty:
        raise RuntimeError("no training rows after feature filtering")
    y_log = np.log1p(train[target].to_numpy(dtype=float))
    if model_name == "median":
        return np.repeat(np.expm1(np.median(y_log)), len(test))
    model = make_model(model_name, seed)
    model.fit(train[features], y_log)
    return np.expm1(model.predict(test[features]))


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    r2 = float(r2_score(true, pred)) if len(true) >= 2 else float("nan")
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": r2,
        "bias": float(np.mean(pred - true)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(
        CONTEXT,
        parse_dates=["min_date", "peak_date", "origin_date", "target_min_date", "target_peak_date"],
    )
    boundaries = pd.read_csv(
        BOUNDARIES,
        parse_dates=["min_date", "peak_date", "next_min_date"],
    )
    aa = pd.read_csv(AA, parse_dates=["date"])
    context = add_aa_features(context, boundaries, aa)
    complete = context.dropna(subset=["target_peak_raw", "target_peak_smooth"]).copy()

    rows = []
    prediction_rows = []
    for target_kind in ("raw", "smooth"):
        target = f"target_peak_{target_kind}"
        w0, w1, aa_features = target_features(target_kind)
        feature_sets = {
            "W0": w0,
            "W0_AA_CONTROL": w0,
            "W1": w1,
            "W1_AA_CONTROL": w1,
            "W2_AA": w1 + aa_features,
        }
        for feature_name, features in feature_sets.items():
            data = complete.dropna(subset=[target, *features]).reset_index(drop=True)
            if feature_name in {"W0_AA_CONTROL", "W1_AA_CONTROL", "W2_AA"}:
                data = data.dropna(subset=aa_features).reset_index(drop=True)
            for model_name in ("median", "ridge", "random_forest"):
                seeds = SEEDS if model_name == "random_forest" else (2021,)
                for seed in seeds:
                    predictions = np.full(len(data), np.nan)
                    for test_index in range(len(data)):
                        train = data.drop(index=test_index)
                        test = data.iloc[[test_index]]
                        predictions[test_index] = fit_predict(
                            train, test, target, features, model_name, seed
                        )[0]
                    rows.append(
                        {
                            "target_kind": target_kind,
                            "feature_set": feature_name,
                            "model": model_name,
                            "seed": seed,
                            "cv": "loocv",
                            "n_test": len(data),
                            **metrics(data[target].to_numpy(), predictions),
                        }
                    )
                    for cycle, true, pred in zip(
                        data["cycle"].astype(int),
                        data[target].to_numpy(),
                        predictions,
                    ):
                        prediction_rows.append(
                            {
                                "target_kind": target_kind,
                                "feature_set": feature_name,
                                "model": model_name,
                                "seed": seed,
                                "cv": "loocv",
                                "test_cycle": cycle,
                                "true": float(true),
                                "pred": float(pred),
                            }
                        )

                    for test_cycle in range(18, 24):
                        train = data.loc[data["cycle"] < test_cycle]
                        test = data.loc[data["cycle"] == test_cycle]
                        if test.empty or len(train) < 8:
                            continue
                        pred = fit_predict(
                            train, test, target, features, model_name, seed
                        )
                        rows.append(
                            {
                                "target_kind": target_kind,
                                "feature_set": feature_name,
                                "model": model_name,
                                "seed": seed,
                                "cv": "expanding",
                                "test_cycle": int(test_cycle),
                                "n_test": len(test),
                                **metrics(test[target].to_numpy(), pred),
                            }
                        )
                        prediction_rows.append(
                            {
                                "target_kind": target_kind,
                                "feature_set": feature_name,
                                "model": model_name,
                                "seed": seed,
                                "cv": "expanding",
                                "test_cycle": int(test_cycle),
                                "true": float(test[target].iloc[0]),
                                "pred": float(pred[0]),
                            }
                        )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "validation_detail.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "validation_predictions.csv", index=False)
    summary = (
        detail.groupby(["target_kind", "feature_set", "model", "cv"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            bias_mean=("bias", "mean"),
            runs=("mae", "size"),
        )
        .sort_values(["target_kind", "cv", "mae_mean"])
    )
    summary.to_csv(OUT / "validation_summary.csv", index=False)

    current = context.loc[context["cycle"] == 25].iloc[[0]]
    current_rows = []
    for target_kind in ("raw", "smooth"):
        target = f"target_peak_{target_kind}"
        w0, w1, aa_features = target_features(target_kind)
        for feature_name, features in {
            "W0": w0,
            "W0_AA_CONTROL": w0,
            "W1": w1,
            "W1_AA_CONTROL": w1,
            "W2_AA": w1 + aa_features,
        }.items():
            train = complete.dropna(subset=[target, *features])
            if feature_name in {"W0_AA_CONTROL", "W1_AA_CONTROL", "W2_AA"}:
                train = train.dropna(subset=aa_features)
            for model_name in ("median", "ridge", "random_forest"):
                seeds = SEEDS if model_name == "random_forest" else (2021,)
                values = []
                for seed in seeds:
                    values.append(
                        float(
                            fit_predict(train, current, target, features, model_name, seed)[0]
                        )
                    )
                current_rows.append(
                    {
                        "target_kind": target_kind,
                        "feature_set": feature_name,
                        "model": model_name,
                        "prediction_median": float(np.median(values)),
                        "prediction_min": float(np.min(values)),
                        "prediction_max": float(np.max(values)),
                        "n_train": len(train),
                        "train_target_min": float(train[target].min()),
                        "train_target_max": float(train[target].max()),
                        "extrapolation": bool(
                            np.median(values) < train[target].min()
                            or np.median(values) > train[target].max()
                        ),
                    }
                )
    current_predictions = pd.DataFrame(current_rows)
    current_predictions.to_csv(OUT / "current_predictions.csv", index=False)

    loocv_best = {}
    aa_increment = {}
    w_increment = {}
    for target_kind in ("raw", "smooth"):
        subset = summary.loc[
            (summary["target_kind"] == target_kind)
            & (summary["cv"] == "loocv")
        ]
        best = subset.sort_values("mae_mean").iloc[0]
        loocv_best[target_kind] = {
            "feature_set": best["feature_set"],
            "model": best["model"],
            "mae": float(best["mae_mean"]),
        }
        aa_increment[target_kind] = {}
        w_increment[target_kind] = {}
        for model in ("ridge", "random_forest"):
            w0_control = subset.loc[
                (subset["feature_set"] == "W0_AA_CONTROL")
                & (subset["model"] == model),
                "mae_mean",
            ]
            control = subset.loc[
                (subset["feature_set"] == "W1_AA_CONTROL")
                & (subset["model"] == model),
                "mae_mean",
            ]
            treatment = subset.loc[
                (subset["feature_set"] == "W2_AA")
                & (subset["model"] == model),
                "mae_mean",
            ]
            if len(control) and len(treatment):
                aa_increment[target_kind][model] = float(
                    treatment.iloc[0] - control.iloc[0]
                )
            if len(w0_control) and len(control):
                w_increment[target_kind][model] = float(
                    control.iloc[0] - w0_control.iloc[0]
                )
    decision = {
        "targets": {
            "raw": "raw monthly maximum",
            "smooth": "13-month centered smoothed maximum",
        },
        "note": (
            "Both definitions remain active; neither replaces the other. "
            "Current Cycle 26 estimates require review and extrapolation checks."
        ),
        "summary_rows": summary.to_dict(orient="records"),
        "current_predictions": current_predictions.to_dict(orient="records"),
        "loocv_best": loocv_best,
        "w_increment_mae": w_increment,
        "aa_increment_mae": aa_increment,
    }
    (OUT / "decision.json").write_text(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else value,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nCurrent predictions")
    print(current_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
