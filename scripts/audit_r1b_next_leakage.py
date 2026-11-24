"""Audit R1B-NEXT cycle-context features for forecast-origin leakage.

The script records evidence only. A failed check is reported as
``blocked_pending_review``; the final scientific judgment remains with 景修.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
CONTEXT = ROOT / "analysis/cycle_context/cycle_context_2026-10.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
OUT = ROOT / "analysis/r1b_next"
SEEDS = (2019, 2020, 2021, 2022, 2023)
TARGET_COLUMNS = (
    "target_peak_raw",
    "target_peak_smooth",
    "target_peak_date",
    "target_months_to_peak",
    "target_min_date",
    "target_months_to_min",
)
EXPANDING_CONTEXT_CYCLES = tuple(range(18, 24))
EXPECTED_BASELINES = {
    ("raw", "ridge"): 40.98101811199377,
    ("raw", "random_forest"): 44.70270853129371,
    ("smooth", "ridge"): 34.44934928522901,
    ("smooth", "random_forest"): 38.63317028537029,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(ROOT / "scripts/build_cycle_context.py", "cycle_builder")
EVALUATOR = load_module(
    ROOT / "scripts/evaluate_r1b_multicycle.py", "r1b_evaluator"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.to_period("M") - start.to_period("M")).n)


def parse_context(frame: pd.DataFrame) -> pd.DataFrame:
    date_columns = [
        "min_date",
        "peak_date",
        "origin_date",
        "target_min_date",
        "target_peak_date",
    ]
    return frame.copy().assign(
        **{column: pd.to_datetime(frame[column]) for column in date_columns}
    )


def build_context(frame: pd.DataFrame, workdir: Path) -> pd.DataFrame:
    workdir.mkdir(parents=True, exist_ok=True)
    data_path = workdir / "sunspot.csv"
    out_dir = workdir / "context"
    frame.to_csv(data_path, index=False)
    original_data = BUILDER.DATA
    original_out = BUILDER.OUT
    try:
        BUILDER.DATA = data_path
        BUILDER.OUT = out_dir
        with contextlib.redirect_stdout(io.StringIO()):
            BUILDER.main()
    finally:
        BUILDER.DATA = original_data
        BUILDER.OUT = original_out
    return parse_context(pd.read_csv(out_dir / "cycle_context_2026-10.csv"))


def feature_columns() -> list[str]:
    raw_w0, _, _ = EVALUATOR.target_features("raw")
    smooth_w0, _, _ = EVALUATOR.target_features("smooth")
    extra = [
        "origin_date",
        "peak_date",
        "smooth_at_origin",
        "decline_24m",
        "current_min_raw",
        "current_min_smooth",
    ]
    return list(dict.fromkeys([*raw_w0, *smooth_w0, *extra]))


def predict_origin(
    context: pd.DataFrame,
    cycle: int,
    target_kind: str,
    model_name: str,
    seed: int,
) -> float:
    target = f"target_peak_{target_kind}"
    w0, _, _ = EVALUATOR.target_features(target_kind)
    train = context.loc[context["cycle"] < cycle].dropna(
        subset=[target, *w0]
    )
    test = context.loc[context["cycle"] == cycle, w0]
    if train.empty or test.empty:
        raise RuntimeError(
            f"insufficient rows for cycle={cycle}, target={target_kind}"
        )
    return float(
        EVALUATOR.fit_predict(
            train, test, target, w0, model_name, seed
        )[0]
    )


def prediction_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    cycle: int,
    check: str,
    rows: list[dict],
) -> None:
    candidate_origin = candidate.loc[candidate["cycle"] == cycle, "origin_date"]
    reference_origin = reference.loc[reference["cycle"] == cycle, "origin_date"]
    same_origin = bool(
        not candidate_origin.empty
        and not reference_origin.empty
        and pd.Timestamp(candidate_origin.iloc[0])
        == pd.Timestamp(reference_origin.iloc[0])
    )
    for target_kind in ("raw", "smooth"):
        settings = [("median", 2021), ("ridge", 2021)]
        settings.extend(("random_forest", seed) for seed in SEEDS)
        for model_name, seed in settings:
            if not same_origin:
                rows.append(
                    {
                        "check": check,
                        "cycle": cycle,
                        "target_kind": target_kind,
                        "model": model_name,
                        "seed": seed,
                        "item": "prediction_same_origin",
                        "reference": np.nan,
                        "candidate": np.nan,
                        "abs_diff": np.nan,
                        "applicable": False,
                        "passed": True,
                        "detail": "origin shifted; prediction invariance is not interpretable",
                    }
                )
                continue
            try:
                reference_train = reference.loc[
                    reference["cycle"] < cycle
                ].dropna(
                    subset=[
                        f"target_peak_{target_kind}",
                        *EVALUATOR.target_features(target_kind)[0],
                    ]
                )
                if reference_train.empty:
                    rows.append(
                        {
                            "check": check,
                            "cycle": cycle,
                            "target_kind": target_kind,
                            "model": model_name,
                            "seed": seed,
                            "item": "prediction",
                            "reference": np.nan,
                            "candidate": np.nan,
                            "abs_diff": np.nan,
                            "applicable": False,
                            "passed": True,
                            "detail": "no training history at this early origin",
                        }
                    )
                    continue
                left = predict_origin(
                    reference, cycle, target_kind, model_name, seed
                )
                right = predict_origin(
                    candidate, cycle, target_kind, model_name, seed
                )
                difference = abs(left - right)
                rows.append(
                    {
                        "check": check,
                        "cycle": cycle,
                        "target_kind": target_kind,
                        "model": model_name,
                        "seed": seed,
                        "item": "prediction",
                        "reference": left,
                        "candidate": right,
                        "abs_diff": difference,
                        "applicable": True,
                        "passed": bool(difference <= 1e-10),
                        "detail": "forecast must be identical",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "check": check,
                        "cycle": cycle,
                        "target_kind": target_kind,
                        "model": model_name,
                        "seed": seed,
                        "item": "prediction",
                        "reference": np.nan,
                        "candidate": np.nan,
                        "abs_diff": np.nan,
                        "applicable": True,
                        "passed": False,
                        "detail": f"exception: {type(error).__name__}: {error}",
                    }
                )


def compare_feature_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    cycle: int,
    check: str,
    columns: list[str],
    rows: list[dict],
) -> None:
    left_row = reference.loc[reference["cycle"] == cycle]
    right_row = candidate.loc[candidate["cycle"] == cycle]
    if left_row.empty or right_row.empty:
        rows.append(
            {
                "check": check,
                "cycle": cycle,
                "target_kind": "all",
                "model": "not_applicable",
                "seed": -1,
                "item": "cycle_row",
                "reference": 1.0 if not left_row.empty else 0.0,
                "candidate": 1.0 if not right_row.empty else 0.0,
                "abs_diff": np.nan,
                "applicable": True,
                "passed": False,
                "detail": "cycle row missing before or after audit",
            }
        )
        return
    left = left_row.iloc[0]
    right = right_row.iloc[0]
    for column in columns:
        left_value = left[column]
        right_value = right[column]
        if pd.isna(left_value) and pd.isna(right_value):
            difference = 0.0
        elif pd.api.types.is_numeric_dtype(reference[column]):
            difference = abs(float(left_value) - float(right_value))
        else:
            difference = 0.0 if left_value == right_value else np.inf
        rows.append(
            {
                "check": check,
                "cycle": cycle,
                "target_kind": "all",
                "model": "not_applicable",
                "seed": -1,
                "item": column,
                "reference": left_value,
                "candidate": right_value,
                "abs_diff": difference,
                "applicable": True,
                "passed": bool(difference <= 1e-10),
                "detail": "feature must be identical after information cutoff",
            }
        )


def baseline_summary(context: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    result = {}
    detail_rows = []
    for target_kind in ("raw", "smooth"):
        target = f"target_peak_{target_kind}"
        w0, _, _ = EVALUATOR.target_features(target_kind)
        for model_name in ("ridge", "random_forest"):
            seeds = SEEDS if model_name == "random_forest" else (2021,)
            per_seed_mae = []
            for seed in seeds:
                errors = []
                for cycle in EXPANDING_CONTEXT_CYCLES:
                    train = context.loc[context["cycle"] < cycle].dropna(
                        subset=[target, *w0]
                    )
                    test = context.loc[context["cycle"] == cycle].dropna(
                        subset=[target, *w0]
                    )
                    predicted = EVALUATOR.fit_predict(
                        train, test, target, w0, model_name, seed
                    )[0]
                    observed = float(test[target].iloc[0])
                    errors.append(abs(predicted - observed))
                    detail_rows.append(
                        {
                            "target_kind": target_kind,
                            "model": model_name,
                            "seed": seed,
                            "context_cycle": cycle,
                            "target_cycle": cycle + 1,
                            "true": observed,
                            "predicted": float(predicted),
                            "absolute_error": abs(predicted - observed),
                        }
                    )
                per_seed_mae.append(float(np.mean(errors)))
            observed = float(np.mean(per_seed_mae))
            expected = EXPECTED_BASELINES[(target_kind, model_name)]
            result[f"{target_kind}_{model_name}"] = {
                "expected": expected,
                "observed": observed,
                "difference": observed - expected,
                "seed_mae": per_seed_mae,
                "reproduced": bool(abs(observed - expected) <= 1e-6),
            }
    return result, pd.DataFrame(detail_rows)


def feature_manifest(context: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    boundary_by_cycle = boundaries.set_index("cycle")
    rows = []
    for record in context.itertuples(index=False):
        origin = pd.Timestamp(record.origin_date)
        cycle = int(record.cycle)
        previous_peak_raw = (
            pd.Timestamp(boundary_by_cycle.loc[cycle - 1, "peak_date"])
            if cycle > 1
            else pd.NaT
        )
        previous_peak_smooth = previous_peak_raw
        available = [
            (
                "current_peak_raw",
                "ssn",
                "start_to_observed_peak",
                pd.Timestamp(record.peak_date),
                0,
            ),
            (
                "previous_peak_raw",
                "ssn",
                "previous_cycle",
                previous_peak_raw,
                0,
            ),
            (
                "current_peak_smooth",
                "ssn",
                "13m_centered_at_peak",
                pd.Timestamp(record.peak_date) - pd.DateOffset(months=6),
                6,
            ),
            (
                "previous_peak_smooth",
                "ssn",
                "13m_centered_at_previous_peak",
                previous_peak_smooth - pd.DateOffset(months=6)
                if pd.notna(previous_peak_smooth)
                else pd.NaT,
                6,
            ),
            (
                "decline_pct_24m",
                "ssn",
                "current_peak_smooth_and_causal_13m_at_origin",
                origin - pd.DateOffset(months=6),
                6,
            ),
            (
                "cycle_parity",
                "cycle_index",
                "cycle_identifier",
                pd.Timestamp(record.min_date),
                0,
            ),
            (
                "previous_cycle_length",
                "cycle_boundaries",
                "previous_and_current_min_date",
                pd.Timestamp(record.min_date),
                0,
            ),
        ]
        for feature, source, calculation, last_available, required_lag in available:
            if pd.isna(last_available):
                lag = np.nan
                applicable = False
                passed = True
            else:
                lag = months_between(last_available, origin)
                applicable = True
                passed = lag >= required_lag and last_available <= origin
            rows.append(
                {
                    "cycle": cycle,
                    "origin_date": origin,
                    "feature": feature,
                    "source_columns": source,
                    "calculation_window": calculation,
                    "last_available_month": last_available,
                    "lag_months": lag,
                    "required_lag_months": required_lag,
                    "applicable": bool(applicable),
                    "passed": bool(passed),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    reference = parse_context(pd.read_csv(CONTEXT))
    boundaries = pd.read_csv(
        BOUNDARIES,
        parse_dates=["min_date", "peak_date", "next_min_date"],
    )

    audit_rows: list[dict] = []
    features = feature_columns()
    with tempfile.TemporaryDirectory(prefix="r1b-next-audit-") as temp:
        temp_root = Path(temp)
        rebuilt = build_context(frame, temp_root / "full")
        cycle_dates = sorted(reference["origin_date"].dropna().unique())

        for origin_value in cycle_dates:
            origin = pd.Timestamp(origin_value)
            cycles = reference.loc[
                reference["origin_date"] == origin, "cycle"
            ].astype(int)
            if cycles.empty:
                continue
            cycle = int(cycles.iloc[0])

            truncated_frame = frame.loc[frame["date"] <= origin].copy()
            try:
                truncated = build_context(
                    truncated_frame, temp_root / f"truncated-{cycle:02d}"
                )
                compare_feature_rows(
                    reference, truncated, cycle, "truncation", features, audit_rows
                )
                prediction_rows(
                    reference, truncated, cycle, "truncation", audit_rows
                )
            except Exception as error:
                audit_rows.append(
                    {
                        "check": "truncation",
                        "cycle": cycle,
                        "target_kind": "all",
                        "model": "not_applicable",
                        "seed": -1,
                        "item": "pipeline",
                        "reference": np.nan,
                        "candidate": np.nan,
                        "abs_diff": np.nan,
                        "applicable": True,
                        "passed": False,
                        "detail": f"exception: {type(error).__name__}: {error}",
                    }
                )

            poisoned_frame = frame.copy()
            future = poisoned_frame["date"] > origin
            for column in poisoned_frame.select_dtypes(include="number").columns:
                poisoned_frame.loc[future, column] = 1.0e9
            try:
                poisoned = build_context(
                    poisoned_frame, temp_root / f"poisoned-{cycle:02d}"
                )
                compare_feature_rows(
                    reference, poisoned, cycle, "future_poisoning", features, audit_rows
                )
                prediction_rows(
                    reference, poisoned, cycle, "future_poisoning", audit_rows
                )
            except Exception as error:
                audit_rows.append(
                    {
                        "check": "future_poisoning",
                        "cycle": cycle,
                        "target_kind": "all",
                        "model": "not_applicable",
                        "seed": -1,
                        "item": "pipeline",
                        "reference": np.nan,
                        "candidate": np.nan,
                        "abs_diff": np.nan,
                        "applicable": True,
                        "passed": False,
                        "detail": f"exception: {type(error).__name__}: {error}",
                    }
                )

            blinded = reference.copy()
            blinded.loc[blinded["cycle"] >= cycle, TARGET_COLUMNS] = np.nan
            prediction_rows(
                reference, blinded, cycle, "target_blindness", audit_rows
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out / "leakage_audit.csv", index=False)

    manifest = feature_manifest(reference, boundaries)
    manifest.to_csv(args.out / "feature_manifest.csv", index=False)

    baselines, baseline_detail = baseline_summary(reference)
    baseline_detail.to_csv(args.out / "baseline_predictions.csv", index=False)

    check_summary = (
        audit.loc[audit["applicable"]]
        .groupby("check", as_index=False)
        .agg(
            checks=("passed", "size"),
            passes=("passed", "sum"),
            failures=("passed", lambda values: int((~values).sum())),
        )
    )
    check_summary["passed"] = check_summary["failures"] == 0
    manifest_summary = {
        "checks": int(manifest["applicable"].sum()),
        "passes": int(
            (manifest["applicable"] & manifest["passed"]).sum()
        ),
        "failures": int(
            (manifest["applicable"] & ~manifest["passed"]).sum()
        ),
        "not_applicable": int((~manifest["applicable"]).sum()),
    }
    baseline_passed = all(item["reproduced"] for item in baselines.values())
    audit_passed = bool(check_summary["passed"].all())
    manifest_passed = manifest_summary["failures"] == 0
    overall = audit_passed and manifest_passed and baseline_passed
    summary = {
        "status": "passed_pending_review" if overall else "blocked_pending_review",
        "review_notice": (
            "Automated checks identify pass/fail candidates. 景修 must review "
            "the evidence before a failure is treated as final."
        ),
        "baseline_reproduction": baselines,
        "baseline_passed": baseline_passed,
        "audit_summary": check_summary.to_dict(orient="records"),
        "audit_passed": audit_passed,
        "manifest_summary": manifest_summary,
        "manifest_passed": manifest_passed,
        "overall_passed": overall,
        "inputs": {
            "data": str(DATA.relative_to(ROOT)),
            "data_sha256": sha256(DATA),
            "context": str(CONTEXT.relative_to(ROOT)),
            "context_sha256": sha256(CONTEXT),
            "builder_sha256": sha256(ROOT / "scripts/build_cycle_context.py"),
            "evaluator_sha256": sha256(
                ROOT / "scripts/evaluate_r1b_multicycle.py"
            ),
            "audit_sha256": sha256(Path(__file__)),
            "seeds": list(SEEDS),
            "expanding_context_cycles": list(EXPANDING_CONTEXT_CYCLES),
            "target_cycles": [cycle + 1 for cycle in EXPANDING_CONTEXT_CYCLES],
        },
    }
    (args.out / "leakage_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(check_summary.to_string(index=False))
    print("\nBaseline")
    print(json.dumps(baselines, indent=2, ensure_ascii=False))
    print("\nManifest", manifest_summary)
    print("Status:", summary["status"])


if __name__ == "__main__":
    main()
