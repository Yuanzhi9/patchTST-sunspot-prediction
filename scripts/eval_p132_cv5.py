"""Evaluate p132 direct-cycle CV5 outputs.

Diagnostic metrics use all sliding windows. The primary p132 metric uses
non-overlapping 132-month calls within each outer fold, so overlapping
windows are not counted as independent cycles.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def physical_target(z: np.ndarray, meta: dict) -> np.ndarray:
    mean = float(meta["scaler_mean"][-1])
    scale = float(meta["scaler_scale"][-1])
    transform = meta.get("target_transform", "")
    if transform:
        raise ValueError(f"evaluation inverse for target_transform={transform!r} not implemented")
    return z * scale + mean


def metric_block(pred: np.ndarray, true: np.ndarray) -> dict:
    diff = pred - true
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def build_direct_cycle_curve(pred: np.ndarray, true: np.ndarray, meta: dict) -> pd.DataFrame:
    starts = np.asarray(meta["window_target_starts"], dtype=int)
    dates = pd.to_datetime([d["start"] for d in meta["target_dates"]])
    outer = meta["fold_spec"]["outer_val"]
    pred_len = int(meta["pred_len"])
    current = int(starts[0])
    rows = []

    while current <= int(outer["end"]):
        matches = np.flatnonzero(starts == current)
        if len(matches) == 0:
            break
        i = int(matches[0])
        for lead in range(pred_len):
            rows.append(
                {
                    "date": dates[i] + pd.DateOffset(months=lead),
                    "true": float(true[i, lead]),
                    "pred": float(pred[i, lead]),
                    "window_index": i,
                    "lead": lead + 1,
                }
            )
        current += pred_len
    return pd.DataFrame(rows)


def smooth(values: pd.Series, window: int = 13) -> pd.Series:
    return values.rolling(window, center=True, min_periods=1).mean()


def extrema_errors(curve: pd.DataFrame) -> dict:
    smooth_true = smooth(curve["true"]).to_numpy()
    smooth_pred = smooth(curve["pred"]).to_numpy()
    distance = max(24, len(curve) // 8)

    true_peaks, _ = find_peaks(smooth_true, distance=distance, prominence=2.0)
    pred_peaks, _ = find_peaks(smooth_pred, distance=distance, prominence=2.0)
    true_troughs, _ = find_peaks(-smooth_true, distance=distance, prominence=2.0)
    pred_troughs, _ = find_peaks(-smooth_pred, distance=distance, prominence=2.0)
    dates = pd.to_datetime(curve["date"]).to_list()

    def match(true_idx: np.ndarray, pred_idx: np.ndarray, kind: str) -> list[dict]:
        records = []
        used: set[int] = set()
        for ti in true_idx:
            if len(pred_idx) == 0:
                break
            candidates = [(abs(int(pi) - int(ti)), int(pi)) for pi in pred_idx if int(pi) not in used]
            if not candidates:
                break
            _, pi = min(candidates)
            used.add(pi)
            records.append(
                {
                    "kind": kind,
                    "true_date": dates[int(ti)].strftime("%Y-%m"),
                    "pred_date": dates[int(pi)].strftime("%Y-%m"),
                    "true_value": float(smooth_true[int(ti)]),
                    "pred_value": float(smooth_pred[int(pi)]),
                    "amplitude_error": float(smooth_pred[int(pi)] - smooth_true[int(ti)]),
                    "time_error_months": int(round((dates[int(pi)] - dates[int(ti)]).days / 30.4375)),
                }
            )
        return records

    peaks = match(true_peaks, pred_peaks, "peak")
    troughs = match(true_troughs, pred_troughs, "trough")

    def summarize(records: list[dict]) -> dict:
        if not records:
            return {
                "n": 0,
                "abs_amplitude_error_mean": None,
                "abs_time_error_months_mean": None,
            }
        return {
            "n": len(records),
            "abs_amplitude_error_mean": float(np.mean([abs(r["amplitude_error"]) for r in records])),
            "abs_time_error_months_mean": float(
                np.mean([abs(r["time_error_months"]) for r in records])
            ),
        }

    return {
        "true_peak_count": int(len(true_peaks)),
        "pred_peak_count": int(len(pred_peaks)),
        "true_trough_count": int(len(true_troughs)),
        "pred_trough_count": int(len(pred_troughs)),
        "peak_summary": summarize(peaks),
        "trough_summary": summarize(troughs),
        "peaks": peaks,
        "troughs": troughs,
    }


def evaluate_one(result_dir: Path) -> dict:
    pred_z = np.load(result_dir / "pred.npy")
    true_z = np.load(result_dir / "true.npy")
    meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
    if int(meta["pred_len"]) != 132:
        raise ValueError(f"expected pred_len=132 in {result_dir}")
    if pred_z.ndim != 3 or true_z.ndim != 3:
        raise ValueError(f"unexpected pred/true shape in {result_dir}")

    pred = physical_target(pred_z[:, :, -1], meta)
    true = physical_target(true_z[:, :, -1], meta)
    curve = build_direct_cycle_curve(pred, true, meta)
    full = metric_block(pred, true)
    cycle = metric_block(curve["pred"].to_numpy(), curve["true"].to_numpy())
    lead_mae = np.mean(np.abs(pred - true), axis=0).tolist()
    extrema = extrema_errors(curve)

    result = {
        "setting": meta["setting"],
        "fold": int(meta["fold"]),
        "outer_val": meta["fold_spec"]["outer_val"],
        "n_windows": int(len(pred)),
        "n_direct_cycle_windows": int(curve["window_index"].nunique()),
        "n_direct_cycle_months": int(len(curve)),
        "full_window_metrics": full,
        "direct_cycle_metrics": cycle,
        "lead_mae": [float(v) for v in lead_mae],
        **extrema,
    }
    (result_dir / "p132_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    curve.to_csv(result_dir / "p132_curve.csv", index=False)
    return result


def aggregate(results: list[dict]) -> dict:
    rows = []
    for result in results:
        peak = result["peak_summary"]
        trough = result["trough_summary"]
        rows.append(
            {
                "fold": result["fold"],
                "cycle_MAE": result["direct_cycle_metrics"]["mae"],
                "cycle_RMSE": result["direct_cycle_metrics"]["rmse"],
                "cycle_bias": result["direct_cycle_metrics"]["bias"],
                "full_MAE": result["full_window_metrics"]["mae"],
                "full_RMSE": result["full_window_metrics"]["rmse"],
                "full_bias": result["full_window_metrics"]["bias"],
                "peak_abs_amp": peak["abs_amplitude_error_mean"],
                "peak_abs_time": peak["abs_time_error_months_mean"],
                "trough_abs_amp": trough["abs_amplitude_error_mean"],
                "trough_abs_time": trough["abs_time_error_months_mean"],
                "n_cycle_windows": result["n_direct_cycle_windows"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("fold")
    numeric_cols = [c for c in summary.columns if c != "fold"]
    aggregate_result = {}
    for col in numeric_cols:
        vals = summary[col].dropna().to_numpy(dtype=float)
        aggregate_result[col] = {
            "mean": float(np.mean(vals)) if len(vals) else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "worst": float(np.max(vals)) if len(vals) else None,
        }
    return {"folds": rows, **aggregate_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate p132 direct-cycle CV5 outputs")
    parser.add_argument("--des", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_glob", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pattern = args.results_glob or f"results/*_{args.des}_F*_seed{args.seed}/"
    dirs = [
        Path(path)
        for path in sorted(glob.glob(pattern))
        if Path(path, "cv_window_meta.json").exists()
    ]
    if not dirs:
        raise SystemExit(f"no CV5 result directories found: {pattern}")

    results = sorted([evaluate_one(path) for path in dirs], key=lambda item: item["fold"])
    out_dir = Path(args.out_dir or f"analysis_p132/{args.des}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(aggregate(results)["folds"])
    summary.to_csv(out_dir / "cv5_summary.csv", index=False)
    (out_dir / "cv5_aggregate.json").write_text(
        json.dumps(aggregate(results), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
