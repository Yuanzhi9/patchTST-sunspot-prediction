"""Evaluate one-cycle curve fit and its peak/trough amplitude and timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def smooth(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, center=True, min_periods=window).mean()


def metric_block(pred: np.ndarray, true: np.ndarray) -> dict:
    diff = pred - true
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "negative_months": int(np.sum(pred < 0)),
    }


def extrema(frame: pd.DataFrame, window: int, order: str) -> dict:
    dates = pd.to_datetime(frame["date"])
    true = frame["true"].astype(float)
    pred = frame["pred"].astype(float)
    smooth_true = smooth(true, window)
    smooth_pred = smooth(pred, window)
    valid = smooth_true.notna() & smooth_pred.notna()
    if not valid.any():
        raise RuntimeError("no valid smoothed values")

    if order == "peak_then_trough":
        idx_true_peak = smooth_true[valid].idxmax()
        idx_pred_peak = smooth_pred[valid].idxmax()
        after_true = valid & (pd.Series(np.arange(len(frame)), index=frame.index) > idx_true_peak)
        after_pred = valid & (pd.Series(np.arange(len(frame)), index=frame.index) > idx_pred_peak)
        idx_true_trough = smooth_true[after_true].idxmin()
        idx_pred_trough = smooth_pred[after_pred].idxmin()
    elif order == "trough_then_peak":
        idx_true_trough = smooth_true[valid].idxmin()
        idx_pred_trough = smooth_pred[valid].idxmin()
        after_true = valid & (pd.Series(np.arange(len(frame)), index=frame.index) > idx_true_trough)
        after_pred = valid & (pd.Series(np.arange(len(frame)), index=frame.index) > idx_pred_trough)
        idx_true_peak = smooth_true[after_true].idxmax()
        idx_pred_peak = smooth_pred[after_pred].idxmax()
    else:
        raise ValueError(f"unsupported extrema order: {order}")

    def timing(pred_idx: int, true_idx: int) -> int:
        return int((dates.iloc[pred_idx].to_period("M") - dates.iloc[true_idx].to_period("M")).n)

    pred_peak = float(smooth_pred.iloc[idx_pred_peak])
    pred_trough = float(smooth_pred.iloc[idx_pred_trough])
    true_peak = float(smooth_true.iloc[idx_true_peak])
    true_trough = float(smooth_true.iloc[idx_true_trough])
    return {
        "smooth_window_months": window,
        "order": order,
        "peak": {
            "true_date": dates.iloc[idx_true_peak].strftime("%Y-%m"),
            "true_value": true_peak,
            "pred_date": dates.iloc[idx_pred_peak].strftime("%Y-%m"),
            "pred_value": pred_peak,
            "amplitude_error": pred_peak - true_peak,
            "time_error_months": timing(idx_pred_peak, idx_true_peak),
        },
        "trough": {
            "true_date": dates.iloc[idx_true_trough].strftime("%Y-%m"),
            "true_value": true_trough,
            "pred_date": dates.iloc[idx_pred_trough].strftime("%Y-%m"),
            "pred_value": pred_trough,
            "amplitude_error": pred_trough - true_trough,
            "time_error_months": timing(idx_pred_trough, idx_true_trough),
        },
        "amplitude_ratio": float((pred.max() - pred.min()) / (true.max() - true.min())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth_window", type=int, default=13)
    parser.add_argument(
        "--order",
        choices=["peak_then_trough", "trough_then_peak"],
        default="peak_then_trough",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.curve, parse_dates=["date"])
    required = {"date", "pred", "true"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    pred = frame["pred"].to_numpy(dtype=float)
    true = frame["true"].to_numpy(dtype=float)
    payload = {
        "curve": str(args.curve),
        "n_months": int(len(frame)),
        "raw_metrics": metric_block(pred, true),
        "smoothed": extrema(frame, args.smooth_window, args.order),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
