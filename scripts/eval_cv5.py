"""Evaluate BASE-2026-11-M CV5 fold outputs.

This is an analysis-only script. It reads outer-fold pred.npy / true.npy and
cv_window_meta.json, converts predictions back to physical SSN, computes
metrics, builds a non-overlapping block24 curve, and extracts peak/trough
values and dates.
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
    target_idx = -1
    mean = float(meta["scaler_mean"][target_idx])
    scale = float(meta["scaler_scale"][target_idx])
    out = z * scale + mean
    transform = meta.get("target_transform", "")
    if transform:
        raise ValueError(f"evaluation inverse for target_transform={transform!r} not implemented")
    return out


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


def build_non_overlapping_curve(pred: np.ndarray, true: np.ndarray, meta: dict) -> pd.DataFrame:
    starts = np.asarray(meta["window_target_starts"], dtype=int)
    dates = pd.to_datetime([d["start"] for d in meta["target_dates"]])
    outer = meta["fold_spec"]["outer_val"]
    current = int(starts[0])

    rows = []
    while current <= int(outer["end"]):
        matches = np.flatnonzero(starts == current)
        if len(matches) == 0:
            break
        i = int(matches[0])
        take = min(int(meta["pred_len"]), int(outer["end"]) - current + 1)
        for lead in range(take):
            rows.append(
                {
                    "date": dates[i] + pd.DateOffset(months=lead),
                    "true": float(true[i, lead]),
                    "pred": float(pred[i, lead]),
                    "window_index": i,
                    "lead": lead + 1,
                }
            )
        current += take
    return pd.DataFrame(rows)


def smooth(series: pd.Series, window: int = 13) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def extrema(curve: pd.DataFrame) -> dict:
    smooth_true = smooth(curve["true"])
    smooth_pred = smooth(curve["pred"])
    distance = max(24, len(curve) // 8)
    peak_idx, peak_props = find_peaks(
        smooth_true.to_numpy(), distance=distance, prominence=2.0
    )
    trough_idx, trough_props = find_peaks(
        -smooth_true.to_numpy(), distance=distance, prominence=2.0
    )

    def records(indices, prominences, kind: str) -> list[dict]:
        out = []
        for i, prominence in zip(indices, prominences):
            d = curve.iloc[i]["date"]
            out.append(
                {
                    "kind": kind,
                    "date": d.strftime("%Y-%m"),
                    "true": float(smooth_true.iloc[i]),
                    "pred": float(smooth_pred.iloc[i]),
                    "amplitude_error": float(smooth_pred.iloc[i] - smooth_true.iloc[i]),
                    "prominence": float(prominence),
                }
            )
        return sorted(out, key=lambda r: r["prominence"], reverse=True)

    peaks = records(peak_idx, peak_props.get("prominences", []), "peak")
    troughs = records(trough_idx, trough_props.get("prominences", []), "trough")
    if not peaks:
        i = int(np.argmax(smooth_true.to_numpy()))
        peaks = records([i], [float("inf")], "peak")
    if not troughs:
        i = int(np.argmin(smooth_true.to_numpy()))
        troughs = records([i], [float("inf")], "trough")
    return {"peaks": peaks, "troughs": troughs}


def evaluate_one(result_dir: Path) -> dict:
    pred_z = np.load(result_dir / "pred.npy")
    true_z = np.load(result_dir / "true.npy")
    meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
    if pred_z.ndim != 3 or true_z.ndim != 3:
        raise ValueError(f"unexpected pred/true shape in {result_dir}")

    pred = physical_target(pred_z[:, :, -1], meta)
    true = physical_target(true_z[:, :, -1], meta)
    curve = build_non_overlapping_curve(pred, true, meta)

    full = metric_block(pred, true)
    block = metric_block(curve["pred"].to_numpy(), curve["true"].to_numpy())
    step_mae = np.mean(np.abs(pred - true), axis=0).tolist()
    ext = extrema(curve)
    result = {
        "setting": meta["setting"],
        "fold": int(meta["fold"]),
        "outer_val": meta["fold_spec"]["outer_val"],
        "n_windows": int(len(pred)),
        "n_block_curve_months": int(len(curve)),
        "full_window_metrics": full,
        "block24_metrics": block,
        "lead_mae": [float(v) for v in step_mae],
        **ext,
    }
    (result_dir / "cv_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    curve.to_csv(result_dir / "cv_block_curve.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BASE-2026-11-M CV5 folds")
    parser.add_argument("--des", required=True, help="experiment ID, e.g. B26M-E00")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--results_glob", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pattern = args.results_glob or f"results/*_{args.des}_F*_seed{args.seed}/"
    dirs = [Path(p) for p in sorted(glob.glob(pattern)) if Path(p, "cv_window_meta.json").exists()]
    if not dirs:
        raise SystemExit(f"no CV5 result directories found: {pattern}")

    results = sorted([evaluate_one(d) for d in dirs], key=lambda r: r["fold"])
    out_dir = Path(args.out_dir or f"analysis/{args.des}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in results:
        row = {
            "fold": r["fold"],
            "test_start": r["outer_val"]["start_date"],
            "test_end": r["outer_val"]["end_date"],
            "n_windows": r["n_windows"],
            "full_MAE": r["full_window_metrics"]["mae"],
            "full_RMSE": r["full_window_metrics"]["rmse"],
            "full_bias": r["full_window_metrics"]["bias"],
            "block24_MAE": r["block24_metrics"]["mae"],
            "block24_RMSE": r["block24_metrics"]["rmse"],
            "block24_bias": r["block24_metrics"]["bias"],
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cv5_summary.csv", index=False)
    aggregate = {
        "des": args.des,
        "seed": args.seed,
        "folds_found": len(results),
        "full_MAE_mean": float(summary["full_MAE"].mean()),
        "full_MAE_std": float(summary["full_MAE"].std(ddof=0)),
        "full_MAE_worst": float(summary["full_MAE"].max()),
        "block24_MAE_mean": float(summary["block24_MAE"].mean()),
        "block24_MAE_std": float(summary["block24_MAE"].std(ddof=0)),
        "block24_MAE_worst": float(summary["block24_MAE"].max()),
    }
    (out_dir / "cv5_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
