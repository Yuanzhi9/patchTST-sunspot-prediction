"""Evaluate saved PR24 cycle-event auxiliary predictions."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "next_peak_raw",
    "months_to_next_peak",
    "months_to_next_trough",
    "cycle_length",
]
MODE_COLUMNS = {
    "none": [],
    "amp": [0],
    "peak_time": [1],
    "trough_time": [2],
    "all": [0, 1, 2],
}


def physical_amplitude(values: np.ndarray, meta: dict) -> np.ndarray:
    mean = float(meta["scaler_mean"][-1])
    scale = float(meta["scaler_scale"][-1])
    return values * scale + mean


def metric_rows(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    mode: str,
    meta: dict,
) -> list[dict]:
    rows = []
    for column in MODE_COLUMNS[mode]:
        valid = mask[:, column].astype(bool)
        if not valid.any():
            continue
        p = pred[valid, column]
        t = true[valid, column]
        if column == 0:
            p = physical_amplitude(p, meta)
            t = physical_amplitude(t, meta)
            unit = "SSN"
        else:
            p = p * 12.0
            t = t * 12.0
            unit = "months"
        error = p - t
        row = {
            "column": COLUMNS[column],
            "unit": unit,
            "n": int(valid.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
            "median_abs_error": float(np.median(np.abs(error))),
        }
        if unit == "months":
            row["within_6m"] = float(np.mean(np.abs(error) <= 6))
            row["within_12m"] = float(np.mean(np.abs(error) <= 12))
        rows.append(row)
    return rows


def evaluate_one(result_dir: Path) -> tuple[list[dict], dict]:
    meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
    pred = np.load(result_dir / "aux_pred.npy")
    true = np.load(result_dir / "aux_true.npy")
    mask = np.load(result_dir / "aux_mask.npy")
    rows = []
    for row in metric_rows(pred, true, mask, meta["mt_mode"], meta):
        rows.append(
            {
                "fold": int(meta["fold"]),
                "mode": meta["mt_mode"],
                **row,
            }
        )
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PR24 multitask aux heads")
    parser.add_argument("--des", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    pattern = f"results/*_{args.des}_F*_seed{args.seed}/"
    result_dirs = [
        Path(path)
        for path in sorted(glob.glob(pattern))
        if Path(path, "aux_pred.npy").exists()
    ]
    if not result_dirs:
        raise SystemExit(f"no multitask aux outputs found: {pattern}")

    all_rows = []
    for result_dir in result_dirs:
        rows, _ = evaluate_one(result_dir)
        all_rows.extend(rows)
    detail = pd.DataFrame(all_rows).sort_values(["column", "fold"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "aux_detail.csv", index=False)

    summary_rows = []
    for column, group in detail.groupby("column", sort=True):
        row = {
            "column": column,
            "folds": int(len(group)),
            "valid_samples": int(group["n"].sum()),
            "mae_mean": float(group["mae"].mean()),
            "mae_std": float(group["mae"].std(ddof=0)),
            "rmse_mean": float(group["rmse"].mean()),
            "bias_mean": float(group["bias"].mean()),
            "median_abs_error_mean": float(group["median_abs_error"].mean()),
        }
        if "within_6m" in group:
            row["within_6m_mean"] = float(group["within_6m"].mean())
            row["within_12m_mean"] = float(group["within_12m"].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "aux_summary.csv", index=False)
    payload = {
        "des": args.des,
        "seed": args.seed,
        "folds_found": len(result_dirs),
        "summary": summary.to_dict(orient="records"),
    }
    (out_dir / "aux_aggregate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
