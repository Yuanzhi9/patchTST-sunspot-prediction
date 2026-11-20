"""Compare frozen p24 and p132 baselines on the same fold/date intersection."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


BLUE = "#0072B2"
ORANGE = "#E69F00"
BLACK = "#111111"


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    diff = pred - true
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def extrema(pred: np.ndarray, true: np.ndarray, dates: list[pd.Timestamp]) -> dict:
    smooth_true = pd.Series(true).rolling(13, center=True, min_periods=1).mean().to_numpy()
    smooth_pred = pd.Series(pred).rolling(13, center=True, min_periods=1).mean().to_numpy()
    distance = max(24, len(true) // 8)
    records = {}
    for kind, sign in (("peak", 1), ("trough", -1)):
        true_idx, _ = find_peaks(sign * smooth_true, distance=distance, prominence=2.0)
        pred_idx, _ = find_peaks(sign * smooth_pred, distance=distance, prominence=2.0)
        matches = []
        used: set[int] = set()
        for ti in true_idx:
            candidates = [(abs(int(pi) - int(ti)), int(pi)) for pi in pred_idx if int(pi) not in used]
            if not candidates:
                break
            _, pi = min(candidates)
            used.add(pi)
            matches.append(
                {
                    "amplitude_error": float(smooth_pred[pi] - smooth_true[ti]),
                    "time_error_months": int(round((dates[pi] - dates[ti]).days / 30.4375)),
                }
            )
        records[kind] = {
            "n": len(matches),
            "abs_amplitude_error_mean": (
                float(np.mean([abs(item["amplitude_error"]) for item in matches]))
                if matches else None
            ),
            "abs_time_error_months_mean": (
                float(np.mean([abs(item["time_error_months"]) for item in matches]))
                if matches else None
            ),
        }
    return records


def find_dirs(root: str, des: str, seed: int, marker: str) -> list[Path]:
    pattern = f"{root.rstrip('/')}/*_{des}_F*_seed{seed}/"
    return [
        Path(path)
        for path in sorted(glob.glob(pattern))
        if Path(path, marker).exists() and Path(path, "cv_window_meta.json").exists()
    ]


def fold_number(path: Path) -> int:
    meta = json.loads((path / "cv_window_meta.json").read_text(encoding="utf-8"))
    return int(meta["fold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare p24 and p132 baselines")
    parser.add_argument("--p24-des", required=True)
    parser.add_argument("--p24-seed", type=int, default=2021)
    parser.add_argument("--p24-results-root", default="/output/BASE-2026-11-M/results")
    parser.add_argument("--p132-des", required=True)
    parser.add_argument("--p132-seed", type=int, default=2021)
    parser.add_argument("--p132-results-root", default="results")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    p24_dirs = {
        fold_number(path): path
        for path in find_dirs(
            args.p24_results_root, args.p24_des, args.p24_seed, "cv_block_curve.csv"
        )
    }
    p132_dirs = {
        fold_number(path): path
        for path in find_dirs(
            args.p132_results_root, args.p132_des, args.p132_seed, "p132_curve.csv"
        )
    }
    folds = sorted(set(p24_dirs) & set(p132_dirs))
    if not folds:
        raise SystemExit("no matching evaluated folds")

    rows = []
    curves = {}
    for fold in folds:
        p24_curve = pd.read_csv(p24_dirs[fold] / "cv_block_curve.csv", parse_dates=["date"])
        p132_curve = pd.read_csv(p132_dirs[fold] / "p132_curve.csv", parse_dates=["date"])
        merged = (
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
        if len(merged) < 132:
            raise ValueError(f"fold {fold}: only {len(merged)} comparable months")

        true = merged["true"].to_numpy()
        date_list = pd.to_datetime(merged["date"]).to_list()
        p24_metrics = metrics(merged["p24"].to_numpy(), true)
        p132_metrics = metrics(merged["p132"].to_numpy(), true)
        p24_extrema = extrema(merged["p24"].to_numpy(), true, date_list)
        p132_extrema = extrema(merged["p132"].to_numpy(), true, date_list)
        row = {
            "fold": fold,
            "n_common_months": len(merged),
            "common_start": date_list[0].strftime("%Y-%m"),
            "common_end": date_list[-1].strftime("%Y-%m"),
            "p24_MAE": p24_metrics["mae"],
            "p24_RMSE": p24_metrics["rmse"],
            "p24_bias": p24_metrics["bias"],
            "p132_MAE": p132_metrics["mae"],
            "p132_RMSE": p132_metrics["rmse"],
            "p132_bias": p132_metrics["bias"],
            "p24_peak_abs_amp": p24_extrema["peak"]["abs_amplitude_error_mean"],
            "p132_peak_abs_amp": p132_extrema["peak"]["abs_amplitude_error_mean"],
            "p24_peak_abs_time": p24_extrema["peak"]["abs_time_error_months_mean"],
            "p132_peak_abs_time": p132_extrema["peak"]["abs_time_error_months_mean"],
            "p24_trough_abs_amp": p24_extrema["trough"]["abs_amplitude_error_mean"],
            "p132_trough_abs_amp": p132_extrema["trough"]["abs_amplitude_error_mean"],
            "p24_trough_abs_time": p24_extrema["trough"]["abs_time_error_months_mean"],
            "p132_trough_abs_time": p132_extrema["trough"]["abs_time_error_months_mean"],
        }
        rows.append(row)
        curves[fold] = merged

    summary = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {}
    for column in summary.columns:
        if column in {"fold", "common_start", "common_end"}:
            continue
        values = summary[column].dropna().to_numpy(dtype=float)
        aggregate[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "worst": float(np.max(values)),
        }
    payload = {
        "p24": {"des": args.p24_des, "seed": args.p24_seed},
        "p132": {"des": args.p132_des, "seed": args.p132_seed},
        "folds": rows,
        "aggregate": aggregate,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(len(folds), 1, figsize=(12, 2.8 * len(folds)), dpi=300, squeeze=False)
    for ax, fold in zip(axes[:, 0], folds):
        curve = curves[fold]
        ax.plot(curve["date"], curve["true"], color=BLACK, lw=1.25, label="Observed")
        ax.plot(curve["date"], curve["p24"], color=ORANGE, lw=1.1, ls="--", label="p24 block")
        ax.plot(curve["date"], curve["p132"], color=BLUE, lw=1.1, ls=":", label="p132 direct")
        row = summary.loc[summary["fold"] == fold].iloc[0]
        ax.set_title(
            f"Fold {fold} | p24 MAE {row['p24_MAE']:.2f} | p132 MAE {row['p132_MAE']:.2f}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.suptitle(
        f"{args.p24_des} vs {args.p132_des} | same-date intersection | seed "
        f"{args.p24_seed}/{args.p132_seed}",
        fontsize=10,
    )
    fig.tight_layout()
    out_png = out_dir / "comparison_curves.png"
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
