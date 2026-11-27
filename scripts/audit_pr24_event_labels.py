"""Build and audit PR24 cycle-event labels for multitask auxiliary heads.

This is a zero-training audit. It creates one event-label row for every CV5
sample window without changing any model or dataset training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
DEFAULT_SMOOTH = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_smoothed13_2026-04.csv"
DEFAULT_BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
DEFAULT_CV = ROOT / "configs/cv5_folds_2026-10.json"
DEFAULT_OUT = ROOT / "analysis/pr24_multitask_extrema/a0_labels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_delta(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return (end.year - start.year) * 12 + (end.month - start.month)


def build_ranges(fold: dict, flag: str) -> list[tuple[int, int]]:
    if flag == "test":
        outer = fold["outer_val"]
        return [(int(outer["start"]), int(outer["end"]))]

    key = "inner_train" if flag == "train" else "inner_val"
    return [
        (int(block[key]["start"]), int(block[key]["end"]))
        for block in fold["train_blocks"]
    ]


def build_windows(
    fold: dict,
    flag: str,
    seq_len: int,
    pred_len: int,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for segment_start, segment_end in build_ranges(fold, flag):
        latest_start = segment_end - pred_len + 1
        for target_start in range(segment_start, latest_start + 1):
            if flag == "train" and target_start - seq_len < segment_start:
                continue
            windows.append((target_start, target_start + pred_len - 1))
    return windows


def next_event(
    origin: pd.Timestamp,
    events: list[dict],
    event_date_key: str,
) -> dict | None:
    future = [event for event in events if event[event_date_key] > origin]
    return min(future, key=lambda event: event[event_date_key]) if future else None


def validate_boundaries(
    raw: pd.DataFrame,
    smooth: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_by_date = raw.set_index("date")["ssn"]
    smooth_by_date = smooth.set_index("date")["ssn"]

    for row in boundaries.itertuples(index=False):
        if row.min_date not in raw_by_date.index or row.min_date not in smooth_by_date.index:
            errors.append(f"cycle {row.cycle}: min_date missing at {row.min_date:%Y-%m}")
        if row.peak_date not in raw_by_date.index or row.peak_date not in smooth_by_date.index:
            errors.append(f"cycle {row.cycle}: peak_date missing at {row.peak_date:%Y-%m}")
        if pd.notna(row.next_min_date) and (
            row.next_min_date not in raw_by_date.index
            or row.next_min_date not in smooth_by_date.index
        ):
            errors.append(
                f"cycle {row.cycle}: next_min_date missing at {row.next_min_date:%Y-%m}"
            )
        if not row.min_date < row.peak_date:
            errors.append(f"cycle {row.cycle}: min_date is not before peak_date")
        if pd.notna(row.next_min_date) and not row.peak_date < row.next_min_date:
            errors.append(f"cycle {row.cycle}: peak_date is not before next_min_date")

        cycle_end = row.next_min_date if pd.notna(row.next_min_date) else raw["date"].iloc[-1]
        cycle_raw = raw.loc[
            (raw["date"] >= row.min_date) & (raw["date"] <= cycle_end), "ssn"
        ]
        if not cycle_raw.empty and not np.isclose(
            float(cycle_raw.max()), float(row.peak_raw), atol=0.05
        ):
            errors.append(
                f"cycle {row.cycle}: peak_raw {row.peak_raw} != interval max {cycle_raw.max()}"
            )
        if row.peak_date in smooth_by_date.index and not np.isclose(
            float(smooth_by_date.loc[row.peak_date]),
            float(row.peak_smooth),
            atol=0.05,
        ):
            warnings.append(
                f"cycle {row.cycle}: peak_smooth differs from smoothed CSV by "
                f"{abs(float(smooth_by_date.loc[row.peak_date]) - float(row.peak_smooth)):.4f}"
            )

    expected_smooth_dates = boundaries["peak_date"].dropna()
    missing_peak_dates = expected_smooth_dates[
        ~expected_smooth_dates.isin(smooth_by_date.index)
    ]
    if not missing_peak_dates.empty:
        errors.append(
            "smoothed CSV lacks peak dates: "
            + ", ".join(date.strftime("%Y-%m") for date in missing_peak_dates)
        )
    return errors, warnings


def coverage(frame: pd.DataFrame, mask_key: str) -> dict:
    return {
        "n": int(len(frame)),
        "known": int(frame[mask_key].sum()),
        "missing": int((~frame[mask_key]).sum()),
        "coverage": float(frame[mask_key].mean()),
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (fold, flag), group in frame.groupby(["fold", "flag"], sort=True):
        peak = coverage(group, "peak_mask")
        trough = coverage(group, "trough_mask")
        cycle = coverage(group, "cycle_length_mask")
        rows.append(
            {
                "fold": int(fold),
                "flag": flag,
                "windows": int(len(group)),
                "peak_known": peak["known"],
                "peak_coverage": peak["coverage"],
                "trough_known": trough["known"],
                "trough_coverage": trough["coverage"],
                "cycle_length_known": cycle["known"],
                "cycle_length_coverage": cycle["coverage"],
                "next_peak_in_24m": float(group["next_peak_in_horizon"].mean()),
                "next_trough_in_24m": float(group["next_trough_in_horizon"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["flag", "fold"])
    totals = {}
    for flag in ["train", "val", "test"]:
        group = frame.loc[frame["flag"] == flag]
        totals[flag] = {
            "windows": int(len(group)),
            "peak_coverage": float(group["peak_mask"].mean()),
            "trough_coverage": float(group["trough_mask"].mean()),
            "cycle_length_coverage": float(group["cycle_length_mask"].mean()),
            "next_peak_in_24m": float(group["next_peak_in_horizon"].mean()),
            "next_trough_in_24m": float(group["next_trough_in_horizon"].mean()),
        }
    return summary, totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PR24 cycle-event labels")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--smooth", type=Path, default=DEFAULT_SMOOTH)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--cv_file", type=Path, default=DEFAULT_CV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seq_len", type=int, default=132)
    parser.add_argument("--pred_len", type=int, default=24)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, parse_dates=["date"]).sort_values("date")
    smooth = pd.read_csv(args.smooth, parse_dates=["date"]).sort_values("date")
    boundaries = pd.read_csv(
        args.boundaries,
        parse_dates=["min_date", "peak_date", "next_min_date"],
    ).sort_values(["cycle"])
    cv_spec = json.loads(args.cv_file.read_text(encoding="utf-8"))

    errors, warnings = validate_boundaries(raw, smooth, boundaries)
    if len(raw) != int(cv_spec["n_months"]):
        errors.append(f"raw rows {len(raw)} != CV n_months {cv_spec['n_months']}")
    if raw["date"].iloc[0].strftime("%Y-%m") != cv_spec["start_date"]:
        errors.append("raw start date does not match CV spec")
    if raw["date"].iloc[-1].strftime("%Y-%m") != cv_spec["end_date"]:
        errors.append("raw end date does not match CV spec")

    peaks = [
        {
            "cycle": int(row.cycle),
            "date": row.peak_date,
            "raw": float(row.peak_raw),
            "smooth": float(row.peak_smooth),
        }
        for row in boundaries.itertuples(index=False)
    ]
    trough_by_date: dict[pd.Timestamp, int] = {}
    for row in boundaries.itertuples(index=False):
        trough_by_date[row.min_date] = int(row.cycle)
        if pd.notna(row.next_min_date):
            trough_by_date[row.next_min_date] = int(row.cycle) + 1
    troughs = [
        {"cycle": cycle, "date": date}
        for date, cycle in sorted(trough_by_date.items(), key=lambda item: item[0])
    ]

    rows: list[dict] = []
    for fold in cv_spec["folds"]:
        fold_id = int(fold["fold"])
        for flag in ["train", "val", "test"]:
            windows = build_windows(fold, flag, args.seq_len, args.pred_len)
            for window_index, (start, end) in enumerate(windows):
                origin = raw["date"].iloc[start]
                target_end = raw["date"].iloc[end]
                next_peak = next_event(origin, peaks, "date")
                next_trough = next_event(origin, troughs, "date")

                current_cycle = boundaries.loc[
                    (boundaries["min_date"] <= origin)
                    & (
                        boundaries["next_min_date"].isna()
                        | (origin < boundaries["next_min_date"])
                    )
                ]
                if len(current_cycle) == 0 and origin < boundaries["min_date"].min():
                    cycle_length = np.nan
                    cycle_number = np.nan
                elif len(current_cycle) != 1:
                    errors.append(
                        f"fold {fold_id} {flag} window {window_index}: "
                        f"{len(current_cycle)} current cycles at {origin:%Y-%m}"
                    )
                    cycle_length = np.nan
                    cycle_number = np.nan
                else:
                    cycle_row = current_cycle.iloc[0]
                    cycle_number = int(cycle_row["cycle"])
                    cycle_length = (
                        month_delta(cycle_row["min_date"], cycle_row["next_min_date"])
                        if pd.notna(cycle_row["next_min_date"])
                        else np.nan
                    )

                peak_date = next_peak["date"] if next_peak else pd.NaT
                trough_date = next_trough["date"] if next_trough else pd.NaT
                months_to_peak = (
                    month_delta(origin, peak_date) if pd.notna(peak_date) else np.nan
                )
                months_to_trough = (
                    month_delta(origin, trough_date) if pd.notna(trough_date) else np.nan
                )

                rows.append(
                    {
                        "fold": fold_id,
                        "flag": flag,
                        "window_index": window_index,
                        "target_start_index": int(start),
                        "target_end_index": int(end),
                        "target_start_date": origin.strftime("%Y-%m"),
                        "target_end_date": target_end.strftime("%Y-%m"),
                        "current_cycle": cycle_number,
                        "next_peak_cycle": next_peak["cycle"] if next_peak else np.nan,
                        "next_peak_date": (
                            peak_date.strftime("%Y-%m") if pd.notna(peak_date) else ""
                        ),
                        "next_peak_raw": next_peak["raw"] if next_peak else np.nan,
                        "next_peak_smooth": next_peak["smooth"] if next_peak else np.nan,
                        "months_to_next_peak": months_to_peak,
                        "next_trough_cycle": next_trough["cycle"] if next_trough else np.nan,
                        "next_trough_date": (
                            trough_date.strftime("%Y-%m") if pd.notna(trough_date) else ""
                        ),
                        "months_to_next_trough": months_to_trough,
                        "cycle_length": cycle_length,
                        "next_peak_in_horizon": bool(
                            pd.notna(months_to_peak) and 0 < months_to_peak < args.pred_len
                        ),
                        "next_trough_in_horizon": bool(
                            pd.notna(months_to_trough)
                            and 0 < months_to_trough < args.pred_len
                        ),
                        "peak_mask": bool(next_peak is not None),
                        "trough_mask": bool(next_trough is not None),
                        "cycle_length_mask": bool(pd.notna(cycle_length)),
                    }
                )

    labels = pd.DataFrame(rows)
    for key in ["months_to_next_peak", "next_peak_raw", "next_peak_smooth"]:
        if (labels.loc[labels["peak_mask"], key] < 0).any():
            errors.append(f"negative value in {key}")
    for key in ["months_to_next_trough", "cycle_length"]:
        if (labels.loc[labels[("trough_mask" if "trough" in key else "cycle_length_mask")], key] <= 0).any():
            errors.append(f"non-positive value in masked {key}")

    # Every training window must have a fully known next-peak event.
    train_labels = labels.loc[labels["flag"] == "train"]
    if not train_labels["peak_mask"].all():
        errors.append("training windows contain masked next-peak labels")
    if not train_labels["trough_mask"].all():
        errors.append("training windows contain masked next-trough labels")

    # Input columns are fixed and intentionally contain no event label.
    input_columns = ["month_sin", "month_cos", "ssn"]
    forbidden_inputs = {
        "next_peak_raw",
        "months_to_next_peak",
        "months_to_next_trough",
        "cycle_length",
        "next_peak_date",
        "next_trough_date",
    }
    leaked = forbidden_inputs.intersection(input_columns)
    if leaked:
        errors.append(f"event labels present in input columns: {sorted(leaked)}")

    summary_table, totals = summarize(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "event_labels_all_windows.csv"
    summary_path = args.output_dir / "coverage_by_fold.csv"
    labels.to_csv(labels_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "raw": {"path": str(args.raw), "sha256": sha256(args.raw)},
            "smooth": {"path": str(args.smooth), "sha256": sha256(args.smooth)},
            "boundaries": {
                "path": str(args.boundaries),
                "sha256": sha256(args.boundaries),
            },
            "cv_file": {"path": str(args.cv_file), "sha256": sha256(args.cv_file)},
        },
        "outputs": {
            "event_labels": {
                "path": str(labels_path),
                "sha256": sha256(labels_path),
            },
            "coverage_by_fold": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
        },
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "input_columns": input_columns,
        "label_columns": sorted(forbidden_inputs),
        "totals": totals,
    }
    result_path = args.output_dir / "audit_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary_table.to_string(index=False))
    print(json.dumps({"status": result["status"], "output": str(result_path)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
