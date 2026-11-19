"""Create the frozen sequential 5-fold definition for BASE-2026-11-M.

This script only describes data blocks. It never trains or scales data.
The resulting JSON is shared by every experiment in the branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def block_record(df: pd.DataFrame, block_id: int, start: int, end: int) -> dict:
    return {
        "block_id": block_id,
        "start": int(start),
        "end": int(end),
        "n_months": int(end - start + 1),
        "start_date": str(df.loc[start, "date"])[:7],
        "end_date": str(df.loc[end, "date"])[:7],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen CV5 fold specification")
    parser.add_argument(
        "--data_csv",
        default="PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv",
    )
    parser.add_argument("--output", default="configs/cv5_folds_2026-10.json")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--inner_val_months", type=int, default=132)
    args = parser.parse_args()

    data_path = Path(args.data_csv)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    required = {"date", "month_sin", "month_cos", "ssn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("date column must be monotonic increasing")
    if dates.duplicated().any():
        raise ValueError("date column contains duplicates")

    block_indices = [a.astype(int) for a in np.array_split(np.arange(len(df)), args.n_folds)]
    blocks = [
        block_record(df, i + 1, int(idx[0]), int(idx[-1]))
        for i, idx in enumerate(block_indices)
    ]

    folds = []
    for fold_idx, outer_idx in enumerate(block_indices, start=1):
        outer = block_record(df, fold_idx, int(outer_idx[0]), int(outer_idx[-1]))
        train_blocks = []
        for block_idx, idx in enumerate(block_indices, start=1):
            if block_idx == fold_idx:
                continue
            start, end = int(idx[0]), int(idx[-1])
            inner_val_start = end - args.inner_val_months + 1
            if inner_val_start <= start:
                raise ValueError(f"block {block_idx} too short for inner validation")
            record = block_record(df, block_idx, start, end)
            record["inner_train"] = {
                "start": start,
                "end": inner_val_start - 1,
                "n_months": inner_val_start - start,
                "start_date": str(df.loc[start, "date"])[:7],
                "end_date": str(df.loc[inner_val_start - 1, "date"])[:7],
            }
            record["inner_val"] = {
                "start": inner_val_start,
                "end": end,
                "n_months": args.inner_val_months,
                "start_date": str(df.loc[inner_val_start, "date"])[:7],
                "end_date": str(df.loc[end, "date"])[:7],
            }
            train_blocks.append(record)

        folds.append(
            {
                "fold": fold_idx,
                "outer_val": outer,
                "train_blocks": train_blocks,
            }
        )

    spec = {
        "schema": "BASE-2026-11-M-cv5-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_csv": str(data_path),
        "data_sha256": sha256(data_path),
        "n_months": int(len(df)),
        "start_date": str(dates.iloc[0])[:7],
        "end_date": str(dates.iloc[-1])[:7],
        "n_folds": args.n_folds,
        "split_rule": "sequential_equal_blocks_no_shuffle",
        "inner_val_months": args.inner_val_months,
        "blocks": blocks,
        "folds": folds,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}")
    for block in blocks:
        print(
            f"F{block['block_id']}: {block['start_date']}..{block['end_date']} "
            f"n={block['n_months']}"
        )


if __name__ == "__main__":
    main()
