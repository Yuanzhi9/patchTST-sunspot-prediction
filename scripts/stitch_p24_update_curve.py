"""Stitch the latest PR24 forecast under a 24-month observed-update schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
    pred_z = np.load(result_dir / "pred.npy")[:, :, -1]
    true_z = np.load(result_dir / "true.npy")[:, :, -1]
    mean = float(meta["scaler_mean"][-1])
    scale = float(meta["scaler_scale"][-1])
    pred = pred_z * scale + mean
    true = true_z * scale + mean

    starts = np.asarray(meta["window_target_starts"], dtype=int)
    start_to_index = {int(start): index for index, start in enumerate(starts)}
    dates = pd.to_datetime([item["start"] for item in meta["target_dates"]])
    outer = meta["fold_spec"]["outer_val"]
    first = int(starts.min())
    last = int(outer["end"])
    pred_len = int(meta["pred_len"])

    final_start = last - pred_len + 1
    scheduled = list(range(first, final_start + 1, pred_len))
    if final_start >= first and final_start not in scheduled:
        scheduled.append(final_start)

    records: dict[pd.Timestamp, tuple[float, float, int, int]] = {}
    for block_index, start in enumerate(scheduled):
        index = start_to_index.get(int(start))
        if index is None:
            raise RuntimeError(f"missing model window for target start {start}")
        for lead in range(pred_len):
            target_index = int(start) + lead
            if target_index > last:
                break
            target_date = dates[index] + pd.DateOffset(months=lead)
            records[target_date] = (
                float(pred[index, lead]),
                float(true[index, lead]),
                lead + 1,
                block_index,
            )

    out = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "pred": values[0],
                "true": values[1],
                "lead": values[2],
                "block_index": values[3],
            }
            for date, values in sorted(records.items())
        ]
    )
    if len(out) != last - first + 1:
        raise RuntimeError(f"stitched months {len(out)} != expected {last - first + 1}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(args.out_csv)


if __name__ == "__main__":
    main()
