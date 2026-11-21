"""Generate a no-future-truth PR24 rolling-block curve from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PatchTST_supervised"))

from exp.exp_main import _load_checkpoint
from models.PatchTST import Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--data_csv", default="PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv")
    parser.add_argument("--horizon", type=int, default=132)
    parser.add_argument("--block_size", type=int, default=24)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / args.data_csv, parse_dates=["date"])
    feature_cols = ["month_sin", "month_cos", "ssn"]
    raw = df[feature_cols].to_numpy(dtype=np.float64)
    mean = np.asarray(meta["scaler_mean"], dtype=np.float64)
    scale = np.asarray(meta["scaler_scale"], dtype=np.float64)
    data_z = (raw - mean) / scale

    starts = np.asarray(meta["window_target_starts"], dtype=int)
    start = int(starts.min())
    seq_len = int(meta["seq_len"])
    pred_len = int(meta["pred_len"])
    if args.block_size <= 0 or args.block_size > pred_len:
        raise ValueError("block_size must be in [1, pred_len]")
    if start < seq_len:
        raise RuntimeError("insufficient history at forecast origin")
    if start + args.horizon > len(df):
        raise RuntimeError("requested horizon exceeds available historical truth")

    checkpoint_path = ROOT / "checkpoints" / meta["setting"] / "full_checkpoint.pth"
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint.get("args") or checkpoint.get("config")
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model = Model(checkpoint_args).float()
    model.load_state_dict(state)
    model.eval()

    window = data_z[start - seq_len : start].copy()
    rows = []
    produced = 0
    block_index = 0
    while produced < args.horizon:
        with torch.no_grad():
            output = model(torch.FloatTensor(window).unsqueeze(0))
        z_pred = output[0, :pred_len, -1].cpu().numpy()
        take = min(args.block_size, args.horizon - produced)
        for lead in range(take):
            target_index = start + produced + lead
            pred_physical = float(z_pred[lead] * scale[-1] + mean[-1])
            rows.append(
                {
                    "date": df["date"].iloc[target_index].strftime("%Y-%m-%d"),
                    "pred": pred_physical,
                    "true": float(df["ssn"].iloc[target_index]),
                    "lead": lead + 1,
                    "block_index": block_index,
                }
            )
            new_row = data_z[target_index].copy()
            new_row[-1] = float(z_pred[lead])
            window = np.vstack([window[1:], new_row])
        produced += take
        block_index += 1

    out_csv = Path(args.out_csv) if args.out_csv else result_dir / "p24_recursive_curve.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
