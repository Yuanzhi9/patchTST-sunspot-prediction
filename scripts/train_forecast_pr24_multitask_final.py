"""All-history retrain and real forecast for the best PR24 multitask structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PatchTST_supervised"))
from models.PatchTSTMultiTask import Model  # noqa: E402


DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
SMOOTH_DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_smoothed13_2026-04.csv"
BOUNDARIES = ROOT / "analysis/cycle_context/cycle_boundaries_2026-10.csv"
FEATURES = ("month_sin", "month_cos", "ssn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model_args() -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=3,
        seq_len=132,
        pred_len=24,
        e_layers=2,
        n_heads=8,
        d_model=128,
        d_ff=2048,
        dropout=0.2,
        fc_dropout=0.2,
        head_dropout=0.0,
        individual=0,
        patch_len=36,
        stride=18,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        activation="gelu",
        mt_mode="all",
        mt_weighting="fixed",
    )


def month_delta(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return (end.year - start.year) * 12 + (end.month - start.month)


def event_targets(
    dates: pd.Series,
    target_starts: list[int],
    scaler: StandardScaler,
    curve_values: pd.Series,
    track: str,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = pd.read_csv(
        BOUNDARIES, parse_dates=["min_date", "peak_date", "next_min_date"]
    )
    peaks = [
        {
            "date": row.peak_date,
            "raw": float(row.peak_raw),
            "smooth": float(curve_values.loc[row.peak_date]),
        }
        for row in boundaries.itertuples(index=False)
    ]
    troughs = sorted(
        set(boundaries["min_date"].dropna()) | set(boundaries["next_min_date"].dropna())
    )
    targets = np.zeros((len(target_starts), 4), dtype=np.float32)
    masks = np.zeros_like(targets)
    amp_mean = float(scaler.mean_[-1])
    amp_scale = float(scaler.scale_[-1])

    for row_index, start in enumerate(target_starts):
        origin = pd.Timestamp(dates.iloc[start])
        future_peaks = [item for item in peaks if item["date"] > origin]
        future_troughs = [date for date in troughs if date > origin]
        current = boundaries.loc[
            (boundaries["min_date"] <= origin)
            & (
                boundaries["next_min_date"].isna()
                | (origin < boundaries["next_min_date"])
            )
        ]

        if future_peaks:
            peak = min(future_peaks, key=lambda item: item["date"])
            peak_amplitude = peak["raw"] if track == "raw" else peak["smooth"]
            targets[row_index, 0] = (peak_amplitude - amp_mean) / amp_scale
            targets[row_index, 1] = month_delta(origin, peak["date"]) / 12.0
            masks[row_index, 0:2] = 1.0
        if future_troughs:
            trough = min(future_troughs)
            targets[row_index, 2] = month_delta(origin, trough) / 12.0
            masks[row_index, 2] = 1.0
        if len(current) == 1 and pd.notna(current.iloc[0]["next_min_date"]):
            cycle_length = month_delta(
                current.iloc[0]["min_date"], current.iloc[0]["next_min_date"]
            )
            targets[row_index, 3] = cycle_length / 12.0
            masks[row_index, 3] = 1.0
    return targets, masks


def build_dataset(
    data_z: np.ndarray,
    dates: pd.Series,
    scaler: StandardScaler,
    curve_values: pd.Series,
    track: str,
    seq_len: int,
    pred_len: int,
) -> TensorDataset:
    starts = list(range(seq_len, len(data_z) - pred_len + 1))
    x = np.stack([data_z[start - seq_len : start] for start in starts])
    y = np.stack([data_z[start : start + pred_len] for start in starts])
    aux, masks = event_targets(dates, starts, scaler, curve_values, track)
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(aux, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
    )


def aux_loss(aux_pred: torch.Tensor, aux_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for column in (0, 1, 2):
        valid = mask[:, column].bool()
        if valid.any():
            losses.append(
                F.smooth_l1_loss(
                    aux_pred[valid, column],
                    aux_true[valid, column],
                    beta=0.5,
                )
            )
    return torch.stack(losses).mean() if losses else aux_pred.sum() * 0.0


def train_one(
    seed: int,
    frame: pd.DataFrame,
    data_z: np.ndarray,
    scaler: StandardScaler,
    track: str,
    data_path: Path,
    out_dir: Path,
    checkpoint_root: Path,
    config_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    horizon: int,
    block_size: int,
) -> pd.DataFrame:
    set_seed(seed)
    model_args = make_model_args()
    setting = f"FINAL-PR24-MT-{track.upper()}_seed{seed}"
    checkpoint_dir = checkpoint_root / setting
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(
        data_z,
        frame["date"],
        scaler,
        frame.set_index("date")["ssn"],
        track,
        model_args.seq_len,
        model_args.pred_len,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    model = Model(model_args).float().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        steps_per_epoch=len(loader),
        pct_start=0.3,
        epochs=epochs,
        max_lr=learning_rate,
    )
    curve_criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        totals, curves, auxiliaries = [], [], []
        for batch_x, batch_y, event_y, event_mask in loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            event_y = event_y.cuda()
            event_mask = event_mask.cuda()
            optimizer.zero_grad(set_to_none=True)
            curve, aux = model(batch_x)
            curve = curve[:, -model_args.pred_len :, -1]
            target = batch_y[:, -model_args.pred_len :, -1]
            curve_loss = curve_criterion(curve, target)
            aux_value = aux_loss(aux, event_y, event_mask)
            total = curve_loss + 0.1 * aux_value
            total.backward()
            optimizer.step()
            scheduler.step()
            totals.append(float(total.detach().cpu()))
            curves.append(float(curve_loss.detach().cpu()))
            auxiliaries.append(float(aux_value.detach().cpu()))
        record = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(totals)),
            "curve_loss": float(np.mean(curves)),
            "aux_loss": float(np.mean(auxiliaries)),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"curve={record['curve_loss']:.8f} aux={record['aux_loss']:.8f} "
            f"seconds={record['seconds']:.3f}",
            flush=True,
        )

    pd.DataFrame(history).to_csv(checkpoint_dir / "train_history.csv", index=False)
    checkpoint = {
        "seed": seed,
        "setting": setting,
        "model_state_dict": model.state_dict(),
        "model_args": vars(model_args),
        "training_protocol": "all-history fixed-epoch retrain; no validation early stopping",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mt_aux_weight": 0.1,
        "n_windows": len(dataset),
    }
    torch.save(checkpoint, checkpoint_dir / "full_checkpoint.pth")
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": "FINAL-PR24-MT-2026-10",
        "setting": setting,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "data_csv": (
            str(data_path.relative_to(ROOT))
            if data_path.is_relative_to(ROOT)
            else str(data_path)
        ),
        "data_sha256": sha256(data_path),
        "target_track": track,
        "last_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "model_args": vars(model_args),
        "mt_aux_weight": 0.1,
        "training_protocol": checkpoint["training_protocol"],
        "n_windows": len(dataset),
        "horizon": horizon,
        "block_size": block_size,
    }
    (config_root / f"{setting}_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    origin_date = frame["date"].iloc[-1]
    mean = float(scaler.mean_[-1])
    scale = float(scaler.scale_[-1])
    window = data_z[-model_args.seq_len :].copy()
    rows = []
    block_rows = []
    produced = 0
    block_index = 0
    while produced < horizon:
        with torch.no_grad():
            curve, aux = model(torch.tensor(window, dtype=torch.float32).unsqueeze(0).cuda())
        prediction = curve[0, :, -1].cpu().numpy()
        aux_np = aux[0].cpu().numpy()
        take = min(block_size, model_args.pred_len, horizon - produced)
        block_rows.append(
            {
                "seed": seed,
                "block_index": block_index,
                "input_end_date": (origin_date + pd.DateOffset(months=produced)).strftime("%Y-%m"),
                "next_peak_raw": float(aux_np[0] * scale + mean),
                "months_to_next_peak": float(aux_np[1] * 12.0),
                "months_to_next_trough": float(aux_np[2] * 12.0),
            }
        )
        for lead in range(take):
            date = origin_date + pd.DateOffset(months=produced + 1)
            physical_raw = float(prediction[lead] * scale + mean)
            physical_clip = max(physical_raw, 0.0)
            rows.append(
                {
                    "date": date,
                    "mode": "clip0",
                    "pred_raw": physical_raw,
                    "pred": physical_clip,
                    "horizon": produced + 1,
                    "block_index": block_index,
                    "lead_in_block": lead + 1,
                    "seed": seed,
                }
            )
            phase = 2.0 * np.pi * date.month / 12.0
            z_value = (physical_clip - mean) / scale
            window = np.vstack(
                [
                    window[1:],
                    np.asarray([np.sin(phase), np.cos(phase), z_value]),
                ]
            )
            produced += 1
        block_index += 1
    forecast = pd.DataFrame(rows)
    forecast.to_csv(out_dir / f"forecast_seed{seed}.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out_dir / f"aux_blocks_seed{seed}.csv", index=False)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--track", choices=["raw", "smooth13"], required=True)
    parser.add_argument("--data_csv", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--checkpoint_root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--config_root", type=Path, default=ROOT / "configs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--block_size", type=int, default=24)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data_path = args.data_csv or (DATA if args.track == "raw" else SMOOTH_DATA)
    out_dir = args.out_dir or (
        ROOT / "analysis/pr24_multitask_future_forecast" / f"{args.track}_target"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
    expected_end = "2026-10" if args.track == "raw" else "2026-04"
    if frame["date"].iloc[-1].strftime("%Y-%m") != expected_end:
        raise ValueError(f"{args.track} data must end at {expected_end}")
    if frame["ssn"].isna().any():
        raise ValueError("target series contains missing values")
    scaler = StandardScaler()
    data_z = scaler.fit_transform(frame[list(FEATURES)].to_numpy(dtype=float))
    train_one(
        seed=args.seed,
        frame=frame,
        data_z=data_z,
        scaler=scaler,
        track=args.track,
        data_path=data_path,
        out_dir=out_dir,
        checkpoint_root=args.checkpoint_root,
        config_root=args.config_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        horizon=args.horizon,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
