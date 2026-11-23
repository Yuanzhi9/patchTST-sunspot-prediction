"""Train and recursively evaluate an oracle-conditioned PatchTST model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim import lr_scheduler

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT / "PatchTST_supervised"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_provider.cv5_cond_data import build_conditional_cv5_data
from eval_cycle_extrema import extrema, metric_block
from models.PatchTSTConditioned import Model
from utils.tools import EarlyStopping, adjust_learning_rate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", required=True)
    p.add_argument("--cond_mode", required=True, choices=["none", "amp", "time", "both"])
    p.add_argument("--cv_file", required=True)
    p.add_argument("--cv_fold", type=int, default=1)
    p.add_argument("--root_path", default="./PatchTST_supervised/dataset/")
    p.add_argument("--data_path", default="sunspot_with_cycle_monthly_2026-10.csv")
    p.add_argument("--boundaries_file", default="analysis/cycle_context/cycle_boundaries_2026-10.csv")
    p.add_argument("--features", default="MS")
    p.add_argument("--target", default="ssn")
    p.add_argument("--freq", default="m")
    p.add_argument("--target_transform", default="")
    p.add_argument("--seq_len", type=int, default=132)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--enc_in", type=int, default=3)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--fc_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--patch_len", type=int, default=36)
    p.add_argument("--stride", type=int, default=18)
    p.add_argument("--padding_patch", default="end")
    p.add_argument("--revin", type=int, default=1)
    p.add_argument("--affine", type=int, default=0)
    p.add_argument("--subtract_last", type=int, default=0)
    p.add_argument("--decomposition", type=int, default=0)
    p.add_argument("--kernel_size", type=int, default=25)
    p.add_argument("--individual", type=int, default=0)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--cond_dim", type=int, default=2)
    p.add_argument("--embed", default="timeF")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_window_limit", type=int, default=0)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lradj", default="type3")
    p.add_argument("--pct_start", type=float, default=0.3)
    p.add_argument("--random_seed", type=int, default=2021)
    p.add_argument("--des", required=True)
    p.add_argument("--checkpoints", default="./checkpoints/")
    p.add_argument("--config_root", default="./configs")
    p.add_argument("--analysis_root", default="./analysis/e1b_oracle_conditioning")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    eval_frame = frame[["date", "true"]].copy()
    eval_frame["pred"] = pred
    ext = extrema(eval_frame, 13, "peak_then_trough")
    return {
        "curve": metric_block(pred, frame["true"].to_numpy(dtype=float)),
        "extrema": ext,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E1b requires CUDA")

    bundle = build_conditional_cv5_data(args)
    train_data, train_loader = bundle["train"]
    val_data, val_loader = bundle["val"]
    test_data, _ = bundle["test"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.analysis_root) / "screen" / args.cycle / args.cond_mode / f"seed{args.random_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(args.config_root)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.des}_{stamp}.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "params": vars(args),
                "data_sha256": sha256(Path(args.root_path) / args.data_path),
                "cv_sha256": sha256(Path(args.cv_file)),
                "boundaries_sha256": sha256(Path(args.boundaries_file)),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    model = Model(args).float().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate,
    )
    early = EarlyStopping(patience=args.patience, verbose=True)
    checkpoint_dir = Path(args.checkpoints) / args.des
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "train_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "seconds", "best_val"])

    print(
        f"start {args.des} mode={args.cond_mode} seed={args.random_seed} "
        f"train={len(train_data)} val={len(val_data)} test={len(test_data)}"
    )
    for epoch in range(args.train_epochs):
        started = time.time()
        model.train()
        losses = []
        for batch_x, batch_y, cond in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cond = cond.float().to(device)
            optimizer.zero_grad()
            output = model(batch_x, cond)[:, -args.pred_len :, -1:]
            target = batch_y[:, -args.pred_len :, -1:]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, cond in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cond = cond.float().to(device)
                output = model(batch_x, cond)[:, -args.pred_len :, -1:]
                target = batch_y[:, -args.pred_len :, -1:]
                val_losses.append(float(criterion(output, target).item()))
        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        early(val_loss, model, optimizer, scheduler, epoch, args, str(checkpoint_dir))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch + 1, train_loss, val_loss, time.time() - started, early.val_loss_min]
            )
        print(
            f"Epoch {epoch + 1}/{args.train_epochs} train={train_loss:.7f} "
            f"val={val_loss:.7f} best={early.val_loss_min:.7f}"
        )
        if early.early_stop:
            break
        adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    checkpoint = torch.load(
        checkpoint_dir / "full_checkpoint.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fold = test_data.fold
    start = int(fold["outer_val"]["start"])
    end = int(fold["outer_val"]["end"])
    horizon = end - start + 1
    data = test_data.data
    mean = test_data.scaler.mean_
    scale = test_data.scaler.scale_
    window = data[start - args.seq_len : start].copy()
    produced = 0
    block_index = 0
    rows = []
    while produced < horizon:
        target_start = start + produced
        cond = test_data.condition_for_target_start(target_start)
        with torch.no_grad():
            output = model(
                torch.tensor(window[None], dtype=torch.float32, device=device),
                torch.tensor(cond[None], dtype=torch.float32, device=device),
            )
        z_pred = output[0, : args.pred_len, -1].cpu().numpy()
        take = min(args.pred_len, horizon - produced)
        for lead in range(take):
            target_index = target_start + lead
            pred_physical = float(z_pred[lead] * scale[-1] + mean[-1])
            rows.append(
                {
                    "date": test_data.dates.iloc[target_index].strftime("%Y-%m-%d"),
                    "pred": pred_physical,
                    "true": float(test_data.df["ssn"].iloc[target_index]),
                    "lead": lead + 1,
                    "block_index": block_index,
                    "peak_offset_months": float(cond[1]),
                }
            )
            new_row = data[target_index].copy()
            new_row[-1] = float(z_pred[lead])
            window = np.vstack([window[1:], new_row])
        produced += take
        block_index += 1

    curve = pd.DataFrame(rows)
    clip_pred = np.clip(curve["pred"].to_numpy(), 0.0, None)
    raw_assessment = assess(curve, curve["pred"].to_numpy())
    clip_assessment = assess(curve, clip_pred)
    curve["pred_clip0"] = clip_pred
    curve_path = run_dir / "recursive_curve.csv"
    curve.to_csv(curve_path, index=False)

    base_cond = test_data.condition_for_target_start(start)
    probe_window = data[start - args.seq_len : start].copy()

    def probe_peak(condition: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            output = model(
                torch.tensor(probe_window[None], dtype=torch.float32, device=device),
                torch.tensor(condition[None], dtype=torch.float32, device=device),
            )
        z = output[0, : args.pred_len, -1].cpu().numpy()
        return z * scale[-1] + mean[-1]

    amp_lo = base_cond.copy()
    amp_hi = base_cond.copy()
    time_lo = base_cond.copy()
    time_hi = base_cond.copy()
    if args.cond_mode in {"amp", "both"}:
        amp_lo[0] -= 2.0
        amp_hi[0] += 2.0
    if args.cond_mode in {"time", "both"}:
        time_lo[1] -= 2.0
        time_hi[1] += 2.0
    probe_amp_lo = probe_peak(amp_lo)
    probe_amp_hi = probe_peak(amp_hi)
    probe_time_lo = probe_peak(time_lo)
    probe_time_hi = probe_peak(time_hi)
    amp_peak_delta = float(probe_amp_hi.max() - probe_amp_lo.max())
    time_peak_index_delta = int(
        np.argmax(probe_time_hi) - np.argmax(probe_time_lo)
    )
    sensitivity = {
        "amp_peak_delta_for_two_std": amp_peak_delta,
        "time_peak_index_delta_for_two_std": time_peak_index_delta,
        "amp_sensitive": bool(amp_peak_delta > 1e-6),
        "time_sensitive": bool(time_peak_index_delta > 0),
        "pass": bool(
            (args.cond_mode not in {"amp", "both"} or amp_peak_delta > 1e-6)
            and (
                args.cond_mode not in {"time", "both"}
                or time_peak_index_delta > 0
            )
        ),
    }

    result = {
        "cycle": args.cycle,
        "cond_mode": args.cond_mode,
        "seed": args.random_seed,
        "setting": args.des,
        "raw": raw_assessment,
        "clip0": clip_assessment,
        "checkpoint": str(checkpoint_dir / "full_checkpoint.pth"),
        "config": str(config_path),
        "curve": str(curve_path),
        "cond_stats": {
            "mean": test_data.cond_mean.tolist(),
            "scale": test_data.cond_scale.tolist(),
        },
        "n_train_windows": len(train_data),
        "n_val_windows": len(val_data),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "condition_sensitivity": sensitivity,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["clip0"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
