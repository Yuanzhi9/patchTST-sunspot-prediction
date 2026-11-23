"""Train PatchTST with two-block teacher forcing or scheduled sampling."""

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

from data_provider.cv5_two_block_data import build_two_block_data
from eval_cycle_extrema import extrema, metric_block
from models.PatchTST import Model
from utils.tools import EarlyStopping, adjust_learning_rate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", required=True)
    p.add_argument("--mode", required=True, choices=["teacher", "scheduled"])
    p.add_argument("--cv_file", required=True)
    p.add_argument("--cv_fold", type=int, default=1)
    p.add_argument("--root_path", default="./PatchTST_supervised/dataset/")
    p.add_argument("--data_path", default="sunspot_with_cycle_monthly_2026-10.csv")
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
    p.add_argument("--embed", default="timeF")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_window_limit", type=int, default=0)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lradj", default="type3")
    p.add_argument("--pct_start", type=float, default=0.3)
    p.add_argument("--scheduled_warmup_epochs", type=int, default=10)
    p.add_argument("--min_checkpoint_epoch", type=int, default=25)
    p.add_argument("--scheduled_max_prob", type=float, default=0.3)
    p.add_argument("--second_block_weight", type=float, default=0.5)
    p.add_argument("--random_seed", type=int, default=2021)
    p.add_argument("--des", required=True)
    p.add_argument("--checkpoints", default="./checkpoints/")
    p.add_argument("--config_root", default="./configs")
    p.add_argument("--analysis_root", default="./analysis/t1_scheduled_sampling")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduled_probability(args, epoch: int) -> float:
    if args.mode == "teacher" or epoch < args.scheduled_warmup_epochs:
        return 0.0
    remaining = max(1, args.train_epochs - args.scheduled_warmup_epochs)
    fraction = min(1.0, (epoch - args.scheduled_warmup_epochs + 1) / remaining)
    return args.scheduled_max_prob * fraction


def forward_two_blocks(model, x, block1, block2, probability):
    pred1 = model(x)[:, -block1.shape[1] :, -1:]
    future = block1.clone()
    if probability > 0:
        use_prediction = (
            torch.rand(x.shape[0], 1, 1, device=x.device) < probability
        ).float()
        future[:, :, -1:] = (
            use_prediction * pred1 + (1.0 - use_prediction) * block1[:, :, -1:]
        )
    x_next = torch.cat([x[:, block1.shape[1] :, :], future], dim=1)
    pred2 = model(x_next)[:, -block2.shape[1] :, -1:]
    return pred1, pred2


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
        raise RuntimeError("T1 scheduled sampling requires CUDA")

    bundle = build_two_block_data(args)
    train_data, train_loader = bundle["train"]
    val_data, val_loader = bundle["val"]
    test_data, _ = bundle["test"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.analysis_root)
        / "screen"
        / args.cycle
        / args.mode
        / f"seed{args.random_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_root) / f"{args.des}_{timestamp}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "params": vars(args),
                "data_sha256": sha256(Path(args.root_path) / args.data_path),
                "cv_sha256": sha256(Path(args.cv_file)),
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
        optimizer,
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
        writer.writerow(["epoch", "train_loss", "rollout_val_loss", "seconds", "best_val", "scheduled_probability"])

    print(
        f"start {args.des} mode={args.mode} seed={args.random_seed} "
        f"train={len(train_data)} val={len(val_data)} test={len(test_data)}"
    )
    for epoch in range(args.train_epochs):
        started = time.time()
        probability = scheduled_probability(args, epoch)
        model.train()
        losses = []
        for x, block1, block2 in train_loader:
            x = x.float().to(device)
            block1 = block1.float().to(device)
            block2 = block2.float().to(device)
            optimizer.zero_grad()
            pred1, pred2 = forward_two_blocks(
                model, x, block1, block2, probability
            )
            loss1 = criterion(pred1, block1[:, :, -1:])
            loss2 = criterion(pred2, block2[:, :, -1:])
            loss = loss1 + args.second_block_weight * loss2
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, block1, block2 in val_loader:
                x = x.float().to(device)
                block1 = block1.float().to(device)
                block2 = block2.float().to(device)
                pred1, pred2 = forward_two_blocks(
                    model, x, block1, block2, 1.0
                )
                val_losses.append(
                    float(
                        criterion(pred1, block1[:, :, -1:])
                        + args.second_block_weight
                        * criterion(pred2, block2[:, :, -1:])
                    )
                )
        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        if epoch + 1 >= args.min_checkpoint_epoch:
            early(val_loss, model, optimizer, scheduler, epoch, args, str(checkpoint_dir))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    time.time() - started,
                    early.val_loss_min,
                    probability,
                ]
            )
        print(
            f"Epoch {epoch + 1}/{args.train_epochs} p={probability:.3f} "
            f"train={train_loss:.7f} rollout_val={val_loss:.7f} "
            f"best={early.val_loss_min:.7f}"
        )
        if early.early_stop:
            break
        adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    if not (checkpoint_dir / "full_checkpoint.pth").exists():
        raise RuntimeError(
            f"no eligible checkpoint at/after epoch {args.min_checkpoint_epoch}"
        )
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
        with torch.no_grad():
            output = model(torch.tensor(window[None], dtype=torch.float32, device=device))
        z_pred = output[0, : args.pred_len, -1].cpu().numpy()
        take = min(args.pred_len, horizon - produced)
        for lead in range(take):
            target_index = start + produced + lead
            pred_physical = float(z_pred[lead] * scale[-1] + mean[-1])
            rows.append(
                {
                    "date": test_data.dates.iloc[target_index].strftime("%Y-%m-%d"),
                    "pred": pred_physical,
                    "true": float(test_data.df["ssn"].iloc[target_index]),
                    "lead": lead + 1,
                    "block_index": block_index,
                }
            )
            new_row = data[target_index].copy()
            new_row[-1] = float(z_pred[lead])
            window = np.vstack([window[1:], new_row])
        produced += take
        block_index += 1

    curve = pd.DataFrame(rows)
    clip_pred = np.clip(curve["pred"].to_numpy(), 0.0, None)
    curve["pred_clip0"] = clip_pred
    curve_path = run_dir / "recursive_curve.csv"
    curve.to_csv(curve_path, index=False)
    raw_result = assess(curve, curve["pred"].to_numpy())
    clip_result = assess(curve, clip_pred)

    metrics = {
        "cycle": args.cycle,
        "mode": args.mode,
        "seed": args.random_seed,
        "setting": args.des,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "test_windows": len(test_data),
        "raw": raw_result,
        "clip0": clip_result,
        "scheduled_max_probability": args.scheduled_max_prob,
        "scheduled_warmup_epochs": args.scheduled_warmup_epochs,
        "min_checkpoint_epoch": args.min_checkpoint_epoch,
        "selected_checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "second_block_weight": args.second_block_weight,
        "checkpoint": str(checkpoint_dir / "full_checkpoint.pth"),
        "config": str(config_path),
        "curve": str(curve_path),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["clip0"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
