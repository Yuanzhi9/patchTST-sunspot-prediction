"""CLI entry point for PR24 curve plus cycle-event auxiliary heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from exp.exp_cv5_multitask import Exp_CV5_MultiTask
from models.PatchTSTMultiTask import MT_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="PatchTST CV5 multitask extrema")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="sunspot")
    parser.add_argument("--model", type=str, default="PatchTSTMultiTask")
    parser.add_argument("--root_path", type=str, default="./PatchTST_supervised/dataset/")
    parser.add_argument(
        "--data_path", type=str, default="sunspot_with_cycle_monthly_2026-10.csv"
    )
    parser.add_argument(
        "--event_labels",
        type=str,
        default="analysis/pr24_multitask_extrema/a0_labels/event_labels_all_windows.csv",
    )
    parser.add_argument("--features", type=str, default="MS")
    parser.add_argument("--target", type=str, default="ssn")
    parser.add_argument("--freq", type=str, default="m")
    parser.add_argument("--target_transform", type=str, default="")
    parser.add_argument("--cv_file", type=str, default="configs/cv5_folds_2026-10.json")
    parser.add_argument("--cv_fold", type=int, required=True, choices=[1, 2, 3, 4, 5])

    parser.add_argument("--seq_len", type=int, default=132)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--enc_in", type=int, default=3)
    parser.add_argument("--dec_in", type=int, default=5)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--patch_len", type=int, default=36)
    parser.add_argument("--stride", type=int, default=18)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--subtract_last", type=int, default=0)
    parser.add_argument("--decomposition", type=int, default=0)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--embed_type", type=int, default=0)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)
    parser.add_argument("--do_predict", action="store_true", default=False)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_window_limit", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--wmse_alpha", type=float, default=1.0)
    parser.add_argument("--lradj", type=str, default="type3")
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--des", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--results_root", type=str, default="./results")
    parser.add_argument("--config_root", type=str, default="./configs")
    parser.add_argument("--mt_mode", type=str, default="none", choices=sorted(MT_MODES))
    parser.add_argument("--mt_aux_weight", type=float, default=0.1)
    parser.add_argument(
        "--mt_weighting",
        type=str,
        default="fixed",
        choices=["fixed", "uncertainty"],
    )

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_setting(args) -> str:
    return (
        f"{args.model_id}_{args.model}_cv5_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
        f"_df{args.d_ff}_p{args.patch_len}_s{args.stride}"
        f"_mt{args.mt_mode}_aw{args.mt_aux_weight:g}_{args.mt_weighting}"
        f"_{args.des}_F{args.cv_fold}_seed{args.random_seed}"
    )


def save_config(args, setting: str) -> str:
    os.makedirs(args.config_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        args.config_root,
        f"{args.des}_mt{args.mt_mode}_F{args.cv_fold}_seed{args.random_seed}_{stamp}.json",
    )
    data_path = os.path.join(args.root_path, args.data_path)
    payload = {
        "experiment_id": args.des,
        "setting": setting,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "data_file": data_path,
        "data_sha256": file_sha256(data_path),
        "event_labels": args.event_labels,
        "event_labels_sha256": file_sha256(args.event_labels),
        "cv_file": args.cv_file,
        "cv_sha256": file_sha256(args.cv_file),
        "params": vars(args),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main():
    args = parse_args()
    if args.features != "MS":
        raise ValueError("PR24 multitask currently supports features=MS only")
    for path in [args.cv_file, args.event_labels]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("--use_gpu 1 but CUDA is unavailable")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(value) for value in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    setting = make_setting(args)
    config_path = save_config(args, setting)
    print("Args in PR24 multitask experiment:")
    print(args)
    print(f"config: {config_path}")
    print(f"setting: {setting}")

    exp = Exp_CV5_MultiTask(args)
    if args.is_training:
        exp.train(setting)
        print(f">>>>>>>testing outer fold {args.cv_fold}: {setting}<<<<<<<")
        exp.test(setting, test=1)
    else:
        print(f">>>>>>>testing outer fold {args.cv_fold}: {setting}<<<<<<<")
        exp.test(setting, test=0)


if __name__ == "__main__":
    main()
