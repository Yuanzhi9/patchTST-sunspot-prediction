"""Dedicated entry point for BASE-2026-11-M sequential blocked CV5.

New file only. The legacy run_longExp.py is intentionally left untouched.
"""

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

from exp.exp_cv5 import Exp_CV5


def parse_args():
    p = argparse.ArgumentParser(description="PatchTST sequential blocked CV5")
    p.add_argument("--is_training", type=int, default=1)
    p.add_argument("--model_id", type=str, default="sunspot")
    p.add_argument("--model", type=str, default="PatchTST")
    p.add_argument("--root_path", type=str, default="./PatchTST_supervised/dataset/")
    p.add_argument("--data_path", type=str, default="sunspot_with_cycle_monthly_2026-10.csv")
    p.add_argument("--features", type=str, default="MS")
    p.add_argument("--target", type=str, default="ssn")
    p.add_argument("--freq", type=str, default="m")
    p.add_argument("--target_transform", type=str, default="")
    p.add_argument("--cv_file", type=str, default="configs/cv5_folds_2026-10.json")
    p.add_argument("--cv_fold", type=int, required=True, choices=[1, 2, 3, 4, 5])

    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--enc_in", type=int, default=3)
    p.add_argument("--dec_in", type=int, default=5)
    p.add_argument("--c_out", type=int, default=1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--distil", action="store_false", default=True)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--fc_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--padding_patch", type=str, default="end")
    p.add_argument("--revin", type=int, default=1)
    p.add_argument("--affine", type=int, default=0)
    p.add_argument("--subtract_last", type=int, default=0)
    p.add_argument("--decomposition", type=int, default=0)
    p.add_argument("--kernel_size", type=int, default=25)
    p.add_argument("--individual", type=int, default=0)
    p.add_argument("--embed_type", type=int, default=0)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--output_attention", action="store_true", default=False)
    p.add_argument("--do_predict", action="store_true", default=False)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--loss", type=str, default="mse")
    p.add_argument("--wmse_alpha", type=float, default=1.0)
    p.add_argument("--lradj", type=str, default="type3")
    p.add_argument("--pct_start", type=float, default=0.3)
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--random_seed", type=int, default=2021)
    p.add_argument("--des", type=str, required=True)
    p.add_argument("--checkpoints", type=str, default="./checkpoints/")
    p.add_argument("--results_root", type=str, default="./results")
    p.add_argument("--config_root", type=str, default="./configs")

    p.add_argument("--use_gpu", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use_multi_gpu", action="store_true", default=False)
    p.add_argument("--devices", type=str, default="0")
    p.add_argument("--test_flop", action="store_true", default=False)
    return p.parse_args()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_setting(args) -> str:
    return (
        f"{args.model_id}_{args.model}_cv5_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
        f"_df{args.d_ff}_p{args.patch_len}_s{args.stride}"
        f"_{args.des}_F{args.cv_fold}_seed{args.random_seed}"
    )


def save_config(args, setting: str) -> str:
    os.makedirs(args.config_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.config_root, f"{args.des}_F{args.cv_fold}_{stamp}.json")
    data_path = os.path.join(args.root_path, args.data_path)
    payload = {
        "experiment_id": args.des,
        "setting": setting,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "data_file": data_path,
        "data_sha256": file_sha256(data_path),
        "cv_file": args.cv_file,
        "cv_sha256": file_sha256(args.cv_file),
        "params": vars(args),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.features != "MS":
        raise ValueError("BASE-2026-11-M currently supports features=MS only")
    if not Path(args.cv_file).exists():
        raise FileNotFoundError(args.cv_file)
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
        device_ids = [int(v) for v in args.devices.split(",")]
        args.device_ids = device_ids
        args.gpu = device_ids[0]

    setting = make_setting(args)
    config_path = save_config(args, setting)
    print("Args in CV5 experiment:")
    print(args)
    print(f"config: {config_path}")
    print(f"setting: {setting}")

    exp = Exp_CV5(args)
    if args.is_training:
        exp.train(setting)
        print(f">>>>>>>testing outer fold {args.cv_fold}: {setting}<<<<<<<")
        exp.test(setting, test=1)
    else:
        print(f">>>>>>>testing outer fold {args.cv_fold}: {setting}<<<<<<<")
        exp.test(setting, test=1)


if __name__ == "__main__":
    main()
