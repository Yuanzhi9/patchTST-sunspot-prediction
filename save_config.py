"""
save_config.py — 训练前冻结配置快照

用法:
  python save_config.py EXP-16 --reason "复现Baseline B在1749+数据上"

输出 configs/EXP-16_YYYY-MM-DD.json

⚠️ 本文件的参数默认值需与 run_sunspot_fixed.py 保持同步。
   改 rsf.py 的参数时，同步更新本文件同名字段。
"""

import argparse
import json
import os
import subprocess
from datetime import date


def main():
    p = argparse.ArgumentParser(description="保存实验配置快照")
    p.add_argument("experiment_id", help="实验 ID，如 EXP-16")
    p.add_argument("--reason", default="", help="实验目的（一句话）")
    p.add_argument("--scaler", default="standard",
                   choices=["standard", "minmax"],
                   help="归一化类型")
    p.add_argument("--scaler_params", type=json.loads, default=None,
                   help="scaler 额外参数，JSON 格式，如 '{\"feature_range\":[0,1]}'")
    p.add_argument("--num_train", type=int, default=3119,
                   help="训练集行数（scaler fit on [0:num_train]）")

    # --- 以下参数与 run_sunspot_fixed.py 保持同步 ---
    p.add_argument("--model_id", type=str, default="sunspot")
    p.add_argument("--model", type=str, default="PatchTST")
    p.add_argument("--data", type=str, default="custom")
    p.add_argument("--root_path", type=str, default="./PatchTST_supervised/dataset/")
    p.add_argument("--data_path", type=str, default="sunspot_with_cycle.csv")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--target", type=str, default="ssn")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--enc_in", type=int, default=3)
    p.add_argument("--dec_in", type=int, default=3)
    p.add_argument("--c_out", type=int, default=1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--patch_len", type=int, default=16,
                   help="⚠️ 必填——目录名不含此字段，只能在此记录")
    p.add_argument("--stride", type=int, default=8,
                   help="⚠️ 同上")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--fc_dropout", type=float, default=0.05)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--revin", type=int, default=1)
    p.add_argument("--affine", type=int, default=0)
    p.add_argument("--subtract_last", type=int, default=0)
    p.add_argument("--individual", type=int, default=0)
    p.add_argument("--embed_type", type=int, default=0)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--loss", type=str, default="mse")
    p.add_argument("--lr", type=float, dest="learning_rate", default=0.0001)
    p.add_argument("--lradj", type=str, default="type3")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--random_seed", type=int, default=2021)
    p.add_argument("--num_workers", type=int, default=0)

    args = p.parse_args()

    git_hash = ""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()[:8]
    except Exception:
        git_hash = "unknown"

    config = {
        "experiment_id": args.experiment_id,
        "date": date.today().isoformat(),
        "reason": args.reason,
        "git_commit": git_hash,
        "scaler": args.scaler,
        "scaler_params": args.scaler_params or {},
        "num_train": args.num_train,
        "params": {
            "model_id": args.model_id,
            "model": args.model,
            "data": args.data,
            "root_path": args.root_path,
            "data_path": args.data_path,
            "features": args.features,
            "target": args.target,
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "pred_len": args.pred_len,
            "enc_in": args.enc_in,
            "dec_in": args.dec_in,
            "c_out": args.c_out,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "e_layers": args.e_layers,
            "d_layers": args.d_layers,
            "d_ff": args.d_ff,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "dropout": args.dropout,
            "fc_dropout": args.fc_dropout,
            "head_dropout": args.head_dropout,
            "revin": args.revin,
            "affine": args.affine,
            "subtract_last": args.subtract_last,
            "individual": args.individual,
            "embed_type": args.embed_type,
            "embed": args.embed,
            "activation": args.activation,
            "loss": args.loss,
            "learning_rate": args.learning_rate,
            "lradj": args.lradj,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "patience": args.patience,
            "random_seed": args.random_seed,
            "num_workers": args.num_workers,
        },
    }

    os.makedirs("configs", exist_ok=True)
    path = f"configs/{args.experiment_id}_{date.today().isoformat()}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Config saved: {path}")


if __name__ == "__main__":
    main()
