"""
save_config.py — 训练前冻结配置快照

用法:
  python save_config.py EXP-16 --reason "复现Baseline B在1749+数据上"

从 run_sunspot_fixed.build_parser() 继承参数定义（单一数据源，不会失同步）。
额外参数（scaler/num_train/reason）来自本脚本。

输出 configs/EXP-16_YYYY-MM-DD.json
"""

import json
import os
import subprocess
from datetime import date

import sys
sys.path.append('.')
from run_sunspot_fixed import build_parser


def main():
    parser = build_parser()

    parser.add_argument("experiment_id", help="实验 ID，如 EXP-16")
    parser.add_argument("--reason", default="", help="实验目的（一句话）")
    parser.add_argument("--scaler", default="standard",
                        choices=["standard", "minmax"],
                        help="归一化类型")
    parser.add_argument("--scaler_params", type=json.loads, default=None,
                        help="scaler 额外参数 JSON，如 '{\"feature_range\":[0,1]}'")
    parser.add_argument("--num_train", type=int, default=3119,
                        help="训练集行数（scaler fit on [0:num_train]）")

    args = parser.parse_args()

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
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
    }

    os.makedirs("configs", exist_ok=True)
    path = f"configs/{args.experiment_id}_{date.today().isoformat()}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Config saved: {path}")


if __name__ == "__main__":
    main()
