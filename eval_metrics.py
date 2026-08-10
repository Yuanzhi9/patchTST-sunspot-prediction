"""
eval_metrics.py — 从 pred.npy / true.npy 反算物理口径评估指标

用法:
  有 config JSON（未来实验）:
    python eval_metrics.py --config configs/EXP-XX_2026-08-10.json

  手动指定（历史实验，没有 config JSON）:
    python eval_metrics.py results/EXP-XX/ --scaler standard --data_csv sunspot_with_cycle.csv --num_train 3119

数据流说明：
  原始 SSN → StandardScaler（本脚本反算的位置）→ z-score 空间
  → 模型（含 RevIN 内部 normalize + denormalize）→ z-score 输出
  → 写入 pred.npy / true.npy → 本脚本 inverse_transform → 物理 SSN
  RevIN 层在模型 EXP-14 的 test() 内部已还原，输出处于 scaler 空间，本脚本只 inverse scaler。

固定输出四个指标 + 误差分层 + R²。
⚠️ 滚动 MAE 本脚本不计算。需要时手动跑 H1 方式（取 step0、推进1月、循环70月），
   或待流程固化后写 roll_eval.py 统一支持 --rolling 参数。
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


def load_data(data_csv, num_train, scaler_name, scaler_kwargs=None):
    df = pd.read_csv(data_csv)
    ssn_train = df["ssn"].values[:num_train].reshape(-1, 1)
    cls = SCALER_MAP[scaler_name]
    scaler = cls(**scaler_kwargs) if scaler_kwargs else cls()
    scaler.fit(ssn_train)
    return df, scaler


def compute(pred_phy, true_phy):
    assert pred_phy.shape == true_phy.shape
    n_windows, n_steps = pred_phy.shape

    diff = pred_phy - true_phy
    mae_all = float(np.mean(np.abs(diff)))
    mae_step0 = float(np.mean(np.abs(diff[0])))  # window 0, all pred_len steps
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true_phy - np.mean(true_phy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    bins = [(0, 50), (50, 100), (100, 150), (150, 1e9)]
    strat = {}
    for lo, hi in bins:
        mask = (true_phy >= lo) & (true_phy < hi)
        strat[f"{lo}-{int(hi) if hi < 1e8 else 'plus'}"] = (
            round(float(np.mean(np.abs(diff[mask]))) if mask.any() else 0.0, 1)
        )

    idx_flat_true = int(np.argmax(true_phy))
    idx_flat_pred = int(np.argmax(pred_phy))
    peak_true = float(true_phy.flat[idx_flat_true])
    peak_pred = float(pred_phy.flat[idx_flat_pred])
    E_r = round(peak_pred - peak_true, 1)
    E_m = round(int(idx_flat_pred // n_steps) + idx_flat_pred % n_steps
                - (int(idx_flat_true // n_steps) + idx_flat_true % n_steps), 0)

    return {
        "step0_MAE": round(mae_step0, 2),
        "full_MAE": round(mae_all, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "E_r": E_r,
        "E_m": E_m,
        "error_stratification": strat,
    }


def main():
    p = argparse.ArgumentParser(description="PatchTST npy 反算评估")
    p.add_argument("results_dir", nargs="?", default=None,
                   help="results/ 目录路径（含 pred.npy / true.npy）")
    p.add_argument("--config", default=None,
                   help="config JSON 路径（自动读取所有参数）")
    p.add_argument("--scaler", default=None,
                   help="归一化类型: standard / minmax")
    p.add_argument("--data_csv", default=None,
                   help="原始 CSV 路径")
    p.add_argument("--num_train", type=int, default=None,
                   help="训练集行数（scaler fit 范围）")
    p.add_argument("--enc_in", type=int, default=3,
                   help="输入通道数（ssn 在最后一列）")
    args = p.parse_args()

    # --- 模式一：从 config JSON 读取全部参数 ---
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        params = cfg["params"]
        scaler_name = cfg.get("scaler", "standard").lower()
        scaler_kwargs = cfg.get("scaler_params", None)
        data_csv = os.path.join(
            params.get("root_path", "PatchTST_supervised/dataset/"),
            params.get("data_path", "sunspot_with_cycle.csv"),
        )
        num_train = cfg.get("num_train", 3119)
        enc_in = params.get("enc_in", 3)

        # 自动推断 results_dir
        default_root = os.path.join(
            "results",
            "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                params.get("model_id", "sunspot"),
                params.get("model", "PatchTST"),
                params.get("data", "custom"),
                params.get("features", "M"),
                params.get("seq_len", 96),
                params.get("label_len", 48),
                params.get("pred_len", 24),
                params.get("d_model", 128),
                params.get("n_heads", 8),
                params.get("e_layers", 2),
                params.get("d_layers", 1),
                params.get("d_ff", 2048),
                params.get("factor", 1),
                params.get("embed", "timeF"),
                params.get("distil", True),
                cfg.get("experiment_id", "0"),
            ),
        )
        results_dir = args.results_dir or default_root

    # --- 模式二：手动指定 ---
    else:
        if not all([args.results_dir, args.scaler, args.data_csv, args.num_train]):
            p.error("手动模式需要: results_dir --scaler --data_csv --num_train")
        results_dir = args.results_dir
        scaler_name = args.scaler.lower()
        scaler_kwargs = None
        data_csv = args.data_csv
        num_train = args.num_train
        enc_in = args.enc_in

    # --- 核心计算 ---
    pred_path = os.path.join(results_dir, "pred.npy")
    true_path = os.path.join(results_dir, "true.npy")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} 不存在")
        sys.exit(1)

    pred_z = np.load(pred_path)
    true_z = np.load(true_path)
    ssn_col = pred_z.shape[-1] - 1  # 最后一列

    pred_ssn_z = pred_z[:, :, ssn_col]
    true_ssn_z = true_z[:, :, ssn_col]

    _, scaler = load_data(data_csv, num_train, scaler_name, scaler_kwargs)
    pred_phy = scaler.inverse_transform(pred_ssn_z.reshape(-1, 1)).reshape(pred_ssn_z.shape)
    true_phy = scaler.inverse_transform(true_ssn_z.reshape(-1, 1)).reshape(true_ssn_z.shape)

    metrics = compute(pred_phy, true_phy)
    strat = metrics.pop("error_stratification")

    print(f"results_dir: {results_dir}")
    print(f"scaler: {scaler_name}  num_train: {num_train}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    print("-" * 40)
    print("误差分层 (MAE):")
    for k, v in strat.items():
        print(f"  SSN {k:>10}: {v}")


if __name__ == "__main__":
    main()
