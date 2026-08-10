"""
roll_eval.py — H1 滚动评估：逐月推进，预测值回填输入，测量误差自回归累积

用法:
  python roll_eval.py --config configs/EXP-XX_YYYY-MM-DD.json
  python roll_eval.py --manual --checkpoint PATH --data_csv PATH --num_train 3119 --scaler standard

逻辑:
  1. 取测试段前 seq_len 个月全真实值作为初始窗口
  2. 模型预测 24 步 → 取 step 0 → inverse 到物理 SSN
  3. 将预测值视为"新观测"回填输入末尾，窗口右移 1 格
  4. 循环 n_roll 次（覆盖全测试段）
  5. 输出滚动 MAE + 逐月误差
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)  # 先搜根目录（含 build_parser）
sys.path.insert(1, os.path.join(ROOT_DIR, 'PatchTST_supervised'))  # 后搜子目录
from run_sunspot_fixed import build_parser as rsf_parser
from models.PatchTST import Model


SCALER_MAP = {"standard": StandardScaler, "minmax": MinMaxScaler}


def load_data(data_csv, num_train, scaler_name, scaler_kwargs=None):
    df = pd.read_csv(data_csv)
    cols = ['month_sin', 'month_cos', 'ssn']
    arr = df[cols].values
    train_arr = arr[:num_train]
    cls = SCALER_MAP[scaler_name]
    scaler = cls(**scaler_kwargs) if scaler_kwargs else cls()
    scaler.fit(train_arr)
    data_z = scaler.transform(arr)
    return df, scaler, data_z


def load_model(args=None, checkpoint_path=None):
    if args is None:
        parser = rsf_parser()
        args = parser.parse_args([])
        args.use_gpu = False
    model = Model(args).float()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def run_rolling(model, scaler, data_z, test_start, seq_len, pred_len, n_roll):
    """
    data_z: shape (N, 3), z-score transformed [month_sin, month_cos, ssn]
    test_start: index of first test month
    初始窗口：test_start 前 seq_len 个月全真实值
    """
    window = data_z[test_start - seq_len : test_start].copy()  # (96, 3)

    preds_phy = []
    trues_phy = []

    for step in range(n_roll):
        # --- ground truth for this step ---
        true_z_month = data_z[test_start + step]                # (3,)
        true_phy = scaler.inverse_transform(true_z_month.reshape(1, -1))[0, 2]

        # --- predict ---
        x = torch.FloatTensor(window).unsqueeze(0)               # (1, 96, 3)
        with torch.no_grad():
            out = model(x)                                       # (1, 24, 3)
        pred_step0_z = out[0, 0, 2].item()                       # scalar, z-score

        # --- inverse to physical ---
        ms_z  = window[-1, 0]
        mc_z  = window[-1, 1]
        pred_row_z = np.array([[ms_z, mc_z, pred_step0_z]])
        pred_phy = scaler.inverse_transform(pred_row_z)[0, 2]

        preds_phy.append(pred_phy)
        trues_phy.append(true_phy)

        # --- slide window ---
        # 用预测的物理 SSN 构造新行 → z-score transform → 追加
        new_month_ms = true_z_month[0]    # 用真实 month_sin (CSV 确定值)
        new_month_mc = true_z_month[1]    # 用真实 month_cos
        new_row_phy = np.array([[0, 0, pred_phy]])               # 只有 SSN 是物理值
        new_row_z = scaler.transform(new_row_phy)
        new_row_z[0, 0] = new_month_ms                           # month_sin 用真实 z-score
        new_row_z[0, 1] = new_month_mc                           # month_cos 用真实 z-score

        window = np.vstack([window[1:], new_row_z])              # drop first, append

    preds_phy = np.array(preds_phy)
    trues_phy = np.array(trues_phy)
    return preds_phy, trues_phy


def main():
    p = argparse.ArgumentParser(description="H1 滚动评估")
    p.add_argument("--config", default=None,
                   help="config JSON 路径")
    p.add_argument("--manual", action="store_true",
                   help="手动指定参数（不使用 config JSON）")
    p.add_argument("--checkpoint", default=None,
                   help="full_checkpoint.pth 路径")
    p.add_argument("--data_csv", default="PatchTST_supervised/dataset/sunspot_with_cycle.csv")
    p.add_argument("--num_train", type=int, default=3119)
    p.add_argument("--num_val", type=int, default=132)
    p.add_argument("--scaler", default="standard")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--n_roll", type=int, default=70,
                   help="滚动步数（测试集月数）")
    args = p.parse_args()

    # --- cfg ---
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        params = cfg['params']
        scaler_name = cfg.get('scaler', 'standard').lower()
        scaler_kwargs = cfg.get('scaler_params', None)
        data_csv = os.path.join(
            params.get('root_path', 'PatchTST_supervised/dataset/'),
            params.get('data_path', 'sunspot_with_cycle.csv'))
        num_train = cfg.get('num_train', 3119)
        num_val = params.get('num_val', 132) if 'num_val' in params else 132
        seq_len = params.get('seq_len', 96)
        pred_len = params.get('pred_len', 24)
        n_roll = args.n_roll if args.n_roll != 70 else params.get('n_roll', 70)

        # 推断 checkpoint 路径
        chkpt_dir = 'sunspot_PatchTST_custom_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            params.get('features', 'M'),
            params.get('seq_len', 96),
            params.get('label_len', 48),
            params.get('pred_len', 24),
            params.get('d_model', 128),
            params.get('n_heads', 8),
            params.get('e_layers', 2),
            params.get('d_layers', 1),
            params.get('d_ff', 2048),
            params.get('factor', 1),
            params.get('embed', 'timeF'),
            params.get('distil', True),
            cfg.get('experiment_id', '0'),
        )
        default_ckpt = os.path.join(
            params.get('checkpoints', 'PatchTST_supervised/checkpoints/'),
            chkpt_dir, 'full_checkpoint.pth')
        ckpt_path = args.checkpoint or default_ckpt

        # 重建 args for model
        parser = rsf_parser()
        model_args = parser.parse_args([])
        for k, v in params.items():
            if hasattr(model_args, k):
                setattr(model_args, k, v)
        model_args.use_gpu = False
    else:
        # 手动模式
        if not args.checkpoint:
            p.error("手动模式需 --checkpoint")
        scaler_name = args.scaler
        scaler_kwargs = None
        data_csv = args.data_csv
        num_train = args.num_train
        num_val = args.num_val
        seq_len = args.seq_len
        pred_len = args.pred_len
        n_roll = args.n_roll
        ckpt_path = args.checkpoint
        model_args = None

    # --- load ---
    df, scaler, data_z = load_data(data_csv, num_train, scaler_name, scaler_kwargs)
    test_start = num_train + num_val

    actual_roll = min(n_roll, len(data_z) - test_start)
    if actual_roll < n_roll:
        print(f"⚠️ 数据不足以滚动 {n_roll} 步，改为 {actual_roll}")
        n_roll = actual_roll

    model = load_model(model_args, ckpt_path)

    # --- run ---
    preds, trues = run_rolling(model, scaler, data_z, test_start,
                               seq_len, pred_len, n_roll)

    # --- report ---
    errors = np.abs(preds - trues)
    mae_rolling = float(np.mean(errors))

    print(f"Checkpoint: {ckpt_path}")
    print(f"scaler: {scaler_name}  num_train: {num_train}")
    print(f"Rolling steps: {n_roll}")
    print("-" * 50)
    print(f"滚动 MAE  (全 {n_roll} 月): {mae_rolling:.2f}")

    # 前 1/3 和 后 1/3 分段
    third = n_roll // 3
    if third >= 4:
        mae_first = float(np.mean(errors[:third]))
        mae_last  = float(np.mean(errors[-third:]))
        print(f"前期 MAE  (前 {third} 月): {mae_first:.2f}")
        print(f"后期 MAE  (后 {third} 月): {mae_last:.2f}")
        ratio = mae_last / mae_first if mae_first > 0 else float('inf')
        print(f"后期/前期: {ratio:.1f}x")
        if ratio > 1.5:
            print("  → 误差自回归累积明显 ✓")
        else:
            print("  → 误差累积不明显？需人工检查")

    # 峰值对比
    peak_idx = int(np.argmax(trues))
    print(f"真实峰值: {trues[peak_idx]:.1f}  预测: {preds[peak_idx]:.1f}  误差: {trues[peak_idx] - preds[peak_idx]:.1f} (第 {peak_idx+1} 月)")


if __name__ == '__main__':
    main()
