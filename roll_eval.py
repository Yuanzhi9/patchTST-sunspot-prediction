"""
roll_eval.py — H1 滚动评估：逐月推进，预测值回填输入，测量误差自回归累积

支持两种管线（自动检测 checkpoint args.enc_in）：
  enc_in == 1：raw SSN → 模型（RevIN 内部处理）→ 物理 SSN 直接输出
  enc_in  > 1：month_sin, month_cos, ssn → StandardScaler → 模型 → inverse scaler

用法:
  python roll_eval.py --checkpoint PATH --data_csv PATH
  python roll_eval.py --config configs/EXP-XX.json

验证: 用 EXP-14 checkpoint 跑，单步推理结果应与 npy 一致（已验证）。
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, 'PatchTST_supervised'))
from run_sunspot_fixed import build_parser as rsf_parser
from models.PatchTST import Model

# [2026-08-15 景修] 目标变换映射表（探索期 EXP-21 系列）。
# 改前：仅 'sqrt' 分支。改后：四变换映射，与 data_loader.py / eval_metrics.py 对应。
TARGET_TRANSFORMS = {
    'sqrt': lambda x: np.sqrt(x),
    'pow07': lambda x: np.power(x, 0.7),
    'pow23': lambda x: np.power(x, 2.0 / 3.0),
    'log1p': lambda x: np.log1p(x),
}
INV_TARGET_TRANSFORMS = {
    'sqrt': lambda y: np.clip(y, 0, None) ** 2,
    'pow07': lambda y: np.clip(y, 0, None) ** (1.0 / 0.7),
    'pow23': lambda y: np.clip(y, 0, None) ** 1.5,
    'log1p': lambda y: np.expm1(y),
}


def load_data_enc1(data_csv, num_train, num_val):
    """enc_in=1 管线：只取 ssn 列，不归一化。RevIN 内部处理。"""
    df = pd.read_csv(data_csv, parse_dates=['date'])
    train_ssn = df['ssn'].values[:num_train]
    scaler = StandardScaler()
    scaler.fit(train_ssn.reshape(-1, 1))
    ssn_arr = df['ssn'].values
    test_start_idx = num_train + num_val
    return df, scaler, ssn_arr, test_start_idx


def load_data_enc3(data_csv, num_train, num_val, scaler_name='standard', target_transform=''):
    """enc_in=3 管线：month_sin + month_cos + ssn → StandardScaler。
    [2026-08-15 景修] target_transform='sqrt' 时 ssn 列先取平方根再 fit scaler（与训练侧一致）。"""
    df = pd.read_csv(data_csv)
    cols = ['month_sin', 'month_cos', 'ssn']
    arr = df[cols].values
    # [2026-08-15 景修] 目标变换（映射表）。改前：仅 'sqrt' 分支。改后：四变换。
    if target_transform:
        fn = TARGET_TRANSFORMS.get(target_transform)
        if fn is None:
            raise ValueError(f"未知 target_transform: {target_transform}")
        arr[:, -1] = fn(np.clip(arr[:, -1], 0, None))

    if scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f'Unknown scaler: {scaler_name}')

    scaler.fit(arr[:num_train])
    data_z = scaler.transform(arr)
    test_start_idx = num_train + num_val
    return df, scaler, data_z, test_start_idx


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'args' in ckpt:
        ckpt_args = ckpt['args']
    elif 'config' in ckpt:
        ckpt_args = ckpt['config']
    else:
        raise ValueError('Checkpoint missing args/config — cannot determine model architecture')

    ckpt_args.use_gpu = False

    model = Model(ckpt_args).float()
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, ckpt_args


def run_rolling_enc1(model, ssn_arr, test_start_idx, seq_len, n_roll):
    """enc_in=1：raw SSN → model（RevIN 内部 norm+denorm）→ 物理 SSN 输出。"""
    window = ssn_arr[test_start_idx - seq_len : test_start_idx].copy()

    preds = []
    trues = []

    for step in range(n_roll):
        true_val = ssn_arr[test_start_idx + step]
        trues.append(float(true_val))

        x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        with torch.no_grad():
            out = model(x)                                        # (1, 24, 1)
        pred_val = out[0, 0, 0].item()
        preds.append(pred_val)

        window = np.append(window[1:], pred_val)

    return np.array(preds), np.array(trues)


def run_rolling_enc3(model, scaler, data_z, test_start_idx, seq_len, n_roll, target_transform=''):
    """enc_in=3：StandardScaler z-score → model → inverse scaler。
    [2026-08-15 景修] target_transform='sqrt' 时 inverse scaler 得 sqrt 空间值，输出前平方回物理空间；
    滚动回填链保持 sqrt 空间一致（scaler 训练于 sqrt 空间）。"""
    window = data_z[test_start_idx - seq_len : test_start_idx].copy()  # (96, 3)

    preds_phy = []
    trues_phy = []

    for step in range(n_roll):
        true_z_month = data_z[test_start_idx + step]
        true_phy_sqrt = scaler.inverse_transform(true_z_month.reshape(1, -1))[0, 2]

        x = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        pred_step0_z = out[0, 0, 2].item()

        pred_row_z = np.array([[window[-1, 0], window[-1, 1], pred_step0_z]])
        pred_phy_sqrt = scaler.inverse_transform(pred_row_z)[0, 2]

        if target_transform:
            inv = INV_TARGET_TRANSFORMS.get(target_transform)
            if inv is None:
                raise ValueError(f"未知 target_transform 逆变换: {target_transform}")
            preds_phy.append(inv(pred_phy_sqrt))
            trues_phy.append(inv(true_phy_sqrt))
        else:
            preds_phy.append(pred_phy_sqrt)
            trues_phy.append(true_phy_sqrt)

        new_row_phy = np.array([[0, 0, pred_phy_sqrt]])
        new_row_z = scaler.transform(new_row_phy)
        new_row_z[0, 0] = true_z_month[0]  # real month_sin
        new_row_z[0, 1] = true_z_month[1]  # real month_cos

        window = np.vstack([window[1:], new_row_z])

    return np.array(preds_phy), np.array(trues_phy)


def report(preds, trues, label, n_roll):
    errors = np.abs(preds - trues)
    mae = float(np.mean(errors))

    print(f"滚动 MAE  (全 {n_roll} 月): {mae:.2f}")

    third = max(n_roll // 3, 1)
    if n_roll >= 6:
        mae_first = float(np.mean(errors[:third]))
        mae_last  = float(np.mean(errors[-third:]))
        print(f"前期 MAE  (前 {third} 月): {mae_first:.2f}")
        print(f"后期 MAE  (后 {third} 月): {mae_last:.2f}")
        ratio = mae_last / mae_first if mae_first > 0 else float('inf')
        print(f"后期/前期: {ratio:.1f}x")

    peak_idx = int(np.argmax(trues))
    print(f"真实峰值: {trues[peak_idx]:.1f}  预测: {preds[peak_idx]:.1f}  误差: {trues[peak_idx] - preds[peak_idx]:.1f} (第 {peak_idx+1} 月)")


def main():
    p = argparse.ArgumentParser(description="H1 滚动评估")
    p.add_argument("--checkpoint", default=None, help="full_checkpoint.pth 路径")
    p.add_argument("--config", default=None, help="config JSON（自动推断 checkpoint 路径）")
    p.add_argument("--data_csv", default="PatchTST_supervised/dataset/sunspot_with_cycle.csv")
    p.add_argument("--num_train", type=int, default=3119)
    p.add_argument("--num_val", type=int, default=132)
    p.add_argument("--n_roll", type=int, default=70)
    p.add_argument("--scaler", default="standard", help="enc_in > 1 时用")
    # [2026-08-15 景修] 新增：目标变换逆操作（与训练侧 --target_transform 对应）
    p.add_argument("--target_transform", default="", help="目标变换: 空/sqrt")
    args = p.parse_args()

    # --- config JSON mode ---
    if args.config and not args.checkpoint:
        with open(args.config) as f:
            cfg = json.load(f)
        params = cfg['params']
        ckpt_dir = 'sunspot_PatchTST_custom_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            params.get('features', 'M'), params.get('seq_len', 96),
            params.get('label_len', 48), params.get('pred_len', 24),
            params.get('d_model', 128), params.get('n_heads', 8),
            params.get('e_layers', 2), params.get('d_layers', 1),
            params.get('d_ff', 2048), params.get('factor', 1),
            params.get('embed', 'timeF'), params.get('distil', True),
            cfg.get('experiment_id', '0'))
        args.checkpoint = os.path.join(
            params.get('checkpoints', 'PatchTST_supervised/checkpoints/'),
            ckpt_dir, 'full_checkpoint.pth')
        args.data_csv = os.path.join(
            params.get('root_path', 'PatchTST_supervised/dataset/'),
            params.get('data_path', 'sunspot_with_cycle.csv'))
        args.num_train = cfg.get('num_train', 3119)
        args.num_val = params.get('num_val', 132) if 'num_val' in params else 132
        # [2026-08-15 景修] config 模式同步读 target_transform
        args.target_transform = params.get('target_transform', '')

    if not args.checkpoint:
        p.error("需要 --checkpoint 或 --config")

    print(f"Checkpoint: {args.checkpoint}")
    model, ckpt_args = load_model(args.checkpoint)
    enc_in = getattr(ckpt_args, 'enc_in', 3)
    seq_len = getattr(ckpt_args, 'seq_len', 96)
    print(f"enc_in={enc_in}  seq_len={seq_len}  d_model={getattr(ckpt_args, 'd_model', '?')}  epochs={getattr(ckpt_args, 'train_epochs', '?')}")

    # --- load data, select pipeline ---
    if enc_in == 1:
        print("管线: raw SSN → RevIN（内部处理）")
        df, scaler, data_arr, t0 = load_data_enc1(args.data_csv, args.num_train, args.num_val)
        n_roll = min(args.n_roll, len(data_arr) - t0)
        preds, trues = run_rolling_enc1(model, data_arr, t0, seq_len, n_roll)
    else:
        print(f"管线: {args.scaler} scaler → model → inverse")
        df, scaler, data_z, t0 = load_data_enc3(args.data_csv, args.num_train, args.num_val, args.scaler, args.target_transform)
        n_roll = min(args.n_roll, len(data_z) - t0)
        preds, trues = run_rolling_enc3(model, scaler, data_z, t0, seq_len, n_roll, args.target_transform)

    report(preds, trues, ckpt_args, n_roll)


if __name__ == '__main__':
    main()
