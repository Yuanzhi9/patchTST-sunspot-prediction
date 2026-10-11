"""
rule_fine_grid.py — 细粒度快照（前 20 轮每 2 轮）+ 规则复评（best-val / val单起点轨迹 / L-B 组合信号）

冻结的决策（景修 2026-10-11）：
  主规则 = val 单起点轨迹（在验证段跑一条 block24 轨迹，取轨迹 MAE 最小者）
  V3（多起点）已撤回；本脚本检验 L-B：把 val 峰值误差并进信号是否更好。
  L-B 权重取等权 + 逐窗口标准化（a priori 定，不用 test 调权）。

每 (窗口×dropout)：1 次训练（--ckpt_every 5 --ckpt_every_early 2 --ckpt_early_until 20）
每快照：测试段 block24（结果）+ val 单起点轨迹 MAE + val 段峰值误差（信号）
用法: python3 scripts/rule_fine_grid.py
"""

import csv
import glob
import os
import re
import subprocess
import sys
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
sys.path.insert(2, os.path.join(ROOT, "scripts"))
from roll_eval import load_model, load_data_enc3     # noqa: E402
from b23_block_eval import block_predict            # noqa: E402

P = "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue"
RUNS = [("W1", "1996-08", "2008-11", 2838, 148, "0.05"), ("W1", "1996-08", "2008-11", 2838, 148, "0.20"),
        ("W2", "2008-12", "2019-11", 2986, 132, "0.05"), ("W2", "2008-12", "2019-11", 2986, 132, "0.20"),
        ("W3", "2019-12", "2025-10", 3118, 71, "0.05"), ("W3", "2019-12", "2025-10", 3118, 71, "0.20")]
VAL = 132
COMMON = ("--model_id sunspot --model PatchTST --data custom "
          "--root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv "
          "--features MS --target ssn --enc_in 3 --seq_len 336 --label_len 48 --pred_len 24 "
          "--d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 "
          "--revin 1 --affine 0 --head_dropout 0.0 --individual 0 "
          "--train_epochs 50 --patience 100 --batch_size 16 --learning_rate 0.0001 --loss mse "
          "--activation gelu --random_seed 2021 --num_workers 0 --itr 1 --use_gpu 1")


def phy(sc, r):
    return float(sc.inverse_transform(np.asarray(r).reshape(1, -1))[0, 2])


def main():
    rows = []
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "PatchTST_supervised"), NVIDIA_TF32_OVERRIDE="0")
    for win, ts, te, nt, ntest, do in RUNS:
        des = f"FINE-{win}-do{do.replace('.', '')}"
        args = (f"{COMMON} --dropout {do} --fc_dropout {do} --test_start {ts} --test_end {te} --des {des}")
        print(f"=== {win} do{do} 训练（细粒度快照）===", flush=True)
        subprocess.run(f"python3 -u run_longExp.py --is_training 1 {args} "
                       f"--ckpt_every 5 --ckpt_every_early 2 --ckpt_early_until 20",
                       cwd=ROOT, shell=True, env=env, capture_output=True, text=True)
        _, sc, dz, t0 = load_data_enc3(os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv"),
                                       nt, VAL, "standard", "")
        v0 = t0 - VAL
        tt = np.array([phy(sc, dz[t0 + t]) for t in range(ntest)])
        tv = np.array([phy(sc, dz[v0 + t]) for t in range(VAL)])
        for ck in sorted(glob.glob(f"checkpoints/{P}_{des}_0/ckpt_e*.pth"),
                         key=lambda f: int(re.search(r"ckpt_e(\d+)", f).group(1))):
            e = int(re.search(r"ckpt_e(\d+)", ck).group(1))
            m, _ = load_model(ck)
            pt = block_predict(m, sc, dz, t0, 336, ntest, 24, 0)
            pv = block_predict(m, sc, dz, v0, 336, VAL, 24, 0)
            rows.append(dict(win=win, do=do, epoch=e,
                             test_mae=float(np.mean(np.abs(pt - tt))),
                             val_traj=float(np.mean(np.abs(pv - tv))),
                             val_peak_err=abs(float(np.max(pv)) - float(np.max(tv)))))
            print(f"  {win} do{do} e{e:02d} | 测试 {rows[-1]['test_mae']:6.2f} | "
                  f"val轨迹 {rows[-1]['val_traj']:6.2f} | val峰误差 {rows[-1]['val_peak_err']:6.1f}", flush=True)
    out = os.path.join(ROOT, "results", f"rule_fine_grid_{date.today()}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nCSV:", out)


if __name__ == "__main__":
    sys.exit(main())
