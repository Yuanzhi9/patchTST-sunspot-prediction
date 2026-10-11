"""
rule_v3_do005.py — do0.05 上的选模规则复核（best-val / 单起点 val 轨迹 / V3 多起点 val 轨迹）

V3（景修 2026-10-11 批准为 rule 6 实现）= 在验证段内多起点跑 block24 轨迹，按段长加权平均误差作选择信号。
全部只用 val（不碰 test）；test 只用于最后评估。

每窗口：1 次训练（--ckpt_every 5，do0.05，seed 2021）→ 10 个逐轮快照
每快照：测试段轨迹 MAE（结果）+ val 单起点轨迹 MAE（v2 信号）+ val 多起点加权 MAE（V3 信号）+ val 峰值误差
用法: python3 scripts/rule_v3_do005.py
"""

import csv
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
from roll_eval import load_model, load_data_enc3          # noqa: E402
from b23_block_eval import block_predict                 # noqa: E402

P = "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue"
WIN = [("W1", "1996-08", "2008-11", 2838, 148), ("W2", "2008-12", "2019-11", 2986, 132),
       ("W3", "2019-12", "2025-10", 3118, 71)]
EPOCHS = list(range(5, 51, 5))
VAL = 132
COMMON = ("--model_id sunspot --model PatchTST --data custom "
          "--root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv "
          "--features MS --target ssn --enc_in 3 --seq_len 336 --label_len 48 --pred_len 24 "
          "--d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 "
          "--revin 1 --affine 0 --fc_dropout 0.05 --head_dropout 0.0 --individual 0 "
          "--train_epochs 50 --patience 100 --batch_size 16 --learning_rate 0.0001 --loss mse "
          "--activation gelu --random_seed 2021 --num_workers 0 --itr 1 --use_gpu 1")


def phy(scaler, row_z):
    return float(scaler.inverse_transform(np.asarray(row_z).reshape(1, -1))[0, 2])


def main():
    rows = []
    for win, ts, te, nt, ntest in WIN:
        des = f"V3-{win}"
        args = f"{COMMON} --dropout 0.05 --test_start {ts} --test_end {te} --des {des}"
        print(f"=== {win} do0.05 训练（--ckpt_every 5）===", flush=True)
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "PatchTST_supervised"), NVIDIA_TF32_OVERRIDE="0")
        subprocess.run(f"python3 -u run_longExp.py --is_training 1 {args} --ckpt_every 5",
                       cwd=ROOT, shell=True, env=env, capture_output=True, text=True)
        _, scaler, data_z, t0 = load_data_enc3(os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv"),
                                               nt, VAL, "standard", "")
        val_start = t0 - VAL
        true_test = np.array([phy(scaler, data_z[t0 + t]) for t in range(ntest)])
        true_val = np.array([phy(scaler, data_z[val_start + t]) for t in range(VAL)])

        for e in EPOCHS:
            ck = f"checkpoints/{P}_{des}_0/ckpt_e{e:02d}.pth"
            model, _ = load_model(ck)
            seq_len = 336
            # 1) 测试段轨迹（结果）
            pt = block_predict(model, scaler, data_z, t0, seq_len, ntest, 24, 0)
            test_mae = float(np.mean(np.abs(pt - true_test)))
            # 2) val 单起点轨迹（v2 信号）
            pv1 = block_predict(model, scaler, data_z, val_start, seq_len, VAL, 24, 0)
            v1 = float(np.mean(np.abs(pv1 - true_val)))
            v1_peak_err = abs(float(np.max(pv1)) - float(np.max(true_val)))
            # 3) V3：val 多起点轨迹（起点每 12 月），按段长加权
            tot, wsum = 0.0, 0
            for s in range(0, VAL - 24, 12):
                n = VAL - s
                ps = block_predict(model, scaler, data_z, val_start + s, seq_len, n, 24, 0)
                tot += float(np.mean(np.abs(ps - true_val[s:]))) * n
                wsum += n
            v3 = tot / wsum
            rows.append(dict(win=win, epoch=e, test_mae=test_mae, val_single=v1, val_v3=v3,
                             val1_peak_err=v1_peak_err))
            print(f"  e{e:02d} | 测试 block24 {test_mae:6.2f} | val 单起点 {v1:6.2f} | val V3 {v3:6.2f} | val 峰误差 {v1_peak_err:6.1f}",
                  flush=True)
    out = os.path.join(ROOT, "results", f"rule_v3_do005_{date.today()}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nCSV:", out)


if __name__ == "__main__":
    sys.exit(main())
