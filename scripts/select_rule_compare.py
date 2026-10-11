"""
select_rule_compare.py — 选模规则比较（do0.2，三窗口）

做法：
  1) 每个窗口训练一次（--ckpt_every 5 → 10 个逐轮快照）
  2) 每个快照跑一次补测（--is_training 0），记录全步 z-MAE（与物理 MAE 严格成比例：MAE_phy = σ·MAE_z）
  3) 五条规则各挑一个模型：①best-val ②平滑 val（3 轮滑动平均）③固定轮次 ④末轮 ⑤权重平均(最后 5 轮)
  4) 对"被挑中"的模型再算 block24 MAE（b23_block_eval，物理口径）

输出：表 + `results/rule_compare_<date>.csv`
用法: python3 scripts/select_rule_compare.py
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import date

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
WIN = [("W1", "1996-08", "2008-11", 2838, 148),
       ("W2", "2008-12", "2019-11", 2986, 132),
       ("W3", "2019-12", "2025-10", 3118, 71)]
EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
COMMON = ("--model_id sunspot --model PatchTST --data custom "
          "--root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv "
          "--features MS --target ssn --enc_in 3 --seq_len 336 --label_len 48 --pred_len 24 "
          "--d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 "
          "--revin 1 --affine 0 --dropout 0.2 --fc_dropout 0.2 --head_dropout 0.0 --individual 0 "
          "--train_epochs 50 --patience 100 --batch_size 16 --learning_rate 0.0001 --loss mse "
          "--activation gelu --random_seed 2021 --num_workers 0 --itr 1 --use_gpu 1")


def run(cmd):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "PatchTST_supervised"),
               NVIDIA_TF32_OVERRIDE="0")
    p = subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True, env=env)
    return p.stdout + p.stderr


def setting_name(des):
    return ("sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_"
            f"fc1_ebtimeF_dtTrue_{des}_0")


def main():
    rows = []
    for win, ts, te, nt, nroll in WIN:
        des = f"RULE-{win}"
        args = f"{COMMON} --test_start {ts} --test_end {te} --des {des}"
        print(f"=== {win}: 训练（--ckpt_every 5）===", flush=True)
        out = run(f"python3 -u run_longExp.py --is_training 1 {args} --ckpt_every 5")
        val_curve = [(int(a), float(c)) for a, b, c in
                     re.findall(r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*([\d.]+)\s*Vali Loss:\s*([\d.]+)", out)]
        print(f"  val 曲线 {len(val_curve)} 轮；快照目录 {des}", flush=True)

        per_epoch = {}
        for e in EPOCHS:
            sub = f"{des}-e{e:02d}"
            d = os.path.join(CKPT, setting_name(sub))
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(CKPT, setting_name(des), f"ckpt_e{e:02d}.pth"),
                        os.path.join(d, "full_checkpoint.pth"))
            o = run(f"python3 -u run_longExp.py --is_training 0 {args} --des {sub}")
            m = re.search(r"mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)", o)
            if not m:
                open(f"/tmp/rule_fail_e{e}.txt", "w").write(o[-3000:])
                raise SystemExit(f"补测失败 {win} e{e}，输出见 /tmp/rule_fail_e{e}.txt")
            per_epoch[e] = float(m.group(2))
            print(f"    e{e:02d}: z-MAE {per_epoch[e]:.4f}", flush=True)

        # 规则 ⑤ 权重平均（最后 5 轮 = e30..e50 中的 e30? 用 e40..e50 的 5 个）
        avg_epochs = [40, 45, 50]
        acc = None
        for e in avg_epochs:
            ck = torch.load(os.path.join(CKPT, setting_name(des), f"ckpt_e{e:02d}.pth"),
                            map_location="cpu", weights_only=False)
            sd = ck["model_state_dict"]
            acc = sd if acc is None else {k: acc[k] + sd[k] for k in acc}
        acc = {k: v / len(avg_epochs) for k, v in acc.items()}
        sub = f"{des}-avg"
        d = os.path.join(CKPT, setting_name(sub))
        os.makedirs(d, exist_ok=True)
        torch.save({"model_state_dict": acc}, os.path.join(d, "full_checkpoint.pth"))
        o = run(f"python3 -u run_longExp.py --is_training 0 {args} --des {sub}")
        m = re.search(r"mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)", o)
        if not m:
            open("/tmp/rule_fail_avg.txt", "w").write(o[-3000:])
            raise SystemExit(f"补测失败 {win} avg，输出见 /tmp/rule_fail_avg.txt")
        avg_mae = float(m.group(2))
        print(f"    avg(权重平均 e{avg_epochs}): z-MAE {avg_mae:.4f}", flush=True)

        rows.append(dict(win=win, nt=nt, nroll=nroll, des=des, val=val_curve,
                         per_epoch=per_epoch, avg=avg_mae))

    out_csv = os.path.join(ROOT, "results", f"rule_compare_{date.today()}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window", "rule", "epoch", "z_MAE_fullstep"])
        for r in rows:
            for e, v in r["per_epoch"].items():
                w.writerow([r["win"], "per_epoch", e, v])
            w.writerow([r["win"], "weight_avg", "40-50", r["avg"]])
    print("\nCSV:", out_csv)


if __name__ == "__main__":
    sys.exit(main())
