"""
rule_mechanism_probe.py — 选模规则的机理论证（零训练，只用已有 10 个逐轮快照）

要回答：
  H1 过拟合：全步误差是否随 epoch 单调变差？（若是 → "早模型更好"是过拟合的结果）
  H2 振幅/形状：block24 轨迹是否偏好"振幅更足"的模型？→ 记录轨迹预测峰值与真值峰的距离
  H3 是规律还是运气：block24 MAE vs epoch 的曲线是光滑的谷还是锯齿？
  ★ 关键检验：**用"验证段的 block24 轨迹误差"选轮次，能否选到测试段的最优轮次？**
     （若能 → 这就是一条合法又贴合交付场景的规则：选择指标与交付场景一致，且只看 val 不看 test）

做法：对每个窗口 × 10 个快照，各算两条 block24 轨迹：
  - 测试段轨迹：--num_train NT --num_val 132 --n_roll <测试段长>
  - 验证段轨迹：--num_train NT --num_val 132 --n_roll 132   （轨迹起点=训练段末尾，整段覆盖 val）
输出：表 + results/rule_mechanism_<date>.csv
用法: python3 scripts/rule_mechanism_probe.py
"""

import csv
import os
import re
import subprocess
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue"
WIN = [("W1", 2838, 148), ("W2", 2986, 132), ("W3", 3118, 71)]
EPOCHS = list(range(5, 51, 5))


def blk(ckpt, nt, nroll, offset=0):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "PatchTST_supervised") + ":" + os.path.join(ROOT, "scripts"),
               NVIDIA_TF32_OVERRIDE="0")
    o = subprocess.run(f"python3 scripts/b23_block_eval.py --checkpoint {ckpt} --num_train {nt} "
                       f"--num_val 132 --n_roll {nroll} --traj_start_offset {offset} --block_size 24",
                       cwd=ROOT, shell=True, capture_output=True, text=True, env=env).stdout
    mae = re.search(r"MAE = ([\d.]+)", o)
    pk = re.search(r"真实峰: ([\d.-]+)\s+预测峰: ([\d.-]+)", o)
    fp = re.search(r"前期 MAE \(前 \d+ 月\): ([\d.]+)\s+后期 \(后 \d+ 月\): ([\d.]+)", o)
    return (float(mae.group(1)) if mae else float("nan"),
            float(pk.group(1)) if pk else float("nan"), float(pk.group(2)) if pk else float("nan"),
            float(fp.group(2)) / max(float(fp.group(1)), 1e-9) if fp else float("nan"))


def main():
    rows = []
    for win, nt, ntest in WIN:
        for e in EPOCHS:
            ck = f"checkpoints/{P}_RULE-{win}-e{e:02d}_0/full_checkpoint.pth"
            t_mae, t_true, t_pred, t_ratio = blk(ck, nt, ntest)
            v_mae, v_true, v_pred, v_ratio = blk(ck, nt, 132, offset=132)
            rows.append(dict(win=win, epoch=e, test_mae=t_mae, test_peak=t_pred, test_true_peak=t_true,
                             test_post_pre=t_ratio, val_mae=v_mae, val_peak=v_pred, val_true_peak=v_true,
                             val_post_pre=v_ratio))
            print(f"{win} e{e:02d} | 测试段 block24 {t_mae:6.2f}（峰 {t_pred:5.1f}/{t_true:.0f}，后/前 {t_ratio:.1f}x）"
                  f" | 验证段 block24 {v_mae:6.2f}（峰 {v_pred:5.1f}/{v_true:.0f}）", flush=True)
    out = os.path.join(ROOT, "results", f"rule_mechanism_{date.today()}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nCSV:", out)


if __name__ == "__main__":
    sys.exit(main())
