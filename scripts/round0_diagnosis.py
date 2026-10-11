"""
round0_diagnosis.py — 第 0 轮体检（L-0，零代码改动）

回答：正式选模口径"最佳 val 模型"是否可靠？
 ① best-val 轮 vs best-test 轮（差多少轮）
 ② val 曲线震荡幅度（前 10 轮相邻差最大值、峰谷比）
 ③ 训练段（末轮模型）vs 补测段（最佳 val 模型）在 test 上的差（result.txt 两行，z-space MAE）
 ④ val 与 test 曲线的相关性

判据（plan_round0_diagnosis_2026-10-11.md）：
  若 ≥2 组 best-test 与 best-val 相差 >5 轮，或 val 震荡 >30% → 判"口径不稳" → 进 L-1（改代码存多轮 checkpoint）

数据源：launch_experiment 落盘的 logs/EXP-*.log（每 epoch 都算了 Train/Vali/Test Loss）
用法: python3 scripts/round0_diagnosis.py
"""

import glob
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*([\d.]+)\s*"
                 r"Vali Loss:\s*([\d.]+)\s*Test Loss:\s*([\d.]+)")

# 实验 → (窗口, dropout)；EOF 过滤
RUNS = {
    "EXP-25": ("W1", "0.05"), "EXP-26": ("W1", "0.20"),
    "EXP-27": ("W2", "0.05"), "EXP-28": ("W2", "0.20"),
    "EXP-29": ("W3", "0.05"), "EXP-30": ("W3", "0.20"),
}


def parse(eid):
    f = os.path.join(ROOT, "logs", f"{eid}.log")
    txt = open(f, encoding="utf-8", errors="ignore").read()
    rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in PAT.findall(txt)]
    return np.array(rows) if rows else None


def main():
    print(f"{'实验':7} {'窗口':4} {'do':5} {'轮数':4} {'best-val@':>9} {'best-test@':>10} "
          f"{'差(轮)':>6} {'val前10相邻maxΔ':>15} {'val前10峰谷比':>12} {'val~test r':>10}")
    first10_osc, gaps = [], []
    for eid, (win, do) in RUNS.items():
        arr = parse(eid)
        if arr is None:
            print(f"{eid:7} 解析失败"); continue
        ep, tr, va, te = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        bv, bt = int(ep[int(np.argmin(va))]), int(ep[int(np.argmin(te))])
        v10 = va[:10]
        maxd = float(np.max(np.abs(np.diff(v10)))) if len(v10) > 1 else float("nan")
        ratio = float((v10.max() - v10.min()) / v10.mean())
        r = float(np.corrcoef(va, te)[0, 1]) if len(va) > 2 else float("nan")
        first10_osc.append(ratio); gaps.append(abs(bt - bv))
        print(f"{eid:7} {win:4} {do:5} {len(arr):4d} {bv:9d} {bt:10d} {abs(bt-bv):6d} "
              f"{maxd:15.3f} {ratio:12.2%} {r:10.2f}")

    n_big = sum(1 for g in gaps if g > 5)
    osc_big = sum(1 for o in first10_osc if o > 0.30)
    print(f"\n相差 >5 轮的组数: {n_big}/6（判据 ≥2 → 口径不稳）")
    print(f"前 10 轮震荡 >30% 的组数: {osc_big}/6（判据 ≥1 亦为警示）")
    verdict = "口径不稳 → 进 L-1" if (n_big >= 2 or osc_big >= 3) else "口径基本可用 → 可直接进 EXP-B00"
    print(f"判定：{verdict}")

    # ③ 训练段 vs 补测段（z-space MAE，来自 result.txt 两行）
    print("\n训练段（末轮模型）vs 补测段（最佳 val 模型）— z-space MAE，来自 result.txt：")
    res = open(os.path.join(ROOT, "result.txt"), encoding="utf-8", errors="ignore").read()
    for eid in RUNS:
        lines = re.findall(rf"_{eid}_0\s*\nmse:([\d.eE-]+), mae:([\d.eE-]+), rse:([\d.eE-]+)", res)
        if len(lines) >= 2:
            last, best = float(lines[0][1]), float(lines[1][1])
            win, do = RUNS[eid]
            print(f"  {eid} ({win} do{do}): 末轮 {last:.4f} | 最佳val {best:.4f} | "
                  f"{'最佳val更好' if best < last else '末轮更好'} | 差 {(best-last)/last*100:+.1f}%")


if __name__ == "__main__":
    main()
