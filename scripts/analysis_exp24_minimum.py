"""
analysis_exp24_minimum.py — EXP-24 极小值专项评估

复用 plot_exploration.py 已验证的推理函数（滚动/block/oracle），
对三窗口谷点做三口径评估：时间误差 / 相对幅度误差 / 绝对幅度误差。

谷点清单（月均口径，2026-08-20 CSV 实测）：
  W1 起点谷 1996-10（后侧评估）、W1 终点谷 2008-08（完整 ±12 月）
  W2 起点谷 2009-08（后侧）、W3 起点谷 2020-02（后侧）
  平滑参照列：轨迹 13 月平滑后找谷，与官方月对比（标注边界效应）

对拍门禁：轨迹 MAE 必须等于已记值（滚动 50.41/66.80/42.05，block 21.29/27.28/23.75）。
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3
from scripts.b23_block_eval import block_predict
from scripts.plot_exploration import trues_array

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")

# (窗口, ckpt名, num_train, n_roll, test_start, 滚动MAE记值, block MAE记值)
WINDOWS = [
    ("W1", "EXP-18-5b", 2838, 148, "1996-08", 50.41, 21.29),
    ("W2", "EXP-19-2", 2986, 132, "2008-12", 66.80, 27.28),
    ("W3", "EXP-19-3", 3118, 71, "2019-12", 42.05, 23.75),
]

# 谷点清单（月均口径）：(窗口, 谷月, 评估模式 full=±12 / after=后侧12)
TROUGHS = [
    ("W1", "1996-10", "after"),
    ("W1", "2008-08", "full"),
    ("W2", "2009-08", "after"),
    ("W3", "2020-02", "after"),
]


def smooth13(x):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 6), min(n, i + 7)
        out[i] = np.mean(x[lo:hi])
    return out


def find_ckpt(name):
    import glob
    hits = glob.glob(os.path.join(ROOT, f"checkpoints/*{name}_0/full_checkpoint.pth"))
    assert hits, f"ckpt not found {name}"
    return hits[0]


def main():
    results = {}
    for wname, ckpt_name, nt, nr, ts, exp_roll, exp_block in WINDOWS:
        model, ckpt_args = load_model(find_ckpt(ckpt_name))
        seq_len = getattr(ckpt_args, "seq_len", 96)
        _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, nt, 132)
        trues = trues_array(scaler, data_z, t0, nr)
        roll_preds, _ = run_rolling_enc3(model, scaler, data_z, t0, seq_len, nr)
        block_preds = block_predict(model, scaler, data_z, t0, seq_len, nr, 24, 0)
        mae_r = float(np.mean(np.abs(roll_preds - trues)))
        mae_b = float(np.mean(np.abs(block_preds - trues)))
        ok = abs(mae_r - exp_roll) < 0.5 and abs(mae_b - exp_block) < 0.5
        print(f"{wname}: 滚动={mae_r:.2f}(记{exp_roll}) block={mae_b:.2f}(记{exp_block}) 对拍{'PASS' if ok else 'FAIL'}")
        assert ok, f"{wname} 对拍失败，停止"
        x = pd.date_range(start=ts, periods=nr, freq="MS")
        results[wname] = (x, trues, roll_preds, block_preds)

    print("\n=== 谷点三口径评估 ===\n")
    for wname, trough_ym, mode in TROUGHS:
        x, trues, roll, block = results[wname]
        ym = np.array([d.strftime("%Y-%m") for d in x])
        ti = int(np.where(ym == trough_ym)[0][0])
        lo = max(0, ti - 12) if mode == "full" else ti
        hi = min(len(trues), ti + 13)
        seg_t = trues[lo:hi]
        true_min = float(np.min(seg_t))
        true_min_i = int(lo + np.argmin(seg_t))

        print(f"--- {wname} {trough_ym} (mode={mode}, 段 {ym[lo]}~{ym[hi-1]}) ---")
        print(f"  真实谷: {ym[true_min_i]} 值 {true_min:.1f}")
        for label, preds in [("滚动", roll), ("block", block)]:
            seg_p = preds[lo:hi]
            p_min = float(np.min(seg_p))
            p_min_i = int(lo + np.argmin(seg_p))
            has_trough = (np.argmin(seg_p) > 0 and np.argmin(seg_p) < len(seg_p) - 1)
            neg_flag = " ⚠️负值崩塌" if p_min < 0 else ""
            if not has_trough:
                print(f"  {label}: 预测侧无谷（段内单调）→ 时间误差=无谷")
                print(f"      段内最小预测 {p_min:.1f}{neg_flag} vs 真实谷 {true_min:.1f}（绝对差 {p_min-true_min:+.1f}）")
                continue
            time_err = (x[p_min_i] - x[true_min_i]).days // 30
            abs_err = p_min - true_min
            # 相对误差仅对真实谷值>=1 有意义（谷值过小分母爆炸失真）
            rel_str = f"{abs_err/max(true_min,1.0)*100:+.0f}%（按 max(真实谷,1) 归一）"
            print(f"  {label}: 预测谷 {ym[p_min_i]} (值 {p_min:.1f}{neg_flag}) vs 真实谷 {ym[true_min_i]} (值 {true_min:.1f})")
            print(f"      时间误差 {time_err:+d} 月 | 绝对幅度误差 {abs_err:+.1f} | 相对幅度误差 {rel_str}")

        # 平滑参照（真实序列）
        seg_s = smooth13(trues[lo:hi])
        s_min_i = int(lo + np.argmin(seg_s))
        print(f"  参照[平滑13月·真实]: 平滑谷 {ym[s_min_i]} (值 {seg_s[np.argmin(seg_s)]:.1f}) ⚠️边界效应")
        print()


if __name__ == "__main__":
    main()
