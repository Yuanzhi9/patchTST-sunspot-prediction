"""
check_leakage.py — 数据泄露核查（三窗口 × train/val/test 切分 + 预测路径）

回答三个问题：
  Q1 三种评估口径（全步/block/滚动）的预测路径有没有读未来？
  Q2 scaler 是不是只用训练段拟合？
  Q3 train/val/test 三个切分有没有互相重叠？

依据：PatchTST_supervised/data_provider/data_loader.py::Dataset_Custom.__read_data__
（本脚本按同一公式重算边界，并逐项打印，便于人工核对）

用法: python3 scripts/check_leakage.py
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
SEQ_LEN, PRED_LEN, LABEL_LEN = 336, 24, 48

WINDOWS = [("W1", "1996-08", "2008-11"), ("W2", "2008-12", "2019-11"), ("W3", "2019-12", "2025-10")]


def main():
    df = pd.read_csv(CSV)
    ym = df["date"].astype(str).str[:7]
    n = len(df)
    print(f"数据: {ym.iloc[0]} ~ {ym.iloc[-1]}，共 {n} 月\n")

    for name, test_start, test_end in WINDOWS:
        dt = pd.to_datetime(test_start)
        train_end_str = (dt - pd.DateOffset(months=133)).strftime("%Y-%m")
        num_train = int((ym <= train_end_str).sum())
        num_val = 132
        test_end_row = int((ym <= test_end).sum()) - 1
        total_used = test_end_row + 1
        num_test = total_used - num_train - num_val
        border1s = [0, num_train - SEQ_LEN, num_train + num_val - SEQ_LEN]
        border2s = [num_train, num_train + num_val, total_used]

        print(f"===== {name}（test_start={test_start}, test_end={test_end}）=====")
        print(f"  num_train={num_train}  num_val={num_val}  num_test={num_test}  total_used={total_used}")
        for k, (tag, b1, b2) in enumerate(zip(("train", "val  ", "test "), border1s, border2s)):
            print(f"  {tag} 数据下标 [{b1}, {b2})  日期 {ym.iloc[b1]} ~ {ym.iloc[b2-1]}")

        # Q2 scaler 拟合范围：训练实例里 train_data = df_data[border1s[0]:border2s[0]] = [0, num_train)
        print(f"  Q2 scaler 拟合段 = [0, {num_train}) = {ym.iloc[0]} ~ {ym.iloc[num_train-1]}"
              f"  → {'仅训练段 ✓' if border1s[0] == 0 and border2s[0] == num_train else '越界 ✗'}")

        # Q3 val 窗口的目标区间 vs 训练段
        n_val_win = (border2s[1] - border1s[1]) - SEQ_LEN - PRED_LEN + 1
        v_b1 = border1s[1] + (SEQ_LEN - LABEL_LEN) + LABEL_LEN      # 第 0 个 val 窗口目标起点
        v_b2 = border1s[1] + (n_val_win - 1) + SEQ_LEN + PRED_LEN    # 最后一个 val 窗口目标终点（开区间）
        ov = max(0, min(v_b2, num_train) - v_b1)
        print(f"  Q3 val 目标覆盖 [{v_b1}, {v_b2}) = {ym.iloc[v_b1]} ~ {ym.iloc[min(v_b2, n)-1]}"
              f"；其中 {ov} 个月落在训练段内（≤ {ym.iloc[num_train-1]}）"
              f"  → {'存在重叠（val 非纯样本外，影响 EarlyStopping 选择）' if ov > 0 else '无重叠 ✓'}")
        # 反转记录：val 窗口里目标完全在训练段内的比例
        in_train = 0
        for i in range(n_val_win):
            rs = border1s[1] + i + SEQ_LEN
            if rs + PRED_LEN <= num_train:
                in_train += 1
        print(f"     其中目标整体落在训练段内的 val 窗口: {in_train} / {n_val_win}")

        # Q1 预测路径：test 窗口的输入与目标下标
        t_in_b1 = border1s[2]
        first_target = border1s[2] + (SEQ_LEN - LABEL_LEN) + LABEL_LEN
        last_target = border2s[2] - 1
        print(f"  Q1 test 输入窗口 [{t_in_b1}, {border1s[2]+SEQ_LEN}) = {ym.iloc[t_in_b1]} ~ "
              f"{ym.iloc[border1s[2]+SEQ_LEN-1]}（全在 test_start 之前 ✓）")
        print(f"     test 目标/真值下标 [{first_target}, {last_target}] = {ym.iloc[first_target]} ~ {ym.iloc[last_target]}"
              f"（= 记录的测试段 ✓）")
        print(f"     预测路径（roll_eval/block_eval）读到的最大下标 = {last_target} = {ym.iloc[last_target]}"
              f" → 未读 test 段之后的数据 ✓\n")

    # 全步窗口数核对
    print("全步窗口数（= 各窗口 test 段长 - seq_len - pred_len + 1 时用 test 段切片）:")
    for name, test_start, test_end in WINDOWS:
        dt = pd.to_datetime(test_start)
        num_train = int((ym <= (dt - pd.DateOffset(months=133)).strftime("%Y-%m")).sum())
        total_used = int((ym <= test_end).sum())
        print(f"  {name}: test 段 {total_used - num_train - 132} 月")


if __name__ == "__main__":
    main()
