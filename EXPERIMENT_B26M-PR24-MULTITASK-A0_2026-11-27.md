# B26M PR24 多任务极值 A0：标签审计

> 日期：2026-11-27  
> 分支：`exp/pr24-multitask-extrema`  
> 性质：零训练标签审计  
> 状态：PASS

## 1. 目的

为 PR24 多任务辅助头建立逐窗口周期事件标签，确认：

- 下一次周期主峰的 raw 峰幅；
- 从预测起点到下一次 13m 主峰的剩余月数；
- 从预测起点到下一次 13m 谷的剩余月数；
- 当前周期长度；
- 标签缺失时的 mask；
- 标签不进入模型输入。

## 2. 实现

脚本：

```text
scripts/audit_pr24_event_labels.py
```

输入：

- raw SSN：`sunspot_with_cycle_monthly_2026-10.csv`
- 13m 数据：`sunspot_with_cycle_smoothed13_2026-04.csv`
- 周期边界：`analysis/cycle_context/cycle_boundaries_2026-10.csv`
- CV5：`configs/cv5_folds_2026-10.json`

输出：

```text
analysis/pr24_multitask_extrema/a0_labels/
  event_labels_all_windows.csv
  coverage_by_fold.csv
  audit_summary.json
```

## 3. 关键结果

| 数据集 | 窗口数 | 峰标签覆盖 | 谷标签覆盖 | 主峰在 24m 内 | 谷在 24m 内 |
|---|---:|---:|---:|---:|---:|
| train | 7596 | 100% | 100% | 18.64% | 18.17% |
| val | 2180 | 99.63% | 88.99% | 11.38% | 21.10% |
| test | 3219 | 99.94% | 98.14% | 16.62% | 17.86% |

训练窗口的峰、谷标签完整；验证和测试端点的缺失来自 Cycle 25 后事件尚未确定，
使用 mask，不填零。

周期事件只有约 18% 落在单个 24 月窗口内。因此辅助头目标是
`months_to_next_peak` 和 `months_to_next_trough`，不是 24 月窗口自身的
`argmax/argmin`。

## 4. 校验

- raw 峰幅逐周期复核为周期区间内的 raw 最大值；
- 时间标签按年月差计算，没有把日差除以 30 近似；
- 训练窗口的峰、谷标签覆盖均为 100%；
- 输入列固定为 `month_sin`、`month_cos`、`ssn`，事件标签不在输入列；
- 输入与输出 SHA256 已写入 `audit_summary.json`；
- 原 PR24 数据和训练路径未修改。

## 5. 已知口径说明

`cycle_boundaries` 的 `peak_smooth` 由项目既有的居中 13 月滚动平滑生成，
而 `sunspot_with_cycle_smoothed13_2026-04.csv` 是另一个 13m 数据文件，两者
平滑值有约 0–6 SSN 差异。本实验只使用周期边界给出的峰谷月份，峰幅使用 raw；
不使用两个 13m 文件的平滑幅度差做标签。

## 6. 结论

A0 通过。可以进入 A1 基线复核和 A2 峰幅单头。
