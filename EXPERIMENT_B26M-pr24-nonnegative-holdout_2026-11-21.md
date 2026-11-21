# B26M PR24 非负基线能力检验

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-21 |
| 实验 ID | `B26M-PR24-NONNEG-HOLDOUT` |
| 类型 | 分析任务；复用三完整周期、五 seed 的独立重训 checkpoint 与预测曲线 |
| 目的 | 在正式非负口径下检验 PR24 能否承担完整周期曲线、峰值/时间、谷值/时间预测 |
| 正式模型 | `PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32` |
| 数据 | SILSO V2.0 月均 raw SSN，1749-01–2026-10 |
| 时间留出 | SC20、SC21、SC24，作为 W1/W2/W3 类三周期检验 |
| 种子 | 2019、2020、2021、2022、2023 |
| 主方法 | PR24 recursive block24；不使用测试期任何真实 SSN |
| 参照 | PR24 observed-update block24；使用测试期真实观测，只作业务更新参照 |
| 非负门禁 | 物理反变换后 `clip0 = max(pred, 0)`；raw 同时保留 |

## 可证伪假设

1. 若 PR24 recursive 已达到可用完整周期基线，三个周期的峰时或谷时至少两个达到
   `±6 月`，且峰幅误差方向与量级不应出现强周期系统性崩溃。
2. 若曲线上相关高但峰谷幅度/时间不稳定，则 PR24 只能作为轮廓候选，不能直接承担
   Cycle 25 谷值和 Cycle 26 峰值点预测。
3. 若模型相对 persistence 的技巧分为负，则递归基线没有业务价值；若为正但峰谷失败，
   则平均误差掩盖了关键任务失败。

## 预尸检

1. 13 月平滑只用于峰谷定义，曲线指标使用 raw 月均。
2. raw 与 clip0 的总体 MAE 差异可能很小；非负门禁主要保证物理合法性，不负责提高模型。
3. observed-update 每 24 月使用真实观测，不能与 recursive 的同分数直接比较。
4. 三个周期的振幅差异很大，不能把 132 个月混合 MAE 当作峰值能力。
5. 五个 seed 只描述训练随机性，不是跨周期总体不确定性的充分样本。

## 成功门槛

- 数据门禁：三周期 × 五 seed × 两方法 × 两口径全部成功计算，无 NaN/Inf。
- 非负门禁：clip0 版本负值月数必须为 0。
- 极值门禁：同时报告峰谷幅度、相对幅度、月份误差、3/6/12 月命中率。
- 稳健性：报告五 seed 均值、标准差、中位数、最小/最大值和均值 bootstrap 95% CI。
- 整体能力：报告 MAE、RMSE、bias、R²、Pearson、Spearman、NRMSE、
  相对 persistence 技巧分和集合中位数经验区间覆盖率。

## 结果

PR24 recursive block24 的正式 clip0 口径：

| 周期 | MAE | RMSE | R² | Pearson r | Persistence skill | 峰幅误差 | 峰时绝对误差 | 谷幅误差 | 谷时绝对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SC20 | 42.39 ± 1.02 | 54.93 ± 1.11 | 0.562 | 0.939 | +0.176 | -88.80 ± 1.76 | 4.8 月 | +15.20 ± 4.18 | 6.0 月 |
| SC21 | 28.20 ± 1.86 | 34.94 ± 2.15 | 0.806 | 0.925 | +0.306 | +20.03 ± 9.92 | 10.8 月 | +3.51 ± 5.60 | 0.8 月 |
| SC24 | 14.45 ± 0.54 | 19.13 ± 0.53 | 0.784 | 0.913 | +0.579 | -0.65 ± 2.43 | 8.0 月 | +2.51 ± 2.08 | 0.0 月 |

峰谷 6 月命中率：

| 周期 | 峰时 ≤6 月 | 谷时 ≤6 月 |
|---|---:|---:|
| SC20 | 3/5 seed | 3/5 seed |
| SC21 | 1/5 seed | 5/5 seed |
| SC24 | 1/5 seed | 5/5 seed |

完整结论见 `project_docs/REPORT_2026-11-21_PR24_nonnegative_baseline.md`。

## 结论

1. `clip0` 把 recursive 基线的负值月数从 SC21 `1.6 ± 2.1`、SC24 `3.0 ± 3.2`
   统一降为 0，正式物理门禁成立。
2. 曲线平均相关较高、对 persistence 技巧分为正，说明 PR24 能学到周期轮廓。
3. 峰时只在 SC20 五 seed 中 3 个命中 ±6 月；SC21/SC24 各仅 1 个，不能称为稳定的
   峰值时间模型。
4. 谷时在 SC21/SC24 全部命中，SC20 为 3/5；谷值时间明显优于峰值时间。
5. SC20 强周期峰幅低估 88.8，SC21 高估 20.0，SC24 基本正确；跨周期方向不稳定，
   当前不能直接承担 Cycle 26 峰值点预测。

## 证据文件

- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_seed_metrics.csv`
- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_seed_extrema.csv`
- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_metric_aggregate.csv`
- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_extrema_aggregate.csv`
- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_ensemble_metrics.csv`
- `analysis/pr24_nonnegative_holdout/pr24_nonnegative_leadtime_aggregate.csv`
- `analysis/pr24_nonnegative_holdout/*.png`
- `analysis/pr24_nonnegative_holdout/*.pdf`
- `logs/pr24_nonnegative_holdout_2026-11-21.log`
