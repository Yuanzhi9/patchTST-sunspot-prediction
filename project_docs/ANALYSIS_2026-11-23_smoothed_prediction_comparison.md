# raw 与 13 月平滑尺度预测对照

> 日期：2026-11-23  
> 性质：分析任务，无重新训练  
> 目的：检查 R1-A 和未来 PR24 曲线在官方 13 月平滑尺度上的表现

## 方法

- R1-A：对已有真值、PR24 和 R1-A 局部修正序列分别做 13 月中心平滑，
  只比较两端都有完整 13 月窗口的月份。
- 未来 PR24：在“历史观测 + 单 seed 未来预测”的拼接序列上做 13 月中心平滑；
  峰谷位置直接复用既有 `forecast_extrema_by_seed.csv`，不从平滑后的中位数轨迹
  重新定义极值。

该平滑是预测后处理与评价口径，不是“用 13m 序列重新训练模型”。

## R1-A 平滑结果

| 周期 | PR24 13m MAE | R1-A Oracle 13m MAE | 原 raw MAE 改善 |
|---|---:|---:|---:|
| SC20 | 41.719 | **24.368** | +28.81% |
| SC21 | 23.096 | **20.592** | +5.01% |
| SC24 | 11.012 | 11.015 | +0.11% |

SC20 在平滑空间仍有明显收益；SC21 有中等收益；SC24 没有实质变化。
SC24 的原因不是平滑后收益消失，而是原 PR24 峰幅已经几乎正确，
局部峰修正只能改变不到 1 SSN。

图：

- `analysis/smoothed_comparison/r1a_smoothed_profiles.png`
- `analysis/smoothed_comparison/r1a_smoothed_profiles.pdf`

## 未来 PR24 平滑结果

从 2026-10 起报：

- Cycle 25 谷时中位：2030-10；
- 谷值中位：约 0.12；
- Cycle 26 峰时中位：2035-06；
- 13m 峰值中位：104.82；
- 五 seed 峰值范围：101.68–108.58。

这些数值与此前报告一致。平滑后的图主要用于与官方周期口径和文献图比较，
不改变“峰幅偏低”的结论。

图：

- `analysis/smoothed_comparison/pr24_future_smoothed_profile.png`
- `analysis/smoothed_comparison/pr24_future_smoothed_profile.pdf`

## 解释边界

1. R1-A 仍是 Oracle 条件：使用真实峰幅和真实峰位。
2. 未来曲线平滑使用模型预测作为未观测月份的输入，是条件平滑，不是官方观测值。
3. 五 seed 带不是概率置信区间。
4. SC24 的“没区别”是真实结果：五个 seed 中 3 个极小幅改善、2 个极小幅恶化，
   平均 MAE 只差约 0.003。

## 复现

```bash
python3 scripts/plot_smoothed_predictions_2026-11-23.py
```
