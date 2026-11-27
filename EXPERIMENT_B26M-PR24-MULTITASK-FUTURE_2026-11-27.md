# PR24 多任务真实未来预测与 R1A/R1B 峰值修正

> 日期：2026-11-27
>
> 分支：`exp/pr24-multitask-extrema`
>
> 模型：PR24 曲线 + 峰幅 + 峰时 + 谷时辅助头，固定权重 0.1
>
> 性质：全历史重训后的真实未来外推；SC26 尚无真值，不作精度评价

## 1. 训练口径

两条数据线完全分开：

| 目标线 | 目标数据 | 起报点 | 预测期限 |
|---|---|---|---:|
| raw | SILSO raw 月均 SSN | 2026-10 | 168 月 |
| 13m | 官方 13 月中心平滑 SSN | 2026-04 | 168 月 |

13m 中心平滑末端只到 2026-04，因此不能与 raw 共用 2026-10 起报点。
旧的“13m 轮廓”是 raw 未来曲线再做 13m 平滑，只作为历史对照；新的 13m
模型是独立训练目标。

每条线五 seed，固定 50 epoch，无验证集早停，正式输出 clip0。

## 2. 新多任务真实预测

| 目标线 | 中位曲线峰值 | 中位曲线峰时 | 逐 seed 事件峰幅中位 | 逐 seed 事件峰时中位 |
|---|---:|---|---:|---|
| raw | 137.2 | 2035-01 | raw 133.3 / 13m 133.8 | 2034-05 |
| 13m | 153.9 | 2034-02 | 13m 155.0 | 2034-07 |

逐 seed 事件法的 raw 峰时范围是 2034-01–2035-07、raw 峰幅范围
128.8–152.3；13m 峰时范围是 2032-12–2036-02、13m 峰幅范围
106.9–207.6。raw 线明显高于旧 PR24 的 106.7–107.2；13m 线中心值更高，
但 seed 范围非常宽，不能视为已经稳定的点预测。

## 3. 旧 PR24 未来轮廓的峰值修正

旧 raw 和旧 13m 轮廓都保留原来的 2026-10 起报点。

| 轮廓 | 未修正峰幅 | R1A+R1B Ridge | R1A+R1B RF |
|---|---:|---:|---:|
| old raw | 107.18 | 205.85 | 190.04 |
| old 13m | 104.82 | 139.84 | 127.00 |

R1A 分别尝试 H=12/24/36 月。旧轮廓的所有组合都收敛，峰时位置不变；
H 越小，修正越像窄峰尖；H 越大，峰区被抬得更宽。

## 4. 多任务未来轮廓的峰值修正

| 轮廓 | 未修正峰幅 | R1A+R1B Ridge H24 | R1A+R1B RF H24 |
|---|---:|---:|---:|
| multitask raw | 137.22 | 205.85 | 190.04 |
| multitask 13m | 153.88 | 139.84 | 127.00 |

关键差别：

- raw 多任务曲线低于 R1B 峰幅目标，因此修正是向上拔高；
- 13m 多任务曲线已经高于 R1B 峰幅目标，因此修正是向下压，不是“拔高”；
- 13m 曲线的峰区较平，H=12 无法在保持主峰位置的前提下达到目标，R1A H12 失败；
- H=24/36 可以达到目标，但会把峰时向后移动约 8–15 月。

这说明当前 13m 多任务轮廓与 R1B 的周期幅度口径存在明显张力；不能把两种
13m 曲线直接平均，也不能用 raw 峰幅目标去修 13m 轮廓。

## 5. 图的完整文件名

```text
analysis/pr24_multitask_future_correction/mt_real_forecast_raw_and_13m.png
analysis/pr24_multitask_future_correction/mt_real_forecast_raw_and_13m.pdf
analysis/pr24_multitask_future_correction/old_profiles_with_R1A_R1B.png
analysis/pr24_multitask_future_correction/old_profiles_with_R1A_R1B.pdf
analysis/pr24_multitask_future_correction/mt_profiles_with_R1A_R1B.png
analysis/pr24_multitask_future_correction/mt_profiles_with_R1A_R1B.pdf
```

数据：

```text
analysis/pr24_multitask_future_correction/future_peak_comparison.csv
analysis/pr24_multitask_future_correction/old_raw_with_R1A_R1B.csv
analysis/pr24_multitask_future_correction/old_smooth13_with_R1A_R1B.csv
analysis/pr24_multitask_future_correction/mt_raw_with_R1A_R1B.csv
analysis/pr24_multitask_future_correction/mt_smooth13_with_R1A_R1B.csv
analysis/pr24_multitask_future_correction/old_R1A_R1B_metadata.csv
analysis/pr24_multitask_future_correction/mt_R1A_R1B_metadata.csv
```

## 6. 解释边界

- SC26 真值未知，新曲线只能作为情景预测；
- R1A Oracle 在真实未来不可用，因为没有 SC26 真峰幅；
- R1A 图展示的是 R1A 局部核在不同带宽下，如何使用 R1B 预测峰幅；
- 不能根据这组未来图选择“最好方法”，只能比较假设和形状；
- 13m 起报点与 raw 相差 6 个月，不能把两条未来曲线的峰时差值直接解释为模型优劣。
