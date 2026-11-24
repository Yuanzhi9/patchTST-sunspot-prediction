# R1B-NEXT：防泄漏审计与受控后续实验

> 计划执行日：2026-11-24
> 状态：审计已执行；未来污染检查标记，等待景修审核定性；下游实验未运行
> 主目标：验证 R1B 峰幅模型是否真正使用起报时可见信息，并判断 RF 是否继续保留

## 1. 当日必须完成

1. 复现当前 R1B-MULTI 的 raw/13m 基线；
2. 完成 forecast-origin 防泄漏审计；
3. 完成固定 W0 特征下的目标表示与模型对照；
4. 给出 RF 的保留/降级结论。

峰时、谷时和单特征消融只在上述门槛通过后执行。

## 2. 基线复现

必须复现：

```text
raw  W0-RF expanding MAE   = 44.70
13m  W0-RF expanding MAE   = 38.63
raw  W0-Ridge expanding MAE = 40.98
13m  W0-Ridge expanding MAE = 34.45
```

输入文件哈希、脚本版本和 seed 集合必须记录。无法复现时停止，不进入后续实验。

## 3. 防泄漏审计

### A. truncation

对每个预测起点 `t0`：

1. 用完整数据生成特征与预测；
2. 删除所有 `date > t0` 的行后重跑；
3. 两次特征和预测必须逐元素一致。

### B. future poisoning

把 `t0` 后的 SSN、aa 和派生列改成不可能值，把目标周期未来极值改成 `NaN`。
预测必须完全不变。

### C. 13m availability

13m 中心平滑在 `t` 时刻需要未来 6 个月数据。起报点 `t0` 的模型输入只允许使用
中心时间 `<= t0 - 6m` 的完整 13m 值。目标评价仍可用完整 13m 真值。

### D. feature manifest

为每个特征记录：

```text
feature
source columns
calculation window
last_available_month
lag_months
```

任何 `last_available_month > origin` 的特征直接判泄漏。

### E. target blindness

测试行的 `target_peak_raw`、`target_peak_smooth`、`target_peak_date`、
`target_months_to_peak`、`target_min_date` 和 `target_months_to_min` 只能作为
标签；将其置为 `NaN` 后不得改变同一起报点的预测。

五项全部通过，才允许运行第 4 节。

## 4. 第一轮受控实验

### 数据与切分

- 周期级峰幅：预测下一周期 raw/13m 峰；
- 主验证：目标周期 SC19–SC24 的时间顺序扩测；
- 辅助验证：24 周期 LOOCV；
- 外层测试周期不参与模型或超参数选择。

### A. 模型和目标表示

固定 `W0` 特征：

```text
current peak
previous peak
decline_pct_24m
cycle_parity
previous_cycle_length
```

每个模型只改变一个变量：

| ID | 模型 | 目标 |
|---|---|---|
| A0 | 训练折内气候中位数 | raw/13m |
| A1 | Ridge | log1p(peak) |
| A2 | RF | log1p(peak) |
| A3 | RF | raw peak |
| A4 | RF | next_peak/current_peak |

RF 使用固定、较小的候选网格；网格在训练折内部选择。不得看见外层测试周期后调整。

### B. 单特征消融

仅在固定最佳模型上执行：

```text
W0
W0 + rise_months
W0 + postpeak_abs_slope_24m
W0 + postpeak_volatility_24m
W0 + elapsed_since_min
```

一次只加一个，不允许先把全部 W1 特征一起加入。

aa 只在同覆盖期对照中评估；不因数据名称不同而改变结论口径。

## 5. 条件执行

仅在 A/B 通过后运行：

- 峰时：`target_months_to_peak`；
- 谷时：`target_months_to_min`；
- 对照：气候中位数、Ridge、RF；
- 指标：MAE、偏差、±6 月命中率、逐周期方向。

时间模型不得使用峰幅或曲线 MAE 的改善作为通过理由。

## 6. 有效性门槛

峰幅候选必须同时满足：

1. 泄漏审计五项全部通过；
2. raw 或 13m 至少一个口径相对同折气候中位数改善 ≥10%；
3. 目标 SC19–SC24 中至少 5/6 周期改善；
4. 任一周期相对同折气候中位数的误差恶化 ≤20%；
5. 五 seed 方向不能由单个 seed 主导；
6. 高值周期不能靠统一压低换取平均改善。

时间模型必须同时满足：

1. MAE 优于同一训练折中位数；
2. ±6 月命中率提高；
3. 主峰幅度和曲线轮廓不得因此恶化。

## 7. 止损

- 当前基线复现失败：停止；
- truncation 或 poisoning 失败：停止并修数据管线；
- 目标表示实验不优于气候中位数：淘汰；
- 只有 LOOCV 好、时间顺序差：淘汰；
- 峰幅改善但高值周期系统性恶化：不进入 R1A；
- 当天无法完成时间模型：记录为未执行，不用弱结果硬补。

## 8. 交付物

- `analysis/r1b_next/leakage_audit.csv`
- `analysis/r1b_next/target_representation_summary.csv`
- `analysis/r1b_next/feature_ablation_summary.csv`
- `analysis/r1b_next/selected_predictions.csv`
- `analysis/r1b_next/r1b_next_comparison.png/.pdf`
- `EXPERIMENT_B26M-R1B-NEXT_2026-11-24.md`
- 更新 `result.txt`、`experiment_history.md`、`STATE.md` 和日记

## 9. 当日结论格式

只允许以下三类结论：

1. **保留**：通过全部门槛，进入 R1A 候选；
2. **降为对照**：有弱信号但未过稳定性门槛；
3. **停止**：不优于气候中位数或存在泄漏。

不得用“看起来更好”“有一定潜力”作为通过结论。
