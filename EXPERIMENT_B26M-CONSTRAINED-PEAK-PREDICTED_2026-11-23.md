# B26M R1-B — 周期级预测峰幅驱动局部重构

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-CONSTRAINED-PEAK-PREDICTED` |
| 类型 | 零训练；把 R1-A Oracle 峰幅换成周期级模型预测 |
| 目的 | 检验 R1-A 的局部重构收益在真实周期特征条件下还剩多少 |
| 输入 | PR24 recursive block24 五 seed；R1-A H=24 修正器 |
| 峰幅来源 | 周期上下文 reduced Ridge；reduced Random Forest；LOOCV |
| 周期映射 | R1 的 SC20/SC21/SC24 对应周期上下文的 cycle 20/21/23 |
| 对照 | PR24 clip0；R1-A Oracle；Ridge；Random Forest |
| 训练入口 | 无 |
| 坑清单回顾 | 周期上下文与 PR24 周期命名错位、LOOCV 非严格时间顺序、弱周期峰幅已接近真值 |

## 方法

R1-A 的局部修正器保持不变：

- 核宽固定 `H=24`；
- 核中心为 PR24 当前预测峰；
- 谷区权重为 0；
- 只替换目标峰值来源。

三个版本：

1. Oracle：真实 13m 峰幅；
2. Ridge：周期上下文 reduced Ridge 的 LOOCV 峰幅；
3. RF：reduced Random Forest 的 LOOCV 峰幅。

本实验回答：周期级模型预测误差通过 R1-A 传播后，是否仍比 PR24 好。

## 通过门槛

任一模型必须同时满足：

1. 至少 2/3 周期曲线 MAE 改善 5%；
2. 任一周期 MAE 恶化不超过 10%；
3. 至少 2/3 周期峰幅误差下降；
4. 谷幅和谷时保持不变；
5. 至少 2/3 周期、至少 4/5 seed 同方向改善。

若 Ridge/RF 都失败：

- R1-A 只保留为 Oracle 上界；
- 不用峰幅模型驱动正式重构；
- 转到条件形状/历史相似轨迹，或独立做 scheduled sampling。

## 预尸检

1. 周期上下文 LOOCV 不是严格未来盲推，可能比真实未来乐观。
2. Ridge/RF 对峰幅的跨周期误差仍很大。
3. SC24 的原 PR24 峰幅已接近真值，错误预测会把好结果改坏。
4. 以当前预测峰时为中心会限制峰时收益，本轮不评价峰时重构。
5. 如果 Ridge/RF 方向相反，不能选历史结果更好的一个冒充稳定方案。

## 产物

- `analysis/r1b_constrained_peak_predicted/seed_detail.csv`
- `analysis/r1b_constrained_peak_predicted/aggregate.csv`
- `analysis/r1b_constrained_peak_predicted/decision.json`
- `analysis/r1b_constrained_peak_predicted/r1b_profiles.png/.pdf`
- `analysis/r1b_constrained_peak_predicted/r1b_mae.png/.pdf`
- `logs/r1b_constrained_peak_predicted.log`

## 结果

周期级峰幅预测值：

| 周期 | 真值 13m 峰 | Ridge | Random Forest |
|---|---:|---:|---:|
| SC20 | 231.708 | 160.779 | 146.225 |
| SC21 | 212.769 | 181.422 | 183.255 |
| SC24 | 115.331 | 141.333 | 154.399 |

R1-A `H=24` 应用到预测峰幅后：

| 周期 | PR24 MAE | Oracle | Ridge | Random Forest |
|---|---:|---:|---:|---:|
| SC20 | 42.392 | **30.181** | **39.037** | 41.758 |
| SC21 | 28.198 | **26.784** | 28.179 | 27.771 |
| SC24 | 14.452 | **14.436** | 17.951 | 20.134 |

Ridge 相对 PR24 的改善为 `+7.92% / +0.07% / -24.21%`；RF 为
`+1.50% / +1.51% / -39.31%`。两者都没有做到 2/3 周期改善 5%，
且均使 SC24 超过 10% 安全线。

Oracle 仍通过全部门槛，`deployable_passed=false`。

## 结论

原结论按 13 月平滑峰值评价。后续核对发现 SC24 的 raw 月均峰值为 146.1，
而 13m 峰值为 115.331；Ridge 预测的 141.333 接近 raw，却高于 13m。

更正后的结论：

1. 在 13m 口径下，Ridge/RF 均不合格，SC24 分别高估 26.00/39.07。
2. 在 raw 月均峰值口径下，Ridge 对 SC24 的误差仅 -1.16，用户指出的“峰值其实
   差不多高，主要错在时间”成立。
3. 但 Ridge/RF 在 SC20/SC21 的 raw 峰值仍严重偏低，当前幅度模型不能整体采用。
4. 峰时没有被本轮局部修正改变，SC24 仍提前约 5–8 月。
5. raw/13m 已改为长期双轨；不能只写“全面失败”，也不能把两个口径平均。

分口径复核：`project_docs/ANALYSIS_2026-11-23_peak_definition_raw_vs_smoothed.md`。

图：`analysis/r1b_constrained_peak_predicted/r1b_mae.png`。
