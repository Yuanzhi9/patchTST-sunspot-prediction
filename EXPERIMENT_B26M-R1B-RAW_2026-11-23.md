# B26M R1B-RAW — raw 月均峰值局部重构

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-R1B-RAW` |
| 类型 | 零训练分析 |
| 目的 | 以 raw 月均峰值为目标，验证 Ridge/RF 峰幅经 R1-A 局部核传播后的效果 |
| 输入 | PR24 recursive block24 五 seed；SC20/21/24 |
| 目标 | `target_peak_raw`，不是 13 月平滑峰值 |
| 对照 | PR24 clip0；R1-A Oracle(13m 目标)；smooth-target Ridge/RF；raw-target Ridge/RF |
| 训练入口 | 无 |
| 坑清单回顾 | raw/13m 混用、周期上下文 LOOCV 非严格时间顺序、峰时未修正 |

## 方法

局部修正核、中心和 H=24 月与 R1-A 相同，只改变幅度目标：

```text
raw 模型：把 raw 月均峰值校正到 predicted_raw_peak
```

raw 峰值预测来自周期上下文 reduced Ridge/RF 的 `target_peak_raw` LOOCV。
同时保留此前 smooth-target 结果，明确展示两种口径差异。

## 通过门槛

1. 至少 2/3 周期 raw 峰幅误差下降；
2. 任一周期曲线 MAE 恶化不超过 10%；
3. 至少 2/3 周期 raw 峰值时间不恶化；
4. 至少两个改善周期中各有 4/5 seed 同方向；
5. 同时报告 13m 峰幅作为次口径。

## 预尸检

1. raw 峰值更容易受单月尖峰影响，局部核可能把峰变得过尖。
2. SC24 原 PR24 raw 峰低约 28.6，但 13m 峰已接近真值；提高 raw 峰值可能恶化平滑峰。
3. Ridge/RF 的 raw 目标预测可能把 SC24 推到 200 左右，导致大幅过调。
4. 峰时仍不由幅值核修正。
5. LOOCV 不是严格未来盲推，只能作为机制筛查。

## 产物

- `analysis/r1b_raw_peak/raw_peak_seed_detail.csv`
- `analysis/r1b_raw_peak/raw_peak_aggregate.csv`
- `analysis/r1b_raw_peak/raw_peak_decision.json`
- `analysis/r1b_raw_peak/r1b_raw_profiles.png/.pdf`
- `analysis/r1b_raw_peak/r1b_raw_errors.png/.pdf`
- `logs/r1b_raw_peak.log`

## 结果

五 seed、H=24：

| 周期 | 方法 | raw 峰幅误差 | 13m 峰幅误差 | 曲线 MAE |
|---|---|---:|---:|---:|
| SC20 | PR24 | -117.45 | -88.80 | 42.392 |
| SC20 | Oracle raw | **0** | +21.23 | **30.313** |
| SC20 | Ridge raw | -53.23 | -28.77 | 32.379 |
| SC20 | RF raw | -47.82 | -23.69 | 31.761 |
| SC21 | PR24 | -41.87 | +20.03 | **28.198** |
| SC21 | Oracle raw | **0** | +59.19 | 32.856 |
| SC21 | Ridge raw | -51.27 | +11.33 | 27.392 |
| SC21 | RF raw | -30.98 | +30.16 | 28.979 |
| SC24 | PR24 | -28.60 | -0.65 | **14.452** |
| SC24 | Oracle raw | **0** | +25.55 | 17.909 |
| SC24 | Ridge raw | +54.70 | +76.72 | 27.165 |
| SC24 | RF raw | +62.08 | +83.63 | 28.453 |

Raw 目标下的 Oracle 可以把 raw 峰幅精确命中，但会改变峰宽和附近轮廓：
SC21/SC24 的曲线 MAE 分别恶化 16.5%/23.9%。

Ridge-raw/RF-raw 的预测峰幅：SC20 约 214/219（改善），SC21 约
233/254（仍低或接近），SC24 约 201/208（真值 146，严重高估）。

## 结论

1. 你的判断在“现有 smooth-target Ridge 的 raw 峰结果”上成立：SC24 raw 峰从
   117.5 抬到 144.9，几乎命中真值 146.1。
2. 但这不是一个统一的 raw 峰值模型。把周期模型目标改成 raw 后，Ridge/RF
   对 SC24 预测为 200.8/208.2，反而严重高估。
3. Raw 峰幅提升能救强周期 SC20，但不能跨强、中、弱周期稳定工作。
4. 峰时没有变化：SC20 仍晚约 8 月，SC21 晚约 11 月，SC24 早约 5 月。
5. R1-B 系列值得继续。RF 保留为峰幅候选，但必须加入保守修正门，
   不能在没有跨周期稳定性证据时直接扩大使用。

图：`analysis/r1b_raw_peak/r1b_raw_profiles.png`、
`analysis/r1b_raw_peak/r1b_raw_errors.png`。
