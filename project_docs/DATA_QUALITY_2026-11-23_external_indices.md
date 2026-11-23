# 外部太阳活动指数数据审计（2026-11-23）

> 范围：aa、F10.7 与 WSO 极区磁场。
> 原则：保留原始文件；缺测不伪造；处理表同时保存 coverage、missing 和来源。

## 1. 当前数据状态

| 数据 | 本地文件 | 有效范围 | 覆盖情况 |
|---|---|---|---|
| aa 3 小时值 | `PatchTST_supervised/dataset/external/raw/aa/aa_1869-01-01_2026-11-23_IAGA2002.dat` | 1869-01-01–2026-11-23 | ISGI IAGA2002-like；999 为缺测 |
| aa 日值 | `PatchTST_supervised/dataset/external/processed/aa_daily_1869_2026.csv` | 57,670 天 | 日均、日最大、3小时记录数、coverage |
| aa 月均 | `PatchTST_supervised/dataset/external/processed/aa_monthly_1869_2026.csv` | 1869-01–2026-10 | 1,894 个有效月；仅 2026-11 因记录不完整缺失 |
| F10.7 月均 | `PatchTST_supervised/dataset/external/processed/f107_monthly_1947_2026.csv` | 1947-02–2026-10 | 957 个月，3 个已知缺月 |
| F10.7 原始日值 | `PatchTST_supervised/dataset/external/raw/f107/` | 2015–2026 日文件 | 1947–2015 使用 NCEI 月均文件；2015–2026 由日文件聚合 |
| WSO 极区场 10 日值 | `PatchTST_supervised/dataset/external/processed/wso_polar_10day_1976_2026.csv` | 1976-05-31–2026-07-18 的记录范围 | 有效值最后到 2026-01-09 |
| WSO 极区场月均 | `PatchTST_supervised/dataset/external/processed/wso_polar_monthly_1976_2026.csv` | 1976-05–2026-10 | 有效月最后到 2025-12 |

F10.7 缺少的 3 个月是 2026-07、2026-08、2026-09；2026-10 月值由 NOAA
SWPC 月表补充。三个月连续缺口不插值。

## 2. 来源与口径

### aa

- 来源：ISGI，IAGA-2002-like 3 小时值。
- 官方缺测码 999 转为 NaN。
- 按 3 小时记录生成日均和月均，同时保留 `aa_max` 与 `aa_p90`。
- 月覆盖率低于 80% 时，月均值、月最大值和 P90 均记为缺失。
- 长缺口不插值，并保留 missing 标记。
- 上传文件的原始 SHA256：
  `2eed23a8b5d812d2a6516cbe7aa8b193f36ddb15b2b3388a4173d1486c082e76`。

### F10.7

- 历史月均：NOAA NCEI Penticton/Ottawa noontime-flux archive。
- 日值：同一 NCEI archive 的 `pent_noontime-flux_YYYY.txt`。
- 近期月值：NOAA SWPC `observed-solar-cycle-indices.json`。
- NCEI 说明：1947 年 Ottawa 开始，1991-06 后转 Penticton；`Obsflux` 是
  实测通量，`Adjflux` 校正到 1 AU，`Absflux` 是绝对校准版本。
- 本轮合并表优先保留 `f107_observed`；不能把 observed/adjusted/absolute
  三种口径混成一个序列。

### 极区磁场

- WSO `Polar.html`，每 10 天给出北、南和平均极区场，包含原始值与
  20 nHz filtered 值。
- 2026-01-09 之后记录为 `XXX`；因此当前 2026-10 起报点不能用 WSO 作为
  完整输入。
- 极性反转是物理过程，不能把带符号场直接相加。后续同时保留：
  - `polar_north_filtered`
  - `polar_south_filtered`
  - `polar_average_filtered`

## 3. 清洗规则

1. `XXXX`、`XXX`、空白和负占位符转为 NaN。
2. 日期统一到月初/记录日，检查单调性和重复日期。
3. F10.7 日值聚合月均时记录有效天数；覆盖率低于 80% 的月份不生成均值。
4. 只有连续且两侧有观测的 ≤2 个月缺口允许线性插值；
   3 个月及以上缺口全部保留 NaN。
5. aa 原始缺测码 999 必须转为 NaN，不参与均值、最大值或分位数。
6. 极区磁场按 10 日记录聚合成月；少于 2 条有效记录时月值为 NaN。
7. 建模时必须带 missing mask；不能用 0 表示未知。
8. 每个变量独立标准化，scaler 只在训练段拟合。
9. 中心平滑只能用于历史标签或回顾性分析；当前端点缺失时不得偷偷回填未来。

## 4. 初步统计

在原始月均 SSN 上：

- F10.7 observed 与同期 SSN 的 Pearson 相关约 0.973，Spearman 约 0.981。
- WSO 极区平均场的带符号月值只用于存储诊断；其与同期 SSN 的直接相关约
  -0.08，不能按普通活动指数解释。
- aa 月均与同期 raw SSN 的 Pearson 相关约 0.431，Spearman 约 0.476；
  这是同期活动响应，不是“下一周期峰值”的前兆验证。
- F10.7 的增益需要通过同训练期控制实验判断，不能因为单变量相关高就认定它能
  改善跨周期峰幅预测。

## 5. 当前 Cycle 25 起报点

- 2026-10 F10.7 observed：108.47 sfu。
- 2026-10 aa：月均 20.98 nT，月最大 194 nT，记录覆盖率 1.0。
- WSO 极区场在 2026-10 没有有效月值。
- 因此：
  - F10.7 可做当前端点的覆盖期控制实验；
  - WSO 极区场只能先用历史周期做能力审计，或等待其他台站交叉校准数据；
  - aa 已可用于当前端点的覆盖期对照，但与 F10.7 分开做单变量实验，
    不先做多变量堆叠。

## 6. 建模边界

当前 PatchTST 的 channel-independent 输出不会自动读取新增输入通道。因此：

- 直接把 `f107_observed` 或 `polar_average_filtered` 加到 CSV，不能保证 SSN
  输出会使用它们；
- 外部月序列需要走显式 fusion；
- 周期级 scalar 特征需要单独编码，再注入形状/振幅/时间头；
- 在 fusion 完成前，只做覆盖期相同的数据审计和传统周期级模型。

## 7. 复现命令

```bash
python3 scripts/build_external_indices.py
python3 scripts/merge_external_features.py
python3 scripts/build_cycle_context.py
python3 scripts/evaluate_cycle_context_models.py
```
