# SILSO 官方数据快照

下载时间：2026-11-19 03:22 UTC

官方来源：WDC-SILSO, Royal Observatory of Belgium, Brussels  
数据页面：https://www.sidc.be/SILSO/datafiles  
DOI：https://doi.org/10.24414/qnza-ac80  
许可：CC BY-NC 4.0

## 文件

| 文件 | 内容 | 有效范围 | 行数 |
|---|---|---|---|
| `SN_m_tot_V2.0_through_2026-10.txt` | Monthly mean total sunspot number，未平滑 | 1749-01 至 2026-10 | 3334 |
| `SN_ms_tot_V2.0_source_through_2026-10.txt` | 官方 13-month smoothed monthly total sunspot number，源文件 | 源文件覆盖至 2026-10；有效平滑值至 2026-04 | 3334 |
| `manifest-paired_2026-11-19.sha256` | 下载当日的 SILSO 官方 SHA-256 manifest | — | — |

两个官方 TXT 的 SHA-256 与 manifest 完全一致：

```text
a34b6b671e6465b266f080d4b87ac13c04aab6aab45a158dd5352ceab9e5304c  SN_m_tot_V2.0.txt
c735c79921abfc68095dba842f306bb5eebf705617747615a34d44c003e80aa5  SN_ms_tot_V2.0.txt
```

## 官方平滑定义

SILSO 的 13-month smoothed series 是**居中的 tapered-boxcar running mean**，不是 Gaussian：

```text
R_smooth(t) = [0.5 R(t-6) + R(t-5) + ... + R(t+5) + 0.5 R(t+6)] / 12
```

即中间 11 个月权重为 1，两端 `t-6`、`t+6` 权重为 0.5，再除以 12。
因为使用居中窗口，序列开头和末尾各 6 个月没有平滑值，官方文件中记为 `-1`。

## 派生训练文件

| 文件 | 用途 | 行数 |
|---|---|---|
| `../sunspot_with_cycle_monthly_2026-10.csv` | 未平滑月均目标，保留原项目特征列 | 3334 |
| `../sunspot_with_cycle_smoothed13_2026-04.csv` | 官方 13 月平滑目标，保留原项目特征列 | 3322 |

派生文件 SHA-256：

```text
675f5ab2bc88ef7a92734b8d87d1fbc6f12be587f4f4f9652a6a7b58d4eff69c  sunspot_with_cycle_monthly_2026-10.csv
97590679284b483b2a56df5e04d38a04f35b3f1412ce2d6be9be15afc7b63929  sunspot_with_cycle_smoothed13_2026-04.csv
```

派生文件额外保留 `provisional`、`n_obs`、`ssn_std`：

- `provisional=1`：官方仍标记为暂定值，可能被 SILSO 后续修订。
- `n_obs`：对应月份使用的观测数。
- `ssn_std`：官方文件中对应月的标准差。

## 注意

- 原始月均值最新到 2026-10，其中 2026-07 至 2026-10 为 provisional。
- 平滑值有效到 2026-04，且 2026-01 至 2026-04 为 provisional。
- 历史数据可能被 SILSO 修订；旧项目文件 `sunspot_with_cycle.csv` 保留不动，以复现旧实验。
- 若把 13 月平滑序列用作模型输入，必须重新检查切分边界的数据泄露；居中平滑在 t 月使用 t+6 的观测。

## 文献依据

- `literature/太阳黑子，来源可疑lrsp-2015-4.pdf`，Hathaway (2015), §4.2：
  13 月 running mean 是传统标准；同时比较了 24-month FWHM Gaussian，后者对高频泄漏更小。
- `literature/Petrovay2020_Article_SolarCyclePrediction.pdf`：
  将 13-month boxcar average（首尾月半权重）称为太阳周期预测的标准做法。
- `literature/stab1159.pdf`，Xiong et al. (2021)：
  实际使用 Gaussian smoothing，`σ=3.6`，替代传统 13-month moving average。
- `literature/数据处理/Solar cycle prediction using a long short-term memory deep learning modelWang_2021_Res._Astron._Astrophys._21_012.pdf`：
  使用 SILSO 13-month smoothed monthly SSN。
- `literature/s11207-025-02577-y.pdf`，Gerçeker et al. (2025)：
  同时比较未平滑月均 SSN 与 SILSO 13-month smoothed SSN。
