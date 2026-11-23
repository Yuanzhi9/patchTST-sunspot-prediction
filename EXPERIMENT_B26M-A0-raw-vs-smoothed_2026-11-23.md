# B26M A0 原始与 13 月平滑口径对照

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-A0RAW-1749-2026-04`；`B26M-A0SMOOTH13-1749-2026-04` |
| 目的 | 在完全相同的日期、切分、窗口和模型参数下，比较原始月均 SSN 与官方 13 月平滑 SSN 的 A0 基线行为 |
| 数据 | SILSO V2.0；两口径均为 1749-07–2026-04，3322 月 |
| 切分 | 顺序块五折；两口径五折日期逐项一致 |
| 模型 | `SL132/PL24/P36/S18/D128/H8/E2/F2048/DO0.2/FC0.2/LR3e-4/BS32/E50` |
| 种子 | 2019、2020、2021、2022、2023 |
| 评价 | 物理目标、Full MAE、非重叠 block24 MAE；另报按各口径真值标准差归一化的误差 |

## 唯一改动与解释边界

`B26M-A0RAW` 使用原始月均 SSN；`B26M-A0SMOOTH13` 使用 SILSO 官方
13 月中心平滑 SSN。两端都按 2026-04 截断，以保持每个 fold 的训练、验证和
测试月份相同。

本实验是 A0 数据口径对照，不是严格的“原始输入 + 平滑目标”实验。当前
`Dataset_BlockCV` 从同一个 `ssn` 列构造输入和目标，因此平滑版也是平滑输入 +
平滑目标。尤其注意：

- 13 月中心平滑在月份 `t` 使用 `t-6..t+6`；
- 当前 2026-10 端点没有可观测的官方 13 月平滑值，官方序列只到 2026-04；
- 因此 `SMOOTH13` 只能作回顾性基线和目标口径分析，不能直接作为 2026-10
  真实未来预报的可用输入。

## 五 seed 结果

两口径的物理 MAE 对应不同难度的目标，不能直接横比。归一化误差以各自真值
标准差缩放，可用于检查“平滑后误差下降是否只是目标变平滑”。

| 口径 | Full MAE | Block24 MAE | Normalized MAE | Normalized RMSE |
|---|---:|---:|---:|---:|
| raw | 26.913 ± 0.122 | 27.331 ± 0.284 | 0.4003 | 0.5554 |
| smoothed13 | 16.439 ± 0.251 | 17.903 ± 0.400 | 0.2632 | 0.3858 |

结论：

1. 13 月平滑版在 Full 和 Block24 上的物理 MAE 都更低，但这是与更平滑的目标
   比；不能据此宣称峰值预测能力同比提高。
2. 按目标尺度归一化后，平滑版 MAE 仍低约 34%，说明误差下降不只是 SSN 数值
   量纲变化，平滑目标确实更易学习；它同时抹去了短时起伏，不能证明峰幅和峰时
   的真实未来泛化更好。
3. raw 五 seed 的 Full/Block24 很稳定，确认当前 PR24 reference baseline
   在原始月均口径下可复现。
4. 正式 2026-10 预测继续以 raw 为主；平滑版作为第二口径，并必须使用对应的
   平滑真值评价。

## 验证

- 两口径 `check_cv5_leakage.py` 均通过：3322 月、日期一致、五折首尾无重叠；
- `test_eval.py` 的 EXP-14 回归锚点全部通过；
- 两口径各 25 个 result 目录，共 50 个 fold×seed 结果。

注意：泄露检查覆盖窗口与 fold 边界，不消除中心平滑本身的回顾性。上述
`SMOOTH13` 未来不可部署性来自数据定义，不是切分脚本错误。

## 产物

- `PatchTST_supervised/dataset/sunspot_a0_raw_1749-07_2026-04.csv`
- `PatchTST_supervised/dataset/sunspot_a0_smoothed13_1749-07_2026-04.csv`
- `configs/cv5_folds_a0_raw_2026-04.json`
- `configs/cv5_folds_a0_smoothed13_2026-04.json`
- `analysis/B26M-A0RAW-1749-2026-04_seed2019/` 至 `seed2023/`
- `analysis/B26M-A0SMOOTH13-1749-2026-04_seed2019/` 至 `seed2023/`
- `analysis/a0_raw_vs_smoothed/a0_raw_smoothed_seed_detail.csv`
- `analysis/a0_raw_vs_smoothed/a0_raw_smoothed_summary.csv`
- `analysis/a0_raw_vs_smoothed/a0_raw_vs_smoothed.png`
- `analysis/a0_raw_vs_smoothed/a0_raw_vs_smoothed.pdf`
- `scripts/prepare_a0_variants.py`
- `scripts/run_a0_raw_smoothed_2026-11-23.sh`
- `scripts/summarize_a0_comparison.py`
