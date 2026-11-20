# B26M 双基线实验 — p24 补强与 p132 独立调参

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-20 |
| 实验ID | `B26M-*`（p24）/ `B26M-P132-*`（p132） |
| 类型 | 训练实验 + 分析任务 |
| 目的 | 分别寻找 p24 与 p132 的最佳可复现基线，再做同日期曲线比较 |
| 改动变量 | 分阶段单变量扫描；p132 首先改变任务 horizon |
| 对照基线 | p24：`B26M-F-P36-SEED2019/2020/2021`；p132：本文件结构轮 seed2021 |
| 训练入口 | 两者均用 `run_base_cv5.py` |
| features | MS |
| 前提 | 2026-10 月均未平滑 SSN；StandardScaler；RevIN=1；MSE |
| 坑清单回顾 | 13,14,15,16,18,19,20,22,26,27,29,32,34,35；新增 p132 shape/窗口/mini-batch 风险 |
| 种子范围 | p24 目标 2019–2023；p132 筛选 seed2021，最终目标 2019–2023 |

## 预检表

- [x] 训练入口：`run_base_cv5.py`
- [x] p24 fold：`configs/cv5_folds_2026-10.json`
- [x] p132 fold：`configs/cv5_folds_p132_2026-10.json`（训练前生成；inner train 507 / inner val 160）
- [x] 数据路径与 SHA256 由每次 config 自动记录
- [x] features=MS，target=ssn，enc_in=3，num_workers=0
- [x] p24 对照 ID：`B26M-F-P36-SEED2019`、`B26M-F-P36-SEED2020`、`B26M-B-P36-BS32`
- [x] p132 对照 ID：本文件阶段 S132/P36
- [x] 只改一个变量：阶段内是；p132 初始锚点跨任务，不归因于单参数
- [x] 可证伪假设、预尸检、止损见预登记
- [x] 不修改旧 `run_longExp.py` / `Dataset_Custom` / `Exp_Main`
- [x] p132 先 smoke，后批量
- [x] 评估后图内数值必须与 summary 对拍
- [x] 提交前核对全部未跟踪文件
- [x] 不 push

## 参数对照

| 参数 | p24 基准 | p24 补强 | p132 基准 |
|---|---|---|---|
| seq_len | 132 | 132 | 132（结构轮另测336） |
| pred_len | 24 | 24 | 132 |
| patch/stride | 36/18 | 36/18 | 36/18（结构轮另测16/8、24/12） |
| lr | 3e-4 | 3e-4 / 1e-3 | 3e-4（lr轮再测） |
| batch | 32 | 16 / 32 | 32（batch轮再测） |
| dropout | 0.2 | 0.2 | 0.2（dropout轮再测） |
| n_heads | 8 | 8 | 8（heads轮再测） |
| epochs | 50 | 50 | 50 |
| seed | 2019/2020/2021 | 2019–2023 | 2021（最终轮2019–2023） |

## 可证伪假设

1. p24 P36 若稳定，五 seed 均值相对三 seed 的变化应小于 3%，且 lr×batch 不出现显著翻转。
2. p132 的可行配置应能在无重叠 132 月曲线上给出不是单纯平线/均值回归的峰谷，并至少在一个折上接近 p24 的同日期误差。
3. 若 p132 的峰谷幅度、时间或 fold 间离散度全面失控，则应停止继续扩大 p132 网格，并保留负结果。

## 预尸检摘要

最可能的失败不是“GPU 不够快”，而是把高度重叠的 132 月滑窗误当成独立周期样本，
导致指标虚高或 checkpoint 偶然选中。主指标因此使用无重叠 132 月直接曲线；滑窗全步仅作诊断。

## 训练与评估产物

### p24 补强完成

共 5 个配置 × 5 折 = 25 次训练；每个配置均找到 5 个外层 fold 产物，无真实 Traceback/NaN/OOM。
日志中的 `UnpicklingError` 字样来自项目既有的安全检查提示，随后以 `weights_only=False`
成功加载自身 checkpoint，不是失败。

#### lr × batch（P36、三 seed）

| lr | batch | Full MAE mean ± seed std | Block24 MAE mean ± seed std |
|---:|---:|---:|---:|
| 3e-4 | 16 | 27.15 ± 0.28 | 26.91 ± 0.26 |
| **3e-4** | **32** | **26.83 ± 0.14** | **26.46 ± 0.35** |
| 1e-3 | 16 | 27.30 ± 0.06 | 27.14 ± 0.58 |
| 1e-3 | 32 | 27.36 ± 0.54 | 27.56 ± 1.20 |

结论：四格中 `lr=3e-4, batch=32` 在 Full 与 Block24 都最好；没有排序翻转。

#### 最终候选五 seed

| seed | Full MAE | Block24 MAE |
|---:|---:|---:|
| 2019 | 26.77 | 26.35 |
| 2020 | 26.74 | 26.18 |
| 2021 | 27.00 | 26.85 |
| 2022 | 26.84 | 26.53 |
| 2023 | 26.83 | 26.51 |
| **五 seed** | **26.83 ± 0.10** | **26.48 ± 0.25** |

五 seed 均值与三 seed 基本重合，未出现超过 3% 的变化。p24 最佳基线暂定为
`BASE-2026-11-M-CAND-P36`。

#### p24 图件

- `plots/fig_p24_structure_stability_2026-11-20.{png,pdf}`
- `plots/fig_p24_candidate_five_seed_stability_2026-11-20.{png,pdf}`
- `analysis/p24_dual_summary/p24_lr_batch_heatmap.{png,pdf}`
- `analysis/p24_dual_summary/p24_candidate_five_seed.{png,pdf}`

### p132 预检与 smoke

- 独立切分：inner train 507（末块 506）/ inner val 160。
- `seq_len=132, pred_len=132`：训练 975 窗口，验证 116，F1 外层 404；
- `seq_len=336, pred_len=132`：训练 159 窗口，验证 116，F1 外层 200；
- 两者 1 epoch smoke 均有限值、shape=(N,132,1)，`steps/epoch` 分别为 30 与 4。

> 早先 469/198 方案因 `seq_len=336` 每块只剩 2 个训练窗口、batch32 会得到 0 个 batch，
> 在正式训练前废弃并改为 507/160。

## 结论

p24 双基线中的 p24 侧已完成补强：P36 + lr3e-4 + bs32 保持最佳。p132 结论待结构轮与后续参数轮完成后补充。
