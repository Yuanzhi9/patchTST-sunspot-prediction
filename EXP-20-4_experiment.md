# EXP-20-4 — sqrt 变换（探索期，Stage 9）

> 探索期队列第四实验。判定标准待议——只收集数据，不下有效/无效结论。

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-15 |
| 实验ID | EXP-20-4 |
| 目的 | 打峰值压制机制2（Scaler锚点/右偏分布）：目标 SSN 做 sqrt 变换，压缩右偏，使 MSE 高值区梯度更均匀 |
| 改动变量 | target_transform（唯一变量）：''(基线) → sqrt |
| 对照基线 | EXP-18-5b（无变换, W1：全步=22.62, R²=0.789, E_r=-24.4, 滚动MAE=50.41, 滚动峰=217.6/244.3） |
| 代码改动 | data_loader/data_factory/run_longExp/run_sunspot_fixed/save_config/eval_metrics/roll_eval 共 7 文件（均注释标注+3 个备份）；回归验证通过（test_eval PASS + EXP-14 锚点逐位一致 + sqrt/默认模式 scaler mean 双验证 PASS） |
| 训练入口 | run_longExp.py（命令链模式） |
| 口径 | 最佳 val 模型 |
| 豁免标注 | 本队列一次性豁免 SOP §8（仅此一次） |

## 可证伪假设

sqrt 把 244 和 146 的差从 98 压到 √244−√146≈3.5，模型在 sqrt 空间更容易顶到峰值。文献传统做法（Hathaway/Petrovay §1.2.3）。

- 若机制2成立：E_r 和滚动峰值误差改善
- 代价预期：低值区分辨率下降（√1=1, √10≈3.2），全步可能持平或略变差
- 注意：E_r 等指标在物理空间（平方后）计算；sqrt 空间的数值不直接报告

## 参数对照表

| 参数 | 基线(5b) | EXP-20-4 | 一致？ |
|------|---------|----------|--------|
| features/窗口/seq_len/pred_len | MS/W1/336/24 | 同 | ✓ |
| d_model/n_heads/e_layers/d_ff | 128/8/2/2048 | 同 | ✓ |
| patch_len/stride/dropout/fc_dropout | 16/8/0.2/0.2 | 同 | ✓ |
| revin/affine/individual/activation/loss | 1/0/0/gelu/mse | 同 | ✓ |
| batch_size/lr/epochs/patience/seed | 16/1e-4/50/100/2021 | 同 | ✓ |
| **target_transform** | **''** | **sqrt** | ✗ 唯一变量 |

## 结果（跑后填）

| 指标 | 基线无变换 (5b) | EXP-20-4 (sqrt) |
|------|:---:|:---:|
| 全步 MAE | 22.62 | 21.40 |
| step0 | 11.01 | 11.82 |
| R² | 0.789 | 0.800 |
| E_r | -24.4 | -29.8 |
| 滚动 MAE | 50.41 | 27.80 |
| 滚动峰值(预测/真实) | 217.6/244.3 | 179.7/244.3 (-64.6) |
| 150+ 分层 | 28.4 | 32.9 |
| 0-50 分层 | 12.3 | 9.1 |

## 结论（判定待议，跑后收集数据）
