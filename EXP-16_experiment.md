# EXP-16 — Baseline B参数×1749+数据复现

> 记录时间：2026-08-10 | SOP 全程对照

## 基本字段

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-10 |
| 实验ID | EXP-16 |
| 目的 | 在 1749+ 数据上用 Baseline B 参数配置训练，建立与 EXP-14 同数据源的对照基线 |
| 改动变量 | 非消融（相对 EXP-14 改 7 项：seq_len/n_heads/e_layers/d_ff/patch_len/stride/epochs） |
| 对照基线 | EXP-14（同数据源） |
| 前提条件 | 1749+, M mode, enc_in=3, StandardScaler |

## 配置

```yaml
data_file: sunspot_with_cycle.csv
时间范围: 1749-2025
enc_in: 3
seq_len: 132
pred_len: 24
d_model: 128
n_heads: 16
e_layers: 3
d_ff: 256
patch_len: 12
stride: 6
dropout: 0.05
RevIN: 1
loss: MSE
lr: 0.0001
batch_size: 16
epochs: 50 (EarlyStopping@12, patience=3)
seed: 2021
scaler: standard
```

配置快照：`configs/EXP-16_2026-08-10.json`

## 结果

```yaml
物理MAE全步: 24.02
物理MAE_step0: 18.07
物理RMSE: 33.88
R²: 0.553
峰值E_r: -85.2
峰值E_m: -12
误差分层(0-50/50-100/100-150/>150): 15.4/10.2/23.1/71.3
滚动MAE: 33.18 (前期21.39/后期54.30/2.5x)
z-score: mse=0.082 mae=0.127 rse=0.310
```

## 结论

**一句话结论（2026-08-10）**：证据不足，待对照 EXP-16b（patience=20）补充分析。ES@12 提前截断可能是欠拟合，不是架构上限。**EXP-16b 结果已出（2026-08-10）：step0 18→14改善，全步 24→25轻微退化，滚动 33→31改善。EXP-16 的差主要来自 patience=3 的欠拟合。**

⚠️ 本实验不认可为正式基线——EXP-16b 替代为正式评估版本。
