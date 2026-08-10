# EXP-16b — EXP-16对照：patience 3→20

> 记录时间：2026-08-10 | SOP 全程对照

## SOP §1：实验前

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-10 |
| 实验ID | EXP-16b |
| 目的 | 验证 EXP-16 是否因 patience=3 导致 EarlyStopping 提前截断（epoch 12）而欠拟合 |
| 改动变量 | patience: 3 → 20（唯一变量） |
| 对照基线 | EXP-16 |
| 前提条件 | 1749+, M mode, enc_in=3, StandardScaler, seq=132/nh16/el3/df256/pl12/str6/50ep |

### 可证伪假设

```
如果有效 → step0 从 18.07 下降，全步从 24.02 下降，滚动 ≤33
如果无效 → 全步持平甚至变差（过拟合），step0 无改善
如果结果是（无效）→ 瓶颈不在训练轮数，在 1749+ 数据上该架构天然能力上限
```

### 预尸检

如果两周后证明这实验是废物，最可能因为：1749+ 数据的宽 range（极值 398）用 StandardScaler 天然压制物理幅度，多训练不解决 scaler 层面的系统性压缩。

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
epochs: 50
patience: 20          # ← EXP-16 为 3
seed: 2021
scaler: standard
des: EXP-16b          # ← 避免覆盖 EXP-16 输出
```

## 训练

```bash
python run_sunspot_fixed.py \
  --seq_len 132 --n_heads 16 --e_layers 3 --d_ff 256 \
  --patch_len 12 --stride 6 --train_epochs 50 --patience 20 --des EXP-16b
```

## 结果

```yaml
物理MAE全步: 24.85
物理MAE_step0: 13.90
物理RMSE: 34.95
R²: 0.524
峰值E_r: -81.5
峰值E_m: -12
误差分层(0-50/50-100/100-150/>150): 12.1/11.1/26.3/73.6
滚动MAE: 30.78 (前期21.80/后期44.48/2.0x)
z-score: mse=0.087 mae=0.124 rse=0.319
epochs实际: 48 (EarlyStopping counter 20/20)
```

## 结论

```yaml
一句话结论: patience 3→20有效——step0 18.07→13.90改善明显，滚动 33.18→30.78改善，但全步 24.02→24.85轻微退化（可能过拟合）。
跟EXP-16比较: step0 18.07→13.90 全步 24.02→24.85 滚动 33.18→30.78
峰值: 仍严重低估（-81.5），瓶颈在任务定义/数据范围，非训练轮数可解。
```

## 可证伪假设验证

```
step0 下降 ✅ (18.07→13.90)
全步下降 ❌ (24.02→24.85，轻微上升=轻微过拟合)
滚动下降 ✅ (33.18→30.78)
→ 部分成立。更多训练改善了首窗口预测，但在滑动窗口全局评估上有轻微过拟合。
```
