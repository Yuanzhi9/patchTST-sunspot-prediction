# EXP-16c — Baseline B严格参数×1749+数据（run_longExp.py原管线）

> 记录时间：2026-08-10 | SOP 全程对照 | 取代 EXP-16/16b

## SOP §1：实验前

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-10 |
| 实验ID | EXP-16c |
| 目的 | 用 run_longExp.py（原管线）+ Baseline B 严格参数在 1749+ 数据上复现基线 |
| 改动（与 Baseline B 比） | 数据：1867+→1749+；itr：2→1（B 未重新训练）；patience：100→20；num_workers 相同=0 |
| 对照基线 | Baseline B |
| 前提条件 | 1749+, MS mode, enc_in=3, StandardScaler, RevIN=1 |
| 训练入口 | run_longExp.py（原管线） |

### 可证伪假设

```
如果 1749+ 上 Baseline B 参数可复现 → step0 10-18，全步 22-26，滚动 < 35
如果不可复现 → step0 > 20 或全步 > 28，数据范围差异是关键瓶颈
```

### 预尸检

失败最可能因为：1749+ 极值 398 拉大 scaler std，结合 StandardScaler 天然压制物理幅度。

## 配置

```yaml
训练脚本: run_longExp.py
data: sunspot_with_cycle.csv (1749+, 3321月)
features: MS  enc_in: 3  target: ssn
seq_len: 132  pred_len: 24
d_model: 128  n_heads: 16  e_layers: 3  d_ff: 256
patch_len: 12  stride: 6  RevIN: 1  dropout: 0.05
train_epochs: 50  patience: 20  batch_size: 32  lr: 0.0001
loss: mse  activation: gelu  num_workers: 0  itr: 1
```

## 训练命令

```bash
python run_longExp.py \
  --is_training 1 --model_id sunspot --model PatchTST --data custom \
  --root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv --freq m \
  --features MS --target ssn --enc_in 3 \
  --seq_len 132 --label_len 48 --pred_len 24 \
  --d_model 128 --n_heads 16 --e_layers 3 --d_ff 256 \
  --patch_len 12 --stride 6 --revin 1 --dropout 0.05 \
  --train_epochs 50 --patience 20 --batch_size 32 --learning_rate 0.0001 --loss mse \
  --itr 1 --num_workers 0 --activation gelu \
  --des EXP-16c
```

## 结果

```yaml
物理MAE全步: 24.54
物理MAE_step0: 11.21
物理RMSE: 34.66
R²: 0.532
E_r: -85.5
E_m: -12
误差分层(0-50/50-100/100-150/>150): 10.8/11.5/26.0/73.0
滚动MAE: 36.37 (前期26.90/后期53.24/2.0x)
z-score: mse=0.257 mae=0.359 rse=0.684
训练epochs: 38 (ES@20, best val@epoch 15)
```

## 结论

```yaml
一句话结论: run_longExp.py原管线复现成功。step0=11.21优于原始Baseline B(13.02)，但滚动36.37差于原始B(33.13)。峰值-85.5系1749+数据宽range的Scaler压制效应，与EXP-14(68.8)同根因。
跟Baseline B比较: step0 13.02→11.21(改善) 全步 23.91→24.54(微差) 滚动 33.13→36.37(变差)
差异来源: 数据1749+(宽range 0-398) vs 1867+(窄range)，Scaler std差异是关键
```
