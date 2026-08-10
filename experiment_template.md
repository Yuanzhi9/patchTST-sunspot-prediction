# 实验记录模板

> 用法：每次跑新实验前，复制此文件，重命名为 `EXPERIMENT_<ID>.md`，填写后存到项目根目录。
> 跑完实验后补充结果和结论段，然后：
> 1. 在 `result.txt` 里追加一行标准化记录
> 2. 在 `experiment_history.md` 快速索引表里追加一行 + 在末尾追加一节详情
> 3. `git add` 所有新增/修改文件，commit

---

## [实验ID] — [一句话描述改动]

| 字段 | 内容 |
|------|------|
| 日期 | YYYY-MM-DD |
| 实验ID | EXP-XX / 自定义 |
| 目的 | 回答什么问题 |
| 改动变量 | 只改了什么（必须是 1 个） |
| 对照基线 | 跟谁比 |
| 训练入口 | run_longExp.py（原管线，必须） |
| features | MS 还是 M |
| 前提（mode/数据/归一化） | — |

---

## 配置

```yaml
# 数据
data_file: sunspot_with_cycle.csv
时间范围: 1749-2025
train_rows: 3119
val_rows: 132
test_rows: 70
enc_in: 3  # month_sin, month_cos, ssn

# 模型
model: PatchTST
d_model: 128
n_heads: 8
e_layers: 2
d_ff: 2048
# ⚠️ patch_len 和 stride 不在目录名中，此处必须填写
patch_len: 16
stride: 8
dropout: 0.2
RevIN: 1

# 序列
seq_len: 96
pred_len: 24

# 训练
mode: M  # M or MS
loss: MSE
lr: 0.0001
batch_size: 16
epochs: 10
seed: 2021
patience: 3
EarlyStopping: True/False
normalization: StandardScaler  # or MinMax

# 其他
激活函数: GELU  # or ReLU, Linear
```

---

## 训练命令

```bash
# 所有参数显式 CLI 传入，不修改 run_longExp.py 源码
# 如下示例为 EXP-14 默认配置，实验时替换为实际参数
PYTHONPATH=PatchTST_supervised python3 run_longExp.py \
  --is_training 1 --model_id sunspot --model PatchTST --data custom \
  --root_path ./PatchTST_supervised/dataset/ \
  --features MS --target ssn --enc_in 3 \
  --seq_len 96 --label_len 48 --pred_len 24 \
  --d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 \
  --patch_len 16 --stride 8 --revin 1 --dropout 0.05 \
  --train_epochs 10 --patience 20 --batch_size 32 --learning_rate 0.0001 --loss mse \
  --itr 1 --num_workers 0 --activation gelu --des EXP-XX
```

---

## 结果

```yaml
# 从 result.txt 和 npy 反算
MSE_z:
MAE_z:
RSE:
物理MAE全步:
物理MAE_step0:
物理RMSE:
R²:
峰值段MAE(SSN>150):
```

---

## 结论

1. 这个实验回答了什么问题
2. 结论是什么
3. 对下一步的约束/建议

---

## 自检（运行后填写，避免无意识记录错误）

- [ ] 本次只改了 1 个变量？
- [ ] 对照基线用的是同一配置？
- [ ] 训练入口确认是 run_longExp.py（PYTHONPATH=PatchTST_supervised）？
- [ ] 指标用物理口径报告（MAE/RMSE/R²），不只报 z-score？
- [ ] result.txt 追加了一行？
- [ ] experiment_history.md 索引表追加了一行？

---

## 示例（填写好的样子）

```markdown
## EXP-14 — dm512→128（降参验证）

| 字段 | 内容 |
|------|------|
| 日期 | 2026-06-14 |
| 实验ID | EXP-14 |
| 目的 | 验证降参(d_model 512→128)对泛化的影响 |
| 改动变量 | d_model: 512→128 |
| 对照基线 | EXP-13 (dm512) |

## 配置
同基线，仅 d_model=128

## 结果
MSE_z: 0.079, MAE_z: 0.125, RSE: 0.304
物理MAE全步: 23.87, step0: 9.08, R²: 0.568
峰值段MAE: 68.8

## 结论
降参有泛化提升（R² +5.4%），但峰值压制未解决。
```
