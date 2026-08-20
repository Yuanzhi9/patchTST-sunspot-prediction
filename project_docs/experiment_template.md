# 实验记录模板

> 用法：每次跑新实验前，复制此文件，重命名为 `EXPERIMENT_<ID>.md`，填写后存到项目根目录。
> 跑完实验后补充结果和结论段，然后：
> 1. 在 `result.txt` 里追加一行标准化记录
> 2. 在 `project_docs/experiment_history.md` 快速索引表里追加一行 + 在末尾追加一节详情
> 3. `git add` 所有新增/修改文件，commit
>
> ⚠️ 2026-08-20 更新：①预检表必须跑前完成，事后补写不算数只作留档；②坑清单回顾必填；
> ③单种子限制必须标注；④分析任务（无训练）用本模板时按 §分析任务变体填写。

---

## [实验ID] — [一句话描述改动]

| 字段 | 内容 |
|------|------|
| 日期 | YYYY-MM-DD |
| 实验ID | EXP-XX / 自定义 |
| 类型 | 训练实验 / 分析任务（无训练，复用 checkpoint） |
| 目的 | 回答什么问题 |
| 改动变量 | 只改了什么（必须是 1 个）；分析任务写"无，评估任务" |
| 对照基线 | 跟谁比 |
| 训练入口 | run_longExp.py（原管线，必须）；分析任务写"无" |
| features | MS 还是 M |
| 前提（mode/数据/归一化） | — |
| **坑清单回顾（必填）** | 本实验可能踩的坑编号是 ___（对照 SOP 坑 1-28） |
| **单种子限制** | 本结果基于 seed=2021 单次训练。弊端：无重复标准差/置信区间，3% 边缘差异不可靠，图无法带误差棒。解决想法见 data_pipeline.md 顶部警示（多种子重跑+bootstrap，不写死） |

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
# ⚠️ 数据口径：ssn 为月均（未平滑）。谷峰引用均月均口径；官方周期界需平滑转换（Q9 待议）

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
target_transform: ''  # 空/sqrt/pow07/pow23/log1p（变换模型必填，评估侧须同传）

# 其他
激活函数: GELU  # or ReLU, Linear
```

---

## 可证伪假设（跑前填完，三行）

- 如果有效，___（指标）≤ / ≥ ___
- 如果无效，___（指标）≈ ___
- 如果结果是 ___ → 推论是 ___

## 预尸检（跑前填完）

如果两周后证明这实验是废物，最可能因为什么？

## 文献先例（跑前填完）

查 `literature/literature_reading_notes.md`（Ctrl+F 关键词）→ 有则记录"别人做到什么精度、踩了什么坑"；无则标注"无文献先例"。

---

## 训练命令

```bash
# 所有参数显式 CLI 传入，不修改 run_longExp.py 源码
# 如下示例为 EXP-14 默认配置，实验时替换为实际参数
# ⚠️ 正式实验用命令链模式（train && 补测），见 SOP §4
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
滚动MAE:
滚动峰误差(pred-true, 负=低估):
```

---

## 结论

1. 这个实验回答了什么问题
2. 结论是什么
3. 对下一步的约束/建议

---

## 自检（运行后填写，避免无意识记录错误）

- [ ] 本次只改了 1 个变量？（分析任务写"评估任务，非对照"）
- [ ] 对照基线用的是同一配置？
- [ ] 训练入口确认是 run_longExp.py（PYTHONPATH=PatchTST_supervised）？
- [ ] 指标用物理口径报告（MAE/RMSE/R²），不只报 z-score？
- [ ] 图内数值与已记值对拍一致（如有画图）？
- [ ] result.txt 追加了一行？
- [ ] project_docs/experiment_history.md 索引表追加了一行？
- [ ] diary 当天日志已写？

---

## 分析任务变体（无训练，2026-08-20 起）

分析任务（如 EXP-24 极小值评估）使用本模板时：
- "类型"填"分析任务"；"训练入口"填"无"；"改动变量"填"无，评估任务"
- 数据来源写清：复用哪些 checkpoint + num_train + target_transform
- 对拍门禁照旧：轨迹/指标必须与 result.txt 已记值一致
- 成功标准改为"数据完整、对拍通过即达成"，不下有效/无效判定
- 参照 EXP-24_experiment.md 的填写方式

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
