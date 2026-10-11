# 数据管线说明

从 CSV 到物理指标评估的完整数据流。
最后更新：2026-08-20（数据口径警示 + 月均vs平滑对照）

## ⚠️ 数据口径警示（2026-08-20 景修提出，重要待议）

- **ssn 列 = SILSO 月均 SSN（Monthly Mean，未平滑）**。全项目训练/评估均用此口径。
- **版本锚点**：2014-04=112.5、2024-10 平滑峰=159.2 与 SILSO V2 官方逐位吻合（2015 年后段落为 V2 口径）；全序列最大值 398.2@1778-05 的版本归属**待与 SILSO 官网逐月核对**（阶段2 数据更新前必须完成）。
- **景修存疑（Q9，待议）**：训练数据是否应改用 13 月平滑 SSN？景修倾向"平滑才是正常口径"。影响：若换则全管线重训 vs 仅阶段3 外推用平滑——待与学姐学长/导师讨论。此条为 AI 记录，未定论。
- **谷峰引用口径**：本文件及实验文档中所有谷峰时间/值，均为**月均口径**。官方周期界（平滑极小值/极大值月）与月均极值月可差 2-8 个月。
- **⚠️ 单种子限制（2026-08-20 统一标注）**：全部实验基于 seed=2021 单次训练。弊端：无重复标准差/置信区间，3% 边缘的差异无法区分"真实差异"与"种子噪声"，图无法带误差棒。解决想法（不写死，可灵活调整）：①对关键实验做多种子重跑（训练方差）；②对现有预测做 bootstrap 重采样（评估方差）——两者性质不同，报告时分开。详见 SOP 追踪表待办。

### 月均 vs 13 月平滑谷峰对照（2026-08-20 实测计算，t-6..t+6）

| 窗口 | 月均谷 | 平滑谷 | 月均峰 | 平滑峰 |
|------|--------|--------|--------|--------|
| W1 | 2008-08 (0.3) | 2008-11 (2.4) ⚠️边界效应 | 2000-07 (244.3) | 2002-03 (180.9) |
| W2 | 2009-08 (0.0) | 2019-11 (2.3) ⚠️边界效应 | 2014-02 (146.1) | 2014-04 (115.3) |
| W3 | 2020-02 (0.2) | 2019-12 (1.9) ⚠️边界效应 | 2024-08 (216.0) | 2024-10 (159.2) |

> ⚠️ 平滑谷均落在测试段边界月（平滑窗口 t-6..t+6 缺数据），数值失真——本数据集截断下"平滑口径谷"算不准，此为结构性限制。

## 归一化与逆操作

归一化方法不固定（历史用过 StandardScaler、MinMaxScaler，未来可能换）。

### 归一化方法 × 逆操作对照表

| scaler | 归一化公式 | 逆操作 | sklearn 方法 | 已验证 |
|--------|----------|--------|-------------|--------|
| `standard` | `z = (x - μ) / σ` | `x = z * σ + μ` | `scaler.inverse_transform()` | ✅ EXP-14（9.08, 23.87 对上了） |
| `minmax` | `z = (x - min) / (max - min)` | `x = z * (max - min) + min` | `scaler.inverse_transform()` | ⚠️ 代码支持但未实测 |

### 切换归一化时，必须同步改的地方

| 文件 | 位置 | 改什么 |
|------|------|--------|
| `data_loader.py` | L237, L274-278 | scaler 类 + fit/transform 调用 |
| `save_config.py` | `--scaler` / `--scaler_params` | CLI 参数默认值或传入值 |
| `scripts/eval_metrics.py` | `SCALER_MAP` | 如果用了新 scaler 类，加到字典里 |
| `roll_eval.py` | `load_data_enc3` | scaler 类 + 逆操作 |
| 本文件（data_pipeline.md） | 上表 | 追加新行，标注"已验证"或"未实测" |

### 验证机制

每换归一化后，用已知 checkpoint 跑一次与物理指标对比。偏差 > 5% = 归一化或逆操作不匹配。

当前验证基准（EXP-14, StandardScaler）: step0=9.08, 全步=23.87

```bash
python scripts/eval_metrics.py results/EXP-14_PATH/ --scaler <新scaler> --data_csv ... --num_train 3119
```

### 通用公式

```
训练: 物理值 → fit(scaler) → transform → 归一化空间 → 模型
评估: pred.npy(归一化空间) → inverse_transform → 物理值
```

只要 scaler 和 inverse 是同一个类、参数一致，公式成立。

---

## 数据流总览

| 阶段 | 输入 | 变换 | 输出 | 关键文件 |
|------|------|------|------|---------|
| 读入 | CSV (8列) | `pd.read_csv` | DataFrame | — |
| 列筛选 | 全列 | `cols=['month_sin','month_cos']` → `[date, ms, mc, ssn]` | 4 列 DataFrame | `data_loader.py:249-251` |
| 归一化 | 3 特征列 | `fit(scaler_config) → transform` | z-score（或归一化空间） | `data_loader.py:273-278` |
| 窗口化 | 归一化数据 | sliding window (seq_len + pred_len) → Dataset | batch (batch, seq_len, enc_in) | `data_loader.py:298-301` |
| 模型 | batch | PatchTST forward + RevIN（内部 norm/denorm） | (batch, pred_len, enc_in) 预测 | `PatchTST.py:80-91` |
| 评估 | pred.npy / true.npy | `inverse_transform`（与归一化对应的逆操作） | 物理 SSN | `scripts/eval_metrics.py` |

## 硬编码约束

以下内容写死在代码里，换数据/特征时需要同步改：

| 约束 | 位置 | 值 |
|------|------|-----|
| 特征列名 | `data_loader.py:249` | `['month_sin', 'month_cos']` |
| 目标列名 | argparse 默认值 | `'ssn'` |
| 训练/验证/测试切分 | `data_loader.py` | **2026-08-12 起支持 date-based**：传 `--test_start`/`--test_end` 时按年月切分；不传时回退旧硬编码 train=3119, val=132, test=70 |
| RevIN 行为 | 模型内部 | norm(输入) → denorm(输出)，逐窗口统计 |

## 回测窗口切分（2026-08-12 新增，阶段1起用）

CLI 传 `--test_start`（YYYY-MM）和 `--test_end`（YYYY-MM），`data_loader.py` 自动：

```
train_end = test_start − 133 个月（留 132 个月给 val）
num_train = count(date_ym ≤ train_end)
val = 其后 132 个月
test = val 后至 test_end
```

三窗口（阶段1使用，数值已实测验证）：

| 窗口 | test_start | test_end | train_end(自动) | num_train | num_test |
|------|-----------|----------|-----------------|-----------|----------|
| W1 | 1996-08 | 2008-11 | 1985-07 | 2838 | 148 |
| W2 | 2008-12 | 2019-11 | 1997-11 | 2986 | 132 |
| W3 | 2019-12 | 2025-10 | 2008-11 | 3118 | 71 |

> `eval_metrics.py` 的 `--num_train` 必须与上表严格一致（差一行=scaler 锚点错位，所有物理值作废）。

### 数据泄露核查（2026-10-11，结论：无泄露）

> 起因：景修问"三种口径（全步/block/滚动）确定了有没有数据泄露"。
> 核查方式：脚本 `scripts/check_leakage.py`（三窗口边界按 `Dataset_Custom.__read_data__` 同式重算并打印），
> 明细见 `GPU-dropout-abl` 分支执行输出。结论：**三种口径均无未来信息泄露**。

**三条证据**

| 项 | 实测 | 判定 |
|----|------|------|
| scaler 拟合范围 | `[0, num_train)`：W1 1749-02~1985-07 / W2 到 1997-11 / W3 到 2008-11 | 仅训练段 ✓ |
| val 目标区间 | `[num_train, num_train+132)`：W1 1985-08~1996-07 等，与训练段**零重叠**（val 窗口回看 336 月的部分落在训练段内，但**目标**不重叠） | ✓ |
| 预测路径最大读取下标 | = 测试段末月（W1 2008-11 / W2 2019-11 / W3 2025-10），未读测试段之后 | ✓ |

**三条口径的具体检查**

- 全步：窗口 i 的输入为 `[border1+i, border1+i+336)`，目标从 `border1+i+336` 起，严格因果（输入窗口尾部含测试段内的**实测**值，这是"有真值锚定的一次性预测"，属正常评估设定，不算泄露）。
- 滚动 / block：回填时写入窗口的 ssn 一律是**模型自己的预测**；只有 `month_sin/month_cos` 用真实值。
- 真值 `data_z[test_start+t]` 只在两处被读：算 MAE、取真实月相——都是评估侧用途，不进预测路径。

**两条"看起来像泄露、实为约定"的说明（以后再被问到直接引这里）**

1. `month_sin/month_cos` 用真实值 = 未来日历已知（月相/季节是可预知的），只有 ssn 被预测。项目既定约定，`roll_eval.py` 与 `b23_block_eval.py` 一致。
2. `label_len=48` 的 decoder 输入取自**输入窗口尾部**（过去 48 月），且 PatchTST 的 `forward(x)` 只消费 `x`，该输入实际未被使用。

**未使用的列（不构成任何问题，但记录在案）**：`year_norm`、`cycle_phase` 两列存在于 CSV，当前 `enc_in=3` 未启用
（`cols=['month_sin','month_cos'] + [target]`）。若要接入（物理先验方向），属改特征、需重训。

## 正式口径（2026-08-14 起）

**正式实验 = 最佳 val 模型口径**：命令链 `train && 补测(--is_training 0)`，见 experiment_SOP.md §4。
背景：train→test 同进程时 test 用内存中的最终模型（exp_main.py L229 加载最佳权重的代码被注释），50ep 下最终模型过拟合（0b-r2 门禁触发记录），故统一为最佳模型口径。

## 归一化历史

| 实验 | 数据 | scaler | features |
|------|------|--------|----------|
| Stage 0 | 1749+（旧） | MinMax | 5 列（含 year_norm, cycle_phase） |
| Stage 1（Baseline B） | 1867+ | standard | MS（3 列） |
| Stage 3（EXP-14） | 1749+ | standard | M（3 列） |
| Stage 7（EXP-16c） | 1749+ | standard | MS（3 列） |
| Stage 8（EXP-17~19，阶段1） | 1749+ | standard | MS（3 列），date-based 三窗口，最佳模型口径 |

---

## 加特征/换目标改动检查清单（2026-08-14，改动前逐项勾选）

> 适用场景：加物理参数特征（如 aa 指数/F10.7）、换目标变换（如 sqrt）等数据管线改动。
> 与 experiment_SOP.md §4-B 配套（§4-B 管"改代码"，本清单管"改数据"）。

改动前勾选（逐项确认，全部 ✓ 才动工）：

- [ ] **1. 数据对齐**：新列按年月 merge 进 CSV。确认时间覆盖（新列起止年份）与缺失值策略（缩短训练段 / 截掉参数 / 前向填充）——后者是决策点，与方案讨论时定，不是技术顺手解决
- [ ] **2. 列选择**：`data_loader.py` 硬编码 `cols=['month_sin','month_cos']` 是否要加新列——忘改=新特征根本没被读入
- [ ] **3. 通道数同步**：`enc_in` 数值在 run_longExp.py CLI、save_config.py 参数、eval_metrics.py、roll_eval.py 四处同步更新——不同步=评估结果作废
- [ ] **4. 评估脚本适配**：`eval_metrics.py` 的 ssn_col（最后一列）逻辑、`roll_eval.py` 的 enc3 管线（新特征列的滚动回填逻辑）——反算错列=物理值全错
- [ ] **5. 目标变换的反变换**（如 sqrt）：预测值 ŷ² 回物理空间后**再**算 E_r/MAE；变换空间的指标与物理空间指标不可混用
- [ ] **6. 量纲差异**：物理参数量纲天差地别（F10.7=60-300，纬度=±40°）。StandardScaler fit 全列后 z 值可比，但需在实验中观察某参数是否因方差极大而主导模型；必要时按参数分组标准化
- [ ] **7. 回归验证锚点**：任何管线改动后对拍已知数值（EXP-14: step0=9.08/全步=23.87；EXP-18-5b W1: 全步=22.62），偏差 >5% 停，先排查
- [ ] **8. 文档同步**：本文件"归一化历史"表和"硬编码约束"表同步更新（特征列名、enc_in 值）；experiment_history 索引表加新实验行
- [ ] **9. 评估脚本回归测试**：改动涉及 eval_metrics/roll_eval 时，跑 `python scripts/test_eval.py` 确认没改坏
