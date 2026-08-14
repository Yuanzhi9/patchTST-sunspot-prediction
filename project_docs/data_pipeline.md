# 数据管线说明

从 CSV 到物理指标评估的完整数据流。
最后更新：2026-08-14（date-based 切分 + 三窗口 + 命令链口径）

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
