# 数据管线说明

从 CSV 到物理指标评估的完整数据流。

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
| 训练/验证/测试切分 | `data_loader.py:258-263` | train=3119, val=132, test=70 |
| RevIN 行为 | 模型内部 | norm(输入) → denorm(输出)，逐窗口统计 |

## 归一化历史

| 实验 | 数据 | scaler | features |
|------|------|--------|----------|
| Stage 0 | 1749+（旧） | MinMax | 5 列（含 year_norm, cycle_phase） |
| Stage 1（Baseline B） | 1867+ | standard | MS（3 列） |
| Stage 3（EXP-14） | 1749+ | standard | M（3 列） |
| Stage 7（EXP-16c） | 1749+ | standard | MS（3 列） |
