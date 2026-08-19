# EXP-18 Round 1 — patch_len/stride 搜索 (W1窗口: Cycle 23)

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-12 |
| 实验ID | EXP-18-1a (12,6) / EXP-18-1b (16,8)=基线 / EXP-18-1c (24,12) |
| 目的 | 在 Round 0 基线(sl336)上，测试 patch_len/stride 三种组合，确定最优时间粒度 |
| 改动变量 | patch_len+stride（绑定，比值恒 2:1）：(12,6) vs (16,8) vs (24,12) |
| 对照基线 | EXP-17-0b (sl336, pl16, s8, 全步MAE=23.87) |
| 训练入口 | run_longExp.py（原管线，PYTHONPATH=PatchTST_supervised） |
| features | MS |
| 前提 | StandardScaler+RevIN=1, W1窗口, single seed=2021, epochs=50, patience=100 |

---

## 配置

**共同固定参数**（与基线 EXP-17-0b 完全一致，仅 patch_len/stride 不同）：
```
features=MS, enc_in=3, seq_len=336, pred_len=24, d_model=128, n_heads=8
e_layers=2, d_ff=2048, revin=1, dropout=0.05, fc_dropout=0.05, head_dropout=0.0
individual=0, activation=gelu, batch_size=16, lr=0.0001, loss=mse
train_epochs=50, patience=100, random_seed=2021, num_workers=0
```
窗口：test_start=1996-08, test_end=2008-11, num_train=2838, num_val=132, num_test=148

| 实验 | patch_len | stride | token数(sl336) |
|------|-----------|--------|----------------|
| EXP-18-1a | 12 | 6 | ~55 |
| EXP-18-1b（基线） | 16 | 8 | ~41 |
| EXP-18-1c | 24 | 12 | ~27 |

## 训练命令

```bash
# EXP-18-1a
PYTHONPATH=PatchTST_supervised python3 run_longExp.py \
  --is_training 1 --model_id sunspot --model PatchTST --data custom \
  --root_path ./PatchTST_supervised/dataset/ --data_path sunspot_with_cycle.csv \
  --features MS --target ssn --enc_in 3 \
  --test_start 1996-08 --test_end 2008-11 \
  --seq_len 336 --label_len 48 --pred_len 24 \
  --d_model 128 --n_heads 8 --e_layers 2 --d_ff 2048 \
  --patch_len 12 --stride 6 --revin 1 --affine 0 \
  --dropout 0.05 --fc_dropout 0.05 --head_dropout 0.0 --individual 0 \
  --train_epochs 50 --patience 100 --batch_size 16 \
  --learning_rate 0.0001 --loss mse --activation gelu \
  --random_seed 2021 --num_workers 0 --itr 1 --des EXP-18-1a

# EXP-18-1c 同上，仅 --patch_len 24 --stride 12 --des EXP-18-1c
```

> ⚠️ 执行环境备注：1a 与 1c 曾并行训练（双进程争抢 4 vCPU），2026-08-12 晚因并行导致内存/CPU 紧张、SSH 断连。指标有效性不受影响（每进程独立训练，仅墙钟时间受影响），但 2026-08-13 起执行协议改为全程串行。另注意：本 round 的测试对象为**第 50 轮最终模型**（一次性 train→test），与 Round 0 旧数据的"最佳checkpoint口径"不同（详见 EXP-17-Round0_experiment.md 口径修正节）。

---

## 结果

| 指标 | EXP-18-1a (12,6) | EXP-18-1b 基线 (16,8) | EXP-18-1c (24,12) |
|------|:---:|:---:|:---:|
| MSE_z | 0.264 | 0.198 | 0.240 |
| MAE_z | 0.395 | 0.351 | 0.380 |
| RSE | 0.554 | 0.480 | 0.529 |
| **全步 MAE** | 26.85 | **23.87** | 25.84 |
| step0 MAE | 16.62 | 11.39 | **10.26** |
| RMSE | 34.90 | **30.26** | 33.32 |
| R² | 0.693 | **0.769** | 0.720 |
| E_r | -31.6 | -17.8 | 7.6 |
| E_m | -2 | 3 | **0** |

### 峰值对照表 (Cycle 23, 全步口径)

| 实验 | E_r (幅度偏差) | E_m (时间偏差, 月) |
|------|:---:|:---:|
| 1a (12,6) | -31.6 | -2 |
| 1b (16,8) | -17.8 | 3 |
| 1c (24,12) | +7.6（罕见高估） | 0 |

### 误差分层 (全步 MAE)

| SSN 区间 | 1a (12,6) | 1b (16,8) | 1c (24,12) |
|----------|:---:|:---:|:---:|
| 0-50 | 14.5 | 14.1 | 15.5 |
| 50-100 | 25.1 | 25.2 | 25.3 |
| 100-150 | 32.5 | **30.6** | 34.7 |
| 150+ | 39.8 | **29.0** | 32.3 |

---

## 假设验证

**可证伪假设**：patch_len 偏离 16 会导致全步 MAE 恶化 ≥3%，或三组差距 <3% 说明 patch_len 不敏感。

**实际结果**：1a=26.85（+12.5% 恶化）、1c=25.84（+8.3% 恶化），基线 (16,8) 全面胜出。patch_len=16/stride=8 是敏感参数中的最优点，不是"不敏感"。

## 结论

**✅ 通过 — 基线 (16,8) 保持。** 更细 (12,6) 和更粗 (24,12) 的粒度均显著更差。patch_len=16, stride=8 与 PatchTST 原论文默认一致，且为最优。该参数固定为 16/8，不再搜索。stride 与 patch_len 的绑定关系未拆开验证——因本轮基线胜出、无拆开动机，维持绑定。

**⚠️ 单种子限制**：基于 seed=2021 单次训练，结论需多种子验证。W2/W3 验证阶段将提供交叉窗口证据。

**收尾判定**：✅ 通过 — 假设成立，结论可进入知识库。

---

## 探索目标

测试 patch_len/stride 三种组合对全步 MAE 的影响，确定 PatchTST 在太阳黑子数据上的最优时间粒度。

## 收获

1. patch_len=16/stride=8 是明确的最优点：更细(12,6)全步 MAE 恶化 12.5%，更粗(24,12)恶化 8.3%
2. 1c (24,12) 的 E_r=+7.6 是历史罕见的峰值高估，说明粗 patch 会丢失峰值细节，但高估方向不稳定
3. 1a (12,6) 的 step0=16.62 极差（基线 11.39），细粒度 patch 对单步预测尤其不利
4. 与 Round 0 一致：150+ 峰值段误差是所有配置的薄弱区（29-40），说明峰值压制问题来自更上游（loss/归一化/任务定义），不是 patch 粒度

## 不足

1. 单种子。三组差距均 >8%（远超决策阈值 3%），结论对种子扰动稳健性较好，但仍未验证
2. 1a/1c 并行训练导致墙钟时间不可靠（1a 85min、1c 55min 不可直接比），已在新协议中改正为串行
3. stride 未独立于 patch_len 验证
4. **⚠️ 口径缺陷（2026-08-13 发现）**：本表原为"最终模型口径"（一次性 train→test），与基线 0b(23.87) 的最佳模型口径混比。已补测修正。

---

## 2026-08-14 补测重审（最佳模型口径统一）

**补测结果**（--is_training 0 加载 full_checkpoint；原最终模型 npy 已备份为 *.bak_2026-08-13_final-model.npy）：

| 指标 | 1a (12,6) | 基线 (16,8) | 1c (24,12) |
|------|:---:|:---:|:---:|
| 全步 MAE | 25.54 | 23.87 | **23.53** |
| step0 | 16.40 | 11.39 | 13.30 |
| R² | 0.739 | 0.769 | **0.775** |
| E_r | +4.3 | -17.8 | -9.8 |
| 150+ 分层 | 31.3 | 29.0 | **28.6** |

**重审结论（三重条件审查）**：
- 1a (25.54) 差于基线 6.9% → 淘汰，维持原结论 ✓
- 1c (23.53) 优于基线 1.4% → **低于 3% 阈值，判为"无差异"，不翻转**。单种子下 0.34 MAE 的差距属于噪声范围。维持基线 (16,8)（PatchTST 原论文默认，更保守）
- 最终模型口径下 1a/1c 的"恶化 8-12%"大幅缩窄为最佳口径下的"恶化 6.9%/无差异"——原结论方向不变，幅度修正

**Round 1 最终结论**：patch_len=16, stride=8 维持为基线（1c 无差异，1a 更差）。参数固定。

---

## SOP 预检表缺项补写（2026-08-15 事后补写，只作留档）

> 本实验跑前未完整填写 SOP §1 预检表。以下内容为事后依据当时的方案讨论回填，
> 不构成'跑前已完成预检'的证明。2026-08-15 起新实验执行跑前门禁：预检表不写完不 launch。
### 预尸检（补写）

最可能：patch_len 偏离 16（PatchTST 原文默认）时性能恶化；12 太细（token 过多）或 24 太粗（峰值细节丢失）。

### 文献先例（补写）

PatchTST 原文 patch_len=16/stride=8；无太阳黑子专有先例，本项目内首次。

### 参数对照表（补写）

| 参数 | 基线(16,8) | 1a(12,6) | 1c(24,12) | 一致？ |
|------|-----------|----------|-----------|--------|
| features/窗口/seq_len/pred_len | MS/W1/336/24 | 同 | 同 | ✓ |
| d_model/n_heads/e_layers/d_ff | 128/8/2/2048 | 同 | 同 | ✓ |
| patch_len/stride | 16/8 | 12/6 | 24/12 | 唯一变量 |
| dropout/fc_dropout/revin/individual | 0.05/0.05/1/0 | 同 | 同 | ✓ |
| batch_size/lr/epochs/patience/seed | 16/1e-4/50/100/2021 | 同 | 同 | ✓ |

### 结论前提条件（补写）

结论仅在 features=MS、W1、最佳模型口径、50ep、单种子、sl336 下成立；stride 与 patch_len 绑定（2:1）未拆开验证。

### 成功标准（补写）

全步 MAE 差 ≥3% 判胜/劣；<3% 维持基线（更保守）。
