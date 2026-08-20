# EXP-18 Round 6 — activation 搜索 (W1窗口: Cycle 23)

> SOP §1 预检表（2026-08-14 跑前填写）→ 结果和结论跑后补。

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-14 |
| 实验ID | EXP-18-6a (relu) |
| 目的 | 在新基线(sl336, pl16/s8, el=2, df=2048, dm=128, do=0.2)上，测试 FFN 激活 gelu→relu |
| 改动变量 | activation（唯一变量）：gelu(基线) → relu |
| 对照基线 | EXP-18-5b（do=0.2，全步MAE=22.62, R²=0.789, E_r=-24.4） |
| 训练入口 | run_longExp.py（命令链模式） |
| features | MS |
| 口径 | 最佳 val 模型 |

---

## 可证伪假设

`--activation` 控制的是 FFN 内部非线性（2026-08-12 已修复 CLI 传递 bug）。文献普遍 gelu ≈ relu，预期差异 <3%。但本项目在 MS+StandardScaler+do=0.2 下从未测过。

- 如果 relu 与 gelu 差 <3%：符合预期，选 gelu（原论文默认）
- 如果 relu 恶化 ≥5%：relu dead neuron 问题在该数据上被放大，维持 gelu
- 如果 relu 意外改善 ≥5%：需警惕（超出文献预期），标记存疑并核实是否激活修复引入新问题

## 预尸检

1. 最可能：<3% 无差异——激活函数是微调级参数
2. 附带价值：本轮是 activation bug 修复后的首次正式测试。若 gelu 结果与历史 gelu 实验（同参数）差异大（>10%），说明 bug 修复引入了问题；若正常，则证明修复正确
3. do=0.2 新基线下 relu 与 gelu 的差异可能被 dropout 噪声掩盖（单种子）
（⚠️ 单种子弊端与解决想法见 data_pipeline.md 顶部警示：无重复标准差、3%边缘差异不可靠；想法=多种子重跑+bootstrap，不写死）

## 参数对照表

| 参数 | 基线(5b) | 6a | 一致？ |
|------|---------|-----|--------|
| features/窗口/seq_len/pred_len | MS/W1/336/24 | 同 | ✓ |
| d_model/n_heads/e_layers/d_ff | 128/8/2/2048 | 同 | ✓ |
| patch_len/stride | 16/8 | 同 | ✓ |
| dropout/fc_dropout/head_dropout | 0.2/0.2/0.0 | 同 | ✓ |
| **activation** | **gelu** | **relu** | ✗ 唯一变量 |
| revin/individual | 1/0 | 同 | ✓ |
| batch_size/lr/loss/epochs/patience/seed | 16/1e-4/mse/50/100/2021 | 同 | ✓ |

## 结论前提条件

features=MS、W1、最佳模型口径、50ep、单种子、sl336+pl16/s8+el=2+df=2048+dm=128+do=0.2 下成立。

## 成功标准

| 结果 | 判定 | 动作 |
|------|------|------|
| 差 <3% | 无差异 | 维持 gelu |
| relu 恶化 ≥5% | relu 有害 | 维持 gelu |
| relu 改善 ≥5% | 意外 | 标记存疑，核实后再决定 |

---

## 结果（跑后填）

| 指标 | 基线 gelu (5b) | EXP-18-6a (relu) |
|------|:---:|:---:|
| 全步 MAE | 22.62 | 22.96 |
| step0 | 11.01 | 12.51 |
| R² | 0.789 | 0.785 |
| E_r | -24.4 | -23.0 |
| 150+ 分层 | 28.4 | 29.0 |

## 结论（跑后填）

## 结论（跑后填）

**✅ 通过 — relu 差 1.5% <3% 无差异，维持 gelu。**

1. 回答了什么：gelu 与 relu 在该配置下无实质差异（22.62 vs 22.96），符合文献预期
2. activation bug 修复（2026-08-12）经本轮正式实验验证正常：--activation relu 正确生效（若未生效，relu 结果应与 gelu 逐位一致，实际 z-score 不同 0.184 vs 0.181，证明路径真正切换）
3. 维度关停：维持 gelu
4. 下一步：Round 7（individual），基线不变

**收尾判定**：✅ 通过。

## 自检

- [x] 只改 1 个变量（activation）
- [x] 命令链模式
- [x] 落盘 commit + push

---

## SOP 预检表缺项补写（2026-08-19 事后补写，只作留档）

> 本实验跑前未完整填写 SOP §1 预检表。以下内容为事后依据当时的方案讨论回填，
> 不构成'跑前已完成预检'的证明。2026-08-19 起新实验执行跑前门禁：预检表不写完不 launch。
### 文献先例（补写）

PatchTST 原文建议值与本研究基线一致；本维度（d_model/dropout/activation/individual）无太阳黑子专有文献先例，标注"本项目内首次干净对照"。
