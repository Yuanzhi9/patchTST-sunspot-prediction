# EXP-18 Round 7 — individual head 搜索 (W1窗口: Cycle 23)

> SOP §1 预检表（2026-08-14 跑前填写）→ 结果和结论跑后补。

| 字段 | 内容 |
|------|------|
| 日期 | 2026-08-14 |
| 实验ID | EXP-18-7a (ind=1) |
| 目的 | 在新基线(sl336, pl16/s8, el=2, df=2048, dm=128, do=0.2, gelu)上，测试 individual 0→1 |
| 改动变量 | individual（唯一变量）：0(共享head基线) → 1(每通道独立head) |
| 对照基线 | EXP-18-5b（do=0.2，全步MAE=22.62, R²=0.789） |
| 训练入口 | run_longExp.py（命令链模式） |
| features | MS |
| 口径 | 最佳 val 模型 |

---

## 可证伪假设

enc_in=3（sin/cos/ssn）+ MS 模式。individual=1 时每通道独立 head，SSN 通道的 head 不受 sin/cos 干扰。但 MS 模式只取 SSN 通道输出，sin/cos 的 head 输出被丢弃（浪费）。两种可能：独立 head 让 SSN 通道表示更纯（改善），或共享 head 提供通道间信息交换（改善）。

- 如果 ind=1 改善 ≥3%：SSN 通道独立 head 有效
- 如果差 <3%：无差异，维持 ind=0（更简洁）

## 预尸检

1. 最可能：<3% 无差异——MS 模式下 ind 的影响有限
2. 若 ind=1 训练明显变慢（多 2 个 head 的梯度计算），时间成本不划算

## 参数对照表

| 参数 | 基线(5b) | 7a | 一致？ |
|------|---------|-----|--------|
| features/窗口/seq_len/pred_len | MS/W1/336/24 | 同 | ✓ |
| d_model/n_heads/e_layers/d_ff | 128/8/2/2048 | 同 | ✓ |
| patch_len/stride | 16/8 | 同 | ✓ |
| dropout/fc_dropout/head_dropout | 0.2/0.2/0.0 | 同 | ✓ |
| activation | gelu | gelu | ✓ |
| revin | 1 | 1 | ✓ |
| **individual** | **0** | **1** | ✗ 唯一变量 |
| batch_size/lr/loss/epochs/patience/seed | 16/1e-4/mse/50/100/2021 | 同 | ✓ |

## 结论前提条件

features=MS、W1、最佳模型口径、50ep、单种子、sl336+pl16/s8+el=2+df=2048+dm=128+do=0.2 下成立。**不推广到 M 模式**（M 模式下 ind 行为完全不同）。

## 成功标准

| 结果 | 判定 | 动作 |
|------|------|------|
| 改善 ≥3% | ind=1 有效 | 更新基线 |
| 差 <3% | 无差异 | 维持 ind=0 |

---

## 结果（跑后填）

| 指标 | 基线 ind=0 (5b) | EXP-18-7a (ind=1) |
|------|:---:|:---:|
| 全步 MAE | 22.62 | 24.13 |
| step0 | 11.01 | 13.26 |
| R² | 0.789 | 0.764 |
| E_r | -24.4 | -14.0 |
| 150+ 分层 | 28.4 | 31.5 |

## 结论（跑后填）

## 结论（跑后填）

**✅ 通过 — ind=1 恶化 6.7%，维持 ind=0。**

1. 回答了什么：MS 模式下共享 head 优于独立 head（22.62 vs 24.13）——共享 head 的通道间信息交换对 SSN 预测有益
2. 维度关停：维持 individual=0
3. 下一步：搜索完成，进入验证阶段（最优配置 × W1/W2/W3）

**收尾判定**：✅ 通过。

## 自检

- [x] 只改 1 个变量（individual）
- [x] 命令链模式
- [x] 落盘 commit + push
