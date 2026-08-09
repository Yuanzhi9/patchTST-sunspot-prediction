# 文献阅读笔记 — PatchTST 太阳黑子预测项目

> 阅读日期：2026-08-09
> 定位：Phase 0 文献奠基 — 系统性阅读笔记
> 共精读 14 篇，含综述 2 篇、方法论文 2 篇、DL+sunspot 预测 5 篇、传统/非线性预测 5 篇

---

## 总览表

| # | 论文 | 类别 | 核心结论 | 与你项目的关系 |
|---|------|------|---------|--------------|
| 1 | Hathaway (2015) The Solar Cycle | 🟢 综述 | SSN 数据 V1→V2 修正~20%；Waldmeier 效应 r=−0.7；地磁 precursor RMS≈27-29 SSN | 物理先验 + 数据预处理的指南 |
| 2 | Petrovay (2020) Solar cycle prediction | 🟢 综述 | Extrapolation 方法历史表现差；polar precursor 最优；给 NN/DL 留了一道窄门 | 你的方法分类定位 + 竞争基线 |
| 3 | DLinear (2023) AAAI | 🟡 方法 | 单层线性 > 所有 Transformer；shuffle 实验证明自注意力不用时序信息 | 解释你的天花板探测结果 |
| 4 | PatchTST (2023) ICLR | 🟡 方法 | Patching + CI；小数据集（ILI）上表现最差；自监督在小数据无益 | 你正在用的模型的局限分析 |
| 5 | Wang et al. (2021) RAA | 🔵 DL+sunspot | LSTM，整周期留出（实验设计最严谨），多步 RMSE=6.1-35.3，SC25 峰值 114.3 | 最优实验设计参照 + 精度基线 |
| 6 | Pala & Atici (2019) Solar Phys | 🔵 DL+sunspot | LSTM vs NNAR，rolling origin CV，RMSE=35.9，SC25 预测 167.3（已被证伪） | LSTM 基线 |
| 7 | Benson et al. (2020) Solar Phys | 🔵 DL+sunspot | WaveNet+LSTM，RMSE=2.93（DL 最优），SC25 预测 106（大幅低估） | DL 方法的精度上限参照 |
| 8 | Kumar & Kumar (2024) ASR | 🔵 DL+sunspot | CNN-BiGRU + GRC，90:10 切分→假预测；Smoothed 数据 RMSE=1.64 | 反面教材：切分方式决定精度真假 |
| 9 | Kumar & Kumar (2025) Solar Phys | 🔵 DL+sunspot | Hybrid Ensemble，60:40 切分→假预测；R²=0.9964 数值虚假膨胀；SC26 峰值 165.35 | 反面教材：堆模型≠好研究 |
| 10 | Xiong et al. (2021) MNRAS | 🟤 precursor | NUIST 自己人；综合前兆 + 多元回归 R=0.95；SC25 峰值 140.2；HWR 单峰局限 | 传统方法基线 + 同校参照 |
| 11 | Chapman (2026) ApJ | 🟤 precursor | Hilbert 相位时钟；switch-off SSN 与下周期峰值 r²=0.71；SC26 弱−中等 | 相位特征工程启发 |
| 12 | Gerçeker et al. (2025) Solar Phys | 🟤 nonlinear | Simplex Projection，raw 月均值；SC26 平滑峰值 137-146，双峰形态 | 非线性预测竞争方法 |
| 13 | ARIMA 预测（中文） | 🟤 classical | ARIMA(27,1,33)，严重过拟合；SC25 峰值预测 79（实际>120，误差>50%） | 无参考价值 |
| 14 | 太阳黑子统计预报（中文 docx） | 🟤 classical | 未成功解析 | 待处理 |

---

## 一、综述类

### 1. Hathaway (2015) "The Solar Cycle"

**基本信息**：Hathaway, D. H. (2015). Living Reviews in Solar Physics, 12, 4. NASA Ames. 100+ 页，242 篇引用。

**三句话**
- Q1 动机：全面综述太阳活动周期的观测、统计特征、物理机制和预测方法
- Q2 方法：系统回顾近 400 年黑子记录 + 10.7cm 射电、黑子面积、地磁指数、宇宙线等多指标
- Q3 与你项目：提供数据版本警示、预测 baseline benchmark、可嵌入模型的物理先验

**关键提取**

**SSN 数据版本问题（§3.2）**
- 1946 年 Waldmeier 接任 Zürich 台长后改变计数规则，给大黑子更高权重（2/3/5 而非 1）
- 估计 1946 年后的现代 SSN 被人为膨胀约 **20%**
- 2015 年 SILSO 推出 V2 数据纠正此偏差
- 对你的影响：用 V2 数据重新训练；1946 年是一个结构性断点

**平滑方法（§4.2）**
- 13-month running mean：高频透射率高达 20%，会保留准双年振荡的噪声
- 24-month FWHM Gaussian：高频透射率 <0.3%，极大值一致性远好于 13-month mean
- 建议：用 24-month FWHM Gaussian 平滑后训练，而非传统 13-month mean

**周期形状函数（§4.5, Eq.6）**
```
F(t) = A · (t-t₀)³ / b³ · [exp(-c·(t-t₀)²/b²)]⁻¹
```
平均参数：A=195, b=56, c=0.8, t₀=−4 月。c 可固定，b 可表为 A 的函数 → 退化为两参数函数。周期开始 2-3 年后参数即可可靠确定。

**Waldmeier 效应（§4.7）**
- 上升时间与振幅成反比：`Rise Time ≈ 35 + 1800/Amplitude`
- SSN 数据 r ≈ −0.7；Group SSN r ≈ −0.34；黑子面积数据中**未观察到**
- 争议：可能是计数规则的人为产物（Dikpati 2008），而非物理本质
- 对你：可作辅助 loss 约束；但 V2 数据上需重新验证

**预测方法精度（Table 6）**
| 方法 | RMS 误差 (SSN) |
|------|-----------------|
| Thompson 地磁 disturbed days | 27.0 |
| Feynman 地磁 aa_I 前兆 | 28.6 |
| Ohl 地磁 aa 极小值 | 28.7 |
| Gleissberg 周期 | 42.1 |
| 气候学平均（无技能 baseline）| 54.4 |

**一句话**：地磁前兆是目前唯一经过定量检验的预测方法（RMS≈27-29 SSN），这是你的 PatchTST 必须显著击败的量化 baseline。

---

### 2. Petrovay (2020) "Solar cycle prediction"

**基本信息**：Petrovay, K. (2020). Living Reviews in Solar Physics, 17, 2. 93 页，423 篇引用。对 2010 版的大幅更新。

**三句话**
- Q1 动机：方法分歧 + SSN V2 修正 + Cycle 25 预报窗口，需要更新综述
- Q2 方法：将方法归类为 precursor / model-based / extrapolation，逐一评价在 Cycle 21-24 的实际表现
- Q3 与你项目：你的 PatchTST 属于 extrapolation 大类——这是历史上表现最差的一类，但 Petrovay 留了门

**方法分类（§1.5）**
| 类别 | 前提假设 | 历史表现 |
|------|---------|---------|
| Precursor | 每个周期是自洽的整体 | 最优，polar precursor 在 Cycle 21-24 全部正确 |
| Model-based | 物理过程可数值模拟 | SFT 模型校准后 ≈ precursor；纯发电机模型偏高 |
| **Extrapolation** | SSN 过程统计齐次（stationary）| 预报值分布 ≈ 猜气候均值 |

**Extrapolation 的根本局限**
1. 前提不成立：太阳活动不是平稳过程（Maunder Minimum 是质变）
2. 样本致命不足：只有 24 个完整周期
3. 噪声 vs 信号无法区分：不确定系统是低维混沌还是随机

**对 NN/DL 的评价（§4.3.4 + §5）**
- Calvo (1995) NN → Cycle 23 预测完全失败
- Maris & Oncica (2006) NN → 同样远离实际
- Petrovay 留门："因为整类差就否定每个具体方法是草率的。NN、SSA 等方法还没被足够多周期检验过"
- 核心挑战：证明能从短时间序列中学到真实非线性结构，而非过拟合噪声

**Sqrt/对数变换（§1.2.3）**
- sqrt(R) 是传统做法；但 Petrovay 明确指出：**物理底层参数未知**，变换选择不应伪装成物理推断
- 零值处理未专门讨论，但实际黑子数不达零（新旧周期重叠）

**Cycle 25 预测汇总（§6）**
- 大多数方法预测 SC25 在 SC24 的 ±20% 范围内
- Dynamo 模型预测略低于 SC24；precursor + SFT 预测相当或略高
- Attia et al. (2013) neuro-fuzzy 预测 90.7±8（对 Cycle 25 已偏得离谱）

**一句话**：你的 PatchTST 必须证明能从 SSN 序列中提取出传统外推方法无法捕捉的模式，baseline 是"猜气候均值"，竞争上限是 polar precursor 的 1σ scatter ≈ 15-25 SSN。

---

## 二、方法论文

### 3. DLinear — Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?"

**基本信息**：Zeng, Chen, Zhang, Xu. AAAI 2023. CUHK & IDEA.

**三句话**
- Q1 动机：自注意力排列不变 → 丢失时序信息。Transformer 对纯数值时间序列真的有效？
- Q2 方法：趋势−季节分解 + 两个独立线性层 = DLinear。在 9 个 benchmark 上 vs 5 种复杂 Transformer
- Q3 与你项目：从机制上解释了为什么你的天花板探测 DLinear-I > PatchTST

**DLinear 架构**
```
输入 X → 移动平均分解 → X_t (趋势) + X_s (残差)
预测 = W_s · X_s + W_t · X_t
```
参数量 2TL，通道独立（不建模变量间相关性）。

**关键实验结果**
- DLinear 比最佳 Transformer（FEDformer）好 20-50%
- ILI 数据集（966 时间步，最接近你的小样本场景）改善 48%
- **Shuffle 实验（Table 5，论文最有杀伤力的证据）**：随机打乱输入序列后，所有 Transformer 预测几乎不变（Exchange-Rate 上 FEDformer 下降仅 0.09%），而线性模型大幅退化。说明 **Transformer 在纯数值数据上根本不使用时序信息**

**为什么线性 > Transformer？**
1. 长时序预测只需趋势 + 周期两个低维信号 → 线性加权求和是最优编码器
2. Transformer 自注意力排列不变 → 丢失时序
3. 参数过多 → 在小样本上过拟合噪声
4. 长信号路径 → 周期信号的梯度被稀释

**局限**
- 存在突变点（change point）时线性模型力不从心
- 回望窗口太短时欠拟合

**一句话**：你的天花板探测 DLinear-I > PatchTST 不是偶然——Transformer 的自注意力机制在无语义的周期性数值时间序列上不仅多余，还会引入噪声。

---

### 4. PatchTST — Nie et al. (2023) "A Time Series is Worth 64 Words"

**基本信息**：Nie, Nguyen, Sinthong, Kalagnanam. ICLR 2023. IBM Research.

**三句话**
- Q1 动机：逐点 tokenization 无语义 + O(N²) 复杂度限制历史窗口长度；DLinear 挑战 Transformer 有效性
- Q2 方法：Patching（子序列 token）+ Channel Independence（每通道独立过 Transformer，权重共享）+ 自监督预训练
- Q3 与你项目：你正在用的模型，需要理解其在小样本场景下的局限

**Patching 机制**
```
N = ⌊(L − P) / S⌋ + 2
```
默认 P=16, S=8, L=336。输入 token 数从 L 降到 ≈ L/S，复杂度降 S² 倍。

**Channel Independence**
- 每个通道的序列独立进入同一个 Transformer backbone（权重共享）
- 不同通道间无 cross-attention，无直接特征交互
- 对小数据更鲁棒：CI 在训练数据减少时比 channel-mixing 退化更慢

**小数据表现**
- ILI（966 步，7 特征）：所有数据集中表现最差。MSE 1.52-1.53，方差大
- 自监督预训练在 ILI 上完全没有实验（数据太少，没法做）
- 原文对小数据集用缩小版模型（H=4, D=16, F=128），建议你也这么做

**对你的数据的判断**
- month_sin + month_cos + ssn：3 个特征太少，channel-mixing 只会增加参数无收益。CI 方向基本正确
- 3600 个时间点 ≈ ILI 的 3.7 倍，但仍属小数据。必须用小模型配置
- 自监督预训练大概率无益

**一句话**：PatchTST 在充足数据下优势明显，但在最小数据集 ILI 上表现最差——你的 3600-sample 太阳黑子数据需要小模型、大 dropout、不碰自监督。

---

## 三、DL + 太阳黑子预测

### 5. Wang et al. (2021) — LSTM

**基本信息**：Wang, Li, Guo (2021). RAA, 21, 012. 中南大学。

**三句话**
- Q1：现有方法预测值分散、误差大
- Q2：优化 LSTM（隐节点 19, batch 20），整周期留出预测 Cycle 22-25
- Q3：**实验设计最严谨的 DL+sunspot 论文**，你的 PatchTST 应该对标它的切分方案

**实验设计**
- Silso V2 13-month smoothed SSN，1750 年起
- **单变量 SSN**，无多特征
- **整周期留出测试**：目标周期之前的所有数据训练，目标完整周期作测试
- 多步预测：720 月（60年）→ 72 月（6年）
- 10 次独立运行取平均

**精度**
| 周期 | 多步 RMSE | 峰值相对误差 |
|------|----------|-------------|
| Cycle 22 | 35.3 | 17.2% |
| Cycle 23 | 28.8 | 6.9% |
| Cycle 24 | 12.1 | 3.0% |
| Cycle 25 预测 | — | 峰值 114.3 (2023) |

**判断**
- 优点：整周期留出是真预测（测试集是完整未见周期）
- 不足：只测 3 个周期，样本太小；未报告 R² 或 MAE；只用单变量
- RMSE 随周期振幅减小而下降（Cycle 22 振幅大→RMSE 大），需归一化指标

**一句话**：实验设计最严谨的参照——你的 PatchTST 应该也用整周期留出方案。

---

### 6. Pala & Atici (2019) — LSTM vs NNAR

**基本信息**：Pala & Atici (2019). Solar Physics. 对比 LSTM, NNAR, ARIMA, Naive。

**三句话**
- Q1：系统对比 DL 与经典方法在 SSN 上的表现
- Q2：LSTM 2 层 50 单元 + rolling origin CV；预测 10 年 (120 月)
- Q3：早期 DL+sunspot 基线

**实验设计**
- Silso V2 月均 SSN, 1749-2018
- Rolling origin CV（6-slice / 12-slice）
- 单变量

**精度**
- LSTM RMSE = 35.9（最佳）
- NNAR = 42.41, ARIMA = 45.60
- SC25 预测峰值 167.3 (2022.07) —— 已被证伪

**判断**：模型容量偏小（2层50单元）。SC25 预测和实际偏差大。零值用前一非零值填充可能引入系统性偏差。

**一句话**：LSTM 基线 RMSE=35.9，是 PatchTST 需要显著超越的早期 DL 基准。

---

### 7. Benson et al. (2020) — WaveNet+LSTM

**基本信息**：Benson et al. (2020). Solar Physics. WaveNet + LSTM。

**三句话**
- Q1：WaveNet 的多层膨胀卷积可以指数级扩大感受野
- Q2：10 层 WaveNet (dilation 1→512) + 1 层 LSTM；5-fold TimeSeriesSplit CV；同时预测 SSN 和总黑子面积
- Q3：**当前 DL 方法最优结果**，RMSE=2.93

**实验设计**
- Silso V2 月均 SSN, 1749-2019
- 输入 528 月 (4 周期) → 输出 132 月 (1 周期)
- 5-fold TimeSeriesSplit
- 单变量

**精度**
- WaveNet+LSTM RMSE = **2.93**
- 单 LSTM = 4.42, 1DConv+LSTM = 3.89
- SC25 预测峰值 **106 ± 19.75**, 时间 2025.03 —— 实际大幅低估
- TSA 峰值 2022.05 与 SSN 峰值 2025.03 差近 3 年（两者实际高度同步）

**判断**：架构有物理直觉（膨胀卷积→长记忆），但单靠单变量 SSN 外推的本质上限仍在。TSA 和 SSN 预测互相矛盾暴露出不稳定性。

**一句话**：RMSE=2.93 是单变量 SSN 预测的 DL 当前最优，但 SC25 预测被实际大幅低估，提示纯数据驱动外推有根本天花板。

---

### 8. Kumar & Kumar (2024) — CNN-BiGRU + GRC

**基本信息**：Advances in Space Research, 73, 4342-4362。CNN-BiGRU + 梯度残差修正。

**三句话**
- Q1：残差未被处理导致预测不准
- Q2：CNN-BiGRU 基础预测 + AdaBoost 残差修正
- Q3：实验设计有致命漏洞

**关键漏洞**
- **90:10 时间序列切分**：测试集紧接训练数据末尾，测的是"序列续写"而非"预测新周期"
- 13-month smoothed SSN 上 RMSE=1.64, MAPE=0.06% —— 问题本身因为平滑+短步长变得 trivial
- SC25 预测峰值 143.6 (2024)

**一句话**：GRC 思路有创意，但 90:10 切分让所有精度数字虚假膨胀——这不是真预测，是序列自相关的 trivial 续写。

---

### 9. Kumar & Kumar (2025) — Hybrid Ensemble

**基本信息**：Solar Physics, 300, 100。4 种混合 DL + Ensemble。

**三句话**
- Q1：复杂混合架构能提升预测精度
- Q2：CNN-DilatedLSTM-BiLSTM-GRU 等 4 种混合模型 + H2&H4 取平均的 Ensemble
- Q3：模型堆砌的典型作品

**关键漏洞**
- **60:40 时间序列切分**：比 2024 篇更差，测试集和训练共享同周期
- 13-month smoothed SSN 上 R²=**0.9964** —— 红牌信号，问题无难度
- Baseline LSTM R²=0.9953，架构建再复杂收益微乎其微
- SC26 预测 165.35 (2036)，方法学上不可信（迭代预测 12 年无不确定性区间）
- Friedman ranking + Holm 校正的统计测试做得全，但评估问题的本质是错的

**一句话**：架构复杂度飙升但实际收益微薄，精度数字因切分方式虚假膨胀。

---

### 10. Xiong et al. (2021) — 南信大 Precursor + 回归

**基本信息**：Xiong, Lu, Zhao, Sun, Gao (2021). MNRAS, 505, 1046-1052. 南信大空间天气研究所。**你自己学校的论文**。

**三句话**
- Q1：单一前兆信噪比不够，需要综合多前兆参数
- Q2：高斯滤波平滑 → 4 个前兆参数 + 交叉项多元回归 → HWR 函数给完整轮廓
- Q3：传统方法基线 + 同校参照

**实验设计**
- 月均 SSN (1867.02-2020.10, NASA OMNIWeb)
- 高斯滤波 σ=3.6（替代 13-month mean）
- 4 个前兆：RI_max(n-1), RI_min(n), skewness s(n-1), aa_max(n-1)
- 11-23 周建模 → SC24 验证；11-24 周 → 预测 SC25
- 仅 13~14 个样本做回归，但含交叉项和 6 个参数

**精度**
- 组合回归 R=0.9505
- SC24 回测峰值 121.3（实际 115.4，误差<6%）
- SC25 预测峰值 **140.2**, 时间 2024.03

**判断**
- 有一个逻辑漏洞：输入参数 RI_min(n)（目标周期谷值）在预测时未知——论文未解释如何处理
- HWR 是单峰函数，无法描述双峰结构（SC24 就是双峰）
- 13 个样本、6 个参数 → 过度参数化风险

**一句话**：同校工作，传统 precursor 方法的参考基线，但模型输入存在时序逻辑漏洞。

---

## 四、非 DL 预测方法

### 11. Chapman (2026) — Hilbert 相位新前兆

**基本信息**：Chapman, S. C. (2026). ApJ, 1003, 159. University of Warwick。

**三句话**
- Q1：找物理有据、能提前 ~7 年预测下一周期最大值的 SSN 前兆
- Q2：Hilbert 变换提取"太阳周期时钟"，发现 switch-off 时的 SSN 与下一周期峰值线性相关
- Q3：相位特征工程启发；SC26 定性预测

**实验设计**
- Silso V2 月均 SSN (1749-2026)
- 13 月平滑 − 40 年趋势 → detrended → Hilbert 解析相位 → switch-off 时间
- 全量回测 Cycle 1-25 (25 个样本做线性回归)

**精度**
- Switch-off SSN → 下一周期峰值：r²=0.71, ρ=0.84
- 下降期时长 → 下一周期峰值：r²=0.48（弱得多）
- 平均提前量：6.9 年 (σ=1.4 年)
- **SC26 预测**：弱-中等周期，弱于或约等于 SC25

**判断**
- 物理意义强：switch-off 对应日冕形态转换、AR 纬度越过 15°
- 精确预测要等 2.5-3.5 年 switch-off 实际发生
- 论文发现强/弱周期形状是二分的（不是连续的），但外推方案仍用平均形状

**一句话**：Hilbert 相位给了太阳周期一个客观时钟，switch-off SSN 是 r²=0.71 的新前兆，SC26 大概是弱-中等周期。

---

### 12. Gerçeker et al. (2025) — Simplex Projection

**基本信息**：Gerçeker, Kilcik, Ozguc, Yurchyshyn (2025). Solar Physics, 300, 169。

**三句话**
- Q1：平滑数据丢失高频动力学信息；需要不假设线性/平稳的非线性方法
- Q2：EDM/Simplex Projection，对**未平滑月均 SSN**做相空间重构 + 最近邻加权外推
- Q3：非线性预测竞争方法，raw 数据直接用的理念与你一致

**实验设计**
- Silso V2 月均 SSN (1749.01-2024.12)
- Takens 嵌入 (E=2~10, τ=1~70 网格搜索)
- 单周期 + 双周期回测（SC20-24）
- 关键漏洞：split point 穷举搜索用了 SC25 前 60 个月的已知观测筛选最优参数 → 数据泄露

**精度**
- 回测 ρ_mean（月均 raw→平滑后）= 0.9959
- SC26 平滑峰值：**137.4-146.2**, 时间 2035.06
- 月均 raw 峰值：150.6-181.5
- 形态：双峰或微起伏平峰，类似 SC20；弱于 SC25、强于 SC24
- SC25 最小值：2030 年中期

**判断**：用 raw 数据保留高频动力学是优点。方法透明（最近邻可检视）。但 split point 优化有数据泄露——实际预测场景中你没有 SC25 的任何已知观测。

**一句话**：非线性的 Simplex Projection 预测 SC26 平滑峰值 137-146，方法透明，但参数优化存在数据泄露。

---

### 13. ARIMA 预测（中文）

**基本信息**：未标注作者和期刊。中文 ARIMA 预测报告。

**实验设计**
- 月均 SSN, 1947.01-2020.04（仅 73 年，丢弃了 1749-1946 的 200 年数据）
- ARIMA(27,1,33)：60 个参数，不足 900 个月数据
- 无训练/测试划分——全量拟合后直接外推
- DW=1.9989（宣称残差白噪声通过检验）

**精度**
- SC25 预测峰值约 **79** (2024.05-06)
- 实际 SC25 13-month smoothed SSN 已超 120 → **误差 >50%**

**判断**：严重过度参数化 + 丢弃大量历史数据 + 无交叉验证 = 无参考价值。典型"统计检验全过但预测完全不准"的案例。

**一句话**：ARIMA(27,1,33) 过度拟合 + 数据过短，SC25 预测误差 >50%，无参考价值。

---

### 14. 太阳黑子第 25 期的统计预报（中文 docx）

未成功解析文本，待用 python-docx 处理。

---

## 五、横向总结与对你项目的综合意义

### 5.1 实验设计——最该学的和最该避的

| 做对了（学）| 做错了（避）|
|------------|-----------|
| Wang (2021)：整周期留出作测试 = 真预测 | Kumar (2024/2025)：90:10 或 60:40 时序切分 = 假预测 |
| Pala & Atici：Rolling origin CV | ARIMA 中文：无划分，全量拟合后外推 |
| Benson：TimeSeriesSplit CV | Wang：只用 3 个测试周期（太少） |

**你的 PatchTST 应该：整周期留出 + Leave-one-cycle-out CV + 多 seed 评估**

### 5.2 精度基线——你需要在哪些数字上赢

| 方法 | 精度 (SSN 单位或 RMSE) | 类型 |
|------|------------------------|------|
| 气候学平均（无技能 baseline）| RMS≈54.4 | 最底线 |
| Ohl/Feynman/Thompson 地磁前兆 | RMS≈27-29 | 传统 benchmark |
| Wang LSTM 多步预测 | RMSE=12.1-35.3 | DL 基线 |
| Pala LSTM | RMSE=35.9 | DL 基线 |
| **Benson WaveNet+LSTM** | RMSE=**2.93** | DL 最优 |
| **Polar precursor** | 1σ scatter≈15-25 | 物理最优 |

PatchTST MAE=23.87 已经接近地磁前兆的 RMS 范围，但离 polar precursor 还有距离。

### 5.3 一个绕不开的事实

所有在 2021 年前发表、预测 SC25 峰值的论文（至少 5 篇），**没有一个准的**。预测范围从 79（ARIMA）到 167.3（Pala LSTM），实际约 120-140（取决于用 raw 还是 smoothed）。这说明：**仅靠历史 SSN 序列做外推存在一个刚性的信息上限。** 你的项目要想有实质贡献，要么突破这个上限（引入物理先验、多变量），要么系统刻画这个上限（可行性边界研究）。

### 5.4 可做的特征工程方向（来自文献）

- **24-month FWHM Gaussian 替代 13-month running mean** 做预处理（Hathaway 2015）
- **Hilbert 相位作为周期位置特征**：表示"当前在周期的哪个位置"（Chapman 2026）
- **地磁 aa 指数**作为辅助输入通道（Hathaway + Petrovay）
- **Waldmeier 效应作为 loss 约束**：惩罚"高振幅+慢上升"的预测
- **sqrt 变换拉平方差**：传统做法，但 Petrovay 提醒底层物理参数未知

### 5.5 DLinear vs PatchTST 的定论

DLinear 原文的 shuffle 实验 + 你的天花板探测共同指向一个结论：**在太阳黑子这种强周期性、无语义的数值时间序列上，Transformer 的自注意力机制是多余的。** 线性模型（DLinear-I）能提取趋势和周期这两个最核心的低维信号，参数少 → 不过拟合。PatchTST 的 patching + channel independence 在方向上是合理的，但自注意力部分可能是负资产。

---

*本文件随文献阅读推进持续更新。下一批待处理：Xiong (2021) NUIST 论文的深入分析、docx 文件解析、Phase 0 结束判定。*
