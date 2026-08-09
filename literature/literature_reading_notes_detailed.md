# 结构化文献笔记（详注版）

> 阅读日期：2026-08-09
> 对应 `literature_reading_notes.md`，此文件为每篇论文的完整深度笔记
> 每篇包含：三句话笔记、实验设计细节、关键数值、方法论判断、与你项目的直接关系

---

## 1. Hathaway (2015) "The Solar Cycle"

**完整引用**：Hathaway, D. H. (2015). The Solar Cycle. *Living Reviews in Solar Physics*, 12, 4. DOI: 10.1007/lrsp-2015-4. NASA Ames.

### 1.1 三句话笔记

- **Q1 作者想解决什么问题？**
  不是解决单一问题。Hathaway 的目标是把太阳活动周期从观测数据、周期特征、长期变化、短期变化、预测方法到物理解释的全链条知识做一个权威整理，为太阳物理和发电机理论研究者提供一个"现状地图"。

- **Q2 他们怎么解决的？**
  通过对近 400 年太阳黑子记录（以及其他活动指标：10.7 cm 射电流量、黑子面积、地磁指数、宇宙线、放射性同位素）的系统回顾，从数据质量→周期特征统计→物理规律（Waldmeier 效应、Spörer 定律等）→预测方法误差对比，逐层推进。最后落到 Babcock 发电机模型的困境上。

- **Q3 跟我的项目有什么关系？**
  直接相关。你的 PatchTST 做太阳黑子周期预测，这篇综述给出了：(a) 你需要用的数据版本问题（V1→V2 改了 20%），(b) 你要预测的目标曲线的数学形式（§4.5 Eq. 6），(c) 你应该对比的 baseline 方法及其误差 benchmark（Table 6），(d) 可以嵌入模型的物理先验（Waldmeier 效应、Gnevyshev-Ohl 规则、周期-振幅关系、地磁前兆）。

### 1.2 核心发现/概念提取

#### §3.2 "Revised Sunspot Numbers" —— SSN 数据版本修正

**具体改了什么：**
- 1946 年 Waldmeier 接任 Zürich 台长后，改变了黑子计数规则：不再每个黑子计 1，而是给大黑子更高权重（2、3 或 5）。
- Svalgaard (2013) 对比有/无加权两种计数方式，估计 **1946 年以来的现代 SSN 被人为膨胀了约 20%**。
- 同期存在的 Group Sunspot Number (Hoyt & Schatten 1998) 与 International SSN 在 20 世纪前有系统性偏差。
- 2015 年 SILSO（比利时皇家天文台）推出 **Version 2（V2）** 的 SSN 数据，纠正了 Waldmeier 加权带来的膨胀，并重新校准了历史数据。
- 论文中提及但未完全覆盖 V2（因为 2015 年修订中增加了 §3.2），但明确指出"正在进行的社区大协调"（当时正在进行的大规模数据修正项目）。

**对你项目的影响：**
1. 如果你的数据源是 2015 年前下载的 SILSO V1 数据（现在很多公开教程和 Kaggle 数据集是 V1），那么 1946 年后的振幅被系统性地高估了 ~20%。用 V2 数据重新训练。
2. 如果你的 PatchTST 用整个 1749-至今的 SSN 序列训练，V1→V2 意味着数据在时间轴上存在一个**结构性断点**（1946 年），这可能在模型训练中引入虚假的模式跳跃。更严格的做法是：在 V2 数据上按 cycle 分 train/test，确保每个 cycle 内部数据版本一致。
3. 考虑用 Group Sunspot Number 作为额外的输入特征/对比数据通道。

#### §4.2 13-month Running Mean 平滑与 24-month FWHM Gaussian 平滑

**13-month running mean 公式：**
> "centered on a given month with equal weights for months –5 to +5 and half-weight for months –6 and +6"

即滑动窗口为 13 个月（当前月 ±6 个月），中间 11 个月权重相等，两端（±6）权重减半。本质是带锥形端点的 boxcar 平滑。

**24-month FWHM Gaussian 平滑公式：**

Eq. (4) & (5)：
```
W(t) = e^(-t² / 2a²) - e^(-2) · (3 - t² / 2a²)
其中 -2a+1 ≤ t ≤ +2a-1
```
- `t` 以月为单位，`2a` 是 FWHM（半峰全宽），取 24 时即为 24-month FWHM Gaussian。
- 滤波器在端点处加了锥形（使权重和其一阶导数在端点归零），避免截断伪迹。
- (注意：这个公式与 Hathaway et al. 1999 中给出的略有差异，本文是修正版。）

**两者优劣比较：**
| | 13-month running mean | 24-month FWHM Gaussian |
|---|---|---|
| 高频抑制 | 差——对 >1 cycle/year 的信号透射率高达 **20%** | 优——对 >1 cycle/year 的信号透射率 **<0.3%** |
| 半周期抑制 | 差 | 对 0.5 cycle/year 透射率仅 ~1% |
| 极大值一致性 | 不同指标给出的极大值日期范围较大 | 一致性远好于 13-month mean，日期范围约减半 |
| 双峰问题 | 容易保留 1-3 年尺度的短周期波动，导致双极大值 | 24-month 窗足够宽，有效滤掉双峰 |
| 使用传统 | 历史惯性强，Waldmeier & McKinnon 表格均基于此 | 更科学，但社区采纳度不够 |

**对你项目的建议：**
1. **用 24-month FWHM Gaussian 平滑后的数据训练 PatchTST**，不要用 13-month mean。因为 13-month mean 保留了过多高频噪声（20% 透射率意味着你的输入数据里有你不想要的 1-3 年准双年振荡信号），模型会花参数学拟合这些噪声而不是真正的周期结构。
2. 如果你在对比其他文献的 benchmark 数值（比如各 cycle 的振幅），要确认他们用的是哪种平滑——结果会差出几十个 SSN 单位。

#### §4.5 太阳周期形状的数学描述

**平均周期形状（cycles 1-23 归一化）：**
- 上升时间：约 **48 个月**（4 年）
- 下降时间：约 **84 个月**（7 年）
- 明确不对称：升快降慢（Waldmeier 1935 首先指出）

**Hathaway et al. (1994) 的 cycle shape 函数（Eq. 6）：**

```
F(t) = A · (t - t₀)³ / b³ · [ exp( -c · (t - t₀)² / b² ) ]^(-1)
```

参数含义：
- `A`：振幅（峰值 SSN）
- `t₀`：开始时间（相对于极小值的时间偏移）
- `b`：上升时间尺度（控制上升的陡峭程度）
- `c`：不对称性参数（控制上升 vs 下降的相对形状）

**平均周期拟合值：** A = 195, b = 56, c = 0.8, t₀ = -4 个月。

**关键简化：** c 可以固定为常数，b 可以表达为振幅的函数——因此整条曲线退化为**两参数**函数（A 和 t₀），这两个参数在周期开始 2-3 年后即可可靠确定。

**对你项目的意义：**
1. **特征工程**：Eq. (6) 不需要你从零学，Hathaway 团队已经验证过这个函数形式能极好地拟合每个 cycle。如果你想把物理先验灌进 PatchTST：可以用 Eq. (6) 的参数 `(A, b)` 作为辅助特征，或用拟合残差作为模型的预测目标。
2. **输出约束**：PatchTST 预测的结果如果严重偏离 Eq. (6) 的两参数形式，几乎一定是过拟合噪声。

#### §4.7 Waldmeier 效应的定量表述

**定义：** 上升时间（从极小值到极大值）与周期振幅**成反比**——振幅越大的周期，上升越快。

**定量公式（Eq. 7）：**
```
Rise Time (months) ≈ 35 + 1800 / Amplitude (SSN)
```

**数据支撑：**
- 使用 International SSN：上升时间与振幅的**反相关系数 r ≈ -0.7**
- 使用 Group Sunspot Number 时：反相关减弱到 **r ≈ -0.34**
- 在**黑子面积**数据中：**未观察到** Waldmeier 效应
- 10.7 cm 射电流量也表现出 Waldmeier 效应，但极大值延迟约 **6 个月**

**物理解释与争议：**
1. **Dikpati et al. (2008b)** 在论文中质疑 Waldmeier 效应可能是 Wolf 黑子数**定义的产物**（artifact），而非太阳发电机本身的特性。他们的论据：效应在 Group SSN 中大幅减弱，在面积数据中消失。
2. **Cameron & Schüssler (2007)** 的解释：周期重叠 + Waldmeier 效应共同产生了地磁前兆（precursor）关系的表象。
3. 也有观点认为并非 artifact——大周期中黑子的纬度分布更宽，到达峰值更快符合物理直觉。

**对你项目的影响：**
1. 这是一个很强的**物理先验约束**：如果你用 PatchTST 预测周期振幅（long-horizon forecast），可以设计一个辅助 loss 项惩罚那些"预测出高振幅但上升很慢"的结果。
2. 在分析模型预测的 explainability 时，可以用 Waldmeier 关系作为 sanity check——如果模型预测违反这个关系，意味着模型可能学了假关联。
3. 但要留个心眼：这个效应在面积数据里不成立。如果后续用 SSN V2 数据（已去 Waldmeier 偏倚）检验，相关可能也会减弱。**在 V2 数据上重新验证 Waldmeier 效应再决定是否用它做约束。**

### 1.3 数值/公式/表格

#### 周期振幅范围与长度

| 指标 | 数值 |
|---|---|
| 平均振幅（cycles 1-23, 13-month mean SSN） | **114.1 ± 40.4** |
| 振幅范围（过去 400 年） | **0（Maunder Minimum）→ 201.3（cycle 19）** |
| Dalton Minimum 振幅 | **49.2 和 48.7** |
| 近年大振幅序列（cycles 18-22） | 151.8 / 201.3 / 110.6 / 164.5 / 158.5 |
| Cycle 23 峰值 | **120.7** (V1 SSN, 2000/04) |
| Cycle 24 峰值 | **81.9** (2014/04) |
| 平均周期长度 | ~11 年（**132 个月**），标准差 **~14 个月** |
| 平均上升时间 | **~48 个月** |
| 平均下降时间 | **~84 个月** |

#### Table 6 —— 各预测方法 RMS 误差（cycles 19-23）

这是对你项目最重要的一张表。它给出了所有 baseline 方法的定量 benchmark。RMS 误差单位是 SSN。

| 预测方法 | RMS 误差 (SSN) |
|---|---|
| **Thompson's Method**（地磁 disturbed days） | **27.0** |
| **Feynman's Method**（地磁 aa_I 前兆） | **28.6** |
| **Ohl's Method**（地磁 aa 极小值） | **28.7** |
| Gleissberg Cycle（7-8 周期波动） | 42.1 |
| Three Cycle Sawtooth | 49.0 |
| Secular Trend（长期趋势外推） | 49.3 |
| Amplitude-Period（周期-振幅关系） | 49.6 |
| Maximum-Minimum（极大-极小关系） | 51.2 |
| Even-Odd（Gnevyshev-Ohl 规则） | 52.0 |
| **Mean Cycle（气候学平均——无技能 baseline）** | **54.4** |

**三种地磁 precursor 的具体误差：**

| 方法 | 公式 | 相关 r | 误差 |
|---|---|---|---|
| **Ohl** | `Rmax = 7.95·min(aa) ± 18` | r = 0.93, r² = 0.86 | RMS = **28.7** |
| **Feynman** | `Rmax = 12.1·Max(aa_I) ± 16` | r = 0.95, r² = 0.91 | RMS = **28.6** |
| **Thompson** | `Rmax(n)+Rmax(n+1) = 0.55·NDD(n) ± 28` | r = 0.95, r² = 0.91 | RMS = **27.0** |

其中：
- `aa_I = aa - aa_R`，`aa_R = 10.9 + 0.097·R`（去除与 SSN 相关的成分后留下的"行星际"分量）
- NDD = Number of geomagnetically Disturbed Days (Ap ≥ 25) in the previous cycle

#### 其他关键统计数字

- 黑子面积与 SSN 的相关系数：**r = 0.994, r² = 0.988**
- 周期 n 的周期长度与周期 n+1 的振幅的反相关：**r = -0.68, r² = 0.46**
- 周期长度与同期振幅的反相关：弱得多，**r = -0.37**（远不如 Waldmeier 显著）
- 上升拐点出现在极小值后 **2-3 年**，这是曲线拟合 prediction 变得可靠的临界点
- N-S 半球相位差不超过 **~10 个月**

### 1.4 与项目的直接关系

**可能影响 PatchTST 实验设计的信息：**
1. **数据预处理不可忽视平滑方法的选择。** 13-month running mean 保留太多高频噪声（20% 透射率），相当于你的模型得到的是被高频污染的"脏"曲线。用 24-month FWHM Gaussian 预平滑，可以让模型专注于低频的周期动力学。建议做 ablation：对比两种平滑对预测精度的影响。
2. **预测 horizon 的选择应该对齐物理时间尺度。** 论文明确指出：在极小值后 **2-3 年内预测不可靠**；cycle-to-cycle 的振幅预测需要用前兆方法。如果你的 PatchTST 用过去 96 个月的 SSN 预测未来 6-12 个月的 SSN（短期继续预报），那很合理。但如果用历史周期振幅预测下一个周期的振幅（跨 cycle），那输入特征里面必须有地磁指数（aa）、极区磁场等前兆量。
3. **Benchmark 设定必须包含 Table 6 中的方法。** 尤其 Ohl 法——3 行代码，r=0.93，RMS=28.7。你的深度学习模型必须 beats this。不做这个对比，审稿人不会接受。
4. **数据版本统一。** 全文（尤其 Figure 6）反复强调 SSN 数据版本之间的差异。如果你混用不同版本的数据（比如训练用 V1，测试某个 cycle 用 V2），相当于在不同标尺上训练和评估。
5. **周期边界不是绝对的。** §4.1 详细讨论了 minima 在不同指标下可以差几个月甚至半年。这对时间序列分割（按 cycle 切分 train/val/test）有影响。

**可以作为特征工程方向的物理学先验：**
1. **Waldmeier 效应作为辅助 loss / 特征**：`rise_time ～ const + k/amplitude`。在训练时可以加入 soft constraint。
2. **Eq. (6) 两参数函数作为趋势项**：用 Hatheway 的 cycle shape 函数拟合出每个 cycle 的 (A, b) 参数，把函数值作为趋势特征输入；把残差作为 PatchTST 的目标——DLinear 的 trend-seasonal 分解思路完全可以用 Eq. (6) 来代替移动平均趋势。
3. **地磁指数（aa/Ap）作为辅助输入通道**：§7.3 和 §3.9 详细给出了 aa 指数与 SSN 的关系。你的 PatchTST 可以做成多变量输入（SSN + aa + 10.7 cm 射电流量），利用这些指标的领先-滞后关系提高预测。
4. **Gnevyshev-Ohl 规则的编码**：偶数-奇数 cycle 对的振幅关系（奇 > 偶）在历史上基本成立（disrupted by cycle 8-9 和 22-23）。可以把 `cycle_parity` 作为 categorical 特征。
5. **极区磁场作为 long-term 前兆**：§7.4 明确讨论了 polar field 的预测能力（predict Rmax(24) ≈ 75 ± 8，实际 81.9）。如果需要跨 cycle 预测，这是最物理的特征。

### 1.5 一句话总结

这篇论文最重要的一件事是：**太阳黑子周期的预测，目前唯一经过定量检验的、有实际预测技能（相对于气候学平均）的方法是基于地磁前兆（Ohl、Feynman、Thompson）、极区磁场前兆和简化发电机模型——它们的 RMS 误差大约在 27-29 SSN，相当于均值振幅的 25%，这是一个深度学习模型必须明确击败的定量 benchmark；而黑子数据本身的版本变化（V1→V2 的 20% 膨胀）意味着如果你不检查数据来源，你的模型可能在一条已经被物理学界放弃的过时标尺上做优化。**

---

## 2. Petrovay (2020) "Solar cycle prediction"

**完整引用**：Petrovay, K. (2020). Solar cycle prediction. *Living Reviews in Solar Physics*, 17, 2. 93 页，423 篇引用。对 2010 版的大幅更新（209→423 篇）。

### 2.1 三句话笔记

- **Q1 动机：** 太阳黑子数（SSN）序列是人类历史上最长的直接天文观测记录，太阳周期预报对空间气候和卫星运行至关重要；经过第 24 周的"教训"（各类方法分歧巨大），2015 年 SSN 数据的重大修订（v1→v2），以及 Cycle 25 预报窗口的开辟，这次升级版综述要回答的根本问题是——到底什么方法能预测太阳黑子振幅？三类方法各凭什么、各做得多好？

- **Q2 方法：** 这是一篇 Living Reviews 综述（literature review），非原创预测研究。它系统梳理了从 Bracewell (1953) 到 2019 年公开的约 420 篇文献，将太阳周期预报方法归纳为三大类（§1.5），逐一评估它们在 Cycle 21–24 的实际表现（§5），最后给出 Cycle 25 的早期预报汇总（§6）。作者本人是 polar precursor 方法的长期推动者。

- **Q3 与你的项目的关系：** 你用的 PatchTST（Transformer 时间序列预测模型）属于 extrapolation methods 大类。按 Petrovay 的分类和评价，这个大类在过去几个周期里表现最差——预报值的分布跟"猜气候均值"差不多。但这不等于你的项目没价值：Petrovay 明确说「不应该因为整类表现差就否定其中每个具体方法，特别是 SSA、phase space reconstruction、neural networks 这些新方法还没经历过几个周期的验证」。你的项目要回答的核心问题是：一个有现代架构（PatchTST）的深度学习模型，能不能突破 extrapolation 类方法的根本局限？

### 2.2 核心内容提取

#### §1.2.3 Alternating series and nonlinear transforms

文献讨论了三种变换：

**（1）幂次变换 R' = R^a （0 < a < 1）**
- 动机之一是**统计分布**：原始黑子数分布远离高斯分布，且太阳周期剖面对正弦形状有强烈偏离（不对称的尖峰），许多标准分析方法依赖正态假设和正弦谱成分。
- a = 0.5（即 sqrt 变换）是最常用的取值。Waldmeier 常用的是 a 的对数极限（即 log R）。
- 动机之二是**物理对应**：黑子数 R 是人为构造的指数，真正底层的物理量可能是环向磁场强度 B 或其平方 B²（磁能）。但作者强调：**没有坚实的物理基础来推断底层参数到底是什么**——"our current understanding of the solar dynamo does not make it possible to guess what the underlying parameter is"。
- **零值处理**：文献没有直接讨论 log 变换中零值的处理方法（黑子数在最小值附近理论上可以为零，但实际从不达到零因为新旧周期重叠），但从 Bracewell (1988) 的 3/2 律可以看出他们在构造物理变量 R_B 时采用了 R 的 2/3 次方，即隐含 R > 0 的假设。

**（2）交变序列 R_±（Bracewell 1953）**
- 基于 Hale 极性规则，给奇偶 Schwabe 周期赋予正负交变符号，把 11 年周期扩展为 22 年周期。奇数周期通常取负号。
- 在极小值附近 R 不为零（因为两周期重叠约 1-2 年），所以交变序列在极小值处有轻微跳跃——在某些应用中还需要引入额外处理来规避。

**（3）Bracewell 3/2 律（Bracewell 1988）**
- R_± = 100 (R_B / 83)^(3/2)， 即 a = 2/3 的幂律。
- 当引入 rectified phase variable φ 补偿周期轮廓的不对称性后，R_B 几乎成为 φ 的正弦函数。
- 物理阐释：这个"3/2 律"意味着较大的黑子群不仅峰值面积大，被观测的时间也更长，因此对年均黑子数贡献不成比例地大。R_B 应该被认为正比于环向磁通量的总浮现量。

**对你的项目的意义：** sqrt 变换在太阳物理领域有传统、有解释。Bracewell 的 3/2 律给了 a=2/3 一个物理解释——但这仍然只是 empirical。如果你项目里用 Box-Cox 或机器学习自动搜索最优变换，那是在弥补原文献"我们不知道真实底层物理参数是啥"这个认知缺口。不过你要注意：**这个领域的物理共识是——底层参数未知，变换选择不应伪装成物理推断。**

#### §1.4.1 Secular activity variations

**Quantitative conclusions on secular variations:**

1. **Gleissberg 周期**：对 SSN 序列做 Gleissberg 滤波（1-2-2-2-1 权重），呈现约 **9–10 个 Schwabe 周期（约 90–110 年）** 的调制周期。

2. **已知的 secular 极值**：
   - Dalton Minimum：异常弱的 Cycle 5–7
   - Gleissberg Minimum：中等偏弱的 Cycle 12–16（大多数不到 1σ 低于长期均值）
   - **Modern Maximum**：强周期 17–23，覆盖 20 世纪下半叶

3. **关键争议——Modern Maximum 到底多强？**
   - SSN v1→v2 修订**大幅削弱**了 secular increase 的幅度
   - 最激进的 GSN 重建（Svalgaard & Schatten 2016）几乎抹掉了上升趋势
   - 但最新 GSN 重建（Chatzistergos et al. 2017）又显示出明显的长期增加趋势
   - **宇宙成因核素（cosmogenic isotope）** 记录毫不含糊地表明：20 世纪下半叶持续高水平太阳活动在数千年历史上没有先例
   - "目前这个问题正在被热烈争论"（hotly debated）

4. **Maunder Minimum（1640–1705）**：与大极小值（grand minima）相比，是一个**质变**而非量变。尽管黑子极其稀少，但 11/22 年周期仍然存在，只是幅值极低。解释 grand minima 的候选机制：混沌行为、随机涨落、双稳态发电机、AR tilt 涨落、"rogue" 黑子。

**对你的项目的意义：** 这段表明在几十年到百年的时间尺度上，SSN 序列决不是一个平稳时间序列——它有趋势、有调制周期、有 regime shift（Maunder Minimum 是定性不同的状态）。非平稳性是一切 extrapolation 方法（包括 PatchTST）的根本软肋——如果你训练数据只覆盖最近几个周期，那测试集可能落在完全不同的 regime。

#### §1.4.2 Does the Sun have a long term memory?

**核心结论：有长期记忆，但对周期到周期预测帮助有限。**

文献通过三条证据链论证：

**(a) 组合统计证据**
- Dalton Minimum（四个最低极大值中三个连续出现）在 24 个随机极大值序列中发生的概率仅约 **5%**。
- 千年长度代理数据表明 grand minima 和 grand maxima 的出现频率高于高斯统计预期。

**(b) Hurst 指数估计**
- 多篇重新标度极差分析（rescaled range analysis）和去趋势波动分析（detrended fluctuation analysis）一致给出：年及以上尺度 **H ≈ 0.85–0.88**，更短时间尺度 H ≈ 0.75。
- H > 0.5 表明太阳活动是 **persistent** 的——强/弱状态倾向于持续。
- 对于长度为太阳黑子记录的有限序列，Monte Carlo 实验证明 H > 0.7 统计显著。
- 但是 Oliver & Ballester (1998) 指出质疑。

**(c) 随机游走模型的反驳**
- Love & Rigler (2012) 曾拟合 ln R 做高斯随机游走，步长约 0.39——这套模型能"解释" Modern Maximum，但无法解释 Maunder Minimum，也无法解释活动水平有上界的事实（宇宙成因核素记录表明活动水平被 bounded from above）。

**作者的重点转折：**
> "The overwhelming evidence for the persistent character of solar activity and for the intermittent appearance of secular cyclicities, however, is not much help when it comes to cycle-to-cycle prediction... the associated errors will still be so large as to make the forecast of little use for individual cycles."

**对你的项目的意义：** PatchTST 利用的正是时间序列中的长期依赖（attention 机制）。Petrovay 承认长记忆存在，但说它不够用。你的模型要挑战的是——能不能从已知的 persistence 特征中挤出比传统外推方法更多的信息？H≈0.85 是关键先验约束。

#### §1.5 三类预测方法的精确定义和分类标准

Petrovay (2020) 在 §1.5 中修订了 v1 版的粗分类，提出了一个连续谱框架：

```
(a) Internal empirical precursors —— 仅依靠 SSN 序列本身的内部经验前兆
(b) External empirical precursors —— 依赖其他活动指标的外部经验前兆
(c) Physical[ly motivated] precursors —— 有物理解释的基础前兆
(d) Forecasts based on SFT models —— 基于表面通量输运模型
(e) Forecasts based on dynamo models —— 基于发电机模型
```

然后将 (a)-(c) 归为 **Precursor methods**，(d)-(e) 归为 **Model-based methods**。

三大类的**精确分类标准**（作者原话）：

| 类别 | 根本前提 (premise) | 核心假设 |
|---|---|---|
| **Precursor methods** | "Each numbered solar cycle is a consistent unit in itself" — 每个太阳周期是一个自洽的整体（Gleissberg 格言）。前兆在极小值时刻评估，预报下一个极大值。 | 存在 intercycle memory（周期间记忆），或者至少 intracycle memory + overlap effect 足以建立预报关系。 |
| **Extrapolation methods**（你的类别） | "The physical process giving rise to the sunspot number record is statistically homogeneous" — 产生 SSN 记录的物理过程在统计上是齐次的（stationary），其变异底层的数学规律在任意时间点都是相同的。 | 时间序列是一个同质随机过程的实现，因此可以外推。 |
| **Model-based methods** | 太阳周期由具体的物理过程驱动（Babcock-Leighton 发电机 + 经向环流）。用数值模型模拟这些过程，预报输出。 | 模型中的物理机制与现实太阳一致，且模型参数已校准。 |

**作者的归类排序：**
> "time series methods have not been particularly successful... have been relegated to a later section of this review, after... the currently much more lively field of the physically more insightful and more successful alternative approaches"

Extrapolation 被"贬"到了第 4 节，这本身就是一个态度。

#### §2 Precursor methods 中各类前兆方法的精度对比

**关键发现：没有统一量化的 MAE/RMSE。** 文献不以单一统计指标评价方法——而是以"对 Cycle 21–24 的实际预报值是否落在合理范围内"作为判据。

**(a) 内部经验前兆（§2.1）：**
- 连续周期极大值相关：r = 0.35（很弱）
- 极小值 → 下一个极大值（线性回归）：**r = 0.68**（排除异常 Cycle 19 后）
- 极小值前 3 年的活动水平 → 极大值：也可观的相关性
- 但依赖 Waldmeier 效应的预报方案"可靠性相当有限"（"the rather low reliability reflected in the correlation coefficients quoted above"）

**(b) 极性前兆（§2.3）—— 最优方法：**
- **Table 1 给出四种极性前兆定义的模拟散点（1σ scatter）**：16.8–25.9 SSN v2 单位，相对均值为约 10–15%
- WSO 极性场 → Cycle 24 预测：Svalgaard et al. (2005) 预测 75±8 (v1) vs 实际 67 (v1)——"predicted a relatively weak Cycle 24 as early as 4 years before the sunspot minimum"
- 极性代理重建与 SSN 的相关性：r = 0.69

**(c) 地磁前兆（§2.4）：**
- Ohl (1966) 方法：aa 指数极小值 → 下一个周期极大值，有物理依据但不直接
- Feynman (1982) 方法：分离"行星际成分"——**在 Cycle 24 预测中惨败**（预测 R_m ≈ 150 而实际约 67 v1 / 116 v2），原因是"Halloween events of 2003"错误地膨胀了行星际成分估计

**精度的「定性排名」：**
Polar precursor >> Geomagnetic (Ohl) > Internal (Waldmeier) > External empirical

对 Cycle 24 预报，**polar precursor 是唯一始终正确的方法**（"has consistently proven its skill in all cycles"）。

#### §4 Extrapolation methods

##### §4.3 Nonlinear methods 对神经网络的整体评价

**(a) 混沌假设的困境（§4.3.1–4.3.3）**
- 1980 年代：很多人赶"混沌"热潮（"jumped on the chaos bandwagon"），用非线性时间序列方法（吸引子重建、嵌入维数估计）分析太阳活动
- 1990 年代以来（特别是 Kantz & Schreiber 1997 后）：逐渐认识到**应用非线性算法本身不证明底层系统的混沌性质**——"stochastic noise superposed on a simple, regular, deterministic skeleton can also give rise to phase space characteristics that are hard to tell from low dimensional chaos"
- 当前主导观点（prevailing view）：**没有明确的证据表明太阳活动数据中存在混沌**（"there is no clear cut evidence for chaos in solar activity data", Panchev & Tsekov 2007）
- 不同研究者报告的吸引子维数互相矛盾

**(b) 用 NN 做太阳周期预测的局限性：**

**§4.3.4 对神经网络的整体评价：**

1. **原理上合理**：NN 是广义非线性映射逼近器，可用来学习时间序列的 attractor 结构中的非线性映射。这在原理上属于 "nonparametric fitting"。

2. **历史表现惨淡**：
   - Calvo et al. (1995) 第一个将 NN 用于黑子数预测——声称能"预测"（实为 hindcast/postdict）早期周期，但**对 Cycle 23 的预测偏得离谱**（预测峰值 166 [v1] vs 实际 121）
   - Maris & Oncica (2006) 对 Cycle 24 的 NN 预测同样"equally far off"
   - Uwamahoro et al. (2009) 更保守，表现相对好一些
   - Attia et al. (2013) 的 neuro-fuzzy 模型预测 Cycle 25 略低于 Cycle 24

3. **Petrovay 的核心批评在 §5**：
   > "Extrapolation methods as a whole have shown a much less impressive performance. Overall, the statistical distribution of maximum amplitude values predicted by 'real' forecasts made using these methods... does not seem to significantly differ from the long term climatological average."

   翻译：外推方法的预报值分布跟猜气候均值差不多。这是最严厉的判词。

4. **但也有留有余地**（§5）：
   > "It would of course be a hasty judgement to dismiss each of the widely differing individual approaches comprised in this class simply due to the poor overall performance of the group. In particular, some novel methods introduced in the last decades, such as SSA, phase space reconstruction or neural networks have hardly had a chance to debut, so their further performance will be worth monitoring in upcoming cycles."

   这是给 NN/DL 方法留的窗口。

#### §5 Summary evaluation：三类方法的整体表现排名

**Cycle 21–22**：Precursor 方法预报一致且正确。
**Cycle 23**：Precursor 方法内部仍一致（范围 150–170 v1），但实际值偏低（121 v1）。
**Cycle 24**：大部分 precursor 一致指示低于均值（70–100 v1），**除了 Feynman 的地磁前兆法误判为 150**。

**排名（定性）：**
```
Precursor methods（尤其是 polar precursor）  >>  Model-based（如果校准了，跟 precursor 等价）
                                                >>  Extrapolation methods（跟猜气候均值差不多）
```

#### §6 Forecasts for Cycle 25

**Table 2 摘要：**

| 类别 | 方法 | 预报峰值 (SSN v2) |
|---|---|---|
| Internal precursor | Li et al. (2015) | 175 (154–202) |
| Polar precursor | 多家平均 | 117 ± 15 |
| Helicity | Hawkes & Berger (2018) | 117 |
| Rush-to-the-poles | Petrovay et al. (2018) | 120 ± 39 / 130 |
| Model-based: SFT | Jiang et al. (2018) | 124 ± 31 |
| Model-based: SFT (AFT) | Upton & Hathaway (2018) | 110 |
| Model-based: dynamo 2×2D | Labonville et al. (2019) | 89 (+29/-14) |
| Model-based: truncated | Kitiashvili (2016) | 90 ± 15 |
| Spectral | Rigozo et al. (2011) | 132 |
| Simplex projection | Singh & Bhargawa (2017) | 103 ± 25 |
| Simplex proj / time-delay | Sarp et al. (2018) | 154 ± 12 |
| **Neural networks / Neuro-fuzzy** | **Attia et al. (2013)** | **90.7 ± 8** |

**Petrovay 的总结**：
> "The overwhelming majority of forecasts agree that the amplitude of Cycle 25 is most likely to lie within ±20% of Cycle 24... Dynamo based predictions indicate that Cycle 25 will peak at somewhat lower values than Cycle 24, while precursor techniques and SFT modelling suggest a cycle amplitude comparable to or slightly higher."

### 2.3 对你的项目的关键判断

#### 2.3.1 Petrovay 是否认为 NN/DL 在太阳黑子预测中有前途？

**答案：审慎中立，给了一道窄门。**

他的论据分三层：
- **否定面**：已有的 NN 预测（Calvo 1995，Maris 2006）在实际周期预报（非回测）中都做得非常差，甚至不如猜气候均值。
- **辩证法**："因为整类表现差就否定每个具体方法"是草率的。NN、SSA、phase space reconstruction 等「还没被足够多的周期检验过」。
- **留门**：如果 NN 能学到时间序列中确实存在的非线性结构，它在原理上并非无用——问题在于时间序列太短（24 个完整周期的年值 = ~ 250 个有效自由度）、噪声太大、且系统可能根本不是低维混沌的。

他的核心态度可以总结为：**NN 不是死路，但过去用得太烂（浅层网络 + 数据太少 + 没检验外推能力）。** 他没有见过 PatchTST 这种现代架构——你的模型要在**不如猜均值**这个 baseline 上有**统计显著**的改善，才能逆转这个评价。

#### 2.3.2 Extrapolation 方法的根本局限是什么？

从 Petrovay 的论述中可以提取三个根本局限：

1. **根本前提不成立**：extrapolation 方法的前提是"产生 SSN 的过程在统计上是齐次的（homogeneous/stational）"。但太阳活动不是平稳过程。它有 Gleissberg 调制、有 grand minima（Maunder Minimum 是质变），Modern Maximum 可能在整个全新世无先例。你的训练数据如果只覆盖最近的 Modern Maximum regime，用外推模型去预测一个新 regime 是结构性错误的。

2. **样本量致命不足**：只有 24 个完整 Schwabe 周期（年值约 250 点，月值约 3000 点），对深度学习来说少得可怜。

3. **噪声 vs 信号的混淆**：你无法区分低维混沌、随机噪声叠加在确定性骨架上、和纯随机过程。这意味着即使你的模型在回测中拟合得好，也无法保证学到了物理本质而非噪声。

#### 2.3.3 "物理先验 + 数据驱动"的混合路线在文献中的支持

**直接提到"数据同化"（data assimilation）的地方：**
- Kitiashvili & Kosovichev (2008) 用 data assimilation 方法将观测数据融入发电机模型做 Cycle 24 预测——**这个预报取得了良好的结果**。
- 但 Petrovay 同时指出这个模型的物理基础"rather far removed from physical reality"

**隐含的混合路线：**
- 极性场前兆方法本质上就是物理先验（Babcock-Leighton 发电机理论的定性推论）配经验校准（线性回归）
- SFT 模型 + 观测数据同化相当于 physics-informed boundary condition

### 2.4 一句话总结

这篇综述告诉你：你的 PatchTST 属于在历史上表现最差的一类方法，但你不是低配版 MLP——你需要证明 Transformer 架构 + 更好的特征工程（如 §1.2.3 的 sqrt/alternating 变换）能从 SSN 序列中提取出传统外推方法无法捕捉的 pattern；Petrovay 给你的 baseline 是"猜气候均值"，polar precursor 给你的竞争上限大概是 1σ scatter ≈ 15–25 SSN 单位——超过这个你就是贡献，超不过就是另一篇"off by a wide margin"的神经网络文献。

---

## 3. DLinear — Zeng et al. (2023) AAAI

**完整引用**：Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. CUHK & IDEA.

### 3.1 三句话笔记

- **Q1 动机：** 现有 Transformer 长时序预测模型依赖自注意力机制，但自注意力本质上是排列不变的（permutation-invariant）——它对 token 之间的"顺序"天然不敏感。对于缺乏语义的纯数值时间序列，"顺序"才是最核心的信息。论文质疑：Transformer 对于时间序列预测真的有效吗？

- **Q2 方法：** 提出一组极其简单的单层线性模型 LTSF-Linear（含三个变体：Vanilla Linear、DLinear、NLinear），采用直接多步（DMS）预测策略，在 9 个真实世界 benchmark 上与 5 种复杂 Transformer 变体对比。

- **Q3 与项目的关系：** 论文从机制上解释了为什么你的天花板探测（DLinear-I > PatchTST, MAE 19.30 vs 20.54）不是一个偶然 —— 长时序预测本质只需要提取趋势和周期性，而这些恰好是线性模型最擅长的事。Transformer 的自注意力对排序不敏感，在纯周期性数据上不仅多余，还会引入噪声和过拟合。

### 3.2 核心内容提取

#### DLinear 的精确定义

**不是单纯一个线性层**，而是"时序分解 + 两个独立线性层"的架构：

1. **分解步骤**：用移动平均核 (kernel size=25，与 Autoformer 相同) 将原始输入 X 分解为：
   - **Trend 分量** X_t：移动平均提取的趋势
   - **Remainder (Seasonal) 分量** X_s：原始序列减去趋势

2. **线性映射**：对两个分量各应用一个单层线性层：
   ```
   X̂_i = W_s · X_{s,i} + W_t · X_{t,i}
   ```
   其中 W_s, W_t ∈ ℝ^{T × L} 是沿时间轴的线性层权重。

3. **关键特性**：
   - 权重在不同变量间共享（Channel Independent，不建模空间相关性）
   - 总参数量：2TL（T 为预测长度，L 为回望窗口）
   - 与 Vanilla Linear (TL 参数) 的区别仅在于分解 + 双线性层

#### NLinear 是什么？与 DLinear 的区别

**NLinear** 是为应对"训练-测试集分布偏移 (distribution shift)"设计的：

- **公式**：X̂_i = W · (X_i − X_{i,last}) + X_{i,last}
- 先减去序列的最后一个值，过线性层，再加回来
- 本质是把预测锚定在最近观测值上，避免测试集均值漂移导致的大误差

**三者区别总结：**

| 模型 | 参数量 | 核心机制 | 适用场景 |
|------|--------|----------|----------|
| Vanilla Linear | TL | 加权求和 | 无特殊需求的通用情况 |
| **DLinear** | 2TL | 趋势-季节性分解 + 双线性层 | 有明显趋势的数据 |
| **NLinear** | TL | 用序列末值做归一化 | 存在分布偏移的数据 (如 ETTh1/ETTh2/ILI) |

#### 实验设置：9 个 benchmark 数据集

| 数据集 | 变量数 | 时间步数 | 采样频率 | 领域 |
|--------|--------|----------|----------|------|
| ETTh1 | 7 | 17,420 | 1小时 | 电力变压器温度 |
| ETTh2 | 7 | 17,420 | 1小时 | 电力变压器温度 |
| ETTm1 | 7 | 69,680 | 5分钟 | 电力变压器温度 |
| ETTm2 | 7 | 69,680 | 5分钟 | 电力变压器温度 |
| Traffic | 862 | 17,544 | 1小时 | 道路占用率 |
| Electricity | 321 | 26,304 | 1小时 | 电力消耗 |
| **Exchange-Rate** | 8 | 7,588 | 1天 | 汇率 |
| Weather | 21 | 52,696 | 10分钟 | 气象指标 |
| **ILI** | 7 | 966 | 1周 | 流感样疾病 |

预测长度：ILI 用 {24, 36, 48, 60}，其余用 {96, 192, 336, 720}。

#### 关键实验结果

**总体结论：DLinear/Linear 比最佳 Transformer (FEDformer) 优势在 20%~50%。**

| 数据集 | 预测长度 | Linear最优MSE | 最佳Transformer MSE | 改善幅度 |
|--------|----------|---------------|---------------------|----------|
| ETTh2 | 96 | 0.081 (DLinear) | 0.148 (FEDformer) | **~45%** |
| Exchange-Rate | 720 | 0.413 (DLinear*) | 0.421 (FEDformer) | 约2% (最小的) |
| ILI | 24 | 1.683 (NLinear) | 3.228 (FEDformer) | **~48%** |
| Traffic | 96 | 0.375 (All Linear) | 0.587 (FEDformer) | ~36% |

**一个惊人的发现**：即使是 naive Repeat（重复最后一个值），在 Exchange-Rate 上比所有 Transformer 方法好约 45%。Transformer 过拟合了突变噪声，导致趋势预测错误。

#### look-back window 消融实验结论

**核心证据（Figure 4, Section 5.3）：**

- **Transformer：** look-back 窗口增大 → 性能恶化或停滞。证据：Figure 4 中 Traffic 和 Electricity 上，随着 L 从 24→720，所有 Transformer 的 MSE 走平甚至上升。
- **LTSF-Linear：** look-back 窗口增大 → 性能显著提升。
- **Far vs Close 实验（Table 3）：** 用距离预测点很远的 96 步和很近的 96 步做输入，Transformer 的预测精度几乎不变 —— 说明它们根本没有从"更远处的历史"中提取到有用信息，只是捕捉邻近时间步的特征。
- **论文原话**："Existing solutions tend to overfit temporal noises instead of extracting temporal information if given a longer sequence, and the input size 96 is exactly suitable for most Transformers."

#### Channel Independent vs Channel Mixing

LTSF-Linear **明确采用 Channel Independent (CI)**：
> "LTSF-Linear shares weights across different variates and does not model any spatial correlations."

结果：即使不建模变量间相关性，CI 的线性模型仍然击败所有建模了交叉变量关系（Channel Mixing）的 Transformer。这意味着对于这些 benchmark，变量间的"空间"相关性要么不存在，要么没有信息增量，甚至可能引入噪声。

#### 自注意力对排序不敏感的消融实验（shuffle 实验）

**Table 5 — 这是整篇论文最有杀伤力的实验：**

两种打乱策略：
- **Shuf.**：完全随机打乱输入序列
- **Half-Ex.**：前半段和后半段交换

**关键结果（MSE, Exchange-Rate 数据集）：**

| 模型 | 原始 (Ori.) | 随机打乱 (Shuf.) | 平均下降 |
|------|-------------|------------------|----------|
| FEDformer | 0.395-0.520 | 0.753-0.846 | **-0.09%** (几乎不变!) |
| Autoformer | 0.455-0.525 | 0.838-0.696 | **+0.09%** (几乎不变!) |
| Informer | 0.974-2.720 | 0.971-2.716 | **-0.12%** (几乎不变!) |
| **Linear** | 0.080-0.806 | 0.133-0.825 | **-81.06%** (剧烈恶化!) |

**结论：** 在没有明显周期性模式的金融数据（Exchange-Rate）上，所有 Transformer 根本不使用时间顺序信息——随机打乱输入后性能纹丝不动。线性模型则天然利用排序，打乱后大幅退化。

### 3.3 对项目的关键判断

**DLinear 的局限——什么时候 Transformer 可能比线性好：**
1. 存在变点 (change point) 或结构性突变
2. 变量间存在真实的非线性交互
3. 数据包含"语义级"信息而非纯数值周期
4. 需要多模态输入（如文本+时序）
5. 回望窗口短时欠拟合（L ≤ 72）

**DLinear 在小样本、强周期性数据上的表现：**
- 与太阳黑子最接近的数据集是 ILI（流感数据）——样本量极小 966 步、有明显周期性
- NLinear 在 ILI 上碾压所有 Transformer：**48% 的改善**
- 论文没有测试类似 11 年周期的数据，但 ILI 的机制最接近

**为什么"线性模型在周期性数据上能赢 Transformer"——机理解释：**
1. **本质假说**：长时序预测只依赖两个东西——趋势 + 周期性。预测越远，回望窗口中"近期扰动"的影响越小，趋势和周期性是唯一可靠的信号。
2. **线性模型天然提取趋势和周期**：加权求和操作可以直接编码周期性模式。通过 DLinear 的权重可视化，可以清晰看到模型学到了 24 小时和 168 小时的权重峰值。
3. **Transformer 的三个根本缺陷**：
   - **排列不变的自注意力**：shuffle 实验证明，Transformer 在纯数值数据上基本不利用时间顺序
   - **过多的参数导致过拟合**：周期性是简单模式，用百万参数去拟合 = 必然过拟合噪声
   - **长信号路径**：Transformer 需要多层传递信息，而线性模型是 O(1) 的最短路径
4. **语义缺失**：NLP 中词语有语义，打乱顺序"语义"部分保留。但"SSN 100 → 101"没有语义，只有数值的先后关系。

### 3.4 与你的天花板探测实验的直接对照

你的发现：**DLinear-I > PatchTST (MAE 19.30 vs 20.54)**。

这篇论文提供了以下直接的理论解释：

1. **Shuffle 实验 (Table 5)**：PatchTST 使用自注意力对子序列 patch 建模 —— 如果太阳黑子数据是纯周期性的，注意力机制本质上就不需要时序信息也能工作（但在有噪声时会过拟合无关模式），而线性模型天然"知道"顺序重要。

2. **Informer→Linear 递进消融 (Table 4)**：去除自注意力 → 替换为线性层 → 再去除 FFN 等辅助模块，每简化一步，性能就提升一步。你的发现是同一逻辑的另一个实例：PatchTST 中的注意力层是多余的。

3. **O(1) 信号路径 vs 多头注意力多跳路径**：太阳黑子的 ~11 年周期是一个全局特征。线性模型通过一次加权求和直接捕获；Transformer 经过多层注意力后，长期依赖的梯度信号可能已经被稀释或混淆。

### 3.5 一句话总结

DLinear 等线性模型在周期性时间序列上击败 Transformer，并非因为线性模型有多强，而是因为长时序预测的本质只需要提取趋势和周期这两个低维信号——线性模型的单层加权求和对这种信号是天然的最优编码器，而 Transformer 的自注意力机制因为排列不变性丢失了时序信息、参数过多导致过拟合噪声、且长信号路径稀释了周期信号。你的太阳黑子实验（DLinear-I > PatchTST）是这一原理在真实数据上的又一验证。

---

## 4. PatchTST — Nie et al. (2023) ICLR

**完整引用**：Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. IBM Research.

### 4.1 三句话笔记

- **Q1 动机：** 之前的 Transformer 时间序列模型做逐点（point-wise）tokenization，单个时间步没有语义含义，而且 self-attention 的 O(N²) 复杂度限制了 look-back window 的长度；DLinear 甚至用简单线性模型在多个 benchmark 上打爆了所有 Transformer 变体，引出核心挑战——"Transformer 到底对时间序列有没有用？"

- **Q2 方法：** 两个核心设计：**Patching**——把时间序列切成 subseries-level 的 patch 作为输入 token，既保留了局部语义，又把输入长度从 L 降到≈L/S（复杂度降 S² 倍），从而可以让模型看到更长的历史窗口；**Channel Independence**——多变量时间序列中每个通道独立过 Transformer backbone（权重共享），不混合通道信息，最后将 M 个通道的预测结果拼接起来。另外支持自监督预训练：随机 mask 掉 40% 的 patch，用 MSE 重建被 mask 的 patch。

- **Q3 与项目的关系：** 你们用 PatchTST 预测太阳黑子周期（month_sin, month_cos, ssn 三特征），核心建模问题是：**channel independence 的假设在你们这种"特征量少（M=3）、特征间弱相关"的场景下是否仍然成立？** 原文的 small dataset（ILI，M=7, timesteps=966）表现明显差于大 dataset，这个 failure mode 和你们的小样本太阳黑子数据高度相似——这个问题是你们项目能否 work 的关键。

### 4.2 核心内容提取

#### Patching 机制的精确定义

每个单变量序列 x⁽ⁱ⁾ ∈ ℝ^(1×L) 被切分为长度为 P 的 patch，步长为 S。

**关键公式（Section 3.1）：**
```
N = ⌊(L - P) / S⌋ + 2
```
即在原始序列末尾用最后一个值填充（pad），保证最后一个值被覆盖到。

**默认超参数（Section 4.1）：**
- `PatchTST/42`：L=336, P=16, S=8 → N=42
- `PatchTST/64`：L=512, P=16, S=8 → N=64
- 自监督预训练时：P=12, S=12（非重叠 patch，以防止信息泄露）

**为什么 patching 而非逐点？**
1. **语义：** 单个时间步没有语义含义，NLP 中的词有语义、CV 中 16×16 的图像块有语义，时间序列需要在 subseries 级别才能提取局部模式。
2. **计算：** 输入 token 数从 L 降到≈L/S，attention 复杂度从 O(L²) 降到 O((L/S)²)，实际训练时间在 Traffic 上快了 22 倍。
3. **收益：** 同样的计算预算下，模型可以看到更长的 look-back window，MSE 从 0.518 降到 0.397（L: 96→336），再通过 patching 进一步降到 0.367。

#### Channel Independence 的精确定义

**数学定义（Section 3.1）：**

输入多元时间序列 (x₁, ..., x_L)，每个 x_t ∈ ℝ^M，被按通道拆成 M 个独立的单变量序列：
```
x⁽ⁱ⁾ = (x₁⁽ⁱ⁾, ..., x_L⁽ⁱ⁾)  ∈ ℝ^{1×L},  i = 1, ..., M
```

每个 x⁽ⁱ⁾ 独立经过同一个 Transformer backbone（**权重共享**），产生各自的预测：
```
x̂⁽ⁱ⁾ = (x̂_{L+1}⁽ⁱ⁾, ..., x̂_{L+T}⁽ⁱ⁾) ∈ ℝ^{1×T}
```

**损失函数：**
```
L = E_x [1/M · Σ_{i=1}^M ‖x̂_{L+1:L+T}⁽ⁱ⁾ - x_{L+1:L+T}⁽ⁱ⁾‖²₂]
```
即每个通道独立计算 MSE，再对 M 个通道取平均。

**实现技巧（Appendix A.1.5）：**
- 输入 batch shape: B × M × L
- After patching: B × M × P × N
- Reshape 到 (B·M) × P × N
- 送入标准 Transformer 实现（不需要特殊算子）
- 这等于说"B·M 个独立样本"并行过同一个 Transformer，通道之间唯一的交互是通过共享权重

#### 自监督预训练部分

**方法（Section 3.2）：**

1. **非重叠 patch**：P=S（supervised 时 S<P 允许重叠），确保观测到的 patch 不包含被 mask patch 的信息
2. 随机选择 40% 的 patch 索引，被选中的 patch 设为全零
3. 去掉预测头，附加一个 D×P 的线性层用于重建
4. 用 MSE loss 重建被 mask 的 patch

**训练流程：**
1. 自监督预训练 100 epochs（masked patch reconstruction）
2. 下游预测：linear probing 20 epochs（冻结 backbone，只训练 head），或两阶段 fine-tuning

**对你们项目的可能用法：**
- 太阳黑子数据量小（~300 年×12 月=约 3600 个时间点），直接用自监督预训练**在自己数据上**意义不大——原文预训练用的是大 dataset（Electricity 26304 timesteps），ILI（只有 966 timesteps）根本没做预训练实验
- 更有价值的可能是：用某个大 corpus 预训练，然后迁移到你们的具体任务上。但实话说：以你们的数据量，自监督预训练大概率不是瓶颈，先保证 supervised 模式下 Patching + CI 能 work 再说

#### 关键消融实验结果

**(1) Patching vs No Patching：**

以 Weather 数据集为例（PatchTST/42, L=336）：

| 配置 | 预测96步 MSE |
|---|---|
| P+CI (全量) | 0.130 |
| CI only (P=S=1, 逐点) | 0.136 |
| P only (有patching, channel-mixing) | 0.196 |
| Original TST (无patching, channel-mixing) | 0.168 |

- **CI 的收益 > Patching 的收益**：CI only（0.136）比 P only（0.196）好得多
- P + CI = 0.130，进一步边际改善

**(2) Channel Independent vs Mixing：**

以 Electricity 96 步为例：
- CI only (no patching): MSE 0.164
- P only (channel-mixing, with patching): MSE 0.168
- P+CI: MSE 0.152

Channel mixing 在大数据集上更容易过拟合（Figure 7）。CI 的样本效率更高——CI 用更少的训练数据就能达到同样的 MSE。

**(3) 不同 patch_len 的影响（Appendix A.4.1, Figure 4）：**

- **MSE 对 P 的选择不敏感**（原文说"robust to the patch length hyperparameter"）
- 总体趋势：较大的 P 略优于较小的 P，且更大的 P 计算量更小
- **推荐范围：P ∈ {8, 16}**

**(4) 自监督预训练的收益（Table 4 / Table 12）：**

- 大 dataset 上收益明显。Electricity 96 步：MSE 从 0.152→0.144
- 小 dataset 上收益不明显甚至倒退。ETTh1 96 步：0.375→0.366（有提升但不大）
- 自监督在 ILI 上**完全没有实验**
- **核心规律：数据越多，自监督预训练收益越大。**

### 4.3 对项目的关键判断

#### Channel Independence 假设在你们的 3 特征弱相关数据上是否成立？

你们的数据：month_sin（月份的正弦编码），month_cos（月份余弦编码），ssn（太阳黑子数）。

**原文的直接证据（Section A.7.1）：**

1. **Adaptability（适应性）：** 每个通道可以学到独立的 attention pattern。不相关的序列注意力图差异很大，相关的序列会有相似的注意力图。"channel mixing 时所有序列共享同一个 attention pattern，如果行为不同的序列混在一起，反而有害。"

2. **样本效率：** CI 模型比 channel-mixing 更随训练数据量大小收敛更快。原文明确指出："widely used time series datasets may not be large enough for channel-mixing models to obtain similar performances in supervised learning."

3. **抗过拟合：** channel-mixing 在几个 epoch 后就过拟合了，CI 模型持续收敛。

**对你们的数据的判断：**
- month_sin 和 month_cos 强相关（共同编码月份循环），但原文说"相关序列会学到相似的 attention map"，共享权重可能给它们分配相似模式——反而不是问题
- month_sin/cos 和 ssn 之间相关性弱，这正是 CI 假设成立的条件
- **整体判断**：CI 假设在你们数据上**基本成立**。三个特征太少，channel-mixing 只会增加参数量而无任何实际收益。ILI（M=7, timesteps=966）小数据集上 CI-only 也远好于 channel-mixing 版本。

#### 原文在哪些类型的数据集上表现最差？

1. **ILI 数据集**（M=7, timesteps=966）：所有数据集中最小的一个。MSE 高达 1.522-1.529。而且不同模型参数配置下方差大——**小样本不稳定**。
2. **ETT 小数据集**（M=7, timesteps=17420）：原文专门对这些小数据集用了缩小版模型参数（H=4, D=16, F=128），以防止过拟合。

**和太阳黑子数据的相似性：** 你们的 3600 个时间点 ≈ ILI 的 3.7 倍，但仍属小数据范畴。不会得益于 PatchTST 的大数据优势（自监督预训练、长 look-back window），但仍然能用 patching 带来的局部语义提取和 CI 的抗过拟合特性。

**一个值得注意的问题：** 太阳黑子数据的周期性很强（~11 年周期），月值序列基本上是类似正弦波+噪声的模式。这种高度周期性的数据，patching 所提取的"局部语义"是否真的有用，还是说一个简单的季节性分解就很够用——这是原文没有讨论的问题。

#### 原文是否讨论了小样本问题？

**没有明确讨论最低数据量要求。** 但可以从以下细节间接推断：

1. 对 ILI 和 ETTh1/ETTh2 使用了缩小的模型配置（H=4, D=16, F=128），隐含建议：小数据→小模型
2. ILI 的实验 epoch 数、look-back window 等都做了单独调整（L=104 for ILI vs L=336 for 其他数据集）
3. Figure 7 展示了样本效率曲线：当训练数据只有全量的 20% 时，CI 模型的退化远小于 channel-mixing
4. 原文在 ablation 实验中把 Electricity 和 Traffic 的最大 epoch 从 100 降到 20，说明参数量大+数据量不足时需要减少训练 epoch

**对你的数据的最低估计：** 3600 个时间步如果使用滚动窗口构造训练样本（比如 L=96, stride=1），大约能产生 3500 个样本——这对比 ILI 的~900 个样本，条件好一些。但仍建议使用小模型配置（H=4, D=16）。

#### 原文中你项目可能直接用到的参数建议

| 参数 | 原文默认值 | 对你的数据的建议 |
|---|---|---|
| patch_len P | 16 (supervised), 12 (self-supervised) | 8-16（原文说这个范围普适性好） |
| stride S | 8 | 月值数据周期性12，可以考虑 S=6 或 12 |
| look-back L | 336 或 512 | 月值数据，L=120-240（10-20年）可能合理 |
| 模型大小 | D=128, H=16 | D=16-64, H=4（小数据用小模型） |
| Dropout | 0.2 | 可以考虑更高（0.3-0.4） |
| Instance Norm | 用 | 有用但不是核心 |

### 4.4 一句话总结

PatchTST 用 patching 还原局部语义 + channel independence 对抗过拟合，在充足数据下大幅超越了所有 Transformer 变体和线性 baseline，但在 ILI（最小数据集，966 steps）上表现最差；对你们的 3600-sample 太阳黑子数据，patching 设计仍然合理（提取子序列级别的模式），但自监督预训练大概率没有收益，且必须用缩小版模型参数来防止过拟合。

---

## 5. Wang et al. (2021) — LSTM 太阳黑子预测

**完整引用**：Wang, Q.-J., Li, J.-C., & Guo, L.-Q. (2021). Solar cycle prediction using a long short-term memory deep learning model. *RAA*, 21, 012. 中南大学。

### 5.1 三句话笔记

- **Q1 动机：** 现有太阳黑子预测方法（谱分析、神经网络、气候学、发电机模型、前兆法）对 Cycle 24 的预测值分布极散（43-185），相对误差多数超过 10%，需要一个更准确的深度学习方法。
- **Q2 方法：** 用优化的 LSTM（隐节点=19，batch size=20）做 SSN 的一步预测和多步预测。一步预测：用前 10 个月预测下 1 个月。多步预测：用前 720 个月（60年）预测后 72 个月（6年）。用目标周期之前的所有数据训练，目标周期作为测试集。
- **Q3 与项目关系：** 实验设计最严谨的 DL+sunspot 论文。你的 PatchTST 应该对标它的整周期留出方案。

### 5.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO, Royal Observatory of Belgium |
| SSN 版本 | Version 2.0 |
| 数据类型 | 13-month smoothed monthly SSN |
| 时间范围 | 1750 年起 |
| 特征 | **仅 SSN**（单变量） |
| 训练/测试划分 | **按整周期留出**。Cycle 22: 1750-1985→1986-1996; Cycle 23: 1750-1996→1997-2007; Cycle 24: 1750-2008→2009-2019 |
| 多步预测 input/output | Input: 720 个月 (60年), Output: 72 个月 (6年) |
| 评估指标 | RMSE、PDRM（峰值振幅相对误差） |
| 多 seed 验证 | 是，每个结果取 **10 次运行平均** |
| LSTM 参数 | 隐节点=19, 层数=2, batch size=20, 训练轮数=10000 |

### 5.3 精度结果

**一步预测：**

| 周期 | RMSE | 实际峰值 | 预测峰值 | 相对误差 |
|------|------|----------|----------|----------|
| Cycle 22 | 6.12 | 212.5 | 199.3 | 6.2% |
| Cycle 23 | 4.28 | 180.3 | 178.8 | 0.8% |
| Cycle 24 | 2.45 | 116.4 | 118.8 | 2.1% |

**多步预测：**

| 周期 | RMSE | 实际峰值 | 预测峰值 | 相对误差 |
|------|------|----------|----------|----------|
| Cycle 22 | 35.3 | 212.5 | 175.9 | 17.2% |
| Cycle 23 | 28.8 | 180.3 | 167.8 | 6.9% |
| Cycle 24 | 12.1 | 116.4 | 112.8 | 3.0% |
| **Cycle 25** | — | — | **114.3** | — |

Cycle 25 预计 2023 年达到峰值 114.3。

### 5.4 方法论判断

**优点：**
- **按整周期留出训练/测试**是太阳黑子预测领域最合理的实验设计。这让"预测"真正成为预测——测试集中是模型从未见过的完整太阳活动周期。这点非常关键。
- 多 seed（10 次运行平均）有一定稳健性。
- 既做了一步预测也做了多步预测（6 年 ahead），贴近实际预报需求。
- V2 SSN 数据有完整引用。

**漏洞/不足：**
- 只测了 3 个周期（Cycle 22-24），样本量太小。更严格应该做 leave-one-cycle-out CV。
- 没有 R² 或 MAE 报告，只有 RMSE 和峰值相对误差。
- 多步预测的 RMSE 从 Cycle 22 的 35.3 降到 Cycle 24 的 12.1，不是因为模型变好了，而是因为 Cycle 22 振幅大（212.5）、Cycle 24 振幅小（81.9）。RMSE 作为绝对指标受量纲影响严重。
- 多步预测的"峰值滞后"问题没有被量化讨论，只在一步预测中提到约 1.5 个月的滞后。

**和你的 PatchTST 实验对比：**
- Wang 的训练/测试划分方案更严谨（整周期留出），但评估指标太少，且只测了 3 个周期。
- 你的 PatchTST 如果用随机时间切分（比如 80/20），在科学意义上就不如 Wang 严谨，因为模型可能见过测试集所在周期内的部分数据。

### 5.5 一句话总结

实验设计思路正确（整周期留出），但模型太简单，评估太粗糙，3 个测试周期不足以证明方法的泛化性。

---

## 6. Pala & Atici (2019) — LSTM vs NNAR

**完整引用**：Pala, Z., & Atici, R. (2019). Forecasting Sunspot Time Series Using Deep Learning Methods. *Solar Physics*.

### 6.1 三句话笔记

- **Q1 动机：** 系统对比深度学习（LSTM, NNAR）与经典算法（ARIMA, Naive）在 SSN 超长时序上的预测性能。
- **Q2 方法：** 月均 SSN 1749-2018 共 3240 条，rolling origin cross validation（6-slice 和 12-slice），对比 6 种算法，LSTM 为 2 层 50 单元，预测 10 年（120 个月）。
- **Q3 与项目：** 本项目最直接的前驱之一——深度学习做黑子预测的早期最佳 baseline；RMSE=35.9 是 PatchTST 需要显著击败的目标。

### 6.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO v2.0 月均 SSN, 1749-2018（3240 个月）|
| 特征 | 单变量 SSN，分解为 trend+seasonal+remainder |
| 训练/测试划分 | Rolling origin forecast resampling。6-slice: 每片 1800 条, train=1200, val=600, test=600, skip=240。12-slice: 每片 1080 条, train=840, val=280, test=240, skip=180 |
| 评估指标 | RMSE |

### 6.3 精度结果

| 模型 | RMSE |
|------|------|
| LSTM 6-slice | **35.9**（最佳） |
| LSTM 12-slice | 36.9 |
| NNAR | 42.41 |
| ARIMA | 45.60 |
| Naive | 90.51 |
| SC25 预测峰值 | **167.3**，时间 2022.07 |

### 6.4 方法论判断

Rolling origin CV 是时间序列预测的正确验证方式，这一设计值得肯定。但 LSTM 仅 2 层 50 单元，模型容量明显偏小。SC25 峰值 167.3 预测时间 2022 年 7 月——现在回头看这是双重错误：实际峰值远晚于 2022 年，且形态是双峰结构。输入窗口仅 44 年不够覆盖 Gleissberg 循环（70-100 年）。SSN 中的零值用前一个非零值填充可能引入系统性偏差。

### 6.5 一句话总结

LSTM 是 SSN 深度学习的早期基线（RMSE=35.9），模型容量偏小，预测 SC25 峰值 167.3/2022.7 已被实际观测证伪。

---

## 7. Benson et al. (2020) — WaveNet+LSTM

**完整引用**：Benson, B., et al. (2020). Forecasting Solar Cycle 25 Using Deep Neural Networks. *Solar Physics*.

### 7.1 三句话笔记

- **Q1 动机：** WaveNet 的多层膨胀卷积能比纯 LSTM 指数级扩大感受野，理论上可同时捕捉尺度差异巨大的太阳周期模式。
- **Q2 方法：** WaveNet（膨胀率 1→512，10 层）+ 单 LSTM 层 132 单元，输入 528 个月(4 周期)，输出 132 个月(1 周期)，5 折 TimeSeriesSplit CV，同时预测 SSN 和总黑子面积。
- **Q3 与项目：** 这是目前深度学习 SSN 预测的顶尖结果（RMSE=2.93）；膨胀卷积扩大感受野的思路与 PatchTST 使用 patch 机制降低复杂度是两种不同的"长记忆捕获"策略。

### 7.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO v2.0 月均 SSN (1749-2019, 3251 条)；TSA 日数据月均后 (1874-2019, 1744 条) |
| 特征 | 单变量 SSN 和单变量 TSA（独立实验） |
| 输入/输出 | 输入 528 个月→输出 132 个月（约 2560 对 SSN, 1085 对 TSA） |
| 训练/测试划分 | 5-fold TimeSeriesSplit，每 fold 训练集增量（432→864→1296→1728→2160 对） |
| 模型架构 | WaveNet(10层1D膨胀因果卷积, dilation=1,2,...,512) + LSTM(132单元) + dropout 30% + batch norm |
| 评估指标 | RMSE；MAE 给出不确定性（SSN 8%, TSA 12%） |

### 7.3 精度结果

| 模型 | RMSE |
|------|------|
| Baseline (平均周期) | 34.15 |
| 单 LSTM | 4.42 |
| 双层 LSTM | 4.09 |
| 1DConv+LSTM | 3.89 |
| **WaveNet+LSTM** | **2.93**（最佳） |
| SC25 SSN 峰值 | **106 ± 19.75**，时间 2025.03 |
| SC25 TSA 峰值 | 1771 ± 381.17，时间 2022.05（与 SSN 峰值差近 3 年） |

### 7.4 方法论判断

WaveNet+LSTM 的架构设计在直觉上是对的——膨胀卷积将感受野指数级扩大。但结果有两处值得警惕：(1) SC25 预测峰值 106 远低于实际 SC25 表现，说明即使是架构最优的 DL 模型，仅靠单变量 SSN 做满 11 年外推的纯数据驱动方法有其根本上限；(2) TSA 预测峰值 2022.05 与 SSN 预测峰值 2025.03 差了近 3 年，而实际这两个量是高度同步的——两个预测互相矛盾暴露了模型的不稳定性。输入窗口固定 528 个月=4 个标准 11 年周期，但实际周期长度在 10-13 年之间变化。

### 7.5 一句话总结

WaveNet+LSTM 的 RMSE=2.93 是目前 SSN 预测的深度学习方法顶尖结果，但 SC25 预测 106 被实际大幅低估，提示单变量时序的 DL 外推有本质局限。

---

## 8. Kumar & Kumar (2024) — CNN-BiGRU + GRC

**完整引用**：Kumar, A., & Kumar, V. (2024). Forecast of solar cycle 25 based on Hybrid CNN-Bidirectional-GRU model and Novel Gradient Residual Correction technique. *Advances in Space Research*, 73, 4342-4362.

### 8.1 三句话笔记

- **Q1 动机：** 传统 DL 模型预测 SSN 时残差未被处理，导致预测精度受限，需要一个能纠正残差的 post-processing 框架。
- **Q2 方法：** 提出两级模型——(1) CNN-BiGRU 混合模型做基础预测；(2) GRC（Gradient Residual Correction）技术：用 AdaBoost 回归器学习训练集预测残差与数据梯度之间的关系，然后预测测试集的残差，叠加到基础预测上得到最终结果。
- **Q3 与项目：** GRC 残差修正思路可作为你的 PatchTST 实验的参考，但这篇的实验设计有严重问题。

### 8.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO, World Data Center |
| 数据类型 | 四种：Daily Total SSN, Monthly Mean Total SSN, 13-month Smoothed Monthly Total, Yearly Mean Total |
| 特征 | 仅 SSN（单变量） |
| 训练/测试划分 | **90:10 时间序列切分**（81% train, 9% val, 10% test） |
| 输入/输出 | Input: lag=11 个时间步；Output: 1 步 ahead |
| 评估指标 | RMSE、MAE、MASE、MAPE |
| 多 seed 验证 | 是，20 次独立运行 |
| GRC | AdaBoost Regressor (100 estimators)，用训练集残差+梯度训练 |

### 8.3 精度结果

| 数据集 | RMSE | MAE | MAPE | R² |
|--------|------|-----|------|-----|
| Daily SSN | 10.86 | 8.54 | 1.36 | 0.9378 |
| Monthly Mean SSN | 8.16 | 5.92 | 1.73 | 0.9871 |
| **13-month Smoothed SSN** | **1.64** | 1.31 | **0.06** | 0.9585 |
| Yearly SSN | 16.46 | 12.99 | 0.81 | 0.7016 |

SC25 预测峰值：143.6 (2024)。

### 8.4 方法论判断

**致命缺陷：** **90:10 时间序列切分。** 测试数据在时间上紧接训练数据，是**整个时间序列末尾的 10%**。这不是"预测新周期"，而是"用过去所有周期数据拟合下一个时间步"。模型的训练集中包含了几乎所有太阳周期（24 个），测试集只是末尾短小一段。这不是真正的预测，而是一次 step-ahead 的序列续写。

**RMSE=1.64 的水分：** 13-month smoothed SSN 本身已经做了 13 个月的平滑，序列高度自相关。lag=11，预测 1 步 ahead——你几乎是拿 11 个月前的平滑值预测下个月的值，这在数学上接近一个 trivial problem。MAPE=0.06% 说明这个任务本身几乎没有难度。

**对 Cycle 25 的预测用了"迭代单步预测"**，论文没有讨论误差传播问题。

**和你的 PatchTST 对比：** 实验设计明显不如 Wang 严谨。90:10 切分意味着你评估的不再是"能否预测未知周期"，而是"能否在数据流末尾做序列续写"。

### 8.5 一句话总结

模型架构有创新（GRC），但 90:10 的时序切分让所有"高精度"都打了折扣——这测的不是预测能力，是序列自相关的 trivial 续写。

---

## 9. Kumar & Kumar (2025) — Hybrid Ensemble

**完整引用**：Kumar, A., & Kumar, V. (2025). Hybrid-Ensemble Deep-Learning Models to Enhance the Sunspot Prediction and Forecasting of Solar Cycle 26. *Solar Physics*, 300, 100.

### 9.1 三句话笔记

- **Q1 动机：** 现有研究对复杂混合架构和 ensemble 方法的探索不足，需要用多种混合 DL 架构的组合来提升预测精度。
- **Q2 方法：** 提出 4 种混合模型（CNN-DilatedLSTM-BiLSTM-GRU / CNN-GRU-RNN / CNN-GRU / CNN-GRU-RNN）+ 一个 Hybrid Ensemble（取 H2 和 H4 的加权平均）。
- **Q3 与项目：** "模型超市"做法可作为反面教材——你的 PatchTST 不需要走这条路。

### 9.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO, WDC, Royal Observatory of Belgium |
| 数据类型 | 三种：13-month smoothed monthly (3306 样本), Yearly (324 样本), Monthly Mean (3306 样本) |
| 特征 | 仅 SSN（单变量） |
| 训练/测试划分 | **60% train / 40% test** 时序切分 |
| 输入/输出 | 输入 14 个时间步；输出单步 ahead |
| 评估指标 | RMSE、MAE、MSE、R² |
| 多 seed 验证 | 是，20 次独立运行 |
| Ensemble 方式 | H2 和 H4 的简单平均 |

### 9.3 精度结果

| 数据集 | 最优模型 | RMSE | MAE | R² |
|--------|---------|------|-----|-----|
| **13-month Smoothed Monthly** | Hybrid Ensemble | **4.062** | 1.338 | **0.9964** |
| Yearly | Hybrid Ensemble | 22.110 | 16.904 | 0.8920 |
| Monthly Mean | Hybrid Ensemble | 24.606 | 17.853 | 0.8826 |

SC26 预测：峰值 165.35 (2036)，谷值 10.41 (2032)。

### 9.4 方法论判断

**致命缺陷：**
1. **60:40 时间序列切分**：测试集的每个周期训练集都见过——测的是"序列续写"，不是"新周期预测"。
2. **R²=0.9964 是红牌信号**：13-month smoothed SSN 上 baseline LSTM 都是 0.9953。问题本身几乎没有难度——smoothed 数据 + 输入 14 个月预测下 1 个月 = 高度自相关。
3. **收益微薄**：LSTM RMSE=4.598 vs Ensemble=4.062，架构建得再复杂提升极小。
4. **H2 和 H4 的 ensemble 没有互补性**：两者架构相同（CNN-GRU-RNN），唯一区别是 dropout。本质上是同一模型的两个训练变体取平均。
5. **SC26 预测不可信**：迭代预测 12 年无不确定性区间、无误差传播讨论。
6. 写作质量有明显 AI 辅助痕迹。

### 9.5 一句话总结

架构复杂度飙升但实际收益微薄，精度数字因 60:40 切分 + smoothed 数据虚假膨胀，SC26 预测在方法论上不可信。

---

## 10. Xiong et al. (2021) — 南信大 Precursor + 回归

**完整引用**：Xiong, Y., Lu, J., Zhao, K., Sun, M., & Gao, Y. (2021). Forecasting solar cycle 25 using comprehensive precursor combination and multiple regression technique. *MNRAS*, 505, 1046-1052. **南京信息工程大学空间天气研究所**。

### 10.1 三句话笔记

- **Q1 动机：** 第 24 周的前兆方法预测集体失败（地磁指数预测均值 157 vs 实际 120.7），单一前兆信噪比不够，需要一个综合多前兆参数+完整周期轮廓的预测模型。
- **Q2 方法：** 用高斯滤波替代传统 13 月滑动平均处理月均 SSN，选四个前兆参数（前一周峰值、本周期谷值、前一周 Skewness、前一周 aa 指数最大值）及其交叉项做多元最小二乘回归预测峰值，再用改进的 HWR 函数给完整单峰轮廓。
- **Q3 与项目：** 南信大自己人的工作。为 PatchTST 实验提供"传统前兆方法"的对比基线——HWR 的单峰局限正是 PatchTST 可以解决的核心问题。

### 10.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | 月均 SSN（NASA OMNIWeb，1867.02-2020.10，共 1844 个月）；地磁 aa 指数（ISGI） |
| SSN 版本 | 未注明 SILSO 版本号，使用 V1 的可能性较大（2021 年发表时 V2 刚发布） |
| 特征选取 | 4 个前兆变量 + 它们的交叉项（含 aamax·skewness、aamax·RI_max、RI_min·skewness 等 5 项） |
| 训练/测试划分 | 第 11-23 周（13 个样本）建模→预测 SC24 做验证；第 11-24 周（14 个样本）→预测 SC25 |
| 平滑方法 | 高斯滤波 σ=3.6（替代传统 13 月滑动平均） |
| 评估指标 | 多元回归 R 系数、F 检验 p 值、峰值误差百分比 |

### 10.3 精度结果

| 指标 | 数值 |
|------|------|
| 组合 1 回归系数 R | 0.9505 |
| SC24 回测峰值 | 121.3（实际 115.4，误差<6%） |
| SC24 回测上升期 | 4.78 年（实际 5.17 年，误差 7.56%） |
| **SC25 预测峰值** | **140.2**，峰值时间 **2024 年 3 月** |
| SC25 预测上升期 | 4.36 年 |

### 10.4 方法论判断

**优点：** 高斯滤波比 13 月滑动平均信息丢失少，能更好地保留整体趋势。综合多前兆参数确实比单一前兆稳健。南信大空间天气研究所是成熟的太阳物理研究团队。

**漏洞：**
1. **时序逻辑死循环：** 模型要求输入 RI_min(n)（预测周期的谷值），但在预测时该值本身是未知的。论文并未解释此矛盾的解决方式。
2. **HWR 单峰局限：** 无法拟合 SC24 的双峰结构。
3. **误差叠加：** 先预测峰值再代入 HWR → 峰后预测更差。
4. **过度参数化：** 13 个样本拟合 6 个参数。
5. **高斯滤波 σ=3.6 是人为选定的**，缺少敏感性分析。

### 10.5 一句话总结

综合前兆+多元回归预测 SC25 峰值 140.2，方法有"先验信息不可得"的逻辑漏洞，单峰 HWR 无能力刻画双峰结构。

---

## 11. Chapman (2026) — Hilbert 相位新前兆

**完整引用**：Chapman, S. C. (2026). A New Declining Phase Precursor and an Early Prediction of Cycle 26 Maximum. *The Astrophysical Journal*, 1003, 159. University of Warwick.

### 11.1 三句话笔记

- **Q1 动机：** 太阳周期长度和振幅各自独立变化，需要能提前约 7 年预测下一周期最大值的物理有据的新前兆指标。
- **Q2 方法：** 对 13 月平滑 SSN(1749 年起)做 Hilbert 变换提取"均匀太阳能周期时钟"，确定每个周期的 switch-off 时间（活动→平静转换点）；发现 switch-off 时的 SSN 值与下一周期最大值线性相关（r²=0.71）。
- **Q3 与项目：** Hilbert 相位可作为一种额外特征注入 PatchTST；"强/弱周期形状不同"的发现与 PatchTST 的多尺度特性高度契合。

### 11.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO v2.0 月均 SSN (1749-2026) |
| 数据处理 | 13 月平滑 SSN − 40 年 rlowess 趋势 → detrended 信号 → Hilbert 解析相位 → 确定 switch-off 时间 |
| 训练/测试划分 | 全量回测第 1-25 周（25 个 sample 的线性回归） |
| 评估指标 | r²（决定系数）、ρ（Pearson 相关系数） |

### 11.3 精度结果

| 指标 | 数值 |
|------|------|
| Switch-off SSN vs 下一周期 max | r²=0.71, ρ=0.84 |
| 下降期时长 vs 下一周期 max | r²=0.48, ρ=0.72 |
| 回测平均提前量 | 6.9 年（σ=1.4 年） |
| SC25 switch-off 时间（预测）| 约 2028.88-2029.16（2.5-3.5 年后）|
| **SC26 预测定性** | **弱-中等周期，弱于或约等于 SC25** |

### 11.4 方法论判断

**优势：** 物理意义强——switch-off 与日冕形态转换和地磁活动特征切换一致，也对应 AR 区域纬度越过 15° 的观测事实。Hilbert 变换提供了一个客观的"周期时钟"，不依赖人为选定的极小值点。

**局限：** SC26 的定量预测依赖三种 SSN 外推方案将当前周期延伸到 switch-off 时间。论文自己发现强/弱周期形状其实是二分的（不是连续的），但外推方案仍用平均形状——"平均形状"可能对强周期和弱周期都偏。精确预测要等 2.5-3.5 年后 switch-off 实际发生时才可得。

### 11.5 一句话总结

Hilbert 相位定义的 switch-off SSN 是与下一周期峰值线性相关的新前兆(r²=0.71)，SC26 大概率是弱于 SC25 的弱-中等周期。

---

## 12. Gerçeker et al. (2025) — Simplex Projection

**完整引用**：Gerçeker, K., Kilcik, A., Ozguc, A., & Yurchyshyn, V. (2025). Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers. *Solar Physics*, 300, 169.

### 12.1 三句话笔记

- **Q1 动机：** 绝大多数预测研究用 13 月平滑 SSN 丢弃了高频动力学信息；需要不假设线性和平稳性的非线性预测方法。
- **Q2 方法：** Simplex Projection（经验动态建模 EDM），对**未平滑的月均 SSN**做相空间重构（Takens 嵌入），在重构相空间中找最近邻历史状态，加权平均预测未来。同时做单周期和双周期回测。
- **Q3 与项目：** 强竞争方法——model-free、非线性、直接使用 raw 数据的理念与 PatchTST 一致；可做集成候选或对比基准；它的"split point 穷举搜索使用未来观测"漏洞需要警惕。

### 12.2 实验设计细节

| 项目 | 内容 |
|------|------|
| 数据来源 | SILSO v2.0 月均 SSN (1749.01-2024.12) 和 13 月平滑 SSN |
| 特征 | 单变量，通过 (E=2~10, τ=1~70) 做网格搜索的 Takens 嵌入重构相空间 |
| 训练/测试划分 | 回测 SC20-24：library 集截止前一周结束。预测 SC25+26：数据截至 SC24 结束 (2019.12) |
| 参数选择 | split point 穷举搜索 1000→2700，用 SC25 前 60 个月已知观测（至 2024.12）筛选最优配置 |
| 评估指标 | MAE + ρ（回测 ρ>0.95 视为成功） |

### 12.3 精度结果

| 指标 | 数值 |
|------|------|
| 回测 ρ_mean（月均 raw→平滑后）| 0.9959 |
| SC25 最小值（月均原始）| 2030 年 5-7 月，3.8-5.3 SSN |
| SC25 最小值（平滑后）| 2030 年 4-6 月，9.9-12.1 SSN |
| SC26 峰值（月均原始）| 150.6-181.5 |
| **SC26 峰值（13 月平滑）** | **137.4-146.2**，时间 **2035 年 6 月** |
| SC26 形态预测 | 双峰或微起伏平峰，类似 SC20；弱于 SC25、强于 SC24 |

### 12.4 方法论判断

**优势：**
- 使用未平滑数据保留高频动力学信息是这个方向的重要贡献——月均 raw 数据反而比平滑数据在非线性框架下给出了更准的周期谷值。
- 方法的"相似历史片段搜索"逻辑透明可检视，这是相对于黑箱方法的一个优势。

**漏洞：**
- **split point 穷举搜索本质上是数据泄露**：先用 SC25 前 60 个月的已知观测作为 label 来反选最佳的 split point 参数组合，再将这个最优组合用于预测整个 SC25+SC26。实际场景中你在 2019 年底根本没有 SC25 的任何已知观测。
- E 和 τ 的阈值（ρ>0.95, MAE<5%）是主观设定的。

### 12.5 一句话总结

Simplex Projection 在 raw 月均 SSN 上实现双周期预测，SC26 平滑峰值 137-146，方法透明但 split point 优化存在数据泄露问题。

---

## 13. ARIMA 预测（中文）

**基本信息**：未标注作者和期刊。中文 ARIMA 预测报告。

### 13.1 三句话笔记

- **Q1 动机：** 用经典时间序列 ARIMA 模型做 SC25 月均黑子数预测。
- **Q2 方法：** ARIMA(p=27,d=1,q=33)对月均 SSN（1947.01-2020.04）一阶差分平稳后建模，ACF/PACF 定阶，残差 DW 检验。
- **Q3 与项目：** 作为经典统计方法基线对比；ARIMA p=27 阶自回归较短——说明短程记忆可能已经捕捉到大部分循环结构，PatchTST 的 patch 长度需要谨慎设计。

### 13.2 实验设计

- 月均 SSN，1947.01-2020.04（仅约 73 年，丢弃了 1749-1946 近 200 年的 14 个完整周期）
- 单变量自回归，AR(27)阶，ARIMA(27,1,33) 含 60 个参数
- 无显式训练/测试划分——全量数据拟合后直接外推 2020.05-2031.05
- DW=1.9989（接近 2.0→残差白噪声）
- SC25 预测峰值约 **79**（2024 年 5-6 月），预测谷值约 20（2029 年 9 月）

### 13.3 方法论判断

这个模型的问题非常严重。ARIMA(27,1,33)含 60 个参数，用不足 900 个月的数据拟合，极度过度参数化。预测峰值 79 与实际 SC25（2024 年中 13 月平滑 SSN 已超 120）相差 >50%。只用 1947 年起的数据丢弃了近 200 年信息。无交叉验证、无不确定性估计。

### 13.4 一句话总结

ARIMA(27,1,33) 严重过度参数化，SC25 预测误差 >50%，无参考价值。

---

## 14. 太阳黑子第 25 期的统计预报（中文 docx）

未成功解析文本，待安装 python-docx 后处理。

---

## 附录：横向对比速查

### 各方法预测 SC25/SC26 一览

| 论文 | 方法 | SC25 峰值 | SC26 峰值 | 数据切分 |
|------|------|----------|----------|---------|
| Wang (2021) | LSTM | 114.3 | — | 整周期留出 ✓ |
| Pala (2019) | LSTM | 167.3 | — | Rolling CV ✓ |
| Benson (2020) | WaveNet+LSTM | 106 | — | TimeSeriesSplit |
| Xiong (2021) | Precursor+回归 | 140.2 | — | 整周期留出 ✓ |
| Kumar (2024) | CNN-BiGRU+GRC | 143.6 | — | 90:10 时序 ✗ |
| Kumar (2025) | Hybrid Ensemble | — | 165.35 | 60:40 时序 ✗ |
| Chapman (2026) | Hilbert 前兆 | — | 弱-中等 | 全量回测 |
| Gerçeker (2025) | Simplex | — | 137-146 | 双周期回测 |
| ARIMA 中文 | ARIMA(27,1,33) | 79 | — | 无划分 ✗ |

### 你的 PatchTST 基线 vs 竞争方法

| 方法 | 类型 | 指标 | 数值 |
|------|------|------|------|
| PatchTST dm128 (你的) | DL | MAE/R² | 23.87 / 0.568 |
| DLinear-I (你的天花板) | 线性 | MAE/R² | 19.30 / 0.751 |
| M4 Waldmeier (师兄) | 物理 | MAE | 3.32 |
| 地磁前兆 (Ohl et al.) | 传统 | RMS | 27-29 |
| Polar precursor | 物理 | 1σ scatter | 15-25 |
| Wang LSTM | DL | RMSE | 12.1-35.3 |
| Benson WaveNet+LSTM | DL | RMSE | 2.93 |
| 气候学平均 | 无技能 | RMS | 54.4 |

---

*此文件为每篇论文的完整深度笔记，与 literature_reading_notes.md 配套使用。*
