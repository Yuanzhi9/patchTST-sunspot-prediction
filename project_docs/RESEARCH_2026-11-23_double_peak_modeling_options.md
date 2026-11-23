# 双峰太阳周期建模：在 24 个周期约束下保留主峰与谷值的可验证路线

> 日期：2026-11-23
> 性质：定向文献与方法调研；不是模型验收
> 主目标：Cycle 25 谷值/时间、Cycle 26 主峰值/时间，轮廓次之
> 当前状态：景修已确认主峰→零训练残差审计→软双峰参数顺序；不启动双峰训练

## 摘要

观测和机制研究都说明双峰是太阳周期峰区的真实结构，但“双峰存在”不等于
“双峰标签稳定”，更不等于“起报时可预测第二峰”。本调研围绕三个问题展开：
历史双峰是否可稳定识别；哪些模型能生成双峰或平顶轮廓；在只有 24 个完整周期
且必须保护 SC25 谷与 SC26 主峰时，哪条路线最可行。现有证据更支持参数化双峰
形状和历史相似轨迹，而不是立即训练端到端双峰 PatchTST。最稳妥的路线是保留
R1A 主峰注入器，先用零训练残差审计验证双峰基函数能否只改变峰区，再用软标签
预测第二峰概率或 2–3 维形状参数。双峰模型预计首先改善曲线轮廓和峰区形态，
对峰幅校准帮助有限，不能替代 R1B 上游幅度模型。残差还必须通过验证集控制的
deploy/no-deploy 门；证据与结构见
`project_docs/RESEARCH_2026-11-23_peak_residual_correction.md`。

## 1. 研究问题

RQ1：历史太阳周期的双峰是否是可稳定识别的离散类别？
RQ2：参数化双峰、形状 PCA、历史相似轨迹和深度条件模型分别能解决什么？
RQ3：在 24 个周期样本下，哪种方案最可能改善 SC26 峰时与轮廓，同时不破坏
SC25 谷时、SC25 谷幅和 SC26 主峰？

## 2. 方法

检索日期为 2026-11-23。使用 OpenAlex、Crossref、Semantic Scholar 和出版社
文章页，围绕四组检索式交叉检索：

```text
Gnevyshev gap / double-peaked maximum / double maxima solar cycle
sunspot cycle shape / principal component / binary Gaussian / binary Laplace
simplex projection / dynamic time warping / analog forecast solar cycle
cycle-conditioned neural network / FiLM / two-stage solar cycle prediction
```

纳入标准：直接研究太阳周期峰区、双峰、周期形状或太阳周期轨迹预测；
或提供本项目采用的通用条件建模方法。排除只有新闻、无方法摘要或不能核验的
结果。最终用于判断的 10 项工作覆盖观测事实、物理机制、形状拟合、历史轨迹
和神经网络条件融合。

| 证据类型 | 能回答 | 不能回答 |
|---|---|---|
| 观测统计 | 双峰是否常见、是否随奇偶或指数变化 | 下一次双峰是否会出现 |
| 发电机模型 | 双峰可能的物理机制 | 用 2026-10 已知观测做业务预测 |
| 参数曲线拟合 | 历史周期能否用双峰函数表示 | 参数能否提前预测 |
| 轨迹回测 | 历史相似模式是否可迁移 | 小样本外能否稳定泛化 |
| 深度学习诊断 | 条件信息是否被模型读取 | 在 24 个周期上是否可可靠训练 |

## 3. 分类

现有路线分成四支：

1. **软标签或概率路线**：不强行判“单峰/双峰”，预测第二峰存在概率；
2. **参数化形状路线**：用二元 Gaussian、Laplace 或 PCA 基函数表示峰区；
3. **历史相似轨迹路线**：用 simplex projection 或 DTW 搜索历史相似周期；
4. **深度条件路线**：在月序列模型上增加周期条件、极值头或多任务 loss。

四支并不互斥。最适合本项目的关系是：主峰极值模型决定“多大、何时”；
双峰分支只决定“主峰附近是一个峰、两个峰还是平顶”。

## 4. 双峰存在，但离散标签不稳定

Basu et al. (2025) 用 SSN、F10.7 和 F30 检查 SC19–25，发现 SC19–21 的
Gnevyshev gap 不显著，而 SC22–25 在部分数据中较清楚，且 gap 强弱与奇偶周
没有系统性关系；这与本项目“不能按奇偶或强/弱周期硬分组”的审计一致。
Kılçık & Özgüç (2013) 从大、小黑子群比例解释双峰，并回看 SC1–23 的峰区
持续时间；这类工作支持双峰有结构，但其判别依赖数据分组。

更关键的是，双峰可能不是“有或无”的简单状态。Georgieva (2011) 认为所有周期
都可能有两个由不同环向场涌浪形成的峰，只是在总黑子数中有时被合并；Karak
et al. (2018) 的 Babcock–Leighton 发电机模型也用极向场涨落解释双峰或尖峰。
因此，总 SSN 上的“单峰”可能只是两个峰相距太近或第二峰太弱。这与本项目
双峰 audit 中 14/24 个周期为 ambiguous、所有周期至少翻转一次标签的结果相容。

结论：应使用 `double_support∈[0,1]` 或连续峰间凹陷指标，而不是把 24 个周期
硬切成二分类。硬标签更适合展示结果，不适合作为唯一训练目标。需要注意，
`double_support` 只是高度相关的 12 种平滑/阈值配置下的支持比例，不是校准概率；
概率模型必须重新校准，不能直接把它当真实概率使用。

## 5. 参数化形状能表示双峰，但不自动解决预测

Li et al. (2017) 用二元 Gaussian 拟合周期形状，六个参数可压成三个，明确
优于此前的二元 Laplace，尤其在峰区；Takalo & Mursula (2018) 对 SC1–23 做
PCA，第一主成分描述平均形状，第二主成分描述不对称性，说明“峰区不对称/
双峰”可以用少量连续形状变量表示，而不必训练大型生成器。

但“能拟合”不等于“能预测”。Sabarinath & Anilkumar (2018) 用多变量回归
预测峰幅，再用二元 Laplace 描述双峰；方法能表达 Gnevyshev gap、两峰间隔和
峰时，但其 SC24 中心峰幅约 78±25，明显低于后来观测到的平滑峰值约 115。
这说明双峰形状模块可以改善“轮廓像不像”，却不能自动修正主峰幅度。

对本项目最合适的不是自由拟合六参数，而是受约束的两分量表示：

```text
y(t) = B(t) + a2 * phi((t - t2) / w2)
```

其中 `B(t)` 是已经固定主峰幅度和时刻的基线曲线，`phi` 是局部 Gaussian 或
raised-cosine 基函数。第一主峰不交给双峰分支决定；第二峰只在峰区活动，
谷区和下降段权重为零。参数上限建议先控制在：

```text
second-peak probability
second-peak time offset
second/main amplitude ratio
gap width or depth
```

四个参数已经足以表达第二个峰，不应一开始让深度网络自由生成整条曲线。

## 6. 历史相似轨迹可保留双峰，但必须做盲切分

Gerçeker et al. (2025) 对 raw 月均 SSN 做 Takens 嵌入和 simplex projection，
直接复制相似历史轨迹的演化，因此能自然产生 SC26 的双峰或平顶。该结果说明
“形状库/相似轨迹”比单峰 HWR 更适合表达双峰，但它的 split point 使用 SC25
前 60 月已知观测反选，不能视为 2019 年起的完全盲预报。

Soman & Sabarinath (2026，预印本) 进一步把 aa 幅度前兆与 DTW 历史轨迹组合，
在 SC21–24 的 24 个阶段回测中，其中 20 个阶段尚未观测到周期峰；报告峰值
MAE 为 `16.1`，等权组合为 `18.6`，仅 aa 前兆为 `25.1`，轨迹误差从 `26.1`
降至 `20.2`。其重要启发不是具体 16.1 可照搬，而是“幅度前兆定强度、
历史轨迹定形状”的分工；但这是预印本，且只回测四个周期，必须独立复现。

这条路线的可行性为中高，受益主要在曲线轮廓、下降段和双峰形态。主要风险是：

- 历史样本只有 24 个，近邻集合很容易被少数强周期主导；
- 参数若用 SC25/26 已知结果调整，会重演 Gerçeker 的 split-point 泄漏；
- 相似周期均值不能提供校准良好的峰值概率区间。

## 7. 深度条件模型不是首选，但可作为最后包装

Xiong et al. (2021) 已经采用“前兆回归预测峰幅 + HWR 生成轮廓”的两阶段
结构，说明先定主峰、再生成形状有现实先例；其 HWR 单峰无法表达双峰，正是
本项目局部残差可以补上的位置。Perez et al. (2018) 的 FiLM 证明条件变量可以
通过特征级仿射调制注入网络，但通用网络能力不保证太阳周期小样本可学。

本项目 E1b 已实际验证：FiLM 的幅度条件能被模型读取，但 amp 模式使 SC24
曲线 MAE 恶化 49.03%。因此不能把“加了 FiLM”当成双峰模型可行；在没有稳定
主峰条件前，端到端双峰 PatchTST 会把“主峰错误、第二峰错误、标签错误”混在
一起，无法归因。

## 8. 方案比较

| 路线 | 最少样本/参数 | 对主峰时间的潜在收益 | 对轮廓的收益 | 最大风险 | 当前判断 |
|---|---|---:|---:|---|---|
| 硬双峰分类 | 二分类标签 | 低 | 中 | 14/24 ambiguous，标签翻转 | 不采用 |
| 软双峰概率 + 局部残差 + 选择性部署 | 连续 support；1–4 个残差参数 | 中 | 中高 | 概率校准难，始终开启残差可能使预测变差 | 优先预研 |
| 二元 Gaussian/Laplace 形状 | 每周期 3–6 形状参数 | 中 | 高 | 历史拟合强、预测能力未知 | 先做表示审计 |
| PCA 形状主成分 | 2–3 个形状分数 | 中 | 中高 | 主成分可解释性有限 | 作为残差基 |
| DTW/simplex 历史轨迹 | 无深度训练 | 中 | 中高 | 近邻少、split 泄漏 | 独立盲测对照 |
| 端到端双峰 PatchTST | 24 个完整周期 + 月序列 | 不确定 | 可能高 | 过拟合且不可归因 | 暂不采用 |

## 9. 推荐路线

### P0：已完成的标签审计

保留 `double_support`、两峰时间、幅度比和峰间凹陷；不把
`all_config_consistent_cycles=0` 误读成所有标签随机。双峰报告以稳定双峰、
ambiguous 和稳定单峰三类展示。

### P1：先完成主峰与谷值

R1A 继续保留为局部峰值注入器。R1B 或其他幅度模型必须先让 raw/13m 双轨主峰
幅度达到可接受误差，并同时报告主峰时间和 SC25 谷时。双峰分支只能在主峰方案
冻结后接入，不能反向修改主峰结论。RF 保留为峰幅候选，不因跨周期不稳定而
删除；它与其他幅度供给器在同一条 R1A 注入器中做受控比较。

### P2：零训练双峰表示审计

在 PR24 或 R1A 基线上，对每个历史周期拟合 1–3 个紧支撑峰区基函数，比较：

- 只放一个主峰；
- 主峰 + 一个第二峰；
- 主峰 + 一个平顶/不对称基函数。

验收不看整条 MAE，而看：

1. 主峰时间是否不变或改善；
2. 主峰幅度是否不变或改善；
3. 谷区和下降段是否保持不变；
4. 峰区轮廓是否改善；
5. 稳定双峰周期与 ambiguous 周期是否出现不同的残差需求。

残差必须在每个预测折的起报点重新定义：

```text
r_fold(t) = observed(t) - forecast_from_fold_origin(t)
```

不能使用全周期最佳拟合后的 best-fit residual。基函数在主峰处和峰区边界应为
零，先保证主峰和谷区不被改变，再评价新增的平顶、不对称或第二峰结构。

若 P2 无法在稳定双峰周期中优于单峰基线，停止参数化双峰路线，避免为了
历史拟合强行制造第二峰。

### P3：软标签概率与低维参数预测

若 P2 通过，先训练 Ridge/logistic/RF 预测：

```text
double_support
second-peak time offset
second/main amplitude ratio
gap depth
```

分母样本仍是约 24 个周期，因此第一版最多使用 2–3 个低维目标，不能同时预测
全部四个参数。无显著概率时，第二峰 weight 直接归零；不能把“没有证据”变成
“必须有两个峰”。

P3 之后加入 selective deployment。CRC (2025) 的结论是 always-on residual
correction 在弱信号或噪声大时会伤害预测，因此每个训练折都要固定一个
deploy/no-deploy 门：

```text
只有验证集证明主峰、谷值和净误差不退化时，才部署残差
```

记录 non-degradation rate，而不是只报告平均 MAE。

### P4：历史相似轨迹盲测

以固定 split point 实现 simplex 或 DTW 历史轨迹，不允许使用 SC25/26 结果选参。
输出样本轨迹集合和范围，不把数字化他人曲线当作真值。该路线作为独立对照，
不能与双峰残差混在一起宣称收益。

### P5：深度模型只在通过前四步后考虑

若软概率和低维残差确有信号，再让 PatchTST 只负责基曲线，轻量头负责概率和
残差参数。不要让一个大 PatchTST 同时从 24 个周期学习峰幅、峰时、双峰类别
和完整轮廓。

## 10. 采纳门槛

双峰候选必须同时满足：

1. SC25 谷时、SC26 主峰时和主峰幅度不劣于当前主基线；
2. 第二峰概率在 LOOCV 或时间顺序验证中优于“按历史频率猜”；
3. 峰区轮廓改善不能靠整条曲线 MAE 掩盖；
4. 谷区和下降段不得出现方向一致恶化；
5. 稳定双峰和 ambiguous 周期分开报告；
6. 所有结果 raw/13m 双轨并报；
7. 只有主峰时间/幅度先通过，才允许双峰分支进入最终未来图。

## 11. 结论

RQ1：双峰真实存在，但当前总 SSN 标签只对少数周期稳定。应采用软标签或连续
峰区指标，不能硬分类。

RQ2：参数化二元 Gaussian/Laplace 和 PCA 能表示双峰；simplex/DTW 能保留
历史轨迹形态；端到端深度模型证据最弱。前两类适合做形状，后一类在 24 个
RQ3：最符合本项目主目标的路线是“R1A 主峰约束 + 1–3 个峰区基函数 +
软双峰概率”。它首先改善 SC26 峰区轮廓和可能的第二峰时间，对峰幅校准帮助
有限。它不应在 R1B 主峰尚未稳定前启动，更不能牺牲 SC25 谷时和 SC26 主峰。

本次定向检索没有找到一项已核验的同行评议研究，能够从起报点同时预测双峰
类别、第二峰参数、完整曲线，并明确证明主峰时间和谷值不退化。这个空白正是
本项目需要用小样本、分层验证去补的部分，也是不能直接照搬大型深度模型的原因。

## 参考文献

[1] S. Basu, A. Saha, and S. Ghosh, "Analyzing the Gnevyshev Gap across Odd–Even Solar Cycles (SC 19–25) Using Sunspot Number and Solar Radio Flux (F10.7 and F30)," Research Notes of the AAS, 2025.

[2] A. Kılçık and A. Özgüç, "One Possible Reason for Double-Peaked Maxima in Solar Cycles: Is a Second Maximum of Solar Cycle 24 Expected?," Solar Physics, 2013.

[3] K. Georgieva, "Why the Sunspot Cycle Is Double Peaked," ISRN Astronomy and Astrophysics, 2011.

[4] B. B. Karak, S. Mandal, and D. Banerjee, "Double Peaks of the Solar Cycle: An Explanation from a Dynamo Model," The Astrophysical Journal, 2018.

[5] F. Y. Li, N. B. Xiang, D. Kong, and J. L. Xie, "The Shape of Solar Cycles Described by a Simplified Binary Mixture of Gaussian Functions," The Astrophysical Journal, 2017.

[6] A. Sabarinath and A. K. Anilkumar, "Sunspot Cycle Prediction Using Multivariate Regression and Binary Mixture of Laplace Distribution Model," Journal of Earth System Science, 2018.

[7] J. Takalo and K. Mursula, "Principal Component Analysis of Sunspot Cycle Shape," Astronomy & Astrophysics, 2018.

[8] K. Gerçeker, A. Kılçık, A. Özgüç, and V. B. Yurchyshyn, "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025.

[9] V. Soman and A. Sabarinath, "Combining a Geomagnetic Precursor with Dynamic Time Warping for Solar-Cycle Peak and Trajectory Forecasting," Research Square preprint, 2026.

[10] Y. Xiong, J. Lu, K. Zhao, M. Sun, and Y. Gao, "Forecasting Solar Cycle 25 Using Comprehensive Precursor Combination and Multiple Regression Technique," Monthly Notices of the Royal Astronomical Society, 2021.

[11] E. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI, 2018.

[12] K. Petrovay, "Solar Cycle Prediction," Living Reviews in Solar Physics, 2020.

[13] A. Kumar and V. Kumar, "Forecast of Solar Cycle 25 Based on Hybrid CNN-Bidirectional-GRU (CNN-BiGRU) Model and Novel Gradient Residual Correction (GRC) Technique," Advances in Space Research, 2024.

[14] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting," ICLR, 2020.

[15] "Post-Training Corrections for Improved Time-Series Forecasting," arXiv:2505.15354, 2025.

[16] "Robust Residual Correction via Selective Deployment for Time Series Forecasting," arXiv:2512.22428, 2025.

[17] "Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals," arXiv:2402.02332, 2024.
