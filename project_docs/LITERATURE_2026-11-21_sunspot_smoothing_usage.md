# 太阳黑子深度学习中的平滑数据：使用方式与可预报边界

> 日期：2026-11-21  
> 范围：与 Cycle 25/26 预测及月度 SSN 深度学习直接相关的代表性论文。  
> 核心判断：平滑数据在文献中很常见，也合法可用；但“论文用了官方平滑值”
> 不等于“该值可以作为预测起点输入”。必须区分平滑发生在输入、标签、评价，
> 还是仅用于事后定义周期极值。

## 1. 问题

RQ1：太阳黑子深度学习论文实际使用什么数据集，13 个月平滑是否常见？  
RQ2：平滑值分别进入输入、标签还是评价，时间顺序上是否合法？  
RQ3：对本项目 Gaussian σ=3.6 与 Cycle 25/26 预测，哪些方案可进入主结果？

## 2. 方法与证据边界

检索采用三类视角：深度学习时间序列预测、太阳周期物理/周期界定义、
滚动或时间顺序验证。证据来自工作区论文原文、SILSO 官方文件说明和 Crossref
元数据。论文方法细节以 PDF 正文为准；只有摘要或元数据时不下具体实现结论。

证据分级：

- A：原文明确写清输入、目标和切分，并能判断预测起点是否只有历史信息。
- B：原文明确写清数据序列，但没有充分说明中心平滑边界或 vintage。
- C：综述或官方说明，用于定义数据口径，不作为某个模型的复现证据。

## 3. 数据集与平滑方式的分类

### 3.1 同一平滑序列同时作为输入和目标

Wang et al. (2021) 明确使用 SILSO V2.0 的 smoothed monthly SSN，LSTM 的
720 月输入预测未来 72 月，并对 Cycle 22–24 做 one-step/multi-step 回测。
Kumar and Kumar (2024) 同时比较 daily、monthly mean、13-month smoothed 和
yearly 四套 SILSO 数据；13-month 版本是把同一平滑序列拆成输入与预测目标。
Kumar and Kumar (2025) 又并行比较 13-month smoothed、yearly 和 monthly mean，
分别训练传统与混合 DL 模型。

这三篇证明“平滑序列直接训练”确实是常见做法，但它们没有共同建立一套
预测起点 vintage 审计。中心平滑在月份 t 使用 t−6 至 t+6；如果历史训练/评价
在任意切分边界把整条平滑序列直接使用，边界附近就可能带入未来 6 个月信息。
论文没有说明这一点时，只能记为 B 级，不能据此宣称业务预报无泄漏。

Kumar and Kumar (2025) 的切分说明还有内部不一致：正文称 60% 训练、40%
验证，同时前文又定义训练/验证/测试三部分，但未充分交代三者日期边界。因此
其 13-month 平滑结果尤其不能作为“真实未来预测”的直接证据。

### 3.2 未平滑月均序列直接训练

Benson et al. (2020) 用 SILSO monthly averaged observations，窗口 528 个月，
预测 132 个月，并采用始终把验证段放在训练段之后的 `TimeSeriesSplit`。
Dang et al. (2022) 用 monthly averaged total SSN，按 1749–1928、1929–1999、
1999–2022 做时间顺序切分，再用全部数据预测未来。Pala and Atici (2019) 使用
SILSO V2.0 monthly mean，采用 rolling-origin 重采样。Gerçeker et al. (2025)
更明确：为了预测 Cycle 25 余段和 Cycle 26，他们故意使用 unsmoothed monthly
mean，并把 13-month smoothed 结果作为另一种报告口径。

这一组说明，深度学习和非线性统计预测并不要求先平滑。原文给出的时间切分
普遍比“同序列中心平滑后直接切开”更容易审计。Gerçeker et al. 还指出，
平滑可能删除短时变化，而这些变化对状态依赖的动力学可能有用。

### 3.3 平滑只作为特征构造、极值定义或条件外推

Xiong et al. (2021) 对月均原始序列做 Gaussian σ=3.6 平滑，但模型不是端到端
LSTM，而是用上一个周期峰值、偏度、aa 指数和本周期起始值做多元回归，再用
HWR 轮廓生成整周期。其最后数据点约为 2020-10，报告的 Cycle 25 峰值是历史
条件下的统计外推，不是“把 2026 年中心平滑值输入网络”。这篇只能支持
Gaussian 轮廓特征的先例，不能直接支持中心平滑输入用于当前端点。

Hathaway (2015) 与 Petrovay (2020) 说明 13-month running mean 长期作为周期
极值标准；Hathaway 同时指出它对 1–3 年变化滤波不足，24-month FWHM Gaussian
给出的极值日期更稳定。这说明平滑首先是“周期界定义”问题，不能和模型输入
混为一谈。

Chapman (2026) 使用 13-month smoothed SSN 和 Hilbert 相位，但没有假装当前
周期末端的中心平滑值已经存在。其做法是先用 NOAA、McIntosh、NMSC 的未来
SSN 投影延长序列，再做平滑和相位分析，并把约 2.5–3 年后的 switch-off
作为更新点。这是本次检索中边界处理最清楚的做法：当前不可观测的中心平滑值
由明确的未来情景投影补齐，而不是偷偷读取真实未来。

## 4. 逐篇对照

| 研究 | 模型实际使用的序列 | 平滑进入哪里 | 时间顺序是否清楚 | 对本项目能支持什么 |
|---|---|---|---|---|
| Wang et al. 2021 | SILSO V2.0 smoothed monthly SSN | 同一序列同时作输入和目标 | 周期回测清楚；中心平滑边界未说明 | 平滑 DL 有先例，不足以证明当前端点因果合法 |
| Kumar & Kumar 2024 | daily、monthly mean、13-month、yearly | 四套独立同序列训练 | 90:10 切分，未给出完整 vintage 审计 | 平滑作为数据口径常见；不能据此免做泄漏检查 |
| Kumar & Kumar 2025 | 13-month、yearly、monthly mean | 三套独立同序列训练 | 60:40 与三段切分表述不一致 | 只能作结果对照，不能作严格实时预测证据 |
| Benson et al. 2020 | SILSO monthly averaged observations | 不平滑 | 是，严格时间顺序 CV | raw 月均是合法基线 |
| Dang et al. 2022 | SILSO monthly averaged total SSN | 不平滑 | 是，三段时间顺序切分 | raw 深度学习与未来外推的强先例 |
| Pala & Atici 2019 | SILSO V2.0 monthly mean | 不平滑 | 是，rolling-origin | raw 滚动回测可行 |
| Gerçeker et al. 2025 | monthly mean；13-month 仅作对照 | 主预测不平滑 | 是，library/prediction split | 支持 raw 主线和双口径报告 |
| Xiong et al. 2021 | monthly mean 经 Gaussian σ=3.6 | 特征与周期轮廓 | 预测依赖已知起始谷；不是实时端点网络 | 支持 σ=3.6 敏感性实验，不支持泄漏输入 |
| Chapman 2026 | 13-month smoothed + Hilbert phase | 相位与极值定义 | 是，先明确未来投影再平滑 | 可仿照“条件外推后平滑”的业务流程 |
| Hathaway 2015；Petrovay 2020 | 多种序列/综述 | 主要定义周期界 | 非模型论文 | 平滑是标准口径，但需说明具体滤波和用途 |

## 5. “合法吗”的两层回答

### 5.1 数据许可

合法。SILSO V2.0 月均、13-month smoothed 和年度数据公开，许可为
CC BY-NC 4.0；使用时需明确署名 WDC-SILSO。

### 5.2 科学方法

不能只回答“合法/不合法”，必须看时间位置：

1. **中心平滑后只用它定义历史周期界**：合法。它本来就是周期极值的标准口径。
2. **中心平滑只作为历史标签 y，输入仍为截至预测起点已知的 raw 或已经发布的
   平滑值**：可以作为回顾性监督目标。标签在 t 月用到 t+6，并不等于输入在
   起点偷看未来；但模型输出的含义变成“最终版平滑 SSN”，必须和同口径真值评价。
3. **中心平滑值直接放入预测起点之前的输入 x**：只有在 t+6 的观测当时已经
   发布时才合法。对当前 2026-10 起点，最新可用官方中心平滑只到 2026-04；
   把 2026-05 至 2026-10 的中心平滑值当输入使用了未来信息。
4. **因果平滑同时进入 x 和 y**：可以业务化，但它是另一种滞后指标。峰时和谷时
   必须与同一因果滤波后的真值比较，不能直接与 raw 月份或官方 13-month 月份混用。
5. **先预测 raw 未来，再在完整“历史+预测”序列上做中心平滑**：合法，也是
   Chapman 一类条件预测的处理方式；结果是条件曲线，不是无条件观测。

## 6. 对 B/C 方案的含义

方案 B 的准确结构是：

```text
输入 x_t = [month_sin_t, month_cos_t, ssn_raw_t]
目标 y_{t+h} = ssn_smoothed13_{t+h}
```

例如预测起点为 2026-10：

- x 只含截至 2026-10 的 raw 月均观测；
- y 是未来月份的最终版 13-month smoothed SSN；
- 预测 2027-04 的 y 会使用 2026-10 至 2027-10 的 raw 值，但这些值是预测
  结果未来才揭晓的标签定义，不在 2026-10 被喂回输入；
- 训练时 x 和 y 是两列，必须有独立 scaler 和逆变换；
- 当前 `Dataset_BlockCV` 用同一个 `ssn` 列同时构造 x/y，所以还没有真正实现 B。

方案 C 是 `x=y=同一个平滑序列`。其因果版采用单边滤波，当前起点的每个 x
都只用过去值，方法合法；centered 版在历史回测边界使用未来值，只能作
回顾性上界。Gaussian σ=3.6 的支持间隔约为 ±15 个月，因此 centered 边界
泄漏不只 6 个月。

## 7. 结论

1. 你记忆中的现象成立：有代表性的太阳黑子 DL 论文确实直接使用 13-month
   smoothed 月均数据集，而且不少使用平滑输入+平滑目标。
2. 但多数论文的重点是拟合或 postcast，不是 2026 年这一时刻的实时 vintage
   预报；不能把它们的平滑分数当作本项目可部署能力的证据。
3. 当前最合规的主结果应是 raw 输入 + raw 或因果平滑目标；centered 平滑模型
   只能列为回顾性上界。官方 13-month 平滑最新值比 raw 晚 6 个月，是这一点
   最直接的现实约束。
4. Gaussian σ=3.6 应先跑 causal 版本；centered 版本保留为文献口径和上界
   对照。若只报告 centered 的较低 MAE，会高估真实预测能力。

## 8. 参考文献

[1] Q.-J. Wang, J.-C. Li, and L.-Q. Guo, "Solar cycle prediction using a long short-term memory deep learning model," Research in Astronomy and Astrophysics, 2021.

[2] A. Kumar and V. Kumar, "Forecast of solar cycle 25 based on Hybrid CNN-Bidirectional-GRU (CNN-BiGRU) model and Novel Gradient Residual Correction (GRC) technique," Advances in Space Research, 2024.

[3] A. Kumar and V. Kumar, "Hybrid-Ensemble Deep-Learning Models to Enhance the Sunspot Prediction and Forecasting of Solar Cycle 26," Solar Physics, 2025.

[4] B. Benson, W. D. Pan, A. Prasad, G. A. Gary, and Q. Hu, "Forecasting Solar Cycle 25 Using Deep Neural Networks," Solar Physics, 2020.

[5] Y. Dang, Z. Chen, H. Li, and H. Shu, "A Comparative Study of Non-deep Learning, Deep Learning, and Ensemble Learning Methods for Sunspot Number Prediction," Applied Artificial Intelligence, 2022.

[6] Z. Pala and R. Atici, "Forecasting Sunspot Time Series Using Deep Learning Methods," Solar Physics, 2019.

[7] K. Gerçeker, A. Kilcik, A. Ozguc, and V. Yurchyshyn, "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025.

[8] Y. Xiong, J. Lu, K. Zhao, M. Sun, and Y. Gao, "Forecasting solar cycle 25 using comprehensive precursor combination and multiple regression technique," Monthly Notices of the Royal Astronomical Society, 2021.

[9] D. H. Hathaway, "The Solar Cycle," Living Reviews in Solar Physics, 2015.

[10] K. Petrovay, "Solar cycle prediction," Living Reviews in Solar Physics, 2020.

[11] S. C. Chapman, "A New Declining Phase Precursor and an Early Prediction of Cycle 26 Maximum," The Astrophysical Journal, 2026.

[12] SILSO World Data Center, "Monthly smoothed total sunspot number" and "What does smoothed monthly mean sunspot number actually mean?", Royal Observatory of Belgium.
