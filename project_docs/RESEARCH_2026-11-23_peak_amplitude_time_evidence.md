# 太阳周期峰幅、峰时与谷时：按起报阶段选择模型

> 日期：2026-11-23
> 性质：定向文献调研；不运行模型
> 主目标：Cycle 25 谷值/时间、Cycle 26 主峰值/时间，曲线轮廓次之
> 当前状态：候选方案与证据已整理，等待景修审核

## 摘要

太阳周期峰幅预测的主要矛盾并不只是模型复杂度，而是起报点所包含的信息量。接近上一周期
极小期时，极区磁场等物理前兆对下一周期峰幅最可靠；在极小期前很早、且没有极区场时，
长提前量模型的跨周期误差显著增大。峰时预测通常比峰幅稍稳，但“精确到月”的长期预报
仍然缺乏证据，谷时与谷幅的证据更少。对 2026-10 起报的 SC26，现成最优路线不是再上
一个大型 PatchTST，而是同时保留三类可解释基线：低参数周期时序列模型、周期级
RF/Ridge，以及无泄漏的历史形状 analog，并按时更新；只有主峰和谷时稳定后，才接
R1A 和峰区残差。双轨 raw/13m 必须保留。

## 1. 研究问题

RQ1：在真正跨周期或严格时间顺序验证下，哪些模型能比气候学基线更可靠地预测峰幅？

RQ2：峰时与谷时是否存在可在 2026-10 起报点使用的稳定预测信息？

RQ3：在约 24 个完整周期样本下，简单周期模型、RF/Ridge、历史 analog 和深度模型应
如何分工？

## 2. 检索与证据标准

检索日期为 2026-11-23。使用 OpenAlex、Crossref、Springer、IOP、arXiv 和项目本地
PDF。重点检索：

```text
solar cycle amplitude precursor / polar field / aa / growth rate
solar cycle maximum time / minimum time / terminator / phase
deep learning solar cycle / N-BEATS / recurrent network / analog DTW
cycle holdout / out-of-sample validation / forecast origin
```

纳入直接讨论太阳周期预测、峰谷时相或跨周期验证的工作。只用题目和摘要不足以判断
验证方式时，读取本地 PDF 或出版社正文摘要。不同论文的数据版本、平滑口径、
起报时刻和误差定义不一致，因此不直接平均它们的峰幅。

| 证据类型 | 可信度 | 不能推出什么 |
|---|---|---|
| 周期留出或严格 out-of-sample | 较高 | 仍未必适用于 2026 这个具体端点 |
| 接近极小期的物理前兆 | 较高，但时窗受限 | 不能提前到任意年份使用 |
| 历史 postcast/拟合 | 中到低 | 不能等同于真实盲预报 |
| 单周期新闻式准确命中 | 低 | 不能证明方法稳定 |
| 多模型后见选择 | 低 | 不能根据 SC25/26 真值回选参数 |

## 3. 预测能力随时相变化

当前研究起报点是 2026-10。SC25 已过主峰，但它到下一极小期还有约 4 年，
SC26 峰还在约 9 年之后。不同方法在这个时点拥有的信息完全不同：

| 时相 | 可用信息 | 主要方法 | 主要限制 |
|---|---|---|---|
| 当前 2026-10 | 历史 SSN、SC25 峰后衰减、aa 历史、有限磁场端点 | 长提前量时序列、analog、周期级 RF | 极区场尚未到下一极小期，SC26 上升相未知 |
| 约 2029 switch-off | SC25 结束相、极区场积累 | Hilbert/terminator、polar precursor | 误差会明显收紧，但仍要更新 |
| 约 2030 极小期 | 轴向偶极/极区场、aa 最小值 | 物理前兆、SFT | 对峰幅最有信息，但已经接近 SC26 起点 |
| 2031–2032 上升相 | SC26 自身增长率 | Waldmeier、增长率和曲线拟合 | 已不是当前最早预测，但可更新 |

Karak 的 2026 年综述分析了约 100 个 SC24 预测和 130 多个 SC25 预测，指出多数方法
没有正确预测峰幅，ML 结果也不理想；极区场及代理量最有物理依据，但若在极小期前
过早使用，误差会增大。该文还指出，在上一周期峰之前预测下一周期没有意义。当前
2026-10 已经晚于 SC25 主峰，满足最低信息条件，但还没有进入极区场前兆最可靠的
时窗。

## 4. 峰幅模型的分层证据

### 4.1 接近极小期的物理前兆最强

Kumar et al. (2021) 比较 WSO、Mt. Wilson、Pulkovo 和 SFT 输出，发现极区前兆在
极向场反转后约 4 年达到 `r=0.8`，通常比下一极小期早 2–3 年；到极小期，
相关系数可超过 `0.9`。这是一条清晰的时间梯度：越靠近极小期，信息越强。

Upton & Hathaway (2023) 将曲线拟合与极区场/aa 前兆结合，预测 SC25
`134±8`、2024 年晚秋达到峰。N-BEATS 预测 `133.9±7.2`、2024-02，
TCN 预测 `135.3`、2024-10。以项目当前 SILSO 数据计算的 SC25 13m 峰为
`159.2`、2024-10，说明即使较新的物理前兆和深度模型也出现了幅度低估。
这些数字依赖数据版本，不能当作精确常数，但方向足以说明：**峰时可能接近，
峰幅仍未校准**。

### 4.2 周期内部增长率先验不能替代当前的外推

Cameron & Schüssler (2008) 证明强周期通常上升更快；Podladchikova et al. (2022)
用上升相最大增长率在 SC1–24 上得到 `r=0.93`，平均提前约 21 个月。该结果适合
SC26 一旦进入上升相后更新峰幅，不能从 2026-10 直接使用，因为 SC26 尚未开始。

### 4.3 低参数时序列模型是必须保留的基线

Petrovay (2024) 用 Gleissberg 振荡加奇偶周期残差建立一个非常低参数的时序列模型，
SC23/SC24 后报正确，并给出 SC26 `96.9±25.1`。这比高参数网络更符合约 24 个周期
的样本量，也可以作为“没有外部数据时的气候外推基线”。它不能证明 SC26 一定弱，
但它的存在说明：**当前的高峰值模型必须证明自己优于这种简单模型，而不是只优于
气候中位数。**

### 4.4 Analog/DTW 是当前起报点最有现实竞争力的数据驱动路线

Soman & Sabarinath (2026) 的 Solar Physics 论文用 DTW 选择历史相似周期，并在
SC21–24 上用早于目标周期的历史做严格 out-of-sample：

- baseline 模式不使用目标周期任何观测，平均 MAE 约 26 smoothed SSN；
- real-time 模式逐月更新，平均 MAE 约 19；
- 对 SC26 的 baseline 峰幅约 143。

该文同时明确使用固定随机种子、冻结数据快照和可复现程序。这是目前最接近本项目
“低参数、可解释、可更新、保留形状”的公开基线。仍需检查其时间误差、raw/13m
口径和本项目独立复现，不能直接抄 143。

Gerçeker et al. (2025) 的 simplex 给出 SC26 13m 峰 `137.4–146.2`、2035-06，
与 Soman 的约 143 相呼应；但它的 split point 用 SC25 前 60 月观测筛选，
不能视为完全盲预报。

### 4.5 深度模型能拟合，不等于跨周期可靠

N-BEATS、TCN 和循环网络在单个 SC25 预测上可以给出接近共识的时间与幅度，但
Karak 的 2026 综述指出这一类方法整体未显示稳定优势；项目自身的强周期 SC20
也出现约 89 SSN 的 13m 峰幅低估。对小样本问题，深度网络的误差很难拆成
“模型复杂度收益”还是“目标周期碰巧像训练数据”。

## 5. 峰时与谷时

### 5.1 峰时比峰幅容易，但长期精确到月仍困难

Uzal et al. (2012) 从极小期曲率同时回归峰幅、峰时和周期总长：峰幅相关
`R≈0.91`，上升时间相关 `R≈−0.77`，但 SC24 峰时的 90% 区间跨
2013-01–2014-09，宽度超过一年。该结果说明峰时有信号，却不支持把月份视为
高置信点值。

Wang et al. (2009) 用相似下降相给出 SC24 峰值 100.2±7.5、峰时
2012-05–10；实际峰时明显更晚。单一 analog 的成功不能代替跨周期验证。

Leamon et al. (2020) 用 Hilbert 相位识别 terminator，提前预测 SC25 onset
为 2020-05（+4/−1.5 月）。这类事件时间比完整曲线稳定，但它预测的是新周期
启动，不是 SC26 主峰月份。

Jha & Upton (2024) 用 AFT/极向场反转预测 SC25 峰在 2024-04–08；
这对峰时是有用证据，但依赖极向场和模型校准。

Foxon (2025) 比较大量 SC25 预测后指出，峰时预测整体似乎好于峰幅，但峰时
的公开样本更少。因此本项目可以把峰时作为独立目标，却不能把“时间比幅度好”
误写成“峰时已经准确”。

### 5.2 谷时和谷幅的证据更弱

Uzal et al. (2012) 对周期总长的相关只有 `R≈−0.56`，明显弱于峰幅和峰时。
Asikainen & Mantere (2023) 预测 SC25 结束于 2029-09，但误差约 `±1.9` 年。
这些结果说明谷值时相可以用周期长度、极小期曲率和 analog 做区间预测，但
目前没有一项被本项目核验的研究能支持从 2026-10 给出精确谷月或谷幅点值。

因此对 SC25 谷值，当前应报告季度到半年区间，并在 terminator/switch-off
接近时更新；单独一个“2030-xx”月份只能作为模型情景，不应作为已校准点值。

## 6. 项目内结果与文献的对照

项目 R1B-MULTI 的时间顺序扩测显示：

| 目标 | W0-RF MAE | 同覆盖期气候中位数 | 判断 |
|---|---:|---:|---|
| raw 峰幅 | 44.70 | 61.63 | 有弱信号 |
| 13m 峰幅 | 38.63 | 49.53 | 有弱信号 |

这支持“现有 SSN 周期上下文不是零信息”，但误差依然远大于接近极小期的物理前兆
文献通常报告的 15–25 SSN 量级，而且起报时相不同，不能直接宣称我们的模型
超过了 polar precursor。当前 RF 应保留，但应作为 2026 时点的候选，而不是
永久主模型。

当前 R1B 的 `W1` 和 aa 特征没有稳定增量，也不应立即解释为“W 效应/aa 无用”。
文献使用方法的时间窗更具体：Xiong et al. (2021) 使用前一周期最后四年
aa 最大值；Cameron 使用极小期前约 3 年的活动水平；Wang & Sheeley (2009)
强调 aa 最小值对应轴向偶极，而不是峰值。当前 W2 的起报覆盖窗与这些物理前兆
并不等同，因此结论只能是“这组 aa 特征失败”，不能扩大成“aa 永远无用”。

## 7. 推荐建模结构

### 7.1 当前 2026-10 的方案

保留 raw/13m 双轨，建立三类平行基线：

1. **气候中位数**：无技能底线；
2. **Petrovay 型低参数周期时序列模型**：检验长周期规律；
3. **W0-RF / W0-Ridge**：检验现有 SSN 周期上下文；
4. **DTW analog 独立盲测**：检验历史相似轨迹；
5. **极区场/aa 物理前兆**：若端点数据可用，只作额外物理对照，不作为默认主模型。

所有模型共享相同 forecast origin、目标定义和跨周期验证。输出不是一个“平均峰值”，
而是各模型及其历史误差区间。

### 7.2 峰时与谷时分开建模

- 峰幅：周期级回归或分类，不用月级 MAE 代替；
- 峰时：事件时间回归/区间预测，至少比较气候中位数、曲率/增长率、analog；
- 谷时：以周期长度和 switch-off 事件为主，先输出半年区间；
- 曲线：只有主峰和谷值稳定后，才接 R1A 和峰区残差。

### 7.3 分阶段更新

```text
2026-10：报告宽区间和模型分歧
2029 switch-off / 极区场更新：缩小 SC26 峰幅区间
2030 极小期：更新极区前兆和 SC25 谷时/谷幅
2031-2032 SC26 上升相：用增长率更新 SC26 峰幅和峰时
```

这不是拖延，而是把可用的信息按时间组织起来。长期预报若拒绝更新，等于把
2026 时点的信息上限误写成整个问题上限。

## 8. 方案评审

完整评审见
`project_docs/EVALUATION_2026-11-23_origin_aware_peak_time.md`。结论为
`Accept with Revisions`：方向值得做，但第一阶段只能比较低参数时序列、W0-RF
和 DTW analog；必须先固定 forecast origin，并单独评价峰时和谷时。

## 9. 结论

RQ1：当前可用模型的峰幅能力随起报时相变化。接近极小期的极区场前兆最有物理
依据；2026-10 只能使用长提前量模型和历史 analog，误差必须如实报告。RF 有弱
信号但尚未证明优于低参数时序列或 analog。

RQ2：峰时在部分研究中比峰幅稳定，但精确到月的长期误差仍大；谷时和谷幅证据
更弱，应报告季度/半年区间并等待 terminator、极小期和 SC26 上升相更新。

RQ3：24 个周期更适合三类低参数模型并行：Petrovay 型周期时序列、RF/Ridge、
DTW analog。PatchTST 应负责曲线轮廓，R1A 负责局部峰注入；端到端大模型只应
在低参数主峰模型稳定后作为对照。

## 参考文献

[1] K. Petrovay, "Solar cycle prediction," Living Reviews in Solar Physics, 2020.

[2] B. B. Karak, "Solar cycle prediction: challenges, progress, and future perspectives," Reviews of Modern Plasma Physics, 2026.

[3] P. Kumar, M. Nagy, A. Lemerle, B. B. Karak, and K. Petrovay, "The Polar Precursor Method for Solar Cycle Prediction: Comparison of Predictors and Their Temporal Range," The Astrophysical Journal, 2021.

[4] L. Upton and D. H. Hathaway, "Solar Cycle Precursors and the Outlook for Cycle 25," Journal of Geophysical Research: Space Physics, 2023.

[5] R. H. Cameron and M. Schüssler, "A Robust Correlation between Growth Rate and Amplitude of Solar Cycles: Consequences for Prediction Methods," The Astrophysical Journal, 2008.

[6] T. Podladchikova et al., "Maximal growth rate of the ascending phase of a sunspot cycle for predicting its amplitude," Astronomy & Astrophysics, 2022.

[7] K. Petrovay, "Predicting Solar Cycles with a Parametric Time Series Model," Universe, 2024.

[8] V. Soman and A. Sabarinath, "Analog-Based Forecasting of Solar Cycles Using Dynamic Time Warping and Residual Bootstrapping: Real-Time Application to Cycle 25," Solar Physics, 2026.

[9] K. Gerçeker, A. Kılçık, A. Özgüç, and V. B. Yurchyshyn, "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025.

[10] S. C. Chapman, "A New Declining Phase Precursor and an Early Prediction of Cycle 26 Maximum," The Astrophysical Journal, 2026.

[11] L. C. Uzal, R. D. Piacentini, and P. F. Verdes, "Predictions of the Maximum Amplitude, Time of Occurrence, and Total Length of Solar Cycle 24," Solar Physics, 2012.

[12] J.-L. Wang et al., "Predicting the start and maximum amplitude of solar cycle 24 using similar phases and a cycle grouping," Research in Astronomy and Astrophysics, 2009.

[13] R. J. Leamon, S. W. McIntosh, S. C. Chapman, and N. W. Watkins, "Timing Terminators: Forecasting Sunspot Cycle 25 Onset," Solar Physics, 2020.

[14] B. K. Jha and L. Upton, "Predicting the Timing of the Solar Cycle 25 Polar Field Reversal," The Astrophysical Journal Letters, 2024.

[15] T. Asikainen and J. Mantere, "Prediction of even and odd sunspot cycles," Journal of Space Weather and Space Climate, 2023.

[16] F. Foxon, "Solar Cycles: Can they be Predicted?," Research Notes of the AAS, 2025.

[17] J. Jiang, J. Wang, Q.-R. Jiao, and J. Cao, "Predictability of the Solar Cycle Over One Cycle," The Astrophysical Journal, 2018.

[18] Y. Xiong, J. Lu, K. Zhao, M. Sun, and Y. Gao, "Forecasting solar cycle 25 using comprehensive precursor combination and multiple regression technique," Monthly Notices of the Royal Astronomical Society, 2021.

[19] X. Su, B. Liang, S. Feng, W. Dai, and Y. Yang, "Solar Cycle 25 Prediction Using N-BEATS," The Astrophysical Journal, 2023.

[20] C. Zhao et al., "Solar Cycle Prediction Using a Temporal Convolutional Network Deep-learning Model with a One-step Pattern," The Astrophysical Journal Supplement Series, 2025.
