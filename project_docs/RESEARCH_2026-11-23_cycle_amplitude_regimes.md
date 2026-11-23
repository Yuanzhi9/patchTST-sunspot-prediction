# 太阳周期峰幅低估与波动分组可行性

> 日期：2026-11-23  
> 主目标：Cycle 25 谷值/时间、Cycle 26 峰值/时间，曲线轮廓次之  
> 性质：机制、文献与低成本可行性筛查；不是新模型验收

## 结论

当前约 105 的 SC26 峰值不能简单判成“必然低估”。文献估计从约
97 到 181，PR24 位于低端；它与 Petrovay 的简单周期时序列模型接近，但明显低于
Gerçeker 的 simplex projection 和高幅磁先验结果。因此准确表述是：

> PR24 给出弱到中等周期情景，峰时位于主流区间，但峰幅尚未校准。

把历史硬分成“高波动、低波动、一般”三组后分别训练，当前证据不支持。
在 24 个完整周期的 leave-one-cycle-out 筛查中，硬分组使峰幅 MAE 变差：

| 目标 | 全局 RF | 按可见波动三分组 RF | 气候中位数 |
|---|---:|---:|---:|
| raw 峰幅 | 57.24 | 60.33 | 59.24 |
| 13m 峰幅 | 47.97 | 48.35 | 47.30 |
| 峰后到峰时月数 | 18.17 | 17.56 | 16.13 |

峰时有一点局部改善，但仍不如“永远猜历史中位月数”。硬分组每组只剩约 8 个周期，
阈值和模型都不稳定。可继续研究的是“因果可见特征的软门控”，不是先验地把周期
贴上类别后拆成三个模型。

该 pilot 的协议是：只用起报时可见的 9 个周期特征；逐周期留一；每一折只用
训练周期重算 `postpeak_volatility_24m` 的 1/3、2/3 分位；测试周期按阈值进入
对应组，组内训练样本少于 8 条时退回全局模型；Random Forest 固定
`n_estimators=120/min_samples_leaf=3/max_features=0.7`。它只用于判断是否值得
正式实现，不作为性能结论。逐项数值见
`analysis/cycle_context/model_screening/regime_split_pilot_summary.csv`。

## 为什么强周期峰幅会被压低

### 1. 周期强度在起点不可唯一确定

SSN 月序列不是统计平稳序列。振幅受极区磁通量、Babcock-Leighton 型输运、
跨赤道磁通事件和随机源区倾斜等共同控制，周期长度、上升速度与形状也随强度变化。
仅看最近 132 月不能唯一确定下一周期的最终振幅。Petrovay 的综述指出，前兆类方法
在过去几个周期的总体表现优于纯外推类方法，并特别强调极区场前兆。

### 2. SSN 分支没有读取附加通道

当前 PatchTST 是 channel-independent。SSN 输出只沿 SSN 自己的通道传播；
`month_sin`、`month_cos` 等通道不会自动进入 SSN 预测。即使把 F10.7 或极区场
直接追加到 CSV，SSN 输出也不会自动利用它们。必须先做显式 fusion 或多任务头。

### 3. 递归回填放大振幅收缩

SC20 observed-update 的峰幅误差为 -30.1，recursive block24 扩大到 -88.8。
这说明一旦未来真实观测被模型自己的平滑预测替代，训练与推理输入分布不一致会进一步
把强周期拉向训练集常见的中等周期形状。

### 4. 月级 MSE 不是极值目标

训练损失对每月误差求平均，低值和中等活动月份数量占优；峰值月份很少，峰幅、
峰时和峰形没有单独损失。PR132 direct 对 SC20 的峰幅误差为 -121.8，
真实峰与预测峰跨周期相关约为 0，表明它主要输出平均周期轮廓，而非可靠地区分
强、中、弱周期。

## 文献给出的可行路线

1. **周期级前兆/统计模型**：Podladchikova 等以上升相最大增长率和极区场作前兆，
   对总 SSN 的周期后报相关达到 `r=0.93`；但其“增长率预测”主要是在同一周期峰值
   到来前预测该周期振幅，不等同于在 2026 年直接预测 SC26 峰幅。
2. **简单周期时序列模型**：Petrovay 用 Gleissberg 变化与奇偶周期规则预测
   SC26 为 `96.9 ± 25.1`。这提示只有约 24 个周期的任务更适合低参数周期模型，
   不应让高参数 PatchTST 从少量完整周期中自行记忆规律。
3. **磁先兆**：Kumar 等把极区场积累速率与 Waldmeier 效应联系起来；Pal 的
   2026 年比较认为轴向偶极矩比单一极区场更稳定。但这些都需要连续、可信的磁场
   数据，WSO 当前端点缺测到 2026-01，不能直接进入 2026-10 正式预测。
4. **近邻轨迹/经验动力学**：Gerçeker 等使用未平滑月均 SSN 的 Takens 延迟嵌入和
   simplex projection，保留历史相似轨迹的振幅和双峰结构；其 SC26 峰时为
   2035-06，raw 峰幅 `150.6–181.5`，13m 峰幅 `137.4–146.2`。

Gerçeker 的图看起来贴合历史并非偶然：它直接找相似轨迹，而不是强迫输出单一
平滑峰型。但其配置选择使用了 SC25 已知前 60 月观测，本地审计不能把它当作
2019 年起完全盲推。可借鉴近邻集合和双峰轮廓；不能照搬事后 split-point 选择。

## 周期级模型为何先做 Ridge/Random Forest

这里的样本单位是“一个周期一条记录”：

```text
输入：当前周期可见特征
输出：下一周期峰幅、峰时、谷时
```

Ridge 用少量线性权重检验信号是否存在；Random Forest 用浅树检查低阶非线性。
两者都是低参数筛查器。PatchTST 的参数单位是“月序列到多步曲线”，适合学轮廓，
但不能拿 24 个周期去训练一个高参数模型并期待可靠跨周期外推。合理顺序是：

1. Ridge/RF 先看信号是否超过气候中位数；
2. 若幅度有信号，再看峰时/谷时；
3. 通过逐周期留一和严格时间顺序后，才把特征做 FiLM/gating 或多任务头融入
   PatchTST；
4. PatchTST 负责轮廓，周期头负责振幅和时间，不能让三者一起端到端乱拟合。

当前筛查只支持“峰幅有弱信号”：raw RF MAE 为 55.73，优于气候中位数 59.24；
13m reduced Ridge 为 45.81，优于中位数 47.29。峰时和谷时没有稳定超过中位数。
因此还不能宣称周期级模型已经成为新基线。

把 2026-10 的 Cycle 25 可见上下文送入这些低参数模型，13m 峰幅为
`130.4–142.9`，raw 峰幅为 `188.0–206.1`；峰时为起报后 `104.6–106.1` 月。
13m 结果与 Gerçeker 的 `137.4–146.2` 接近，说明 PR24 的约 105 确实可能偏低；
但周期模型的逐周期留一 MAE 仍有约 46–58 SSN，不能把 `142` 当作校准后的点值。
这个差异只能说明“幅度的可识别范围尚未解决”，下一步应做周期头与曲线的条件组合，
而不是直接在两套结果间取平均。

## 建议顺序

| 优先级 | 实验 | 可行性 | 预期收益 | 主要风险 |
|---:|---|---|---|---|
| 1 | 周期级 Ridge/RF + 气候中位数基线 | 高 | 先证伪/证实周期特征是否有信息 | 仅约 24 个样本 |
| 2 | 极值辅助头：峰幅、峰时、谷时 | 中 | 直接把优化目标对齐主任务 | 辅助权重与标签定义敏感 |
| 3 | rollout-aware / scheduled sampling | 中高 | 缩小 observed-update 与 recursive 的约 6.25 MAE 差距 | 训练不稳，可能伤短期精度 |
| 4 | 两阶段：极值模型 + 条件曲线 | 高 | 曲线轮廓服务于已定极值 | 条件曲线可能只在少数周期成立 |
| 5 | 因果软门控/混合专家 | 中 | 比硬分组保留样本，按可见特征调模型 | 24 个周期仍易过拟合 |
| 6 | 硬分三组训练 | 低 | 可能局部拟合更好 | 未来组别不可知，样本太少 |
| 7 | 分位数/情景预测 | 高 | 明确表达弱、中、强情景 | 不会自动修正中心偏差 |

第一批正式实验应先加入 aa；F10.7 和极区磁场必须同时训练“不使用该特征、但覆盖期
相同”的对照。否则年代较短的数据集带来的分布变化会冒充特征收益。

## 数据可用性

| 数据 | 当前状态 | 进入模型的限制 |
|---|---|---|
| SSN raw | 1749-01–2026-10 已有 | 主数据 |
| 13m smoothed SSN | 有效到 2026-04 | 中心平滑末端缺 6 月，只能回顾或条件补齐 |
| aa | 待 ISGI 邮箱后下载 | 1868 起；必须做相同覆盖期控制 |
| F10.7 | 1947-02–2026-10 已清洗 | 2026-07/08/09 缺，long gap 不插值；只能缩短训练段对照 |
| WSO 极区场 | 1976 起已清洗；有效月值最后 2025-12 | 2026-10 起报点缺测，当前不能进入正式特征 |
| 周期上下文 | 已由 SSN 生成 25 条周期记录 | 24 条有完整目标，统计功效有限 |

## 参考文献

[1] K. Petrovay, "Solar cycle prediction," Living Reviews in Solar Physics, 2020. DOI: 10.1007/s41116-020-0022-z.

[2] K. Petrovay, "Predicting Solar Cycles with a Parametric Time Series Model," Universe, 2024. DOI: 10.3390/universe10090364.

[3] R. H. Cameron, J. Jiang, M. Schüssler, and L. Gizon, "Physical causes of solar cycle amplitude variability," Journal of Geophysical Research: Space Physics, 2014. DOI: 10.1002/2013JA019498.

[4] R. H. Cameron and M. Schüssler, "A Robust Correlation between Growth Rate and Amplitude of Solar Cycles: Consequences for Prediction Methods," The Astrophysical Journal, 2008. DOI: 10.1086/591079.

[5] T. Podladchikova et al., "Maximal growth rate of the ascending phase of a sunspot cycle for predicting its amplitude," Astronomy & Astrophysics, 2022. DOI: 10.1051/0004-6361/202243509.

[6] P. Kumar, A. Biswas, and B. B. Karak, "Physical link of the polar field buildup with the Waldmeier effect broadens the scope of early solar cycle prediction," MNRAS Letters, 2022. DOI: 10.1093/mnrasl/slac043.

[7] K. Gerçeker, A. Kılçık, A. Özgüç, and V. Yurchyshyn, "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025. DOI: 10.1007/s11207-025-02577-y.

[8] S. Pal, "Which Polar Precursor Better Predicts Solar Cycles: Axial Dipole Moment or Hemispheric Polar Flux of the Sun?," The Astrophysical Journal, 2026. DOI: 10.3847/1538-4357/ae4c5a.
