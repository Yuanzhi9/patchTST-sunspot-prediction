# PR24 基线验收与长时距方案决定（2026-11-23）

> 决定人：景修  
> 决策：接受当前 D128 模型作为 **PR24 reference baseline**，用于后续改进的
> 工程和统计对照；不把它宣传为已校准的 SC26 峰值预测模型。

## 1. 资质判断

当前模型可以作为基线，但必须区分两种“基线”：

| 类型 | 是否接受 | 含义 |
|---|---|---|
| PR24 工程参考基线 | ✅ 接受 | 配置、数据、训练、推理和指标都已冻结，后续改动必须与它同口径比较 |
| PR24 高精度峰值基线 | ❌ 不接受 | SC20 强周期峰幅低估 88.8，峰时命中率不足，不能当作可靠点预测 |

规范 ID：

```text
PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32
```

其中 `D128` 只表示 Transformer 的隐藏维度 `d_model=128`；当前 checkpoint
约 1,214,238 个参数。132 月输入经 patch36/stride18 和 end padding 形成 7 个
patch token，输出 24 月。

接受依据：

1. 五 seed 参数稳定性已确认，Full/Block24 = 26.83±0.10 / 26.48±0.25；
2. 容量审计完成，D112/D144/D160 未形成稳定改善；
3. 三周期时间顺序盲测、raw/clip0、峰谷误差和 seed 范围均已留档；
4. 预测脚本可从 checkpoint 重算现有 CSV，差异仅浮点误差；
5. 峰值能力边界已显式记录，不会把弱预测能力误写成强预测能力。

## 2. 后续改进的采纳门槛

任何候选改进必须使用本基线相同的数据快照、时间切分、输出长度、物理空间
反变换和 `clip0` 门禁。比较至少包括 SC20、SC21、SC24；若改动模型结构，
扩展到 SC18、SC19、SC22、SC23 作为补充稳健性检查。

主目标按以下顺序评价：

1. Cycle 25 谷值月份；
2. Cycle 26 峰值月份；
3. 峰幅和谷幅；
4. 完整曲线 MAE/RMSE；
5. seed 范围与区间覆盖率。

采纳规则：

- 时间：峰时或谷时在至少 2/3 周期上不劣于基线，且命中 ±6 月的 seed 比例提高；
- 振幅：至少 2/3 周期改善峰值幅度，单个周期恶化不得超过 10%；
- 曲线：三周期平均 MAE 不得恶化超过 5%；
- 稳定性：五 seed 方向不得由单个 seed 主导；
- 未来：不能根据 Cycle 25/26 尚无真值的结果回调模型；
- 报告：必须同时报告峰时、峰幅、曲线 MAE 和 seed 范围，不得只挑 MAE。

只有通过以上门槛，候选方法才可以替换 `PR24-SL132-PL36-S18-D128...` 成为
下一版正式参考基线。

## 3. “周期层面的幅度/时间模型”是什么

它指训练样本从“一个月窗”改为“一个太阳周期”：

```text
输入：周期起报时已经可见的信息
  - 上一周期峰幅、周期长度
  - 当前周期峰幅、峰后衰减速度
  - 距离最近极小的月数
  - 12/24 月斜率、曲率、波动
  - 可选 aa、F10.7、极区磁场

输出：
  - 下一周期峰幅
  - 下一周期峰时
  - 下一周期起点的谷时
```

它直接对应景修的主目标：25 周期谷值/时间、26 周期峰值/时间和曲线轮廓。
曲线可以随后由 PatchTST 或参数化形状模型生成，但“多大、何时”不再完全交给
24 月递归曲线自己推断。

评估必须是 leave-one-cycle-out 或严格时间顺序，不能随机拆月份。

## 4. 递归峰幅衰减的处理路线

项目内时间顺序盲测五 seed 平均 MAE：

| 推理方式 | SC20 | SC21 | SC24 | 三周期均值 | 当前可部署 |
|---|---:|---:|---:|---:|---:|
| PR24 observed update | 25.27 | 26.30 | 14.78 | 22.12 | 否，使用未来观测 |
| PR24 recursive block24 | 42.39 | 28.23 | 14.48 | 28.37 | 是 |
| PR24 recursive block12 | 47.97 | 32.86 | 13.19 | 31.34 | 是 |
| PR24 recursive block6 | 49.27 | 42.78 | 16.46 | 36.17 | 是 |
| PR132 direct | 52.15 | 38.84 | 29.82 | 40.27 | 是 |

结论：

1. 当前已有的简单方案里，block24 recursive 最稳健；继续扫描 block 长度优先级低。
2. PR132 direct 一次输出 132 月已经实测更差，不能把它当成现成替代方案。
3. observed-update 与完整递归相差约 6.25 MAE，这个差距说明误差累积是主要改进空间。

可尝试的新方案，按优先级排列：

| 方案 | 机制 | 可行性 | 预期收益 | 主要风险 |
|---|---|---|---|---|
| 周期级极值 + 条件曲线 | 先预测峰幅/峰时/谷时，再生成轮廓 | 高 | 最贴合主目标 | 24 个完整周期样本小 |
| rollout-aware 训练 / scheduled sampling | 训练时逐步混入模型自己的预测，缩小训练-推理差异 | 中高 | 可能缩小 observed-update 与递归的差距 | 训练不稳定，可能牺牲短期精度 |
| 多任务极值辅助头 | 曲线损失外加峰幅、峰时、谷时损失 | 中 | 让优化目标直接关心极值 | 辅助权重难定，可能扭曲曲线 |
| 直接多输出 132 月重新设计 | 一次输出但增加周期/振幅条件 | 中 | 避免块间反馈 | 一次决定整周期，方差大 |
| 概率/分位数输出 | 输出弱/中/强情景或区间 | 高 | 显著改善不确定性表达 | 不会自动修正中心偏差 |
| 硬分“高低波动”三组 | 分别训练三个模型 | 低 | 可能局部拟合更好 | 未来类别未知、每组仅约 8 个周期 |
| 多起点 block 集成 | 多窗口多次预测取中位数 | 低 | 降低随机波动 | 项目内失败模式高度相关，不能补信息 |

文献对递归、直接和多输出策略没有普遍赢家；策略优劣依赖序列长度、模型是否
正确指定和预测跨度 [1][2]。本项目的实测也支持“已有 block24 不差，但需要
处理递归反馈”，而不是简单换成一次性 132 月。

## 5. 外部数据与周期上下文

2026-11-23 已完成外部数据与周期上下文初筛。SSN、F10.7、WSO 极区场和由
SSN 计算出的周期特征和 aa、F10.7、WSO 原始/处理数据已有本地文件。

| 数据 | 本地状态 | 公共来源 | 可用起点 | 主要代价 |
|---|---|---|---|---|
| aa 指数 | 已清洗至 2026-10；有效从 1869-01 | ISGI | 1869 | 丢 1749–1868；月均需聚合 |
| F10.7 | 已清洗至 2026-10；缺 2026-07–09 | NCEI/NOAA SWPC | 1947 | 丢约 200 年历史；只能做缩短段对照 |
| 极区磁场 | WSO 1976 起已清洗；有效月值止于 2025-12 | WSO | 约 1976 | 当前端点缺测，不能进入 2026-10 正式预测 |
| 周期相位/历时/斜率 | 25 个周期上下文已生成 | 本项目 | 1749 | 历史端点须避免未来信息；仅 24 条完整目标 |
| 前一周峰幅、峰后衰减、波动 | 已进入周期上下文表 | 本项目 | 足够早的历史周期 | 24 个周期，统计功效有限 |

数据入口：

- aa：<https://isgi.unistra.fr/indices_aa.php>、<https://isgi.unistra.fr/data_download.php>
- F10.7：<https://lasp.colorado.edu/lisird/data/noaa_radio_flux/>
- WSO 极区磁场：<http://wso.stanford.edu/>

周期上下文 leave-one-cycle-out 初筛已完成：峰幅有一点信号，峰时/谷时没有
稳定超过中位数。建议下一批正式实验先用纯 SSN 周期上下文，再接 aa；
F10.7 和极区磁场作为缩短训练段后的覆盖期对照，而不是直接替换 1749 起的主数据。
详见 `project_docs/RESEARCH_2026-11-23_cycle_amplitude_regimes.md`。

## 6. 文献曲线对比

三联图：

- `analysis/literature_comparison/pr24_vs_literature_sc26_2026-11-23.png`
- `analysis/literature_comparison/pr24_vs_literature_sc26_2026-11-23.pdf`
- 绘图脚本：`scripts/plot_pr24_literature_comparison.py`
- 数字化数据：`analysis/literature_comparison/literature_sc26_monthly_digitized.csv`
- 极值表：`analysis/literature_comparison/sc26_extrema_comparison.csv`

图中的 Gerçeker 和 Dang 月曲线由论文图数字化得到，只用于比较形态、峰时和
振幅；它们不是论文原始模型数组。Kumar 使用论文图公开的年尺度数值。

主要观察：

- 峰时：PR24 `2035-06`，Gerçeker `2035-06`，Dang `2035-11`，Kumar `2036`，
  NASA `2034-12`；时间落在同一窗口。
- 峰幅：PR24 raw 约 106.95；其余高估计约 150–181；PR24 位于明显低端。
- 形态：Gerçeker 偏双峰/平顶；PR24 和 Dang 接近单峰；Kumar 为年尺度平滑曲线。

这支持保留当前峰时，同时把峰幅标为“低尾且可能低估”，不能把约 105 当作
已校准的 Cycle 26 峰值。

## 7. 后续待办

| 优先级 | 待办 | 状态 |
|---:|---|---|
| 1 | 修正 PatchTST channel-independent 导致 SSN 输出不读取其他输入通道的问题 | 待讨论 |
| 2 | 建立周期级幅度/时间特征表并用 leave-one-cycle-out 验证 | ✅ 初筛完成，峰幅弱信号；峰时/谷时未通过 |
| 3 | 设计极值头或两阶段条件曲线模型 | 待讨论 |
| 4 | 比较 scheduled sampling / rollout loss 与当前 block24 | 待预登记 |
| 5 | 接入 aa；F10.7/极区磁场作为缩短训练段对照 | aa/F10.7/WSO 均已清洗；训练暂缓 |
| 6 | 对文献三联图补充 CSV 来源、数字化误差和许可说明 | 部分完成 |
| 7 | 用同一组时间顺序周期评价所有候选改进 | 待执行 |

## 8. 参考文献

[1] S. Ben Taieb, G. Bontempi, A. F. Atiya, and A. Sorjamaa, "A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition," Expert Systems with Applications, 2012. DOI: 10.1016/j.eswa.2012.01.039.

[2] M. Marcellino, J. H. Stock, and M. W. Watson, "A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series," Journal of Econometrics, 2006. DOI: 10.1016/j.jeconom.2005.07.020.

[3] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks," NeurIPS, 2015. arXiv:1506.03099.

[4] K. Petrovay, "Solar cycle prediction," Living Reviews in Solar Physics, 2020. DOI: 10.1007/s41116-020-0022-z.

[5] R. H. Cameron, J. Jiang, M. Schüssler, and L. Gizon, "Physical causes of solar cycle amplitude variability," Journal of Geophysical Research: Space Physics, 2014. DOI: 10.1002/2013JA019498.

[6] P. Kumar, A. Biswas, and B. B. Karak, "Physical link of the polar field buildup with the Waldmeier effect broadens the scope of early solar cycle prediction," MNRAS Letters, 2022. DOI: 10.1093/mnrasl/slac043.

[7] K. Gerçeker et al., "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025. DOI: 10.1007/s11207-025-02577-y.

[8] Y. Dang et al., "A comparative study of non-deep learning, deep learning, and ensemble learning methods for sunspot number prediction," Applied Artificial Intelligence, 2022. DOI: 10.1080/08839514.2022.2074129.

[9] A. Kumar and V. Kumar, "Hybrid-Ensemble Deep-Learning Models to Enhance the Sunspot Prediction and Forecasting of Solar Cycle 26," Solar Physics, 2025. DOI: 10.1007/s11207-025-02510-3.
