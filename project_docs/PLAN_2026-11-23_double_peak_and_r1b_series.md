# 双峰轮廓与 R1B 幅度系列实验计划

> 日期：2026-11-23
> 状态：双轨峰值口径；R1B扩周期与双峰审计完成待审核；双峰训练/Gerçeker暂缓
> 主目标：Cycle 25 谷值/时间、Cycle 26 峰值/时间；完整轮廓次之

## 1. 研究问题

RQ1：太阳周期双峰是否具有可预测性，还是只能在事后识别？
RQ2：在单峰主轮廓上增加可选第二峰或残差，能否改善轮廓而不破坏主峰幅度和时间？
RQ3：R1B 的峰幅重构能否分别用 raw 峰和 13m 峰训练、验证，并扩展到更多周期？
RQ4：W 效应、aa 等前兆能否提供下一周期峰幅信息？

## 2. 文献结论

1. Gnevyshev gap 和双峰真实存在，但不同周期强弱不同。Basu et al. (2025)
   在 SC19–25 中发现 SC19–21 的 gap 不明显，SC22–25 更清楚；没有发现
   与奇偶周一对应的稳定规律。
2. Takalo & Mursula (2018) 的周期形状主成分分析指出，第二主成分描述周期
   不对称程度；Gnevyshev gap 可作为周期两部分的分离点。
3. Gerçeker et al. (2025) 的 simplex projection 保留历史相似轨迹，更容易
   出现双峰或平顶；参数化单峰模型会系统性删掉这种结构。
4. 项目此前的 M4 + PatchTST 残差实验失败，原因是训练使用 best-fit residual，
   测试使用 forecast residual，分布不一致。新的残差必须按相同预测起点定义。

结论：双峰可以研究，但不能强制每个周期都产生第二峰。应采用“单峰主轮廓 +
可关闭第二峰/残差”的结构。

## 3. 分阶段方案

### P0：双峰标签审计（零训练）

对全部历史周期，分别从 raw 和 13m 曲线提取：

- 是否存在至少两个显著局部峰；
- 第一峰、第二峰的日期和幅度；
- 第二峰是否高于主峰；
- 两峰间隔；
- 峰间最小凹陷深度；
- prominence 阈值 3/5/10 SSN；
- 13m 平滑窗口 13/17/21 月敏感性。

如果标签随阈值或窗口显著翻转，则停止“硬双峰分类”，只保留连续残差模型。

### P1：R1B 幅度模型重构

raw 峰与 13m 峰必须分开训练、分开评价、并列报告。两者都是正式比较口径，
不预先冻结单一主口径；最终由景修审核两套结果后决定采用哪一种或同时报告。
Ridge/RF 都保留为峰幅候选，尤其是 RF 的拔高效果继续研究；不因跨周期误差
不稳定而先删除候选。

目标表示优先试验：

```text
log(peak amplitude)
peak amplitude / previous peak
raw peak or 13m peak separately
```

特征集：

| 模型 | 特征 | 目的 |
|---|---|---|
| W0 | 上一周期峰幅、当前峰幅、衰减、奇偶 | 零外部依赖 |
| W1 | W0 + 上升月数、上升速率、下降曲率 | 检验 W 系列特征 |
| W2 | W1 + aa，同覆盖期控制 | 检验外部前兆增量 |

保守修正门：

```text
只有预测峰幅与 PR24 峰幅的差异超过训练验证误差，
且特征不处于训练范围外时，才允许改写峰幅；
否则保留 PR24。
```

这不是“压高低值”，而是避免没有证据地产生异常高值。

### P2：可选第二峰与残差

#### P2-A：概率模型

先预测第二峰存在概率：

```text
p_second = P(存在显著第二峰 | 起报时可见信息)
```

只有概率超过校准阈值时，再预测第二峰时间偏移、与主峰幅度比和峰间凹陷。
第一版使用 logistic/Ridge/RF，不使用大型神经网络。

#### P2-B：低维残差模型

```text
y_raw(t) = B(t) + sum_k beta_k phi_k(t)
```

`B(t)` 是 R1A 注入主峰后的平滑主轮廓；`phi_k(t)` 是只在预测主峰两侧
±H 月内活动的紧支撑基函数，并满足：

- `phi_k(t_peak)=0`，不直接改写主峰幅度；
- `phi_k(t_peak±H)=0`，不改变峰区外下降段和谷区；
- 第二峰幅度必须低于主峰；
- 无第二峰证据时，第二峰系数可以精确为 0。

第一版最多使用三类低维基函数：

- 峰区展宽/收窄基，表达尖峰或平顶；
- 左右不对称基，表达主峰前后肩部；
- 第二峰/峰间凹陷基，表达 Gnevyshev gap；

PCA 前 2–3 个周期形状主成分只作为固定基函数的对照，不与自由第二峰同时加入。

残差必须定义为：

```text
r_fold(t) = observed(t) - forecast_from_fold_origin(t)
```

不能复用 best-fit residual。

残差系数必须经过选择性部署门：

```text
只有当验证折证明主峰、谷值和净误差不退化时，才部署修正
```

记录 non-degradation rate。该设计来自 GRC、N-BEATS basis expansion 与
CRC selective deployment 的组合；详见
`project_docs/RESEARCH_2026-11-23_peak_residual_correction.md` 和
`project_docs/EVALUATION_2026-11-23_constrained_peak_residual.md`。

#### P2-C：Gerçeker 相似轨迹对照

实现 raw 月均 SSN 的 Takens 嵌入和 simplex projection：

- 搜索历史近邻状态；
- 对近邻轨迹加权；
- 输出主峰、双峰概率与轮廓范围；
- 固定 split 规则，禁止用 SC25/26 结果回调配置。

### P3：验证周期

主验证：SC20、SC21、SC24。  
扩展验证：SC18、SC19、SC22、SC23。

先做固定超参数的时间顺序验证，再做 leave-one-cycle-out 辅助检查。

### P4：T2 复审时机

T1-v2 只在 seed2021 筛选，且未达到 2/3 周期改善门槛，因此暂不直接启动
rollout loss。但这不构成对 T2 的永久否决。T2 使用固定的多块展开损失，
与 T1 的随机输入混合不是同一机制；等 R1B 幅度实验完成后重新审核一次。

## 4. 指标

1. Cycle 25 谷时；
2. Cycle 26 主峰时；
3. raw 峰幅、13m 峰幅分开报告；
4. 主峰是否为第一/第二峰；
5. 第二峰概率、日期和幅度；
6. 曲线 MAE。

高值不是错误；错误是高峰值没有对应前兆证据，或为了拉高 raw 峰而破坏
13m 峰和下降段。

## 5. 可证伪假设

- R1B raw 峰模型应至少在 2/3 周期降低 raw 峰误差，并且不能把 SC24 再推到 200；
- `log(A)` 或峰幅比应比直接预测绝对峰幅更稳定；
- 双峰概率模型应优于按历史频率随机猜；
- 第二峰时间应优于固定放在主峰后 N 月；
- 双峰残差不得降低主峰幅度和时间精度。
- 残差部署门应提高 non-degradation rate，而不是只在平均 MAE 上略好。

## 6. 预尸检

1. 强行制造双峰：没有 Gnevyshev gap 的周期被硬拆成两个峰。
2. 标签不稳定：平滑窗口或 prominence 改变后双峰标签翻转。
3. 样本不足：24 个周期中显著双峰更少，复杂模型易过拟合。
4. 残差分布漂移：重演 M4 best-fit residual 与 forecast residual 不一致。
5. 主峰身份歧义：第二峰偶尔高于第一峰，必须定义“主峰=最终最大峰”。
6. 高值惩罚：损失或正则化天然偏爱低值，导致 398 级历史峰值学不到。
7. raw/13m 混用：再次出现 SC24 这样的口径误判。
8. 时间被忽略：峰幅改善但峰时提前 6–12 月。

## 7. 启动顺序

并行阶段（只做数据审计，不训练深度模型）：

1. P0 双峰标签审计；
2. P1 的全部历史周期、raw/13m 双轨幅度模型。

串行阶段：

3. 先完成 P1 R1B 主峰幅度模型，由景修审核 raw/13m 双轨结果；
4. 主峰通过后，再做 P2-A/P2-B 条件双峰与残差；
5. P2-C Gerçeker 相似轨迹作为独立可部署对照；
6. T2 rollout loss 在 P1 完成后复审；
7. 通过门槛后才生成未来曲线并更新文献三联图。

并行只针对不依赖未来训练结果的数据审计。双峰形状模型不得与 R1B
主峰模型同时开始，避免主峰错误和第二峰错误无法归因。

## 8. 当前不做

- 不立即训练双峰 PatchTST；
- 不把双峰存在作为所有周期的硬约束；
- 不一次性加入 aa/F10.7/WSO；
- 不在主峰幅度未稳定时生成新的正式未来预测。

## 9. 参考文献

1. S. Basu, A. Saha, and S. Ghosh, "Analyzing the Gnevyshev Gap across Odd–Even Solar Cycles (SC 19–25) Using Sunspot Number and Solar Radio Flux," RNAAS, 2025.
2. J. Takalo and K. Mursula, "Principal component analysis of sunspot cycle shape," Astronomy & Astrophysics, 2018.
3. A. Sabarinath and A. K. Anilkumar, "Sunspot cycle prediction using multivariate regression and binary mixture of Laplace distribution model," Journal of Earth System Science, 2018.
4. K. Gerçeker et al., "Simplex Projection Predictions of the Remainder of Solar Cycle 25 and the Next Solar Cycle 26 Based on the Monthly Mean Sunspot Numbers," Solar Physics, 2025.
5. A. Kılçık and A. Özgüç, "One Possible Reason for Double-Peaked Maxima in Solar Cycles," Solar Physics, 2013.
