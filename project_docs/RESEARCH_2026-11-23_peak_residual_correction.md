# 太阳周期峰区残差校正：从 GRC 到选择性部署的证据与方案

> 日期：2026-11-23
> 性质：残差文献定向调研与方法设计；不运行模型
> 主目标：保护 SC25 谷值/时间与 SC26 主峰值/时间，再改善峰区和双峰轮廓
> 当前状态：景修已确认结构方向；等待零训练表示审计，不运行模型

## 摘要

文献库中的 GRC 把 CNN-BiGRU 的基础预测和 AdaBoost 残差修正分成两级，但它
只做了短滞后、单步预测和 90:10 顺序切分，不能直接证明十年期太阳周期预测
可用。外部残差研究给出更重要的两条结论：N-BEATS 表明低维 basis 和残差堆叠
适合可解释时间序列预测；2025 年的“选择性部署”研究则表明，残差信号弱或噪声
大时，始终开启的修正器会过拟合并使预测变差。因此本项目不应训练一个无约束
残差网络，而应采用“主峰锁定 + 1–3 个峰区基函数 + 验证集选择是否部署残差”
的结构。该结构最有希望改善峰区轮廓和第二峰形态；它不能替代 R1B 对主峰幅度
的预测。

## 1. 研究问题

RQ1：已有残差校正方法分别在什么预测设定下有效？
RQ2：GRC 的“后处理残差”能否迁移到本项目完整周期预测？
RQ3：怎样把残差限制在峰区，既不改变 SC25 谷值和 SC26 主峰，又能表达
Gnevyshev gap、第二峰或平顶？

## 2. 检索与证据

检索日期为 2026-11-23。除文献库精读笔记外，使用 OpenAlex、Crossref 和
arXiv 核对题目、作者、年份和摘要。关键词包括：

```text
gradient residual correction solar cycle
post-training correction time series forecasting
selective residual deployment forecasting
neural basis expansion residual forecasting
progressive residual learning time series
```

证据强度分为：

| 工作 | 设定 | 证据等级 | 对本项目的限制 |
|---|---|---|---|
| GRC, Kumar & Kumar (2024) | lag11、单步、90:10 | 同行评议，但切分弱 | 不能证明跨周期十年预测有效 |
| N-BEATS (2020) | 多领域单变量点预测 | 强方法基线 | 非太阳周期，无峰区约束 |
| Post-training corrections (2025) | 已有预测器后处理 | 预印本 | 未验证 24 个周期的小样本 |
| CRC selective deployment (2025) | 多个骨干和长序列 | 预印本 | 最支持“不是所有残差都该启用” |
| Minusformer (2024) | 渐进残差学习 | 预印本 | 深度残差器对 24 周期仍偏重 |
| 项目 Level 3 | M4 + PatchTST 残差 | 本地实测 | best-fit 与 forecast residual 分布不一致，失败 |

## 3. GRC 能借鉴什么

Kumar & Kumar (2024) 的基础流程是：

```text
SSN 历史 -> CNN-BiGRU -> 基础预测
训练残差 + 数据梯度 -> AdaBoost -> 预测残差 -> 叠加
```

它可借鉴的是“基础预测与纠错器解耦”，而不是论文报告的 RMSE：

1. 基础模型负责已冻结的主轮廓；
2. 纠错器不应重写整条曲线；
3. 残差可以有独立的低维输入；
4. 必须保留基础模型作为无修正对照。

其关键局限是输入 11 个月、输出 1 个月且测试集紧接训练集。它测的是序列
续写，不是从 2026-10 外推十年。13 个月平滑序列的短步 ahead 预测受时间自相关
帮助很大，不能把 RMSE `1.64` 当成跨周期峰幅误差。对本项目而言，GRC 是
“后处理架构”的先例，不是性能证据。

## 4. 残差基函数与主峰锁定

N-BEATS 的 basis expansion 和本项目已有的 raised-cosine 峰修正共同提示：
残差可以由少量固定形状基函数表示，而不必用一个大网络预测整段残差。建议：

```text
y_raw(t) = B(t) + sum_k beta_k phi_k(t)
```

其中：

- `B(t)` 是 PR24 + R1A 主轮廓；
- `phi_k(t)` 是一组紧支撑基函数；
- 仅在预测主峰 `t_peak` 两侧 ±H 月内非零；
- `phi_k(t_peak)=0`，因此加残差不会直接改变主峰高度；
- `phi_k(t_peak±H)=0`，因此不会连续改变谷区和下降段；
- 第二峰基函数还要满足 `A2/A1 < 1`；
- 若无第二峰证据，`beta_second` 必须允许精确为 0。

第一版最多使用三种基：

1. 峰区展宽/收窄基：表达平顶或尖峰；
2. 左右不对称基：表达主峰前后肩部；
3. 第二峰或峰间凹陷基：表达 Gnevyshev gap。

建议不在第一版学习峰时。`t_peak` 由主极值模型给出，残差基围绕它构造；这样
峰时错误和形状错误可以分开归因。若必须允许小范围峰时修正，应单独加入
`δt_peak`，不要把峰时藏在自由残差里。

## 5. “选择性部署”比“残差越强越好”更重要

CRC (2025) 的摘要指出，always-on residual correction 在残差弱或噪声大时
会过拟合并增加总误差；它使用验证集校准的部署门，只在净误差下降时应用修正，
并报告非退化率。Post-training corrections (2025) 也采用轻量、模型无关的
后处理修正，而不是重新训练基础模型。

这与本项目高度相关，因为：

- R1A Oracle 证明峰区有可修正空间；
- R1A 换成 Ridge/RF 峰幅后，SC24 等周期会因幅度供给错误而恶化；
- 双峰标签有 14/24 个周期 ambiguous；
- 残差一旦在低置信周期开启，很容易把正确主峰改坏。

因此最终方案必须有“部署防火墙”：

```text
if 验证折预测置信度不足 or 修正后主极值/谷值恶化:
    do not deploy residual
else:
    deploy local residual
```

这不是看到结果后挑周期，而是每个训练折内用 nested validation 固定规则。

## 6. Minusformer 与深度残差器的取舍

Minusformer 通过多级减法逐步学习残差，报告了平均预测改善和一定抗过拟合能力。
其机制说明“渐进残差学习”可以做，但其目标仍是长序列平均误差，不是保护峰谷。
在本项目只有约 24 个完整周期、且 main extrema 尚未稳定的条件下，直接引入
深度残差器会把训练误差、主峰误差和第二峰概率混在一起。

建议先用 0 参数的固定基函数做表示审计，再决定是否需要可学习 basis。只有固定
基函数能够拟合历史 stable-double 周期的峰区残差时，学习残差系数才有意义。

## 7. 与项目 Level 3 失败的边界

Level 3 的失败原因必须保留：

```text
训练残差 = observed - best-fit M4
测试残差 = observed - forecast M4
```

两者不是同一分布，PatchTST 学到零均值噪声后把 MAE 从 3.32 推到 4.48。

新方案必须使用 forecast-origin residual：

```text
r_fold(t) = observed(t) - curve_generated_at_fold_origin(t)
```

即每一折都从该折起报点生成完整周期曲线，再用同一预测起点的真值计算残差。
不能对一个周期先做全周期最佳拟合，再拿拟合残差训练，再用于新的预测起点。

## 8. 修订后的残差架构

```text
Stage 1: 主极值模型
  - raw/13m 分离
  - 峰幅、峰时、SC25 谷时
  - RF 保留为候选幅度供给器，不作为淘汰项

Stage 2: 基础轮廓
  - PR24 recursive block24
  - R1A H=24 局部峰值注入
  - 主峰位置和幅度作为硬约束

Stage 3: 零训练残差表示审计
  - raw/13m 分开拟合，不共享残差系数
  - 固定 1–3 个紧支撑峰区基函数
  - 每个历史周期只在 forecast-origin residual 上拟合
  - 检查主峰、谷区是否保持不变

Stage 4: 少量残差系数预测
  - Ridge/RF 先做信号筛查
  - 输出峰区展宽、不对称、第二峰/凹陷三个低维系数
  - 不直接生成新的 P、T

Stage 5: 选择性部署
  - 修炼外/嵌套验证固定门槛
  - 修正后主峰或谷值恶化则不部署
  - 记录 non-degradation rate
```

## 9. 评估指标

主指标：

1. SC25 谷时误差；
2. SC25 谷幅误差；
3. SC26 raw/13m 主峰时误差；
4. SC26 raw/13m 主峰幅误差；
5. 五 seed 方向和最差周期。

双峰指标：

1. 第二峰概率 Brier score；
2. 峰间凹陷时间与深度误差；
3. 第二峰相对主峰幅度比误差；
4. 峰区 ±24 月 MAE；
5. 残差部署前后的主极值退化率。

辅助指标：整条曲线 MAE/RMSE。不得用整条 MAE 掩盖主峰或谷值恶化。

## 10. 可行性

可行性较高，因为固定基函数和 Ridge/RF 的训练成本低，现有 GPU 足够。主要
风险不是算力，而是“残差信号是否存在”和“残差是否 cross-cycle 稳定”。

预期收益应分栏判断：

| 目标 | 预期收益 | 原因 |
|---|---|---|
| SC25 谷时 | 低 | 残差限制在主峰区，谷区不动 |
| SC26 主峰时 | 中 | 平顶/第二峰可改变峰期形状，但不能替代主峰时模型 |
| SC26 主峰幅 | 低到中 | 需要 R1B/RF 先给对幅度，残差不能无中生有 |
| 双峰与峰区轮廓 | 中高 | 1–3 个基函数足够表达主要结构 |
| 不确定性表达 | 中 | 可输出单峰/平顶/双峰情景 |

## 11. 结论

GRC 提供的是“基础预测 + 后处理残差”的架构，不是可迁移的性能数字。N-BEATS
和 R1A 支持低维 basis 表达；CRC 则说明必须选择性启用修正，否则弱信号残差
会使结果变差。项目 Level 3 的失败进一步约束：只能学习 forecast-origin
residual。

当前最合理的方案是：

```text
保留 RF 作为主峰幅度候选
+ R1A 作为局部注入器
+ 1–3 个紧支撑、零主峰值、零谷区权重的峰区基函数
+ 验证集控制的 deploy/no-deploy 门
```

这比训练双峰 PatchTST 更直接，也比无条件残差叠加更安全。下一步仍应先做
零训练表示审计，不进入训练。

## 参考文献

[1] A. Kumar and V. Kumar, "Forecast of Solar Cycle 25 Based on Hybrid CNN-Bidirectional-GRU (CNN-BiGRU) Model and Novel Gradient Residual Correction (GRC) Technique," Advances in Space Research, 2024.

[2] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting," ICLR, 2020.

[3] "Post-Training Corrections for Improved Time-Series Forecasting," arXiv:2505.15354, 2025.

[4] "Robust Residual Correction via Selective Deployment for Time Series Forecasting," arXiv:2512.22428, 2025.

[5] "Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals," arXiv:2402.02332, 2024.
