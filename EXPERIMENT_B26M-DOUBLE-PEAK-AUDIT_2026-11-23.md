# B26M P0 — 双峰标签稳定性审计

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-DOUBLE-PEAK-AUDIT` |
| 类型 | 零训练分析 |
| 目的 | 判断历史周期中的双峰是否稳定、是否适合建立第二峰分类器 |
| 当前状态 | 景修已审核：取消硬双峰方向，训练与相似轨迹模型暂缓 |
| 数据 | SILSO V2 raw；周期边界来自 `cycle_boundaries_2026-10.csv`；SC01–SC24 |
| 平滑窗口 | raw、13、17、21 月 |
| prominence 阈值 | 3、5、10 SSN |
| 训练入口 | 无 |
| 坑清单回顾 | 平滑窗口敏感、prominence 阈值敏感、主峰/第二峰身份、周期样本少 |

## 方法

在每个历史周期区间内寻找显著局部峰：

```text
find_peaks(
    distance=6 months,
    prominence=threshold,
)
```

每个周期有 12 种配置：4 种序列口径 × 3 个 prominence 阈值。

定义：

- `double`：至少两个峰，且两峰间隔 ≥6 月；
- `support`：12 种配置中判为双峰的比例；
- `stable_double`：support ≥0.75；
- `stable_single`：support ≤0.25；
- 其余为 ambiguous。

## 可证伪假设

- 若双峰有稳定类别，至少 1/3 周期在多数配置中一致；
- 若标签高度受窗口/阈值控制，则不能使用硬分类，只能做连续残差；
- 若第二峰普遍高于主峰或峰值身份频繁交换，需要重定义“主峰”。

## 预尸检

1. raw 序列的小抖动被误判为第二峰。
2. 17/21 月平滑把弱的 Gnevyshev gap 抹掉。
3. 周期边界附近的首尾峰被截断。
4. 两个峰值接近时，主峰/第二峰身份可能交换。
5. 24 个周期中稳定双峰样本可能不足以训练分类器。

## 产物

- `analysis/double_peak_audit/peak_config_detail.csv`
- `analysis/double_peak_audit/cycle_stability.csv`
- `analysis/double_peak_audit/robust_double_cycles.csv`
- `analysis/double_peak_audit/double_peak_stability.png/.pdf`
- `logs/double_peak_audit_2026-11-23.log`

## 结果

24 个完整周期的分类结果：

| 类别 | 周期数 | 周期号 |
|---|---:|---|
| stable double | 9 | SC2、7、12、14、16、18、22、23、24 |
| stable single | 1 | SC3 |
| ambiguous | 14 | 其余周期 |

支持率定义：

```text
stable_double = 12 种窗口/阈值配置中至少 75% 判为双峰
stable_single = 12 种配置中至多 25% 判为双峰
ambiguous     = 其余周期
```

图：

- `analysis/double_peak_audit/double_peak_stability.png`
- `analysis/double_peak_audit/double_peak_stability.pdf`

图中黑格表示该配置检出至少两个显著峰，白格表示单峰或未检出；横轴依次为
raw、13m、17m、21m 平滑乘 3、5、10 SSN 的 prominence 阈值。

关键读数：

1. SC2、7、12、14、16、18、22、23、24 在至少 9/12 个配置中为双峰，标签较稳；
2. SC3 只有 3/12 个配置判为双峰，是稳定单峰；
3. 其余 14 个周期随平滑窗口或阈值改变而翻转，不能强行二分；
4. `all_config_consistent_cycles=0` 表示没有任何周期在全部 12 个配置中完全不变，
   不是“所有标签随机”或“一致率统计为零”。

历史双峰存在并不等于可在起报时稳定预判。当前结果为后续模型指出的可证伪方向
是“第二峰概率/低维残差”，不是立即训练硬分类器。

## 解释边界

- 当前标签依赖事后周期边界和窗口选择，不是起报时可观测标签；
- 只审计了是否存在两个显著局部峰，没有证明两个峰具有同一物理机制；
- 稳定性高不等于可预测性高；
- 9 个稳定双峰样本不足以支持一个高参数第二峰生成器；
- 如果以后建模，应先尝试连续“双峰支持度”或低维峰区残差，并以较低复杂度作为
  优先对照。

## 结论

1. 双峰标签存在，但对平滑窗口和 prominence 阈值敏感。
2. 只有 9/24 个周期是稳定双峰，14 个 ambiguous，样本与标签不足以支持硬分类。
3. 当前不启动双峰 PatchTST，也不启动 Gerçeker 相似轨迹模型；等 R1B 主峰幅度
   问题稳定后再讨论。
4. 后续若恢复双峰路线，应优先做概率模型或低维残差，并保留“单峰主轮廓”作为
   可关闭的基线。
5. 本报告已审核收口；双峰训练和相似轨迹模型仍不启动。
