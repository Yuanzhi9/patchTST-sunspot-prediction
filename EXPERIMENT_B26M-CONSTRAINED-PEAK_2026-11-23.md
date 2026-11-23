# B26M R1-A — Oracle 峰局部幅值重构

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-CONSTRAINED-PEAK` |
| 类型 | 零训练 Oracle 上界诊断 |
| 目的 | 检验只在峰附近修改预测、保持谷区不变时，能否保留 E1a 的幅度收益 |
| 输入 | PR24 recursive block24 五 seed；SC20/SC21/SC24 |
| Oracle 条件 | 目标周期的真实 13m 峰幅和真实峰时位置 |
| 主方法 | 以当前预测峰为中心的 raised-cosine 局部修正 |
| 对照 | 同 seed `clip0` PR24；E1a 全局幅值缩放 |
| 训练入口 | 无 |
| 坑清单回顾 | 坑 25/26；峰时错位、平滑后峰移、谷区被误改、核宽事后挑选 |

## 方法

对每个 seed：

```text
y0(t) = max(PR24_recursive(t), 0)
t_pred_peak = argmax smooth13(y0)
A_true = max smooth13(true)
w(t; H) = raised-cosine kernel centered at t_pred_peak
          support: |t - t_pred_peak| <= H
y_local(t) = max(y0(t) + Δ × w(t; H), 0)
```

`Δ` 通过迭代校准，使 `smooth13(y_local)` 的最大值等于 `A_true`。修正核在
支撑区外严格为 0，因此峰前后很远处的曲线和谷区完全不变。

主实验固定 `H=24` 月。另跑 `H=12` 和 `H=36` 仅作核宽敏感性分析，
不根据结果事后替换主设定。

## 可证伪假设

- 如果“峰幅误差局部修正”可行，SC20/SC21 至少两个周期明显改善，
  且 SC24 与谷区不被显著破坏。
- 如果局部修正只能改善峰幅而总 MAE 不降，说明峰宽/尾部结构也是主要误差。
- 如果不同 `H` 方向相反，说明方法对人为核宽过敏感，不应进入预测条件阶段。

## 通过门槛

同时满足：

1. SC20 和 SC21 的曲线 MAE 相对 PR24 改善至少 5%；
2. SC24 的曲线 MAE 恶化不超过 10%；
3. 三周期绝对谷幅误差相对 PR24 均不得恶化超过 10%；
4. 至少 2/3 周期峰幅误差下降；
5. `H=24` 的结果不能由单个 seed 主导；
6. `H=12/24/36` 至少两种核宽在 2/3 周期上方向一致。

失败处理：

- 不进入用 Ridge/RF 预测峰幅的 R1-B；
- 记录“局部加法修正不足”，转到条件形状/模拟轨迹。

## 预尸检

1. 使用真实峰时位置定位核中心会过于有利，因此本轮结果只能作上界。
2. 13 月平滑会把局部尖峰摊平，简单一次加法可能无法命中目标峰幅。
3. 强周期 SC20 需要较大修正，可能改变峰值宽度。
4. SC21 当前峰已经偏高，修正方向为向下，可能产生新拐点。
5. SC24 接近正确，任何大改动都可能由“救治本来正常的周期”造成。
6. 核宽是人为选择；必须同时报告 12/24/36 月。
7. Oracle 精确峰时尚未被预测，R1-A 不能成为正式预测结果。

## 运行命令

```bash
python3 -u scripts/eval_constrained_peak_2026-11-23.py 2>&1 \
  | tee logs/r1a_constrained_peak.log
```

## 产物

- `analysis/r1a_constrained_peak/seed_detail.csv`
- `analysis/r1a_constrained_peak/aggregate.csv`
- `analysis/r1a_constrained_peak/decision.json`
- `analysis/r1a_constrained_peak/<cycle>/seed*.csv`
- `analysis/r1a_constrained_peak/r1a_profiles.png/.pdf`
- `analysis/r1a_constrained_peak/r1a_kernel_sensitivity.png/.pdf`

## 结果

固定 `H=24` 的 Oracle 结果：

| 周期 | PR24 clip0 MAE | 局部峰修正 MAE | 改善 | 峰幅误差 | 谷幅误差 |
|---|---:|---:|---:|---:|---:|
| SC20 | 42.392 | **30.181** | **28.81%** | -88.80 → 0 | 15.20 → 15.20 |
| SC21 | 28.198 | **26.784** | **5.01%** | +20.03 → 0 | 3.51 → 3.51 |
| SC24 | 14.452 | 14.436 | **+0.11%** | -0.65 → 0 | 2.51 → 2.51 |

核宽敏感性：

- `H=12`：SC20/SC21/SC24 MAE 为 `36.00/26.83/14.44`；
- `H=24`：`30.18/26.78/14.44`；
- `H=36`：`25.25/26.68/14.43`。

三种核宽在 SC20/SC21 改善方向一致；SC24 仅 3/5 seed 改善，但变化量约
0.1%，不能当作实质退化。

全部登记门槛通过。修正核在谷区权重严格为 0，因此三周期谷幅和谷时完全不变。

## 结论

1. “只修峰、不动谷”的局部幅值重构在 Oracle 条件下成立。
2. 它明显优于 E1a 的全局乘法：同样命中峰幅，但避免了 SC20 谷区被放大。
3. `H=24` 保留为主设定；`H=36` 数值更好只能作为敏感性证据，不能看到结果后
   改成主方法。
4. 当前收益仍使用真实目标峰幅定位与校准，不能当作可部署预测。
5. 下一阶段 R1-B 必须把 Oracle 峰幅替换成周期级 Ridge/RF 的预测值，检查收益残存。

图：`analysis/r1a_constrained_peak/r1a_profiles.png`、
`analysis/r1a_constrained_peak/r1a_kernel_sensitivity.png`。

补充的 13 月平滑对照：

- SC20 PR24 41.719→R1-A 24.368；
- SC21 23.096→20.592；
- SC24 11.012→11.015。

平滑尺度没有改变结论。详见
`project_docs/ANALYSIS_2026-11-23_smoothed_prediction_comparison.md`。
