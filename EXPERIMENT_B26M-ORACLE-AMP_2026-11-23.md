# B26M E1a — Oracle 峰幅校正诊断

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-ORACLE-AMP` |
| 类型 | 分析任务（零训练；复用 PR24 cycle-holdout checkpoint 预测） |
| 目的 | 判断“即使峰幅已知，单靠强制峰幅正确”能否显著改善完整周期曲线和谷值 |
| 改动变量 | 无，评估 PR24 recursive block24 的不可部署 Oracle 幅值上界 |
| 对照 | 同 seed 的 `clip0` PR24 recursive block24 |
| 数据 | 1749-01–2026-10 raw 月均 SSN，冻结基线 |
| 周期 | SC20、SC21、SC24 |
| 种子 | 2019、2020、2021、2022、2023 |
| 评价 | 13 月平滑峰谷；曲线 MAE/RMSE/R²；峰谷幅度与时间 |
| 训练入口 | 无 |
| 坑清单回顾 | 坑 25（口径必须一致）、坑 26（曲线 MAE 不能替代极值）、跨周期样本少、Oracle 不可部署 |

## 方法

对每条已有 132 月 recursive block24 预测：

```text
baseline(t) = max(prediction(t), 0)
scale = true_13m_peak / predicted_13m_peak
oracle_amp(t) = max(baseline(t) × scale, 0)
```

`true_13m_peak` 从同一曲线真值计算。峰幅条件只在评价脚本读取，不进入模型训练。
峰时没有被校正，因此本实验只回答幅度条件能够贡献多少。

## 可证伪假设

- 如果“峰幅是主要瓶颈”，Oracle 幅值校正后至少 2/3 周期的曲线 MAE 改善，
  三周期平均改善 ≥5%。
- 如果无效，三周期 MAE 变化接近零或变差，说明还需要峰时/形状条件。
- 如果峰幅校正改善但谷值同步恶化，说明“乘法幅度条件”过于粗糙，不能直接进入 E3。

## 通过门槛

同时满足：

1. 至少 2/3 周期的曲线 MAE 改善；
2. 三周期平均 MAE 改善至少 5%；
3. 每个改善周期内至少 4/5 seed 同方向；
4. 至多一个周期的绝对谷幅误差恶化超过 25%。

不满足时：

- 不启动以峰幅乘法为主的 E1b；
- 优先检查峰时和归一化形状；
- Oracle 结果仍作为机制上界保留。

## 预尸检

1. 缩放把所有月份一起放大，可能把已经偏高的谷区推得更高。
2. SC20 的缩放倍数远大于 SC21/24，可能由强周期峰幅失真主导总体结论。
3. 只校正峰幅不会修正峰时，曲线 MAE 仍可能很差。
4. 五 seed 共用同一真峰幅，seed 改善方向可能高度相关；“4/5 seed”只检查随机性，
   不等于跨周期泛化。
5. 13 月平滑峰值和 raw 曲线峰值不是同一数值，脚本必须统一从平滑真值定义。
6. Oracle 结果不可部署，不能进入正式未来预测摘要。

## 运行命令

```bash
python3 scripts/eval_oracle_amplitude_2026-11-23.py
```

## 预期产物

- `analysis/e1_oracle_amplitude/oracle_amplitude_seed_detail.csv`
- `analysis/e1_oracle_amplitude/oracle_amplitude_aggregate.csv`
- `analysis/e1_oracle_amplitude/oracle_amplitude_decision.json`
- `analysis/e1_oracle_amplitude/<cycle>/oracle_amp_seed*.csv`
- `analysis/e1_oracle_amplitude/e1a_oracle_amplitude.png`
- `analysis/e1_oracle_amplitude/e1a_oracle_amplitude.pdf`
- `logs/e1a_oracle_amplitude.log`

## 结果

| 周期 | PR24 clip0 MAE | Oracle amplitude MAE | 改善 | 谷幅误差变化 |
|---|---:|---:|---:|---:|
| SC20 | 42.392 | 30.544 | +27.95% | +15.20 → +33.16 |
| SC21 | 28.198 | 24.482 | +13.18% | +3.51 → +2.16 |
| SC24 | 14.452 | 14.496 | -0.30% | +2.51 → +2.57 |

三周期平均 MAE 改善 `13.61%`。SC20/SC21 均为 5/5 seed 改善，
SC24 为 3/5。峰时不受幅值缩放影响，仍为 SC20/21/24
`4.8/10.8/8.0` 月。

预设门槛：

- 2/3 周期改善：通过；
- 平均改善 ≥5%：通过；
- 改善周期的至少 4/5 seed 同方向：通过；
- 谷幅恶化超过 25% 的周期不超过一个：SC20 被放大 118%，SC21 改善，
  SC24 基本不变，通过但触发警报。

最终判定：`proceed_to_e1b`，但 **禁止采用纯乘法幅值融合作为最终方案**。
SC20 高估的谷区被同比放大，说明条件模型必须同时约束谷值和形状。

## 结论

1. 峰幅确实是主要误差来源之一：只使用不可部署的真实峰幅，就能把
   SC20/SC21 的曲线 MAE 明显拉低。
2. 峰幅校正不能替代峰时预测，峰时误差完全没有变化。
3. 纯乘法缩放会把强周期本已偏高的谷区进一步抬高，SC20 谷幅误差增至
   `+33.16`；因此 E1b 不接受“将整条曲线乘以真实峰值”的方案。
4. 下一阶段应测试显式条件融合，同时把谷值/形状误差纳入训练或条件向量；
   如果仍只能改善峰幅而不能改善峰时与谷值，则只保留“极端条件上界”结论。

决策证据：`analysis/e1_oracle_amplitude/oracle_amplitude_decision.json`。
