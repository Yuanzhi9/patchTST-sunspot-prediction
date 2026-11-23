# B26M T1 — Block-level scheduled sampling

| 字段 | 内容 |
|---|---|
| 日期 | 2026-11-23 |
| 实验 ID | `B26M-T1-SCHEDULED` |
| 类型 | 训练策略对照 |
| 目的 | 检验训练时逐步使用模型自己的 24 月预测作为下一块输入，能否缩小递归推理误差 |
| 对照 | `T0`：相同两 block 损失，但第二块始终使用真实输入 |
| 数据 | SILSO V2 raw；SC20/SC21/SC24 时间顺序留出 |
| 固定参数 | PR24 D128 配置；50 epoch；batch32；lr3e-4 |
| 筛选 seed | 2021 |
| 扩展 seed | 2019–2023，仅筛选通过后 |
| 训练入口 | `run_scheduled_sampling.py` |
| 坑清单回顾 | 坑 25/26、训练/验证口径不一致、长时间 rollout 不稳定、额外两 block loss 混淆归因 |

## 方法

每个训练样本提供两个连续 24 月目标块：

```text
x_t       = 132 月真实历史
y_t:t+24  = 第一目标块
y_t+24:t+48 = 第二目标块
```

T0：

```text
pred1 = model(x_t)
x_next = [x_t[24:], y_t:t+24]
pred2 = model(x_next)
L = L1 + 0.5 L2
```

T1：

```text
pred1 = model(x_t)
以概率 p(epoch) 把 x_next 中 SSN 通道替换为 pred1
L = L1 + 0.5 L2
```

`p=0` 至 epoch10，之后从 0 线性升到 0.3。只替换 SSN 通道；月份特征使用
未来已确定的时间编码。

验证损失固定使用“第二块输入预测值”，不随训练 `p` 改变，专门衡量 rollout 能力。

### v2 选模修正（重跑前登记）

首个 smoke 中，SC20 的 teacher 与 scheduled 测试曲线逐位相同。日志显示验证损失
最低点发生在第 3 epoch，而 scheduled sampling 从第 11 epoch 才启动，两个 run
因此都选了干预发生前的同一 checkpoint。这是实验设计缺陷，不是方法结果。

旧 run 保留但不进入结论。`T1-v2` 规定：

- teacher 与 scheduled 使用完全相同的候选 epoch；
- 只允许从第 25 epoch 起选择最佳 rollout-validation checkpoint；
- 第 25 epoch 时 scheduled sampling 已进入爬升段；
- 若第 25–50 epoch 无新低点，使用第 25 epoch checkpoint；
- 正式结果只写 `T1-v2` 目录和新实验 ID。

## 可证伪假设

- 如果 scheduled sampling 有效，T1 的 132 月递归曲线和峰幅应优于 T0。
- 如果只改善第二块训练损失、但完整递归曲线不变，说明收益未迁移到实际推理。
- 如果 T1 改善远期但显著破坏 24 月短期预测，则不能采纳。

## seed2021 筛选门槛

1. T1 相对 T0，至少 2/3 周期 132 月 MAE 改善 5%；
2. 任一周期 MAE 恶化不超过 10%；
3. 至少 2/3 周期峰幅和峰时不同时恶化；
4. T1 的 24 月首块误差不得比 T0 恶化超过 5%；
5. 无 NaN/Inf、无负值爆炸、日期与 132 月长度正确。

通过后扩展 T1 与 T0 的五 seed。未通过则不跑 rollout loss T2，先审阅失败机制。

## 预尸检

1. T0 本身新增第二块 loss；必须区分“长时距训练”和“使用预测回填”的贡献。
2. p 过大可能让前期训练崩塌；最多升到 0.3。
3. 验证阶段使用预测回填可能过早停止。
4. 只有约 2833 个训练窗口，加入第二块后可用窗口会减少。
5. 24 月条件只覆盖两次前向，不等于完整 132 月误差结构。
6. 若只有 SC20 改善，不能把收益推广到弱周期。
7. 若 24 月首块掉点明显，说明训练在牺牲真实近期预测。

## 产物

- `analysis/t1_scheduled_sampling_v2/screen/<cycle>/<mode>/seed2021/`
- `analysis/t1_scheduled_sampling_v2/t1_screen_detail.csv`
- `analysis/t1_scheduled_sampling_v2/t1_screen_summary.csv`
- `analysis/t1_scheduled_sampling_v2/t1_screen_decision.json`
- `logs/B26M-T1V2-*.log`

## 结果

T1-v2 seed2021：

| 周期 | Teacher MAE | Scheduled MAE | 改善 | 峰幅误差 Teacher→Scheduled | 峰时误差 Teacher→Scheduled |
|---|---:|---:|---:|---:|---:|
| SC20 | 38.221 | 38.796 | -1.50% | -79.95→-80.00 | 5→4 |
| SC21 | 34.668 | **31.500** | **+9.14%** | +46.72→+38.82 | 9→8 |
| SC24 | 13.159 | **12.895** | **+2.01%** | -2.62→-6.73 | -10→-10 |

选择到的 checkpoint epoch 分别为：

- SC20：teacher 37，scheduled 35；
- SC21：teacher 25，scheduled 34；
- SC24：teacher 29，scheduled 31。

所有 run 均为 132 月、无 NaN/Inf、clip0 无负值。没有周期 MAE 恶化超过 10%，
但只有 SC21 达到 5% 改善，门槛要求至少两个周期，因此筛选失败。

## 结论

1. Block-level scheduled sampling 在中等强度 SC21 有明确方向性收益，但不是
   跨周期稳定优势。
2. 强周期 SC20 的峰幅没有改善，弱点仍在“判断周期会有多强”。
3. 弱周期 SC24 总 MAE 略改善，但峰幅误差从 -2.62 增至 -6.73，不能称为极值改善。
4. `passed=false`，不扩展五 seed，不启动 T2 rollout loss。
5. 该负结果说明：仅改变训练时的输入混合方式，不能替代对周期幅度信息的建模。

图：`analysis/t1_scheduled_sampling_v2/t1_screen.png`。
作废的 v1 诊断：`analysis/t1_scheduled_sampling/INVALIDATED.md`。
