# B26M PR24 多任务极值辅助头：A0–A6 实验记录

> 日期：2026-11-27  
> 分支：`exp/pr24-multitask-extrema`  
> 对照：`PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32`  
> 状态：实验完整；PR24 全量辅助头不采纳

## 1. 结论

PR24 全量多任务模型在当前标签、权重和训练协议下没有稳定净收益，暂不替换
PR24 reference baseline。峰时的跨周期改善说明该结构仍有继续优化价值。

五 seed CV5：

| 模型 | Full MAE | Block24 MAE |
|---|---:|---:|
| PR24 baseline | 26.821 | 26.461 |
| 曲线 + 峰幅 + 峰时 + 谷时 | 26.988 | 26.797 |
| 变化 | +0.62% | +1.27% |

全量多任务在五个 seed 中只有 seed2021 的 CV5 结果改善；2019、2020、2022、2023
均未稳定改善。

时间顺序盲测：

| 周期 | baseline recursive MAE | 多任务 recursive MAE | 峰时绝对误差 baseline → MT | 峰幅误差 baseline → MT |
|---|---:|---:|---:|---:|
| SC20 | 42.392 | 44.372 | 4.8 → 4.4 月 | -88.80 → -94.49 |
| SC21 | 28.230 | 30.415 | 10.8 → 8.4 月 | +20.04 → +32.62 |
| SC24 | 14.483 | 13.403 | 8.0 → 6.8 月 | -0.66 → -5.08 |

峰时在 3/3 周期改善，但曲线 MAE 在 2/3 周期恶化，峰幅在 2/3 周期恶化。
SC21 曲线恶化 7.74%，超过 5% 门禁。因此这是“峰时有稳定信号、当前联合配置
未通过整体门槛”的权衡结果；应保留峰时分支继续改进，而不是否定整条路线。

## 2. 标签与实现

A0 标签审计：

- 训练窗口下一主峰、下一谷标签覆盖率均为 100%；
- 单 24 月窗口内实际包含下一主峰的只有 18.64%；
- 因此辅助目标使用“到下一主峰/谷的剩余月数”，不用块内 argmax；
- 峰幅用 raw，周期峰谷时间用官方 13m 边界；
- 周期事件标签只进入训练目标，不进入输入。

新增代码：

```text
PatchTST_supervised/data_provider/cv5_event_data.py
PatchTST_supervised/models/PatchTSTMultiTask.py
PatchTST_supervised/exp/exp_cv5_multitask.py
run_cv5_multitask.py
scripts/audit_pr24_event_labels.py
scripts/eval_pr24_multitask_aux.py
scripts/rollout_p24_multitask_cycle.py
scripts/plot_pr24_multitask_profiles.py
```

原 `cv5_data.py`、`PatchTST.py`、`exp_cv5.py` 未修改。

## 3. 单头和权重筛选

seed2021、fold 1–4 筛选：

| 实验 | Full MAE | Block24 MAE | 判断 |
|---|---:|---:|---|
| A1 baseline | 27.550 | 27.191 | 对照 |
| A2 峰幅单头 | 27.594 | 27.058 | full 略差，block 小幅改善 |
| A3 峰时单头 | 27.856 | 27.424 | 两项均变差 |
| A4 谷时单头 | 27.416 | 27.081 | 小幅波动 |
| A5 全量固定 w=0.1 | 27.361 | 26.671 | 四折最好，但五 seed 未稳定 |
| A6 固定 w=0.03 | 27.487 | 26.978 | 弱于 w=0.1 |
| A6 固定 w=0.3 | 27.416 | 26.774 | 弱于 w=0.1 |
| A6 不确定度权重 | 27.443 | 26.808 | 弱于 w=0.1 |

A5 辅助头自身也不强：

- 五折下一主峰幅度 MAE：59.10 SSN；
- 到下一主峰时间 MAE：33.29 月，±6 月命中约 10.1%；
- 到下一谷时间 MAE：31.93 月，±6 月命中约 9.1%。

## 4. 全量时间顺序盲测

三种权重方案都做五 seed：

| 权重 | SC20 recursive MAE | SC21 recursive MAE | SC24 recursive MAE | 判断 |
|---|---:|---:|---:|---|
| baseline | 42.392 | 28.230 | 14.483 | 对照 |
| 固定 w=0.1 | 44.372 | 30.415 | 13.403 | MT 中最好，但未过门槛 |
| 固定 w=0.3 | 45.953 | 31.916 | 14.964 | 未通过门槛 |
| 不确定度 | 43.633 | 35.719 | 16.612 | 未通过门槛且方差更大 |

固定 w=0.1 的峰时结果值得保留为机制线索；但辅助头自身峰时误差大，不能证明
曲线改善来自正确的事件时间预测。更可能是辅助损失对共享表示的弱正则化。

## 5. 审计与错误修正

并行执行时发现两类结果治理错误，均已修正并写入 SOP 坑 37–40：

1. 不同周期使用同名 `des`，会命中/复用同一结果目录；
2. config 文件名不含 seed，同秒并发启动会覆盖配置。

修复后：

- V2 的 `des` 包含周期号；
- config 文件名包含 seed；
- 每次评估前机器校验 result meta 的 `outer_val` 与 config；
- 旧的错误结果、配置、checkpoint 和日志整体移出 workspace；
- 最终 45 个 V2 chrono run 审计 PASS，结果数、seed、config 和周期切分全部一一对应。

审计证据：

```text
analysis/pr24_multitask_v2_audit.json
```

## 6. 曲线图

固定 w=0.1：

```text
analysis/pr24_multitask_figures/fixed_v2/
```

不确定度权重：

```text
analysis/pr24_multitask_figures/unc_v2/
```

固定 w=0.3：

```text
analysis/pr24_multitask_figures/w030_v2/
```

每个目录包含 SC20/SC21/SC24 的 recursive 与 observed-update 独立 PNG/PDF。

## 7. 最终判断

PR24 多任务辅助头应保留为“当前配置未稳定胜出”的改进分支：

- 峰时监督有方向性信号；
- 峰幅头、谷时头和联合曲线没有稳定增益；
- 不确定度权重和更高/更低固定权重均未解决曲线退化；
- teacher-forced 辅助监督没有自动解决 recursive 误差累积。

下一步若继续，应只围绕“峰时正则化”拆出极小实验，而不是继续堆四头权重。
PR132 的周期长度版本尚未启动；必须先决定是否值得把已失败的 PR24 结构迁移到
更长输出。
