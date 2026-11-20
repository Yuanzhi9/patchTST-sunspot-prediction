# PR132 标准 PatchTST 基线搜索预案

> 状态：待景修批准后执行  
> 任务：只搜索标准 PatchTST 的模型容量和训练超参数  
> 不包含：loss 改动、目标变换、真实相位、周期变量、aa/F10.7、数据增强、架构替换

## 一、为什么先做 PR132

PR24 已完成 seq、patch、lr、batch、dropout、heads、lr×batch 和五 seed 的主要证据；
PR132 当前只覆盖 26 个顺序配置，尚未搜索 `d_model/e_layers/d_ff` 的联合容量，
也没有覆盖 `SL264`、小数据模型和顶部候选同等多种子。

因此先补 PR132 标准基线。PR24 的容量审计排在 PR132 基线收口之后，作为最终冻结前的补充检查。

## 二、固定条件

| 项 | 值 |
|---|---|
| 数据 | `sunspot_with_cycle_monthly_2026-10.csv`，SHA256 `675f5ab2...eff69c` |
| 目标 | 月均未平滑 SSN |
| 输入 | `month_sin`, `month_cos`, `ssn` |
| 模型 | 标准 PatchTST |
| pred_len | 132 |
| 输出制度 | direct132，每 132 月一次调用 |
| loss | MSE |
| target_transform | 无 |
| scaler | StandardScaler，只用 inner train 拟合 |
| fold | `configs/cv5_folds_p132_2026-10.json` |
| checkpoint | inner val loss 最小 |
| 外部数据 | 无 |

## 三、搜索空间

采用条件约束的随机/TPE 搜索，不做完整笛卡尔积。

| 参数 | 候选 |
|---|---|
| seq_len | 132, 264, 336 |
| patch_len | 16, 24, 36 |
| stride | patch_len / 2 |
| d_model | 16, 32, 64, 128 |
| n_heads | 2, 4, 8，且必须整除 d_model |
| e_layers | 1, 2, 3 |
| d_ff | 128, 256, 512, 1024, 2048 |
| dropout | 0.1, 0.2, 0.3, 0.4, 0.5 |
| fc_dropout | 0.1, 0.2, 0.3 |
| learning_rate | log-uniform [1e-4, 3e-3] |
| batch_size | 16, 32, 64 |
| train_epochs | 50 |
| seed（搜索） | 2021 |

约束：

- `d_ff >= 2 * d_model`；
- 每折训练窗口数必须大于 0；
- batch/dropout 外其余参数不得隐式改变；
- 不调用 `train_window_limit`，避免把抽样控制混入基线。

## 四、评价与选择规则

开发阶段只使用 fold 1–4 的 `direct-cycle MAE`。fold 5 在搜索期间不读取。

主指标：

```text
mean_cycle_MAE(fold1..4)
```

并列指标：

- 最差开发折 cycle MAE；
- 全滑窗 MAE；
- 预测振幅/真实振幅比；
- 负值比例；
- seed 标准差。

选择规则：

1. 以平均 direct-cycle MAE 为主；
2. 差距小于 3% 时，最差折更好者优先；
3. 仍接近时，模型更小、推理更便宜者优先；
4. 不使用 fold 5 决定超参数。

## 五、执行阶段

### B0：预检与 smoke

- 生成搜索 manifest；
- 单 fold smoke：至少覆盖 `SL264`、`D16/H4/F128`、`batch16`；
- 检查 shape、日期、scaler、窗口数和有限值；
- 重新跑 p132 泄露检查。

预计 10–15 分钟。

### B1：首轮搜索

- 96 个 trial；
- 每个 trial 跑 fold 1–4；
- seed 2021；
- 固定 50 epochs；
- 保存每个 trial 的 aggregate 和曲线摘要。

按当前 A6000 吞吐估算，约 2–3 小时。

### B2：three-seed 复核

- 首轮前 10 名；
- 补 seed 2020、2022；
- 每名 2 × 4 fold = 8 次额外训练。

预计 20–40 分钟。

### B3：five-seed 复核

- 前三名；
- 补 seed 2019、2023；
- 每名 2 × 4 fold = 8 次额外训练。

预计 10–20 分钟。

### B4：deferred-fold 审计

- 冻结一个最佳候选；
- 五 seed 跑 fold 5；
- 该结果只用于审计，不再根据它改超参数。

预计 10–15 分钟。

### B5：收口

- 五折曲线；
- 逐提前期误差；
- 预测振幅比；
- 峰谷评价（修好匹配后）；
- 更新 result.txt、实验索引、STATE、AGENTS、roadmap；
- 提交但不自动合并。

预计 30–60 分钟。

总预算：约 3–5 小时，取决于搜索命中的大模型和 seq336 比例。

## 六、停止条件

- 最近 20 个 trial 没有超过 0.5% 的新最佳：停止首轮搜索；
- 全部候选都无法优于现有 PR132 候选约 41 SSN：停止普通超参数扩展；
- 出现 NaN、OOM、shape 错误：停止并修根因；
- 顶部配置明显依赖单折：不进入 five-seed；
- fold 5 与开发结果严重冲突：保留为审计警告，不通融参数。

## 七、与后续改进的边界

本次搜索结束后，才允许依次进入：

1. PR132-update12/24 正式评价；
2. DLinear/Ridge 架构对照；
3. loss/target transform；
4. 真实相位与周期变量；
5. aa/F10.7；
6. Waldmeier/残差；
7. 合成预训练与迁移学习。

## 八、评价隔离说明

当前五个 fold 已在上一轮 26 个配置中全部出现过，因此不存在完全未被看过的历史 holdout。
本轮把 fold 1–4 作为开发集、fold 5 作为“延迟审计”只能减少直接选择偏差，不能恢复成完全干净的测试。
论文级最终确认仍需 nested CV 或未来新周期数据。
