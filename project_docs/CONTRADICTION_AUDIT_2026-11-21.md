# 2026-11-21 文档矛盾审计

> 规则：以实验产物为事实源；较晚的实测结论覆盖早期机制假设；历史实验名称和
> 数值保留，但错误解释必须加明确修正。

## 已发现并修正

| 原表述 | 与什么冲突 | 处理 |
|---|---|---|
| StandardScaler 单独造成峰值压制 | StandardScaler 是线性可逆变换；`MSE_z=MSE_phys/σ²`，不能单独解释峰幅偏差 | phase1_summary、ONBOARDING、roadmap、experiment_history、EXP-16b/c 加修正；保留历史实验数值 |
| RevIN 反变换是峰值压制元凶 | EXP-20-1 关闭 RevIN 后滚动崩溃，证据反而支持 RevIN 是稳定性支柱 | 统一改为“机制已证伪，RevIN 保留” |
| PR24 容量审计已收口，但中间容量只有 seed2021 | D112/D144/D160 未完成五 seed，无法据此最终判定 | 补完五 seed；D112/D144/D160 均未稳定超过 D128，最终保留 D128 |
| PR24 raw 基线可直接作为正式预测 | raw 在 SC21/SC24 出现负值 | 正式输出统一为物理逆变换后 clip0；raw 只作诊断 |
| PR24 未来预测尚未生成 | 用户后续明确要求继续执行 | 已完成 2026-10 起报五 seed 预测与图件 |
| 当前候选尚未冻结 | 已有五 seed 稳定性、容量审计、非负能力检验 | ONBOARDING 和 STATE 已更新为 D128 技术冻结、clip0 正式输出 |
| 两个分支均已推送 | 当前 `BASE-2026-11-M` 新增本地提交尚未 push | STATE 改为“待景修审核后 push” |

## 保留但没有冲突的历史内容

- 早期“峰值压制机制假设”作为历史诊断保留，并标注后续修正，不删除历史实验。
- raw 与 clip0 的数值并列保留。二者不是重复实验：
  - raw 描述模型未经物理约束的原始输出；
  - clip0 是正式物理输出。
- PR24 与 PR132 的 MAE 只在同一真值、同一日期范围、同一曲线口径下比较。

## 最终事实源

- 容量细化：`analysis/p24_capacity_refine/capacity_refine_summary.csv`
- PR24 非负能力：`analysis/pr24_nonnegative_holdout/`
- 2026-10 未来预测：`analysis/pr24_future_forecast/`
- 状态：`project_docs/STATE.md`
- 交接：`project_docs/HANDOFF_REVIEW_2026-11-21.md`

## 当前检查

- 文档指针检查：通过；
- Stage 唯一性检查：通过；
- 本轮训练/评估进程：无；
- 持久化快照：`/openbayes/home/p24_review_snapshot_2026-11-21`。
