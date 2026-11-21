# 2026-11-21 离场交接与明日审核

> 状态：容量细化、PR24 非负能力检验和 2026-10 未来预测均已完成；
> 等待景修复核。当前未执行 git push。

## 明天先做什么

1. 审核 PR24 非负基线报告与五张图。
2. 审核 2026-10 起报未来预测、两张预测图与文献对照。
3. 审核 Gaussian σ=3.6 causal 结果，确认它只作辅助诊断，不替换 raw 主基线。
4. 明确批准或拒绝 push 当前本地提交。

## 当前运行状态

- 没有后台训练、推理或评估进程。
- 已完成实验不需要重跑。
- Gaussian centered 回顾性上界实验尚未启动；是否继续由景修决定。

## 持久化位置

- `/output` 是 CephFS 持久挂载，不是容器临时层。
- `/openbayes/home` 是指向 `/output` 的符号链接。
- 项目：`/openbayes/home/BASE-2026-11-M`
- Gaussian causal：`/openbayes/home/BASE-2026-11-M/analysis/gaussian_s36/causal`
- PR24 非负能力检验：
  `/openbayes/home/BASE-2026-11-M/analysis/pr24_nonnegative_holdout`
- 2026-10 未来预测：
  `/openbayes/home/BASE-2026-11-M/analysis/pr24_future_forecast`
- 独立 safety snapshot：
  `/openbayes/home/p24_review_snapshot_2026-11-21`

## 需要审核后批准的动作

### Push

- 当前分支：`BASE-2026-11-M`
- 远端：`origin/BASE-2026-11-M`
- 今日完成前已有 6 个本地提交尚未 push；今日实验提交为 `b03d8b1`。
- 景修明确回复“批准 push BASE-2026-11-M”后，才执行：

```bash
git push origin BASE-2026-11-M
```

不得自动 rebase、reset、force push 或合并其他分支。

### 未来预测审核

- 起报点：2026-10。
- Cycle 25 谷值：中位 2030-10，五 seed 范围 2030-07–2030-10。
- Cycle 26 峰值：中位 2035-06，五 seed 范围 2035-02–2035-12。
- Cycle 26 峰值 13m 平滑幅度：中位 104.82，范围 101.68–108.58。
- 需景修决定：是否接受“时间可用、幅度偏低风险高”的解释作为本日基线结果。

## PR24 非负能力检验图件

1. `analysis/pr24_nonnegative_holdout/pr24_nonnegative_profiles_2026-11-21.png`
2. `analysis/pr24_nonnegative_holdout/pr24_nonnegative_extrema_errors_2026-11-21.png`
3. `analysis/pr24_nonnegative_holdout/pr24_nonnegative_metric_distributions_2026-11-21.png`
4. `analysis/pr24_nonnegative_holdout/pr24_nonnegative_leadtime_mae_2026-11-21.png`
5. `analysis/pr24_nonnegative_holdout/pr24_nonnegative_peak_response_2026-11-21.png`

对应 PDF 同名存在。

## 2026-10 未来预测图件

1. `analysis/pr24_future_forecast/pr24_future_profiles_2026-10.png`
2. `analysis/pr24_future_forecast/pr24_future_extrema_2026-10.png`

对应 PDF 同名存在。

## Gaussian σ=3.6 图件状态

- causal 重训有逐 seed 曲线 CSV 和峰谷 JSON，但当前没有生成合并的总图。
- 原因是景修随后明确“只到测试 1 的图会重点检查”，并决定先完成 PR24
  非负基线检验；Gaussian 只保留为辅助实验。
- 原始留档完整，可随时无重训重画。

## 今日关键结论

- D112/D144/D160 五 seed 均未稳定超过 D128；D160 配对变化为
  Full +0.143%、Block24 +0.034%。
- PR24 clip0 recursive MAE：SC20 `42.39`、SC21 `28.20`、SC24 `14.45`。
- R²：`0.562 / 0.806 / 0.784`。
- 相对 persistence 技巧分：`+0.176 / +0.306 / +0.579`。
- 谷时：SC21/SC24 五 seed 全部命中 ±6 月，SC20 为 3/5。
- 峰时：SC20 为 3/5，SC21/SC24 各 1/5。
- 峰幅：`-88.80 / +20.03 / -0.65`，跨周期不稳定。
- 结论：PR24 可作曲线轮廓和谷时基线；当前不能稳定承担 Cycle 26 峰值点预测。
- 2026-10 未来预测时间：SC25 谷 2030-10、SC26 峰 2035-06；
  峰值显著低于 Gerçeker/Kumar/Dang 的高估计，必须保留偏低风险提示。

## 关键文件

- `project_docs/REPORT_2026-11-21_PR24_nonnegative_baseline.md`
- `project_docs/REPORT_2026-11-21_PR24_future_forecast_2026-10.md`
- `EXPERIMENT_B26M-pr24-nonnegative-holdout_2026-11-21.md`
- `EXPERIMENT_B26M-pr24-future-2026-10.md`
- `EXPERIMENT_B26M-gaussian-s36_2026-11-21.md`
- `project_docs/LITERATURE_2026-11-21_sunspot_smoothing_usage.md`
- `project_docs/STATE.md`
- `diary/2026-11-21.md`

## 审核签发

- [ ] 已检查 PR24 图与统计口径
- [ ] 已确认 nonnegative = physical inverse 后 clip0
- [ ] 已确认 Gaussian causal 不替换 raw 主基线
- [ ] 已确认没有运行中实验
- [ ] 已审核 2026-10 未来预测和峰幅偏低风险
- [ ] 已审核容量细化最终保留 D128
- [ ] 已批准 push
- [ ] 已确认未来预测是否作为正式基线结果
