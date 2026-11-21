# 项目状态 — STATE.md（唯一事实源）

> **规矩**：本文件只放字段，**不放段落**；每行必须带证据指针；≤40 行。
> **其他文档不许再写"当前状态"**——需要状态时引用本文件（2026-10-11 景修批准的 L1 维护方案）。
> **核对方式**：`python3 scripts/check_docs.py`（三查：指针有效 / 时效 / 基线 ID 唯一），失败会非零退出。
> **更新时机**：① 每个实验收尾（跑后五件事的第五件）② 换分支/换环境 ③ 一轮讨论定了决策。

| 字段 | 值 | 证据指针 |
|---|---|---|
| 当前阶段 | PR24 容量审计与三周期时间顺序留出盲测完成；当前未来 Cycle25/26 预测待做 | `EXPERIMENT_B26M-cycle-holdout_2026-11-21.md` |
| 当前任务 | 基于冻结 PR24 准备当前未来预测方案；不从盲测结果回改参数 | `EXPERIMENT_B26M-cycle-holdout_2026-11-21.md` |
| p24 原预案缺口 | 容量审计完成；参数搜索范围收口；剩余仅最终全量训练和当前未来预测 | `EXPERIMENT_B26M-capacity-audit_2026-11-21.md` |
| B26M 锚点 | E00 Full/Block=33.62/35.05；G1 seq132=27.82/27.86；完整结果见账本 | `result.txt`、`project_docs/experiment_history.md` Stage 10 |
| PR24 参数冻结基线 | `PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32`；预测交付待非负门禁 | `project_docs/PR24_BASELINE_FREEZE_2026-11-21.md` |
| p24 补强结论 | lr×batch 2×2 无排序翻转；seed2022/2023 未造成五 seed 漂移；稳定性图已出 | `EXPERIMENT_B26M-dual-baseline_2026-11-20.md` §p24 |
| 当前基线（月均） | 旧配置 EXP-18-5b / GPU 复核 EXP-26；仅作历史锚点，数据为 2025-10 | experiment_history：EXP-18-5b 行、EXP-26 行 |
| 时间顺序盲测 | SC20/21/24 各 5 seed；PR24 recursive block24 MAE=42.39/28.23/14.48，PR132 direct=52.15/38.84/29.82 | `analysis/cycle_holdout/cycle_holdout_aggregate.csv` |
| 非负门禁 | raw 预测存在物理负值；未处理前禁止把结果称为正式可交付基线 | `analysis/negative_smoothing/negative_summary.csv` |
| 平滑路线待做 | 13月平滑/ Gaussian σ=3.6 先做目标侧实验；居中平滑不能直接作为业务输入 | `project_docs/ANALYSIS_2026-11-21_negative_smoothing_cycle_dependence.md` |
| 工作分支 | p24=`BASE-2026-11-M`；p132=`BASE-2026-11-M-P132`；两分支均独立保留 | `git worktree list`、`BRANCHES.md` |
| 远端状态 | 两个分支均已推送；分支不合并主线 | `git ls-remote origin refs/heads/BASE-2026-11-M refs/heads/BASE-2026-11-M-P132` |
| 主线与分流 | **`EXP-17-stage1`** 仍是主线；`BASE-2026-11-M` 独立保留、暂不合并（景修 2026-11-19 决定） | `BRANCHES.md`、`diary/2026-11-19.md` |
| 文档线（原） | `env-newserver-2026-10`（内容已并入主线；标签待归档） | git log |
| 分支登记表 | 分支太多时先看 `BRANCHES.md`；**收敛计划待景修批** | `project_docs/BRANCHES.md` |
| 命名规范 | `PR24`/`PR132` 表示 prediction length；`PL36` 表示 patch length | `project_docs/NAMING.md` |
| 运行环境 | A6000 48G / torch 2.8.0+cu128 / numpy 2.2 / 无 venv | AGENTS.md 环境维护 |
| 数据口径 | 当前官方快照：月均未平滑至 **2026-10**；13月平滑有效至 **2026-04**；历史实验结果仍基于 2025-10 快照 | `plan_base_2026-11_monthly_cv5.md` §一、data_pipeline.md |
| 用法口径 | **block24 无重叠**（minimax 判据） | EXP-23-R3 |
| 跨环境规则 | 全步 ≤4% 可用；block ≤10.6% 辅证；**滚动仅同环境** | EXP-23-R2 |
| 泄露核查 | 三种口径**无泄露** | data_pipeline.md §数据泄露核查 |
| 选模口径 | ⚠️ **未定死（残余不确定性）**：候选 = best-val 与 "val 单起点轨迹"；细粒度复评后两者基本打平（最坏 22.5% vs 20.9%，n=6 组单种子）→ 暂按**双轨并列报告** | `EXP-RULE-*_2026-10-11.md`、`discuss_2026-10-11_选模规则.md` |
| B26M 残余不确定性 | ①lr=3e-4 仅三 seed 略优 ②batch 结论限当前 50 epoch 调度 ③seq 效应已部分隔离但仍有结构混杂 ④五折不等于未来顺序审计 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 冻结队列 | C 类六条：平滑 / 相位特征 / 非负约束 / 谷时分类 / scheduled sampling / 多配置集成 | `plan_preregistration_2026-10-11.md` |
| 未决 | ①候选口径复核 ②主任务选择 ③正式冻结 ④最终全量训练/未来曲线 | `/output/p132失效机制核查_2026-11-20.md` |
| 最后核对 | 2026-11-21（容量审计 + 完整周期盲测完成，待景修复核） | `EXPERIMENT_B26M-cycle-holdout_2026-11-21.md` |
