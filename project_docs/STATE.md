# 项目状态 — STATE.md（唯一事实源）

> **规矩**：本文件只放字段，**不放段落**；每行必须带证据指针；≤40 行。
> **其他文档不许再写"当前状态"**——需要状态时引用本文件（2026-10-11 景修批准的 L1 维护方案）。
> **核对方式**：`python3 scripts/check_docs.py`（三查：指针有效 / 时效 / 基线 ID 唯一），失败会非零退出。
> **更新时机**：① 每个实验收尾（跑后五件事的第五件）② 换分支/换环境 ③ 一轮讨论定了决策。

| 字段 | 值 | 证据指针 |
|---|---|---|
| 当前阶段 | 2026-10 月均数据的 24 月基线验证收口（顺序块五折） | `plan_base_2026-11_monthly_cv5.md` |
| 当前任务 | B26M 主扫描与三项后续验证已完成；P36 为暂定候选，尚未冻结正式基线 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 原预案缺口 | lr×batch 小复核、seed 2022/2023、稳定性辅图、最终全量训练/未来曲线仍未完成 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` §原预案完成情况 |
| B26M 锚点 | E00 Full/Block=33.62/35.05；G1 seq132=27.82/27.86；完整结果见账本 | `result.txt`、`project_docs/experiment_history.md` Stage 10 |
| 暂定候选 | P36：seq132、p36/s18、lr3e-4、bs32、do0.2、nh8；3 seed Full/Block=26.83/26.46 | `EXPERIMENT_B26M-tuning_rounds.md` |
| 补强结论 | lr3e-4 三 seed 略优但非绝对；bs128/256 在当前配方五折全败；seq132 等窗口后仍胜 seq336 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` §结果 |
| 当前基线（月均） | 旧配置 EXP-18-5b / GPU 复核 EXP-26；仅作历史锚点，数据为 2025-10 | experiment_history：EXP-18-5b 行、EXP-26 行 |
| 下一阶段基线 | `BASE-2026-11-M`（P36 待景修确认后启用）；其后再做平滑口径 | `plan_base_2026-11_monthly_cv5.md` |
| 工作分支 | `BASE-2026-11-M`（2026-11-19 建） | `git worktree list`、`BRANCHES.md` |
| 远端状态 | `origin/BASE-2026-11-M` 已推送；分支独立保留、不合并主线 | `git ls-remote origin refs/heads/BASE-2026-11-M` |
| 主线与分流 | **`EXP-17-stage1`** 仍是主线；`BASE-2026-11-M` 独立保留、暂不合并（景修 2026-11-19 决定） | `BRANCHES.md`、`diary/2026-11-19.md` |
| 文档线（原） | `env-newserver-2026-10`（内容已并入主线；标签待归档） | git log |
| 分支登记表 | 分支太多时先看 `BRANCHES.md`；**收敛计划待景修批** | `project_docs/BRANCHES.md` |
| 运行环境 | A6000 48G / torch 2.8.0+cu128 / numpy 2.2 / 无 venv | AGENTS.md 环境维护 |
| 数据口径 | 当前官方快照：月均未平滑至 **2026-10**；13月平滑有效至 **2026-04**；历史实验结果仍基于 2025-10 快照 | `plan_base_2026-11_monthly_cv5.md` §一、data_pipeline.md |
| 用法口径 | **block24 无重叠**（minimax 判据） | EXP-23-R3 |
| 跨环境规则 | 全步 ≤4% 可用；block ≤10.6% 辅证；**滚动仅同环境** | EXP-23-R2 |
| 泄露核查 | 三种口径**无泄露** | data_pipeline.md §数据泄露核查 |
| 选模口径 | ⚠️ **未定死（残余不确定性）**：候选 = best-val 与 "val 单起点轨迹"；细粒度复评后两者基本打平（最坏 22.5% vs 20.9%，n=6 组单种子）→ 暂按**双轨并列报告** | `EXP-RULE-*_2026-10-11.md`、`discuss_2026-10-11_选模规则.md` |
| B26M 残余不确定性 | ①lr=3e-4 仅三 seed 略优 ②batch 结论限当前 50 epoch 调度 ③seq 效应已部分隔离但仍有结构混杂 ④五折不等于未来顺序审计 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 冻结队列 | C 类六条：平滑 / 相位特征 / 非负约束 / 谷时分类 / scheduled sampling / 多配置集成 | `plan_preregistration_2026-10-11.md` |
| 未决 | ①原预案剩余复核 ②候选冻结 ③最终全量训练与未来曲线 ④`pred_len=132` 另开 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 最后核对 | 2026-11-19（AI 核对，景修待签） | — |
