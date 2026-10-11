# 项目状态 — STATE.md（唯一事实源）

> **规矩**：本文件只放字段，**不放段落**；每行必须带证据指针；≤40 行。
> **其他文档不许再写"当前状态"**——需要状态时引用本文件（2026-10-11 景修批准的 L1 维护方案）。
> **核对方式**：`python3 scripts/check_docs.py`（三查：指针有效 / 时效 / 基线 ID 唯一），失败会非零退出。
> **更新时机**：① 每个实验收尾（跑后五件事的第五件）② 换分支/换环境 ③ 一轮讨论定了决策。

| 字段 | 值 | 证据指针 |
|---|---|---|
| 当前阶段 | GPU 新环境重建基线 + 调参中 | roadmap §一 状态表 |
| 当前任务 | 找 GPU **月均口径** 基线（lr / batch_size 优先） | `plan_basemodel_search_2026-10-11.md` |
| 当前基线（月均） | 阶段1 配置 EXP-18-5b；GPU 环境复核 = **EXP-26** | experiment_history：EXP-18-5b 行、EXP-26 行 |
| 下一阶段基线 | **平滑口径**（参数沿用月均基线），排在月均基线之后 | `plan_preregistration_2026-10-11.md` §P6 |
| 工作分支 | `GPU-basemodel-search`（本分支，2026-10-11 建） | `git worktree list` |
| 文档线 | `env-newserver-2026-10`（A/B 对齐已完成；C 类落本分支） | git log |
| 分支登记表 | 分支太多时先看 `BRANCHES.md`；**收敛计划待景修批** | `project_docs/BRANCHES.md` |
| 运行环境 | A6000 48G / torch 2.8.0+cu128 / numpy 2.2 / 无 venv | AGENTS.md 环境维护 |
| 数据口径 | SILSO **月均未平滑**，1749-02 ~ 2025-10 | data_pipeline.md |
| 用法口径 | **block24 无重叠**（minimax 判据） | EXP-23-R3 |
| 跨环境规则 | 全步 ≤4% 可用；block ≤10.6% 辅证；**滚动仅同环境** | EXP-23-R2 |
| 泄露核查 | 三种口径**无泄露** | data_pipeline.md §数据泄露核查 |
| 选模口径 | 最佳 val 模型（**体检中**，见 `plan_round0_diagnosis_2026-10-11.md`） | SOP §4 |
| 冻结队列 | C 类六条：平滑 / 相位特征 / 非负约束 / 谷时分类 / scheduled sampling / 多配置集成 | `plan_preregistration_2026-10-11.md` |
| 未决 | ①判定标准 6 项批注 ②推送与合并（目标 `EXP-17-stage1`）③交付口径形式 ④STATE.md 维护办法实施 | SOP 追踪表 |
| 最后核对 | 2026-10-11（AI 核对，景修待签） | — |
