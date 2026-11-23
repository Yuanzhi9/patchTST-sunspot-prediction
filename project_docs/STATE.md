# 项目状态 — STATE.md（唯一事实源）

> **规矩**：本文件只放字段，**不放段落**；每行必须带证据指针；≤40 行。
> **其他文档不许再写"当前状态"**——需要状态时引用本文件（2026-10-11 景修批准的 L1 维护方案）。
> **核对方式**：`python3 scripts/check_docs.py`（三查：指针有效 / 时效 / 基线 ID 唯一），失败会非零退出。
> **更新时机**：① 每个实验收尾（跑后五件事的第五件）② 换分支/换环境 ③ 一轮讨论定了决策。

| 字段 | 值 | 证据指针 |
|---|---|---|
| 当前阶段 | E1、R1、T1 与平滑尺度复核完成；R1-A Oracle 通过，其余部署路径失败 | `project_docs/ANALYSIS_2026-11-23_smoothed_prediction_comparison.md` |
| 当前任务 | 先冻结 raw/13m 峰值主口径，再决定 R1-B 与下一轮幅度模型 | `project_docs/ANALYSIS_2026-11-23_peak_definition_raw_vs_smoothed.md` |
| p24 原预案缺口 | 中间容量审计完成；D160 五 seed 未稳定超过 D128；继续使用 D128 | `EXPERIMENT_B26M-capacity-refine_2026-11-21.md` |
| 当前基线（月均） | PR24 `D128` reference；历史锚点 E00/G1 见账本 | `result.txt`、`project_docs/experiment_history.md` Stage 10 |
| PR24 参考基线 | `PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32`；正式输出采用物理反变换后 clip0。可作改进对照，不宣称峰幅已校准 | `project_docs/PR24_BASELINE_ACCEPTANCE_2026-11-23.md` |
| A0 口径对照 | 同日期 raw 五 seed Full/Block24=26.913/27.331；13m=16.439/17.903；归一化误差 0.400/0.263 | `EXPERIMENT_B26M-A0-raw-vs-smoothed_2026-11-23.md` |
| E1 Oracle | 幅值上界平均改善 13.61% 但谷值恶化；FiLM amp 使 SC24 恶化 49.03%，安全门槛失败 | `EXPERIMENT_B26M-ORACLE-AMP_2026-11-23.md`、`EXPERIMENT_B26M-ORACLE-FUSION_2026-11-23.md` |
| R1 局部峰重构 | Oracle H24 通过；13m 口径 Ridge/RF 破坏 SC24，但 raw 口径 Ridge SC24 峰仅差1.16 | `project_docs/ANALYSIS_2026-11-23_peak_definition_raw_vs_smoothed.md` |
| T1 scheduled sampling | seed2021 仅 SC21 改善9.14%；2/3门槛失败，不扩五seed、不跑T2 | `EXPERIMENT_B26M-SCHEDULED-SAMPLING_2026-11-23.md` |
| 时间顺序盲测 | SC20/21/24 各 5 seed；PR24 recursive block24 MAE=42.39/28.23/14.48，PR132 direct=52.15/38.84/29.82 | `analysis/cycle_holdout/cycle_holdout_aggregate.csv` |
| 非负能力检验 | raw/clip0 双口径完成；clip0 负值=0；recursive MAE=42.39/28.20/14.45，峰时仍不稳定 | `project_docs/REPORT_2026-11-21_PR24_nonnegative_baseline.md` |
| Gaussian σ=3.6 | causal 三周期×五 seed 完成；centered 回顾上界暂停；causal 未解决强周期峰幅压制 | `EXPERIMENT_B26M-gaussian-s36_2026-11-21.md` |
| 工作分支 | p24=`BASE-2026-11-M`；p132=`BASE-2026-11-M-P132`；两分支均独立保留 | `git worktree list`、`BRANCHES.md` |
| 远端状态 | `BASE-2026-11-M` 已推送并与本地同步；P132 分支已推送；分支不合并主线 | `git status --short --branch`、`git ls-remote origin refs/heads/BASE-2026-11-M` |
| 主线与分流 | **`EXP-17-stage1`** 仍是主线；`BASE-2026-11-M` 独立保留、暂不合并（景修 2026-11-19 决定） | `BRANCHES.md`、`diary/2026-11-19.md` |
| 分支登记表 | 分支太多时先看 `BRANCHES.md`；**收敛计划待景修批** | `project_docs/BRANCHES.md` |
| 命名规范 | `PR24`/`PR132` 表示 prediction length；`PL36` 表示 patch length | `project_docs/NAMING.md` |
| 运行环境 | 当前 RTX 5090 32G；11.21 及更早记录为 A6000 48G；torch 2.8.0+cu128 / numpy 2.2 / 无 venv | `nvidia-smi`（2026-11-23）、AGENTS.md 环境维护 |
| 数据口径 | 当前官方快照：raw 至 **2026-10**；13m 有效至 **2026-04**；A0 对照两端统一截到 **2026-04** | `EXPERIMENT_B26M-A0-raw-vs-smoothed_2026-11-23.md`、`data_pipeline.md` |
| 外部数据 | aa 有效月值至 2026-10；F10.7 至 2026-10；WSO 有效月值止于 2025-12；周期上下文 25 条 | `project_docs/DATA_QUALITY_2026-11-23_external_indices.md` |
| 用法口径 | **block24 无重叠**（minimax 判据） | EXP-23-R3 |
| 跨环境规则 | 全步 ≤4% 可用；block ≤10.6% 辅证；**滚动仅同环境** | EXP-23-R2 |
| 泄露核查 | 三种口径**无泄露** | data_pipeline.md §数据泄露核查 |
| 选模口径 | ⚠️ **未定死（残余不确定性）**：候选 = best-val 与 "val 单起点轨迹"；细粒度复评后两者基本打平（最坏 22.5% vs 20.9%，n=6 组单种子）→ 暂按**双轨并列报告** | `EXP-RULE-*_2026-10-11.md`、`discuss_2026-10-11_选模规则.md` |
| B26M 残余不确定性 | ①lr=3e-4 仅三 seed 略优 ②batch 结论限当前 50 epoch 调度 ③seq 效应已部分隔离但仍有结构混杂 ④五折不等于未来顺序审计 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 冻结队列 | C 类六条：平滑 / 相位特征 / 非负约束 / 谷时分类 / scheduled sampling / 多配置集成 | `plan_preregistration_2026-10-11.md` |
| 未来预测 | 起报 2026-10；SC25 谷 2030-10（范围 2030-07–10）；SC26 峰 2035-06（范围 2035-02–12），13m 峰值 104.82（101.68–108.58） | `analysis/pr24_future_forecast/` |
| 文献对照 | 三联图已完成；峰时与文献一致，raw 峰幅约 106.95，处于文献高估计 150–181 的低端 | `analysis/literature_comparison/`、`project_docs/PR24_BASELINE_ACCEPTANCE_2026-11-23.md` §6 |
| 长时距备选 | 当前已测方案中 block24 recursive 最稳；下一步比较周期极值条件曲线与 rollout-aware 训练 | `project_docs/PR24_BASELINE_ACCEPTANCE_2026-11-23.md` §4 |
| 未决 | 峰值主口径未冻结：SC24 raw=146.1、13m=115.33；T2暂不启动；aa已就绪 | `project_docs/ANALYSIS_2026-11-23_peak_definition_raw_vs_smoothed.md` |
| 最后核对 | 2026-11-23（E1、R1、T1、aa 清洗；最新结果尚未推送） | `EXPERIMENT_B26M-SCHEDULED-SAMPLING_2026-11-23.md` |
