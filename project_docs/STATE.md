# 项目状态 — STATE.md（唯一事实源）

> **规矩**：本文件只放字段，**不放段落**；每行必须带证据指针；≤40 行。
> **其他文档不许再写"当前状态"**——需要状态时引用本文件（2026-10-11 景修批准的 L1 维护方案）。
> **核对方式**：`python3 scripts/check_docs.py`（三查：指针有效 / 时效 / 基线 ID 唯一），失败会非零退出。
> **更新时机**：① 每个实验收尾（跑后五件事的第五件）② 换分支/换环境 ③ 一轮讨论定了决策。
| 字段 | 值 | 证据指针 |
|---|---|---|
| 当前阶段 | 双峰审计、R1B 多周期双轨、RF 保留、残差方案评审完成；只差主峰方案收敛 | `project_docs/RESEARCH_2026-11-23_double_peak_modeling_options.md` |
| 当前任务 | 下一批 R1B-NEXT：先过五项泄漏审计，再做 W0 目标表示/单特征消融；未运行 | `project_docs/PLAN_2026-11-24_R1B_NEXT.md` |
| 当前基线（月均） | PR24 `D128` reference；D112/D144/D160 未稳定超过；历史锚点见账本 | `EXPERIMENT_B26M-capacity-refine_2026-11-21.md` |
| PR24 参考基线 | `PR24-SL132-PL36-S18-D128-H8-E2-F2048-DO02-FC02-LR3E4-BS32`；物理反变换后 clip0，不宣称峰幅已校准 | `project_docs/PR24_BASELINE_ACCEPTANCE_2026-11-23.md` |
| A0 口径对照 | raw Full/Block24=26.913/27.331；13m=16.439/17.903；归一化误差 0.400/0.263 | `EXPERIMENT_B26M-A0-raw-vs-smoothed_2026-11-23.md` |
| E1 Oracle | 幅值上界平均改善 13.61% 但谷值恶化；FiLM amp 使 SC24 恶化 49.03% | `EXPERIMENT_B26M-ORACLE-AMP_2026-11-23.md`、`EXPERIMENT_B26M-ORACLE-FUSION_2026-11-23.md` |
| R1 局部峰重构 | R1A Oracle H24 通过；SC20/21 MAE 改善 28.81%/5.01%，谷区不变；机制保留 | `EXPERIMENT_B26M-CONSTRAINED-PEAK_2026-11-23.md` |
| R1B 多周期双轨 | 24 周期 LOOCV + 目标 19–24 顺序扩测；W0 有信号；W1/aa 无稳定增量；RF 保留候选 | `EXPERIMENT_B26M-R1B-MULTICYCLE_2026-11-23.md` |
| R1B-RAW | Oracle 命中 raw 峰但改变轮廓；旧 Ridge/RF raw 对 SC24 预测 200.8/208.2 | `EXPERIMENT_B26M-R1B-RAW_2026-11-23.md` |
| 双峰审计 | 24 周期：9 stable double、1 stable single、14 ambiguous；禁止硬分类，训练暂缓 | `EXPERIMENT_B26M-DOUBLE-PEAK-AUDIT_2026-11-23.md` |
| T1 / T2 | T1 仅 SC21 改善 9.14%，未过 2/3 门槛；T2 暂停，R1B 后复审 | `EXPERIMENT_B26M-SCHEDULED-SAMPLING_2026-11-23.md` |
| 时间顺序盲测 | SC20/21/24 各 5 seed；recursive block24 MAE=42.39/28.23/14.48；PR132=52.15/38.84/29.82 | `analysis/cycle_holdout/cycle_holdout_aggregate.csv` |
| 非负能力检验 | raw/clip0 完成；clip0 负值=0；峰时仍不稳定 | `project_docs/REPORT_2026-11-21_PR24_nonnegative_baseline.md` |
| Gaussian σ=3.6 | causal 三周期×五 seed 完成；未解决强周期峰幅压制 | `EXPERIMENT_B26M-gaussian-s36_2026-11-21.md` |
| 工作分支 | p24=`BASE-2026-11-M`；p132 独立；8 条 worktree/分支登记全部保留，未清场 | `git worktree list`、`BRANCHES.md` |
| 远端状态 | `BASE-2026-11-M` 远端为 `ed49300`；本次调研归档本地待推；P132 不合并 | `git status --short --branch`、`git ls-remote origin refs/heads/BASE-2026-11-M` |
| 主线与分流 | **`EXP-17-stage1`** 仍是主线；`BASE-2026-11-M` 独立保留 | `BRANCHES.md`、`diary/2026-11-19.md` |
| 命名规范 | `PR24`/`PR132` 表示 prediction length；`PL36` 表示 patch length | `project_docs/NAMING.md` |
| 运行环境 | RTX 5090 32G；torch 2.8.0+cu128 / numpy 2.2 / 无 venv；历史 A6000 结果不可直接横比 | `AGENTS.md` 环境维护 |
| 数据口径 | raw 至 **2026-10**；13m 有效至 **2026-04**；A0 对照统一截到 **2026-04** | `data_pipeline.md` |
| 外部数据 | aa/F10.7 至 2026-10；WSO 有效月值止于 2025-12；周期上下文 25 条 | `project_docs/DATA_QUALITY_2026-11-23_external_indices.md` |
| 用法口径 | **block24 无重叠**（minimax 判据） | EXP-23-R3 |
| 跨环境规则 | 全步 ≤4% 可用；block ≤10.6% 辅证；**滚动仅同环境** | EXP-23-R2 |
| 泄露核查 | 三种口径**无泄露** | `data_pipeline.md` §数据泄露核查 |
| 选模口径 | ⚠️ 未定死：best-val 与 val 单起点轨迹基本打平，双轨并列报告 | `EXP-RULE-*_2026-10-11.md` |
| B26M 残余不确定性 | lr 仅三 seed 略优；batch/seq 结论限当前配方；五折不等于未来顺序审计 | `EXPERIMENT_B26M-followup_validation_2026-11-19.md` |
| 冻结队列 | 平滑 / 相位特征 / 非负约束 / 谷时分类 / scheduled sampling / 多配置集成 | `plan_preregistration_2026-10-11.md` |
| 未来预测 | 旧 PR24 基线：SC25 谷 2030-10；SC26 峰 2035-06；raw 106.95，13m 104.82；非新定稿 | `analysis/pr24_future_forecast/` |
| 文献对照 | 峰时与文献接近；raw 峰幅约 106.95，处于高估计 150–181 的低端 | `analysis/literature_comparison/` |
| 长时距备选 | block24 仍最稳；主峰候选按 forecast origin 比较，P2 残差审计未启动 | `project_docs/RESEARCH_2026-11-23_peak_amplitude_time_evidence.md` |
| 未决 | R1B-NEXT 未运行；RF/R1A 保留；时间模型仅在峰幅与泄漏门通过后启动 | `project_docs/PLAN_2026-11-24_R1B_NEXT.md` |
| 最后核对 | 2026-11-23 调研与明日计划已归档；本地提交待推送 | `diary/2026-11-23.md` |
