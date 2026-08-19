# PatchTST 太阳黑子预测 项目 — AGENTS.md v2

## 项目主控文档
**`project_docs/project_roadmap.md`** — 所有阶段、任务、待办集合在此。本文件为项目规范和技术参考。

> **新来的 AI？** 先读 `project_docs/ONBOARDING.md`。3 分钟了解她是谁、项目做什么、今天踩了什么坑、怎么跟她沟通。

## 实验记录约定（2026-08-10 新增）
- **操作规程**：`project_docs/experiment_SOP.md`，跑任何实验前对照检查（全流程版 / 批量版）
- **快速索引**：`project_docs/experiment_history.md` 顶部索引表，Ctrl+F 查找实验
- **记录模板**：`project_docs/experiment_template.md`，每次跑新实验前复制、填写
- **数据管线**：`project_docs/data_pipeline.md`（归一化方法/数据流/硬编码约束）
- **工具回归测试**：`scripts/test_eval.py`（改 eval/roll_eval 后跑一次确认没坏）
- **跑实验后三件事**：① 追加一行到 `result.txt` ② 追加到索引表 ③ commit 推送
- **硬规则**：一次只改一个变量，每次标明对照基线是谁

## 文献笔记

14 篇精读结构化笔记，位于 `literature/`：
- **总览表 + 横向对比**：`literature/literature_reading_notes.md`（473 行）
  — 每篇一句话结论 + 与你项目的关系
- **每篇完整详注**：`literature/literature_reading_notes_detailed.md`（1214 行）
  — 三句话笔记、实验设计细节、关键数值、方法论判断
- 涉及方向：DL+sunspot 预测、物理综述（Hathaway/Petrovay）、GRC 残差校正、PatchTST/DLinear 原文、ARIMA 中文等
- 新实验前 Ctrl+F 搜索关键词 → 找到对应的详注 → 了解别人做到什么精度、踩了什么坑

## 当前阶段：探索期完成，判定标准待景修批注（2026-08-15）

阶段1（参数搜索+三窗口验证）✅、探索期（变换/loss/外推策略三系列 14 实验）✅、线B文献准备 ✅ 全部完成。判定标准 6 项待景修批注（`project_docs/judgment_criteria_discussion_2026-08-15.md`），批注后回填结论并启动组合实验 C-1（pow23+wmse_th）/ C-2（block+wmse_th）。详见 `project_docs/project_roadmap.md` 和 `project_docs/exploration_analysis.md`。

## 环境维护

| 项目 | 状态 |
|------|------|
| Git credential | `~/.git-credentials` 已配置，有效期至 **2026-10-10**。过期后需重新生成 GitHub token 并更新 |
| Token 获取 | https://github.com/settings/tokens → Generate new token (classic) → 勾选 `repo`

## 项目目标（当前定位）
使用深度学习模型探索太阳黑子数（SSN）预测的可行性边界，核心关注第 25、26 太阳活动周的完整曲线预测。M4 Waldmeier 参数化曲线（师兄代码）做物理方法对照基线。当前定位偏向"可行性边界研究"——系统刻画纯数据驱动方法在该问题上的能力上限和失效模式。待文献读完 + 导师确认后再定最终定位。

## 项目路径
- 代码根目录：`PatchTST_supervised/`
- **主训练入口（新实验必须用）**：`run_longExp.py`（原管线，运行时需 `PYTHONPATH=PatchTST_supervised`）
- 快速校准工具（仅 EXP-13/14/15）：根目录 `run_sunspot_fixed.py`
- 命令行版：`PatchTST_supervised/run_longExp.py`
- 数据：`PatchTST_supervised/dataset/`
- 实验分支：`level3-residual-prediction`（M4 + PatchTST 残差预测，已完成，结论：不可行）
- EXP-16 分支：`EXP-16` worktree（`cd ../patchTST-sunspot-prediction-EXP16`）
- 日记：`diary/`
- M4 Waldmeier 师兄代码：`PatchTST_supervised/shixiong_m4/coding/solar_cycle_optimization_framework_package(1)/`

## 目录约定
- 新的实验另起 worktree 分支，命名 `phase-training-vN`，不污染已验证分支
- 分阶段划分脚本统一放 `phase_analysis/`，训练脚本放 `PatchTST_supervised/`
- 训练输出（checkpoint、日志、图表）统一放 `phase_training/output/`
- 不要提交 `__pycache__/`、`.pth`、`.pyc`

## 技术栈
- Python 3.10 + PyTorch 1.11
- 数据处理：numpy, pandas, scipy, scikit-learn
- 模型：PatchTST（Patch Time Series Transformer），源代码位于 `layers/`、`models/`
- 训练框架：`exp/exp_main.py`（继承自 `exp_basic.py`）

## 关键参数约定
- **对标基线**：任何实验必须与基线使用相同参数配置，否则不可比：
  - seq_len=96, pred_len=24, d_model=128, n_heads=8, e_layers=2, d_ff=2048
  - patch_len=16, stride=8, RevIN=1, StandardScaler, MSE loss
  - batch_size=16, train_epochs=10
  - （旧基线 d_model=512，2026-06-14 验证 d_model=128 泛化更好，全面优于 512）
- **⚠️ 2026-08-14 阶段1基线更新**：EXP-17 系列起使用 seq_len=336, train_epochs=50, patience=100, features=MS, W1回测窗口（test_start=1996-08, test_end=2008-11, num_train=2838），且**口径=最佳val模型（命令链 train&&补测，见 SOP §4）**。0b基线全步MAE=23.87。
- ⚠️ **五月 Baseline B（1867+ 数据，step0 MAE=13.02）与当前基线 EXP-14（1749+ 数据，step0 MAE=9.08）不可直接比较：** 两者数据范围（1867+ vs 1749+）、seq_len（132 vs 96）、n_heads（16 vs 8）、e_layers（3 vs 2）、d_ff（256 vs 2048）、patch_len（12 vs 16）、stride（6 vs 8）、dropout（0.05 vs 0.2）、epochs（50 vs 10）共 9 项参数不同。step0 的 13.02→9.08 说明 EXP-14 在自身配置下单步更优，但不可归因到任何一个具体参数变更，也不是严格意义上的"进步"。
- 分阶段训练允许缩小模型（d_model 128~256）以适应阶段数据量
- 所有修改需在独立 worktree 上进行，主分支只保留稳定版

## 完整训练 baseline（2026-06-14 完成 d_model=512 vs 128 对比）
- 全量数据（1749-2025，3321 月），10 epochs，参数全同仅 d_model 不同
- d_model=512：MSE(z)=0.085, MAE(z)=0.141, RSE=0.316, MAE物理=25.27, RMSE物理=34.41, R²=0.539
- **d_model=128 (EXP-14，当前主基线)**：MSE(z)=0.079, MAE(z)=0.125, RSE=0.304, MAE物理=23.87（全47步平均）, RMSE物理=33.29, R²=0.568
  - step 0 单次 2 年预测物理 MAE=9.08（优于全步平均，说明模型在多窗口滑动评估中退化严重）
  - 峰值区域（SSN>150）误差均值 68.8（全步平均），MSE loss 压制峰值问题未因降参解决
- 基线数据来源：`result.txt` line 64-68，npy 反算验证
- 完整实验时间线见 `project_docs/experiment_history.md`

## 探索期结论（2026-08-15，Stage 9，判定标准待景修批注）

> 阶段1 之后的自主探索：数据表示（变换）/训练目标（loss）/使用策略（block）三系列 14 个实验。
> ⚠️ 以下"最佳"为 AI 暂定，判定标准批注前不作正式结论；全部单种子 seed=2021。

- **机制4（新识破）**：右偏分布是滚动误差爆炸的主要放大器——所有目标变换让滚动 MAE 改善 45-56%（基线 50.41 → pow23 22.13 / log1p 22.63）
- **机制3 证伪**：RevIN 不是峰值压制元凶，是滚动稳定性支柱（revin=0 滚动崩溃 1967 万 MAE）
- **loss 最佳候选 wmse_th**（阈值加权，tau=1.5 即 SSN>185）：滚动峰 -6.2（改善 77%）、滚动 33.92（改善 33%）、低值损伤最小；α=1.0 的线性 wmse 峰值更好（-2.2）但全步恶化 14.6%
- **外推策略最佳候选 block24 无重叠**：三窗口优于滚动 44-59%（W1 21.29/W2 27.28/W3 23.75）；重叠 12 月反而更差
- **口径分裂**：全步（125 独立窗口）与滚动/block（自回归轨迹）结论常相反；阶段3 是外推场景，以滚动/block 口径为主
- pred_len=48 是滚动禁区（step0 退化致滚动崩溃）；pl=24 维持
- 组合实验待判定批注后：C-1 pow23+wmse_th、C-2 block+wmse_th
- 线 B 完成：phys_params_survey（Xiong四参数+aa）、M4 同口径对拍（⚠️ 结论 AI 判断，景修持保留）、discussion_agenda

## 分阶段训练已确认的问题
1. d_model=512 对每阶段 ~200 样本严重过拟合 → 下次降低模型参数
2. train/val/test 切分有数据泄漏 → 需修 `Dataset_Phase._read_data()`
3. 四阶段 Scaler 不一致导致预测不可用 → 统一 Scaler，存为文件
4. MSE loss 压制峰值 → 改用 HuberLoss（EXP-9 验证：⚠️ 该实验在 MS mode 下进行，Huber 在 M mode 下尚未验证；此结论暂限 MS mode）
5. EarlyStopping 在验证集太小（13 样本）时不可靠 → 关掉 patience
6. `enc_in` 参数需与数据列数一致（固定 3：month_sin, month_cos, ssn）
7. d_model=128 下峰值误差分层确认：SSN 0-50 误差 6.6、50-100 误差 12.4、100-150 误差 27.1、>150 误差 68.8。降参不解决峰值压制，需改任务定义（预测残差）

## Level 3 实验结论（2026-07-13，worktree: `level3-residual-prediction`）
- M4 Waldmeier Gamma 曲线对 Cycle 25 测试段的预报 MAE = 3.32（最好成绩，远优于 PatchTST 基线 MAE=23.87）
- Level 3 残差预测失败：best-fit 包络的残差（ssn_smooth - M4_bestfit）在训练集上 std=10.16，但无可学习的预测信号——每个周期的 Gamma 偏差是独有的、不重复的结构。PatchTST 学到的是输出均值≈0，叠加到 M4 预报上只会加噪声（MAE 从 3.32 退化到 4.48）
- 根本原因：best-fit 残差 ≠ 预报残差。训练用 best-fit 残差（零均值小噪声），测试用预报残差（含 M4 的系统性预测偏差），两者分布和结构不同
- M4 包络生成与残差管道代码位于 worktree 分支 `level3-residual-prediction`，文件：`PatchTST_supervised/prepare_level3_residual.py`（M4 校准+预报+残差计算）、`PatchTST_supervised/eval_level3_residual.py`（评估）

## 天花板探测实验（2026-07-17，worktree: `ceiling-probe-v1`）

五组实验在测试集（70 月，2020-01~2025-10，全 47 窗口评估）上的结果。

⚠️ **PatchTST 组（A-C）和 DLinear 组（D1-D2）使用了不同的归一化方式（RevIN=1 vs RevIN=0）、不同的学习率（0.0001 vs 0.005）、不同的训练轮数，不能直接跨组对比数值。** 各组内部可比。

| 实验 | 模型 | seq_len | RevIN | lr | epoch | step 0 物理 MAE | 全步 物理 MAE | R² |
|------|------|---------|-------|-----|-------|----------------|-------------|-----|
| A | PatchTST | 96 | 1 | 0.0001 | 10 | 9.08 | 23.87 | 0.568 |
| B | PatchTST | 192 | 1 | 0.0001 | 30 | 12.30 | 22.02 | 0.625 |
| C | PatchTST | 336 | 1 | 0.0001 | 30 | 10.87 | 20.54 | 0.692 |
| D1 | DLinear | 96 | 0 | 0.005 | 30 | 19.04 | 20.31 | 0.722 |
| D2 | DLinear-I | 96 | 0 | 0.005 | 30 | 19.36 | 19.30 | 0.751 |

**各组内部结论：**
- PatchTST 组（A/B/C）：seq_len 96→336，全步 MAE 从 23.87→20.54。更多上下文有帮助但边际递减。step 0 最优为 sl96 (MAE=9.08)。
- DLinear 组（D1/D2）：individual=1 比 individual=0 更好（19.30 vs 20.31）。

**跨组注意事项：**
- PatchTST sl96 step 0 (MAE=9.08) 是全五组中最好的单次 2 年预测结果
- DLinear-I 全步平均 (MAE=19.30) 是全五组中最稳定的多窗口平均结果
- 所有模型 peak 区域误差 50+ SSN
- M4 物理方法 Cycle 25 MAE=3.32 仍是强基准
- 全部数值经 npy 反算验证，数据来源和反算代码见 `project_docs/experiment_history.md`

**之前 AGENTS.md 和 project_summary 中记录的「硬天花板 MAE≈19.30」是基于跨组比较得出的，可比性前提不成立，已废弃。**

## 下一步（2026-08-15 更新）

1. 景修批注判定标准 6 项（`judgment_criteria_discussion_2026-08-15.md`）
2. 批注后：回填 EXP-20~23 实验结论 → 组合实验 C-1/C-2（需景修重新预先批准，属新队列）
3. 极小值专项评估（阶段3 前置）、优胜配置 W2/W3 验证
4. 开学讨论：学姐（Xiong 数据）/ 学长（M4 对质 + block 策略）/ 导师（物理参数数据集方向）
5. 阶段2：数据更新 2026.07 → 全量重训（配置待判定+组合实验后冻结）
6. 阶段3：滚动/block 外推 25 周期极小值 + 26 周期完整曲线

纪律：预检表跑前完成（事后补写不算数）；坑清单 1-28 跑前回顾；主控文档顶部快照随进度同步更新。路线图见 `project_docs/project_roadmap.md`。
