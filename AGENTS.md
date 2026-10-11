# PatchTST 太阳黑子预测 项目 — AGENTS.md v2

## 项目主控文档
**`project_docs/project_roadmap.md`** — 所有阶段、任务、待办集合在此。本文件为项目规范和技术参考。

## ⚠️ 主线定位（2026-10-11）

**`EXP-17-stage1`（本分支）= 项目唯一主线**，由 2026-10-11 的收敛合并产生：
把 `env-fix-torch28-numpy2` / `env-newserver-2026-10` / `GPU-exp25` / `GPU-exp26` / `GPU-dropout-abl` /
`GPU-basemodel-search` 六条线的内容**全部并入**（`git merge-base --is-ancestor` 六条全 PASS）。

- **状态与账本只在主线写、只在主线读**（见下方「分支与账本规范」）
- 分支全貌与收敛记录：`project_docs/BRANCHES.md`；当前状态：`project_docs/STATE.md`
- `main` = **临时遗留标签**（2026-08-10 冻结），景修 2026-10-11 决定暂不改名、先不管
- 当前阶段：GPU 新环境重建基线 + 调参中（本轮只动模型配置类变量）

## 分支与账本规范（方案 C，2026-10-11 景修批准）

**为什么有这条**：2026-10-10~11 攒出 6 条并行线，导致「状态文档在每条线各一份、只有一处更新」→ 其余副本过期但看起来是真的；
且多条线同时往 `result.txt` / 索引表追加 → 每次合并必冲突。**根因不是"分了线"，是"分错了东西"。**

| 东西 | 是否按分支隔离 | 理由 |
|---|---|---|
| 代码 / 实验产物（config / log / npy / 记录文件 / 图） | ✅ 隔离 | 实验可能失败或被否，不该污染已定论的线 |
| **状态文档**（`STATE.md` / 本文件 / `project_roadmap.md` / `experiment_SOP.md`） | ❌ **不隔离** | 一份就够；多份必然漂移 |
| **账本**（`result.txt` / `experiment_history.md` 索引表 / `diary/`） | ❌ **不隔离** | 追加式单文件，多线同时写必然冲突 |

**四条铁律**

1. **状态只在主线写、只在主线读。** 实验分支里的状态文档副本一律视为「过期副本」，不作为依据。
2. **分支短命**：一次实验 = 一条分支，**做完当天或次日合回主线或明确丢弃**；同时在飞的分支 **≤3 条**。
3. **开分支必须当天登记** `project_docs/BRANCHES.md`（目的 / 开分支日期 / 计划合并日期）。
   → 已机器兜底：`scripts/check_docs.py` 的 **D 查**会对"本地存在但未登记的分支"报警。
4. **账本行在合并时统一处理**：追加式文件冲突按「两边都保留、按日期排序」解（两分钟机械操作，不是事故）。

**什么时候值得开分支**

| 场景 | 开分支？ | 形态 |
|---|---|---|
| 调参 / 找基线 | 是 | 工作线（当前 = `GPU-basemodel-search`） |
| 训练口径改变（如平滑口径） | 是 | **平行线**：有自己的账本，数值与月均线不可比，不混进同一张表 |
| 探索新方法（相位 / 非负 / 谷时分类 / scheduled sampling 等） | 是 | 短命方法分支，失败即丢，成功才合 |
| 纯文档修订 | 否 | 直接在主线改 |

> **备选增强（方案 D，未采纳）**：把 `result.txt` 改成"一实验一文件 + 脚本汇总"，可彻底消灭账本冲突。
> 若之后"合并撞账本"仍频繁发生，再启动。

> **新来的 AI？** 先读 `project_docs/ONBOARDING.md`。3 分钟了解她是谁、项目做什么、今天踩了什么坑、怎么跟她沟通。

## 实验记录约定（2026-08-10 新增）
- **操作规程**：`project_docs/experiment_SOP.md`，跑任何实验前对照检查（全流程版 / 批量版）
- **快速索引**：`project_docs/experiment_history.md` 顶部索引表，Ctrl+F 查找实验
- **记录模板**：`project_docs/experiment_template.md`，每次跑新实验前复制、填写
- **数据管线**：`project_docs/data_pipeline.md`（归一化方法/数据流/硬编码约束）
- **工具回归测试**：`scripts/test_eval.py`（改 eval/roll_eval 后跑一次确认没坏）
- **跑实验后四件事**：① 追加一行到 `result.txt` ② 追加到索引表 ③ **commit 前核对未跟踪文件**
  （`git status --untracked-files=all` 逐条归类：`logs/EXP-*.log`、npy、config、图=证据必须提交；`*.pid`、`__pycache__`、`.pth`=ignore）④ commit 推送
  —— 2026-10-11 加第③件：EXP-27~30 的四个原始训练日志曾漏提交（SOP 坑 35）
- **硬规则**：一次只改一个变量，每次标明对照基线是谁

## 文献笔记

14 篇精读结构化笔记，位于 `literature/`：
- **总览表 + 横向对比**：`literature/literature_reading_notes.md`（473 行）
  — 每篇一句话结论 + 与你项目的关系
- **每篇完整详注**：`literature/literature_reading_notes_detailed.md`（1214 行）
  — 三句话笔记、实验设计细节、关键数值、方法论判断
- 涉及方向：DL+sunspot 预测、物理综述（Hathaway/Petrovay）、GRC 残差校正、PatchTST/DLinear 原文、ARIMA 中文等
- 新实验前 Ctrl+F 搜索关键词 → 找到对应的详注 → 了解别人做到什么精度、踩了什么坑

## 当前阶段：GPU 新环境重建基线 + 调参中（2026-10-11）

- 阶段1（参数搜索+三窗口验证）✅、探索期（变换/loss/外推策略三系列 14 实验）✅、线B文献准备 ✅ 完成；
  **判定标准 6 项仍待景修批注**（`project_docs/judgment_criteria_discussion_2026-08-15.md`），批注后才启动组合实验 C-1/C-2
- 2026-10-10 换新云服务器（A6000 / torch 2.8），可比性断裂 → 新环境重跑 EXP-25~30：
  EXP-25 作废（误用旧 ARGS）、EXP-26 复现 50.41（差 0.24%，证明段长是"30.32 vs 50.41"的全部原因）、
  EXP-27~30 得三条修正（口径分裂只在 2/3 窗口成立；**do0.05 的滚动是高方差不是更好**；撤回"环境差异≤1%"）
- **当前基线（GPU 环境）**＝EXP-18-5b 的配置在新环境的复核结果 **EXP-26**（W1 全步 22.87 / step0 11.78 / 滚动 50.29，148 月）
- **用法口径＝block24 无重叠**（判据=minimax"最不易翻车"，EXP-23-R3；不是因为它平均最好）
- **当前任务＝找 GPU 基线（调参）**；**新方法一律冻结**，想法队列见 `project_docs/plan_preregistration_2026-10-11.md`
  （`GPU-dropout-abl` 分支）；还没扫过的调参维度：`learning_rate` / `batch_size` / `label_len` / epochs-patience
- 跨环境比较：只用**全步**（≤4%）与 **block24**（≤10.6%）作辅证；**滚动数只能同环境自比**（≤28%）
- 数据泄露：三种口径已核查**无泄露**，见 `project_docs/data_pipeline.md`「数据泄露核查」节
- 未决：判定标准批注、文档对齐（A/B 类已做，C 类待定）、推送与合并（对齐后再合，目标 `EXP-17-stage1`）
- 详见 `project_docs/project_roadmap.md`、`project_docs/exploration_analysis.md`

## 环境维护

### 运行环境（2026-10-10 换新云服务器，景修确认）

| 项目 | 当前值 |
|------|--------|
| 机器 | 新云服务器；GPU = NVIDIA RTX A6000（48 GB），CUDA 可用 |
| Python | 3.10.18（`/usr/local/bin/python3`）。**无 venv/conda 虚拟环境**，直接用系统解释器 |
| PyTorch | 2.8.0+cu128（旧环境：1.11，纯 CPU。换机后不再适用） |
| 其他依赖 | numpy 2.2.6、pandas 2.3.2、scipy 1.15.3、scikit-learn 1.7.2、matplotlib 3.10.6、tqdm |
| 依赖缺口 | `einops`/`pyraformer`/`tvm`/`fbm`（Pyraformer、FEDformer 对照基线用）、`pmdarima`（`models/Stat_models.py`，该模块未被主路径引用）、`ptflops`（`utils/tools.py` 内惰性调用）。**主训练路径 `run_longExp.py` 不需要这些** |
| ⚠️ 可比性断裂 | torch 2.8 vs 1.11、GPU vs CPU 同时变了。**在本机重跑得到的数值不能直接与 `result.txt` 的历史基线对比**，需按 SOP 重新建立基线 |
| ⚠️ 全局环境污染 | 系统解释器同时承载 pdf/docx skill 等工具的依赖。**装任何新包前先确认它不会升级 numpy / torch**，否则会静默改变实验环境 |
| torch.load 兼容 | torch ≥2.6 默认 `weights_only=True`。部分脚本已改（`check_pred.py`、`manual_inference.py` 等），`roll_eval.py:85` 和 `Formers/*/exp/exp_main.py` 仍是默认值，可能加载失败 |

### Git / GitHub

| 项目 | 状态 |
|------|------|
| 仓库位置 | `/output/patchTST-sunspot-prediction`（持久卷），当前分支 `EXP-17-stage1` |
| Git credential | `/openbayes/home/.git-credentials`（权限 600），helper = `store --file=...`。2026-10-10 重新生成，有效期至 **2027-01-08** |
| ⚠️ 持久化陷阱 | `/root` 是容器临时层，重启即丢。真实配置在 `/openbayes/home/.gitconfig`，`/root/.gitconfig` 只是由平台生成、通过 `include` 引入。**重启后若 git 不认身份，跑一次 `git config --global include.path /openbayes/home/.gitconfig`**；仓库本地 `.git/config` 已冗余配置一份，不受影响 |
| Token 获取 | https://github.com/settings/tokens → Generate new token (classic) → 勾选 `repo`（只读 token 能 clone 不能 push） |

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
- **分支纪律见上方「分支与账本规范（方案 C）」**（旧规矩"新实验一律另起 worktree 分支"已作废——
  它只负责隔离，没规定回收，是 2026-10-10 攒出 6 条并行线的直接原因）
- 分阶段划分脚本统一放 `phase_analysis/`，训练脚本放 `PatchTST_supervised/`
- 训练输出（checkpoint、日志、图表）统一放 `phase_training/output/`
- 不要提交 `__pycache__/`、`.pth`、`.pyc`

## 技术栈
- Python 3.10.18 + PyTorch 2.8.0+cu128（当前云服务器实况，详见上方「运行环境」。历史实验是在 Python 3.10 + PyTorch 1.11 纯 CPU 上跑的）
- 数据处理：numpy, pandas, scipy, scikit-learn
- 模型：PatchTST（Patch Time Series Transformer），源代码位于 `layers/`、`models/`
- 训练框架：`exp/exp_main.py`（继承自 `exp_basic.py`）

## 关键参数约定
- **数据口径警示（2026-08-20）**：ssn 列为 **SILSO 月均 SSN（未平滑）**。阶段3 若引用官方周期界（平滑极小值/极大值）需 13 月平滑转换。**Q9 待议**：训练数据是否改用 13 月平滑（景修倾向应该用）——待与学姐学长/导师讨论。谷峰数字引用均为月均口径。版本锚点=V2（2014/2024 已对拍），1778-05=398.2 版本归属待官网核对。
- **对标基线**：任何实验必须与基线使用相同参数配置，否则不可比：
  - seq_len=96, pred_len=24, d_model=128, n_heads=8, e_layers=2, d_ff=2048
  - patch_len=16, stride=8, RevIN=1, StandardScaler, MSE loss
  - batch_size=16, train_epochs=10
  - （旧基线 d_model=512，2026-06-14 验证 d_model=128 泛化更好，全面优于 512）
- **⚠️ 2026-08-14 阶段1基线更新**：EXP-17 系列起使用 seq_len=336, train_epochs=50, patience=100, features=MS, W1回测窗口（test_start=1996-08, test_end=2008-11, num_train=2838），且**口径=最佳val模型（命令链 train&&补测，见 SOP §4）**。0b基线全步MAE=23.87。
- **⚠️ 当前基线：见 `project_docs/STATE.md`（唯一事实源）。**（阶段1 定论配置为 `EXP-18-5b`：在上述配置上把 `dropout`/`fc_dropout` 由 0.05 改为 **0.2**；W1 全步 MAE 22.62、step0 11.01、R² 0.789、E_r −24.4、滚动 50.41（n_roll=148）；GPU 环境复核 = EXP-26。）
  **本文件不复述基线定义**——凡"基线"必须写实验 ID（坑 34），定义只在 STATE.md 维护。
  - ⚠️ **SOP §4 那段"标准 ARGS"曾长期写着 `dropout 0.05`（Round 5 之前的 0b 配置）**，2026-10-10 已在 SOP 改为 0.2。教训：**凡"基线"描述必须写明是哪个实验 ID**，不写 ID 的一律视为不可信（见 SOP 坑 34）。
  - ⚠️ **滚动口径必须连段长一起记**：W1 整段是 148 月（不是 70）。同一 checkpoint 在 70 月口径下是 30.32、148 月口径下是 41.63，差 40%。见 SOP 坑 32。
- ⚠️ **五月 Baseline B（1867+ 数据，step0 MAE=13.02）与当前基线 EXP-14（1749+ 数据，step0 MAE=9.08）不可直接比较：** 两者数据范围（1867+ vs 1749+）、seq_len（132 vs 96）、n_heads（16 vs 8）、e_layers（3 vs 2）、d_ff（256 vs 2048）、patch_len（12 vs 16）、stride（6 vs 8）、dropout（0.05 vs 0.2）、epochs（50 vs 10）共 9 项参数不同。step0 的 13.02→9.08 说明 EXP-14 在自身配置下单步更优，但不可归因到任何一个具体参数变更，也不是严格意义上的"进步"。
- 分阶段训练允许缩小模型（d_model 128~256）以适应阶段数据量
- **分支纪律见上方「分支与账本规范（方案 C）」**（旧句"所有修改需在独立 worktree 上进行"已作废：
  它等于要求每条改动都开线，与"分支短命、账本不隔离"直接冲突）

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
