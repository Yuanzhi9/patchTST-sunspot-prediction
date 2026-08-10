# PatchTST 太阳黑子预测 项目 — AGENTS.md v2

## 项目主控文档
**`project_roadmap.md`** — 所有阶段、任务、待办集合在此。本文件为项目规范和技术参考。

## 当前阶段：Phase 0 — 文献奠基
正在系统性阅读太阳黑子周期物理基础和 DL+sunspot 预测文献。完成后再进入 Phase 1（项目自审）和 Phase 2（实验设计）。详见 `project_roadmap.md`。

## 项目目标（当前定位）
使用深度学习模型探索太阳黑子数（SSN）预测的可行性边界，核心关注第 25、26 太阳活动周的完整曲线预测。M4 Waldmeier 参数化曲线（师兄代码）做物理方法对照基线。当前定位偏向"可行性边界研究"——系统刻画纯数据驱动方法在该问题上的能力上限和失效模式。待文献读完 + 导师确认后再定最终定位。

## 项目路径
- 代码根目录：`PatchTST_supervised/`
- 主训练入口：
  - **完整训练**：根目录 `run_sunspot_fixed.py`（train_epochs=10，完整基线参数）
  - 快速测试：`PatchTST_supervised/run_sunspot_fixed.py`（train_epochs=1）
  - 命令行版：`PatchTST_supervised/run_longExp.py`
- 数据：`PatchTST_supervised/dataset/`
- 实验分支：`level3-residual-prediction`（M4 + PatchTST 残差预测，已完成，结论：不可行）
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
- 完整实验时间线见 `experiment_history.md`

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
- 全部数值经 npy 反算验证，数据来源和反算代码见 `experiment_history.md`

**之前 AGENTS.md 和 project_summary 中记录的「硬天花板 MAE≈19.30」是基于跨组比较得出的，可比性前提不成立，已废弃。**

## 下一步（2026-07-23 更新 → 2026-08-09 修正）
当前处于 Phase 0（文献阅读）。先读文献 → 自审项目 → 定义科学问题 → 找导师 → 再定实验路线。**不跑新实验。** 路线图见 `project_roadmap.md`。完整实验时间线见 `experiment_history.md`。
