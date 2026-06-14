# PatchTST 太阳黑子预测 项目 — AGENTS.md v1

## 项目目标
使用 PatchTST 模型预测太阳黑子数（SSN），核心目标是预测第 25、26 太阳活动周的完整变化曲线。当前路线：M4 Waldmeier 参数化曲线（shixiong_m4/coding/）做周期包络（战略层） + PatchTST 做残差修正（战术层），双线协同。

## 项目路径
- 代码根目录：`PatchTST_supervised/`
- 主训练入口：
  - **完整训练**：根目录 `run_sunspot_fixed.py`（train_epochs=10，完整基线参数）
  - 快速测试：`PatchTST_supervised/run_sunspot_fixed.py`（train_epochs=1）
  - 命令行版：`PatchTST_supervised/run_longExp.py`
- 数据：`PatchTST_supervised/dataset/`
- 分阶段相关（均在 worktree 分支，不在 main）：`dataset_phase/`、`phase_analysis/`、`experiments/`、`phase_training/output/`
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
- 分阶段训练允许缩小模型（d_model 128~256）以适应阶段数据量
- 所有修改需在独立 worktree 上进行，主分支只保留稳定版

## 完整训练 baseline（2026-06-14 完成 d_model=512 vs 128 对比）
- 全量数据（3321 月，1867-2025），10 epochs，参数全同仅 d_model 不同
- d_model=512：MSE(z)=0.085, MAE(z)=0.141, RSE=0.316, MAE物理=25.27, RMSE物理=34.41, R²=0.539
- d_model=128：MSE(z)=0.079, MAE(z)=0.125, RSE=0.304, MAE物理=23.87, RMSE物理=33.29, R²=0.568
- d_model=128 全面优于 512。但峰值区域（SSN>150）误差依然严重（均值 68.8），MSE loss 压制峰值问题未因降参解决

## 分阶段训练已确认的问题
1. d_model=512 对每阶段 ~200 样本严重过拟合 → 下次降低模型参数
2. train/val/test 切分有数据泄漏 → 需修 `Dataset_Phase._read_data()`
3. 四阶段 Scaler 不一致导致预测不可用 → 统一 Scaler，存为文件
4. MSE loss 压制峰值 → 改用 HuberLoss
5. EarlyStopping 在验证集太小（13 样本）时不可靠 → 关掉 patience
6. `enc_in` 参数需与数据列数一致（目前为 3：month_sin, month_cos, ssn）
7. d_model=128 下峰值误差分层确认：SSN 0-50 误差 6.6、50-100 误差 12.4、100-150 误差 27.1、>150 误差 68.8。降参不解决峰值压制，需改任务定义（预测残差）

## 下一步
- 跑通 M4 代码（shixiong_m4/coding/），生成历史每月的 M4 包络值
- PatchTST 改为预测残差（SSN - M4_envelope），任务从回归 SSN 转回归残差
- Cycle 24/25 留出验证：M4 alone vs M4+PatchTST 对比
- Cycle 26 三情景：M4 出弱/中/强包络，PatchTST 叠加残差预测
