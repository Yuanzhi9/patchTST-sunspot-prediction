# PatchTST 太阳黑子预测 项目 — AGENTS.md v1

## 项目目标
使用 PatchTST 模型预测太阳黑子数（SSN），核心目标是预测第 25、26 太阳活动周的完整变化曲线。当前阶段采用按物理阶段（上升/峰值/下降/低谷）分拆训练的新思路。

## 项目路径
- 代码根目录：`PatchTST_supervised/`
- 主训练入口：`run_longExp.py`
- 数据：`PatchTST_supervised/dataset/`（原始）、`dataset_phase/`（分阶段）
- 阶段划分：`phase_analysis/`
- 实验记录：`experiments/`（最终版）、`phase_training/output/`（训练日志）
- 日记：`diary/`

## 目录约定
- 新的实验另起 worktree 分支，命名 `phase-training-vN`，不污染已验证分支
- 分阶段划分脚本统一放 `phase_analysis/`，训练脚本放 `PatchTST_supervised/`
- 训练输出（checkpoint、日志、图表）统一放 `phase_training/output/`
- 不要提交 `__pycache__/`、`.pth`、`.pyc`

## 技术栈
- Python 3.10 + PyTorch 2.x
- 数据处理：numpy, pandas, scipy, scikit-learn
- 模型：PatchTST（Patch Time Series Transformer），源代码位于 `layers/`、`models/`
- 训练框架：`exp/exp_main.py`（继承自 `exp_basic.py`）

## 关键参数约定
- **对标基线**：任何分阶段实验必须与基线使用相同参数配置，否则不可比：
  - seq_len=96, pred_len=24, d_model=512, n_heads=8, e_layers=2, d_ff=2048
  - patch_len=16, stride=8, RevIN=1, StandardScaler, MSE loss
  - batch_size=16, train_epochs=10
- 分阶段训练允许缩小模型（d_model 128~256）以适应阶段数据量
- 所有修改需在独立 worktree 上进行，主分支只保留稳定版

## 已确认的问题
1. d_model=512 对每阶段 ~200 样本严重过拟合 → 下次降低模型参数
2. train/val/test 切分有数据泄漏 → 需修 `Dataset_Phase._read_data()`
3. 四阶段 Scaler 不一致导致预测不可用 → 统一 Scaler，存为文件
4. MSE loss 压制峰值 → 改用 HuberLoss
5. EarlyStopping 在验证集太小（13 样本）时不可靠 → 关掉 patience
6. `enc_in` 参数需与数据列数一致（目前为 3：month_sin, month_cos, ssn）

## 下一步
- `phase-training-v3`：用修正方案（d_model=128, d_ff=256, dropout=0.3, Huber loss, 统一 Scaler, 无数据泄漏）重新训练
- 在同一台机器上用相同参数跑全量数据 baseline，拿到可比的对照 MAE
- 实现不依赖自回归的阶段衔接预测方法
