# PatchTST 太阳黑子预测 项目 — AGENTS.md v1

## 项目目标
使用 PatchTST 模型预测太阳黑子数（SSN），核心目标是预测第 25、26 太阳活动周的完整变化曲线。当前路线：M4 Waldmeier 参数化曲线（shixiong_m4/coding/）做周期包络预测，PatchTST 辅助修正。Level 3 残差预测方案已验证不可行（见「Level 3 实验结论」），下一步考虑 Level 2（M4 作为 PatchTST 输入特征）。

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
6. `enc_in` 参数需与数据列数一致（固定 3：month_sin, month_cos, ssn）
7. d_model=128 下峰值误差分层确认：SSN 0-50 误差 6.6、50-100 误差 12.4、100-150 误差 27.1、>150 误差 68.8。降参不解决峰值压制，需改任务定义（预测残差）

## Level 3 实验结论（2026-07-13，worktree: `level3-residual-prediction`）
- M4 Waldmeier Gamma 曲线对 Cycle 25 测试段的预报 MAE = 3.32（最好成绩，远优于 PatchTST 基线 MAE=23.87）
- Level 3 残差预测失败：best-fit 包络的残差（ssn_smooth - M4_bestfit）在训练集上 std=10.16，但无可学习的预测信号——每个周期的 Gamma 偏差是独有的、不重复的结构。PatchTST 学到的是输出均值≈0，叠加到 M4 预报上只会加噪声（MAE 从 3.32 退化到 4.48）
- 根本原因：best-fit 残差 ≠ 预报残差。训练用 best-fit 残差（零均值小噪声），测试用预报残差（含 M4 的系统性预测偏差），两者分布和结构不同
- M4 包络生成与残差管道代码位于 worktree 分支 `level3-residual-prediction`，文件：`PatchTST_supervised/prepare_level3_residual.py`（M4 校准+预报+残差计算）、`PatchTST_supervised/eval_level3_residual.py`（评估）

## 下一步（2026-07-13 更新）
- M4 预报表现已确认优秀，可作为 Cycle 25/26 预测的直接方案
- 两条潜在路线：
  1. Level 2：将 M4 包络值作为 PatchTST 额外输入特征（enc_in=4），不改变预测目标（仍为 SSN）
  2. 直接用 M4 + ±18% 不确定度带，不做 PatchTST 修正
 - 分阶段训练的六个问题（L44-51）均未解决，优先级低于 M4 路线

## 天花板探测实验（2026-07-17，worktree: `ceiling-probe-v1`）

### 目的
探测纯数据驱动方法在太阳黑子预测中的性能上限：
1. 信息量饱和测试：seq_len 96→192→336（数据里的有用信号是否已耗尽？）
2. 模型复杂度测试：DLinear vs DLinear-I vs PatchTST（Transformer 架构是否有真实贡献？）

### 实验矩阵
全部统一: seed=2021, bs=16, epoch=30, patience=5, MSE loss, 同一份数据/切分

| 实验 | 模型 | seq_len | 特殊参数 | 
|------|------|---------|----------|
| A | PatchTST | 96 | dm128, patch16 (EXP-14 基线, 复用结果) |
| B | PatchTST | 192 | 同 A |
| C | PatchTST | 336 | 同 A |
| D1 | DLinear | 96 | individual=0, lr=0.005 |
| D2 | DLinear-I | 96 | individual=1, lr=0.005 |

### 已完成结果 (2026-07-17)
| 实验 | MAE(物理) | R² | >150 MAE | 峰值 bias | 判断 |
|------|----------|-----|----------|-----------|------|
| A (PatchTST sl96) | 23.87 | 0.568 | 68.4 | -75.8 | 基线 |
| B (PatchTST sl192) | 22.02 | 0.625 | 64.7 | -85.2 | 历史有微量帮助 |
| C (PatchTST sl336) | 20.54 | 0.692 | 55.7 | -66.9 | 历史边际递减 |
| D1 (DLinear ind0) | 20.31 | 0.722 | 47.2 | -56.5 | 纯线性>Transformer |
| D2 (DLinear-I ind1) | **19.30** | **0.751** | **42.5** | **-50.3** | **当前最优** |

### 核心结论
1. **信息量未完全饱和**: seq96→192→336, MAE 持续下降 (23.87→22.02→20.54), 但边际递减 (7.7%→6.7%)
2. **Transformer 架构在此问题上没有增益**: DLinear-I (纯线性, MAE=19.30) 全面超过 PatchTST sl336 (MAE=20.54)。注意力机制+128 维嵌入+a 打不过一层线性映射。
3. **纯数据驱动的硬天花板**: 最优数据驱动方法 MAE=19.30。物理方法 M4 (MAE=3.32) 好 6 倍。这个差距不太可能靠加数据或加模型弥合——物理先验才是关键瓶颈。
4. **峰值压制是系统性偏置**: 所有模型（包括 DLinear）都对峰值预测偏低 50+ SSN。这是 MSE loss + 数据分布的固有性质，不是模型选择问题。

### 运行命令
```bash
cd /root/code/patchTST-sunspot-prediction-ceiling-probe
python3 -u run_dlinear.py      # DLinear (individual=0)
python3 -u run_dlinear_i.py    # DLinear-I (individual=1)
python3 -u run_seq192.py       # PatchTST seq_len=192
python3 -u run_seq336.py       # PatchTST seq_len=336
python3 -u compare_all.py      # 出对比图表
