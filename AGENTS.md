# Level 3 残差预测实验 — 快照分支

> ⚠️ **此分支为 2026-07-13 实验快照，不再维护。**
> 最新项目结论、参数约定、实验历史以主项目文档为准：
> - 规范与基线：`/root/code/patchTST-sunspot-prediction/AGENTS.md`（v2, 2026-08-09）
> - 实验历史：`/root/code/patchTST-sunspot-prediction/experiment_history.md`

## Level 3 最终结论（2026-07-13）

**不可行。**

- M4 Waldmeier Gamma 曲线 Cycle 25 预报 MAE = 3.32
- M4 + PatchTST 残差预测 MAE = 4.48（**退化**）
- 根因：best-fit 残差（训练集 std=10.16，零均值无预测信号）≠ 预报残差（含 M4 系统性预测偏差），两者分布不同。PatchTST 学到输出≈0，叠加到 M4 预报上只加噪声。

## 本分支关键文件

- `PatchTST_supervised/prepare_level3_residual.py` — M4 校准 + 预报 + 残差生成
- `PatchTST_supervised/eval_level3_residual.py` — 残差预测评估
- `run_sunspot_residual_fullfit.py` — 残差训练入口
