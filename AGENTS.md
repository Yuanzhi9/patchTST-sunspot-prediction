# EXP-16 实验分支 — 快照

> ⚠️ **此分支为 EXP-16/16b/16c 实验专用快照，不再维护。**
> 最新项目结论、参数约定、实验历史、操作规程以主项目文档为准：
> - 规范与基线：`/root/code/patchTST-sunspot-prediction/AGENTS.md`（v2, 2026-08-10）
> - 实验历史：`/root/code/patchTST-sunspot-prediction/project_docs/experiment_history.md`
> - 操作规程：`/root/code/patchTST-sunspot-prediction/project_docs/experiment_SOP.md`
> - 路线图：`/root/code/patchTST-sunspot-prediction/project_docs/project_roadmap.md`

## 本分支内容

| 文件/目录 | 用途 |
|-----------|------|
| `configs/EXP-16*.json` | 实验配置快照 |
| `result.txt` | 实验运行记录 |
| `checkpoints/` | 训练好的模型权重 |
| `results/sunspot_...EXP-16c_0/` | EXP-16c 预测结果（pred.npy/true.npy） |
| `run_longExp.py` + `PatchTST_supervised/` | 原管线代码（可复现训练） |

## 实验结果速查

| 实验 | 状态 | step0 | 全步 | 滚动 |
|------|------|-------|------|------|
| EXP-16 | ❌ 无效（误用 run_sunspot_fixed.py） | — | — | — |
| EXP-16b | ❌ 无效（同上） | — | — | — |
| **EXP-16c** | ✅ 有效（run_longExp.py 原管线） | 11.21 | 24.54 | 36.37 |

EXP-16c 配置：Baseline B 参数（seq=132/nh16/el3/df256/pl12/str6/MS/bs32/50ep）在 1749+ 数据上。
