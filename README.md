# PatchTST 太阳黑子预测

使用深度学习模型探索太阳黑子数（SSN）预测的可行性边界。

## 快速开始

| 想看什么 | 打开 |
|---------|------|
| 项目规范、基线参数、红线 | [`AGENTS.md`](AGENTS.md) |
| 实验操作规程（SOP） | [`project_docs/experiment_SOP.md`](project_docs/experiment_SOP.md) |
| 全部实验历史 | [`project_docs/experiment_history.md`](project_docs/experiment_history.md) |
| 项目路线图 | [`project_docs/project_roadmap.md`](project_docs/project_roadmap.md) |

## 代码结构

| 目录 | 内容 |
|------|------|
| `PatchTST_supervised/` | 模型代码、数据、训练管线、师兄 M4 代码 |
| `scripts/` | 评估和可视化工具（eval_metrics.py 等） |
| `preprocessing/` | 数据预处理脚本 |
| `literature/` | 14 篇文献 PDF + 阅读笔记 |
| `project_docs/` | 项目文档、实验记录、模板 |

## 环境

- Python 3.10.18 + PyTorch 2.8.0+cu128，**NVIDIA RTX A6000（48 GB）**，无 venv/conda
- 历史实验（2026-04 ~ 2026-09）跑在 Python 3.10 + PyTorch 1.11 **纯 CPU** 上；2026-10-10 换机后可比性断裂
  （同配置跨环境差：全步 ≤4%、block ≤10.6%、滚动 ≤28%）——**滚动数只能同环境自比**
- 实况与坑见 [`AGENTS.md`](AGENTS.md)「环境维护」节；数据口径与泄露核查见 [`project_docs/data_pipeline.md`](project_docs/data_pipeline.md)
- `pip install -r requirements.txt`
