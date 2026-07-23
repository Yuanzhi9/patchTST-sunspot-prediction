# 太阳黑子预测 — 项目说明

> **当前阶段：Phase 0 — 文献奠基。** 不跑新实验。详见 `project_roadmap.md`。

## 环境要求
- Python 3.10+
- PyTorch 1.11
- pandas, numpy, matplotlib, scikit-learn, scipy

## 关键文件
- `project_roadmap.md`：主控文档（阶段规划、任务看板）
- `AGENTS.md`：项目规范、基线参数、实验结论速查
- `project_summary_2026-07-17.md`：7 月 17 日完整阶段总结
- `PatchTST_supervised/report_2026-06-14.md`：6 月 14 日详细汇报

## 运行入口
- `run_sunspot_fixed.py`：完整训练（根目录，train_epochs=10）
- `PatchTST_supervised/run_sunspot_fixed.py`：快速测试（train_epochs=1）
- `PatchTST_supervised/run_longExp.py`：命令行参数完整版

## 运行命令

```bash
# 完整训练（10 epochs）
python3 run_sunspot_fixed.py

# 快速测试（1 epoch）
cd PatchTST_supervised && python3 run_sunspot_fixed.py
```

## 核心结果

### 完整训练（2026-06-14）
- 数据：全量 3321 月，参数：seq_len=96, pred_len=24

| Model | MAE(物理) | R² |
|---|---|---|
| PatchTST d_model=512 | 25.27 | 0.539 |
| PatchTST d_model=128 | 23.87 | 0.568 |

### 天花板探测（2026-07-17）
- DLinear-I (纯线性) MAE=19.30, R²=0.751 → **当前最优纯数据驱动**
- PatchTST sl336 MAE=20.54 → Transformer 无增益
- M4 Waldmeier 物理方法 Cycle 25 MAE=3.32 → 好 6 倍
