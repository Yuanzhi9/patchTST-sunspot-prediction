# 天花板探测实验 — 快照分支

> ⚠️ **此分支为 2026-07-17 实验快照，不再维护。**
> 最新项目结论、参数约定、实验历史以主项目文档为准：
> - 规范与基线：`/root/code/patchTST-sunspot-prediction/AGENTS.md`（v2, 2026-08-09）
> - 实验历史：`/root/code/patchTST-sunspot-prediction/experiment_history.md`
> - 路线图：`/root/code/patchTST-sunspot-prediction/project_roadmap.md`
>
> ⚠️ 跨组比较 "DLinear-I 全面超过 PatchTST" 与"硬天花板 MAE≈19.30"已废止（归一化/lr 不同，不可比）。

## 本分支运行命令

```bash
cd /root/code/patchTST-sunspot-prediction-ceiling-probe
python3 -u run_dlinear.py      # DLinear (individual=0)
python3 -u run_dlinear_i.py    # DLinear-I (individual=1)
python3 -u run_seq192.py       # PatchTST seq_len=192
python3 -u run_seq336.py       # PatchTST seq_len=336
python3 -u compare_all.py      # 出对比图表
```

## 本分支内可比的实验组

| 组 | 实验 | 可比性 |
|----|------|--------|
| PatchTST 组内 | seq96 / seq192 / seq336 | ✅ 可比（仅 seq_len 不同） |
| DLinear 组内 | DLinear(ind=0) / DLinear-I(ind=1) | ✅ 可比（仅 individual 不同） |
| PatchTST vs DLinear | — | ❌ 不可跨组比（RevIN/lr 不同） |
