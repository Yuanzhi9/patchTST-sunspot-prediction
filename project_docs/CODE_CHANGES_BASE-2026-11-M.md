# BASE-2026-11-M 代码改动说明

> 原则：新实验使用独立新增文件；旧 `run_longExp.py`、`Dataset_Custom`、`Exp_Main` 和 `data_factory.py` 保持原样。

## 新增文件

| 新文件 | 参考旧文件 | 保留的逻辑 | 新增或改变的逻辑 | 目的 |
|---|---|---|---|---|
| `scripts/make_cv5_folds.py` | 无 | 原始月份顺序、五等分要求 | 生成固定五折 JSON、数据 SHA256、内外层范围 | 冻结可复现切分 |
| `scripts/check_cv5_leakage.py` | `data_pipeline.md` 泄露检查思路 | 无重叠、训练段拟合 scaler | 自动检查五块覆盖、内外层不重叠、窗口可用 | 跑前门禁 |
| `PatchTST_supervised/data_provider/cv5_data.py` | `data_loader.Dataset_Custom` | 特征列、滑窗格式、StandardScaler、时间特征 | 五块轮换、内层训练/验证、外层验证、共享 scaler、窗口不得跨段 | 提供正确 CV5 样本 |
| `PatchTST_supervised/exp/exp_cv5.py` | `exp.exp_main.Exp_Main` | 模型构建、loss、vali、checkpoint 格式 | 训练阶段不计算外层 test loss；外层只评价一次；保存 fold 元数据 | 防止 outer fold 进入早停和训练观察 |
| `run_base_cv5.py` | `run_longExp.py` | 显式 CLI 参数、随机种子、checkpoint/result 目录习惯 | `--cv_fold`、`--cv_file`、自动配置快照、独立 setting 命名 | 新实验专用入口 |
| `scripts/eval_cv5.py` | `scripts/eval_metrics.py`、`scripts/plot_exp25_baseline.py` | 物理反归一化、MAE/RMSE/bias、峰谷分析思路 | 读取 CV5 元数据、构建无重叠曲线、五折汇总 | 评估与汇总 |
| `scripts/plot_cv5.py` | `scripts/plot_exp25_baseline.py` | 学术图画法、颜色与标注 | 五折曲线、block24 轨迹、峰谷数值/时间标注 | 可视化检查 |

## 2026-11-19 短上下文修正

G3 `seq_len=1` 冒烟暴露：旧 `Dataset_Custom` 风格会返回
`label_len + pred_len = 48 + 24` 个标签位置；当 `seq_len` 小于
`label_len` 时，不同窗口的 `seq_y` 长度会不一致，DataLoader 无法 collate。

PatchTST 主路径只消费 `seq_x`，`label_len` 并未进入模型前向；因此
`Dataset_BlockCV` 对 BASE-2026-11-M 统一只返回 `pred_len` 个目标位置。
训练目标、模型输入和评估口径不变；该修正允许短上下文组正常 collate。
失败原始日志保留为 `logs/B26M-G3-S01-F1_prelabel-fix.log`。

## 未修改文件

以下文件在本分支中保持原样：

- `run_longExp.py`
- `PatchTST_supervised/data_provider/data_loader.py`
- `PatchTST_supervised/data_provider/data_factory.py`
- `PatchTST_supervised/exp/exp_main.py`

## 验证顺序

1. `python3 scripts/make_cv5_folds.py`
2. `python3 scripts/check_cv5_leakage.py --seq_len 336 --pred_len 24`
3. 单折、短 epoch smoke test
4. 检查 scaler 拟合月数、训练/内层验证/外层验证窗口数和日期
5. 单折完成后运行 `scripts/eval_cv5.py`
6. 单折图通过后再跑五折 E00
