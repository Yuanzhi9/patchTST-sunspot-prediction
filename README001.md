# 太阳黑子预测 — PatchTST 代码说明

## 环境要求
- Python 3.10+
- PyTorch 1.11
- pandas, numpy, matplotlib, scikit-learn, scipy

## 文件说明
- `run_sunspot_fixed.py`：**完整训练入口**（根目录，train_epochs=10，基线参数已内置）
- `PatchTST_supervised/run_sunspot_fixed.py`：快速测试版（train_epochs=1，适合验证代码能跑通）
- `PatchTST_supervised/run_longExp.py`：命令行参数完整版
- `PatchTST_supervised/exp/exp_main.py`：训练/测试逻辑
- `PatchTST_supervised/data_provider/data_loader.py`：数据加载
- `PatchTST_supervised/dataset/sunspot_with_cycle.csv`：特征数据（month_sin, month_cos, ssn）

## 运行命令

```bash
# 完整训练（推荐，10 epochs）
python3 run_sunspot_fixed.py

# 快速测试（1 epoch）
cd PatchTST_supervised && python3 run_sunspot_fixed.py

# 命令行版本
cd PatchTST_supervised && python3 run_longExp.py \
  --is_training 1 --model_id sunspot --model PatchTST \
  --data custom --root_path ./dataset/ --data_path sunspot_with_cycle.csv \
  --features M --target ssn --freq m \
  --seq_len 96 --pred_len 24 --enc_in 3 --dec_in 3 --c_out 1 \
  --batch_size 16 --train_epochs 10 --num_workers 0 --use_gpu False
```

## 完整训练结果对比（2026-06-14）
- 数据：全量 sunspot_with_cycle.csv（3321 月，1867-2025）
- 参数：seq_len=96, pred_len=24, batch_size=16, 其他全同

| Model | MSE(z) | MAE(z) | RSE | MAE(物理) | RMSE(物理) | R² |
|---|---|---|---|---|---|---|
| d_model=512 | 0.085 | 0.141 | 0.316 | 25.27 | 34.41 | 0.539 |
| d_model=128 | 0.079 | 0.125 | 0.304 | 23.87 | 33.29 | 0.568 |

- d_model=128 全面优于 512
- 误差集中在 SSN>150 峰值区域，需改任务定义为残差预测
