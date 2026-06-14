# 太阳黑子预测 — PatchTST 代码目录说明

## 环境要求
- Python 3.10+
- PyTorch 2.x
- pandas, numpy, matplotlib, scikit-learn, scipy

## 文件说明
- `run_longExp.py`：主训练入口（命令行参数完整版）
- `run_sunspot_fixed.py`：简化入口（参数内置，适合快速实验）
- `exp/exp_main.py`：训练/测试逻辑
- `data_provider/data_loader.py`：数据加载（Dataset_Custom）
- `dataset/sunspot_with_cycle.csv`：特征数据（month_sin, month_cos, ssn）
- `models/PatchTST.py`：模型定义
- `layers/PatchTST_backbone.py`：Transformer backbone

## 运行命令
```bash
# 快速测试（1 epoch，验证代码能跑通）
python3 run_sunspot_fixed.py

# 完整训练（10 epochs，用根目录入口）
cd .. && python3 run_sunspot_fixed.py

# 命令行版本
python3 run_longExp.py \
  --is_training 1 --model_id sunspot --model PatchTST \
  --data custom --root_path ./dataset/ --data_path sunspot_with_cycle.csv \
  --features M --target ssn --freq m \
  --seq_len 96 --pred_len 24 --enc_in 3 --dec_in 3 --c_out 1 \
  --batch_size 16 --train_epochs 10 --num_workers 0 --use_gpu False
```
