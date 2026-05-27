# 太阳黑子预测代码说明

## 环境要求
- Python 3.8+
- PyTorch
- pandas, numpy, matplotlib

## 文件说明
- `run_longExp.py`：主运行文件
- `exp/exp_main.py`：训练逻辑（已修改保存方式）
- `utils/tools.py`：EarlyStopping（已修改保存方式）
- `data_provider/data_loader.py`：数据加载
- `dataset/sunspot_with_features.csv`：特征数据
- `prepare_features.py`：特征生成脚本

## 运行命令
```bash
# 单变量版本
python run_longExp.py --is_training 1 --model_id sunspot_96_96 --model PatchTST --data custom --root_path ./dataset/ --data_path sunspot_monthly_clean.csv --features S --target ssn --seq_len 96 --pred_len 96 --enc_in 1 --dec_in 1 --c_out 1 --batch_size 16 --train_epochs 10 --num_workers 0 --use_gpu False

# 加特征版本（最佳结果）
python run_longExp.py --is_training 1 --model_id sunspot_feat3_epoch10 --model PatchTST --data custom --root_path ./dataset/ --data_path sunspot_with_features.csv --features M --target ssn --seq_len 96 --pred_len 96 --enc_in 3 --dec_in 3 --c_out 1 --batch_size 16 --train_epochs 10 --num_workers 0 --use_gpu Fals