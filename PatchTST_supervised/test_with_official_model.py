import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')
from exp.exp_main import Exp_Main
import argparse

# 1. 创建与训练时相同的参数配置
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.model_id = 'sunspot_96_132'
args.model = 'PatchTST'
args.data = 'custom'
args.root_path = './dataset'
args.data_path = 'sunspot_with_cycle.csv'
args.features = 'M'
args.target = 'ssn'
args.seq_len = 96
args.label_len = 48
args.pred_len = 132
args.freq = 'h'
args.enc_in = 4
args.dec_in = 4
args.c_out = 1
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.fc_dropout = 0.05
args.head_dropout = 0.0
args.patch_len = 16
args.stride = 8
args.padding_patch = 'end'
args.revin = 1
args.affine = 0
args.subtract_last = 0
args.decomposition = 0
args.kernel_size = 25
args.individual = 0
args.dropout = 0.05
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.use_gpu = False
args.use_multi_gpu = False
args.batch_size = 1

# 2. 创建实验对象并加载模型
exp = Exp_Main(args)
setting = 'sunspot_96_132_PatchTST_custom_ftM_sl96_ll48_pl132_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'
checkpoint_path = f'./checkpoints/{setting}/full_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # 改这里

if 'model_state_dict' in checkpoint:
    exp.model.load_state_dict(checkpoint['model_state_dict'])
else:
    exp.model.load_state_dict(checkpoint)
exp.model.eval()

# 3. 手动加载数据（与你手动推理脚本一致）
df = pd.read_csv('./dataset/sunspot_with_cycle.csv')
feature_cols = ['ssn', 'month_sin', 'month_cos', 'cycle_phase']
data = df[feature_cols].values

# 只对 ssn 归一化
scaler = StandardScaler()
ssn_scaled = scaler.fit_transform(data[:, 0:1])
data_scaled = np.concatenate([ssn_scaled, data[:, 1:]], axis=1)

# 取最后 96 个点作为输入
seq_len = 96
x_input = data_scaled[-seq_len:, :]  # (96, 4)
x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # (1, 96, 4)

# 4. 模型预测
with torch.no_grad():
    dec_inp = torch.zeros(1, 132, 4)
    pred = exp.model(x_input, None, dec_inp, None)  # (1, 132, 1)
    pred_ssn_norm = pred[:, :, 0].numpy().flatten()

# 5. 反归一化
pred_ssn = scaler.inverse_transform(pred_ssn_norm.reshape(-1, 1)).flatten()

# 6. 保存并与手动脚本结果比较
np.save('official_model_manual_data_pred.npy', pred_ssn)
manual_pred = np.load('pred_manual.npy')
print(f"官方模型+手动数据预测范围: {pred_ssn.min():.1f} ~ {pred_ssn.max():.1f}")
print(f"手动脚本预测范围: {manual_pred.min():.1f} ~ {manual_pred.max():.1f}")
print(f"两者 MAE 差异: {np.mean(np.abs(pred_ssn - manual_pred)):.2f}")