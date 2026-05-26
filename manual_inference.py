import torch
import numpy as np
import pandas as pd
from models.PatchTST import Model

# 加载模型
class Args:
    pass
args = Args()
args.model = 'PatchTST'
args.enc_in = 4
args.dec_in = 4
args.c_out = 1
args.seq_len = 96
args.pred_len = 132
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.dropout = 0.05
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
args.embed_type = 0
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.use_gpu = False

model = Model(args)
checkpoint_path = './checkpoints/sunspot_96_132_PatchTST_custom_ftM_sl96_ll48_pl132_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 数据准备
df = pd.read_csv('./dataset/sunspot_with_cycle.csv')
last_228 = df.tail(228)
input_data = last_228.iloc[:96, [2,4,5,7]].values  # ssn, month_sin, month_cos, cycle_phase
true_ssn = last_228.iloc[96:, 2].values

# 推理
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
pred_ssn = output[0, :, 0].numpy()

# 保存与打印
np.save('pred_manual.npy', pred_ssn)
np.save('true_manual.npy', true_ssn)
print('前10个预测值:', pred_ssn[:10])
print('前10个真实值:', true_ssn[:10])
print('MAE:', np.mean(np.abs(pred_ssn - true_ssn)))
