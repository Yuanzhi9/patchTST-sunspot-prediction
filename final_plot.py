import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载 scaler ==========
df = pd.read_csv('./dataset/sunspot_with_cycle.csv')
ssn = df['ssn'].values.reshape(-1, 1)

# 用训练集 fit scaler（和训练时一致）
train_end = int(len(ssn) * 0.7)
scaler = StandardScaler()
scaler.fit(ssn[:train_end])

# ========== 2. 加载模型并获取预测 ==========
import argparse
from exp.exp_main import Exp_Main

args = argparse.Namespace(
    is_training=0,
    model='PatchTST',
    data='custom',
    root_path='./dataset/',
    data_path='sunspot_with_cycle.csv',
    features='M',
    target='ssn',
    freq='h',
    seq_len=96,
    label_len=48,
    pred_len=96,
    enc_in=4,
    dec_in=4,
    c_out=1,
    batch_size=16,
    num_workers=0,
    use_gpu=False,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1,2,3',
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    moving_avg=25,
    factor=1,
    distil=True,
    dropout=0.05,
    fc_dropout=0.05,
    head_dropout=0.0,
    embed='timeF',
    activation='gelu',
    output_attention=False,
    patch_len=16,
    stride=8,
    padding_patch='end',
    revin=1,
    affine=0,
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    embed_type=0,
    train_epochs=10,
    patience=100,
    learning_rate=0.0001,
    lradj='type3',
    pct_start=0.3,
    loss='mse',
    use_amp=False,
    des='test',
)

exp = Exp_Main(args)
checkpoint = torch.load('./checkpoints/sunspot_cycle23_train_PatchTST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth', map_location='cpu', weights_only=False)
exp.model.load_state_dict(checkpoint['model_state_dict'])
exp.model.eval()

test_data, test_loader = exp._get_data(flag='test')

all_preds = []
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float()
        outputs = exp.model(batch_x)
        all_preds.append(outputs[:, :, 0].cpu().numpy())

preds = np.concatenate(all_preds, axis=0)  # (n_samples, pred_len)
pred_first = preds[:, 0]  # 每个样本的第一个预测点

# ========== 3. 时间对齐 ==========
n_test = len(pred_first)
test_start_idx = len(ssn) - n_test
true_aligned = ssn[test_start_idx + 1:test_start_idx + 1 + n_test - 1].flatten()
pred_aligned = pred_first[:-1]

# 还原到原始尺度
true_orig = scaler.inverse_transform(true_aligned.reshape(-1, 1)).flatten()
pred_orig = scaler.inverse_transform(pred_aligned.reshape(-1, 1)).flatten()

# ========== 4. 画图 ==========
plt.figure(figsize=(14, 6))
x = range(len(true_orig))

plt.plot(x, true_orig, 'k-', label='真实值', linewidth=2)
plt.plot(x, pred_orig, 'r--', label='预测值', linewidth=1.5)
plt.fill_between(x, true_orig, pred_orig, alpha=0.2, color='red')

plt.xlabel('时间')
plt.ylabel('太阳黑子数')
plt.title('实验D预测效果（原始尺度）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_prediction.png', dpi=300)
plt.savefig('final_prediction.pdf')

print("✅ 最终预测图已生成: final_prediction.png")
