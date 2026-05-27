import torch
import argparse
from exp.exp_main import Exp_Main

# 从实验D的日志里复制的完整参数
args = argparse.Namespace(
    # 基础参数
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
    
    # 模型结构
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    moving_avg=25,
    factor=1,
    distil=True,
    dropout=0.05,
    fc_dropout=0.05,  # ← 加上这个
    head_dropout=0.0,  # ← 加上这个
    embed='timeF',
    activation='gelu',
    output_attention=False,
    
    # PatchTST 特有
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
    
    # 训练参数
    train_epochs=10,
    patience=100,
    learning_rate=0.0001,
    lradj='type3',
    pct_start=0.3,
    loss='mse',
    use_amp=False,
    des='test',
    
    # checkpoint
    checkpoint_path='./checkpoints/sunspot_cycle23_train_PatchTST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth'
)

# 加载模型
exp = Exp_Main(args)
model = exp.model
checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("模型加载成功")

# 获取测试数据
test_data, test_loader = exp._get_data(flag='test')

# 取一个batch测试
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float()
        outputs = model(batch_x)
        print(f"模型输出形状: {outputs.shape}")
        print(f"第一个样本的前10个预测步的ssn值: {outputs[0, :10, 0]}")
        break