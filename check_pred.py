import sys
import torch
import numpy as np
import pandas as pd
from data_provider.data_factory import data_provider
from models.PatchTST import Model

# 这些参数完全复制自你训练时的配置
args = type('Args', (), {
    'model': 'PatchTST',
    'data': 'custom',
    'root_path': './dataset/',
    'data_path': 'sunspot_with_cycle.csv',
    'features': 'M',
    'target': 'ssn',
    'freq': 'h',
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 132,
    'enc_in': 4,
    'dec_in': 4,
    'c_out': 1,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'dropout': 0.05,
    'fc_dropout': 0.05,
    'head_dropout': 0.0,
    'patch_len': 16,
    'stride': 8,
    'padding_patch': 'end',
    'revin': 1,
    'affine': 0,
    'subtract_last': 0,
    'decomposition': 0,
    'kernel_size': 25,
    'individual': 0,
    'embed_type': 0,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'use_gpu': False,
    'use_multi_gpu': False,
    'devices': '0'
})()

# 加载模型
model = Model(args)
checkpoint_path = './checkpoints/sunspot_96_132_PatchTST_custom_ftM_sl96_ll48_pl132_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # 关键修改
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 获取测试数据
_, test_loader = data_provider(args, flag='test')

with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        outputs = model(batch_x)
        print('预测形状:', outputs.shape)  # 应该是 [batch, 132, 1]
        print('第一个样本的前10个预测值:', outputs[0, :10, 0].numpy())
        break