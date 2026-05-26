import torch
from data_provider.data_factory import data_provider
from models.PatchTST import Model

checkpoint_path = './checkpoints/sunspot_96_132_PatchTST_custom_ftM_sl96_ll48_pl132_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
args = ckpt['args']
print("从checkpoint加载参数成功")

# 补充可能缺失的属性
if not hasattr(args, 'batch_size'):
    args.batch_size = 8
if not hasattr(args, 'num_workers'):
    args.num_workers = 0
if not hasattr(args, 'use_amp'):
    args.use_amp = False
if not hasattr(args, 'use_gpu'):
    args.use_gpu = False
if not hasattr(args, 'gpu'):
    args.gpu = 0

model = Model(args)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("模型加载完成")

_, test_loader = data_provider(args, flag='test')

# ========== 关键检查 ==========
print(f"test_loader 长度（样本数）: {len(test_loader)}")

if len(test_loader) == 0:
    print("❌ test_loader 为空，无法进行预测")
    exit()
else:
    print(f"✅ test_loader 有 {len(test_loader)} 个样本")
    # 取第一个 batch
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        print(f"batch_x 形状: {batch_x.shape}")
        print(f"batch_y 形状: {batch_y.shape}")
        with torch.no_grad():
            outputs = model(batch_x)
        print(f"模型输出形状: {outputs.shape}")
        print("第一个样本的前10个预测值（ssn列）:")
        print(outputs[0, :10, 0].numpy())
        break