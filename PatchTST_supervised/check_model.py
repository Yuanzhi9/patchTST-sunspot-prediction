import torch

ckpt = torch.load('./checkpoints/sunspot_cycle23_train_PatchTST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/full_checkpoint.pth', map_location='cpu', weights_only=False)
print('checkpoint 包含的键:', list(ckpt.keys()))