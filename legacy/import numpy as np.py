import numpy as np
pred = np.load(r"F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised\results\Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_1\pred.npy")
true = np.load(r"F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised\results\Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_1\true.npy")
print("pred shape:", pred.shape)
print("true shape:", true.shape)
print("pred min:", pred.min(), "max:", pred.max())


import numpy as np
true = np.load("F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised\results\Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_1\true.npy")
print("测试样本数:", len(true))
print("对应的月份跨度:", len(true) + 132 - 1 + 24)  # 粗略估计

print("true min:", true.min(), "max:", true.max())
print("negative rate:", (pred < 0).mean())
