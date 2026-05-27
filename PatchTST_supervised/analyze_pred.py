import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ===== 你需要根据实际情况修改的三个路径 =====
DATA_PATH = 'F:/Downloads/patchTST_main/PatchTST-main/PatchTST_supervised/dataset/sunspot_1867-02_2025-10_original_sincos.csv'   # CSV 文件路径
RESULTS_DIR = './results/'                                            # 结果文件夹路径
SETTING_NAME = 'Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_1'  # 你的 setting 名称
# =========================================

# 1. 读取 CSV
df_raw = pd.read_csv(DATA_PATH)
cols = ['month_sin', 'month_cos']
df_raw = df_raw[['date'] + cols + ['ssn']]
df_data = df_raw[df_raw.columns[1:]]  # 取 [month_sin, month_cos, ssn] 三列

# 2. 计算与实验完全一致的数据划分
num_test = 70
num_val = 132
num_train = len(df_raw) - num_val - num_test
train_data = df_data.iloc[:num_train]

# 3. 用训练集拟合 StandardScaler（和训练时完全一致）
scaler = StandardScaler()
scaler.fit(train_data.values)
print(f"scaler.mean_ = {scaler.mean_}")
print(f"scaler.scale_ = {scaler.scale_}")
print()

# 4. 加载预测和真实值（Z-Score 空间）
folder_path = os.path.join(RESULTS_DIR, SETTING_NAME)
pred_z = np.load(os.path.join(folder_path, 'pred.npy'))
true_z = np.load(os.path.join(folder_path, 'true.npy'))

print(f"pred_z shape = {pred_z.shape}")
print(f"true_z shape = {true_z.shape}")
print()

# 5. 反归一化
# pred_z / true_z 只包含目标列（ssn），在 df_data 中 ssn 是第 3 列（索引 2）
# 需要补全另外两列才能调用 inverse_transform
batch, pred_len, n_targets = pred_z.shape  # 通常 (47? 32? 24, 1)
total_samples = pred_z.shape[0] * pred_z.shape[1]

pred_full = np.zeros((total_samples, 3))
true_full = np.zeros((total_samples, 3))

pred_full[:, 2] = pred_z.reshape(-1)
true_full[:, 2] = true_z.reshape(-1)

pred_phys = scaler.inverse_transform(pred_full)[:, 2]
true_phys = scaler.inverse_transform(true_full)[:, 2]

# 6. 分析
print("========== 物理空间统计 ==========")
print(f"预测值 - 最大值: {pred_phys.max():.2f}, 最小值: {pred_phys.min():.2f}")
print(f"真实值 - 最大值: {true_phys.max():.2f}, 最小值: {true_phys.min():.2f}")
print(f"预测负值数量: {(pred_phys < 0).sum()} / {len(pred_phys)} ({(pred_phys < 0).mean() * 100:.2f}%)")
print(f"负值列表（前20个）: {pred_phys[pred_phys < 0][:20]}")
print()

print("========== 预测值分段统计 ==========")
for threshold in [50, 100, 150, 180, 200]:
    count = (pred_phys >= threshold).sum()
    print(f"  预测 >= {threshold}: {count} / {len(pred_phys)}")

print()
print("========== 前 24 个样本的预测 vs 真实 ==========")
for i in range(min(24, len(pred_phys))):
    marker = " ***" if pred_phys[i] > 180 else ""
    print(f"  {i:3d}: 预测={pred_phys[i]:8.2f}{marker}  真实={true_phys[i]:8.2f}")

train_ssn = df_data['ssn'].iloc[:num_train]
print(f"训练集 ssn>150: {(train_ssn > 150).sum()} / {len(train_ssn)}")
print(f"训练集 ssn>200: {(train_ssn > 200).sum()} / {len(train_ssn)}")



import matplotlib.pyplot as plt

# 你已经有了 pred_phys 和 true_phys，直接画图
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(pred_phys))
ax.plot(x, true_phys, 'b-', label='True', alpha=0.8, linewidth=1)
ax.plot(x, pred_phys, 'r-', label='Pred', alpha=0.8, linewidth=1)
ax.axhline(150, color='gray', linestyle='--', alpha=0.5, label='150 threshold')
ax.set_title(f'True vs Predicted (Max Pred={pred_phys.max():.1f}, Max True={true_phys.max():.1f})')
ax.set_xlabel('Sample Point')
ax.set_ylabel('Sunspot Number')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_timeseries.png', dpi=150)
plt.show()


