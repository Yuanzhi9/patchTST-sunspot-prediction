import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 1. 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

# ===================== 2. 配置参数 =====================
class Config:
    # 新模型路径（当前结果）
    pred_path = './results/Sunspot_PatchTST_MS_PatchTST_custom_ftMS_sl132_ll66_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy'
    # 旧模型路径（备份）
    old_pred_path = './results_Sunspot_PatchTST_MS_backup_20260404_1018/pred.npy'
    
    data_path = './dataset/sunspot_with_cycle.csv'
    save_fig_path = 'sunspot_prediction_comparison.png'
    
    seq_len = 132
    pred_len = 24
    train_end_idx = 2520
    border1 = 3057
    border2 = 3321

config = Config()

# ===================== 3. 数据加载 =====================
print("="*60)
print("加载数据")
print("="*60)

# 加载新模型预测
new_pred = np.load(config.pred_path)
print(f"新模型预测值形状: {new_pred.shape}")
new_pred_ssn = new_pred[:, :, 0]

# 加载旧模型预测
old_pred = np.load(config.old_pred_path)
print(f"旧模型预测值形状: {old_pred.shape}")
old_pred_ssn = old_pred[:, :, 0]

# 加载真实数据
df = pd.read_csv(config.data_path)
test_ssn_all = df['ssn'].iloc[config.border1:config.border2].values
print(f"测试集长度: {len(test_ssn_all)}")

# 构建滑动窗口真实值
true_ssn_windows = []
n_samples = min(new_pred_ssn.shape[0], old_pred_ssn.shape[0])
for i in range(n_samples):
    start = i + config.seq_len
    end = start + config.pred_len
    if end <= len(test_ssn_all):
        true_ssn_windows.append(test_ssn_all[start:end])

true_ssn = np.array(true_ssn_windows)
new_pred_ssn = new_pred_ssn[:len(true_ssn_windows)]
old_pred_ssn = old_pred_ssn[:len(true_ssn_windows)]
print(f"有效样本数: {len(true_ssn_windows)}")

# ===================== 4. 反归一化 =====================
train_data = df['ssn'].iloc[:config.train_end_idx].values
train_mean = train_data.mean()
train_std = train_data.std()
print(f"训练集统计量: 均值={train_mean:.2f}, 标准差={train_std:.2f}")

new_pred_original = new_pred_ssn * train_std + train_mean
old_pred_original = old_pred_ssn * train_std + train_mean
true_original = true_ssn

print(f"\n反归一化后范围:")
print(f"真实值: {true_original.min():.1f} ~ {true_original.max():.1f}")
print(f"旧模型: {old_pred_original.min():.1f} ~ {old_pred_original.max():.1f}")
print(f"新模型: {new_pred_original.min():.1f} ~ {new_pred_original.max():.1f}")

# ===================== 5. 误差指标 =====================
def calc_metrics(pred, true):
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return mae, rmse

old_mae, old_rmse = calc_metrics(old_pred_original, true_original)
new_mae, new_rmse = calc_metrics(new_pred_original, true_original)

print(f"\n误差对比:")
print(f"旧模型 MAE: {old_mae:.2f}, RMSE: {old_rmse:.2f}")
print(f"新模型 MAE: {new_mae:.2f}, RMSE: {new_rmse:.2f}")
print(f"MAE 改善: {(old_mae - new_mae)/old_mae*100:.1f}%")

# ===================== 6. 可视化 =====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_axis = np.arange(1, config.pred_len + 1)

# 颜色定义
color_true = '#1f77b4'  # 蓝色
color_old = '#7f7f7f'   # 灰色
color_new = '#d62728'   # 红色

# ---------- 图1: 单样本对比 ----------
ax = axes[0, 0]
sample_idx = 0
ax.plot(x_axis, true_original[sample_idx], color=color_true, linewidth=2, label='真实值')
ax.plot(x_axis, old_pred_original[sample_idx], '--', color=color_old, linewidth=1.5, label=f'旧模型 (RevIN=1)')
ax.plot(x_axis, new_pred_original[sample_idx], '--', color=color_new, linewidth=2, label=f'新模型 (RevIN=0)')
ax.set_title(f'样本 {sample_idx+1}: 预测对比')
ax.set_xlabel('预测步数（月）')
ax.set_ylabel('太阳黑子数')
ax.legend()

# ---------- 图2: 平均值对比 ----------
ax = axes[0, 1]
ax.plot(x_axis, true_original.mean(axis=0), color=color_true, linewidth=2, label='真实值')
ax.plot(x_axis, old_pred_original.mean(axis=0), '--', color=color_old, linewidth=1.5, label='旧模型')
ax.plot(x_axis, new_pred_original.mean(axis=0), '--', color=color_new, linewidth=2, label='新模型')
ax.set_title('平均预测对比')
ax.set_xlabel('预测步数（月）')
ax.set_ylabel('太阳黑子数')
ax.legend()

# ---------- 图3: 旧模型散点图 ----------
ax = axes[0, 2]
ax.scatter(true_original.flatten(), old_pred_original.flatten(), alpha=0.3, s=10, color=color_old)
min_val, max_val = true_original.min(), true_original.max()
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.set_title(f'旧模型散点图 (MAE={old_mae:.2f})')

# ---------- 图4: 新模型散点图 ----------
ax = axes[1, 0]
ax.scatter(true_original.flatten(), new_pred_original.flatten(), alpha=0.3, s=10, color=color_new)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.set_title(f'新模型散点图 (MAE={new_mae:.2f})')

# ---------- 图5: 各步MAE对比（旧模型）----------
ax = axes[1, 1]
old_step_mae = np.mean(np.abs(old_pred_original - true_original), axis=0)
ax.bar(x_axis, old_step_mae, color=color_old, alpha=0.7)
ax.set_xlabel('预测步数（月）')
ax.set_ylabel('MAE')
ax.set_title('旧模型各步MAE')

# ---------- 图6: 各步MAE对比（新模型）----------
ax = axes[1, 2]
new_step_mae = np.mean(np.abs(new_pred_original - true_original), axis=0)
bars = ax.bar(x_axis, new_step_mae, color=color_new, alpha=0.7)
# 标注数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    if i % 2 == 0:  # 每隔一个标注，避免拥挤
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', 
                ha='center', va='bottom', fontsize=7)
ax.set_xlabel('预测步数（月）')
ax.set_ylabel('MAE')
ax.set_title('新模型各步MAE')

plt.tight_layout()
plt.savefig(config.save_fig_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: {config.save_fig_path}")

# ===================== 7. 汇总 =====================
print("\n" + "="*60)
print("对比汇总")
print("="*60)
print(f"样本数: {len(true_ssn_windows)}")
print(f"旧模型范围: {old_pred_original.min():.1f} ~ {old_pred_original.max():.1f}, MAE: {old_mae:.2f}")
print(f"新模型范围: {new_pred_original.min():.1f} ~ {new_pred_original.max():.1f}, MAE: {new_mae:.2f}")
print(f"真实值范围: {true_original.min():.1f} ~ {true_original.max():.1f}")
print(f"改善幅度: MAE {(old_mae - new_mae)/old_mae*100:.1f}%, 上限提升 {new_pred_original.max() - old_pred_original.max():.1f}")
print("="*60)