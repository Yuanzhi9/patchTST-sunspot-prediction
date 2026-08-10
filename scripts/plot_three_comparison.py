import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

class Config:
    old_pred_path = './results_Sunspot_PatchTST_MS_backup_20260404_1018/pred.npy'
    es_pred_path = './results/Sunspot_PatchTST_MS_PatchTST_custom_ftMS_sl132_ll66_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy'
    noes_pred_path = './results/Sunspot_PatchTST_MS_NoES_PatchTST_custom_ftMS_sl132_ll66_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy'
    data_path = './dataset/sunspot_with_cycle.csv'
    save_fig_path = 'sunspot_three_model_comparison.png'
    seq_len = 132
    pred_len = 24
    train_end_idx = 2520
    border1 = 3057
    border2 = 3321

config = Config()

# ===================== 数据加载 =====================
print("="*60)
print("三模型对比分析")
print("="*60)

old_pred = np.load(config.old_pred_path)
es_pred = np.load(config.es_pred_path)
noes_pred = np.load(config.noes_pred_path)

print(f"旧模型 (RevIN=1): {old_pred.shape}")
print(f"新模型+ES: {es_pred.shape}")
print(f"新模型无ES: {noes_pred.shape}")

n_samples = min(old_pred.shape[0], es_pred.shape[0], noes_pred.shape[0])
old_pred = old_pred[:n_samples, :, 0]
es_pred = es_pred[:n_samples, :, 0]
noes_pred = noes_pred[:n_samples, :, 0]

df = pd.read_csv(config.data_path)
test_ssn_all = df['ssn'].iloc[config.border1:config.border2].values

true_windows = []
for i in range(n_samples):
    start = i + config.seq_len
    end = start + config.pred_len
    if end <= len(test_ssn_all):
        true_windows.append(test_ssn_all[start:end])

true_ssn = np.array(true_windows)
old_pred = old_pred[:len(true_windows)]
es_pred = es_pred[:len(true_windows)]
noes_pred = noes_pred[:len(true_windows)]

print(f"有效样本数: {len(true_windows)}")

# ===================== 反归一化 =====================
train_data = df['ssn'].iloc[:config.train_end_idx].values
train_mean = train_data.mean()
train_std = train_data.std()

old_original = old_pred * train_std + train_mean
es_original = es_pred * train_std + train_mean
noes_original = noes_pred * train_std + train_mean
true_original = true_ssn

print(f"\n反归一化后范围:")
print(f"真实值: {true_original.min():.1f} ~ {true_original.max():.1f}")
print(f"旧模型: {old_original.min():.1f} ~ {old_original.max():.1f}")
print(f"新+ES: {es_original.min():.1f} ~ {es_original.max():.1f}")
print(f"新无ES: {noes_original.min():.1f} ~ {noes_original.max():.1f}")

# ===================== 误差计算 =====================
def calc_metrics(pred, true):
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return mae, rmse

old_mae, old_rmse = calc_metrics(old_original, true_original)
es_mae, es_rmse = calc_metrics(es_original, true_original)
noes_mae, noes_rmse = calc_metrics(noes_original, true_original)

print(f"\n误差对比:")
print(f"旧模型 (RevIN=1): MAE={old_mae:.2f}, RMSE={old_rmse:.2f}")
print(f"新+EarlyStop: MAE={es_mae:.2f}, RMSE={es_rmse:.2f}")
print(f"新无EarlyStop: MAE={noes_mae:.2f}, RMSE={noes_rmse:.2f}")

# ===================== 可视化 =====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_axis = np.arange(1, config.pred_len + 1)

c_true = '#1f77b4'
c_old = '#7f7f7f'
c_es = '#2ca02c'
c_noes = '#d62728'

# 图1: 单样本
ax = axes[0, 0]
s = 0
ax.plot(x_axis, true_original[s], c=c_true, lw=2, label='真实值')
ax.plot(x_axis, old_original[s], '--', c=c_old, lw=1.5, label='旧(RevIN=1)')
ax.plot(x_axis, es_original[s], '--', c=c_es, lw=2, label='新+ES')
ax.plot(x_axis, noes_original[s], '--', c=c_noes, lw=2, label='新无ES')
ax.set_title(f'样本{s+1}对比')
ax.set_xlabel('预测步数(月)')
ax.set_ylabel('太阳黑子数')
ax.legend(fontsize=8)

# 图2: 平均值
ax = axes[0, 1]
ax.plot(x_axis, true_original.mean(axis=0), c=c_true, lw=2, label='真实值')
ax.plot(x_axis, old_original.mean(axis=0), '--', c=c_old, lw=1.5, label='旧模型')
ax.plot(x_axis, es_original.mean(axis=0), '--', c=c_es, lw=2, label='新+ES')
ax.plot(x_axis, noes_original.mean(axis=0), '--', c=c_noes, lw=2, label='新无ES')
ax.set_title('平均预测对比')
ax.set_xlabel('预测步数(月)')
ax.legend(fontsize=8)

# 图3: 箱线图（修复版）
ax = axes[0, 2]
data_for_box = [old_original.flatten(), es_original.flatten(), noes_original.flatten(), true_original.flatten()]
bp = ax.boxplot(data_for_box, labels=['旧\nRevIN=1', '新+ES', '新无ES', '真实值'], patch_artist=True)
ax.set_title('预测值分布对比')
ax.set_ylabel('太阳黑子数')
colors = [c_old, c_es, c_noes, c_true]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

# 图4-6: 散点图
models = [
    (old_original, '旧模型(RevIN=1)', c_old, old_mae),
    (es_original, '新模型+EarlyStop', c_es, es_mae),
    (noes_original, '新模型无EarlyStop', c_noes, noes_mae)
]

for idx, (pred, title, color, mae) in enumerate(models):
    ax = axes[1, idx]
    ax.scatter(true_original.flatten(), pred.flatten(), alpha=0.3, s=8, c=color)
    min_v = min(true_original.min(), pred.min())
    max_v = max(true_original.max(), pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'{title}\nMAE={mae:.2f}')

plt.tight_layout()
plt.savefig(config.save_fig_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {config.save_fig_path}")

# ===================== 汇总 =====================
print("\n" + "="*60)
print("三模型对比汇总")
print("="*60)
print(f"{'模型':<20} {'MAE':<8} {'RMSE':<8} {'范围':<20}")
print("-"*60)
print(f"{'旧(RevIN=1)':<20} {old_mae:<8.2f} {old_rmse:<8.2f} {f'{old_original.min():.0f}~{old_original.max():.0f}':<20}")
print(f"{'新+EarlyStop':<20} {es_mae:<8.2f} {es_rmse:<8.2f} {f'{es_original.min():.0f}~{es_original.max():.0f}':<20}")
print(f"{'新无EarlyStop':<20} {noes_mae:<8.2f} {noes_rmse:<8.2f} {f'{noes_original.min():.0f}~{noes_original.max():.0f}':<20}")
print(f"{'真实值':<20} {'-':<8} {'-':<8} {f'{true_original.min():.0f}~{true_original.max():.0f}':<20}")
print("="*60)
print(f"最佳模型: 新+EarlyStop (MAE {es_mae:.2f})")
print(f"vs 旧模型改善: {(old_mae-es_mae)/old_mae*100:.1f}%")
print(f"vs 无ES改善: {(noes_mae-es_mae)/noes_mae*100:.1f}%")