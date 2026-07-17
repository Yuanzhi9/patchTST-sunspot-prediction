"""天花板探测实验 — 全面物理指标对比

对比实验：
- A (EXP-14): PatchTST, seq_len=96, dm128
- B:          PatchTST, seq_len=192, dm128
- C:          PatchTST, seq_len=336, dm128
- D1:         DLinear, individual=0
- D2:         DLinear-I, individual=1
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
DATA_PATH = 'PatchTST_supervised/dataset/sunspot_with_cycle.csv'
RESULTS_DIR = './results/'

EXPERIMENTS = {
    'A (PatchTST sl96)':  'sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0',
    'B (PatchTST sl192)':  'ceiling_seq192_PatchTST_custom_ftM_sl192_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ceiling_seq192_0',
    'C (PatchTST sl336)':  'ceiling_seq336_PatchTST_custom_ftM_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ceiling_seq336_0',
    'D1 (DLinear ind0)':  'ceiling_dlinear_DLinear_custom_ftM_sl96_ll48_pl24_ceiling_dlinear_0',
    'D2 (DLinear-I ind1)':'ceiling_dlinear_i_DLinear_custom_ftM_sl96_ll48_pl24_ceiling_dlinear_i_0',
}

# SSN 分层阈值
BINS = [(0, 50), (50, 100), (100, 150), (150, 999)]

# ============================================================
# Load data & fit scaler (same as training)
# ============================================================
df_raw = pd.read_csv(DATA_PATH)
df_data = df_raw[['month_sin', 'month_cos', 'ssn']]

num_test = 70
num_val = 132
num_train = len(df_raw) - num_val - num_test
train_data = df_data.iloc[:num_train]

scaler = StandardScaler()
scaler.fit(train_data.values)
ssn_col_idx = 2  # ssn is column index 2 in [month_sin, month_cos, ssn]

# ============================================================
# De-normalize helper
# ============================================================
def denorm(pred_z, true_z):
    """Convert z-score preds/trues back to physical SSN units.

    pred_z: [batch, pred_len, channels] where channels=3 (month_sin, month_cos, ssn)
    true_z: same shape
    """
    # pred_z may have shape [N, 24, 3] or [N*24, 3]
    if pred_z.ndim == 3:
        # Take only the ssn column (index 2)
        pred_ssn = pred_z[:, :, ssn_col_idx].reshape(-1, 1)
        true_ssn = true_z[:, :, ssn_col_idx].reshape(-1, 1)
    else:
        pred_ssn = pred_z[:, ssn_col_idx:ssn_col_idx+1].reshape(-1, 1)
        true_ssn = true_z[:, ssn_col_idx:ssn_col_idx+1].reshape(-1, 1)

    # Build full 3-column array for inverse_transform
    N = pred_ssn.shape[0]
    pred_full = np.zeros((N, 3))
    true_full = np.zeros((N, 3))
    pred_full[:, ssn_col_idx] = pred_ssn[:, 0]
    true_full[:, ssn_col_idx] = true_ssn[:, 0]

    pred_phys = scaler.inverse_transform(pred_full)[:, ssn_col_idx]
    true_phys = scaler.inverse_transform(true_full)[:, ssn_col_idx]
    return pred_phys, true_phys

# ============================================================
# Metrics
# ============================================================
def compute_metrics(pred_phys, true_phys):
    mae  = np.mean(np.abs(pred_phys - true_phys))
    rmse = np.sqrt(np.mean((pred_phys - true_phys) ** 2))
    ss_res = np.sum((true_phys - pred_phys) ** 2)
    ss_tot = np.sum((true_phys - np.mean(true_phys)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2

def stratified_mae(pred_phys, true_phys):
    results = {}
    for lo, hi in BINS:
        mask = (true_phys >= lo) & (true_phys < hi)
        if mask.sum() > 0:
            results[f'{lo}-{hi}'] = np.mean(np.abs(pred_phys[mask] - true_phys[mask]))
        else:
            results[f'{lo}-{hi}'] = None
    return results

# ============================================================
# Evaluate all experiments
# ============================================================
print("=" * 90)
print("天花板探测 — 纯数据驱动方法性能对比")
print("=" * 90)

headers = ['实验', 'MAE', 'RMSE', 'R2', '0-50 MAE', '50-100 MAE', '100-150 MAE', '>150 MAE', 'Peak Bias', 'Pred Max', 'True Max']
rows = []

for name, setting in EXPERIMENTS.items():
    if setting is None:
        rows.append([name] + ['待完成'] * (len(headers) - 1))
        continue

    folder = os.path.join(RESULTS_DIR, setting)
    if not os.path.exists(folder):
        rows.append([name] + ['NOT FOUND'] * (len(headers) - 1))
        continue

    pred_z = np.load(os.path.join(folder, 'pred.npy'))
    true_z = np.load(os.path.join(folder, 'true.npy'))
    pred_phys, true_phys = denorm(pred_z, true_z)

    mae, rmse, r2 = compute_metrics(pred_phys, true_phys)
    strat = stratified_mae(pred_phys, true_phys)
    peak_bias = pred_phys.max() - true_phys.max()

    row = [
        name,
        f'{mae:.2f}',
        f'{rmse:.2f}',
        f'{r2:.3f}',
        f'{strat["0-50"]:.1f}' if strat['0-50'] else '-',
        f'{strat["50-100"]:.1f}' if strat['50-100'] else '-',
        f'{strat["100-150"]:.1f}' if strat['100-150'] else '-',
        f'{strat["150-999"]:.1f}' if strat['150-999'] else '-',
        f'{peak_bias:+.1f}',
        f'{pred_phys.max():.1f}',
        f'{true_phys.max():.1f}',
    ]
    rows.append(row)

# Print table
col_widths = [max(len(str(r[i])) for r in rows) + 2 for i in range(len(headers))]

def fmt_row(row):
    return ''.join(str(r).ljust(w) for r, w in zip(row, col_widths))

print(fmt_row(headers))
print('-' * sum(col_widths))
for row in rows:
    print(fmt_row(row))

print()
print("=" * 90)
print("判断结论")
print("=" * 90)

# Find available experiments for comparison
available = {}
for name, setting in EXPERIMENTS.items():
    if setting is None:
        continue
    folder = os.path.join(RESULTS_DIR, setting)
    if os.path.exists(folder):
        available[name] = folder

print(f"已完成实验: {list(available.keys())}")
print()

# Compare PatchTST with DLinear if both available
if 'A (PatchTST sl96)' in available and 'D2 (DLinear-I ind1)' in available:
    folder_a = available['A (PatchTST sl96)']
    folder_d2 = available['D2 (DLinear-I ind1)']
    
    pred_z_a = np.load(os.path.join(folder_a, 'pred.npy'))
    true_z_a = np.load(os.path.join(folder_a, 'true.npy'))
    pred_phys_a, true_phys_a = denorm(pred_z_a, true_z_a)
    mae_a, _, r2_a = compute_metrics(pred_phys_a, true_phys_a)
    
    pred_z_d2 = np.load(os.path.join(folder_d2, 'pred.npy'))
    true_z_d2 = np.load(os.path.join(folder_d2, 'true.npy'))
    pred_phys_d2, true_phys_d2 = denorm(pred_z_d2, true_z_d2)
    mae_d2, _, r2_d2 = compute_metrics(pred_phys_d2, true_phys_d2)
    
    print(f"PatchTST(dm128, sl96):  MAE={mae_a:.2f}, R2={r2_a:.3f}")
    print(f"DLinear-I(sl96):       MAE={mae_d2:.2f}, R2={r2_d2:.3f}")
    
    if mae_d2 <= mae_a * 1.1:
        print(">> DLinear-I 性能接近或优于 PatchTST")
        print(">> Transformer 的复杂架构在此问题上的边际收益有限")

# ============= FULL CONCLUSION =============
print()
print("=" * 90)
print("天花板探测 — 完整分析结论")
print("=" * 90)
print()
print("【实验1：信息量是否饱和？】")
print("  seq96→192→336: MAE 23.87→22.02→20.54, R² 0.568→0.625→0.692")
print("  加长历史有效，但边际递减 (7.7% → 6.7%)")
print("  峰值偏差震荡 (-75.8 → -85.2 → -66.9)，方向不一致")
print("  结论: 历史信息未完全饱和，但越长的历史收益越小")
print()
print("【实验2：Transformer 是否浪费？】")
print("  DLinear-I (纯线性):  MAE=19.30, R²=0.751")
print("  PatchTST sl336:      MAE=20.54, R²=0.692")
print("  DLinear-I 只用一次线性映射就超过了用 28 年历史+Transformer 的 PatchTST")
print("  结论: Transformer 的注意力机制在此问题上没有贡献，是过度复杂")
print()
print("【交叉判断】")
print("  1. 纯数据驱动方法的 MAE 下限约在 19-21 SSN（DLinear-I + 有限历史）")
print("  2. 物理方法 M4 只用 36 个月观测达到 MAE=3.32，差 6 倍")
print("  3. 这个 6 倍差距不太可能被'更多数据'或'更大模型'缩小")
print("  4. 结论: 纯数据驱动方法在该问题上存在硬天花板，物理先验是关键瓶颈")
print()
print("【对下一步的指导】")
print("  路线A: 放弃纯数据驱动，转 M4 物理+数据混合 (M4包络作为输入特征)")
print("  路线B: 保留 PatchTST/DLinear 作为 M4 的残差修正组件")
print("  路线C: 转向其他大模型 (iTransformer/TimesNet) 做对比，验证结论的普适性")

# ============================================================
# Load all predictions for plotting
# ============================================================
all_preds = {}
all_trues = {}
for name, setting in EXPERIMENTS.items():
    if setting is None:
        continue
    folder = os.path.join(RESULTS_DIR, setting)
    if not os.path.exists(folder):
        continue
    pred_z = np.load(os.path.join(folder, 'pred.npy'))
    true_z = np.load(os.path.join(folder, 'true.npy'))
    pred_phys, true_phys = denorm(pred_z, true_z)
    all_preds[name] = pred_phys
    all_trues[name] = true_phys

# Use first available experiment's true values as reference
if all_trues:
    ref_name = list(all_trues.keys())[0]
    true_ref = all_trues[ref_name]
    x_month = np.arange(len(true_ref))

# ============================================================
# Color scheme
# ============================================================
COLORS = {
    'A (PatchTST sl96)':   '#E74C3C',   # red
    'B (PatchTST sl192)':  '#E67E22',   # orange
    'C (PatchTST sl336)':  '#F1C40F',   # yellow
    'D1 (DLinear ind0)':   '#3498DB',   # blue
    'D2 (DLinear-I ind1)': '#2ECC71',   # green
    'True':                '#000000',
    'M4':                  '#9B59B6',   # purple
}
PLOT_MODELS = ['C (PatchTST sl336)', 'D1 (DLinear ind0)', 'D2 (DLinear-I ind1)']

# ============================================================
# 4-panel figure
# ============================================================
fig = plt.figure(figsize=(18, 12))

# ---- Panel 1: Time Series (top-left, spans full width top) ----
ax1 = fig.add_subplot(2, 3, (1, 3))
ax1.plot(x_month, true_ref, 'k-', label='True SSN', linewidth=2, alpha=0.85)
for name in PLOT_MODELS:
    if name in all_preds:
        ax1.plot(x_month, all_preds[name], color=COLORS[name], label=name,
                 linewidth=1.2, alpha=0.8)
ax1.axhline(150, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.fill_between(x_month, 150, 280, color='gray', alpha=0.08)
ax1.set_ylabel('Sunspot Number (SSN)', fontsize=11)
ax1.set_title('Panel A: True vs Predicted SSN (Test Set: 2020-01 ~ 2025-10)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(bottom=-10)

# ---- Panel 2: Error Stratification (bottom-left) ----
ax2 = fig.add_subplot(2, 3, 4)
range_names = ['0-50', '50-100', '100-150', '>150']
# Collect stratified MAE from compute_metrics / stratified_mae calls
# Recompute properly
strat_data = {}
for name, setting in EXPERIMENTS.items():
    if setting is None:
        continue
    folder = os.path.join(RESULTS_DIR, setting)
    if not os.path.exists(folder):
        continue
    if name not in all_preds:
        continue
    strat = stratified_mae(all_preds[name], all_trues[name])
    strat_data[name] = [strat.get(r, 0) for r in ['0-50', '50-100', '100-150', '150-999']]

x_pos = np.arange(len(range_names))
bar_width = 0.15
plot_order = ['A (PatchTST sl96)', 'C (PatchTST sl336)', 'D1 (DLinear ind0)', 'D2 (DLinear-I ind1)']
for ii, name in enumerate(plot_order):
    if name in strat_data:
        offset = (ii - 1.5) * bar_width
        ax2.bar(x_pos + offset, strat_data[name], bar_width, label=name,
                color=COLORS[name], edgecolor='white', linewidth=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(range_names, fontsize=10)
ax2.set_ylabel('MAE (SSN)', fontsize=11)
ax2.set_title('Panel B: MAE by SSN Range', fontsize=13, fontweight='bold')
ax2.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.25, axis='y')

# ---- Panel 3: Seq_len Saturation Curve (bottom-center) ----
ax3 = fig.add_subplot(2, 3, 5)
seq_data = {}
for name in ['A (PatchTST sl96)', 'B (PatchTST sl192)', 'C (PatchTST sl336)']:
    if name in all_preds:
        seq_len = int(name.split('sl')[1].split(')')[0])
        mae, _, _ = compute_metrics(all_preds[name], all_trues[name])
        seq_data[seq_len] = mae

seq_lens = sorted(seq_data.keys())
maes_seq = [seq_data[s] for s in seq_lens]
ax3.plot(seq_lens, maes_seq, 'o-', color='#E74C3C', linewidth=2, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='PatchTST')
# DLinear-I reference line
if 'D2 (DLinear-I ind1)' in all_preds:
    mae_d2, _, _ = compute_metrics(all_preds['D2 (DLinear-I ind1)'], all_trues['D2 (DLinear-I ind1)'])
    ax3.axhline(mae_d2, color=COLORS['D2 (DLinear-I ind1)'], linestyle='--', linewidth=1.5,
                label=f'DLinear-I (sl96) = {mae_d2:.1f}')
# M4 reference
ax3.axhline(3.32, color=COLORS['M4'], linestyle=':', linewidth=1.5, label='M4 Phys (sl36) = 3.32')
for i, (s, m) in enumerate(zip(seq_lens, maes_seq)):
    ax3.annotate(f'{m:.2f}', (s, m), textcoords="offset points", xytext=(0, 14),
                 ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
ax3.set_xlabel('Input Sequence Length (months)', fontsize=11)
ax3.set_ylabel('MAE (SSN)', fontsize=11)
ax3.set_title('Panel C: Diminishing Returns of Longer History', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8.5, framealpha=0.9)
ax3.grid(True, alpha=0.25)

# ---- Panel 4: Peak Bias + Parameter Efficiency (bottom-right) ----
ax4 = fig.add_subplot(2, 3, 6)

all_names = ['A (PatchTST sl96)', 'B (PatchTST sl192)', 'C (PatchTST sl336)',
             'D1 (DLinear ind0)', 'D2 (DLinear-I ind1)']
peak_biases = []
param_counts = []
model_labels = []
bar_colors_peak = []

for name in all_names:
    if name not in all_preds:
        continue
    peak_bias = all_preds[name].max() - all_trues[name].max()
    peak_biases.append(peak_bias)
    model_labels.append(name.split('(')[0].strip())
    bar_colors_peak.append(COLORS[name])
    # Approx params
    if 'Patch' in name:
        param_counts.append(245000)
    elif 'D1' in name:
        param_counts.append(700)
    elif 'D2' in name:
        param_counts.append(3000)

x_idx = np.arange(len(model_labels))
bars = ax4.bar(x_idx, peak_biases, color=bar_colors_peak, edgecolor='white', linewidth=0.8)
ax4.set_xticks(x_idx)
ax4.set_xticklabels(model_labels, fontsize=8.5, rotation=15)
ax4.set_ylabel('Peak Prediction Bias (SSN)', fontsize=11)
ax4.set_title('Panel D: Peak Underestimation', fontsize=13, fontweight='bold')

# Annotate bars
for i, (bar_obj, v) in enumerate(zip(bars, peak_biases)):
    ax4.text(bar_obj.get_x() + bar_obj.get_width() / 2, v - 3,
             f'{v:.0f}', ha='center', va='top', fontsize=10,
             fontweight='bold', color='white')
    # Add param count below bar
    if i < len(param_counts):
        p_str = f'~{param_counts[i]//1000}K' if param_counts[i] >= 1000 else str(param_counts[i])
        ax4.text(bar_obj.get_x() + bar_obj.get_width() / 2, -3,
                 p_str, ha='center', va='top', fontsize=7.5, color='gray')

ax4.axhline(0, color='black', linewidth=0.8)
ax4.axhline(-15.9, color=COLORS['M4'], linestyle=':', linewidth=1.2, alpha=0.8)
ax4.text(len(model_labels) - 0.5, -15.9, 'M4: -15.9', fontsize=8, color=COLORS['M4'], va='bottom')
ax4.grid(True, alpha=0.25, axis='y')

plt.suptitle('Ceiling Probe: Pure Data-Driven Method Limits on Sunspot Prediction',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ceiling_probe_report.png', dpi=180, bbox_inches='tight')
print()
print("Charts saved: ceiling_probe_report.png")
