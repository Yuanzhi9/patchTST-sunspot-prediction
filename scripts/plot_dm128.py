import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'PatchTST_supervised/dataset/sunspot_with_cycle.csv'
RESULT_DIR = 'results'
SETTING = 'sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0'

# ---- load data ----
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw[['date'] + ['month_sin', 'month_cos'] + ['ssn']]
df_data = df_raw[df_raw.columns[1:]]

num_test = 70
num_val = 132
num_train = len(df_raw) - num_val - num_test
train_data = df_data.iloc[:num_train]

scaler = StandardScaler()
scaler.fit(train_data.values)

# ---- load predictions ----
pred_z = np.load(f'{RESULT_DIR}/{SETTING}/pred.npy')
true_z = np.load(f'{RESULT_DIR}/{SETTING}/true.npy')

n_samples, pred_len, n_feat = pred_z.shape  # (47, 24, 3)

pred_2d = pred_z.reshape(-1, n_feat)
true_2d = true_z.reshape(-1, n_feat)
pred_phys = scaler.inverse_transform(pred_2d)[:, 2].reshape(n_samples, pred_len)
true_phys = scaler.inverse_transform(true_2d)[:, 2].reshape(n_samples, pred_len)

# ---- map to dates ----
# border1_test = len(df_raw) - 70 - 96
N = len(df_raw)
border1_test = N - 70 - 96  # N - 166
# For sample i, pred step j -> row index = border1_test + i + 96 + j = N - 70 + i + j
all_dates = pd.to_datetime(df_raw['date'].values)

# Build per-timestep (each of 70 test months) prediction collection
# Month k (0..69) has predictions from all (i,j) where i + j = k
n_months = 70
global_preds = {k: [] for k in range(n_months)}
global_trues = np.zeros(n_months)

for i in range(n_samples):
    for j in range(pred_len):
        k = i + j
        if k < n_months:
            global_preds[k].append(pred_phys[i, j])
            global_trues[k] = true_phys[i, j]

# ---- aggregate per month ----
months = []
means = []
stds = []
for k in range(n_months):
    months.append(k)
    means.append(np.mean(global_preds[k]))
    stds.append(np.std(global_preds[k]))

means = np.array(means)
stds = np.array(stds)

month_dates = []
month_labels = []
start_idx = border1_test + 96  # N - 70
for k in range(n_months):
    idx = start_idx + k
    d = all_dates[idx]
    month_dates.append(d)
    month_labels.append(d.strftime('%Y-%m'))

errors = np.abs(means - global_trues)

# ---- find peak error periods ----
# get top error months
top_n = 5
top_idx = np.argsort(errors)[-top_n:][::-1]

# ---- plot ----
fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1 = axes[0]

# draw all prediction windows as faint lines
for i in range(n_samples):
    ks = list(range(i, i + pred_len))
    lines = []
    for j, k in enumerate(ks):
        if k < n_months:
            lines.append((k, pred_phys[i, j]))
    if lines:
        xs = [month_labels[l[0]] for l in lines]
        ys = [l[1] for l in lines]
        ax1.plot(xs, ys, 'red', alpha=0.08, linewidth=0.6)

# true SSN
ax1.plot(month_labels, global_trues, 'b-', linewidth=2, label='True SSN', zorder=10)

# mean prediction
ax1.plot(month_labels, means, 'orange', linewidth=1.5, linestyle='--', label='Mean Prediction', zorder=9)

# fill std band
ax1.fill_between(range(n_months), means - stds, means + stds, alpha=0.15, color='orange', label='±1σ')

# annotate peak errors
for idx in top_idx:
    d = month_dates[idx]
    ax1.annotate(
        f'{d.strftime("%Y-%m")}\nΔ={errors[idx]:.0f}',
        xy=(month_labels[idx], global_trues[idx]),
        xytext=(0, 15), textcoords='offset points',
        fontsize=9, fontweight='bold', color='darkred',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2)
    )

ax1.set_ylabel('SSN (Sunspot Number)', fontsize=12)
ax1.set_title(f'PatchTST d_model=128: 24-Month-Ahead Rolling Predictions vs True SSN', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, n_months)
# thin x ticks
tick_step = max(1, n_months // 15)
ax1.set_xticks(range(0, n_months, tick_step))
ax1.set_xticklabels([month_labels[i] for i in range(0, n_months, tick_step)], rotation=45, ha='right', fontsize=8)

# ---- error subplot ----
ax2 = axes[1]
colors = ['darkred' if e > np.percentile(errors, 80) else 'steelblue' for e in errors]
bars = ax2.bar(range(n_months), errors, color=colors, width=0.8, edgecolor='none', alpha=0.9)

ax2.axhline(y=np.mean(errors), color='gray', linestyle='--', alpha=0.6, label=f'Mean Error = {np.mean(errors):.1f}')

# annotate top errors on bars
for idx in top_idx[:3]:
    ax2.annotate(
        f'{all_dates[start_idx + idx].strftime("%Y-%m")}',
        xy=(idx, errors[idx]),
        xytext=(0, 8), textcoords='offset points',
        fontsize=8, color='darkred', ha='center', fontweight='bold'
    )

ax2.set_ylabel('|Error| (SSN)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-1, n_months)
ax2.set_xticks(range(0, n_months, tick_step))
ax2.set_xticklabels([month_labels[i] for i in range(0, n_months, tick_step)], rotation=45, ha='right', fontsize=8)

# ---- summary stats ----
print()
print("=" * 60)
print("Error Analysis Summary")
print("=" * 60)
print(f"Test period: {month_dates[0].strftime('%Y-%m')} to {month_dates[-1].strftime('%Y-%m')} ({n_months} months)")
print(f"Mean |Error|: {np.mean(errors):.1f} SSN")
print(f"Median |Error|: {np.median(errors):.1f} SSN")
print(f"Max |Error|: {errors.max():.1f} SSN")
print(f"Std of |Error|: {np.std(errors):.1f} SSN")
print()
print("Top error periods:")
for idx in top_idx:
    d = month_dates[idx]
    true_val = global_trues[idx]
    pred_val = means[idx]
    print(f"  {d.strftime('%Y-%m')}: true={true_val:.1f}, pred={pred_val:.1f}, |error|={errors[idx]:.1f}")
print()

# ---- error distribution analysis ----
# group by SSN magnitude
print("Error by SSN magnitude:")
bins = [(0, 50), (50, 100), (100, 150), (150, 250)]
for lo, hi in bins:
    mask = (global_trues >= lo) & (global_trues < hi)
    if mask.sum() > 0:
        print(f"  SSN {lo}-{hi}: n={mask.sum()}, mean|error|={errors[mask].mean():.1f}, max|error|={errors[mask].max():.1f}")

print()
print("=" * 60)

plt.tight_layout()
plt.savefig('dm128_prediction_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to dm128_prediction_error_analysis.png")
