import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = 'PatchTST_supervised/dataset/sunspot_with_cycle.csv'
RESULT_DIR = 'results'

SETTINGS = {
    'd_model=512': 'sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0',
    'd_model=128': 'sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0',
}

cols = ['month_sin', 'month_cos']
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw[['date'] + cols + ['ssn']]
df_data = df_raw[df_raw.columns[1:]]

num_test = 70
num_val = 132
num_train = len(df_raw) - num_val - num_test
train_data = df_data.iloc[:num_train]

scaler = StandardScaler()
scaler.fit(train_data.values)

results = {}
for name, setting in SETTINGS.items():
    folder = f'{RESULT_DIR}/{setting}'
    pred_z = np.load(f'{folder}/pred.npy')
    true_z = np.load(f'{folder}/true.npy')

    n_samples, pred_len, n_feat = pred_z.shape
    pred_2d = pred_z.reshape(-1, n_feat)
    true_2d = true_z.reshape(-1, n_feat)

    pred_phys = scaler.inverse_transform(pred_2d)[:, 2]
    true_phys = scaler.inverse_transform(true_2d)[:, 2]

    # normalized space metrics
    mse_z = np.mean((true_z - pred_z) ** 2)
    mae_z = np.mean(np.abs(true_z - pred_z))

    # physical space metrics
    mae = mean_absolute_error(true_phys, pred_phys)
    rmse = np.sqrt(np.mean((true_phys - pred_phys) ** 2))
    r2 = r2_score(true_phys, pred_phys)

    results[name] = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'MSE_z': mse_z, 'MAE_z': mae_z
    }

print()
print(f"{'Model':<14} {'MAE':>8} {'RMSE':>8} {'R²':>8}  {'MSE(z)':>8} {'MAE(z)':>8}")
print("-" * 62)
for name in ['d_model=512', 'd_model=128']:
    r = results[name]
    print(f"{name:<14} {r['MAE']:8.2f} {r['RMSE']:8.2f} {r['R2']:8.3f}  {r['MSE_z']:8.5f} {r['MAE_z']:8.5f}")
print()
print("MAE/RMSE 单位 = SSN（太阳黑子数），R² 无量纲")
print("MSE(z)/MAE(z) = StandardScaler 归一化空间指标")
