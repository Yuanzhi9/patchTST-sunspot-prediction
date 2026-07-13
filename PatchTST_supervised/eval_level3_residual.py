"""
eval_level3_v2.py
Level 3 残差预测评估（修正版）。

评估对象（均限制在 test 段 2020-2025）：
  a) M4 forecast alone — 纯预报包络
  b) M4 forecast + PatchTST residual — 包络+残差叠加
  c) 对比提升量
"""

import numpy as np
import pandas as pd
import os
import glob
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = "/root/code/patchTST-sunspot-prediction-level3/PatchTST_supervised"
RESIDUAL_CSV = f"{PROJECT_ROOT}/dataset/sunspot_with_residual_fullfit_v2.csv"
RESULTS_DIR = "/root/code/patchTST-sunspot-prediction-level3/results"


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp) if len(yt) > 1 and np.std(yt) > 0 else np.nan
    bias = float(np.mean(yp - yt))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias, "N": int(mask.sum())}


def main():
    print("=" * 60)
    print("Level 3 残差预测 — 评估 (v2, M4 预报)")
    print("=" * 60)

    # ---- Load data ----
    df = pd.read_csv(RESIDUAL_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Dataset_Custom split
    num_test = 70
    num_val = 132
    num_train = len(df) - num_val - num_test  # 3119

    # Test in data_loader: border1 = len(df) - num_test - seq_len = 3321 - 70 - 96 = 3155
    # Test rows in CSV: 3155 to 3320 (0-indexed)
    test_start_idx = num_train + num_val  # 3119 + 132 = 3251
    test_end_idx = len(df)               # 3321

    # But pred.npy aligns with data_loader's test borders
    # test_dl_border1 = len(df) - num_test - seq_len = 3321 - 70 - 96 = 3155
    test_dl_start = len(df) - num_test - 96  # 3155

    df_test = df.iloc[test_dl_start:]
    print(f"\n  测试段 CSV 行: {test_dl_start} ~ {len(df)-1} ({df_test['date'].min().date()} ~ {df_test['date'].max().date()})")

    # ---- Find results ----
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "sunspot_residual_*v2*")))
    if not dirs:
        # Fallback to latest sunspot_residual
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "sunspot_residual_*")))
    if not dirs:
        raise FileNotFoundError("No results found")
    setting = os.path.basename(dirs[-1])
    result_path = os.path.join(RESULTS_DIR, setting)

    pred_npy = os.path.join(result_path, "pred.npy")
    true_npy = os.path.join(result_path, "true.npy")
    print(f"\n  结果: {setting}")

    preds_z = np.load(pred_npy)  # [N, 24, 3]
    trues_z = np.load(true_npy)  # [N, 24, 3]
    preds_z = preds_z[:, :, -1:]  # residual channel
    trues_z = trues_z[:, :, -1:]

    # ---- Inverse transform with StandardScaler ----
    cols = ["month_sin", "month_cos", "residual"]
    train_data = df[cols].iloc[:num_train]
    scaler = StandardScaler()
    scaler.fit(train_data.values)
    scale_r = scaler.scale_[-1]
    mean_r = scaler.mean_[-1]

    preds_r = preds_z * scale_r + mean_r  # physical residuals
    trues_r = trues_z * scale_r + mean_r

    n_samples, pred_len, _ = preds_r.shape

    # ---- Build M4 forecast envelope and true SSN_smooth for each prediction step ----
    m4_all = df["m4_envelope"].values
    ssn_smooth_all = df["ssn_smooth"].values

    m4_preds = np.zeros((n_samples, pred_len))
    ssn_true = np.zeros((n_samples, pred_len))
    residual_true = np.zeros((n_samples, pred_len))

    for i in range(n_samples):
        start_row = test_dl_start + i + 96  # border1 + i + seq_len
        for j in range(pred_len):
            row = start_row + j
            if row < len(df):
                m4_preds[i, j] = m4_all[row]
                ssn_true[i, j] = ssn_smooth_all[row]
                residual_true[i, j] = ssn_smooth_all[row] - m4_all[row]

    # ---- Reconstruct ----
    ssn_pred = m4_preds + preds_r[:, :, 0]  # M4 + PatchTST

    mask = np.isfinite(m4_preds) & np.isfinite(preds_r[:, :, 0]) & np.isfinite(ssn_true)

    # ---- Evaluation ----
    print("\n" + "=" * 60)
    print("Test 段评估 (物理单位 SSN_smooth)")
    print("=" * 60)

    m_m4 = calc_metrics(ssn_true[mask], m4_preds[mask])
    m_r = calc_metrics(residual_true[mask], preds_r[:, :, 0][mask])
    m_comb = calc_metrics(ssn_true[mask], ssn_pred[mask])

    r2_label = "R2"
    print(f"\n  [M4 forecast]          M4 预报包络 vs 平滑 SSN:")
    print(f"    MAE={m_m4['MAE']:.2f}, RMSE={m_m4['RMSE']:.2f}, "
          f"{r2_label}={m_m4['R2']:.4f}, Bias={m_m4['Bias']:.2f}")

    print(f"\n  [PatchTST residual]    残差预测 vs 真实残差:")
    print(f"    MAE={m_r['MAE']:.2f}, RMSE={m_r['RMSE']:.2f}, "
          f"{r2_label}={m_r['R2']:.4f}, Bias={m_r['Bias']:.2f}")

    print(f"\n  [M4+PatchTST]          M4预报 + 残差预测 vs 平滑 SSN:")
    print(f"    MAE={m_comb['MAE']:.2f}, RMSE={m_comb['RMSE']:.2f}, "
          f"{r2_label}={m_comb['R2']:.4f}, Bias={m_comb['Bias']:.2f}")

    if m_m4['MAE'] > 0:
        imp = (m_m4['MAE'] - m_comb['MAE']) / m_m4['MAE'] * 100
        print(f"\n  PatchTST 相对 M4 forecast 的 MAE 提升: {imp:+.1f}%")

    # ---- 按 SSN 区间分层 ----
    print("\n" + "=" * 60)
    print("按平滑 SSN 区间分层")
    print("=" * 60)

    ssn_flat = ssn_true[mask]
    m4_flat = m4_preds[mask]
    comb_flat = ssn_pred[mask]

    bins = [0, 50, 100, 150, 1000]
    labels = ["0-50", "50-100", "100-150", ">150"]

    print(f"\n{'区间':>8s} {'N':>5s}  {'M4_MAE':>8s} {'组合MAE':>8s} {'提升%':>8s}")
    print("-" * 42)

    summary_rows = []
    for label, lo, hi in zip(labels, bins[:-1], bins[1:]):
        bin_mask = (ssn_flat >= lo) & (ssn_flat < hi)
        if bin_mask.sum() < 3:
            continue
        m4_b = calc_metrics(ssn_flat[bin_mask], m4_flat[bin_mask])
        comb_b = calc_metrics(ssn_flat[bin_mask], comb_flat[bin_mask])
        imp_b = (m4_b['MAE'] - comb_b['MAE']) / max(m4_b['MAE'], 0.01) * 100
        print(f"{label:>8s} {bin_mask.sum():>5d}  "
              f"{m4_b['MAE']:>7.2f}  {comb_b['MAE']:>7.2f}  {imp_b:>+7.1f}%")
        summary_rows.append({
            "label": label, "N": bin_mask.sum(),
            "M4_MAE": m4_b['MAE'], "Comb_MAE": comb_b['MAE'],
            "Impr": imp_b
        })

    # ---- Write to result.txt ----
    result_file = os.path.join(PROJECT_ROOT, "result.txt")
    with open(result_file, 'a') as f:
        f.write("\n=== Level 3 Residual v2 (" + setting + ") ===\n")
        f.write("M4 forecast: MAE={:.2f}, RMSE={:.2f}, R2={:.4f}\n".format(m_m4['MAE'], m_m4['RMSE'], m_m4['R2']))
        f.write("M4+PatchTST:  MAE={:.2f}, RMSE={:.2f}, R2={:.4f}\n".format(m_comb['MAE'], m_comb['RMSE'], m_comb['R2']))
        f.write("Improvement: {:.1f}%\n".format(imp))
        for r in summary_rows:
            f.write("  {}: N={}, M4={:.2f}, Comb={:.2f}, Impr={:.1f}%\n".format(
                r["label"], r["N"], r["M4_MAE"], r["Comb_MAE"], r["Impr"]))

    print(f"\n  结果已写入: {result_file}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
