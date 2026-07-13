"""
prepare_level3_residual.py
Level 3 残差预测数据管线（修正版）。

M4 校准周期: 1867-2008 完整周期 (Cycle 12-23)
训练残差: ssn_smooth - M4 best-fit (完整周期，无泄漏)
测试残差: ssn_smooth - M4 forecast (仅用校准先验 + Cycle 25 前36月观测，纯参数曲线)

输出: sunspot_with_residual_fullfit_v2.csv
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Config
# =============================================================================
PROJECT_ROOT = "/root/code/patchTST-sunspot-prediction-level3/PatchTST_supervised"
SUNSPOT_CSV = f"{PROJECT_ROOT}/dataset/sunspot_with_cycle.csv"
OUT_CSV = f"{PROJECT_ROOT}/dataset/sunspot_with_residual_fullfit_v2.csv"

SMOOTH_WINDOW = 13
CALIB_START_YEAR = 1867   # 校准周期起始年
CALIB_END_YEAR = 2008     # 校准周期结束年
FORECAST_OBS_MONTHS = 36  # Cycle 25 预报时使用的早期观测月数


# =============================================================================
# Gamma curve
# =============================================================================
def gamma_cycle_curve(t, A, tp, alpha, floor=0.0):
    t = np.asarray(t, dtype=float)
    tt = np.maximum(t + 1.0, 1e-3)
    tp = max(tp, 1.0)
    alpha = max(alpha, 0.1)
    y = floor + A * (tt / tp) ** alpha * np.exp(alpha * (1.0 - tt / tp))
    return np.maximum(y, 0.0)


def centered_smooth(y, window=13):
    return y.rolling(window, center=True, min_periods=1).mean()


# =============================================================================
# Cycle detection
# =============================================================================
def detect_cycles(df):
    """返回 cycle_id -> {start_idx, end_idx, complete}"""
    y_s = df["ssn_smooth"].values
    min_idx, _ = find_peaks(-y_s, distance=90, prominence=5)

    ref_date = pd.Timestamp("2008-12-01")
    min_dates = pd.to_datetime(df.loc[min_idx, "date"]).reset_index(drop=True)
    idx_24 = int(np.argmin(np.abs((min_dates - ref_date).dt.days.values)))

    cycles = {}
    for i, start_i in enumerate(min_idx):
        cnum = 24 + (i - idx_24)
        if i < len(min_idx) - 1:
            end_i = min_idx[i + 1] - 1
            complete = True
        else:
            end_i = len(df) - 1
            complete = False
        cycles[cnum] = {
            "start_idx": int(start_i),
            "end_idx": int(end_i),
            "complete": complete,
        }
    return cycles


# =============================================================================
# Per-cycle best-fit (完整周期拟合)
# =============================================================================
def fit_gamma_bestfit(y_smooth):
    t = np.arange(len(y_smooth), dtype=float)
    sigma = max(8.0, float(np.nanstd(y_smooth)) * 0.35)

    def residual(par):
        A, tp, alpha, floor = par
        yhat = gamma_cycle_curve(t, A, tp, alpha, floor)
        r_data = (yhat - y_smooth) / sigma
        r_floor = floor / 8.0
        tail = gamma_cycle_curve(np.array([150.0, 170.0]), A, tp, alpha, floor) / 35.0
        return np.concatenate([r_data, [r_floor], tail])

    A_init = max(np.nanmax(y_smooth), 20.0)
    tp_init = max(float(np.nanargmax(y_smooth)), 24.0)
    x0 = np.array([A_init, tp_init, 3.0, max(0, np.nanmin(y_smooth))])
    lb = np.array([20.0, 24.0, 0.8, 0.0])
    ub = np.array([350.0, 95.0, 8.0, 25.0])

    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=5000)
    A, tp, alpha, floor = res.x
    return A, tp, alpha, floor


# =============================================================================
# Waldmeier calibration (M4 的早期特征 → 峰值/峰时 RidgeCV)
# =============================================================================
def early_features(y_smooth, obs_m):
    m = min(obs_m, len(y_smooth))
    yy = np.asarray(y_smooth[:m], dtype=float)
    if len(yy) < 6:
        yy = np.pad(yy, (0, 6 - len(yy)), constant_values=np.nanmean(yy) if len(yy) else 0)
    idx_max = int(np.nanargmax(yy))
    maxv = float(np.nanmax(yy))
    meanv = float(np.nanmean(yy))
    last12 = float(np.nanmean(yy[-min(12, len(yy)):]))
    first12 = float(np.nanmean(yy[:min(12, len(yy))]))
    slope = (last12 - first12) / max(1, m)
    rise_rate = maxv / max(1, idx_max + 1)
    trend = float(np.polyfit(np.arange(len(yy)), yy, 1)[0]) if len(yy) >= 3 else 0.0
    return np.array([m, maxv, meanv, last12, first12, slope, idx_max, rise_rate, trend], dtype=float)


def train_waldmeier_calibration(df, calib_cycles):
    """用校准周期训练 RidgeCV 模型，从早期特征预测 A 和 tp。"""
    X_list, y_amp_list, y_tp_list = [], [], []
    for cid, cinfo in sorted(calib_cycles.items()):
        seg = df.iloc[cinfo["start_idx"]:cinfo["end_idx"] + 1]
        y_s = seg["ssn_smooth"].values.astype(float)
        if len(y_s) < FORECAST_OBS_MONTHS + 12:
            continue
        feat = early_features(y_s, FORECAST_OBS_MONTHS)
        X_list.append(feat)
        y_amp_list.append(float(np.nanmax(y_s)))
        y_tp_list.append(int(np.nanargmax(y_s)))

    if len(X_list) < 4:
        raise RuntimeError("校准周期不足（<4）。")

    X = np.vstack(X_list)
    y_amp = np.array(y_amp_list)
    y_tp = np.array(y_tp_list)
    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    amp_model = RidgeCV(alphas=alphas).fit(X, y_amp)
    tp_model = RidgeCV(alphas=alphas).fit(X, y_tp)
    return amp_model, tp_model, y_amp, y_tp


# =============================================================================
# M4 parametric forecast (纯参数曲线，不替换早期观测)
# =============================================================================
def m4_parametric_forecast(df, cycle_info, amp_model, tp_model, hist_amp, hist_tp):
    """
    使用 M4 预报逻辑对目标周期生成包络。
    返回纯参数曲线（仅用偏移校正对齐最后一个观测点），不替换早期观测。
    """
    seg = df.iloc[cycle_info["start_idx"]:cycle_info["end_idx"] + 1]
    y_s = seg["ssn_smooth"].values.astype(float)
    n = len(y_s)
    obs_m = min(FORECAST_OBS_MONTHS, n)

    target_feat = early_features(y_s, obs_m)
    A_prior = float(amp_model.predict(target_feat.reshape(1, -1))[0])
    tp_prior = float(tp_model.predict(target_feat.reshape(1, -1))[0])

    A_prior = float(np.clip(A_prior, np.percentile(hist_amp, 5) * 0.70, np.percentile(hist_amp, 95) * 1.30))
    tp_prior = float(np.clip(tp_prior, np.percentile(hist_tp, 5), np.percentile(hist_tp, 95) + 10))
    alpha_prior = 3.0

    yobs = y_s[:obs_m]
    t_obs = np.arange(obs_m, dtype=float)
    sigma = max(8.0, float(np.nanstd(yobs)) * 0.35)

    def residual(par):
        A, tp, alpha, floor = par
        yhat = gamma_cycle_curve(t_obs, A, tp, alpha, floor)
        r_data = (yhat - yobs) / sigma
        r_prior = np.array([
            (A - A_prior) / 35.0,
            (tp - tp_prior) / 10.0,
            (alpha - alpha_prior) / 1.8,
            floor / 8.0,
        ])
        tail = gamma_cycle_curve(np.array([150.0, 170.0]), A, tp, alpha, floor) / 35.0
        return np.concatenate([r_data, r_prior, tail])

    x0 = np.array([max(np.nanmax(yobs), A_prior), max(24, tp_prior), alpha_prior, max(0, np.nanmin(yobs))])
    lb = np.array([max(20, np.nanmax(yobs) * 0.85), 24, 0.8, 0.0])
    ub = np.array([350.0, 95.0, 8.0, 25.0])

    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=5000)
    A, tp, alpha, floor = res.x

    t_full = np.arange(n, dtype=float)
    pred = gamma_cycle_curve(t_full, A, tp, alpha, floor)

    # 仅在最近观测点做偏移校正（不替换早期段）
    offset = y_s[obs_m - 1] - pred[obs_m - 1]
    decay = np.exp(-np.arange(n) / 24.0)
    pred = pred + offset * decay

    return pred, (A, tp, alpha, floor)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Level 3 残差数据管线 (修正版)")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    df = pd.read_csv(SUNSPOT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ssn_smooth"] = centered_smooth(df["ssn"], SMOOTH_WINDOW)

    print(f"\n  数据: {len(df)} 月 ({df['date'].min().date()} ~ {df['date'].max().date()})")

    # ---- 2. 检测周期 ----
    cycles = detect_cycles(df)
    print(f"\n  检测到 {len(cycles)} 个周期 (Cycle {min(cycles.keys())} ~ {max(cycles.keys())})")

    # ---- 3. 分类周期 ----
    calib_cycles = {}
    train_complete = {}
    test_cycles = {}

    for cid, cinfo in sorted(cycles.items()):
        end_date = df.loc[cinfo["end_idx"], "date"]
        if cinfo["complete"]:
            train_complete[cid] = cinfo
            if CALIB_START_YEAR <= end_date.year <= CALIB_END_YEAR:
                calib_cycles[cid] = cinfo
        else:
            test_cycles[cid] = cinfo

    print(f"\n  校准周期 (1867-2008 完整): {list(calib_cycles.keys())}")
    print(f"  训练周期 (所有完整): {len(train_complete)} 个")
    print(f"  测试周期 (未完成): {list(test_cycles.keys())}")

    # ---- 4. 训练 M4 Waldmeier 校准模型 ----
    print(f"\n  [M4 校准] 训练 Waldmeier 先验模型...")
    amp_model, tp_model, hist_amp, hist_tp = train_waldmeier_calibration(df, calib_cycles)
    print(f"    校准样本数: {len(hist_amp)}")
    print(f"    hist_amp: {hist_amp.min():.1f} ~ {hist_amp.max():.1f}")
    print(f"    hist_tp: {hist_tp.min():.0f} ~ {hist_tp.max():.0f}")

    # ---- 5. 生成 M4 包络：训练周期用 best-fit，测试周期用预报 ----
    df["m4_envelope"] = np.nan

    for cid, cinfo in sorted(train_complete.items()):
        seg = df.iloc[cinfo["start_idx"]:cinfo["end_idx"] + 1]
        y_s = seg["ssn_smooth"].values.astype(float)
        A, tp, alpha, floor = fit_gamma_bestfit(y_s)
        envelope = gamma_cycle_curve(np.arange(len(y_s), dtype=float), A, tp, alpha, floor)
        df.loc[seg.index, "m4_envelope"] = envelope

    # ---- 6. 预报测试周期的包络 ----
    for cid, cinfo in sorted(test_cycles.items()):
        envelope, (A, tp, alpha, floor) = m4_parametric_forecast(
            df, cinfo, amp_model, tp_model, hist_amp, hist_tp
        )
        seg = df.iloc[cinfo["start_idx"]:cinfo["end_idx"] + 1]
        df.loc[seg.index, "m4_envelope"] = envelope
        print(f"\n  [M4 预报] Cycle {cid}: A={A:.1f}, tp={tp:.1f}, alpha={alpha:.2f}, floor={floor:.2f}")
        print(f"    预报包络范围: {envelope.min():.1f} ~ {envelope.max():.1f}")
        print(f"    实测平滑范围: {seg['ssn_smooth'].min():.1f} ~ {seg['ssn_smooth'].max():.1f}")

    # ---- 7. 回填 Cycle 1 前的数据 ----
    first_start = min(cinfo["start_idx"] for cinfo in cycles.values())
    if first_start > 0:
        pre_seg = df.iloc[:first_start]
        # 用 Cycle 1 的拟合参数回推
        c1_info = cycles.get(1, cycles[min(cycles.keys())])
        c1_seg = df.iloc[c1_info["start_idx"]:c1_info["end_idx"] + 1]
        y1 = c1_seg["ssn_smooth"].values.astype(float)
        A1, tp1, alpha1, floor1 = fit_gamma_bestfit(y1)
        n_pre = len(pre_seg)
        t_back = np.arange(len(y1) - n_pre, len(y1), dtype=float)
        df.loc[pre_seg.index, "m4_envelope"] = gamma_cycle_curve(t_back, A1, tp1, alpha1, floor1)

    # ---- 8. 检查缺口 ----
    missing = df["m4_envelope"].isna().sum()
    if missing > 0:
        df["m4_envelope"] = df["m4_envelope"].interpolate(method="linear", limit_area="inside")
        df["m4_envelope"] = df["m4_envelope"].fillna(0.0)
        print(f"\n  插值修复缺失: {missing} 月")

    # ---- 9. 计算残差 ----
    df["residual"] = df["ssn_smooth"] - df["m4_envelope"]

    # ---- 10. 按 train/test 分区统计 ----
    # Dataset_Custom split: num_test=70, num_val=132
    num_test = 70
    num_val = 132
    num_train = len(df) - num_val - num_test  # 3119
    train_mask = np.arange(len(df)) < num_train
    test_start = num_train + num_val  # 3119 + 132 = 3251
    test_mask = np.arange(len(df)) >= test_start

    r_train = df.loc[train_mask, "residual"]
    r_test = df.loc[test_mask, "residual"]
    s_train = df.loc[train_mask, "ssn_smooth"]
    s_test = df.loc[test_mask, "ssn_smooth"]
    m4_train = df.loc[train_mask, "m4_envelope"]
    m4_test = df.loc[test_mask, "m4_envelope"]

    print(f"\n{'='*60}")
    print(f"残差统计")
    print(f"{'='*60}")

    print(f"\n  训练集 ({len(r_train)} 月, {df.loc[train_mask, 'date'].min().date()} ~ {df.loc[train_mask, 'date'].max().date()}):")
    print(f"    SSN_smooth: {s_train.mean():.1f} ± {s_train.std():.1f}  [{s_train.min():.1f}, {s_train.max():.1f}]")
    print(f"    M4 best-fit: {m4_train.mean():.1f} ± {m4_train.std():.1f}")
    print(f"    残差: {r_train.mean():.2f} ± {r_train.std():.2f}  [{r_train.min():.1f}, {r_train.max():.1f}]")

    print(f"\n  测试集 ({len(r_test)} 月, {df.loc[test_mask, 'date'].min().date()} ~ {df.loc[test_mask, 'date'].max().date()}):")
    print(f"    SSN_smooth: {s_test.mean():.1f} ± {s_test.std():.1f}  [{s_test.min():.1f}, {s_test.max():.1f}]")
    print(f"    M4 forecast: {m4_test.mean():.1f} ± {m4_test.std():.1f}")
    print(f"    残差: {r_test.mean():.2f} ± {r_test.std():.2f}  [{r_test.min():.1f}, {r_test.max():.1f}]")

    # M4 forecast MAE on test
    from sklearn.metrics import mean_absolute_error
    m4_test_mae = mean_absolute_error(s_test, m4_test)
    print(f"\n    M4 forecast MAE (test): {m4_test_mae:.2f} SSN")

    # ---- 11. 输出 ----
    out = df[["date", "month_sin", "month_cos", "residual", "ssn_smooth", "m4_envelope"]].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"\n  输出: {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
