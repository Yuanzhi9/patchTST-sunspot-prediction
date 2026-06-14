# -*- coding: utf-8 -*-
"""
Solar Cycle Optimization Framework
----------------------------------
目标：
1) 自动识别太阳活动周；
2) 对 Solar Cycle 24 和 Solar Cycle 25 做严格留出验证；
3) 对比多条优化路线：
   - predecessor-successor analog：只用上一活动周预测下一活动周；
   - early-shape analog：使用目标活动周早期观测修正未来段；
   - waldmeier-calibrated analog：加入上升率/峰值后验校正；
4) 基于 Cycle 25 当前观测，给出 Cycle 25 剩余段 + Cycle 26 弱/中/强情景趋势；
5) 输出 CSV、图和总结文本。

运行方法：
    python solar_cycle_optimization_framework.py

请在 CONFIG 中修改 INPUT_CSV 与 OUT_DIR。
"""

import os
import math
import zipfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
INPUT_CSV = r"/mnt/data/a0e146b8-e2f4-43ca-acca-a4540a20209a.csv"
OUT_DIR = r"/mnt/data/solar_cycle_optimization_framework_outputs"

SMOOTH_WINDOW = 13
FORECAST_TOTAL_MONTHS = 180          # 输出每个活动周最多 15 年趋势
EARLY_OBS_MONTHS_LIST = [36, 48, 60] # 早期观测更新验证窗口
SCENARIO_FORECAST_MONTHS = 240       # Cycle 25 剩余 + Cycle 26 情景总长度
RANDOM_SEED = 20260531
np.random.seed(RANDOM_SEED)

# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def month_range(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="MS")


def centered_smooth(y: pd.Series, window: int = 13) -> pd.Series:
    return y.rolling(window, center=True, min_periods=1).mean()


def resample_array(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.full(n, np.nan)
    if len(y) == 1:
        return np.full(n, y[0])
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, y)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return np.sum(values * w[:, None], axis=0)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    # values: [n_members, n_time]
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    out = []
    for j in range(values.shape[1]):
        v = values[:, j]
        order = np.argsort(v)
        v_sorted = v[order]
        w_sorted = weights[order]
        cdf = np.cumsum(w_sorted) / (np.sum(w_sorted) + 1e-12)
        out.append(np.interp(q, cdf, v_sorted))
    return np.array(out)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {"MAE": np.nan, "RMSE": np.nan, "R": np.nan, "R2": np.nan, "Bias": np.nan, "MAPE": np.nan, "N": int(mask.sum())}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r = np.corrcoef(yt, yp)[0, 1] if len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0 else np.nan
    r2 = r2_score(yt, yp)
    bias = float(np.mean(yp - yt))
    mape = float(np.mean(np.abs((yp - yt) / np.maximum(np.abs(yt), 1.0))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R": r, "R2": r2, "Bias": bias, "MAPE": mape, "N": int(mask.sum())}


def peak_metrics(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 6:
        return {}
    dd = pd.Series(pd.to_datetime(pd.Series(dates).values[mask])).reset_index(drop=True)
    yt = y_true[mask]
    yp = y_pred[mask]
    i_true = int(np.argmax(yt))
    i_pred = int(np.argmax(yp))
    true_date = pd.Timestamp(dd.iloc[i_true])
    pred_date = pd.Timestamp(dd.iloc[i_pred])
    month_error = (pred_date.year - true_date.year) * 12 + (pred_date.month - true_date.month)
    return {
        "true_peak_date": true_date.strftime("%Y-%m"),
        "pred_peak_date": pred_date.strftime("%Y-%m"),
        "peak_month_error": int(month_error),
        "true_peak_ssn": float(yt[i_true]),
        "pred_peak_ssn": float(yp[i_pred]),
        "peak_amp_error": float(yp[i_pred] - yt[i_true]),
        "peak_amp_abs_error": float(abs(yp[i_pred] - yt[i_true])),
    }


def phase_label(month_in_cycle: int, length: int, peak_month: int) -> str:
    # 用周期内峰值位置构造更物理的四阶段：上升、峰值平台、下降、低谷
    if length <= 0:
        return "unknown"
    if month_in_cycle <= max(2, int(0.65 * peak_month)):
        return "rise"
    if month_in_cycle <= min(length - 1, int(1.35 * peak_month)):
        return "peak"
    if month_in_cycle <= int(0.82 * length):
        return "decline"
    return "trough"


@dataclass
class Cycle:
    cycle: int
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp]
    dates: pd.Series
    y: np.ndarray
    raw: np.ndarray
    complete: bool

    @property
    def length(self) -> int:
        return len(self.y)

    @property
    def peak_idx(self) -> int:
        return int(np.nanargmax(self.y)) if len(self.y) > 0 else 0

    @property
    def peak_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates.iloc[self.peak_idx])

    @property
    def peak_amp(self) -> float:
        return float(np.nanmax(self.y)) if len(self.y) > 0 else np.nan

    @property
    def start_amp(self) -> float:
        return float(self.y[0]) if len(self.y) > 0 else np.nan

    def normalized_shape(self, n: int = 96) -> np.ndarray:
        arr = resample_array(self.y, n)
        amp = max(np.nanmax(arr), 1e-6)
        return arr / amp

    def early_features(self, obs_months: int) -> np.ndarray:
        m = min(obs_months, len(self.y))
        yy = np.asarray(self.y[:m], dtype=float)
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

    def full_features(self) -> np.ndarray:
        return np.array([
            self.length,
            self.peak_amp,
            self.peak_idx,
            self.peak_idx / max(1, self.length),
            self.start_amp,
            float(np.nanmean(self.y)),
            float(np.nanstd(self.y)),
        ], dtype=float)


# =============================================================================
# Cycle detection and table construction
# =============================================================================

def load_and_detect_cycles(input_csv: str) -> Tuple[pd.DataFrame, Dict[int, Cycle], pd.DataFrame]:
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ssn_smooth"] = centered_smooth(df["ssn"], SMOOTH_WINDOW)

    y = df["ssn_smooth"].values
    min_idx, _ = find_peaks(-y, distance=90, prominence=5)
    max_idx, _ = find_peaks(y, distance=80, prominence=20)

    # 以 2008-11/12 附近为 Cycle 24 起点，自动反推 cycle number
    min_dates = pd.to_datetime(df.loc[min_idx, "date"]).reset_index(drop=True)
    target_date = pd.Timestamp("2008-12-01")
    idx_24 = int(np.argmin(np.abs((min_dates - target_date).dt.days.values)))
    cycle_numbers = {int(min_idx[i]): 24 + (i - idx_24) for i in range(len(min_idx))}

    cycles: Dict[int, Cycle] = {}
    for i, start_i in enumerate(min_idx):
        cnum = cycle_numbers[int(start_i)]
        if i < len(min_idx) - 1:
            end_i = min_idx[i + 1] - 1
            complete = True
            end_date = pd.Timestamp(df.loc[end_i, "date"])
        else:
            end_i = len(df) - 1
            complete = False
            end_date = None
        seg = df.iloc[start_i:end_i + 1].copy().reset_index(drop=True)
        cycles[cnum] = Cycle(
            cycle=cnum,
            start_date=pd.Timestamp(seg["date"].iloc[0]),
            end_date=end_date,
            dates=seg["date"],
            y=seg["ssn_smooth"].values.astype(float),
            raw=seg["ssn"].values.astype(float),
            complete=complete,
        )

    # 标注原始 df 中每月的 cycle / phase
    df["cycle"] = np.nan
    df["month_in_cycle"] = np.nan
    df["cycle_phase"] = "unknown"
    for c, cyc in cycles.items():
        idx = df.index[df["date"].isin(set(cyc.dates))].tolist()
        for j, ii in enumerate(idx):
            df.loc[ii, "cycle"] = c
            df.loc[ii, "month_in_cycle"] = j
            df.loc[ii, "cycle_phase"] = phase_label(j, cyc.length, cyc.peak_idx)

    cycle_rows = []
    for c, cyc in sorted(cycles.items()):
        cycle_rows.append({
            "cycle": c,
            "start_date": cyc.start_date.strftime("%Y-%m"),
            "end_date": cyc.end_date.strftime("%Y-%m") if cyc.end_date is not None else "partial",
            "complete": cyc.complete,
            "length_months": cyc.length,
            "peak_date": cyc.peak_date.strftime("%Y-%m"),
            "peak_month_in_cycle": cyc.peak_idx,
            "peak_ssn_smooth": cyc.peak_amp,
            "start_ssn_smooth": cyc.start_amp,
        })
    cycle_table = pd.DataFrame(cycle_rows)
    return df, cycles, cycle_table


# =============================================================================
# Forecast models
# =============================================================================

def get_complete_cycles_before(cycles: Dict[int, Cycle], target_cycle: int) -> List[Cycle]:
    return [cyc for c, cyc in sorted(cycles.items()) if c < target_cycle and cyc.complete]


def analog_weights_from_dist(dist: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    if len(dist) == 0:
        return dist
    if temperature is None:
        temperature = np.nanmedian(dist) + 1e-6
    w = np.exp(-dist / max(temperature, 1e-6))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(dist)
    return w / w.sum()


def predecessor_successor_forecast(
    cycles: Dict[int, Cycle], target_cycle: int, total_months: int = FORECAST_TOTAL_MONTHS, n_shape: int = 96
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """只用 target_cycle-1 与历史 predecessor 周期相似度，预测 target_cycle。"""
    prev = cycles[target_cycle - 1]
    candidate_pairs = []
    for c in sorted(cycles.keys()):
        if c + 1 >= target_cycle:
            continue
        if c in cycles and c + 1 in cycles and cycles[c].complete and cycles[c + 1].complete:
            candidate_pairs.append((cycles[c], cycles[c + 1]))
    if len(candidate_pairs) < 3:
        raise RuntimeError("Not enough predecessor-successor pairs for validation.")

    prev_shape = prev.normalized_shape(n_shape)
    feat_target = prev.full_features()
    feats = np.vstack([p.full_features() for p, s in candidate_pairs])
    scaler = StandardScaler().fit(feats)
    feat_dist = np.linalg.norm(scaler.transform(feats) - scaler.transform(feat_target.reshape(1, -1)), axis=1)
    shape_dist = np.array([np.sqrt(np.mean((p.normalized_shape(n_shape) - prev_shape) ** 2)) for p, s in candidate_pairs])
    dist = 0.65 * shape_dist + 0.35 * (feat_dist / (np.nanmedian(feat_dist) + 1e-6))
    w = analog_weights_from_dist(dist)

    member_paths = np.vstack([resample_array(s.y, total_months) for p, s in candidate_pairs])
    y_mean = weighted_mean(member_paths, w)
    y_p10 = weighted_quantile(member_paths, w, 0.10)
    y_p90 = weighted_quantile(member_paths, w, 0.90)

    start_date = cycles[target_cycle].start_date
    dates = month_range(start_date, total_months)
    out = pd.DataFrame({
        "date": dates,
        "cycle": target_cycle,
        "month_in_cycle": np.arange(total_months),
        "model": "M1_predecessor_successor_analog",
        "pred": y_mean,
        "p10": y_p10,
        "p90": y_p90,
    })
    weights = pd.DataFrame({
        "target_cycle": target_cycle,
        "candidate_predecessor_cycle": [p.cycle for p, s in candidate_pairs],
        "candidate_successor_cycle": [s.cycle for p, s in candidate_pairs],
        "distance": dist,
        "weight": w,
        "successor_peak_ssn": [s.peak_amp for p, s in candidate_pairs],
        "successor_peak_month": [s.peak_idx for p, s in candidate_pairs],
    }).sort_values("weight", ascending=False)
    return out, weights


def fit_waldmeier_models(train_cycles: List[Cycle], obs_months: int):
    X = np.vstack([cyc.early_features(obs_months) for cyc in train_cycles])
    y_amp = np.array([cyc.peak_amp for cyc in train_cycles])
    y_peak_month = np.array([cyc.peak_idx for cyc in train_cycles])
    y_length = np.array([cyc.length for cyc in train_cycles])
    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    amp_model = RidgeCV(alphas=alphas).fit(X, y_amp)
    peak_model = RidgeCV(alphas=alphas).fit(X, y_peak_month)
    len_model = RidgeCV(alphas=alphas).fit(X, y_length)
    return amp_model, peak_model, len_model


def time_warp_cycle_to_target(cyc: Cycle, target_length: int, target_peak_month: int) -> np.ndarray:
    """把候选周期按峰值位置做简单时间变形。"""
    src_peak = max(1, cyc.peak_idx)
    src_len = max(2, cyc.length)
    target_length = max(24, int(target_length))
    target_peak_month = int(np.clip(target_peak_month, 6, target_length - 6))
    target_t = np.arange(target_length)
    src_query = np.zeros(target_length, dtype=float)
    for i, t in enumerate(target_t):
        if t <= target_peak_month:
            src_query[i] = t / max(1, target_peak_month) * src_peak
        else:
            src_query[i] = src_peak + (t - target_peak_month) / max(1, target_length - target_peak_month - 1) * (src_len - src_peak - 1)
    return np.interp(src_query, np.arange(src_len), cyc.y)


def early_shape_forecast(
    cycles: Dict[int, Cycle], target_cycle: int, obs_months: int, total_months: int = FORECAST_TOTAL_MONTHS,
    calibrated: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """使用目标周期早期观测，与历史完整周期早期段匹配，预测后续。calibrated=True 时加入 Waldmeier 后验校正。"""
    target = cycles[target_cycle]
    obs_m = min(obs_months, len(target.y))
    train_cycles = get_complete_cycles_before(cycles, target_cycle)
    train_cycles = [c for c in train_cycles if c.length > obs_m + 12]
    if len(train_cycles) < 4:
        raise RuntimeError("Not enough training cycles for early-shape forecast.")

    target_early = target.y[:obs_m]
    target_norm = target_early / max(np.nanmax(target_early), 1e-6)
    target_feat = target.early_features(obs_m)
    feats = np.vstack([c.early_features(obs_m) for c in train_cycles])
    scaler = StandardScaler().fit(feats)
    feat_dist = np.linalg.norm(scaler.transform(feats) - scaler.transform(target_feat.reshape(1, -1)), axis=1)
    shape_dist = []
    for c in train_cycles:
        ce = c.y[:obs_m]
        cn = ce / max(np.nanmax(ce), 1e-6)
        shape_dist.append(np.sqrt(np.mean((cn - target_norm) ** 2)))
    shape_dist = np.array(shape_dist)
    dist = 0.70 * shape_dist + 0.30 * (feat_dist / (np.nanmedian(feat_dist) + 1e-6))
    w = analog_weights_from_dist(dist)

    diagnostics = {
        "pred_peak_amp": np.nan,
        "pred_peak_month": np.nan,
        "pred_length": np.nan,
    }

    member_paths = []
    if calibrated:
        amp_model, peak_model, len_model = fit_waldmeier_models(train_cycles, obs_m)
        pred_amp = float(amp_model.predict(target_feat.reshape(1, -1))[0])
        pred_peak_month = float(peak_model.predict(target_feat.reshape(1, -1))[0])
        pred_length = float(len_model.predict(target_feat.reshape(1, -1))[0])
        # 经验边界，防止小样本回归发散
        hist_amp = np.array([c.peak_amp for c in train_cycles])
        hist_len = np.array([c.length for c in train_cycles])
        hist_peak = np.array([c.peak_idx for c in train_cycles])
        pred_amp = float(np.clip(pred_amp, np.percentile(hist_amp, 5) * 0.75, np.percentile(hist_amp, 95) * 1.25))
        pred_length = float(np.clip(pred_length, np.percentile(hist_len, 10), np.percentile(hist_len, 90) + 18))
        pred_peak_month = float(np.clip(pred_peak_month, np.percentile(hist_peak, 10), np.percentile(hist_peak, 90) + 12))
        diagnostics = {"pred_peak_amp": pred_amp, "pred_peak_month": pred_peak_month, "pred_length": pred_length}
        for c in train_cycles:
            path = time_warp_cycle_to_target(c, int(round(pred_length)), int(round(pred_peak_month)))
            # 峰值幅度校正
            path = path * (pred_amp / max(np.nanmax(path), 1e-6))
            path = resample_array(path, total_months)
            member_paths.append(path)
    else:
        # 只做早期段相似匹配，并按早期平均幅度缩放
        for c in train_cycles:
            path = resample_array(c.y, total_months)
            c_early = path[:obs_m]
            scale = (np.nanmean(target_early[-min(12, obs_m):]) + 1e-6) / (np.nanmean(c_early[-min(12, obs_m):]) + 1e-6)
            scale = float(np.clip(scale, 0.5, 2.5))
            member_paths.append(path * scale)

    member_paths = np.vstack(member_paths)
    y_mean = weighted_mean(member_paths, w)
    y_p10 = weighted_quantile(member_paths, w, 0.10)
    y_p90 = weighted_quantile(member_paths, w, 0.90)

    # 已知早期观测段直接采用真实平滑值，未来段采用预测；并在衔接处做偏差平移
    if obs_m > 0:
        offset = target.y[obs_m - 1] - y_mean[obs_m - 1]
        decay = np.exp(-np.arange(total_months) / 36.0)
        y_mean = y_mean + offset * decay
        y_p10 = y_p10 + offset * decay
        y_p90 = y_p90 + offset * decay
        y_mean[:obs_m] = target.y[:obs_m]
        y_p10[:obs_m] = target.y[:obs_m]
        y_p90[:obs_m] = target.y[:obs_m]

    start_date = target.start_date
    dates = month_range(start_date, total_months)
    model_name = "M3_waldmeier_calibrated_early_shape" if calibrated else "M2_early_shape_analog"
    out = pd.DataFrame({
        "date": dates,
        "cycle": target_cycle,
        "month_in_cycle": np.arange(total_months),
        "obs_months_used": obs_m,
        "model": model_name,
        "pred": y_mean,
        "p10": y_p10,
        "p90": y_p90,
    })
    weights = pd.DataFrame({
        "target_cycle": target_cycle,
        "obs_months_used": obs_m,
        "candidate_cycle": [c.cycle for c in train_cycles],
        "distance": dist,
        "weight": w,
        "candidate_peak_ssn": [c.peak_amp for c in train_cycles],
        "candidate_peak_month": [c.peak_idx for c in train_cycles],
    }).sort_values("weight", ascending=False)
    return out, weights, diagnostics



def gamma_cycle_curve(t: np.ndarray, A: float, tp: float, alpha: float, floor: float = 0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    tt = np.maximum(t + 1.0, 1e-3)
    tp = max(tp, 1.0)
    alpha = max(alpha, 0.1)
    y = floor + A * (tt / tp) ** alpha * np.exp(alpha * (1.0 - tt / tp))
    return np.maximum(y, 0.0)


def parametric_waldmeier_forecast(
    cycles: Dict[int, Cycle], target_cycle: int, obs_months: int, total_months: int = FORECAST_TOTAL_MONTHS
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """M4: gamma/Hathaway-like parametric cycle curve fitted to early observations with Waldmeier priors."""
    from scipy.optimize import least_squares
    target = cycles[target_cycle]
    obs_m = min(obs_months, len(target.y))
    train_cycles = get_complete_cycles_before(cycles, target_cycle)
    train_cycles = [c for c in train_cycles if c.length > obs_m + 12]
    if len(train_cycles) < 4:
        raise RuntimeError("Not enough training cycles for M4.")
    target_feat = target.early_features(obs_m)
    amp_model, peak_model, len_model = fit_waldmeier_models(train_cycles, obs_m)
    A_prior = float(amp_model.predict(target_feat.reshape(1, -1))[0])
    tp_prior = float(peak_model.predict(target_feat.reshape(1, -1))[0])
    hist_amp = np.array([c.peak_amp for c in train_cycles])
    hist_peak = np.array([c.peak_idx for c in train_cycles])
    A_prior = float(np.clip(A_prior, np.percentile(hist_amp, 5) * 0.70, np.percentile(hist_amp, 95) * 1.30))
    tp_prior = float(np.clip(tp_prior, np.percentile(hist_peak, 5), np.percentile(hist_peak, 95) + 10))
    alpha_prior = 3.0
    yobs = target.y[:obs_m]
    t_obs = np.arange(obs_m, dtype=float)
    sigma = max(8.0, float(np.nanstd(yobs)) * 0.35)

    def residual(par):
        A, tp, alpha, floor = par
        yhat = gamma_cycle_curve(t_obs, A, tp, alpha, floor)
        r_data = (yhat - yobs) / sigma
        # priors prevent explosive fits when only rising branch is visible
        r_prior = np.array([
            (A - A_prior) / 35.0,
            (tp - tp_prior) / 10.0,
            (alpha - alpha_prior) / 1.8,
            floor / 8.0,
        ])
        # force curve to go near zero by 13-15 years
        tail = gamma_cycle_curve(np.array([150.0, 170.0]), A, tp, alpha, floor) / 35.0
        return np.concatenate([r_data, r_prior, tail])

    x0 = np.array([max(np.nanmax(yobs), A_prior), max(24, tp_prior), alpha_prior, max(0, np.nanmin(yobs))])
    lb = np.array([max(20, np.nanmax(yobs) * 0.85), 24, 0.8, 0.0])
    ub = np.array([350.0, 95.0, 8.0, 25.0])
    res = least_squares(residual, x0, bounds=(lb, ub), max_nfev=5000)
    A, tp, alpha, floor = res.x
    t_full = np.arange(total_months, dtype=float)
    pred = gamma_cycle_curve(t_full, A, tp, alpha, floor)
    # smooth transition from actual observed to fitted future
    offset = target.y[obs_m - 1] - pred[obs_m - 1]
    decay = np.exp(-np.arange(total_months) / 24.0)
    pred = pred + offset * decay
    pred[:obs_m] = target.y[:obs_m]
    # uncertainty band from loose amplitude/timing perturbation
    lo = gamma_cycle_curve(t_full, A * 0.82, max(24, tp - 8), alpha, floor)
    hi = gamma_cycle_curve(t_full, A * 1.18, min(95, tp + 8), alpha, floor)
    lo = lo + offset * decay
    hi = hi + offset * decay
    lo[:obs_m] = target.y[:obs_m]
    hi[:obs_m] = target.y[:obs_m]
    p10 = np.minimum(lo, hi)
    p90 = np.maximum(lo, hi)
    dates = month_range(target.start_date, total_months)
    out = pd.DataFrame({
        "date": dates,
        "cycle": target_cycle,
        "month_in_cycle": np.arange(total_months),
        "obs_months_used": obs_m,
        "model": "M4_parametric_waldmeier_curve",
        "pred": pred,
        "p10": p10,
        "p90": p90,
    })
    diag = {"A_fit": float(A), "tp_fit": float(tp), "alpha_fit": float(alpha), "floor_fit": float(floor),
            "A_prior": float(A_prior), "tp_prior": float(tp_prior), "cost": float(res.cost)}
    return out, diag

# =============================================================================
# Validation and plotting
# =============================================================================

def merge_forecast_with_observed(forecast: pd.DataFrame, cycles: Dict[int, Cycle], target_cycle: int) -> pd.DataFrame:
    target = cycles[target_cycle]
    obs_df = pd.DataFrame({"date": pd.to_datetime(target.dates), "observed": target.y, "observed_raw": target.raw})
    out = forecast.merge(obs_df, on="date", how="left")
    out["has_observation"] = out["observed"].notna()
    return out


def evaluate_forecast(df: pd.DataFrame, model_label: str, target_cycle: int, obs_months_used: int, eval_future_only: bool) -> Tuple[Dict, Dict, pd.DataFrame]:
    sub = df[df["has_observation"]].copy()
    if eval_future_only and obs_months_used > 0:
        sub = sub[sub["month_in_cycle"] >= obs_months_used].copy()
    metrics = calc_metrics(sub["observed"].values, sub["pred"].values)
    metrics.update({
        "cycle": target_cycle,
        "model": model_label,
        "obs_months_used": obs_months_used,
        "eval_mode": "future_only" if eval_future_only else "full_observed_window",
    })
    pm = peak_metrics(sub["date"], sub["observed"].values, sub["pred"].values)
    pm.update({"cycle": target_cycle, "model": model_label, "obs_months_used": obs_months_used, "eval_mode": metrics["eval_mode"]})

    # phase-wise metrics
    phase_rows = []
    for ph, g in sub.groupby(sub["month_in_cycle"].map(lambda m: phase_label(int(m), max(1, int(sub["month_in_cycle"].max())+1), int(np.nanargmax(sub["observed"].values)) if len(sub)>0 else 0))):
        mt = calc_metrics(g["observed"].values, g["pred"].values)
        mt.update({"cycle": target_cycle, "model": model_label, "obs_months_used": obs_months_used, "phase": ph})
        phase_rows.append(mt)
    return metrics, pm, pd.DataFrame(phase_rows)


def plot_validation_curve(all_df: pd.DataFrame, target_cycle: int, out_path: str, title: str):
    plt.figure(figsize=(13, 6))
    obs = all_df[all_df["has_observation"]].drop_duplicates("date")
    plt.plot(obs["date"], obs["observed"], linewidth=2.8, label=f"Observed SC{target_cycle} 13-month smoothed")
    for model, g in all_df.groupby("model_run"):
        plt.plot(g["date"], g["pred"], linewidth=1.6, alpha=0.95, label=model)
    plt.xlabel("Date")
    plt.ylabel("Smoothed monthly sunspot number")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual(df: pd.DataFrame, target_cycle: int, out_path: str, title: str):
    plt.figure(figsize=(13, 5))
    for model, g in df[df["has_observation"]].groupby("model_run"):
        plt.plot(g["date"], g["pred"] - g["observed"], linewidth=1.5, label=model)
    plt.axhline(0, linewidth=1.0)
    plt.xlabel("Date")
    plt.ylabel("Prediction - observation")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, out_path: str):
    # 只画 future_only 的 MAE/RMSE 便于比较
    m = metrics_df.copy()
    m["label"] = "SC" + m["cycle"].astype(str) + " | " + m["model"] + " | obs=" + m["obs_months_used"].astype(str)
    m = m.sort_values(["cycle", "MAE"])
    plt.figure(figsize=(13, max(5, 0.35 * len(m))))
    y = np.arange(len(m))
    plt.barh(y, m["MAE"].values, alpha=0.75, label="MAE")
    plt.barh(y, m["RMSE"].values, alpha=0.45, label="RMSE")
    plt.yticks(y, m["label"].values, fontsize=7)
    plt.xlabel("Error")
    plt.title("Cycle 24 / Cycle 25 validation error comparison")
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Cycle 26 scenario generation
# =============================================================================

def cycle26_scenarios(cycles: Dict[int, Cycle], current_cycle: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """基于 Cycle 25 已观测形态，生成 Cycle 25 剩余 + Cycle 26 弱/中/强情景。"""
    c25 = cycles[current_cycle]
    obs_m = len(c25.y)
    # 先用 Waldmeier calibrated early shape 预测 Cycle25 全段
    c25_forecast, c25_weights, diag = early_shape_forecast(cycles, current_cycle, obs_m, total_months=180, calibrated=True)
    # 估计 Cycle25 结束：预测曲线下降到峰后低谷附近或全局后段最小
    pred = c25_forecast["pred"].values
    peak_i = int(np.argmax(pred))
    # 从峰后 48 个月以后找局部最小/全局最小，避免把当前最低点当成结束
    search_start = min(len(pred)-1, max(peak_i + 48, obs_m + 24))
    if search_start < len(pred) - 6:
        end_i = search_start + int(np.argmin(pred[search_start:]))
    else:
        end_i = min(len(pred)-1, obs_m + 72)
    cycle26_start = pd.Timestamp(c25_forecast["date"].iloc[end_i]) + pd.DateOffset(months=1)

    # 当前 Cycle25 估计完整形态作为 predecessor，与历史 predecessor-successor pair 匹配，生成 Cycle26 情景
    candidate_pairs = []
    for c in sorted(cycles.keys()):
        if c + 1 >= current_cycle:
            continue
        if cycles[c].complete and cycles[c + 1].complete:
            candidate_pairs.append((cycles[c], cycles[c + 1]))
    # 构造 Cycle25 pseudo predecessor 特征
    pseudo_y = pred[:end_i+1]
    pseudo_cycle = Cycle(current_cycle, c25.start_date, pd.Timestamp(c25_forecast["date"].iloc[end_i]),
                         pd.Series(month_range(c25.start_date, len(pseudo_y))), pseudo_y, pseudo_y, True)
    n_shape = 96
    target_shape = pseudo_cycle.normalized_shape(n_shape)
    feats = np.vstack([p.full_features() for p, s in candidate_pairs])
    target_feat = pseudo_cycle.full_features()
    scaler = StandardScaler().fit(feats)
    feat_dist = np.linalg.norm(scaler.transform(feats) - scaler.transform(target_feat.reshape(1, -1)), axis=1)
    shape_dist = np.array([np.sqrt(np.mean((p.normalized_shape(n_shape) - target_shape) ** 2)) for p, s in candidate_pairs])
    dist = 0.60 * shape_dist + 0.40 * (feat_dist / (np.nanmedian(feat_dist) + 1e-6))
    w = analog_weights_from_dist(dist)

    # 历史 successor 形态作为 Cycle26 候选
    succ_peaks = np.array([s.peak_amp for p, s in candidate_pairs])
    succ_peak_months = np.array([s.peak_idx for p, s in candidate_pairs])
    succ_lengths = np.array([s.length for p, s in candidate_pairs])
    weak_amp = float(weighted_quantile(succ_peaks[:, None], w, 0.20)[0])
    mid_amp = float(weighted_quantile(succ_peaks[:, None], w, 0.50)[0])
    strong_amp = float(weighted_quantile(succ_peaks[:, None], w, 0.80)[0])
    mid_peak_month = int(round(weighted_quantile(succ_peak_months[:, None], w, 0.50)[0]))
    mid_length = int(round(weighted_quantile(succ_lengths[:, None], w, 0.50)[0]))

    # 生成一个中位形态模板，再缩放到弱/中/强幅度
    member_paths = np.vstack([resample_array(s.y / max(s.peak_amp, 1e-6), 180) for p, s in candidate_pairs])
    template = weighted_mean(member_paths, w)
    template = template / max(np.max(template), 1e-6)
    # 简单调整到中位长度，输出固定 180 月，前 mid_length 后衰减到低谷
    dates26 = month_range(cycle26_start, 180)
    rows = []
    for name, amp in [("weak", weak_amp), ("medium", mid_amp), ("strong", strong_amp)]:
        y = template * amp
        for i, d in enumerate(dates26):
            rows.append({"date": d, "cycle": 26, "scenario": name, "month_in_cycle": i, "pred": y[i]})
    sc26 = pd.DataFrame(rows)

    # Cycle25 剩余 + Cycle26 中情景拼接用于主图
    scenario_summary = pd.DataFrame([
        {"item": "cycle25_start", "value": c25.start_date.strftime("%Y-%m")},
        {"item": "latest_observation", "value": c25.dates.iloc[-1].strftime("%Y-%m")},
        {"item": "estimated_cycle26_start", "value": cycle26_start.strftime("%Y-%m")},
        {"item": "estimated_cycle26_medium_peak_date", "value": (cycle26_start + pd.DateOffset(months=int(np.argmax(template)))).strftime("%Y-%m")},
        {"item": "cycle26_weak_peak_ssn", "value": f"{weak_amp:.2f}"},
        {"item": "cycle26_medium_peak_ssn", "value": f"{mid_amp:.2f}"},
        {"item": "cycle26_strong_peak_ssn", "value": f"{strong_amp:.2f}"},
        {"item": "cycle26_weighted_median_peak_month", "value": str(mid_peak_month)},
        {"item": "cycle26_weighted_median_length_months", "value": str(mid_length)},
        {"item": "cycle25_waldmeier_pred_peak_amp", "value": f"{diag['pred_peak_amp']:.2f}"},
        {"item": "cycle25_waldmeier_pred_peak_month", "value": f"{diag['pred_peak_month']:.1f}"},
        {"item": "cycle25_waldmeier_pred_length", "value": f"{diag['pred_length']:.1f}"},
    ])
    weights = pd.DataFrame({
        "candidate_predecessor_cycle": [p.cycle for p, s in candidate_pairs],
        "candidate_successor_cycle": [s.cycle for p, s in candidate_pairs],
        "distance": dist,
        "weight": w,
        "successor_peak_ssn": succ_peaks,
        "successor_peak_month": succ_peak_months,
        "successor_length": succ_lengths,
    }).sort_values("weight", ascending=False)
    return c25_forecast, sc26, scenario_summary, weights


def plot_cycle26_scenarios(df_all: pd.DataFrame, c25_forecast: pd.DataFrame, sc26: pd.DataFrame, cycles: Dict[int, Cycle], out_path: str):
    plt.figure(figsize=(14, 6))
    # historical recent observed
    recent = df_all[df_all["date"] >= pd.Timestamp("1996-01-01")]
    plt.plot(recent["date"], recent["ssn_smooth"], linewidth=2.4, label="Observed smoothed SSN")
    # c25 forecast after latest observation
    c25 = cycles[25]
    cf = c25_forecast[c25_forecast["date"] >= c25.dates.iloc[-1]]
    plt.plot(cf["date"], cf["pred"], linewidth=2.0, linestyle="--", label="Cycle 25 remaining forecast")
    plt.fill_between(cf["date"], cf["p10"], cf["p90"], alpha=0.18)
    for scen, g in sc26.groupby("scenario"):
        lw = 2.2 if scen == "medium" else 1.4
        ls = "-" if scen == "medium" else "--"
        plt.plot(g["date"], g["pred"], linewidth=lw, linestyle=ls, label=f"Cycle 26 {scen} scenario")
    plt.xlabel("Date")
    plt.ylabel("Smoothed monthly sunspot number")
    plt.title("Cycle 25 remaining trend and Solar Cycle 26 weak / medium / strong scenarios")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    ensure_dir(OUT_DIR)
    df, cycles, cycle_table = load_and_detect_cycles(INPUT_CSV)
    df.to_csv(os.path.join(OUT_DIR, "monthly_ssn_with_detected_cycle_phase.csv"), index=False, encoding="utf-8-sig")
    cycle_table.to_csv(os.path.join(OUT_DIR, "detected_solar_cycles.csv"), index=False, encoding="utf-8-sig")

    all_forecasts = []
    metric_rows = []
    peak_rows = []
    phase_rows = []
    weight_files = []
    diagnostic_rows = []

    for target_cycle in [24, 25]:
        # M1: before-start next-cycle forecast
        m1, w1 = predecessor_successor_forecast(cycles, target_cycle, total_months=FORECAST_TOTAL_MONTHS)
        m1o = merge_forecast_with_observed(m1, cycles, target_cycle)
        m1o["model_run"] = "M1 prior successor analog"
        all_forecasts.append(m1o)
        mt, pm, ph = evaluate_forecast(m1o, "M1 prior successor analog", target_cycle, 0, eval_future_only=False)
        metric_rows.append(mt); peak_rows.append(pm); phase_rows.append(ph)
        w1.to_csv(os.path.join(OUT_DIR, f"weights_cycle{target_cycle}_M1_prior_successor_analog.csv"), index=False, encoding="utf-8-sig")
        weight_files.append(f"weights_cycle{target_cycle}_M1_prior_successor_analog.csv")

        # M2/M3 early updates
        for obs_m in EARLY_OBS_MONTHS_LIST:
            if len(cycles[target_cycle].y) <= obs_m + 6:
                continue
            for calibrated in [False, True]:
                fc, ww, diag = early_shape_forecast(cycles, target_cycle, obs_m, total_months=FORECAST_TOTAL_MONTHS, calibrated=calibrated)
                fco = merge_forecast_with_observed(fc, cycles, target_cycle)
                label = ("M3 calibrated early shape" if calibrated else "M2 early shape analog") + f" ({obs_m}m)"
                fco["model_run"] = label
                all_forecasts.append(fco)
                mt, pm, ph = evaluate_forecast(fco, label, target_cycle, obs_m, eval_future_only=True)
                metric_rows.append(mt); peak_rows.append(pm); phase_rows.append(ph)
                tag = "M3_calibrated" if calibrated else "M2_early_shape"
                ww.to_csv(os.path.join(OUT_DIR, f"weights_cycle{target_cycle}_{tag}_{obs_m}m.csv"), index=False, encoding="utf-8-sig")
                diagnostic_rows.append({"cycle": target_cycle, "model": label, "obs_months_used": obs_m, **diag})

            # M4: parametric Waldmeier/Hathaway-like curve. It is often more stable for long decline forecasts.
            fc4, diag4 = parametric_waldmeier_forecast(cycles, target_cycle, obs_m, total_months=FORECAST_TOTAL_MONTHS)
            fc4o = merge_forecast_with_observed(fc4, cycles, target_cycle)
            label4 = f"M4 parametric Waldmeier curve ({obs_m}m)"
            fc4o["model_run"] = label4
            all_forecasts.append(fc4o)
            mt, pm, ph = evaluate_forecast(fc4o, label4, target_cycle, obs_m, eval_future_only=True)
            metric_rows.append(mt); peak_rows.append(pm); phase_rows.append(ph)
            diagnostic_rows.append({"cycle": target_cycle, "model": label4, "obs_months_used": obs_m, **diag4})

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecasts_df.to_csv(os.path.join(OUT_DIR, "cycle24_cycle25_all_validation_forecasts.csv"), index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(metric_rows).sort_values(["cycle", "MAE"])
    peaks_df = pd.DataFrame(peak_rows).sort_values(["cycle", "peak_amp_abs_error"])
    phases_df = pd.concat(phase_rows, ignore_index=True) if phase_rows else pd.DataFrame()
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "cycle24_cycle25_validation_metrics.csv"), index=False, encoding="utf-8-sig")
    peaks_df.to_csv(os.path.join(OUT_DIR, "cycle24_cycle25_peak_metrics.csv"), index=False, encoding="utf-8-sig")
    phases_df.to_csv(os.path.join(OUT_DIR, "cycle24_cycle25_phase_metrics.csv"), index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(os.path.join(OUT_DIR, "waldmeier_calibration_diagnostics.csv"), index=False, encoding="utf-8-sig")

    # comparison plot per target cycle
    for cyc in [24, 25]:
        sub = forecasts_df[forecasts_df["cycle"] == cyc]
        plot_validation_curve(sub, cyc, os.path.join(OUT_DIR, f"fig_validation_cycle{cyc}_all_methods.png"),
                              f"Solar Cycle {cyc} validation: prior forecast vs early-updated optimized forecasts")
        plot_residual(sub, cyc, os.path.join(OUT_DIR, f"fig_residual_cycle{cyc}_all_methods.png"),
                      f"Solar Cycle {cyc} validation residuals")
    plot_metric_bars(metrics_df, os.path.join(OUT_DIR, "fig_cycle24_25_metric_comparison.png"))

    # Cycle26 scenarios
    c25_forecast, sc26, scenario_summary, c26_weights = cycle26_scenarios(cycles, 25)
    c25_forecast.to_csv(os.path.join(OUT_DIR, "cycle25_remaining_forecast_from_current_observation.csv"), index=False, encoding="utf-8-sig")
    sc26.to_csv(os.path.join(OUT_DIR, "cycle26_weak_medium_strong_scenarios.csv"), index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(os.path.join(OUT_DIR, "cycle26_scenario_summary.csv"), index=False, encoding="utf-8-sig")
    c26_weights.to_csv(os.path.join(OUT_DIR, "cycle26_scenario_analog_weights.csv"), index=False, encoding="utf-8-sig")
    plot_cycle26_scenarios(df, c25_forecast, sc26, cycles, os.path.join(OUT_DIR, "fig_cycle25_remaining_cycle26_scenarios.png"))

    # Method selection table
    best_by_cycle = metrics_df.loc[metrics_df.groupby("cycle")["MAE"].idxmin()].copy()
    best_by_cycle.to_csv(os.path.join(OUT_DIR, "best_method_by_cycle.csv"), index=False, encoding="utf-8-sig")

    roadmap = pd.DataFrame([
        {"step": 1, "name": "Cycle 24/25 strict leave-out validation", "purpose": "用两个真实留出周期判断方法是否稳定，而不是只看 Cycle 25", "expected_gain": "发现弱周/强周下的失效模式"},
        {"step": 2, "name": "Predecessor-successor analog baseline", "purpose": "检验上一活动周是否能预测下一活动周", "expected_gain": "建立最低基线，通常会低估突增强周期"},
        {"step": 3, "name": "Early-shape update", "purpose": "加入目标周期前 36/48/60 个月观测", "expected_gain": "改善上升段后的剩余趋势和峰值位置"},
        {"step": 4, "name": "Waldmeier calibrated analog", "purpose": "用早期上升率校正峰值幅度和峰值时间", "expected_gain": "缓解峰值低估，是当前单变量 SSN 下最关键优化"},
        {"step": 5, "name": "Parametric Waldmeier/Hathaway curve", "purpose": "用带物理形态的偏斜周期曲线约束长期下降段", "expected_gain": "避免相似模板在下降段长期偏高，改善 Cycle 24 弱周期验证"},
        {"step": 6, "name": "Cycle 24 vs Cycle 25 error diagnosis", "purpose": "比较弱 Cycle 24 和偏强 Cycle 25 的峰值误差", "expected_gain": "明确模型是否系统性低估强周"},
        {"step": 7, "name": "Cycle 26 scenario forecast", "purpose": "不做武断单点预测，输出弱/中/强情景", "expected_gain": "结果更稳，更适合报告和论文"},
        {"step": 8, "name": "Add precursor variables", "purpose": "加入 F10.7、极区磁场、太阳黑子面积、aa/Ap 等", "expected_gain": "真正提高下一周峰值强度预报上限"},
    ])
    roadmap.to_csv(os.path.join(OUT_DIR, "optimization_roadmap.csv"), index=False, encoding="utf-8-sig")

    # Summary text
    lines = []
    lines.append("Solar Cycle Optimization Framework Summary")
    lines.append("=" * 70)
    lines.append("Detected cycles:")
    lines.append(cycle_table.to_string(index=False))
    lines.append("\nCycle 24 / Cycle 25 validation metrics sorted by MAE:")
    lines.append(metrics_df[["cycle", "model", "obs_months_used", "MAE", "RMSE", "R", "R2", "Bias", "MAPE", "N"]].to_string(index=False))
    lines.append("\nPeak metrics sorted by peak amplitude absolute error:")
    cols = [c for c in ["cycle", "model", "obs_months_used", "true_peak_date", "pred_peak_date", "peak_month_error", "true_peak_ssn", "pred_peak_ssn", "peak_amp_error", "peak_amp_abs_error"] if c in peaks_df.columns]
    lines.append(peaks_df[cols].to_string(index=False))
    lines.append("\nBest method by cycle:")
    lines.append(best_by_cycle[["cycle", "model", "obs_months_used", "MAE", "RMSE", "R", "R2", "Bias"]].to_string(index=False))
    lines.append("\nCycle 26 scenario summary:")
    lines.append(scenario_summary.to_string(index=False))
    lines.append("\nInterpretation:")
    lines.append("1. M1 is the strict prior forecast: it tests whether one can forecast a whole next cycle using only the previous cycle. This is intentionally hard and often underestimates strong cycles.")
    lines.append("2. M2 uses early observations of the target cycle, therefore it is a current-cycle update method. It should be judged mainly on future_only metrics after the observation window.")
    lines.append("3. M3 adds Waldmeier-effect style calibration, using early rise speed and early amplitude to correct peak amplitude, peak timing and cycle length. Under single-variable SSN constraints, this is the main optimization route.")
    lines.append("4. M4 uses a parametric cycle curve with Waldmeier priors and is especially useful for long decline forecasting.")
    lines.append("5. Cycle 24 and Cycle 25 must be compared together. If a method performs well on Cycle 24 but underestimates Cycle 25 peak, the error is likely related to strong-cycle amplitude calibration rather than ordinary curve fitting.")
    lines.append("6. Cycle 26 should be reported as weak/medium/strong scenarios unless additional precursor data such as polar field, F10.7, sunspot area, aa/Ap is included.")
    with open(os.path.join(OUT_DIR, "README_summary_and_interpretation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Zip package
    zip_path = os.path.join(os.path.dirname(OUT_DIR), "solar_cycle_optimization_framework_package.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(__file__, arcname="solar_cycle_optimization_framework.py")
        for root, dirs, files in os.walk(OUT_DIR):
            for file in files:
                full = os.path.join(root, file)
                arc = os.path.relpath(full, OUT_DIR)
                zf.write(full, arcname=arc)
    print("DONE")
    print("OUT_DIR:", OUT_DIR)
    print("ZIP:", zip_path)
    print(metrics_df[["cycle", "model", "obs_months_used", "MAE", "RMSE", "R", "R2", "Bias", "MAPE", "N"]].to_string(index=False))
    print("\nPeak metrics:")
    print(peaks_df[["cycle", "model", "obs_months_used", "true_peak_date", "pred_peak_date", "peak_month_error", "true_peak_ssn", "pred_peak_ssn", "peak_amp_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
