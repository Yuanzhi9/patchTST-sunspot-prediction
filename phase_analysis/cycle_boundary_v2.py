import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

SMOOTH_WINDOW = 13
MIN_CYCLE_LENGTH = 60
MAX_CYCLE_LENGTH = 180
MAIN_PEAK_PROMINENCE = 20
SECONDARY_PEAK_RATIO = 0.82

SILSO_MINIMA = {
    11: (1867, 2), 12: (1878, 9), 13: (1890, 2), 14: (1902, 1),
    15: (1913, 7), 16: (1923, 8), 17: (1933, 9), 18: (1944, 2),
    19: (1954, 4), 20: (1964, 10), 21: (1976, 6), 22: (1986, 9),
    23: (1996, 8),
}

SILSO_MAXIMA = {
    11: (1870, 8), 12: (1883, 12), 13: (1894, 1), 14: (1907, 2),
    15: (1917, 8), 16: (1928, 4), 17: (1937, 4), 18: (1947, 5),
    19: (1958, 3), 20: (1968, 11), 21: (1979, 12), 22: (1989, 11),
    23: (2001, 11),
}


@dataclass
class Cycle:
    cycle_num: int
    start_idx: int
    end_idx: int
    peak_idx: int
    start_ssn: float
    end_ssn: float
    peak_ssn: float
    start_date: str
    end_date: str
    peak_date: str
    n_months: int
    secondary_peaks: List[int] = field(default_factory=list)
    secondary_peak_ssns: List[float] = field(default_factory=list)
    silso_min_date: str = ''
    silso_max_date: str = ''


def load_data(csv_path: str, start_year: int = 1867, end_year: int = 2008) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)].copy()
    df = df.reset_index(drop=True)
    return df


def smooth_ssn(ssn: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(ssn) < window:
        return ssn.copy()
    return uniform_filter1d(ssn.astype(float), size=window, mode='nearest')


def find_cycle_minima(smoothed: np.ndarray, ssn_raw: np.ndarray) -> np.ndarray:
    inverted = -smoothed
    peaks, _ = find_peaks(inverted, distance=MIN_CYCLE_LENGTH, prominence=MAIN_PEAK_PROMINENCE)
    if len(peaks) == 0:
        peaks, _ = find_peaks(inverted, distance=40)
    refined = []
    for p in peaks:
        lo = max(0, p - 3)
        hi = min(len(ssn_raw), p + 4)
        refined.append(int(lo + np.argmin(ssn_raw[lo:hi])))
    if ssn_raw[0] < 5 and (len(refined) == 0 or refined[0] > 6):
        refined.insert(0, 0)
    tail_len = min(60, len(ssn_raw))
    tail_min_rel = int(np.argmin(smoothed[-tail_len:]))
    tail_min_abs = len(smoothed) - tail_len + tail_min_rel
    lo = max(0, tail_min_abs - 6)
    hi = min(len(ssn_raw), tail_min_abs + 7)
    local_min = int(lo + np.argmin(ssn_raw[lo:hi]))
    if local_min > refined[-1] + 60:
        refined.append(local_min)
    return np.array(sorted(set(refined)))


def find_cycle_maxima(smoothed: np.ndarray, ssn_raw: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(smoothed, distance=MIN_CYCLE_LENGTH, prominence=MAIN_PEAK_PROMINENCE)
    if len(peaks) == 0:
        peaks, _ = find_peaks(smoothed, distance=40)
    refined = []
    for p in peaks:
        lo = max(0, p - 6)
        hi = min(len(ssn_raw), p + 7)
        refined.append(int(lo + np.argmax(ssn_raw[lo:hi])))
    return np.array(sorted(set(refined)))


def find_secondary_peaks(ssn_raw: np.ndarray, start_idx: int, end_idx: int,
                          main_peak_idx: int, main_peak_ssn: float) -> Tuple[List[int], List[float]]:
    segment = ssn_raw[start_idx:end_idx + 1].astype(float)
    peaks, props = find_peaks(segment, prominence=8, distance=24)
    secondary = []
    ssn_vals = []
    threshold = SECONDARY_PEAK_RATIO * main_peak_ssn
    for p in peaks:
        abs_idx = start_idx + p
        ssn_val = ssn_raw[abs_idx]
        if abs_idx != main_peak_idx and abs_idx != start_idx and abs_idx != end_idx:
            if ssn_val >= threshold and abs(abs_idx - main_peak_idx) >= 20:
                secondary.append(int(abs_idx))
                ssn_vals.append(float(ssn_val))
    return secondary, ssn_vals


def build_cycles(minima: np.ndarray, maxima: np.ndarray,
                 ssn_raw: np.ndarray, dates: pd.Series,
                 start_cycle_num: int = 11) -> List[Cycle]:
    cycles = []
    max_pos = 0
    for i in range(len(minima) - 1):
        start_idx = minima[i]
        end_idx = minima[i + 1]
        while max_pos < len(maxima) and maxima[max_pos] <= start_idx:
            max_pos += 1
        if max_pos >= len(maxima):
            break
        peak_idx = maxima[max_pos]
        if peak_idx >= end_idx:
            while max_pos + 1 < len(maxima) and maxima[max_pos + 1] < end_idx:
                max_pos += 1
                peak_idx = maxima[max_pos]
            if peak_idx >= end_idx:
                continue
        n_months = end_idx - start_idx + 1
        if n_months < MIN_CYCLE_LENGTH or n_months > MAX_CYCLE_LENGTH:
            continue
        second_idx, second_ssn = find_secondary_peaks(
            ssn_raw, start_idx, end_idx, peak_idx, float(ssn_raw[peak_idx]))
        cnum = start_cycle_num + len(cycles)
        silso_min = SILSO_MINIMA.get(cnum, ('?', '?'))
        silso_max = SILSO_MAXIMA.get(cnum, ('?', '?'))
        cycle = Cycle(
            cycle_num=cnum,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            peak_idx=int(peak_idx),
            start_ssn=float(ssn_raw[start_idx]),
            end_ssn=float(ssn_raw[end_idx]),
            peak_ssn=float(ssn_raw[peak_idx]),
            start_date=str(dates.iloc[start_idx].date()),
            end_date=str(dates.iloc[end_idx].date()),
            peak_date=str(dates.iloc[peak_idx].date()),
            n_months=int(end_idx - start_idx + 1),
            secondary_peaks=second_idx,
            secondary_peak_ssns=second_ssn,
            silso_min_date=f"{silso_min[0]}-{silso_min[1]:02d}-01" if isinstance(silso_min[0], int) else '?',
            silso_max_date=f"{silso_max[0]}-{silso_max[1]:02d}-01" if isinstance(silso_max[0], int) else '?',
        )
        cycles.append(cycle)
    return cycles


def find_cycles(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, List[Cycle]]:
    df = load_data(csv_path)
    ssn_raw = df['ssn'].values
    dates = df['date']
    smoothed = smooth_ssn(ssn_raw)
    minima = find_cycle_minima(smoothed, ssn_raw)
    maxima = find_cycle_maxima(smoothed, ssn_raw)
    cycles = build_cycles(minima, maxima, ssn_raw, dates)
    return df, smoothed, cycles
