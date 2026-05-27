import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple


SMOOTH_WINDOW = 13
MIN_CYCLE_LENGTH = 60   
MAX_CYCLE_LENGTH = 180  
PEAK_PROMINENCE = 20    


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


def load_data(csv_path: str, start_year: int = 1867, end_year: int = 2008) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)].copy()
    df = df.reset_index(drop=True)
    return df


def smooth_ssn(ssn: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(ssn) < window:
        return ssn.copy()
    smoothed = uniform_filter1d(ssn.astype(float), size=window, mode='nearest')
    return smoothed


def find_cycle_minima(smoothed: np.ndarray, ssn_raw: np.ndarray) -> np.ndarray:
    inverted = -smoothed
    peaks, props = find_peaks(inverted, distance=MIN_CYCLE_LENGTH, prominence=PEAK_PROMINENCE)

    if len(peaks) == 0:
        peaks, props = find_peaks(inverted, distance=40)

    refined = []
    for p in peaks:
        search_start = max(0, p - 3)
        search_end = min(len(ssn_raw), p + 4)
        local_min = search_start + np.argmin(ssn_raw[search_start:search_end])
        refined.append(local_min)

    if ssn_raw[0] < 5 and (len(refined) == 0 or refined[0] > 6):
        refined.insert(0, 0)

    tail_len = min(60, len(ssn_raw))
    tail_smoothed = smoothed[-tail_len:]
    tail_min_rel = int(np.argmin(tail_smoothed))
    tail_min_abs = len(smoothed) - tail_len + tail_min_rel

    search_start = max(0, tail_min_abs - 6)
    search_end = min(len(ssn_raw), tail_min_abs + 7)
    local_min = int(search_start + np.argmin(ssn_raw[search_start:search_end]))

    if local_min > refined[-1] + 60:
        refined.append(local_min)

    return np.array(sorted(set(refined)))


def find_cycle_maxima(smoothed: np.ndarray, ssn_raw: np.ndarray) -> np.ndarray:
    peaks, props = find_peaks(smoothed, distance=MIN_CYCLE_LENGTH, prominence=PEAK_PROMINENCE)

    if len(peaks) == 0:
        peaks, props = find_peaks(smoothed, distance=40)

    refined = []
    for p in peaks:
        search_start = max(0, p - 6)
        search_end = min(len(ssn_raw), p + 7)
        local_max = search_start + np.argmax(ssn_raw[search_start:search_end])
        refined.append(local_max)
    return np.array(sorted(set(refined)))


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

        cycle = Cycle(
            cycle_num=start_cycle_num + len(cycles),
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
