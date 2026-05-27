import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from cycle_boundary_v2 import find_cycles, SILSO_MINIMA, SILSO_MAXIMA

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = '../PatchTST_supervised/dataset/sunspot_monthly_clean.csv'
OUTPUT_DIR = 'output'

PHASE_LABELS = ['rise', 'peak', 'decline', 'trough']
PHASE_COLORS = {'rise': '#1f77b4', 'peak': '#d62728',
                'decline': '#2ca02c', 'trough': '#ff7f0e'}

TH_HIGH_RANGE = np.arange(0.35, 0.91, 0.02)
TH_LOW_RANGE = np.arange(0.03, 0.31, 0.02)


def all_peak_indices(cycle):
    indices = [cycle.peak_idx] + cycle.secondary_peaks
    return sorted(set(indices))


def assign_phases_method_a(ssn_series, start_idx, end_idx, peak_indices,
                           peak_ssn, min_start_ssn, t_high, t_low):
    n = end_idx - start_idx + 1
    phases = np.full(n, '', dtype=object)
    ssn_cycle = ssn_series[start_idx:end_idx + 1]

    high_thresh = t_high * peak_ssn
    low_thresh = min_start_ssn + t_low * (peak_ssn - min_start_ssn)

    peak_start = min(peak_indices) - start_idx
    peak_end = max(peak_indices) - start_idx

    for i in range(n):
        s = ssn_cycle[i]
        if s >= high_thresh:
            phases[i] = 'peak'
        elif s <= low_thresh:
            phases[i] = 'trough'
        elif i < peak_start:
            phases[i] = 'rise'
        elif i > peak_end:
            phases[i] = 'decline'
        else:
            phases[i] = 'peak'

    return phases


def validate_phase_continuity(phases, cycle_num):
    saw_start = False
    saw_peak = False
    saw_decline = False
    errors = []
    for i, p in enumerate(phases):
        if p == 'rise' and not saw_start:
            saw_start = True
        elif p == 'peak' and saw_start and not saw_decline:
            saw_peak = True
        elif p == 'decline' and saw_peak:
            saw_decline = True
        elif p == 'trough':
            if saw_decline:
                pass
        elif p == '':
            pass
    if not saw_start:
        errors.append('no rise')
    if not saw_peak:
        errors.append('no peak')
    if not saw_decline:
        errors.append('no decline')
    return errors


def score_balance(counts):
    if any(c == 0 or c is None for c in counts):
        return 1e9
    return float(np.std(counts))


def optimize_thresholds(ssn_series, start_idx, end_idx, peak_indices,
                        peak_ssn, min_start_ssn):
    best_score = 1e9
    best_params = (0.50, 0.15)
    best_phases = None
    best_counts = {}
    for th in TH_HIGH_RANGE:
        for tl in TH_LOW_RANGE:
            phases = assign_phases_method_a(ssn_series, start_idx, end_idx,
                                            peak_indices, peak_ssn, min_start_ssn, th, tl)
            counts = {label: int(np.sum(phases == label)) for label in PHASE_LABELS}
            score = score_balance(list(counts.values()))
            if score < best_score:
                best_score = score
                best_params = (th, tl)
                best_phases = phases
                best_counts = counts
    return best_params, best_phases, best_counts, best_score


def month_diff(d1_str, d2_str):
    try:
        d1 = pd.to_datetime(d1_str)
        d2 = pd.to_datetime(d2_str)
        return abs((d1.year - d2.year) * 12 + (d1.month - d2.month))
    except Exception:
        return 999


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/cycles', exist_ok=True)

    print("=" * 70)
    print("  方法A v2：精细阈值 + 双峰处理 + SILSO外部对照")
    print("=" * 70)

    df, smoothed, cycles = find_cycles(CSV_PATH)

    df['cycle'] = 0
    df['phase'] = ''
    df['is_split_point'] = 0

    all_results = []
    continuity_issues = []

    for i, c in enumerate(cycles):
        start, end = c.start_idx, c.end_idx
        peak_indices = all_peak_indices(c)

        ssn_cycle = df['ssn'].values[start:end + 1]
        best_params, phases, counts, score = optimize_thresholds(
            df['ssn'].values, start, end, peak_indices,
            c.peak_ssn, c.start_ssn)

        t_high, t_low = best_params
        for j in range(len(phases)):
            idx = start + j
            df.at[idx, 'cycle'] = c.cycle_num
            df.at[idx, 'phase'] = phases[j]
        df.at[start, 'is_split_point'] = 1

        errors = validate_phase_continuity(phases, c.cycle_num)
        if errors:
            continuity_issues.append((c.cycle_num, errors))

        reordered = [counts.get(l, 0) for l in PHASE_LABELS]
        min_dev = month_diff(c.start_date, c.silso_min_date)
        peak_dev = month_diff(c.peak_date, c.silso_max_date)

        print(f"  周期 {c.cycle_num:2d}: t_high={t_high:.2f} t_low={t_low:.2f}  "
              f"rise={reordered[0]:3d} peak={reordered[1]:3d} "
              f"decline={reordered[2]:3d} trough={reordered[3]:3d}  "
              f"std={score:.1f}  "
              f"最小值偏差={min_dev}月  峰值偏差={peak_dev}月")

        if c.secondary_peaks:
            sec_str = ', '.join([f"{df['ssn'].values[p]:.0f}({df['date'].iloc[p].date()})"
                                 for p in c.secondary_peaks])
            print(f"         次峰: {sec_str}")

        all_results.append({
            'cycle': c.cycle_num,
            't_high': t_high, 't_low': t_low,
            'rise_n': reordered[0], 'peak_n': reordered[1],
            'decline_n': reordered[2], 'trough_n': reordered[3],
            'std': score, 'n_months': c.n_months, 'peak_ssn': c.peak_ssn,
            'start_ssn': c.start_ssn, 'end_ssn': c.end_ssn,
            'silso_min_dev_months': min_dev,
            'silso_peak_dev_months': peak_dev,
            'n_secondary_peaks': len(c.secondary_peaks),
            'start_date': c.start_date, 'end_date': c.end_date,
            'peak_date': c.peak_date,
        })

    if continuity_issues:
        print(f"\n[WARNING] 相位连续性问题 ({len(continuity_issues)} 个周期):")
        for cnum, errs in continuity_issues:
            print(f"  周期 {cnum}: {errs}")

    min_devs = [r['silso_min_dev_months'] for r in all_results]
    peak_devs = [r['silso_peak_dev_months'] for r in all_results]
    print(f"\n[SILSO对照] 最小值偏差: 均值={np.mean(min_devs):.1f}月  max={max(min_devs)}月")
    print(f"[SILSO对照] 峰值偏差: 均值={np.mean(peak_devs):.1f}月  max={max(peak_devs)}月")

    tagged = df.loc[:, ['date', 'ssn', 'cycle', 'phase', 'is_split_point']].copy()
    tagged = tagged[tagged['cycle'] > 0]
    tagged.to_csv(f'{OUTPUT_DIR}/tagged_cycles_a_v2.csv', index=False)
    print(f"\n[标记数据] {OUTPUT_DIR}/tagged_cycles_a_v2.csv  ({len(tagged)}行, {tagged['cycle'].nunique()}周期)")

    for label in PHASE_LABELS:
        n = int((tagged['phase'] == label).sum())
        print(f"  {label:8s}: {n:4d} 个月")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f'{OUTPUT_DIR}/method_a_v2_summary.csv', index=False)

    n = len(cycles)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    for i, c in enumerate(cycles):
        start, end = c.start_idx, c.end_idx
        peak_indices = all_peak_indices(c)
        ssn_cycle = df['ssn'].values[start:end + 1]
        phases_local = df['phase'].values[start:end + 1]
        n_local = end - start + 1
        x = np.arange(n_local)

        ax = axes[i]
        for label in PHASE_LABELS:
            mask = phases_local == label
            if mask.any():
                ax.scatter(x[mask], ssn_cycle[mask], c=PHASE_COLORS[label],
                          s=10, label=label, alpha=0.8)
        for pi in peak_indices:
            ax.axvline(x=pi - start, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=n_local - 1, color='gray', linestyle=':', alpha=0.3)
        row = all_results[i]
        ax.set_title(f'Cycle {c.cycle_num} ({c.start_date}~{c.end_date})\n'
                     f"th={row['t_high']:.2f} tl={row['t_low']:.2f}  "
                     f"R{row['rise_n']} P{row['peak_n']} D{row['decline_n']} T{row['trough_n']}",
                     fontsize=8)
        ax.set_xlabel('Months from cycle start')
        ax.set_ylabel('SSN')

    for j in range(len(cycles), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    fig.suptitle('Method A v2: Refined Threshold + Double Peak Handling',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/method_a_v2_cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] {OUTPUT_DIR}/method_a_v2_cycles.png")

    print("\n=== 方法A v2 完成 ===")


if __name__ == '__main__':
    main()
