import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from cycle_boundary import find_cycles, Cycle

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = '../PatchTST_supervised/dataset/sunspot_monthly_clean.csv'
OUTPUT_DIR = 'output'

PHASE_LABELS = ['rise', 'peak', 'decline', 'trough']
PHASE_COLORS = {'rise': '#1f77b4', 'peak': '#d62728',
                'decline': '#2ca02c', 'trough': '#ff7f0e'}

TH_HIGH_RANGE = np.arange(0.40, 0.91, 0.05)
TH_LOW_RANGE = np.arange(0.05, 0.36, 0.05)


def assign_phases_method_a(ssn_series, start_idx, end_idx, peak_idx,
                           peak_ssn, min_start_ssn, t_high, t_low):
    n = end_idx - start_idx + 1
    phases = np.full(n, '', dtype=object)
    ssn_cycle = ssn_series[start_idx:end_idx + 1]

    high_thresh = t_high * peak_ssn
    low_thresh = min_start_ssn + t_low * (peak_ssn - min_start_ssn)

    for i in range(n):
        s = ssn_cycle[i]
        if s >= high_thresh:
            phases[i] = 'peak'
        elif s <= low_thresh:
            phases[i] = 'trough'
        elif i < (peak_idx - start_idx):
            phases[i] = 'rise'
        else:
            phases[i] = 'decline'

    return phases


def score_balance(counts):
    if any(c == 0 for c in counts):
        return 1e9
    return np.std(counts)


def optimize_thresholds(ssn_series, start_idx, end_idx, peak_idx,
                        peak_ssn, min_start_ssn):
    best_score = 1e9
    best_params = (0.50, 0.15)
    best_phases = None

    for th in TH_HIGH_RANGE:
        for tl in TH_LOW_RANGE:
            phases = assign_phases_method_a(ssn_series, start_idx, end_idx,
                                            peak_idx, peak_ssn, min_start_ssn,
                                            th, tl)
            counts = {label: int(np.sum(phases == label)) for label in PHASE_LABELS}
            score = score_balance(list(counts.values()))
            if score < best_score:
                best_score = score
                best_params = (th, tl)
                best_phases = phases
                best_counts = counts

    return best_params, best_phases, best_counts, best_score


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== 方法A：自适应百分比阈值 阶段划分 ===\n")

    df, smoothed, cycles = find_cycles(CSV_PATH)

    df['cycle'] = 0
    df['phase'] = ''
    df['is_split_point'] = 0

    fig, axes = plt.subplots(5, 3, figsize=(20, 25))
    axes = axes.flatten()

    all_results = []

    for i, c in enumerate(cycles):
        start = c.start_idx
        end = c.end_idx
        peak = c.peak_idx

        ssn_cycle = df['ssn'].values[start:end + 1]
        best_params, phases, counts, score = optimize_thresholds(
            df['ssn'].values, start, end, peak,
            c.peak_ssn, c.start_ssn)

        t_high, t_low = best_params

        for j in range(len(phases)):
            idx = start + j
            df.at[idx, 'cycle'] = c.cycle_num
            df.at[idx, 'phase'] = phases[j]

        df.at[start, 'is_split_point'] = 1

        reordered_counts = [counts.get(l, 0) for l in PHASE_LABELS]

        print(f"  周期 {c.cycle_num:2d}: t_high={t_high:.2f} t_low={t_low:.2f}  "
              f"rise={reordered_counts[0]:3d} peak={reordered_counts[1]:3d} "
              f"decline={reordered_counts[2]:3d} trough={reordered_counts[3]:3d}  "
              f"std={score:.1f}  峰值={c.peak_ssn:.0f}")

        all_results.append({
            'cycle': c.cycle_num,
            't_high': t_high,
            't_low': t_low,
            'rise_n': reordered_counts[0],
            'peak_n': reordered_counts[1],
            'decline_n': reordered_counts[2],
            'trough_n': reordered_counts[3],
            'std': score,
            'n_months': c.n_months,
            'peak_ssn': c.peak_ssn,
            'start_ssn': c.start_ssn,
        })

        ax = axes[i]
        x = np.arange(start - c.start_idx, end - c.start_idx + 1)
        for label in PHASE_LABELS:
            mask = phases == label
            if mask.any():
                ax.scatter(x[mask], ssn_cycle[mask], c=PHASE_COLORS[label],
                          s=15, label=label, alpha=0.8)

        ax.axvline(x=peak - start, color='black', linestyle='--', alpha=0.4,
                   label=f'peak {c.peak_ssn:.0f}')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=end - start, color='gray', linestyle=':', alpha=0.3)
        ax.set_title(f'Cycle {c.cycle_num} ({c.start_date} ~ {c.end_date})\n'
                     f"t_high={t_high:.2f} t_low={t_low:.2f}  "
                     f"rise={reordered_counts[0]} peak={reordered_counts[1]} "
                     f"decl={reordered_counts[2]} tro={reordered_counts[3]}",
                     fontsize=9)
        ax.set_xlabel('Months from cycle start')
        ax.set_ylabel('SSN')

    for j in range(len(cycles), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    fig.suptitle('Method A: Adaptive Percentage Threshold Phase Division',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/method_a_cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[图已保存] {OUTPUT_DIR}/method_a_cycles.png")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f'{OUTPUT_DIR}/method_a_summary.csv', index=False)
    print(f"[摘要已保存] {OUTPUT_DIR}/method_a_summary.csv")

    cols = ['date', 'ssn', 'cycle', 'phase', 'is_split_point']
    tagged = df.loc[:, cols].copy()
    tagged = tagged[tagged['cycle'] > 0]
    tagged['phase'] = tagged['phase'].astype(str)
    tagged.to_csv(f'{OUTPUT_DIR}/tagged_cycles_a.csv', index=False)
    print(f"[标记数据已保存] {OUTPUT_DIR}/tagged_cycles_a.csv "
          f"({len(tagged)} 行, {tagged['cycle'].nunique()} 个周期)")

    for label in PHASE_LABELS:
        n = int(tagged[tagged['phase'] == label].shape[0])
        print(f"  {label:8s}: {n:4d} 个月")

    print("\n=== 方法A 完成 ===")


if __name__ == '__main__':
    main()
