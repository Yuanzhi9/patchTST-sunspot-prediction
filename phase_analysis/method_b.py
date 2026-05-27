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


def assign_phases_method_b(n_months, peak_idx_in_cycle):
    base = n_months // 4
    remainder = n_months % 4

    boundaries = [0]
    for k in range(4):
        extra = 1 if k < remainder else 0
        boundaries.append(boundaries[-1] + base + extra)

    if peak_idx_in_cycle < boundaries[1]:
        shift = boundaries[1] - peak_idx_in_cycle
        for k in range(1, 4):
            boundaries[k] += shift
        boundaries[4] += shift
        for k in range(4):
            boundaries[k] = max(0, min(n_months, boundaries[k]))
        boundaries = sorted(set(boundaries))
        if len(boundaries) < 5:
            boundaries = [0, n_months // 4, 2 * n_months // 4,
                          3 * n_months // 4, n_months]

    phases = np.full(n_months, '', dtype=object)
    for i in range(n_months):
        if i < boundaries[1]:
            phases[i] = 'rise'
        elif i < boundaries[2]:
            phases[i] = 'peak'
        elif i < boundaries[3]:
            phases[i] = 'decline'
        else:
            phases[i] = 'trough'

    return phases


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== 方法B：固定等分 + 峰值锚定 阶段划分 ===\n")

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
        n_months = c.n_months
        peak_in_cycle = peak - start

        phases = assign_phases_method_b(n_months, peak_in_cycle)

        for j in range(n_months):
            idx = start + j
            df.at[idx, 'cycle'] = c.cycle_num
            df.at[idx, 'phase'] = phases[j]

        df.at[start, 'is_split_point'] = 1

        counts = {label: int(np.sum(phases == label)) for label in PHASE_LABELS}
        std_val = float(np.std(list(counts.values())))
        reordered = [counts.get(l, 0) for l in PHASE_LABELS]

        print(f"  周期 {c.cycle_num:2d}: rise={reordered[0]:3d} peak={reordered[1]:3d} "
              f"decline={reordered[2]:3d} trough={reordered[3]:3d}  "
              f"std={std_val:.1f}  峰值SSN={c.peak_ssn:.0f}")

        all_results.append({
            'cycle': c.cycle_num,
            'rise_n': reordered[0],
            'peak_n': reordered[1],
            'decline_n': reordered[2],
            'trough_n': reordered[3],
            'std': std_val,
            'n_months': n_months,
            'peak_ssn': c.peak_ssn,
            'peak_idx_in_cycle': peak_in_cycle,
        })

        ax = axes[i]
        ssn_cycle = df['ssn'].values[start:end + 1]
        x = np.arange(n_months)
        for label in PHASE_LABELS:
            mask = phases == label
            if mask.any():
                ax.scatter(x[mask], ssn_cycle[mask], c=PHASE_COLORS[label],
                          s=15, label=label, alpha=0.8)

        ax.axvline(x=peak_in_cycle, color='black', linestyle='--', alpha=0.4)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=n_months - 1, color='gray', linestyle=':', alpha=0.3)
        ax.set_title(f'Cycle {c.cycle_num} ({c.start_date} ~ {c.end_date})\n'
                     f"rise={reordered[0]} peak={reordered[1]} "
                     f"decl={reordered[2]} tro={reordered[3]}",
                     fontsize=9)
        ax.set_xlabel('Months from cycle start')
        ax.set_ylabel('SSN')

    for j in range(len(cycles), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    fig.suptitle('Method B: Equal Time Division with Peak Anchoring',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/method_b_cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[图已保存] {OUTPUT_DIR}/method_b_cycles.png")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f'{OUTPUT_DIR}/method_b_summary.csv', index=False)
    print(f"[摘要已保存] {OUTPUT_DIR}/method_b_summary.csv")

    cols = ['date', 'ssn', 'cycle', 'phase', 'is_split_point']
    tagged = df.loc[:, cols].copy()
    tagged = tagged[tagged['cycle'] > 0]
    tagged['phase'] = tagged['phase'].astype(str)
    tagged.to_csv(f'{OUTPUT_DIR}/tagged_cycles_b.csv', index=False)
    print(f"[标记数据已保存] {OUTPUT_DIR}/tagged_cycles_b.csv "
          f"({len(tagged)} 行, {tagged['cycle'].nunique()} 个周期)")

    for label in PHASE_LABELS:
        n = int(tagged[tagged['phase'] == label].shape[0])
        print(f"  {label:8s}: {n:4d} 个月")

    print("\n=== 方法B 完成 ===")


if __name__ == '__main__':
    main()
