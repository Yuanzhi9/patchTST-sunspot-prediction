import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PHASE_LABELS = ['rise', 'peak', 'decline', 'trough']
PHASE_COLORS = {'rise': '#1f77b4', 'peak': '#d62728',
                'decline': '#2ca02c', 'trough': '#ff7f0e'}


def main():
    df_a = pd.read_csv('output/tagged_cycles_a.csv')
    df_b = pd.read_csv('/tmp/patchTST-method-b/phase_analysis/output/tagged_cycles_b.csv')

    print("=" * 60)
    print("  方法A vs 方法B 对比分析")
    print("=" * 60)

    print("\n--- 各阶段总月数对比 ---")
    for label in PHASE_LABELS:
        na = int((df_a['phase'] == label).sum())
        nb = int((df_b['phase'] == label).sum())
        print(f"  {label:8s}:  A={na:4d}  B={nb:4d}")

    print("\n--- 各周期各阶段长度 ---")
    cycles = sorted(df_a['cycle'].unique())
    for c in cycles:
        row_a = {l: int(((df_a['cycle'] == c) & (df_a['phase'] == l)).sum())
                 for l in PHASE_LABELS}
        row_b = {l: int(((df_b['cycle'] == c) & (df_b['phase'] == l)).sum())
                 for l in PHASE_LABELS}
        print(f"  周期 {int(c):2d}: "
              f"A rise={row_a['rise']:3d} peak={row_a['peak']:3d} "
              f"decl={row_a['decline']:3d} tro={row_a['trough']:3d}  |  "
              f"B rise={row_b['rise']:3d} peak={row_b['peak']:3d} "
              f"decl={row_b['decline']:3d} tro={row_b['trough']:3d}")

    print("\n--- 各周期阶段长度标准差 ---")
    for c in cycles:
        row_a = [int(((df_a['cycle'] == c) & (df_a['phase'] == l)).sum())
                 for l in PHASE_LABELS]
        row_b = [int(((df_b['cycle'] == c) & (df_b['phase'] == l)).sum())
                 for l in PHASE_LABELS]
        print(f"  周期 {int(c):2d}: A std={np.std(row_a):.1f}  B std={np.std(row_b):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cycles_arr = [int(c) for c in cycles]
    x = np.arange(len(cycles_arr))
    width = 0.35

    for idx, label in enumerate(['rise', 'peak']):
        a_vals = [int(((df_a['cycle'] == c) & (df_a['phase'] == label)).sum()) for c in cycles]
        b_vals = [int(((df_b['cycle'] == c) & (df_b['phase'] == label)).sum()) for c in cycles]
        axes[0].bar(x + width * idx - width / 2, a_vals, width, 
                   label=f'A-{label}', alpha=0.7, color=PHASE_COLORS[label])
        axes[0].bar(x + width * idx - width / 2 + width * 2, b_vals, width,
                   label=f'B-{label}', alpha=0.7, color=PHASE_COLORS[label], hatch='//')

    for idx, label in enumerate(['decline', 'trough']):
        a_vals = [int(((df_a['cycle'] == c) & (df_a['phase'] == label)).sum()) for c in cycles]
        b_vals = [int(((df_b['cycle'] == c) & (df_b['phase'] == label)).sum()) for c in cycles]
        axes[0].bar(x + width * (idx + 2) - width / 2, a_vals, width,
                   label=f'A-{label}', alpha=0.7, color=PHASE_COLORS[label])
        axes[0].bar(x + width * (idx + 2) - width / 2 + width * 2, b_vals, width,
                   label=f'B-{label}', alpha=0.7, color=PHASE_COLORS[label], hatch='//')

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(cycles_arr)
    axes[0].set_xlabel('Solar Cycle')
    axes[0].set_ylabel('Number of Months')
    axes[0].set_title('Phase Length by Cycle')
    axes[0].legend(bbox_to_anchor=(1.02, 1), fontsize=7)

    a_totals = [int((df_a['phase'] == l).sum()) for l in PHASE_LABELS]
    b_totals = [int((df_b['phase'] == l).sum()) for l in PHASE_LABELS]
    x2 = np.arange(4)
    axes[1].bar(x2 - width / 2, a_totals, width, label='Method A', alpha=0.8)
    axes[1].bar(x2 + width / 2, b_totals, width, label='Method B', alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(PHASE_LABELS)
    axes[1].set_ylabel('Total Months')
    axes[1].set_title('Total Phase Distribution')
    axes[1].legend()

    a_stds = [np.std([int(((df_a['cycle'] == c) & (df_a['phase'] == l)).sum())
                     for c in cycles]) for l in PHASE_LABELS]
    b_stds = [np.std([int(((df_b['cycle'] == c) & (df_b['phase'] == l)).sum())
                     for c in cycles]) for l in PHASE_LABELS]
    axes[2].bar(x2 - width / 2, a_stds, width, label='Method A', alpha=0.8)
    axes[2].bar(x2 + width / 2, b_stds, width, label='Method B', alpha=0.8)
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(PHASE_LABELS)
    axes[2].set_ylabel('Std Dev of Months across Cycles')
    axes[2].set_title('Phase Length Variability')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('output/comparison_a_vs_b.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[对比图已保存] output/comparison_a_vs_b.png")

    summary_a = pd.read_csv('output/method_a_summary.csv')
    summary_b = pd.read_csv('/tmp/patchTST-method-b/phase_analysis/output/method_b_summary.csv')

    print("\n--- 方法A 全局评分 ---")
    print(f"  总周期数: {len(summary_a)}")
    print(f"  平均周期长度: {summary_a['n_months'].mean():.0f} 月")
    print(f"  各阶段平均长度: rise={summary_a['rise_n'].mean():.0f} peak={summary_a['peak_n'].mean():.0f} decline={summary_a['decline_n'].mean():.0f} trough={summary_a['trough_n'].mean():.0f}")
    print(f"  阶段长度标准差(均值): {summary_a['std'].mean():.1f}")
    print(f"  t_high 均值: {summary_a['t_high'].mean():.2f}")
    print(f"  t_low 均值: {summary_a['t_low'].mean():.2f}")

    print("\n--- 方法B 全局评分 ---")
    print(f"  总周期数: {len(summary_b)}")
    print(f"  平均周期长度: {summary_b['n_months'].mean():.0f} 月")
    print(f"  各阶段平均长度: rise={summary_b['rise_n'].mean():.0f} peak={summary_b['peak_n'].mean():.0f} decline={summary_b['decline_n'].mean():.0f} trough={summary_b['trough_n'].mean():.0f}")
    print(f"  阶段长度标准差(均值): {summary_b['std'].mean():.1f}")

    print("\n=== 对比分析完成 ===")


if __name__ == '__main__':
    main()
