"""EXP-25 新环境基线 — 预测 vs 真实对比图（W1 窗口，全步口径）

故事问题：新环境重建的基线，模型在 W1 回测窗口上预测得怎么样？误差随提前期如何退化？

铁律遵守（figure_guide.md §2）：
  1) 每张图回答一个故事问题 —— 见上
  2) 数据源=对应 checkpoint 的推理产物（补测段 pred/true.npy，即最佳 val 模型推理）；
     **图内 MAE 与 result.txt 已记值逐位一致才出图**，不一致直接 abort
  3) 无 target_transform（本次为空），不触发坑 25
  4) 色盲友好色板：Okabe-Ito 蓝/橙/灰，弃红绿
  5) caption 自包含：口径/窗口/种子/指标定义/单种子警示
  6) single seed 图内标注
  7) 学术风：白底/英文/dpi300/字号 9pt/仅横向浅灰网格/去顶右边框

用法：python scripts/plot_exp25_baseline.py
输出：plots/fig_EXP25_baseline_newenv_2026-10-10.png
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

RESULT_DIR = ('results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2'
              '_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-25_0')
DATA_CSV = 'PatchTST_supervised/dataset/sunspot_with_cycle.csv'
NUM_TRAIN = 2838          # W1 窗口（data_pipeline.md 三窗口表）
NUM_VAL = 132
EXPECTED = {'full_MAE': 23.37, 'step0_MAE': 11.41, 'tol': 0.02}   # 已记值，对拍门禁
OUT = 'plots/fig_EXP25_baseline_newenv_2026-10-10.png'

BLUE, ORANGE, GRAY, BLACK = '#0072B2', '#E69F00', '#999999', '#000000'


def main():
    df = pd.read_csv(DATA_CSV)
    date_col = 'date' if 'date' in df.columns else df.columns[0]

    scaler = StandardScaler()
    scaler.fit(df['ssn'].values[:NUM_TRAIN].reshape(-1, 1))

    pred_z = np.load(os.path.join(RESULT_DIR, 'pred.npy'))[:, :, 0]
    true_z = np.load(os.path.join(RESULT_DIR, 'true.npy'))[:, :, 0]
    pred = scaler.inverse_transform(pred_z.reshape(-1, 1)).reshape(pred_z.shape)
    true = scaler.inverse_transform(true_z.reshape(-1, 1)).reshape(true_z.shape)

    # ---- 对拍门禁：图内 MAE 必须与已记值一致 ----
    full_mae = float(np.mean(np.abs(pred - true)))
    step0_mae = float(np.mean(np.abs(pred[0] - true[0])))
    ok = (abs(full_mae - EXPECTED['full_MAE']) <= EXPECTED['tol']
          and abs(step0_mae - EXPECTED['step0_MAE']) <= EXPECTED['tol'])
    print(f'对拍: full_MAE={full_mae:.4f} (期望 {EXPECTED["full_MAE"]})  '
          f'step0_MAE={step0_mae:.4f} (期望 {EXPECTED["step0_MAE"]})  '
          f'-> {"PASS" if ok else "FAIL"}')
    if not ok:
        print('❌ 图内 MAE 与已记值不一致（figure_guide §2 铁律）→ 不出图')
        sys.exit(1)

    n_win, pred_len = pred.shape
    test_start = NUM_TRAIN + NUM_VAL
    dates = pd.to_datetime(df[date_col].iloc[test_start:test_start + n_win + pred_len - 1]).values
    obs = df['ssn'].values[test_start:test_start + n_win + pred_len - 1]
    step_mae = np.mean(np.abs(pred - true), axis=0)          # 提前期剖面 (24,)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), dpi=300,
        gridspec_kw={'height_ratios': [2.1, 1.0], 'hspace': 0.45})

    # ---------- Panel A：预测 vs 真实 ----------
    for i in range(n_win):                                    # 全窗口预测扇形（灰）
        ax1.plot(dates[i:i + pred_len], pred[i], color=GRAY, lw=0.4, alpha=0.16,
                 zorder=1, solid_capstyle='round')
    ax1.plot([], [], color=GRAY, lw=1.0, alpha=0.6, label='All 125 sliding-window forecasts (steps 1-24)')
    # ⚠️ 两个"step 0"含义不同，必须分开标注（SOP 坑 20：禁止简化过度）：
    #   step_mae[0]  = 所有窗口在"提前 1 个月"位置的 MAE = 18.44
    #   step0_mae    = 第 0 号窗口整段 24 步的 MAE = 11.41（记录里的 step0 MAE 是这个）
    ax1.plot(dates[:n_win], pred[:, 0], color=BLUE, lw=1.3, ls='--',
             label=f'Lead = 1 month (step 1 of each window), MAE = {step_mae[0]:.2f}')
    ax1.plot(dates[pred_len - 1:], pred[:, pred_len - 1], color=ORANGE, lw=1.3, ls='--',
             label=f'Lead = 24 months (step 24 of each window), MAE = {step_mae[-1]:.2f}')
    ax1.plot(dates, obs, color=BLACK, lw=1.6, zorder=5, label='Observation (monthly mean SSN)')
    ax1.set_ylabel('Sunspot number (SSN)')
    ax1.set_title('EXP-25 new-environment baseline — W1 backtest (test period 1996-08 to 2008-11)',
                  fontsize=10, pad=8)
    ax1.legend(loc='upper left', fontsize=7.2, frameon=False, ncol=1)

    # ---------- Panel B：提前期剖面 ----------
    steps = np.arange(1, pred_len + 1)
    ax2.plot(steps, step_mae, color=BLUE, lw=1.5, marker='o', ms=2.6)
    ax2.axhline(full_mae, color=GRAY, lw=1.0, ls='--')
    ax2.text(1, full_mae + 0.5, f'all-step MAE = {full_mae:.2f}',
             color=GRAY, fontsize=7.2, va='bottom', ha='left')
    ax2.set_xlabel('Forecast lead time (months ahead)')
    ax2.set_ylabel('MAE (SSN)')
    ax2.set_title('Error growth with lead time', fontsize=9, pad=6)
    ax2.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax2.yaxis.set_major_locator(MaxNLocator(5))

    for ax in (ax1, ax2):                                     # 学术风
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8, length=3)
        ax.grid(axis='y', color='#DDDDDD', lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
    ax1.tick_params(labelbottom=False)
    ax1.text(0.995, 0.03, 'single seed (n = 1)', transform=ax1.transAxes,
             fontsize=7, color=ORANGE, ha='right', va='bottom')

    caption = (
        'Environment: Python 3.10.18 / PyTorch 2.8.0+cu128 / NVIDIA RTX A6000, TF32 disabled (NVIDIA_TF32_OVERRIDE=0).\n'
        'Config: PatchTST — seq_len 336 / pred_len 24 / d_model 128 / n_heads 8 / e_layers 2 / patch 16 / stride 8 / 50 epochs;\n'
        'window W1 = date-based split, num_train 2838, val 132, test 148 months; predictions from the best-validation\n'
        'checkpoint (train && re-test chain, SOP §4); MAE on physical monthly-mean SSN after inverse scaling.\n'
        f'Recorded "step0 MAE" = {step0_mae:.2f} is the mean over the 24 steps of window #0 only — NOT the lead-1 MAE\n'
        f'({step_mae[0]:.2f}). Recorded all-step MAE = {full_mae:.2f}.  Single seed (2021), no error bars: differences of a\n'
        'few percent are within seed noise (data_pipeline.md single-seed caveat).'
    )
    fig.text(0.012, 0.006, caption, fontsize=5.6, color='#444444', ha='left', va='bottom',
             linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.235)
    os.makedirs('plots', exist_ok=True)
    fig.savefig(OUT, facecolor='white')
    print(f'已保存 {OUT}')
    print(f'step 1 MAE={step_mae[0]:.2f}, step 12 MAE={step_mae[11]:.2f}, step 24 MAE={step_mae[-1]:.2f}')


if __name__ == '__main__':
    main()
