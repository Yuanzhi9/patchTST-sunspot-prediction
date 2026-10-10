"""dropout 口径分裂跨窗口验证 — 出图（三组）

图 A：每窗口一张「滚动预测 vs 真实，do0.05 / do0.2 双线」
图 B：汇总分组柱（三窗口 × 四指标 × 两个 dropout）
图 C：总览权衡散点（本次全部实验点放一张图：x=全步 MAE，y=滚动 MAE，同窗口两点连线）

铁律（figure_guide §2）：图内数值与已记值逐位对拍一致才出图；色盲友好；单种子标注；
英文；白底 dpi300；去顶右边框；caption 自包含。
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, 'PatchTST_supervised'))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3

DATA_CSV = os.path.join(ROOT, 'PatchTST_supervised/dataset/sunspot_with_cycle.csv')
PLOT_DIR = os.path.join(ROOT, 'plots')
DATE = '2026-10-10'
TOL = 0.05

BLUE, ORANGE, GRAY, BLACK = '#0072B2', '#E69F00', '#999999', '#000000'

# name, 窗口, checkpoint, num_train, test_end, 预期(全步MAE, 滚动MAE, E_r, 滚动峰误差)
RUNS = [
    dict(w='W1', tag='do0.05', ck='/output/GPU-exp25/checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-25_0/full_checkpoint.pth',
         res='/output/GPU-exp25/results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-25_0',
         nt=2838, ts='1996-08', te='2008-11', nroll=148, exp=(23.37, 41.63, -6.8, -20.0)),
    dict(w='W1', tag='do0.2', ck='/output/GPU-exp26/checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-26_0/full_checkpoint.pth',
         res='/output/GPU-exp26/results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-26_0',
         nt=2838, ts='1996-08', te='2008-11', nroll=148, exp=(22.87, 50.29, -20.3, -32.2)),
    dict(w='W2', tag='do0.05', ck=os.path.join(ROOT, 'checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-27_0/full_checkpoint.pth'),
         res=os.path.join(ROOT, 'results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-27_0'),
         nt=2986, ts='2008-12', te='2019-11', nroll=132, exp=(21.80, 109.62, 23.4, -67.8)),
    dict(w='W2', tag='do0.2', ck=os.path.join(ROOT, 'checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-28_0/full_checkpoint.pth'),
         res=os.path.join(ROOT, 'results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-28_0'),
         nt=2986, ts='2008-12', te='2019-11', nroll=132, exp=(20.69, 47.95, 12.6, -17.0)),
    dict(w='W3', tag='do0.05', ck=os.path.join(ROOT, 'checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-29_0/full_checkpoint.pth'),
         res=os.path.join(ROOT, 'results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-29_0'),
         nt=3118, ts='2019-12', te='2025-10', nroll=71, exp=(20.99, 23.28, -72.4, -114.7)),
    dict(w='W3', tag='do0.2', ck=os.path.join(ROOT, 'checkpoints/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-30_0/full_checkpoint.pth'),
         res=os.path.join(ROOT, 'results/sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_EXP-30_0'),
         nt=3118, ts='2019-12', te='2025-10', nroll=71, exp=(20.25, 43.65, -75.8, -131.6)),
]


def compute(r):
    """重算每个 run 的四指标（对拍门禁）。"""
    model, ca = load_model(r['ck'])
    seq_len = getattr(ca, 'seq_len', 336)
    df, scaler, data_z, t0 = load_data_enc3(DATA_CSV, r['nt'], 132)
    n_roll = min(r['nroll'], len(data_z) - t0)
    preds, trues = run_rolling_enc3(model, scaler, data_z, t0, seq_len, n_roll)
    roll_mae = float(np.mean(np.abs(preds - trues)))
    pk = int(np.argmax(trues))
    roll_pk = float(preds[pk] - trues[pk])

    pz = np.load(os.path.join(r['res'], 'pred.npy'))[:, :, 0]
    tz = np.load(os.path.join(r['res'], 'true.npy'))[:, :, 0]
    # 全步口径的逆变换用「只 fit ssn 列」的 scaler —— 与 scripts/eval_metrics.py 一致。
    # （roll_eval.load_data_enc3 的 scaler 是 fit 3 列的，ssn 列的 mean/std 相同，
    #   但 inverse_transform 需要 3 列输入，故此处单独建一个 1 列 scaler。）
    s1 = StandardScaler()
    s1.fit(pd.read_csv(DATA_CSV)['ssn'].values[:r['nt']].reshape(-1, 1))
    pred = s1.inverse_transform(pz.reshape(-1, 1)).reshape(pz.shape)
    true = s1.inverse_transform(tz.reshape(-1, 1)).reshape(tz.shape)
    full_mae = float(np.mean(np.abs(pred - true)))
    e_r = float(pred.max() - true.max())
    return dict(preds=preds, trues=trues, full=full_mae, roll=roll_mae, e_r=e_r, roll_pk=roll_pk)


def main():
    for r in RUNS:
        c = compute(r)
        got = (round(c['full'], 2), round(c['roll'], 2), round(c['e_r'], 1), round(c['roll_pk'], 1))
        ok = all(abs(a - b) <= TOL for a, b in zip(got, r['exp']))
        print(f"{r['w']} {r['tag']}: 全步 {got[0]} 滚动 {got[1]} E_r {got[2]} 滚动峰 {got[3]} "
              f"| 期望 {r['exp']} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit('❌ 对拍不一致 → 不出图（figure_guide §2）')
        r.update(c)

    os.makedirs(PLOT_DIR, exist_ok=True)
    wins = ['W1', 'W2', 'W3']

    # ---------- 图 A：每窗口 滚动对比 ----------
    for w in wins:
        rs = [r for r in RUNS if r['w'] == w]
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=300)
        rs0 = rs[0]
        dates = pd.date_range(start=rs0['ts'], periods=len(rs0['trues']), freq='MS')
        ax.plot(dates, rs0['trues'], color=BLACK, lw=1.5, label='Observation (monthly mean SSN)', zorder=5)
        for r, col in zip(rs, (BLUE, ORANGE)):
            ax.plot(dates, r['preds'], color=col, lw=1.3, ls='--',
                    label=f"Rolling 1-month-ahead, {r['tag']}  (MAE {r['roll']:.2f})")
        pk = int(np.argmax(rs0['trues']))
        ax.axvline(dates[pk], color=GRAY, lw=0.8, ls=':')
        ax.set_title(f"Dropout ablation — window {w} ({rs0['ts']} to {rs0['te']}): "
                     f"rolling prediction vs observation", fontsize=10, pad=8)
        ax.set_ylabel('Sunspot number (SSN)')
        ax.legend(loc='upper left', fontsize=7.2, frameon=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#DDDDDD', lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.text(0.995, 0.03, 'single seed (n = 1)', transform=ax.transAxes, fontsize=7,
                color=ORANGE, ha='right', va='bottom')
        fig.text(0.012, 0.015,
                 f"Environment: PyTorch 2.8.0+cu128 / RTX A6000, TF32 off  |  window {w}: num_train={rs0['nt']}, "
                 f"test {rs0['ts']}~{rs0['te']} ({len(rs0['trues'])} months, full period)  |  "
                 f"best-validation checkpoint (train && re-test)  |  MAE on physical monthly-mean SSN  |  single seed 2021",
                 fontsize=6.0, color='#444444')
        fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.16)
        out = os.path.join(PLOT_DIR, f'fig_abl_{w}_roll_dropout_{DATE}.png')
        fig.savefig(out, facecolor='white'); plt.close(fig)
        print('已保存', out)

    # ---------- 图 B：汇总分组柱 ----------
    metrics = [('full', 'Full-step MAE', None), ('roll', 'Rolling MAE (full period)', None),
               ('e_r', 'E_r  (peak-amplitude error, full step)', None),
               ('roll_pk', 'Rolling peak error (pred - true)', None)]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), dpi=300)
    x = np.arange(len(wins))
    for ax, (key, title, _) in zip(axes, metrics):
        v05 = [next(r[key] for r in RUNS if r['w'] == w and r['tag'] == 'do0.05') for w in wins]
        v02 = [next(r[key] for r in RUNS if r['w'] == w and r['tag'] == 'do0.2') for w in wins]
        ax.bar(x - 0.19, v05, 0.36, color=BLUE, label='dropout 0.05')
        ax.bar(x + 0.19, v02, 0.36, color=ORANGE, label='dropout 0.2')
        for xi, (a, b) in enumerate(zip(v05, v02)):
            ax.text(xi - 0.19, a, f'{a:.1f}', ha='center', va='bottom' if a >= 0 else 'top', fontsize=6.5)
            ax.text(xi + 0.19, b, f'{b:.1f}', ha='center', va='bottom' if b >= 0 else 'top', fontsize=6.5)
        ax.axhline(0, color='#888888', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(wins, fontsize=8)
        ax.set_title(title, fontsize=8.2, pad=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#DDDDDD', lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Dropout ablation across three backtest windows — do0.05 vs do0.2 (lower is better; '
                 'peak error: negative = underestimation)', fontsize=9.5, y=0.995)
    fig.text(0.012, 0.01,
             'Same environment (PyTorch 2.8.0+cu128, RTX A6000, TF32 off), same seed 2021, all else identical; '
             'rolling = autoregressive 1-month-ahead over the full test period (W1 148 / W2 132 / W3 71 months); '
             'single seed — no error bars.',
             fontsize=6.0, color='#444444')
    fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.20, wspace=0.28)
    out = os.path.join(PLOT_DIR, f'fig_abl_summary_metrics_{DATE}.png')
    fig.savefig(out, facecolor='white'); plt.close(fig)
    print('已保存', out)

    # ---------- 图 C：总览权衡散点 ----------
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=300)
    for w, col in zip(wins, (BLUE, ORANGE, GRAY)):
        a = next(r for r in RUNS if r['w'] == w and r['tag'] == 'do0.05')
        b = next(r for r in RUNS if r['w'] == w and r['tag'] == 'do0.2')
        ax.annotate('', xy=(b['full'], b['roll']), xytext=(a['full'], a['roll']),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4, alpha=0.85))
        ax.scatter([a['full']], [a['roll']], s=55, color=col, marker='o', zorder=5)
        ax.scatter([b['full']], [b['roll']], s=55, color=col, marker='s', zorder=5)
        ax.annotate(f"{w}  do0.05", (a['full'], a['roll']), textcoords='offset points',
                    xytext=(6, 6), fontsize=7.5, color=col)
        ax.annotate(f"{w}  do0.2", (b['full'], b['roll']), textcoords='offset points',
                    xytext=(6, -12), fontsize=7.5, color=col)
    ax.scatter([], [], marker='o', color='#555555', label='dropout 0.05')
    ax.scatter([], [], marker='s', color='#555555', label='dropout 0.2')
    ax.set_xlabel('Full-step MAE (SSN)  — worse →')
    ax.set_ylabel('Rolling MAE (SSN, full test period)  — worse →')
    ax.set_title('Where does dropout 0.05 → 0.2 move the model?\n'
                 'Arrow points from do0.05 to do0.2 within each window', fontsize=9.5, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(color='#DDDDDD', lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    ax.margins(x=0.13, y=0.10)
    fig.text(0.012, 0.010,
             'Same environment and seed; only dropout/fc_dropout differs.\n'
             'W1: do0.2 improves full-step but worsens rolling (reversal).  W2: do0.2 improves both.  W3: reversal again.\n'
             'Full test period per window (W1 148 / W2 132 / W3 71 months).  Single seed (2021), no error bars.',
             fontsize=6.0, color='#444444', linespacing=1.5)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.86, bottom=0.19)
    out = os.path.join(PLOT_DIR, f'fig_abl_tradeoff_{DATE}.png')
    fig.savefig(out, facecolor='white'); plt.close(fig)
    print('已保存', out)


if __name__ == '__main__':
    main()
