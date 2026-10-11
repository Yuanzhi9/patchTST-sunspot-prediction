"""
block_size_sweep.py — P2-a 块长扫描 + P2-b 集成（零训练，评估复跑）

故事问题：
  P2-a：回填频率（block_size）和 MAE 是什么关系？→ 判"逐月回填的漂移"是不是主因
  P2-b：把几条轨迹合起来（取中位数）能不能压住单条轨迹发飙？

门禁（硬）：block_size=1 必须逐位等于 roll_eval.py 的滚动 MAE（记录值）；
          不一致直接报错退出，不出表不出图。

可部署性标注（重要）：
  - 多块长中位数 {6,12,24}：三条轨迹都从预测起点出发 → **可部署**
  - 多起点中位数（起点 0,6,12,...）：后面的起点要用到"起点之后才发生的真值窗口" → **回测专用，不可部署**

用法: python3 scripts/block_size_sweep.py [--plot]
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "PatchTST_supervised"))
from roll_eval import load_model, load_data_enc3, run_rolling_enc3
from b23_block_eval import block_predict

DATA_CSV = os.path.join(ROOT, "PatchTST_supervised/dataset/sunspot_with_cycle.csv")
SIZES = [1, 2, 3, 6, 12, 24]

# (窗口, 工作区, checkpoint 目录, num_train, n_roll, {dropout: (滚动记录, block24 记录)})
RUNS = [
    ("W1", "/output/GPU-exp26" if False else "/output/GPU-exp25",
     "EXP-25", 2838, 148, {"0.05": (41.63, 21.95)}),
    ("W1", "/output/GPU-exp26", "EXP-26", 2838, 148, {"0.20": (50.29, 23.55)}),
    ("W2", ROOT, "EXP-27", 2986, 132, {"0.05": (109.62, 33.54)}),
    ("W2", ROOT, "EXP-28", 2986, 132, {"0.20": (47.95, 28.44)}),
    ("W3", ROOT, "EXP-29", 3118, 71, {"0.05": (23.28, 22.86)}),
    ("W3", ROOT, "EXP-30", 3118, 71, {"0.20": (43.65, 22.17)}),
]


def ckpt_path(ws, eid):
    pat = "sunspot_PatchTST_custom_ftMS_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_"
    return os.path.join(ws, "checkpoints", pat + eid + "_0", "full_checkpoint.pth")


def block_offset(model, scaler, data_z, t0, seq_len, n_roll, block_size, start):
    """从 start 月开始、用 start 之前的真值窗口起步的 block 轨迹，返回覆盖 [start, n_roll) 的预测。"""
    from b23_block_eval import block_predict as _bp
    n_eff = n_roll - start
    sub = _bp(model, scaler, data_z, t0 + start, seq_len, n_eff, block_size, 0)
    out = np.full(n_roll, np.nan)
    out[start:] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    table, ens_rows = {}, []
    for name, ws, eid, nt, n_roll, dinfo in RUNS:
        drop, (rec_roll, rec_block24) = list(dinfo.items())[0]
        model, ck = load_model(ckpt_path(ws, eid))
        seq_len = int(getattr(ck, "seq_len", 336))
        _, scaler, data_z, t0 = load_data_enc3(DATA_CSV, nt, 132, "standard", "")
        trues = np.array([float(scaler.inverse_transform(data_z[t0 + t].reshape(1, -1))[0, 2])
                          for t in range(n_roll)])

        traj, maes = {}, {}
        for bs in SIZES:
            p = block_predict(model, scaler, data_z, t0, seq_len, n_roll, bs, 0)
            traj[bs] = p
            maes[bs] = float(np.mean(np.abs(p - trues)))

        # ---- 门禁 ----
        if round(maes[1], 2) != rec_roll or round(maes[24], 2) != rec_block24:
            raise SystemExit(f"[门禁失败] {name} do{drop}: block=1 {maes[1]:.2f}(记录 {rec_roll}) "
                             f"block24 {maes[24]:.2f}(记录 {rec_block24})")
        table[(name, drop)] = maes
        print(f"{name} do{drop}  门禁 PASS（block=1 {maes[1]:.2f} = 滚动记录）")
        print("    " + "  ".join(f"{bs}:{maes[bs]:6.2f}" for bs in SIZES))
        # 单条轨迹是否出现负值
        neg = {bs: int((traj[bs] < 0).sum()) for bs in SIZES}
        print("    轨迹中负值月数: " + "  ".join(f"{bs}:{neg[bs]}" for bs in SIZES))

        # ---- P2-b 集成 ----
        ens_mb = np.median(np.vstack([traj[6], traj[12], traj[24]]), axis=0)   # 可部署
        starts = list(range(0, max(n_roll - 24, 1), 6))
        stack = np.vstack([block_offset(model, scaler, data_z, t0, seq_len, n_roll, 24, s) for s in starts])
        with np.errstate(all="ignore"):
            ens_ms = np.nanmedian(stack, axis=0)                              # 回测专用
        ens_ms[:24] = traj[24][:24]      # 头 24 月只有一条轨迹，保持原值（避免 nan）
        m_mb = float(np.mean(np.abs(ens_mb - trues)))
        m_ms = float(np.mean(np.abs(ens_ms - trues)))
        ens_rows.append((name, drop, maes[24], m_mb, m_ms,
                         float(np.max(traj[24])), float(np.max(ens_mb)), float(np.max(ens_ms)),
                         int((traj[24] < 0).sum()), int((ens_mb < 0).sum()), int((ens_ms < 0).sum())))
        print(f"    集成: 单条block24 {maes[24]:6.2f} | 多块长中位数 {m_mb:6.2f} | 多起点中位数(回测专用) {m_ms:6.2f}")

    print("\n===== P2-b 集成汇总 =====")
    print(f"{'win':3} {'do':4} {'单block24':>9} {'多块长中位':>10} {'多起点中位*':>11} "
          f"{'峰(单/块长/多起点)':>22} {'负值月数(单/块长/多起点)':>24}")
    for (n, d, a, b, c, pk1, pk2, pk3, g1, g2, g3) in ens_rows:
        print(f"{n:3} {d:4} {a:9.2f} {b:10.2f} {c:11.2f}   {pk1:6.1f}/{pk2:6.1f}/{pk3:6.1f}      "
              f"{g1:4d}/{g2:4d}/{g3:4d}")
    print("* 多起点集成不可部署（后面的起点要用到起点之后才有的真值窗口），仅作回测诊断。")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        BLUE, ORANGE, GRAY, BLACK = "#0072B2", "#E69F00", "#999999", "#000000"
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
        ax = axes[0]
        for (name, drop), maes in table.items():
            ls = "-" if drop == "0.20" else "--"
            col = {"W1": BLACK, "W2": BLUE, "W3": ORANGE}[name]
            ax.plot(SIZES, [maes[b] for b in SIZES], ls, color=col, marker="o", ms=3,
                    label=f"{name} do{drop}")
        ax.set_xlabel("block_size (months refilled per step)")
        ax.set_ylabel("physical MAE over the whole test period")
        ax.set_xticks(SIZES)
        ax.set_title("P2-a: MAE vs refill granularity", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax = axes[1]
        idx = np.arange(len(ens_rows))
        w = 0.27
        ax.bar(idx - w, [r[2] for r in ens_rows], w, color=GRAY, label="single block24")
        ax.bar(idx, [r[3] for r in ens_rows], w, color=ORANGE, label="median of block {6,12,24}")
        ax.bar(idx + w, [r[4] for r in ens_rows], w, color=BLUE, label="median over start months*")
        for j, r in enumerate(ens_rows):
            for off, v in ((-w, r[2]), (0, r[3]), (w, r[4])):
                ax.text(j + off, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.8)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{r[0]}\ndo{r[1]}" for r in ens_rows], fontsize=7)
        ax.set_ylabel("physical MAE")
        ax.set_title("P2-b: ensembles", fontsize=9)
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        fig.suptitle("Refill granularity and ensembles — new GPU env, monthly-mean SSN, single seed (n = 1)",
                     fontsize=9.5, y=1.0)
        fig.text(0.01, -0.06,
                 "* the multi-start median is backtest-only: later starts use truth windows that only exist after those start months. "
                 "Gate: block_size = 1 reproduces the recorded rolling MAE (50.29/47.95/43.65 and 41.63/109.62/23.28) digit-for-digit.",
                 fontsize=6.5, color="#444444")
        out = os.path.join(ROOT, "plots", "fig_block_size_sweep_2026-10-11.png")
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n出图: {out}")


if __name__ == "__main__":
    main()
