"""Plot overlapping PR132 forecasts with periodic observation updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def result_dir(root: Path, experiment: str, seed: int, fold: int, marker: str) -> Path:
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and f"_{experiment}_F{fold}_seed{seed}" in path.name
        and (path / marker).exists()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one result for {experiment} F{fold} seed{seed}, got {len(matches)}"
        )
    return matches[0]


def physical(array: np.ndarray, meta: dict) -> np.ndarray:
    return (
        array[:, :, 0] * float(meta["scaler_scale"][-1])
        + float(meta["scaler_mean"][-1])
    )


def latest_updated_curve(
    pred: np.ndarray, true: np.ndarray, meta: dict, stride: int
) -> pd.DataFrame:
    """Use the newest forecast available at every target month."""
    starts = np.asarray(meta["window_target_starts"], dtype=int)
    index = {int(start): i for i, start in enumerate(starts)}
    first = int(starts[0])
    end = int(meta["fold_spec"]["outer_val"]["end"])
    dates = pd.date_range(
        pd.Timestamp(meta["target_dates"][0]["start"]),
        periods=end - first + 1,
        freq="MS",
    )

    records = {}
    for start in np.arange(first, end + 1, stride):
        i = index.get(int(start))
        if i is None:
            continue
        for lead in range(stride):
            global_index = int(start) + lead
            if global_index > end:
                break
            records[dates[global_index - first]] = (
                float(pred[i, lead]),
                float(true[i, lead]),
            )
    return pd.DataFrame(
        [(date, values[0], values[1]) for date, values in records.items()],
        columns=["date", "pred", "true"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p132_root", default=".")
    parser.add_argument("--p24_root", default="/output/BASE-2026-11-M")
    parser.add_argument(
        "--p132_des", default="B26M-P132-S132-P36-LR1E3-DO03"
    )
    parser.add_argument("--p24_des", default="B26M-B-P36-BS32")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out_dir", default="analysis_p132/update_cadence")
    args = parser.parse_args()

    p132_root = Path(args.p132_root)
    p24_root = Path(args.p24_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), dpi=260, squeeze=False)
    rows = []
    for fold in range(1, 6):
        result = result_dir(
            p132_root / "results",
            args.p132_des,
            args.seed,
            fold,
            "p132_curve.csv",
        )
        meta = json.loads((result / "cv_window_meta.json").read_text(encoding="utf-8"))
        pred = physical(np.load(result / "pred.npy"), meta)
        true = physical(np.load(result / "true.npy"), meta)
        one_shot = pd.read_csv(result / "p132_curve.csv", parse_dates=["date"])
        update12 = latest_updated_curve(pred, true, meta, 12)
        update24 = latest_updated_curve(pred, true, meta, 24)

        p24_result = result_dir(
            p24_root / "results", args.p24_des, args.seed, fold, "cv_block_curve.csv"
        )
        p24_curve = pd.read_csv(
            p24_result / "cv_block_curve.csv", parse_dates=["date"]
        )

        ax = axes[fold - 1, 0]
        ax.plot(one_shot["date"], one_shot["true"], color="black", lw=1.25, label="Observed")
        ax.plot(
            one_shot["date"],
            one_shot["pred"],
            color="#0072B2",
            lw=1.1,
            ls="--",
            label="PR132 one-shot",
        )
        ax.plot(
            update12["date"],
            update12["pred"],
            color="#009E73",
            lw=1.0,
            label="PR132 update12 latest",
        )
        ax.plot(
            update24["date"],
            update24["pred"],
            color="#E69F00",
            lw=1.0,
            label="PR132 update24 latest",
        )
        ax.plot(
            p24_curve["date"],
            p24_curve["pred"],
            color="#999999",
            lw=0.9,
            alpha=0.8,
            label="PR24 block24",
        )

        def common_mae(curve: pd.DataFrame) -> float:
            merged = curve.merge(
                one_shot[["date", "true"]],
                on="date",
                suffixes=("", "_observed"),
            )
            return float((merged["pred"] - merged["true_observed"]).abs().mean())

        values = {
            "one_shot": float((one_shot["pred"] - one_shot["true"]).abs().mean()),
            "update12": common_mae(update12),
            "update24": common_mae(update24),
            "p24": common_mae(p24_curve),
        }
        ax.set_title(
            f"Fold {fold} | same-date MAE: one={values['one_shot']:.2f}, "
            f"u12={values['update12']:.2f}, u24={values['update24']:.2f}, "
            f"PR24={values['p24']:.2f}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("SSN")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if fold == 1:
            ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper right")
        rows.append({"fold": fold, **values})

    fig.suptitle(
        f"PR132 observation-update diagnostic | seed {args.seed} | historical blocked CV",
        fontsize=10,
    )
    fig.tight_layout()
    output = out_dir / f"p132_update_cadence_seed{args.seed}.png"
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    print(output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
