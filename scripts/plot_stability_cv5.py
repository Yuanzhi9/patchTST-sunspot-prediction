"""Plot lead-time profiles and sliding-window error distributions for CV5 runs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


def physical_ssn(z: np.ndarray, meta: dict) -> np.ndarray:
    transform = meta.get("target_transform", "")
    if transform:
        raise ValueError(f"target_transform={transform!r} is not supported")
    return z * float(meta["scaler_scale"][-1]) + float(meta["scaler_mean"][-1])


def parse_spec(value: str) -> tuple[str, str, int]:
    label, rest = value.split("=", 1)
    des, seed_text = rest.rsplit(":", 1)
    return label, des, int(seed_text)


def load_spec(label: str, des: str, seed: int) -> dict:
    pattern = f"results/*_{des}_F*_seed{seed}/"
    dirs = [
        Path(path)
        for path in sorted(glob.glob(pattern))
        if Path(path, "pred.npy").exists() and Path(path, "cv_window_meta.json").exists()
    ]
    if not dirs:
        raise SystemExit(f"no result directories for {label}: {pattern}")

    lead_profiles = []
    window_mae = []
    for result_dir in dirs:
        pred_z = np.load(result_dir / "pred.npy")
        true_z = np.load(result_dir / "true.npy")
        meta = json.loads((result_dir / "cv_window_meta.json").read_text(encoding="utf-8"))
        pred = physical_ssn(pred_z[:, :, -1], meta)
        true = physical_ssn(true_z[:, :, -1], meta)
        lead_profiles.append(np.mean(np.abs(pred - true), axis=0))
        window_mae.append(np.mean(np.abs(pred - true), axis=1))
    return {
        "label": label,
        "des": des,
        "seed": seed,
        "lead_profiles": np.asarray(lead_profiles),
        "window_mae": np.concatenate(window_mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CV5 stability diagnostics")
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="label=experiment_id:seed (repeatable)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="CV5 stability")
    args = parser.parse_args()

    records = [load_spec(*parse_spec(value)) for value in args.spec]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    for i, record in enumerate(records):
        color = COLORS[i % len(COLORS)]
        profiles = record["lead_profiles"]
        mean = profiles.mean(axis=0)
        low = profiles.min(axis=0)
        high = profiles.max(axis=0)
        x = np.arange(1, mean.size + 1)
        ax1.plot(x, mean, color=color, lw=1.5, label=f"{record['label']} mean")
        ax1.fill_between(x, low, high, color=color, alpha=0.12, lw=0)

    ax1.set_xlabel("Forecast lead (months)")
    ax1.set_ylabel("MAE (SSN)")
    ax1.set_title("Lead-time error profile (line=mean, band=fold range)", loc="left", fontsize=9)
    ax1.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=7, frameon=False)

    box = ax2.boxplot(
        [record["window_mae"] for record in records],
        tick_labels=[record["label"] for record in records],
        showfliers=False,
        patch_artist=True,
    )
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.32)
    ax2.set_ylabel("Per-window MAE (SSN)")
    ax2.set_title("All sliding-window error distribution", loc="left", fontsize=9)
    ax2.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    seeds = sorted({record["seed"] for record in records})
    fig.suptitle(f"{args.title} | seeds={seeds}", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
