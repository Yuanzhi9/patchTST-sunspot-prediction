"""Aggregate PR24 capacity-audit seed/fold metrics without plotting."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/p24_capacity"
OUT = ANALYSIS / "capacity_audit_summary.csv"

CONFIGS = {
    "B26M-B-P36-BS32": (128, 2, 2048),
    "B26M-CAP-D16": (16, 2, 2048),
    "B26M-CAP-D32": (32, 2, 2048),
    "B26M-CAP-D64": (64, 2, 2048),
    "B26M-CAP-D256": (256, 2, 2048),
    "B26M-CAP-E1": (128, 1, 2048),
    "B26M-CAP-E3": (128, 3, 2048),
    "B26M-CAP-F512": (128, 2, 512),
    "B26M-CAP-F1024": (128, 2, 1024),
}


def baseline_seed_means() -> dict[int, tuple[float, float]]:
    path = ROOT / "analysis/p24_dual_summary/candidate_five_seed_detail.csv"
    result = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        result[int(row["seed"])] = (float(row["full_MAE"]), float(row["block24_MAE"]))
    return result


def main() -> None:
    rows = []
    baseline = baseline_seed_means()
    for des, (d_model, e_layers, d_ff) in CONFIGS.items():
        if des == "B26M-B-P36-BS32":
            seed_metrics = baseline
        else:
            seed_metrics = {}
            for path in sorted(ANALYSIS.glob(f"{des}_seed*/cv5_aggregate.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                seed_metrics[int(payload["seed"])] = (
                    float(payload["full_MAE_mean"]),
                    float(payload["block24_MAE_mean"]),
                )
        if not seed_metrics:
            continue
        full = np.asarray([value[0] for value in seed_metrics.values()], dtype=float)
        block = np.asarray([value[1] for value in seed_metrics.values()], dtype=float)
        rows.append(
            {
                "des": des,
                "d_model": d_model,
                "e_layers": e_layers,
                "d_ff": d_ff,
                "n_seeds": len(seed_metrics),
                "seeds": "/".join(str(seed) for seed in sorted(seed_metrics)),
                "full_MAE_mean": float(np.mean(full)),
                "full_MAE_seed_std": float(np.std(full, ddof=1)) if len(full) > 1 else None,
                "full_MAE_worst_seed": float(np.max(full)),
                "block24_MAE_mean": float(np.mean(block)),
                "block24_MAE_seed_std": float(np.std(block, ddof=1)) if len(block) > 1 else None,
                "block24_MAE_worst_seed": float(np.max(block)),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["full_MAE_mean", "block24_MAE_mean"])
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
