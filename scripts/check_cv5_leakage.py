"""Audit the frozen CV5 split before any model training.

Checks:
1. Blocks cover the complete ordered series exactly once.
2. Outer validation never belongs to training or inner-validation ranges.
3. Inner train and inner validation are disjoint.
4. Every training target has the full seq_len history inside its inner-train block.
5. Every outer target has full target horizon inside the outer block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BASE-2026-11-M CV5 split")
    parser.add_argument("--cv_file", default="configs/cv5_folds_2026-10.json")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=24)
    args = parser.parse_args()

    spec = json.loads(Path(args.cv_file).read_text(encoding="utf-8"))
    errors: list[str] = []

    blocks = spec["blocks"]
    n_months = int(spec["n_months"])
    covered = []
    for block in blocks:
        covered.extend(range(block["start"], block["end"] + 1))
    if covered != list(range(n_months)):
        errors.append("blocks do not cover every month exactly once in order")

    for fold in spec["folds"]:
        outer = fold["outer_val"]
        for tr in fold["train_blocks"]:
            inner_train = tr["inner_train"]
            inner_val = tr["inner_val"]
            if ranges_overlap(inner_train["start"], inner_train["end"], inner_val["start"], inner_val["end"]):
                errors.append(f"fold {fold['fold']}: inner train/val overlap in block {tr['block_id']}")
            if ranges_overlap(outer["start"], outer["end"], inner_train["start"], inner_train["end"]):
                errors.append(f"fold {fold['fold']}: outer val overlaps inner train in block {tr['block_id']}")
            if ranges_overlap(outer["start"], outer["end"], inner_val["start"], inner_val["end"]):
                errors.append(f"fold {fold['fold']}: outer val overlaps inner val in block {tr['block_id']}")
            usable = inner_train["n_months"] - args.seq_len - args.pred_len + 1
            if usable <= 0:
                errors.append(f"fold {fold['fold']}: block {tr['block_id']} has no training windows")

        outer_len = outer["n_months"]
        if outer_len < args.pred_len:
            errors.append(f"fold {fold['fold']}: outer block shorter than pred_len")

    print(json.dumps(
        {
            "cv_file": str(args.cv_file),
            "n_months": n_months,
            "date_range": [spec["start_date"], spec["end_date"]],
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "errors": errors,
        },
        indent=2,
        ensure_ascii=False,
    ))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
