"""CV5 dataset returning two consecutive future blocks for rollout training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .cv5_data import Dataset_BlockCV


class TwoBlockDataset(Dataset_BlockCV):
    def _build_windows(self) -> None:
        if self.flag == "train":
            ranges = [
                (block["inner_train"]["start"], block["inner_train"]["end"])
                for block in self.fold["train_blocks"]
            ]
            require_context_in_segment = True
        elif self.flag == "val":
            ranges = [
                (block["inner_val"]["start"], block["inner_val"]["end"])
                for block in self.fold["train_blocks"]
            ]
            require_context_in_segment = False
        else:
            outer = self.fold["outer_val"]
            ranges = [(outer["start"], outer["end"])]
            require_context_in_segment = False

        for segment_start, segment_end in ranges:
            latest_start = segment_end - 2 * self.pred_len + 1
            for target_start in range(segment_start, latest_start + 1):
                if (
                    require_context_in_segment
                    and target_start - self.seq_len < segment_start
                ):
                    continue
                self._append_window(
                    target_start, target_start + 2 * self.pred_len - 1
                )
        if not self.window_origins:
            raise ValueError(f"fold {self.fold_id} {self.flag}: no two-block windows")

    def __getitem__(self, index: int):
        origin = self.window_origins[index]
        target_start = self.window_target_starts[index]
        seq_x = self.data[origin : origin + self.seq_len]
        block1 = self.data[target_start : target_start + self.pred_len]
        block2_start = target_start + self.pred_len
        block2 = self.data[block2_start : block2_start + self.pred_len]
        return seq_x, block1, block2


def build_two_block_data(args):
    common = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        fold_spec_path=args.cv_file,
        fold_id=args.cv_fold,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=0 if args.embed != "timeF" else 1,
        freq=args.freq,
        target_transform=args.target_transform,
        train_window_limit=args.train_window_limit,
    )
    train_set = TwoBlockDataset(flag="train", **common)
    val_set = TwoBlockDataset(flag="val", scaler=train_set.scaler, **common)
    test_set = TwoBlockDataset(flag="test", scaler=train_set.scaler, **common)

    def make_loader(dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

    return {
        "train": (train_set, make_loader(train_set, True, True)),
        "val": (val_set, make_loader(val_set, False, False)),
        "test": (test_set, make_loader(test_set, False, False)),
    }
