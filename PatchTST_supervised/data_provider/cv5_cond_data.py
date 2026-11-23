"""CV5 dataset wrapper with oracle cycle-level conditions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .cv5_data import Dataset_BlockCV


MODE_MASK = {
    "none": np.array([0.0, 0.0], dtype=np.float32),
    "amp": np.array([1.0, 0.0], dtype=np.float32),
    "time": np.array([0.0, 1.0], dtype=np.float32),
    "both": np.array([1.0, 1.0], dtype=np.float32),
}


class ConditionalBlockCV(Dataset_BlockCV):
    def __init__(
        self,
        *args,
        cond_mode: str,
        boundaries_file: str,
        cond_stats: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        if cond_mode not in MODE_MASK:
            raise ValueError(f"unsupported cond_mode: {cond_mode}")
        self.cond_mode = cond_mode
        self.boundaries_file = str(boundaries_file)
        super().__init__(*args, **kwargs)
        self._build_conditions(cond_stats)

    def _build_conditions(self, cond_stats: tuple[np.ndarray, np.ndarray] | None) -> None:
        boundaries = pd.read_csv(self.boundaries_file, parse_dates=["min_date", "peak_date", "next_min_date"])
        rows = []
        for target_start in self.window_target_starts:
            target_date = self.dates.iloc[target_start].normalize()
            valid = boundaries.loc[
                (boundaries["min_date"] <= target_date)
                & (target_date < boundaries["next_min_date"])
            ]
            if len(valid) != 1:
                raise ValueError(
                    f"{target_date:%Y-%m} matched {len(valid)} cycle boundaries"
                )
            cycle = valid.iloc[0]
            amp = float(np.log1p(cycle["peak_smooth"]))
            peak_offset = float(
                (
                    cycle["peak_date"].to_period("M")
                    - self.dates.iloc[target_start].to_period("M")
                ).n
            )
            rows.append((amp, peak_offset))
        raw_conditions = np.asarray(rows, dtype=np.float32)

        if self.flag == "train":
            self.cond_mean = raw_conditions.mean(axis=0)
            cond_scale = raw_conditions.std(axis=0)
            self.cond_scale = np.where(cond_scale == 0, 1.0, cond_scale).astype(np.float32)
        else:
            if cond_stats is None:
                raise ValueError("val/test conditional dataset requires train condition stats")
            self.cond_mean, self.cond_scale = (
                np.asarray(cond_stats[0], dtype=np.float32),
                np.asarray(cond_stats[1], dtype=np.float32),
            )

        normalized = (raw_conditions - self.cond_mean) / self.cond_scale
        self.conditions = (
            normalized * MODE_MASK[self.cond_mode].reshape(1, 2)
        ).astype(np.float32)

    def condition_for_target_start(self, target_start: int) -> np.ndarray:
        boundaries = pd.read_csv(
            self.boundaries_file,
            parse_dates=["min_date", "peak_date", "next_min_date"],
        )
        target_date = self.dates.iloc[target_start].normalize()
        valid = boundaries.loc[
            (boundaries["min_date"] <= target_date)
            & (target_date < boundaries["next_min_date"])
        ]
        if len(valid) != 1:
            raise ValueError(f"{target_date:%Y-%m} matched {len(valid)} cycle boundaries")
        cycle = valid.iloc[0]
        raw = np.array(
            [
                np.log1p(float(cycle["peak_smooth"])),
                float(
                    (
                        cycle["peak_date"].to_period("M")
                        - self.dates.iloc[target_start].to_period("M")
                    ).n
                ),
            ],
            dtype=np.float32,
        )
        normalized = (raw - self.cond_mean) / self.cond_scale
        return (normalized * MODE_MASK[self.cond_mode]).astype(np.float32)

    def __getitem__(self, index: int):
        x, y, x_mark, y_mark = super().__getitem__(index)
        return x, y, self.conditions[index]


def build_conditional_cv5_data(args):
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
        cond_mode=args.cond_mode,
        boundaries_file=args.boundaries_file,
    )
    train_set = ConditionalBlockCV(flag="train", **common)
    scaler = train_set.scaler
    cond_stats = (train_set.cond_mean, train_set.cond_scale)
    val_set = ConditionalBlockCV(
        flag="val", scaler=scaler, cond_stats=cond_stats, **common
    )
    test_set = ConditionalBlockCV(
        flag="test", scaler=scaler, cond_stats=cond_stats, **common
    )

    def loader(dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

    return {
        "train": (train_set, loader(train_set, True, True)),
        "val": (val_set, loader(val_set, False, False)),
        "test": (test_set, loader(test_set, False, False)),
    }
