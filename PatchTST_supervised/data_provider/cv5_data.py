"""Sequential blocked 5-fold dataset for BASE-2026-11-M.

New file only. The legacy Dataset_Custom remains untouched.

The dataset creates samples only within explicitly allowed ranges:
- train: inner-train portion of the four non-outer-validation blocks;
- val: inner-validation portion of those same four blocks;
- test: the complete outer-validation block.

A sample is represented by absolute slices of one transformed full series so
that train/val/test use the same scaler. No sample window may cross a range
boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features

from .data_loader import TARGET_TRANSFORMS


def _load_fold(fold_spec_path: str | Path, fold_id: int) -> dict:
    spec = json.loads(Path(fold_spec_path).read_text(encoding="utf-8"))
    if int(spec["n_folds"]) != 5:
        raise ValueError("BASE-2026-11-M requires exactly five folds")
    for fold in spec["folds"]:
        if int(fold["fold"]) == int(fold_id):
            return fold
    raise ValueError(f"fold {fold_id} not found in {fold_spec_path}")


class Dataset_BlockCV(Dataset):
    """Five-fold blocked dataset with a shared training-only scaler."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        fold_spec_path: str,
        fold_id: int,
        flag: str,
        size: Iterable[int],
        features: str = "MS",
        target: str = "ssn",
        timeenc: int = 1,
        freq: str = "m",
        target_transform: str = "",
        scaler: StandardScaler | None = None,
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train, val or test")
        if features not in {"MS"}:
            raise ValueError("BASE-2026-11-M currently supports features=MS only")

        self.seq_len, self.label_len, self.pred_len = [int(v) for v in size]
        self.flag = flag
        self.features = features
        self.target = target
        self.fold_id = int(fold_id)
        self.fold_spec_path = str(fold_spec_path)
        self.fold = _load_fold(fold_spec_path, fold_id)
        self.timeenc = timeenc
        self.freq = freq
        self.target_transform = target_transform

        self.df = pd.read_csv(Path(root_path) / data_path)
        required = {"date", "month_sin", "month_cos", target}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        self.dates = pd.to_datetime(self.df["date"])
        if not self.dates.is_monotonic_increasing:
            raise ValueError("date must be monotonic increasing")

        feature_cols = ["month_sin", "month_cos", target]
        self.data_frame = self.df[feature_cols].copy()
        if target_transform:
            fn = TARGET_TRANSFORMS.get(target_transform)
            if fn is None:
                raise ValueError(f"unsupported target_transform: {target_transform}")
            self.data_frame[target] = fn(self.data_frame[target].clip(lower=0))

        if flag == "train":
            self.scaler = StandardScaler()
            fit_rows = []
            for block in self.fold["train_blocks"]:
                r = block["inner_train"]
                fit_rows.append(self.data_frame.iloc[r["start"] : r["end"] + 1].values)
            self.scaler.fit(np.concatenate(fit_rows, axis=0))
        else:
            if scaler is None:
                raise ValueError("val/test dataset requires the train fold scaler")
            self.scaler = scaler

        self.data = self.scaler.transform(self.data_frame.values)
        stamp = pd.DataFrame({"date": self.dates})
        if timeenc == 0:
            stamp["month"] = stamp["date"].dt.month
            stamp["day"] = stamp["date"].dt.day
            stamp["weekday"] = stamp["date"].dt.weekday
            stamp["hour"] = stamp["date"].dt.hour
            self.data_stamp = stamp.drop(columns=["date"]).values
        elif timeenc == 1:
            self.data_stamp = time_features(pd.DatetimeIndex(self.dates), freq=freq).transpose(1, 0)
        else:
            raise ValueError("timeenc must be 0 or 1")

        self.window_origins: list[int] = []
        self.window_target_starts: list[int] = []
        self.window_target_ends: list[int] = []
        self._build_windows()

    def _append_window(self, target_start: int, target_end: int) -> None:
        origin = target_start - self.seq_len
        if origin < 0:
            return
        self.window_origins.append(int(origin))
        self.window_target_starts.append(int(target_start))
        self.window_target_ends.append(int(target_end))

    def _build_target_windows(
        self, starts: list[tuple[int, int]], require_context_in_segment: bool = False
    ) -> None:
        for segment_start, segment_end in starts:
            latest_start = segment_end - self.pred_len + 1
            for target_start in range(segment_start, latest_start + 1):
                if require_context_in_segment and target_start - self.seq_len < segment_start:
                    continue
                self._append_window(target_start, target_start + self.pred_len - 1)

    def _build_windows(self) -> None:
        if self.flag == "train":
            ranges = []
            for block in self.fold["train_blocks"]:
                r = block["inner_train"]
                ranges.append((r["start"], r["end"]))
            self._build_target_windows(ranges, require_context_in_segment=True)
        elif self.flag == "val":
            ranges = []
            for block in self.fold["train_blocks"]:
                r = block["inner_val"]
                ranges.append((r["start"], r["end"]))
            self._build_target_windows(ranges)
        else:
            r = self.fold["outer_val"]
            self._build_target_windows([(r["start"], r["end"])])

        if not self.window_origins:
            raise ValueError(f"fold {self.fold_id} {self.flag}: no valid windows")

    def __getitem__(self, index: int):
        origin = self.window_origins[index]
        target_start = self.window_target_starts[index]
        seq_x = self.data[origin : origin + self.seq_len]
        seq_x_mark = self.data_stamp[origin : origin + self.seq_len]
        y_start = target_start - self.label_len
        y_end = target_start + self.pred_len
        seq_y = self.data[y_start:y_end]
        seq_y_mark = self.data_stamp[y_start:y_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.window_origins)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def metadata(self) -> dict:
        target_dates = []
        for start, end in zip(self.window_target_starts, self.window_target_ends):
            target_dates.append(
                {
                    "start": self.dates.iloc[start].strftime("%Y-%m"),
                    "end": self.dates.iloc[end].strftime("%Y-%m"),
                }
            )
        return {
            "fold": self.fold_id,
            "flag": self.flag,
            "seq_len": self.seq_len,
            "label_len": self.label_len,
            "pred_len": self.pred_len,
            "n_windows": len(self),
            "window_origins": self.window_origins,
            "window_target_starts": self.window_target_starts,
            "window_target_ends": self.window_target_ends,
            "target_dates": target_dates,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "feature_columns": ["month_sin", "month_cos", self.target],
            "target_transform": self.target_transform,
            "fold_spec": self.fold,
        }


def build_cv5_data(args):
    """Build train/val/test datasets and loaders with one shared scaler."""

    timeenc = 0 if args.embed != "timeF" else 1
    common = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        fold_spec_path=args.cv_file,
        fold_id=args.cv_fold,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        target_transform=args.target_transform,
    )

    train_set = Dataset_BlockCV(flag="train", **common)
    scaler = train_set.scaler
    val_set = Dataset_BlockCV(flag="val", scaler=scaler, **common)
    test_set = Dataset_BlockCV(flag="test", scaler=scaler, **common)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return {
        "train": (train_set, train_loader),
        "val": (val_set, val_loader),
        "test": (test_set, test_loader),
    }
