"""CV5 dataset wrapper that attaches audited cycle-event targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .cv5_data import Dataset_BlockCV


EVENT_COLUMNS = [
    "next_peak_raw",
    "months_to_next_peak",
    "months_to_next_trough",
    "cycle_length",
]


class Dataset_BlockCV_Event(Dataset_BlockCV):
    """Use the frozen CV5 windows and append event labels as targets only."""

    def __init__(self, *args, event_labels_path: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event_labels_path = str(event_labels_path)
        if not Path(event_labels_path).exists():
            raise FileNotFoundError(event_labels_path)

        labels = pd.read_csv(event_labels_path)
        labels = labels.loc[
            (labels["fold"] == self.fold_id) & (labels["flag"] == self.flag)
        ].copy()
        labels = labels.set_index("target_start_index")
        if labels.index.duplicated().any():
            raise ValueError("event labels contain duplicate target_start_index")

        required = {
            *EVENT_COLUMNS,
            "peak_mask",
            "trough_mask",
            "cycle_length_mask",
            "target_start_date",
        }
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"event labels missing columns: {sorted(missing)}")

        selected = labels.reindex(self.window_target_starts)
        if selected[list(required)].isna().all(axis=1).any():
            raise ValueError(
                f"fold {self.fold_id} {self.flag}: event labels do not cover all windows"
            )
        observed_dates = [
            self.dates.iloc[start].strftime("%Y-%m")
            for start in self.window_target_starts
        ]
        if observed_dates != selected["target_start_date"].tolist():
            raise ValueError(
                f"fold {self.fold_id} {self.flag}: event-label date alignment failed"
            )

        target_mean = float(self.scaler.mean_[-1])
        target_scale = float(self.scaler.scale_[-1])
        amplitude = selected["next_peak_raw"].to_numpy(dtype=float)
        amplitude_z = (amplitude - target_mean) / target_scale
        amplitude_mask = selected["peak_mask"].to_numpy(dtype=bool)
        amplitude_z = np.where(amplitude_mask, amplitude_z, 0.0)

        peak_time = selected["months_to_next_peak"].to_numpy(dtype=float) / 12.0
        trough_time = selected["months_to_next_trough"].to_numpy(dtype=float) / 12.0
        cycle_length = selected["cycle_length"].to_numpy(dtype=float) / 12.0
        peak_time = np.where(amplitude_mask, peak_time, 0.0)

        trough_mask = selected["trough_mask"].to_numpy(dtype=bool)
        trough_time = np.where(trough_mask, trough_time, 0.0)
        cycle_mask = selected["cycle_length_mask"].to_numpy(dtype=bool)
        cycle_length = np.where(cycle_mask, cycle_length, 0.0)

        self.event_targets = np.stack(
            [amplitude_z, peak_time, trough_time, cycle_length], axis=1
        ).astype(np.float32)
        self.event_masks = np.stack(
            [amplitude_mask, amplitude_mask, trough_mask, cycle_mask], axis=1
        ).astype(np.float32)

    def __getitem__(self, index: int):
        seq_x, seq_y, seq_x_mark, seq_y_mark = super().__getitem__(index)
        return (
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
            self.event_targets[index],
            self.event_masks[index],
        )

    def metadata(self) -> dict:
        meta = super().metadata()
        meta["event_labels_path"] = self.event_labels_path
        meta["event_columns"] = EVENT_COLUMNS
        meta["event_target_units"] = [
            "target standard deviations",
            "years",
            "years",
            "years",
        ]
        return meta


def build_cv5_event_data(args):
    """Build loaders while keeping the original CV5 scaler and split logic."""

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
        train_window_limit=args.train_window_limit,
        event_labels_path=args.event_labels,
    )

    train_set = Dataset_BlockCV_Event(flag="train", **common)
    scaler = train_set.scaler
    val_set = Dataset_BlockCV_Event(flag="val", scaler=scaler, **common)
    test_set = Dataset_BlockCV_Event(flag="test", scaler=scaler, **common)

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
