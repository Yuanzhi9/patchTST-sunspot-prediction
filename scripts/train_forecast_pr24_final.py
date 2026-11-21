"""Final all-history PR24 retraining and Cycle 25/26 recursive forecast.

Hyperparameters come from CV selection. This script deliberately does not create
a new validation split: it performs a fixed-schedule all-history retrain, as a
separate final-fit stage. The formal forecast path feeds non-negative physical
predictions back into the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "PatchTST_supervised"
import sys

sys.path.insert(0, str(sys_path))
from models.PatchTST import Model  # noqa: E402

DEFAULT_DATA = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
DEFAULT_OUT = ROOT / "analysis/pr24_future_forecast"
DEFAULT_CKPT = ROOT / "checkpoints"
DEFAULT_CONFIGS = ROOT / "configs"
FEATURE_COLUMNS = ("month_sin", "month_cos", "ssn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model_args(pred_len: int, d_model: int) -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=3,
        seq_len=132,
        pred_len=pred_len,
        e_layers=2,
        n_heads=8,
        d_model=d_model,
        d_ff=2048,
        dropout=0.2,
        fc_dropout=0.2,
        head_dropout=0.0,
        individual=0,
        patch_len=36,
        stride=18,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        activation="gelu",
    )


def build_windows(data_z: np.ndarray, seq_len: int, pred_len: int) -> TensorDataset:
    xs = []
    ys = []
    for target_start in range(seq_len, len(data_z) - pred_len + 1):
        xs.append(data_z[target_start - seq_len : target_start])
        ys.append(data_z[target_start : target_start + pred_len])
    x_tensor = torch.tensor(np.asarray(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(ys), dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def train_one(
    seed: int,
    frame: pd.DataFrame,
    data_z: np.ndarray,
    data_path: Path,
    pred_len: int,
    d_model: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_root: Path,
    config_root: Path,
    exp_id: str,
) -> tuple[Path, list[dict]]:
    set_seed(seed)
    setting = f"{exp_id}_d{d_model}_seed{seed}"
    checkpoint_dir = checkpoint_root / setting
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "train_history.csv"

    model_args = make_model_args(pred_len, d_model)
    model = Model(model_args).float().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = build_windows(data_z, model_args.seq_len, pred_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=len(loader),
        pct_start=0.3,
        epochs=epochs,
        max_lr=learning_rate,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)[:, -pred_len:, -1]
            target = batch_y[:, -pred_len:, -1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"loss={record['train_loss']:.8f} seconds={record['seconds']:.3f}",
            flush=True,
        )
    pd.DataFrame(history).to_csv(history_path, index=False)

    payload = {
        "seed": seed,
        "setting": setting,
        "model_state_dict": model.state_dict(),
        "model_args": vars(model_args),
        "training_protocol": "all-history fixed-epoch retrain; no validation early stopping",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_windows": len(dataset),
    }
    checkpoint_path = checkpoint_dir / "full_checkpoint.pth"
    torch.save(payload, checkpoint_path)
    config = {
        "experiment_id": exp_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setting": setting,
        "data_csv": str(data_path.relative_to(ROOT))
        if data_path.is_relative_to(ROOT)
        else str(data_path),
        "data_sha256": sha256(data_path),
        "last_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "model_args": vars(model_args),
        "training_protocol": payload["training_protocol"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_windows": len(dataset),
        "seed": seed,
    }
    config_path = config_root / f"{setting}_{datetime.now():%Y%m%d_%H%M%S}.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path, history


def month_features(date: pd.Timestamp) -> np.ndarray:
    # Matches the existing dataset convention: January = sin/cos(pi/6).
    phase = 2.0 * np.pi * date.month / 12.0
    return np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float64)


def forecast_one(
    checkpoint_path: Path,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    data_z: np.ndarray,
    horizon: int,
    block_size: int,
) -> pd.DataFrame:
    payload = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model_args = SimpleNamespace(**payload["model_args"])
    model = Model(model_args).float().cuda()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    mean = float(scaler.mean_[-1])
    scale = float(scaler.scale_[-1])
    origin_date = frame["date"].iloc[-1]
    window = data_z[-model_args.seq_len :].copy()
    rows = []

    for mode in ("raw", "clip0"):
        produced = 0
        block_index = 0
        current_window = window.copy()
        while produced < horizon:
            with torch.no_grad():
                prediction = model(
                    torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).cuda()
                )[0, :, -1].detach().cpu().numpy()
            take = min(block_size, model_args.pred_len, horizon - produced)
            for lead in range(take):
                date = origin_date + pd.DateOffset(months=produced + 1)
                physical_raw = float(prediction[lead] * scale + mean)
                physical_formal = max(physical_raw, 0.0)
                rows.append(
                    {
                        "date": date,
                        "mode": mode,
                        "pred_raw": physical_raw,
                        "pred": physical_formal if mode == "clip0" else physical_raw,
                        "horizon": produced + 1,
                        "block_index": block_index,
                        "lead_in_block": lead + 1,
                    }
                )
                use_value = physical_formal if mode == "clip0" else physical_raw
                z_value = (use_value - mean) / scale
                month = month_features(date)
                current_window = np.vstack(
                    [current_window[1:], np.asarray([month[0], month[1], z_value])]
                )
                produced += 1
            block_index += 1
    return pd.DataFrame(rows)


def detect_cycle25_trough_and_cycle26_peak(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    horizon: int,
) -> dict:
    history_start = frame["date"].iloc[-1] - pd.DateOffset(months=180)
    history = frame[frame["date"] >= history_start][["date", "ssn"]].copy()
    future = forecast[["date", "pred"]].rename(columns={"pred": "ssn"})
    combined = pd.concat([history, future], ignore_index=True)
    combined["smooth13"] = combined["ssn"].rolling(13, center=True, min_periods=13).mean()
    origin = frame["date"].iloc[-1]
    origin_index = int(combined.index[combined["date"] == origin][0])

    trough_end = min(origin_index + 84, len(combined) - 1)
    trough_candidates = combined.iloc[origin_index + 1 : trough_end + 1]
    trough_candidates = trough_candidates[trough_candidates["smooth13"].notna()]
    if trough_candidates.empty:
        raise RuntimeError("no valid smoothed trough candidates")
    trough_index = int(trough_candidates["smooth13"].idxmin())

    peak_start = min(trough_index + 24, len(combined) - 1)
    peak_end = min(trough_index + 108, len(combined) - 1, origin_index + horizon)
    peak_candidates = combined.iloc[peak_start : peak_end + 1]
    peak_candidates = peak_candidates[peak_candidates["smooth13"].notna()]
    if peak_candidates.empty:
        raise RuntimeError("no valid smoothed peak candidates")
    peak_index = int(peak_candidates["smooth13"].idxmax())

    if peak_index <= trough_index:
        raise RuntimeError("Cycle 26 peak was not found after Cycle 25 trough")
    return {
        "cycle25_trough_date": combined.loc[trough_index, "date"].strftime("%Y-%m"),
        "cycle25_trough_smoothed": float(combined.loc[trough_index, "smooth13"]),
        "cycle25_trough_raw": float(combined.loc[trough_index, "ssn"]),
        "cycle26_peak_date": combined.loc[peak_index, "date"].strftime("%Y-%m"),
        "cycle26_peak_smoothed": float(combined.loc[peak_index, "smooth13"]),
        "cycle26_peak_raw": float(combined.loc[peak_index, "ssn"]),
        "months_to_trough": (
            combined.loc[trough_index, "date"].to_period("M") - origin.to_period("M")
        ).n,
        "months_to_peak": (
            combined.loc[peak_index, "date"].to_period("M") - origin.to_period("M")
        ).n,
    }


def summarize_forecast(
    curves: dict[int, pd.DataFrame],
    extrema: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = "clip0"
    pivot = None
    for seed, frame in curves.items():
        selected = frame[frame["mode"] == mode][["date", "pred"]].rename(
            columns={"pred": f"seed{seed}"}
        )
        pivot = selected if pivot is None else pivot.merge(selected, on="date", how="inner")
    columns = [column for column in pivot.columns if column.startswith("seed")]
    values = pivot[columns].to_numpy(dtype=float)
    band = pd.DataFrame(
        {
            "date": pivot["date"],
            "median": np.median(values, axis=1),
            "q10": np.quantile(values, 0.10, axis=1),
            "q90": np.quantile(values, 0.90, axis=1),
            "min_seed": np.min(values, axis=1),
            "max_seed": np.max(values, axis=1),
        }
    )
    rows = []
    for column in (
        "cycle25_trough_smoothed",
        "cycle25_trough_raw",
        "cycle26_peak_smoothed",
        "cycle26_peak_raw",
        "months_to_trough",
        "months_to_peak",
    ):
        values = extrema[column].to_numpy(dtype=float)
        rows.append(
            {
                "quantity": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return band, pd.DataFrame(rows)


def plot_forecast(
    frame: pd.DataFrame,
    curves: dict[int, pd.DataFrame],
    band: pd.DataFrame,
    extrema: pd.DataFrame,
    out_dir: Path,
) -> None:
    recent = frame[frame["date"] >= frame["date"].iloc[-1] - pd.DateOffset(months=180)]
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=240)
    ax.plot(recent["date"], recent["ssn"], color="black", lw=1.2, label="Observed raw SSN")
    ax.fill_between(
        band["date"],
        band["q10"],
        band["q90"],
        color="#D55E00",
        alpha=0.18,
        linewidth=0,
        label="Five-seed 10-90% range",
    )
    ax.plot(
        band["date"],
        band["median"],
        color="#D55E00",
        lw=1.8,
        label="PR24 recursive block24 clip0 median",
    )
    for seed, curve in curves.items():
        clip = curve[curve["mode"] == "clip0"]
        ax.plot(clip["date"], clip["pred"], color="#999999", lw=0.55, alpha=0.45)
    for column, label, color in (
        ("cycle25_trough_date", "Cycle 25 trough", "#0072B2"),
        ("cycle26_peak_date", "Cycle 26 peak", "#009E73"),
    ):
        for _, row in extrema.iterrows():
            date = pd.Timestamp(f"{row[column]}-01")
            ax.axvline(date, color=color, lw=0.5, alpha=0.15)
    band_indexed = band.set_index("date")
    for column, label, color in (
        ("cycle25_trough_date", "Cycle 25 trough", "#0072B2"),
        ("cycle26_peak_date", "Cycle 26 peak", "#009E73"),
    ):
        median_date = pd.Timestamp(f"{extrema[column].mode().iloc[0]}-01")
        median_value = float(band_indexed.loc[median_date, "median"])
        ax.scatter(median_date, median_value, color=color, s=40, zorder=6)
        ax.annotate(
            f"{label}\n{median_date:%Y-%m}, {median_value:.1f}",
            (median_date, median_value),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax.set_ylabel("Monthly SSN")
    ax.set_title(
        "PR24 final all-history retrain | origin 2026-10 | formal output clip0",
        loc="left",
    )
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            out_dir / f"pr24_future_profiles_2026-10{suffix}",
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=240)
    axes[0].boxplot(
        [
            extrema["cycle25_trough_smoothed"],
            extrema["cycle26_peak_smoothed"],
        ],
        tick_labels=["Cycle 25 trough", "Cycle 26 peak"],
        patch_artist=True,
    )
    for position, values in enumerate(
        (extrema["cycle25_trough_smoothed"], extrema["cycle26_peak_smoothed"]),
        start=1,
    ):
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[0].scatter(
            position + jitter,
            values,
            color="black",
            s=14,
            zorder=4,
        )
    axes[0].set_ylabel("13-month smoothed SSN")
    axes[0].set_title("Amplitude across five seeds", loc="left", fontsize=9)
    rng = np.random.default_rng(20211121)
    for column, baseline, label, color in (
        ("cycle25_trough_date", 1.0, "Cycle 25 trough", "#0072B2"),
        ("cycle26_peak_date", 0.0, "Cycle 26 peak", "#009E73"),
    ):
        jitter = rng.uniform(-0.12, 0.12, len(extrema))
        dates = pd.to_datetime(extrema[column] + "-01")
        axes[1].scatter(
            dates,
            baseline + jitter,
            label=label,
            color=color,
            alpha=0.75,
        )
        for date, y, seed in zip(dates, baseline + jitter, extrema["seed"]):
            axes[1].annotate(
                str(seed),
                (date, y),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=6,
                color=color,
            )
    axes[1].set_yticks([0, 1], ["Cycle 26 peak", "Cycle 25 trough"])
    axes[1].set_title("Timing across five seeds", loc="left", fontsize=9)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="x", color="#DDDDDD", lw=0.5)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            out_dir / f"pr24_future_extrema_2026-10{suffix}",
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config_root", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--exp_id", default="FINAL-PR24-2026-10")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--block_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="2019,2020,2021,2022,2023")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen final-training protocol")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data_csv, parse_dates=["date"])
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("date must be monotonic increasing")
    if frame["ssn"].isna().any():
        raise ValueError("ssn contains missing values")

    scaler = StandardScaler()
    data_z = scaler.fit_transform(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    seed_values = [int(value) for value in args.seeds.split(",") if value]
    if args.plot_only:
        curves = {
            seed: pd.read_csv(args.out_dir / f"forecast_seed{seed}.csv", parse_dates=["date"])
            for seed in seed_values
        }
        extrema = pd.read_csv(args.out_dir / "forecast_extrema_by_seed.csv")
        band, aggregate = summarize_forecast(curves, extrema)
        band.to_csv(args.out_dir / "forecast_seed_band.csv", index=False)
        aggregate.to_csv(args.out_dir / "forecast_extrema_aggregate.csv", index=False)
        plot_forecast(frame, curves, band, extrema, args.out_dir)
        print(aggregate.to_string(index=False))
        return
    curves = {}
    extrema_rows = []
    checkpoint_paths = []
    for seed in seed_values:
        checkpoint_path, history = train_one(
            seed=seed,
            frame=frame,
            data_z=data_z,
            data_path=args.data_csv,
            pred_len=args.pred_len,
            d_model=args.d_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_root=args.checkpoint_root,
            config_root=args.config_root,
            exp_id=args.exp_id,
        )
        checkpoint_paths.append(str(checkpoint_path))
        forecast = forecast_one(
            checkpoint_path,
            frame,
            scaler,
            data_z,
            args.horizon,
            args.block_size,
        )
        forecast.to_csv(args.out_dir / f"forecast_seed{seed}.csv", index=False)
        curves[seed] = forecast
        extrema = detect_cycle25_trough_and_cycle26_peak(
            frame,
            forecast[forecast["mode"] == "clip0"].reset_index(drop=True),
            args.horizon,
        )
        extrema["seed"] = seed
        extrema_rows.append(extrema)

    extrema = pd.DataFrame(extrema_rows)
    extrema.to_csv(args.out_dir / "forecast_extrema_by_seed.csv", index=False)
    band, aggregate = summarize_forecast(curves, extrema)
    band.to_csv(args.out_dir / "forecast_seed_band.csv", index=False)
    aggregate.to_csv(args.out_dir / "forecast_extrema_aggregate.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": args.exp_id,
        "origin": frame["date"].iloc[-1].strftime("%Y-%m"),
        "data_csv": str(args.data_csv.relative_to(ROOT)),
        "data_sha256": sha256(args.data_csv),
        "formal_forecast": "clip0 recursive block24",
        "training_protocol": "all-history fixed-epoch retrain; no validation early stopping",
        "d_model": args.d_model,
        "epochs": args.epochs,
        "seeds": seed_values,
        "horizon": args.horizon,
        "checkpoints": checkpoint_paths,
        "extrema_aggregate": aggregate.to_dict(orient="records"),
    }
    (args.out_dir / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_forecast(frame, curves, band, extrema, args.out_dir)
    print(extrema.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(args.out_dir)


if __name__ == "__main__":
    main()
