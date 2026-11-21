"""Create centered and causal Gaussian-smoothed SSN datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
CENTERED = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_gaussian_sigma3p6_centered_2026-10.csv"
CAUSAL = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_gaussian_sigma3p6_causal_2026-10.csv"
SIGMA = 3.6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    output = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - radius)
        past = values[start : index + 1][::-1]
        weights = kernel[: len(past)]
        output[index] = np.sum(past * weights) / np.sum(weights)
    return output


def save_copy(frame: pd.DataFrame, values: np.ndarray, path: Path) -> None:
    output = frame.copy()
    output["ssn_raw"] = output["ssn"]
    output["ssn"] = values
    output.to_csv(path, index=False)
    print(path, sha256(path))


def main() -> None:
    frame = pd.read_csv(SOURCE, parse_dates=["date"])
    raw = frame["ssn"].to_numpy(dtype=float)
    centered = gaussian_filter1d(raw, sigma=SIGMA, mode="nearest")
    causal = causal_gaussian(raw, SIGMA)
    save_copy(frame, centered, CENTERED)
    save_copy(frame, causal, CAUSAL)


if __name__ == "__main__":
    main()
