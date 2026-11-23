"""Compare the PR24 Cycle 25/26 forecast with published SC26 forecasts.

The Dang and Gerceker curves are digitized from the published figures. They are
intended for visual comparison of shape, timing, and amplitude, not for
asserting numerical equivalence with the source models.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "analysis/pr24_future_forecast"
OUT_DIR = ROOT / "analysis/literature_comparison"
DATA_CSV = ROOT / "PatchTST_supervised/dataset/sunspot_with_cycle_monthly_2026-10.csv"
SOURCE_CACHE = Path("/tmp/pr24_literature_sources")

SOURCES = {
    "dang_fig6": (
        "https://arxiv.org/html/2203.05757v2/figs/future_preds.png"
    ),
    "gerceker_fig5": (
        "https://media.springernature.com/original/springer-static/image/"
        "art%3A10.1007%2Fs11207-025-02577-y/MediaObjects/"
        "11207_2025_2577_Fig5_HTML.png"
    ),
}


def fetch_image(name: str) -> Image.Image:
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    path = SOURCE_CACHE / f"{name}.png"
    if not path.exists():
        with urllib.request.urlopen(SOURCES[name], timeout=30) as response:
            path.write_bytes(response.read())
    return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    red, green, blue = (
        image[:, :, 0].astype(float),
        image[:, :, 1].astype(float),
        image[:, :, 2].astype(float),
    )
    if color == "red":
        return (red > 150) & (red > 1.4 * green) & (red > 1.25 * blue)
    if color == "blue":
        return (blue > 90) & (blue > 1.2 * red) & (blue > 1.05 * green)
    raise ValueError(f"unsupported color: {color}")


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, _ = label(mask, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        raise RuntimeError("no color component found")
    component = int(np.argmax(counts[1:]) + 1)
    return labels == component


def extract_column_series(
    mask: np.ndarray,
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for x in range(x_bounds[0], x_bounds[1] + 1):
        ys = np.where(mask[y_bounds[0] : y_bounds[1] + 1, x])[0]
        if len(ys):
            x_values.append(float(x))
            y_values.append(float(np.median(ys) + y_bounds[0]))
    if len(x_values) < 20:
        raise RuntimeError("too few digitized points")
    return np.asarray(x_values), np.asarray(y_values)


def digitize_dang() -> pd.DataFrame:
    image = np.asarray(fetch_image("dang_fig6"))
    x_bounds = (158, 1134)
    y_bounds = (71, 521)

    red_mask = largest_component(color_mask(image, "red"))
    red_x, red_y = extract_column_series(red_mask, x_bounds, y_bounds)

    blue_mask = color_mask(image, "blue")
    # Remove the two short colored legend samples in the upper-right corner.
    blue_mask[88:95, 970:1012] = False
    blue_mask[111:118, 970:1012] = False
    blue_x, blue_y = extract_column_series(blue_mask, x_bounds, y_bounds)

    def convert(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        years = (
            2008.0
            + (x - x_bounds[0])
            / (x_bounds[1] - x_bounds[0])
            * (2041.5 - 2008.0)
        )
        values = 175.0 * (y_bounds[1] - y) / (y_bounds[1] - y_bounds[0])
        return pd.DataFrame({"year": years, "value": values})

    red = convert(red_x, red_y).rename(columns={"value": "nasa"})
    blue = convert(blue_x, blue_y).rename(columns={"value": "dang_xgboost_dl"})
    return red.merge(blue, on="year", how="outer").sort_values("year")


def digitize_gerceker() -> pd.DataFrame:
    image = np.asarray(fetch_image("gerceker_fig5"))
    mask = color_mask(image, "red")
    mask[:2, :] = False
    mask[527:, :] = False
    mask[:, :155] = False
    mask[:, 1276:] = False
    # Remove legend samples; the forecast curve is far below this region after 2037.
    mask[:110, 900:1319] = False
    x, y = extract_column_series(mask, (155, 1275), (2, 526))
    years = 2020.0 + (x - 155.0) / (1275.0 - 155.0) * (2042.0 - 2020.0)
    # Tick positions in the published figure are y=507 -> 0 and y=21 -> 250.
    values = (507.0 - y) * (250.0 / (507.0 - 21.0))
    return pd.DataFrame({"year": years, "gerceker_pred1": values}).sort_values(
        "year"
    )


def load_pr24() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(FORECAST_DIR.glob("forecast_seed[0-9][0-9][0-9][0-9].csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        frames.append(frame[frame["mode"] == "raw"][["date", "pred"]])
    if not frames:
        raise FileNotFoundError("PR24 raw forecast files are missing")
    monthly = pd.concat(
        [frame.set_index("date")["pred"].rename(f"seed{i}") for i, frame in enumerate(frames)],
        axis=1,
    )
    summary = pd.DataFrame(
        {
            "median": monthly.median(axis=1),
            "min_seed": monthly.min(axis=1),
            "max_seed": monthly.max(axis=1),
        }
    )
    summary.index.name = "date"
    observed = pd.read_csv(DATA_CSV, parse_dates=["date"])
    observed = observed.loc[observed["date"] >= "2020-01-01", ["date", "ssn"]]
    return summary.reset_index(), observed


def monthly_interpolation(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    clean = frame[["year", column]].dropna().sort_values("year")
    dates = pd.date_range("2020-01-01", "2042-12-01", freq="MS")
    decimal_year = dates.year + (dates.month - 0.5) / 12.0
    values = np.interp(
        decimal_year,
        clean["year"].to_numpy(float),
        clean[column].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    return pd.DataFrame({"date": dates, column: values})


def annual_means(frame: pd.DataFrame, column: str) -> pd.Series:
    clean = frame.dropna(subset=["date", column]).copy()
    clean["year"] = clean["date"].dt.year
    return clean.groupby("year")[column].mean()


def plot(
    pr24: pd.DataFrame,
    observed: pd.DataFrame,
    dang: pd.DataFrame,
    gerceker: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dang_monthly = (
        monthly_interpolation(dang, "dang_xgboost_dl")
        .merge(monthly_interpolation(dang, "nasa"), on="date", how="outer")
        .sort_values("date")
    )
    gerceker_monthly = monthly_interpolation(gerceker, "gerceker_pred1")

    kumar = pd.DataFrame(
        {
            "year": np.arange(2025, 2038),
            "kumar_annual": [
                112.4,
                128.6,
                122.2,
                90.1,
                51.5,
                25.8,
                10.9,
                10.4,
                33.4,
                93.6,
                137.9,
                165.4,
                155.2,
            ],
        }
    )

    pr24_curve = pd.concat(
        [
            observed.rename(columns={"ssn": "median"}).assign(
                min_seed=np.nan, max_seed=np.nan
            ),
            pr24,
        ],
        ignore_index=True,
    ).sort_values("date")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 13),
        dpi=220,
        gridspec_kw={"height_ratios": [1.35, 0.9, 1.0]},
    )

    ax = axes[0]
    ax.plot(
        observed["date"],
        observed["ssn"],
        color="black",
        lw=1.2,
        label="Observed monthly mean",
    )
    ax.fill_between(
        pr24["date"],
        pr24["min_seed"],
        pr24["max_seed"],
        color="#D55E00",
        alpha=0.15,
        linewidth=0,
        label="PR24 five-seed range",
    )
    ax.plot(
        pr24["date"],
        pr24["median"],
        color="#D55E00",
        lw=2.2,
        label="PR24 recursive block24 median",
    )
    ax.plot(
        gerceker_monthly["date"],
        gerceker_monthly["gerceker_pred1"],
        color="#009E73",
        lw=1.3,
        label="Gerceker Prediction 1 (digitized)",
    )
    ax.plot(
        dang_monthly["date"],
        dang_monthly["dang_xgboost_dl"],
        color="#0072B2",
        lw=1.3,
        label="Dang XGBoost-DL (digitized)",
    )
    ax.plot(
        dang_monthly["date"],
        dang_monthly["nasa"],
        color="#777777",
        lw=1.0,
        ls="--",
        label="NASA forecast shown by Dang (digitized)",
    )
    ax.axvline(pd.Timestamp("2026-10-01"), color="black", lw=0.8, alpha=0.35)
    ax.text(pd.Timestamp("2026-11-01"), 3, "origin", fontsize=7, va="bottom")
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2042-01-01"))
    ax.set_ylim(-5, 225)
    ax.set_ylabel("Monthly mean SSN")
    ax.set_title(
        "A. Raw monthly profiles: PR24 versus published SC26 forecasts",
        loc="left",
        fontsize=10,
    )
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    ax = axes[1]
    pr24_ann = annual_means(pr24_curve, "median")
    ger_ann = annual_means(gerceker_monthly, "gerceker_pred1")
    dang_ann = annual_means(dang_monthly, "dang_xgboost_dl")
    nasa_ann = annual_means(dang_monthly, "nasa")
    ax.plot(pr24_ann.index, pr24_ann.values, "o-", color="#D55E00", lw=2, label="PR24")
    ax.plot(
        ger_ann.index,
        ger_ann.values,
        "s-",
        color="#009E73",
        lw=1.3,
        label="Gerceker (from digitized curve)",
    )
    ax.plot(
        dang_ann.index,
        dang_ann.values,
        "^-",
        color="#0072B2",
        lw=1.3,
        label="Dang (from digitized curve)",
    )
    ax.plot(
        nasa_ann.index,
        nasa_ann.values,
        "v--",
        color="#777777",
        lw=1.0,
        label="NASA (from digitized curve)",
    )
    ax.plot(
        kumar["year"],
        kumar["kumar_annual"],
        "D-",
        color="#CC79A7",
        lw=1.4,
        label="Kumar & Kumar annual SSN",
    )
    ax.set_xlim(2024.5, 2042.5)
    ax.set_ylim(0, 190)
    ax.set_ylabel("Annual mean SSN")
    ax.set_title(
        "B. Annual means remove month-to-month rendering differences",
        loc="left",
        fontsize=10,
    )
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    ax = axes[2]
    rows = [
        (
            "PR24",
            pd.Timestamp("2030-10-01"),
            pd.Timestamp("2035-06-01"),
            106.95,
            (101.42, 109.69),
            "#D55E00",
        ),
        (
            "Gerceker",
            pd.Timestamp("2030-06-01"),
            pd.Timestamp("2035-06-01"),
            166.05,
            (150.6, 181.5),
            "#009E73",
        ),
        (
            "Dang",
            pd.Timestamp("2031-01-01"),
            pd.Timestamp("2035-11-01"),
            164.62,
            None,
            "#0072B2",
        ),
        (
            "NASA / Dang",
            pd.Timestamp("2031-01-01"),
            pd.Timestamp("2034-12-01"),
            161.2,
            None,
            "#777777",
        ),
        (
            "Kumar",
            pd.Timestamp("2032-06-01"),
            pd.Timestamp("2036-06-01"),
            165.35,
            None,
            "#CC79A7",
        ),
    ]
    for y, (label, trough, peak, amplitude, interval, color) in enumerate(rows):
        ax.plot([trough, peak], [y, y], color=color, lw=1.3, alpha=0.6)
        ax.scatter(trough, y, marker="o", s=42, color=color, zorder=3)
        ax.scatter(peak, y, marker="^", s=55, color=color, zorder=3)
        if interval is None:
            text = f"{amplitude:.1f}"
        else:
            text = f"{amplitude:.1f} [{interval[0]:.1f}, {interval[1]:.1f}]"
        ax.annotate(
            text,
            (peak, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(pd.Timestamp("2029-06-01"), pd.Timestamp("2037-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(
        "C. Timing and amplitude: circles are Cycle 25 trough, triangles Cycle 26 peak",
        loc="left",
        fontsize=10,
    )
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.01,
        0.005,
        "Literature curves are digitized from the cited published figures for shape comparison only. "
        "They are not the original model arrays and should not be read as a precision benchmark.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUT_DIR / f"pr24_vs_literature_sc26_2026-11-23{suffix}",
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)

    pr24_monthly = pr24[["date", "median"]].rename(columns={"median": "pr24"})
    monthly = pd.DataFrame({"date": pr24["date"]}).merge(
        pr24_monthly,
        on="date",
        how="outer",
    ).merge(gerceker_monthly, on="date", how="outer").merge(
        dang_monthly, on="date", how="outer"
    )
    monthly.sort_values("date").to_csv(
        OUT_DIR / "literature_sc26_monthly_digitized.csv", index=False
    )
    pd.DataFrame(
        rows,
        columns=["source", "trough_date", "peak_date", "peak_value", "peak_range", "color"],
    ).to_csv(OUT_DIR / "sc26_extrema_comparison.csv", index=False)


def main() -> None:
    pr24, observed = load_pr24()
    dang = digitize_dang()
    gerceker = digitize_gerceker()
    plot(pr24, observed, dang, gerceker)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
