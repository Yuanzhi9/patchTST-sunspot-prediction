"""Download and normalize public F10.7 and WSO polar-field observations.

This script never fills long gaps. All source files are cached under
PatchTST_supervised/dataset/external/raw, and processed tables carry explicit
coverage and validity fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "PatchTST_supervised/dataset/external"
RAW_F107 = EXTERNAL / "raw/f107"
RAW_WSO = EXTERNAL / "raw/wso_polar"
RAW_AA = EXTERNAL / "raw/aa"
PROCESSED = EXTERNAL / "processed"

F107_MONTHLY_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-radio/noontime-flux/penticton/"
    "penticton_observed/penticton_solflux-monthly-average.txt"
)
F107_DAILY_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-radio/noontime-flux/penticton/"
    "pent_noontime-flux_{year}.txt"
)
NOAA_SOLAR_CYCLE_URL = (
    "https://services.swpc.noaa.gov/json/"
    "solar-cycle/observed-solar-cycle-indices.json"
)
WSO_POLAR_URL = "http://wso.stanford.edu/Polar.html"


def download(url: str, path: Path, refresh: bool = False) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    return payload


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_f107_monthly(payload: bytes) -> pd.DataFrame:
    rows = []
    for line in payload.decode("ascii", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        year, month = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:5]]
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=1),
                "f107_observed": values[0],
                "f107_adjusted": values[1],
                "f107_absolute": values[2],
                "n_days": np.nan,
                "coverage": np.nan,
                "source": "NCEI monthly average",
            }
        )
    return pd.DataFrame(rows)


def parse_f107_daily(payload: bytes) -> pd.DataFrame:
    rows = []
    for line in payload.decode("ascii", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 3 or not re.fullmatch(r"\d{6}", fields[0]):
            continue
        station = fields[1]
        raw = fields[2]
        value = np.nan if raw.upper() in {"XXXX", "XXX", ""} else float(raw)
        rows.append(
            {
                "date": datetime.strptime(fields[0], "%y%m%d"),
                "station": station,
                "f107_observed": value,
            }
        )
    return pd.DataFrame(rows)


def build_f107(refresh: bool = False) -> pd.DataFrame:
    monthly_payload = download(
        F107_MONTHLY_URL, RAW_F107 / "penticton_monthly.txt", refresh=refresh
    )
    monthly = parse_f107_monthly(monthly_payload)

    daily_frames = []
    for year in range(2015, 2027):
        url = F107_DAILY_URL.format(year=year)
        payload = download(
            url,
            RAW_F107 / f"pent_noontime-flux_{year}.txt",
            refresh=refresh,
        )
        daily_frames.append(parse_f107_daily(payload))
    daily = pd.concat(daily_frames, ignore_index=True)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month

    daily_monthly = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(
            f107_observed=("f107_observed", "mean"),
            n_days=("f107_observed", "count"),
        )
        .assign(date=lambda frame: pd.to_datetime(dict(year=frame.year, month=frame.month, day=1)))
    )
    daily_monthly["expected_days"] = [
        monthrange(int(year), int(month))[1]
        for year, month in zip(daily_monthly["year"], daily_monthly["month"])
    ]
    daily_monthly["coverage"] = daily_monthly["n_days"] / daily_monthly["expected_days"]

    historical = monthly.loc[monthly["date"] < "2015-01-01"].copy()
    recent = daily_monthly.loc[daily_monthly["date"] >= "2015-01-01"].copy()

    # The NCEI daily archive ends in September 2026. Extend only the monthly
    # value from the official SWPC JSON, preserving its source label.
    noaa_payload = download(
        NOAA_SOLAR_CYCLE_URL,
        RAW_F107 / "swpc_observed_solar_cycle.json",
        refresh=refresh,
    )
    noaa = pd.DataFrame(json.loads(noaa_payload))
    noaa = noaa.loc[noaa["time-tag"] >= "2026-10", ["time-tag", "f10.7"]].copy()
    noaa = noaa.loc[noaa["f10.7"] > 0].rename(columns={"time-tag": "month", "f10.7": "f107_observed"})
    if not noaa.empty:
        noaa["date"] = pd.to_datetime(noaa["month"] + "-01")
        noaa["n_days"] = np.nan
        noaa["coverage"] = np.nan
        noaa["source"] = "NOAA SWPC monthly"
        noaa_merge = noaa[
            ["date", "f107_observed", "n_days", "coverage", "source"]
        ].rename(
            columns={
                "f107_observed": "f107_observed_noaa",
                "n_days": "n_days_noaa",
                "coverage": "coverage_noaa",
                "source": "source_noaa",
            }
        )
        recent = recent.merge(noaa_merge, on="date", how="outer")
        for column in ("f107_observed", "n_days", "coverage", "source"):
            if column not in recent:
                recent[column] = np.nan
            recent[column] = recent[column].combine_first(
                recent[f"{column}_noaa"]
            )
            recent = recent.drop(columns=f"{column}_noaa")

    recent["f107_adjusted"] = np.nan
    recent["f107_absolute"] = np.nan
    recent["source"] = recent["source"].fillna("NCEI daily archive")
    output = pd.concat(
        [
            historical[
                [
                    "date",
                    "f107_observed",
                    "f107_adjusted",
                    "f107_absolute",
                    "n_days",
                    "coverage",
                    "source",
                ]
            ],
            recent[
                [
                    "date",
                    "f107_observed",
                    "f107_adjusted",
                    "f107_absolute",
                    "n_days",
                    "coverage",
                    "source",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values("date")
    output = output.drop_duplicates("date", keep="last")
    full_months = pd.DataFrame(
        {"date": pd.date_range("1947-02-01", "2026-10-01", freq="MS")}
    )
    output = full_months.merge(output, on="date", how="left")

    # Only complete bounded gaps of at most two months are interpolated.
    # Longer gaps remain entirely missing. The original value is always kept.
    original = output["f107_observed"].copy()
    interpolated = original.interpolate(method="linear", limit_area="inside")
    missing = original.isna()
    missing_groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    fill_mask = pd.Series(False, index=output.index)
    for _, positions in original.loc[missing].groupby(missing_groups.loc[missing]).groups.items():
        positions = list(positions)
        bounded = positions[0] > 0 and positions[-1] < len(output) - 1
        if bounded and len(positions) <= 2:
            fill_mask.loc[positions] = True
    output["f107_filled_short_gap"] = original.where(~fill_mask, interpolated)
    output["short_gap_flag"] = fill_mask
    output["missing"] = output["f107_observed"].isna().astype(int)
    return output


def parse_wso_polar(payload: bytes) -> pd.DataFrame:
    text = payload.decode("ascii", errors="replace")
    pattern = re.compile(
        r"^(\d{4}):(\d{2}):(\d{2})_"
        r".*?\s+([-0-9X]+)N\s+([-0-9X]+)S\s+([-0-9X]+)Avg\s+"
        r"20nhz\s+filt:\s+([-0-9X]+)Nf\s+([-0-9X]+)Sf\s+([-0-9X]+)Avgf"
    )

    def number(value: str) -> float:
        if value.upper().startswith("X"):
            return np.nan
        return float(value)

    rows = []
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        year, month, day = map(int, match.groups()[:3])
        values = [number(value) for value in match.groups()[3:]]
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=day),
                "polar_north_raw": values[0],
                "polar_south_raw": values[1],
                "polar_average_raw": values[2],
                "polar_north_filtered": values[3],
                "polar_south_filtered": values[4],
                "polar_average_filtered": values[5],
            }
        )
    return pd.DataFrame(rows).sort_values("date")


def build_wso(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = download(WSO_POLAR_URL, RAW_WSO / "Polar.html", refresh=refresh)
    ten_day = parse_wso_polar(payload)
    ten_day["year"] = ten_day["date"].dt.year
    ten_day["month"] = ten_day["date"].dt.month
    monthly = (
        ten_day.groupby(["year", "month"], as_index=False)
        .agg(
            polar_north_filtered=("polar_north_filtered", "mean"),
            polar_south_filtered=("polar_south_filtered", "mean"),
            polar_average_filtered=("polar_average_filtered", "mean"),
            n_10day=("polar_average_filtered", "count"),
        )
        .assign(date=lambda frame: pd.to_datetime(dict(year=frame.year, month=frame.month, day=1)))
    )
    full_months = pd.DataFrame(
        {"date": pd.date_range("1976-05-01", "2026-10-01", freq="MS")}
    )
    monthly = full_months.merge(monthly, on="date", how="left")
    monthly["coverage"] = monthly["n_10day"] / 3.0
    monthly.loc[monthly["n_10day"] < 2, monthly.filter(like="polar_").columns] = np.nan
    monthly["missing"] = monthly["polar_average_filtered"].isna().astype(int)
    return ten_day, monthly


def parse_aa_iaga(payload: bytes) -> pd.DataFrame:
    rows = []
    for line in payload.decode("ascii", errors="replace").splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields[0]):
            continue
        if len(fields) < 2:
            raise ValueError(f"malformed aa record: {line}")
        if len(fields) < 4 or fields[3] in {"999", "-", ""}:
            value = np.nan
        else:
            value = float(fields[3])
        rows.append(
            {
                "datetime": pd.Timestamp(f"{fields[0]} {fields[1]}"),
                "aa": value,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no aa records parsed")
    return frame.sort_values("datetime").reset_index(drop=True)


def build_aa() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    candidates = sorted(RAW_AA.glob("*IAGA2002.dat"))
    if len(candidates) != 1:
        raise ValueError(f"expected one aa IAGA2002 raw file, found {len(candidates)}")
    raw_path = candidates[0]
    three_hour = parse_aa_iaga(raw_path.read_bytes())
    three_hour["date"] = three_hour["datetime"].dt.normalize()
    three_hour["year"] = three_hour["datetime"].dt.year
    three_hour["month"] = three_hour["datetime"].dt.month

    daily = (
        three_hour.groupby("date", as_index=False)
        .agg(
            aa_daily_mean=("aa", "mean"),
            aa_daily_max=("aa", "max"),
            n_3h=("aa", "count"),
        )
    )
    daily["coverage"] = daily["n_3h"] / 8.0
    daily.loc[daily["n_3h"] < 7, ["aa_daily_mean", "aa_daily_max"]] = np.nan

    monthly = (
        three_hour.groupby(["year", "month"], as_index=False)
        .agg(
            aa_mean=("aa", "mean"),
            aa_max=("aa", "max"),
            n_3h=("aa", "count"),
        )
        .assign(
            date=lambda frame: pd.to_datetime(
                dict(year=frame["year"], month=frame["month"], day=1)
            )
        )
    )
    monthly["expected_3h"] = [
        8 * monthrange(int(year), int(month))[1]
        for year, month in zip(monthly["year"], monthly["month"])
    ]
    monthly["coverage"] = monthly["n_3h"] / monthly["expected_3h"]
    monthly.loc[monthly["coverage"] < 0.8, ["aa_mean", "aa_max"]] = np.nan
    quantiles = (
        three_hour.dropna(subset=["aa"])
        .groupby(["year", "month"])["aa"]
        .quantile(0.9)
    )
    monthly["aa_p90"] = [
        float(quantiles.loc[(int(year), int(month))])
        if coverage >= 0.8 and (int(year), int(month)) in quantiles.index
        else np.nan
        for year, month, coverage in zip(
            monthly["year"], monthly["month"], monthly["coverage"]
        )
    ]
    monthly["missing"] = monthly["aa_mean"].isna().astype(int)
    monthly = monthly[
        ["date", "aa_mean", "aa_max", "aa_p90", "n_3h", "coverage", "missing"]
    ]
    return daily, monthly, raw_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download F10.7/WSO sources instead of using cached raw files.",
    )
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    f107 = build_f107(refresh=args.refresh)
    polar_10day, polar_monthly = build_wso(refresh=args.refresh)
    aa_daily, aa_monthly, aa_raw_path = build_aa()

    f107_path = PROCESSED / "f107_monthly_1947_2026.csv"
    polar_path = PROCESSED / "wso_polar_10day_1976_2026.csv"
    polar_monthly_path = PROCESSED / "wso_polar_monthly_1976_2026.csv"
    aa_daily_path = PROCESSED / "aa_daily_1869_2026.csv"
    aa_monthly_path = PROCESSED / "aa_monthly_1869_2026.csv"
    f107.to_csv(f107_path, index=False)
    polar_10day.to_csv(polar_path, index=False)
    polar_monthly.to_csv(polar_monthly_path, index=False)
    aa_daily.to_csv(aa_daily_path, index=False)
    aa_monthly.to_csv(aa_monthly_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "f107": {
            "url": F107_MONTHLY_URL,
            "daily_url_pattern": F107_DAILY_URL,
            "coverage": [str(f107["date"].min().date()), str(f107["date"].max().date())],
            "rows": len(f107),
            "missing_months": int(f107["missing"].sum()),
            "sha256": sha256(f107_path.read_bytes()),
        },
        "wso_polar": {
            "url": WSO_POLAR_URL,
            "coverage": [
                str(polar_10day["date"].min().date()),
                str(polar_10day["date"].max().date()),
            ],
            "valid_10day_rows": int(polar_10day["polar_average_filtered"].notna().sum()),
            "last_valid_10day": str(
                polar_10day.loc[
                    polar_10day["polar_average_filtered"].notna(), "date"
                ].max().date()
            ),
            "monthly_rows": len(polar_monthly),
            "monthly_missing": int(polar_monthly["missing"].sum()),
            "sha256": sha256(polar_monthly_path.read_bytes()),
        },
        "aa": {
            "raw_file": str(aa_raw_path.relative_to(ROOT)),
            "raw_sha256": sha256(aa_raw_path.read_bytes()),
            "daily_rows": len(aa_daily),
            "monthly_rows": len(aa_monthly),
            "monthly_missing": int(aa_monthly["missing"].sum()),
            "coverage": [
                str(aa_monthly["date"].min().date()),
                str(aa_monthly["date"].max().date()),
            ],
            "monthly_sha256": sha256(aa_monthly_path.read_bytes()),
        },
    }
    (PROCESSED / "external_indices_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
