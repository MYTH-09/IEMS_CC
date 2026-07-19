"""
IEMS – NASA POWER API Client
=============================
Fetches hourly solar irradiance and meteorological data from NASA's POWER
(Prediction Of Worldwide Energy Resources) service.

API endpoint : https://power.larc.nasa.gov/api/temporal/hourly/point
Community    : RE (Renewable Energy)
Parameters   : GHI, DNI, DHI, 2m temperature, 2m wind speed
Timezone     : UTC on fetch → converted to IST (Asia/Kolkata) on return

Usage
-----
    from src.solar.nasa_power import fetch_solar_data

    df = fetch_solar_data(
        lat=27.4924, lon=77.6737,
        start_date="2019-05-01", end_date="2021-10-31",
        location_name="Mathura"
    )
    # Returns hourly DataFrame with columns: timestamp, ghi, dni, dhi, temp_air, wind_speed
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from loguru import logger


# ── API config ────────────────────────────────────────────────────────────────
_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA POWER parameter codes → clean names
_PARAM_MAP = {
    "ALLSKY_SFC_SW_DWN": "ghi",        # Global Horizontal Irradiance (W/m²)
    "ALLSKY_SFC_SW_DNI": "dni",        # Direct Normal Irradiance (W/m²)
    "ALLSKY_SFC_SW_DIFF": "dhi",       # Diffuse Horizontal Irradiance (W/m²)  ← two F's
    "T2M":               "temp_air",   # Air temperature at 2 m (°C)
    "WS2M":              "wind_speed", # Wind speed at 2 m (m/s)
}

_FILL_VALUES  = {-99.0, -999.0}   # hourly uses -99; handle both variants defensively
_REQUEST_SLEEP = 3.0      # Seconds between requests (be a good API citizen)
_TIMEOUT       = 90       # Request timeout (seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_year_chunk(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetch one calendar year of hourly data from NASA POWER.
    Returns a DataFrame with clean column names in UTC.

    NASA POWER max window is 366 days, so we call once per year.
    """
    params = {
        "parameters":    ",".join(_PARAM_MAP.keys()),
        "community":     "RE",
        "longitude":     round(lon, 4),
        "latitude":      round(lat, 4),
        "start":         f"{year}0101",
        "end":           f"{year}1231",
        "format":        "JSON",
        "time-standard": "UTC",
    }

    logger.debug(f"    GET NASA POWER: lat={lat}, lon={lon}, year={year}")
    resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
    if not resp.ok:
        logger.error(f"NASA POWER API error {resp.status_code}: {resp.text[:400]}")
    resp.raise_for_status()

    raw = resp.json()
    hourly = raw["properties"]["parameter"]   # dict of {param: {timestamp: value}}

    # Build DataFrame from the nested dict
    first_param = next(iter(hourly))
    ts_keys = list(hourly[first_param].keys())   # "YYYYMMDDHH" strings in UTC

    data = {_PARAM_MAP[p]: [hourly[p][ts] for ts in ts_keys]
            for p in _PARAM_MAP if p in hourly}

    df = pd.DataFrame(data)
    df["timestamp_utc"] = pd.to_datetime(ts_keys, format="%Y%m%d%H", utc=True)

    # Replace NASA fill values with NaN
    irr_cols = ["ghi", "dni", "dhi"]
    for col in df.columns:
        if col != "timestamp_utc":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda v: np.nan if v in _FILL_VALUES else v)

    # Irradiance can't be negative — clip nighttime numerical noise
    for col in irr_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _utc_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UTC timestamp column to IST (Asia/Kolkata, UTC+5:30).
    IST offsets land on half-hours (:00 and :30), so after conversion
    the timestamps will be at :30 marks.  The caller (pv_model) resamples
    these to clean 15-min IST slots aligned with the load data.
    """
    df = df.copy()
    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)           # drop tzinfo → naive IST datetimes
    )
    df = df.drop(columns=["timestamp_utc"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_solar_data(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    location_name: str = "",
    cache_dir: str = "data/solar",
) -> pd.DataFrame:
    """
    Fetch hourly solar irradiance + meteorological data from NASA POWER
    for a location and date range.

    Results are cached to `{cache_dir}/{location_name}_{year}.parquet`
    so repeated calls don't hit the API again.

    Parameters
    ----------
    lat, lon      : Decimal degrees (WGS84).
    start_date    : "YYYY-MM-DD" — earliest date needed.
    end_date      : "YYYY-MM-DD" — latest date needed.
    location_name : Label used for cache filenames and log messages.
    cache_dir     : Directory to store/read cached parquet files.

    Returns
    -------
    pd.DataFrame
        Hourly rows, IST timezone (naive), columns:
        timestamp, ghi, dni, dhi, temp_air, wind_speed
    """
    os.makedirs(cache_dir, exist_ok=True)

    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    years = list(range(start.year, end.year + 1))

    label = location_name or f"{lat:.2f}N_{lon:.2f}E"
    logger.info(f"Fetching solar data for {label} | {start_date} → {end_date}")

    frames = []
    for year in years:
        cache_path = os.path.join(cache_dir, f"{label}_{year}.parquet")

        if os.path.exists(cache_path):
            logger.info(f"  Cache hit: {os.path.basename(cache_path)}")
            chunk = pd.read_parquet(cache_path)
        else:
            logger.info(f"  Fetching from NASA POWER: {label}, {year} ...")
            chunk_utc = _fetch_year_chunk(lat, lon, year)
            chunk     = _utc_to_ist(chunk_utc)
            chunk.to_parquet(cache_path, index=False)
            logger.info(f"  Cached → {cache_path}")
            time.sleep(_REQUEST_SLEEP)

        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Trim to exact date range requested
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end + pd.Timedelta("1D"))]

    logger.success(
        f"Solar data ready: {len(df):,} hourly rows | "
        f"{df['timestamp'].min()} → {df['timestamp'].max()}"
    )
    return df


def load_cached(location_name: str, cache_dir: str = "data/solar") -> pd.DataFrame:
    """
    Load all cached RAW WEATHER parquet files for a location and combine them.

    ⚠ Pattern is deliberately strict: matches ONLY '{location_name}_YYYY.parquet'
    (a 4-digit year), NOT other files that happen to share the location_name
    prefix — e.g. 'Mathura_solar_15min.parquet' (the PROCESSED pvlib output
    saved by scripts/02_solar_pipeline.py lives in the same data/solar/
    directory and would otherwise match a permissive glob, silently mixing
    raw hourly weather with already-simulated 15-min PV output — different
    resolutions, overlapping date ranges, duplicate timestamps, and a hard
    crash downstream in pv_model's resample step. This bit us once — the
    strict pattern is the actual fix.
    """
    import glob
    import re

    all_matches = glob.glob(os.path.join(cache_dir, f"{location_name}_*.parquet"))
    year_pattern = re.compile(rf"^{re.escape(location_name)}_\d{{4}}\.parquet$")
    files = sorted(f for f in all_matches if year_pattern.match(os.path.basename(f)))

    if not files:
        raise FileNotFoundError(
            f"No cached raw-weather solar data for '{location_name}' in '{cache_dir}'. "
            "Run fetch_solar_data() first."
        )

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Defensive: even with the strict glob, de-duplicate any repeated
    # timestamps (e.g. from re-running the pipeline with overlapping
    # date ranges) rather than let a duplicate silently propagate into
    # a hard crash several layers downstream.
    before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    if len(df) < before:
        logger.warning(f"load_cached({location_name}): dropped {before - len(df)} duplicate timestamp rows")

    return df.sort_values("timestamp").reset_index(drop=True)