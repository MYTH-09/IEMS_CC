"""
IEMS – Data Loader
==================
Loads the 4 raw CEEW smart-meter CSV files and combines them into a
single DataFrame with clean column names.

Memory strategy
---------------
* float32 for all numeric columns  → ~50 % RAM saving vs float64
* category dtype for meter_id/city → ~90 % RAM saving for those columns
* Files are loaded individually, then concatenated once (avoids peak RAM spike)

Typical combined size in RAM after loading: ~600 MB (was ~2 GB with defaults)
"""

import os
import re
import glob
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger


# ── Column rename map (raw name → clean name) ─────────────────────────────────
COL_RENAME: dict[str, str] = {
    "x_Timestamp":           "timestamp",
    "t_kWh":                 "energy_kwh",
    "z_Avg Voltage (Volt)":  "voltage_v",
    "z_Avg Current (Amp)":   "current_a",
    "y_Freq (Hz)":           "frequency_hz",
    "meter":                 "meter_id",
}

# ── dtype overrides applied at read time ──────────────────────────────────────
_DTYPE_MAP: dict[str, str] = {
    "t_kWh":                 "float32",
    "z_Avg Voltage (Volt)":  "float32",
    "z_Avg Current (Amp)":   "float32",
    "y_Freq (Hz)":           "float32",
    "meter":                 "category",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _city_from_filename(filepath: str) -> str:
    """Derive city name from the CSV filename."""
    name = os.path.basename(filepath).lower()
    if "bareilly" in name:
        return "Bareilly"
    if "mathura" in name:
        return "Mathura"
    return "Unknown"


def _year_from_filename(filepath: str) -> Optional[int]:
    """Derive data year from the CSV filename (e.g. '2020')."""
    match = re.search(r"(\d{4})", os.path.basename(filepath))
    return int(match.group(1)) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_single_file(filepath: str) -> pd.DataFrame:
    """
    Load one CEEW CSV and return a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to a raw CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, energy_kwh, voltage_v, current_a,
                 frequency_hz, meter_id, city, data_year
    """
    city      = _city_from_filename(filepath)
    data_year = _year_from_filename(filepath)
    fname     = os.path.basename(filepath)
    size_mb   = os.path.getsize(filepath) / 1024 / 1024

    logger.info(f"Loading: {fname}  ({size_mb:.0f} MB, {city}, {data_year})")

    df = pd.read_csv(
        filepath,
        dtype=_DTYPE_MAP,
        parse_dates=["x_Timestamp"],
    )

    df = df.rename(columns=COL_RENAME)

    # Add metadata columns
    df["city"]      = pd.Categorical([city]      * len(df))
    df["data_year"] = np.int16(data_year) if data_year else np.int16(0)

    # Sort within file for safety (should already be sorted, but be explicit)
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    n_meters  = df["meter_id"].nunique()
    date_min  = df["timestamp"].min().date()
    date_max  = df["timestamp"].max().date()
    logger.info(f"  → {len(df):>10,} rows | {n_meters:>3} meters | {date_min} → {date_max}")

    return df


def load_all_files(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and concatenate all CEEW CSV files found in `data_dir`.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw CSVs (default: 'data/raw').

    Returns
    -------
    pd.DataFrame
        Combined dataset, sorted by meter_id then timestamp.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in `data_dir`.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}/'. "
            "Place the Kaggle CSVs there and retry."
        )

    logger.info(f"Found {len(csv_files)} CSV file(s) in '{data_dir}/'")

    frames = [load_single_file(f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    logger.success(
        f"Combined dataset ready: {len(combined):,} rows | "
        f"{combined['meter_id'].nunique()} meters | "
        f"{combined['city'].unique().tolist()}"
    )
    return combined


# ── Parquet helpers (fast I/O for downstream modules) ─────────────────────────

def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a DataFrame to Parquet (compressed, column-oriented)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"Saved → {path}  ({size_mb:.1f} MB)")


def load_parquet(path: str) -> pd.DataFrame:
    """Load a previously saved Parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parquet file not found: '{path}'. "
            "Run scripts/01_data_pipeline.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded ← {path}  ({len(df):,} rows, {df['meter_id'].nunique()} meters)")
    return df
