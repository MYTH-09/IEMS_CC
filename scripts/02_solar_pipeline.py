"""
IEMS – Script 02: Solar Pipeline
=================================
Fetches NASA POWER irradiance data for Mathura and Bareilly, simulates
PV output with pvlib, and saves 15-min solar generation parquet files.

What it does
------------
1. Fetch hourly GHI / DNI / DHI / temp / wind from NASA POWER API
   (cached per city per year → no repeated API calls on re-runs)
2. Run pvlib PVWatts simulation for a default 3 kWp south-facing system
3. Save 15-min solar generation → data/solar/{city}_solar_15min.parquet
4. Print a diagnostic summary (monthly yield, capacity factor)

Usage (from project root with venv active):
    python scripts/02_solar_pipeline.py

Expected runtime:
    First run  : ~2–3 min  (6 NASA POWER API calls + pvlib simulation)
    Subsequent : ~30 sec   (cached API responses, pvlib only)

Outputs
-------
    data/solar/Mathura_2019.parquet      ← cached hourly NASA data
    data/solar/Mathura_2020.parquet
    data/solar/Mathura_2021.parquet
    data/solar/Bareilly_2020.parquet
    data/solar/Bareilly_2021.parquet
    data/solar/Mathura_solar_15min.parquet    ← 15-min AC power output
    data/solar/Bareilly_solar_15min.parquet
"""

import os
import sys

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solar.nasa_power import fetch_solar_data
from src.solar.pv_model   import (
    simulate_pv, save_solar_data,
    annual_yield_kwh, capacity_factor,
    DEFAULT_SYSTEM,
)


# ── Site definitions ──────────────────────────────────────────────────────────
# Coordinates and dataset date ranges per city
SITES = {
    "Mathura": {
        "lat":        27.4924,
        "lon":        77.6737,
        "altitude_m": 182.0,
        "start":      "2019-05-01",
        "end":        "2020-12-31",
    },
    "Bareilly": {
        "lat":        28.3670,
        "lon":        79.4304,
        "altitude_m": 172.0,
        "start":      "2020-01-01",
        "end":        "2021-10-31",
    },
}

SOLAR_DIR  = "data/solar"
SYSTEM_KWP = DEFAULT_SYSTEM["capacity_kw"]   # 3 kWp default (matches config)


# ─────────────────────────────────────────────────────────────────────────────

def run_site(name: str, cfg: dict) -> pd.DataFrame:
    """Fetch + simulate PV for one city. Returns the 15-min DataFrame."""
    logger.info(f"\n{'─'*50}")
    logger.info(f"  Processing city: {name}")
    logger.info(f"{'─'*50}")

    # 1. Fetch / load cached irradiance data
    weather = fetch_solar_data(
        lat           = cfg["lat"],
        lon           = cfg["lon"],
        start_date    = cfg["start"],
        end_date      = cfg["end"],
        location_name = name,
        cache_dir     = SOLAR_DIR,
    )

    # 2. Run pvlib simulation
    solar_15min = simulate_pv(
        weather_hourly = weather,
        lat            = cfg["lat"],
        lon            = cfg["lon"],
        altitude_m     = cfg["altitude_m"],
        # All other params use defaults from DEFAULT_SYSTEM (3 kWp, 27° tilt, etc.)
    )

    return solar_15min


def print_monthly_summary(name: str, solar_15min: pd.DataFrame) -> None:
    """Print a monthly energy yield table for a city."""
    df = solar_15min.copy()
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")

    monthly = (
        df.groupby("month")["solar_energy_kwh"]
        .sum()
        .rename("yield_kwh")
        .reset_index()
    )
    monthly["yield_kwh"] = monthly["yield_kwh"].round(1)

    logger.info(f"\n  Monthly PV yield ({name}, {SYSTEM_KWP} kWp system):")
    logger.info(f"  {'Month':<12}  {'Yield (kWh)':>12}")
    logger.info(f"  {'─'*26}")
    for _, row in monthly.iterrows():
        logger.info(f"  {str(row['month']):<12}  {row['yield_kwh']:>12.1f}")

    cf  = capacity_factor(solar_15min, SYSTEM_KWP)
    tot = annual_yield_kwh(solar_15min)
    logger.info(f"\n  Total yield          : {tot:,.0f} kWh")
    logger.info(f"  Capacity factor      : {cf:.3f}  ({cf*100:.1f}%)")
    logger.info(f"  Specific yield       : {tot/SYSTEM_KWP:,.0f} kWh/kWp")

    # Sanity check: Indian rooftop CF should be 0.16–0.22
    if 0.12 <= cf <= 0.28:
        logger.success(f"  ✓ Capacity factor {cf:.2f} is in expected range [0.16–0.22]")
    else:
        logger.warning(
            f"  ⚠ Capacity factor {cf:.2f} is outside expected range. "
            "Check irradiance data quality."
        )


def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS Solar Pipeline  —  START")
    logger.info("━" * 55)

    results = {}

    for city, cfg in SITES.items():
        solar_15min = run_site(city, cfg)

        # Save to parquet
        out_path = os.path.join(SOLAR_DIR, f"{city}_solar_15min.parquet")
        save_solar_data(solar_15min, out_path)

        # Print diagnostics
        print_monthly_summary(city, solar_15min)

        results[city] = solar_15min

    logger.success("\n" + "━" * 55)
    logger.success("  Solar Pipeline COMPLETE")
    logger.success("━" * 55)
    logger.success("\n  Outputs:")
    for city in SITES:
        logger.success(f"    data/solar/{city}_solar_15min.parquet")
    logger.success("\n  Next: run scripts/03_battery_grid_check.py")


if __name__ == "__main__":
    main()
