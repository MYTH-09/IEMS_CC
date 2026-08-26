"""
IEMS – Script 02: Solar Pipeline
=================================
Fetches NASA POWER irradiance data for major Indian cities (Bengaluru, Chennai,
Hyderabad, Mysuru, Mumbai, Delhi, Kolkata, Mathura, Bareilly, etc.), simulates
PV output with pvlib, and saves 15-min solar generation parquet files.

What it does
------------
1. Fetch hourly GHI / DNI / DHI / temp / wind from NASA POWER API
   (cached per city per year → no repeated API calls on re-runs)
2. Run pvlib PVWatts simulation for a default 3 kWp south-facing system with optimal tilt
3. Save 15-min solar generation → data/solar/{city}_solar_15min.parquet
4. Print a diagnostic summary (monthly yield, capacity factor)

Usage:
    # Process default showcase cities:
    python scripts/02_solar_pipeline.py

    # Process specific city (e.g. Bengaluru, Chennai, Hyderabad, Mysuru):
    python scripts/02_solar_pipeline.py --city Bengaluru
    python scripts/02_solar_pipeline.py --city Mysuru

    # Process multiple cities:
    python scripts/02_solar_pipeline.py --cities Bengaluru Chennai Hyderabad Mysuru Mumbai Delhi

    # List all supported Indian cities:
    python scripts/02_solar_pipeline.py --list

    # Process all major Indian cities:
    python scripts/02_solar_pipeline.py --all
"""

import os
import sys
import argparse
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solar.locations import get_city, list_cities, get_city_names, CityInfo
from src.solar.nasa_power import fetch_solar_data
from src.solar.pv_model   import (
    simulate_pv, save_solar_data,
    annual_yield_kwh, capacity_factor,
    get_or_generate_solar_15min,
    DEFAULT_SYSTEM,
)


SOLAR_DIR  = "data/solar"
SYSTEM_KWP = DEFAULT_SYSTEM["capacity_kw"]   # 3 kWp default (matches config)

# Default showcase cities when no specific arguments are given
DEFAULT_SHOWCASE_CITIES = ["Mathura", "Bareilly", "Bengaluru", "Chennai", "Hyderabad", "Mysuru"]


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

    # Sanity check: Indian rooftop CF should be 0.14–0.26
    if 0.12 <= cf <= 0.28:
        logger.success(f"  ✓ Capacity factor {cf:.2f} is in expected range [0.14–0.26]")
    else:
        logger.warning(
            f"  ⚠ Capacity factor {cf:.2f} is outside expected range. "
            "Check irradiance data quality."
        )


def process_city(city_info: CityInfo, start_date: str, end_date: str, force: bool = False) -> pd.DataFrame:
    """Fetch/simulate solar data for a single city."""
    logger.info(f"\n{'─'*55}")
    logger.info(f"  Processing: {city_info.display_name} (lat={city_info.lat}, lon={city_info.lon}, alt={city_info.altitude_m}m, tilt={city_info.optimal_tilt}°)")
    logger.info(f"{'─'*55}")

    solar_15min = get_or_generate_solar_15min(
        city_name=city_info.name,
        start_date=start_date,
        end_date=end_date,
        cache_dir=SOLAR_DIR,
        capacity_kw=SYSTEM_KWP,
        force_refresh=force,
    )
    print_monthly_summary(city_info.name, solar_15min)
    return solar_15min


def main() -> None:
    parser = argparse.ArgumentParser(description="IEMS NASA POWER Solar Generation Pipeline for Indian Cities")
    parser.add_argument("--city", type=str, default=None, help="Process a single city (e.g. 'Bengaluru', 'Chennai', 'Hyderabad', 'Mysuru')")
    parser.add_argument("--cities", nargs="+", default=None, help="Process multiple specified cities")
    parser.add_argument("--all", action="store_true", help="Process all 50+ registered Indian cities")
    parser.add_argument("--list", action="store_true", help="List all supported Indian cities and exit")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD, default: 2020-01-01)")
    parser.add_argument("--end", type=str, default="2021-12-31", help="End date (YYYY-MM-DD, default: 2021-12-31)")
    parser.add_argument("--force", action="store_true", help="Force re-fetch and re-simulate even if cached")

    args = parser.parse_args()

    all_registered = list_cities()

    if args.list:
        print("\nSupported Indian Cities in IEMS Location Registry:")
        print("=" * 70)
        print(f"{'City Name':<25} {'State':<25} {'Lat / Lon':<18}")
        print("-" * 70)
        for c in all_registered:
            print(f"{c.name:<25} {c.state:<25} {c.lat:>5.2f}N, {c.lon:>5.2f}E")
        print("=" * 70)
        print(f"Total: {len(all_registered)} cities\n")
        return

    # Determine which cities to process
    targets: list[CityInfo] = []
    if args.city:
        city_info = get_city(args.city)
        if not city_info:
            logger.error(f"City '{args.city}' not recognized. Use --list to see all supported cities.")
            sys.exit(1)
        targets = [city_info]
    elif args.cities:
        for c_str in args.cities:
            c_info = get_city(c_str)
            if not c_info:
                logger.warning(f"City '{c_str}' not recognized — skipping.")
            else:
                targets.append(c_info)
    elif args.all:
        targets = all_registered
    else:
        # Default run: run for Mathura, Bareilly and prominent metros
        for name in DEFAULT_SHOWCASE_CITIES:
            c_info = get_city(name)
            if c_info:
                targets.append(c_info)

    logger.info("━" * 60)
    logger.info(f"  IEMS Solar Pipeline  —  Processing {len(targets)} Cities")
    logger.info(f"  Date range: {args.start} → {args.end}")
    logger.info("━" * 60)

    for city_info in targets:
        try:
            process_city(city_info, start_date=args.start, end_date=args.end, force=args.force)
        except Exception as e:
            logger.error(f"Failed to process {city_info.name}: {e}")

    logger.success("\n" + "━" * 60)
    logger.success("  Solar Pipeline Run COMPLETE")
    logger.success("━" * 60)
    logger.info("Outputs generated in data/solar/:")
    for city_info in targets:
        parquet_file = os.path.join(SOLAR_DIR, f"{city_info.name}_solar_15min.parquet")
        if os.path.exists(parquet_file):
            logger.success(f"  ✓ {parquet_file}")


if __name__ == "__main__":
    main()
