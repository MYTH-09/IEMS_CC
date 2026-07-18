"""
IEMS – Script 06: Sizing Grid Search Pipeline
=================================================
Runs the full panel/battery sizing sweep (§9.1) for a household and
prints top recommendations by three different priorities: fastest
payback, best 25-year NPV, best outage backup coverage.

⚠ Data length requirement: EnergyEnv samples a random start index within
the provided series for each simulated episode, so the input load/solar
series must be LONGER than one episode. For sizing, episode_days=365
(a full year), so the series must span MORE than 365 days — this script
uses 400 days of synthetic data to leave headroom. When wiring this to
real data (Module 1's profile generator), request n_days≥400 from
`generate_profile()` for the same reason.

Usage (from project root with venv active):
    python scripts/06_sizing_pipeline.py

For a real run against actual project data, replace `build_test_series()`
with calls to the profile_generator (Module 1) and pv_model (Module 2a)
outputs — this script uses synthetic data so it runs standalone without
requiring the full pipeline to have been run first.
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid.grid_model import GridModel
from src.sizing.grid_search import run_grid_search, top_recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test data (standalone — swap for real pipeline output later)
# ─────────────────────────────────────────────────────────────────────────────

def build_test_series(n_days: int = 400, seed: int = 42):
    """
    Build synthetic load + per-kWp solar series long enough for a full
    365-day sizing episode plus random-start headroom.
    """
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="15min")
    # DatetimeIndex.hour/.minute/.dayofyear return pandas Index objects, not
    # plain arrays — casting to numpy explicitly here prevents Index from
    # silently propagating through the downstream arithmetic below.
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)

    # Household load: morning + evening peaks, with mild seasonal variation
    seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # peak in summer
    load = (
        0.3
        + 0.4 * np.exp(-((hours - 8) ** 2) / 4)
        + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
    ) * seasonal_factor + rng.normal(0, 0.05, n_steps)
    load = np.clip(load, 0.05, None).astype(np.float32)

    # Solar output PER INSTALLED kWp — i.e. what a 1 kWp system would produce.
    # Peak ~0.8 kW/kWp at solar noon on a clear day (realistic derate from 1.0
    # nameplate due to temperature/system losses), with cloudy-day variation.
    clear_sky = 0.8 * np.exp(-((hours - 12) ** 2) / 8)
    clear_sky = np.where((hours < 6) | (hours > 18), 0.0, clear_sky)
    cloud_factor = np.clip(rng.normal(0.9, 0.15, n_steps), 0.3, 1.0)
    solar_per_kwp = np.clip(clear_sky * cloud_factor, 0, None).astype(np.float32)

    return load, solar_per_kwp, timestamps


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS Sizing Grid Search Pipeline  —  START")
    logger.info("━" * 55)

    load_kw, solar_per_kwp, timestamps = build_test_series(n_days=400)
    logger.info(f"Test series: {len(load_kw)} steps ({len(load_kw)/96:.0f} days)")
    logger.info(f"  Load  — mean: {load_kw.mean():.2f} kW, peak: {load_kw.max():.2f} kW")
    logger.info(f"  Solar — mean: {solar_per_kwp.mean():.3f} kW/kWp, peak: {solar_per_kwp.max():.3f} kW/kWp")

    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])

    # Search grid — using a reduced set here for a fast demonstration run;
    # the full config default is panel_sizes_kw=[1,2,3,4,5,6,8,10],
    # battery_sizes_kwh=[0,5,7.5,10,15,20] (48 combinations, ~2-3 min runtime).
    panel_sizes_kw    = [2.0, 3.0, 4.0, 5.0]
    battery_sizes_kwh = [0.0, 5.0, 10.0]

    financial_config = dict(
        panel_cost_per_kw=45000.0,
        battery_cost_per_kwh=15000.0,
        balance_of_system_cost=10000.0,
        subsidy_rate_up_to_2kw=0.40,
        subsidy_rate_2_to_3kw=0.20,
        subsidy_max_kw=3.0,
        analysis_period_years=25,
        discount_rate=0.08,
        electricity_inflation_rate=0.05,
        panel_degradation_rate=0.005,
        maintenance_cost_annual_inr=2000.0,
    )

    results_df = run_grid_search(
        load_kw=load_kw, solar_kw_per_kwp=solar_per_kwp, timestamps=timestamps,
        grid=grid, panel_sizes_kw=panel_sizes_kw, battery_sizes_kwh=battery_sizes_kwh,
        financial_config=financial_config, seed=42,
    )

    logger.info("\n" + "─" * 90)
    logger.info("  FULL RESULTS TABLE (sorted by 25yr NPV)")
    logger.info("─" * 90)
    display_cols = [
        "panel_kw", "battery_kwh", "annual_savings_inr", "total_capex_inr",
        "payback_years", "npv_25yr_inr", "irr_pct", "annual_unmet_load_kwh",
    ]
    logger.info(f"\n{results_df[display_cols].to_string(index=False)}")

    recs = top_recommendations(results_df, n=3)

    logger.info("\n" + "━" * 55)
    logger.info("  TOP RECOMMENDATIONS")
    logger.info("━" * 55)

    logger.info("\n  ⚡ Fastest Payback:")
    logger.info(f"\n{recs['fastest_payback'][display_cols].to_string(index=False)}")

    logger.info("\n  💰 Best 25-Year NPV:")
    logger.info(f"\n{recs['best_npv'][display_cols].to_string(index=False)}")

    logger.info("\n  🔋 Best Outage Backup Coverage:")
    logger.info(f"\n{recs['best_backup'][display_cols].to_string(index=False)}")

    # Sanity checks
    assert len(results_df) == len(panel_sizes_kw) * len(battery_sizes_kwh), \
        "Result count doesn't match search grid size"
    assert (results_df["total_capex_inr"] > 0).all(), "All non-zero systems should have positive CAPEX"

    zero_system = results_df[(results_df["panel_kw"] == panel_sizes_kw[0]) & (results_df["battery_kwh"] == 0)]
    largest_system = results_df[
        (results_df["panel_kw"] == max(panel_sizes_kw)) & (results_df["battery_kwh"] == max(battery_sizes_kwh))
    ]
    logger.info(
        f"\n  Sanity: smallest system CAPEX=₹{zero_system['total_capex_inr'].iloc[0]:,.0f}, "
        f"largest system CAPEX=₹{largest_system['total_capex_inr'].iloc[0]:,.0f}"
    )
    assert zero_system["total_capex_inr"].iloc[0] < largest_system["total_capex_inr"].iloc[0], \
        "Larger systems should cost more than smaller ones"
    logger.success("  ✓ CAPEX scales sensibly with system size")

    logger.success("\n" + "━" * 55)
    logger.success("  SIZING PIPELINE COMPLETE ✓")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()