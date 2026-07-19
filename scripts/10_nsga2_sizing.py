"""
IEMS – Script 10: NSGA-II Sizing (§9.2, Advanced/Stretch Mode)
====================================================================
Runs the multi-objective GA sizing search and prints the resulting
Pareto front — a set of non-dominated system configurations, each the
best possible trade-off for its position on the CAPEX/NPV/payback/backup
frontier, rather than one single "optimal" answer.

TWO solar-curve modes, controlled by `--real-solar`:

  Default (synthetic): uses a lightweight analytical stand-in for how
  tilt/azimuth affect solar yield, so this script runs standalone without
  needing cached NASA weather data. Good for testing the NSGA-II wiring
  itself, NOT for real sizing decisions.

  --real-solar: wraps `src.solar.pv_model.simulate_pv()` against ALREADY
  CACHED weather data (from `scripts/02_solar_pipeline.py` — no network
  calls needed if that has been run at least once for this location).
  This is what a real sizing decision should use.

Usage:
    python scripts/10_nsga2_sizing.py                    # quick synthetic smoke test
    python scripts/10_nsga2_sizing.py --real-solar --city Mathura --full
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid.grid_model import GridModel
from src.sizing.nsga2_sizing import run_nsga2
from src.sizing.grid_search import simulate_system_size


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic solar-curve stand-in (testing only — NOT physically accurate)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticSolarCurve:
    """
    Picklable callable approximating how yield degrades as tilt deviates
    from latitude and azimuth deviates from true south (180°) —
    directionally correct, NOT a substitute for pvlib. Used only so this
    script can be exercised without cached NASA weather data.

    ⚠ Implemented as a class (not a closure/nested function) specifically
    so it survives being pickled and sent to worker processes when
    n_workers > 1 — Python's standard `pickle` module cannot serialize
    closures, which would otherwise crash every parallel evaluation with
    "Can't pickle local object ...".
    """

    def __init__(self, timestamps: pd.DatetimeIndex, latitude_deg: float = 27.5):
        hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
        base_curve = 0.8 * np.exp(-((hours - 12) ** 2) / 8)
        self.base_curve = np.where((hours < 6) | (hours > 18), 0.0, base_curve).astype(np.float32)
        self.latitude_deg = latitude_deg

    def __call__(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        tilt_penalty = 1.0 - 0.003 * abs(tilt_deg - self.latitude_deg)
        azimuth_penalty = 1.0 - 0.0015 * abs(azimuth_deg - 180.0)
        derate = max(0.5, tilt_penalty * azimuth_penalty)
        return (self.base_curve * derate).astype(np.float32)


def make_synthetic_solar_curve_fn(timestamps: pd.DatetimeIndex, latitude_deg: float = 27.5) -> "SyntheticSolarCurve":
    """Thin wrapper kept for call-site readability — returns a picklable SyntheticSolarCurve instance."""
    return SyntheticSolarCurve(timestamps, latitude_deg)


class RealSolarCurve:
    """
    Picklable callable wrapping the real pvlib simulation (§4.2) against
    cached NASA weather data for `city`. Requires
    `scripts/02_solar_pipeline.py` to have been run at least once for
    this city — no network calls happen here.

    ⚠ Implemented as a class for the same reason as SyntheticSolarCurve
    above: closures cannot be pickled for dispatch to worker processes
    when n_workers > 1. The cached weather DataFrame is loaded ONCE here
    (in __init__) and stored as a plain attribute, which pickles fine —
    it's just data, unlike a closure's captured-variable machinery.
    """

    def __init__(self, city: str, lat: float, lon: float, altitude_m: float, timestamps_target: pd.DatetimeIndex):
        from src.solar.nasa_power import load_cached

        self.lat = lat
        self.lon = lon
        self.altitude_m = altitude_m
        self.timestamps_target = timestamps_target
        self.weather_hourly = load_cached(city, cache_dir="data/solar")
        logger.info(f"Loaded cached weather for {city}: {len(self.weather_hourly)} hourly rows")

        # Defensive check: this class previously had a real bug where target
        # timestamps from an unrelated calendar range (e.g. 2025) got
        # nearest-neighbor-reindexed against weather data from a completely
        # different era (e.g. 2019-2020), collapsing the solar curve to one
        # flat repeated value with no day/night or seasonal cycle. Fail
        # loudly here instead of silently repeating that failure mode.
        weather_start = self.weather_hourly["timestamp"].min()
        weather_end = self.weather_hourly["timestamp"].max()
        target_start = timestamps_target.min()
        target_end = timestamps_target.max()
        overlap = not (target_end < weather_start or target_start > weather_end)
        if not overlap:
            raise ValueError(
                f"RealSolarCurve: timestamps_target range ({target_start} → {target_end}) "
                f"does not overlap weather data range ({weather_start} → {weather_end}). "
                "Nearest-neighbor alignment would silently collapse to one flat repeated "
                "value across the whole target range. Build the household on the SAME "
                "calendar dates as the weather cache — see build_test_household(start_date=...)."
            )

    def __call__(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        from src.solar.pv_model import simulate_pv

        result = simulate_pv(
            weather_hourly=self.weather_hourly, lat=self.lat, lon=self.lon, altitude_m=self.altitude_m,
            capacity_kw=1.0,   # per-kWp reference curve — scaled by panel_kw downstream
            tilt_deg=tilt_deg, azimuth_deg=azimuth_deg,
        )
        curve = result.set_index("timestamp")["solar_power_kw"]
        aligned = curve.reindex(self.timestamps_target, method="nearest", fill_value=0.0)
        return aligned.to_numpy(dtype=np.float32)


def make_real_solar_curve_fn(city: str, lat: float, lon: float, altitude_m: float, timestamps_target: pd.DatetimeIndex) -> "RealSolarCurve":
    """Thin wrapper kept for call-site readability — returns a picklable RealSolarCurve instance."""
    return RealSolarCurve(city, lat, lon, altitude_m, timestamps_target)


# ─────────────────────────────────────────────────────────────────────────────

def build_test_household(n_days: int = 400, seed: int = 42, start_date: str = "2025-01-01"):
    """
    ⚠ `start_date` MUST match the calendar range of whatever solar data
    this household will be paired with. Using an arbitrary disconnected
    date range (e.g. a 2025 household against 2019-2020 cached weather)
    silently breaks the real-solar path: RealSolarCurve's nearest-neighbor
    reindex would then map every single household timestamp to the SAME
    boundary weather record (since all 2025 dates are chronologically
    after all 2019-2020 data), collapsing solar generation to one flat
    repeated value with no day/night or seasonal cycle at all. This was a
    real, confirmed bug — not a hypothetical — caught via a full
    --real-solar --full run.
    """
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    timestamps = pd.date_range(start_date, periods=n_steps, freq="15min")
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)

    seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
    load = (
        0.3 + 0.4 * np.exp(-((hours - 8) ** 2) / 4) + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
    ) * seasonal_factor + rng.normal(0, 0.05, n_steps)
    load = np.clip(load, 0.05, None).astype(np.float32)

    return load, timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-solar", action="store_true", help="Use real cached pvlib/NASA data instead of the synthetic stand-in")
    parser.add_argument("--city", default="Mathura", help="City name matching cached weather files (only used with --real-solar)")
    parser.add_argument("--full", action="store_true", help="Run a full-size search (pop=40, gen=50) instead of the quick smoke test (pop=8, gen=3)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: auto-detect CPU count, capped at 8). Use --workers 1 to disable parallelization.")
    args = parser.parse_args()

    logger.info("━" * 60)
    logger.info("  IEMS NSGA-II Sizing  —  START")
    logger.info("━" * 60)

    if args.real_solar:
        city_coords = {
            "Mathura":  dict(lat=27.4924, lon=77.6737, altitude_m=182.0),
            "Bareilly": dict(lat=28.3670, lon=79.4304, altitude_m=172.0),
        }
        if args.city not in city_coords:
            logger.error(f"Unknown city '{args.city}'. Choose from: {list(city_coords.keys())}")
            sys.exit(1)
        coords = city_coords[args.city]

        # Load the weather cache FIRST so the household's timestamps can be
        # built on the SAME calendar range. Building them independently
        # (e.g. an arbitrary "2025-01-01" start) and force-aligning later
        # via nearest-neighbor reindex is what caused the real bug fixed
        # here — see build_test_household()'s docstring for the full story.
        from src.solar.nasa_power import load_cached
        weather_hourly = load_cached(args.city, cache_dir="data/solar")
        weather_start = weather_hourly["timestamp"].min()
        weather_end = weather_hourly["timestamp"].max()
        weather_span_days = (weather_end - weather_start).days
        logger.info(
            f"Cached weather for {args.city} spans {weather_start.date()} → "
            f"{weather_end.date()} ({weather_span_days} days)"
        )

        n_days = min(400, weather_span_days - 5)   # small safety margin below the weather span
        load_kw, timestamps = build_test_household(n_days=n_days, start_date=str(weather_start.date()))
        logger.info(f"Household timestamps aligned to weather range: {timestamps.min()} → {timestamps.max()}")

        solar_curve_fn = RealSolarCurve(args.city, timestamps_target=timestamps, **coords)
        # RealSolarCurve already loaded weather once internally too (cheap —
        # it's a cached parquet read, not a network call); reusing the
        # `weather_hourly` loaded above only for date-range inspection here
        # is fine and keeps RealSolarCurve fully self-contained/picklable.
        logger.info(f"Using REAL pvlib solar curves for {args.city} (cached weather, no network calls)")
    else:
        load_kw, timestamps = build_test_household(n_days=400)
        solar_curve_fn = make_synthetic_solar_curve_fn(timestamps)
        logger.warning(
            "Using SYNTHETIC solar stand-in (for wiring validation only). "
            "Run with --real-solar for an actual sizing decision."
        )

    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])

    financial_config = dict(
        panel_cost_per_kw=45000.0, battery_cost_per_kwh=15000.0, balance_of_system_cost=10000.0,
        subsidy_rate_up_to_2kw=0.40, subsidy_rate_2_to_3kw=0.20, subsidy_max_kw=3.0,
        analysis_period_years=25, discount_rate=0.08,
        electricity_inflation_rate=0.05, panel_degradation_rate=0.005,
        maintenance_cost_annual_inr=2000.0,
    )

    # Baseline: zero solar/battery cost, for computing savings
    logger.info("Computing no-solar baseline...")
    zero_solar = np.zeros_like(load_kw)
    baseline = simulate_system_size(
        load_kw=load_kw, solar_kw_per_kwp=zero_solar, timestamps=timestamps,
        panel_kw=0.0, battery_kwh=0.0, grid=grid, episode_days=365, seed=42,
    )
    baseline_annual_cost_inr = baseline["annual_cost_inr"]
    logger.info(f"Baseline annual cost: ₹{baseline_annual_cost_inr:,.0f}")

    pop_size, n_generations = (40, 50) if args.full else (8, 3)
    if not args.full:
        logger.warning(
            f"Quick mode: pop_size={pop_size}, generations={n_generations} "
            f"({pop_size*n_generations} evaluations) — for wiring validation "
            "only, not a converged Pareto front. Use --full for a real run."
        )

    pareto_df = run_nsga2(
        load_kw=load_kw, timestamps=timestamps, grid=grid,
        solar_curve_fn=solar_curve_fn, financial_config=financial_config,
        baseline_annual_cost_inr=baseline_annual_cost_inr,
        pop_size=pop_size, n_generations=n_generations,
        episode_days=365, seed=42, n_workers=args.workers,
    )

    logger.info("\n" + "─" * 100)
    logger.info("  PARETO FRONT (non-dominated solutions, sorted by NPV)")
    logger.info("─" * 100)
    logger.info(f"\n{pareto_df.to_string(index=False)}")

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/nsga2_pareto_front.csv"
    pareto_df.to_csv(out_path, index=False)
    logger.success(f"\nPareto front saved → {out_path}")

    logger.success("\n" + "━" * 60)
    logger.success("  NSGA-II SIZING COMPLETE")
    logger.success("━" * 60)


if __name__ == "__main__":
    main()