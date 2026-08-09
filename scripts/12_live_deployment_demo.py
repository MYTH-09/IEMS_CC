"""
IEMS – Script 12: Live Deployment Demo
==========================================
Demonstrates what MPC control looks like operating on a REAL, full
calendar year of solar generation and household load — "streamed" day
by day, as a stand-in for a genuinely live deployment where a real
weather-based forecast would arrive each morning.

This is DELIBERATELY SEPARATE from every other comparison already built
(scripts/09_compare_controllers.py, validate_candidates.py) — those use
MPCController's default PERSISTENCE forecasting specifically to keep the
DQN-vs-MPC comparison fair. This script instead gives MPC a genuinely
accurate forecast (real historical data, replayed chronologically) —
demonstrating MPC's *realistic best-case* behavior under a live
deployment's actual operating conditions, not a fairness-controlled
research comparison. Both numbers are legitimate; they answer different
questions. See historical_forecast_provider.py's docstring for the full
reasoning on why this isn't "cheating."

Requires (from the real project, not this demo's mock fallback):
    data/solar/{city}_solar_15min.parquet   — from scripts/02_solar_pipeline.py
    data/processed/meters_15min.parquet     — from scripts/01_data_pipeline.py
    models/load_profiles.parquet, profile_kmeans.pkl — from scripts/01_data_pipeline.py

Falls back to synthetic mock data if these aren't found, so the demo
still runs standalone for testing the streaming mechanics themselves.

Usage:
    python scripts/12_live_deployment_demo.py --city Mathura --bill 2400
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.controllers.historical_forecast_provider import HistoricalForecastProvider


_PROCESSED_DATA_PATH = "data/processed/meters_15min.parquet"


def derive_monthly_outage_multipliers(city: str) -> dict[int, float]:
    """Same logic validated earlier — real monsoon-vs-dry seasonal ratio
    from Module 1's measured data. Falls back to no adjustment (all 1.0)
    if the processed data isn't available."""
    if not os.path.exists(_PROCESSED_DATA_PATH):
        logger.warning(f"'{_PROCESSED_DATA_PATH}' not found — no seasonal outage adjustment applied")
        return {}
    df = pd.read_parquet(_PROCESSED_DATA_PATH, columns=["city", "timestamp", "is_outage"])
    city_df = df[df["city"] == city]
    if len(city_df) == 0:
        return {}
    city_df = city_df.copy()
    city_df["month"] = pd.to_datetime(city_df["timestamp"]).dt.month
    overall_rate = city_df["is_outage"].mean()
    if overall_rate <= 0:
        return {}
    monthly_rate = city_df.groupby("month")["is_outage"].mean()
    return {int(k): round(float(v / overall_rate), 3) for k, v in monthly_rate.items()}


def load_real_or_mock_data(city: str, monthly_bill_inr: float):
    """Load the real cached solar + real load profile if available;
    otherwise build a synthetic stand-in so this script's mechanics can
    still be exercised standalone."""
    solar_path = f"data/solar/{city}_solar_15min.parquet"

    if os.path.exists(solar_path):
        logger.info(f"Using REAL cached solar data: {solar_path}")
        solar_df = pd.read_parquet(solar_path)
        timestamps = pd.DatetimeIndex(solar_df["timestamp"])
        solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=np.float32)

        try:
            from src.data.profile_generator import load_model, generate_profile
            profiles, kmeans = load_model("models")
            daily_kwh = monthly_bill_inr / 5.0 / 30.0   # rough ₹/kWh -> daily kWh
            n_days = len(timestamps) // 96
            raw = generate_profile(profiles, kmeans, target_daily_kwh=daily_kwh,
                                    season="summer", is_weekend=False, n_days=n_days)
            reps = int(np.ceil(len(timestamps) / len(raw)))
            load_kw = (np.tile(raw, reps)[: len(timestamps)] * 4.0).astype(np.float32)
            logger.info("Using REAL profile-generator load shape (Module 1)")
        except FileNotFoundError:
            logger.warning("No trained profile model found — using a synthetic load shape instead")
            hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
            shape = 0.3 + 0.4*np.exp(-((hours-8)**2)/4) + 0.6*np.exp(-((hours-19)**2)/4)
            load_kw = shape.astype(np.float32)
        return solar_kw, load_kw, timestamps, True

    logger.warning(f"'{solar_path}' not found — using MOCK data for demo mechanics only")
    n_days = 365
    timestamps = pd.date_range("2024-01-01", periods=n_days * 96, freq="15min")
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)
    seasonal = 1.0 + 0.2*np.sin(2*np.pi*(day_of_year-172)/365)
    solar_kw = np.clip(3.0*np.exp(-((hours-12)**2)/8)*seasonal, 0, None).astype(np.float32)
    solar_kw = np.where((hours < 6) | (hours > 18), 0.0, solar_kw).astype(np.float32)
    load_kw = (0.3 + 0.4*np.exp(-((hours-8)**2)/4) + 0.6*np.exp(-((hours-19)**2)/4)).astype(np.float32)
    return solar_kw, load_kw, timestamps, False


def run_streaming_year(load_kw, solar_kw, timestamps, grid, controller, panel_kw=4.0, battery_kwh=10.0, seed=42):
    """
    Streams a full year day-by-day. At each 96-step (24h) boundary the
    controller effectively receives a "new day's forecast" — for
    HistoricalForecastProvider this is genuinely accurate (real data);
    for rule-based it's irrelevant (no forecast used at all).
    """
    # The cached solar_kw is simulated output for a fixed 3.0 kWp reference
    # system (pv_model.py's DEFAULT_SYSTEM) — convert to per-kWp, then
    # scale to the requested panel_kw, same convention used throughout
    # the sizing modules (grid_search.py, nsga2_sizing.py).
    REFERENCE_SYSTEM_KWP = 3.0
    solar_kw_scaled = (solar_kw / REFERENCE_SYSTEM_KWP) * panel_kw

    battery_config = dict(
        capacity_kwh=battery_kwh, max_charge_rate_kw=min(3.0, battery_kwh) if battery_kwh > 0 else 0.01,
        max_discharge_rate_kw=min(3.0, battery_kwh) if battery_kwh > 0 else 0.01,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )
    n_days = min(365, len(load_kw) // 96 - 2)
    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw_scaled,
        timestamps=timestamps, battery_config=battery_config, grid=grid,
        episode_days=n_days, seed=seed,
    )

    obs, info = env.reset(seed=seed)
    controller.reset()

    daily_costs = []
    day_reward_accum = 0.0
    terminated = truncated = False
    step_in_day = 0
    day_count = 0

    while not (terminated or truncated):
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        day_reward_accum += reward
        step_in_day += 1

        if step_in_day >= 96:
            daily_costs.append(-day_reward_accum)
            day_count += 1
            if day_count % 60 == 0:
                logger.info(f"  ... day {day_count}/{n_days} streamed, running cost so far: ₹{sum(daily_costs):,.0f}")
            day_reward_accum = 0.0
            step_in_day = 0

    return {
        "total_cost_inr": info["episode_cost_inr"],
        "grid_import_kwh": info["episode_grid_import_kwh"],
        "grid_export_kwh": info["episode_grid_export_kwh"],
        "unmet_load_kwh": info["episode_unmet_load_kwh"],
        "n_days": day_count,
        "daily_costs": daily_costs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Mathura", choices=["Mathura", "Bareilly"])
    parser.add_argument("--bill", type=float, default=2400.0)
    parser.add_argument("--panel-kw", type=float, default=4.0)
    parser.add_argument("--battery-kwh", type=float, default=10.0)
    args = parser.parse_args()

    logger.info("━" * 60)
    logger.info("  IEMS Live Deployment Demo — Streaming a Real Year")
    logger.info("━" * 60)

    solar_kw, load_kw, timestamps, is_real = load_real_or_mock_data(args.city, args.bill)
    logger.info(f"Data source: {'REAL' if is_real else 'MOCK (demo mechanics only)'}, "
                f"{len(timestamps)} steps ({len(timestamps)/96:.0f} days)")

    monthly_mult = derive_monthly_outage_multipliers(args.city)
    if monthly_mult:
        logger.info(f"Seasonal outage multipliers applied: {monthly_mult}")
    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22],
                      monthly_outage_multiplier=monthly_mult)

    logger.info("\n── Running RULE-BASED (no forecast) across the full streamed year ──")
    rb_result = run_streaming_year(load_kw, solar_kw, timestamps, grid, RuleBasedController(),
                                     panel_kw=args.panel_kw, battery_kwh=args.battery_kwh)

    logger.info("\n── Running MPC WITH REAL STREAMED FORECAST across the full year ──")
    logger.info("  (each simulated day, MPC receives the ACTUAL real weather for the next 24h —")
    logger.info("   standing in for what a live deployment's morning forecast would provide)")
    mpc_streaming = MPCController(horizon_hours=24.0, forecast_provider=HistoricalForecastProvider())
    mpc_streamed_result = run_streaming_year(load_kw, solar_kw, timestamps, grid, mpc_streaming,
                                      panel_kw=args.panel_kw, battery_kwh=args.battery_kwh)

    logger.info("\n── Running MPC WITH DEFAULT PERSISTENCE FORECAST (no forecast_provider) ──")
    logger.info("  (identical MPC algorithm, identical data — only the forecast SOURCE differs:")
    logger.info("   'tomorrow = 24h ago', the same assumption used in every other validated")
    logger.info("   comparison in this project, e.g. scripts/09_compare_controllers.py)")
    mpc_persistence = MPCController(horizon_hours=24.0)   # forecast_provider=None -> default persistence
    mpc_persistence_result = run_streaming_year(load_kw, solar_kw, timestamps, grid, mpc_persistence,
                                      panel_kw=args.panel_kw, battery_kwh=args.battery_kwh)

    logger.info("\n" + "━" * 60)
    logger.info("  FULL-YEAR STREAMING RESULTS")
    logger.info("━" * 60)
    logger.info(f"  Rule-based                  : cost=₹{rb_result['total_cost_inr']:,.0f}  "
                f"import={rb_result['grid_import_kwh']:,.0f}kWh  "
                f"unmet={rb_result['unmet_load_kwh']:,.0f}kWh")
    logger.info(f"  MPC (persistence forecast)  : cost=₹{mpc_persistence_result['total_cost_inr']:,.0f}  "
                f"import={mpc_persistence_result['grid_import_kwh']:,.0f}kWh  "
                f"unmet={mpc_persistence_result['unmet_load_kwh']:,.0f}kWh")
    logger.info(f"  MPC (real streamed forecast): cost=₹{mpc_streamed_result['total_cost_inr']:,.0f}  "
                f"import={mpc_streamed_result['grid_import_kwh']:,.0f}kWh  "
                f"unmet={mpc_streamed_result['unmet_load_kwh']:,.0f}kWh")

    savings_vs_rulebased = rb_result["total_cost_inr"] - mpc_streamed_result["total_cost_inr"]
    forecast_only_gain = mpc_persistence_result["total_cost_inr"] - mpc_streamed_result["total_cost_inr"]
    algorithm_only_gain = rb_result["total_cost_inr"] - mpc_persistence_result["total_cost_inr"]

    logger.info("\n" + "─" * 60)
    logger.info("  ATTRIBUTION — how much comes from WHAT")
    logger.info("─" * 60)
    logger.success(f"  MPC's algorithm alone (vs rule-based, same forecast quality)  : ₹{algorithm_only_gain:,.0f}")
    logger.success(f"  Real forecast alone (vs persistence, same MPC algorithm)      : ₹{forecast_only_gain:,.0f}")
    logger.success(f"  Combined total (real-forecast MPC vs rule-based)              : ₹{savings_vs_rulebased:,.0f}")
    logger.success("━" * 60)


if __name__ == "__main__":
    main()