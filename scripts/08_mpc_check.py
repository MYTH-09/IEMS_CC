"""
IEMS – Script 08: MPC Controller Check
=========================================
Validates the MPC controller by running it through a full episode and
comparing against the rule-based baseline on the same household.

Usage:
    python scripts/08_mpc_check.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController


def build_household_series(n_days: int = 60, seed: int = 999, load_scale: float = 1.2):
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="15min")
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)

    load = (
        0.3 + 0.4 * np.exp(-((hours - 8) ** 2) / 4) + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
    ) * load_scale + rng.normal(0, 0.05 * load_scale, n_steps)
    load = np.clip(load, 0.05, None).astype(np.float32)

    solar = 3.0 * np.exp(-((hours - 12) ** 2) / 8)
    solar = np.where((hours < 6) | (hours > 18), 0.0, solar)
    solar = np.clip(solar + rng.normal(0, 0.02, n_steps), 0, None).astype(np.float32)

    return load, solar, timestamps


def run_episode(env: EnergyEnv, controller, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return {
        "total_reward_inr": total_reward,
        "grid_import_kwh": info["episode_grid_import_kwh"],
        "grid_export_kwh": info["episode_grid_export_kwh"],
        "unmet_load_kwh": info["episode_unmet_load_kwh"],
    }


def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS MPC Controller Check  —  START")
    logger.info("━" * 55)

    load, solar, timestamps = build_household_series(n_days=60, seed=999, load_scale=1.2)
    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])
    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    rb_env = EnergyEnv(load_kw=load, solar_kw=solar, timestamps=timestamps,
                        battery_config=battery_config, grid=grid, episode_days=30, seed=999)
    mpc_env = EnergyEnv(load_kw=load, solar_kw=solar, timestamps=timestamps,
                         battery_config=battery_config, grid=grid, episode_days=30, seed=999)

    rb_controller = RuleBasedController()
    logger.info("Running rule-based controller...")
    t0 = time.time()
    rb_stats = run_episode(rb_env, rb_controller, seed=999)
    logger.info(f"  Rule-based done in {time.time()-t0:.1f}s")

    mpc_controller = MPCController(horizon_hours=24.0)
    logger.info("Running MPC controller (this solves an LP at every one of 2880 steps — expect it to take a while)...")
    t0 = time.time()
    mpc_stats = run_episode(mpc_env, mpc_controller, seed=999)
    elapsed = time.time() - t0
    logger.info(f"  MPC done in {elapsed:.1f}s ({elapsed/2880*1000:.1f}ms/step)")

    logger.info(f"\n  Rule-based : reward=₹{rb_stats['total_reward_inr']:.2f}  "
                f"import={rb_stats['grid_import_kwh']:.1f}kWh  "
                f"export={rb_stats['grid_export_kwh']:.1f}kWh  "
                f"unmet={rb_stats['unmet_load_kwh']:.1f}kWh")
    logger.info(f"  MPC        : reward=₹{mpc_stats['total_reward_inr']:.2f}  "
                f"import={mpc_stats['grid_import_kwh']:.1f}kWh  "
                f"export={mpc_stats['grid_export_kwh']:.1f}kWh  "
                f"unmet={mpc_stats['unmet_load_kwh']:.1f}kWh")

    logger.success("\n" + "━" * 55)
    logger.success("  MPC CHECK COMPLETE")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()