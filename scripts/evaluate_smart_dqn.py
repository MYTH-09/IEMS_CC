"""
IEMS – Script: Evaluate and Benchmark Smart Forecast-Aware DQN Controller
==========================================================================
Compares:
  1. Smart DQN (Forecast-Aware 32-dim with Safety Shield)
  2. Legacy DQN (17-dim baseline)
  3. Rule-Based Baseline
  4. MPC Controller (Model Predictive Control)

Evaluates on:
  - Scenario 1: Rain & Outage at 6 PM (SMART_DQN.md challenge)
  - Scenario 2: Multi-day Monsoon Spell (low solar + frequent outages)
  - Scenario 3: Generalization across held-out households (seeds 888, 777, 666)

Outputs results to console and saves structured JSON to:
  outputs/smart_dqn_benchmark_results.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.environment.energy_env import EnergyEnv
from src.environment.forecast_provider import ForecastProvider
from src.grid.grid_model import GridModel
from src.controllers.dqn_controller import DQNController
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.controllers.safety_shield import SafetyShield


def simulate_controller(env: EnergyEnv, controller, seed: int = 42) -> Dict[str, Any]:
    """Runs one full episode simulation for a controller on the given env."""
    obs, info = env.reset(seed=seed)
    controller.reset()

    total_reward = 0.0
    soc_history = []
    actions_taken = []
    unmet_kwh_list = []
    grid_import_kwh_list = []

    terminated = truncated = False
    step_count = 0

    while not (terminated or truncated):
        action = controller.select_action(env)
        actions_taken.append(int(action))

        idx_before = env._data_idx
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        soc_history.append(float(env.battery.soc))
        unmet_kwh_list.append(float(info.get("unmet_load_kwh", 0.0)))
        grid_import_kwh_list.append(float(info.get("grid_import_kwh", 0.0)))
        step_count += 1

    return {
        "reward": round(total_reward, 2),
        "grid_import_kwh": round(float(env.episode_grid_import_kwh), 2),
        "grid_export_kwh": round(float(env.episode_grid_export_kwh), 2),
        "unmet_load_kwh": round(float(env.episode_unmet_load_kwh), 2),
        "total_cost_inr": round(float(env.episode_cost_inr), 2),
        "avg_soc_pct": round(float(np.mean(soc_history) * 100), 1),
        "min_soc_pct": round(float(np.min(soc_history) * 100), 1),
        "action_distribution": {
            "idle": int(np.sum(np.array(actions_taken) == 0)),
            "charge_solar": int(np.sum(np.array(actions_taken) == 1)),
            "charge_grid": int(np.sum(np.array(actions_taken) == 2)),
            "discharge": int(np.sum(np.array(actions_taken) == 3)),
        },
        "soc_history": soc_history,
    }


def build_rain_at_6pm_scenario():
    """
    Builds the explicit benchmark scenario from SMART_DQN.md:
    - 2-day simulation (192 steps)
    - Day 1:
      - 06:00 to 15:00: Normal load and solar
      - 15:00 (3 PM): Rain forecast for 6 PM (rain prob 85%, cloud 90%, expected solar 0.2 kW)
      - 18:00 (6 PM) to 21:00 (9 PM): Heavy rain strikes + grid blackout occurs!
    """
    n_steps = 288  # 3 days of data for a 2-day episode
    timestamps = pd.date_range("2025-07-15 00:00", periods=n_steps, freq="15min")
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)

    # Load: evening peak of 2.2 kW
    load_kw = 0.3 + 0.4 * np.exp(-((hours - 8.5) ** 2) / 3.0) + 1.2 * np.exp(-((hours - 19.5) ** 2) / 4.0)
    load_kw = np.clip(load_kw, 0.1, 3.5).astype(np.float32)

    # Solar: 5 kWp panel, clear sky until 17:30, then severe rain collapse
    solar_kw = 4.0 * np.sin(np.pi * np.clip((hours - 6.0) / 12.0, 0.0, 1.0))
    solar_kw = np.where((hours < 6.0) | (hours > 18.0), 0.0, solar_kw)

    # Inject storm at 18:00 - 21:30 on day 1 (steps 72 to 86)
    solar_kw[72:86] = 0.05

    # Outage on day 1 from 18:00 to 21:00 (steps 72 to 84)
    outage_steps = set(range(72, 85))

    class StormGridModel(GridModel):
        def is_outage(self, ts) -> bool:
            # Check if index is in storm window
            idx = int((ts - timestamps[0]).total_seconds() / 900)
            if idx in outage_steps:
                return True
            return super().is_outage(ts)

    grid = StormGridModel(fallback_outage_hours_per_day=2.0)
    battery_config = {
        "capacity_kwh": 7.5,
        "max_charge_rate_kw": 3.0,
        "max_discharge_rate_kw": 3.0,
        "round_trip_efficiency": 0.90,
        "min_soc": 0.10,
        "max_soc": 0.95,
        "initial_soc": 0.50, # 50% at start
    }

    env = EnergyEnv(
        load_kw=load_kw,
        solar_kw=solar_kw,
        timestamps=timestamps,
        battery_config=battery_config,
        grid=grid,
        episode_days=2,
        seed=123,
        use_forecast=True,
    )
    return env, timestamps


def run_benchmark():
    os.makedirs("outputs", exist_ok=True)
    logger.info("━" * 65)
    logger.info("  IEMS SMART FORECAST-AWARE DQN BENCHMARK")
    logger.info("━" * 65)

    # Initialize controllers
    smart_dqn_path = "models/dqn_smart_policy.zip"
    if not os.path.exists(smart_dqn_path) and os.path.exists("models/checkpoints_smart/best_model.zip"):
        smart_dqn_path = "models/checkpoints_smart/best_model.zip"

    legacy_dqn_path = "models/archive/dqn_legacy_17obs_backup.zip"
    if not os.path.exists(legacy_dqn_path):
        legacy_dqn_path = "models/dqn_policy.zip"

    controllers = {
        "Smart DQN (Forecast-Aware)": DQNController(model_path=smart_dqn_path),
        "Legacy DQN (17-dim)": DQNController(model_path=legacy_dqn_path),
        "Rule-Based": RuleBasedController(),
    }

    all_results = {}

    # ── Test 1: The Rain at 6 PM & Outage Challenge ───────────────────────────
    logger.info("\n[Test 1] Running Challenge: 'Rain at 6 PM & Outage at 6 PM'...")
    t1_results = {}
    for name, ctrl in controllers.items():
        env, ts = build_rain_at_6pm_scenario()
        res = simulate_controller(env, ctrl, seed=123)
        # SOC right when the storm & outage hits at step 72 (18:00)
        soc_at_storm = res["soc_history"][72] if len(res["soc_history"]) > 72 else 0.0
        res["soc_at_storm_pct"] = round(soc_at_storm * 100, 1)
        del res["soc_history"]  # keep output clean
        t1_results[name] = res

        logger.info(
            f"  {name:26s} | Reward: INR {res['reward']:7.1f} | Unmet: {res['unmet_load_kwh']:4.1f} kWh | "
            f"Cost: INR {res['total_cost_inr']:6.1f} | SOC@18:00: {res['soc_at_storm_pct']:4.1f}% | "
            f"Grid Charge Actions: {res['action_distribution']['charge_grid']}"
        )

    all_results["rain_at_6pm_challenge"] = t1_results

    # ── Test 2: Generalization Across Households ──────────────────────────────
    logger.info("\n[Test 2] Running Generalization Across Diverse Held-Out Households...")
    import runpy
    ns_07 = runpy.run_path("scripts/07_train_dqn.py")
    build_household_series = ns_07["build_household_series"]
    seeds = [(888, 0.9), (777, 1.4), (666, 0.7)]
    gen_results = {}

    for h_seed, l_scale in seeds:
        load, solar, timestamps = build_household_series(n_days=60, seed=h_seed, load_scale=l_scale)
        grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])
        b_cfg = {
            "capacity_kwh": 10.0, "max_charge_rate_kw": 3.0, "max_discharge_rate_kw": 3.0,
            "round_trip_efficiency": 0.90, "min_soc": 0.10, "max_soc": 0.95, "initial_soc": 0.50,
        }

        h_key = f"seed_{h_seed}_scale_{l_scale}"
        gen_results[h_key] = {}
        logger.info(f"  Evaluating household {h_key}:")

        for name, ctrl in controllers.items():
            env = EnergyEnv(
                load_kw=load, solar_kw=solar, timestamps=timestamps,
                battery_config=b_cfg, grid=grid, episode_days=30, seed=h_seed,
                use_forecast=True,
            )
            res = simulate_controller(env, ctrl, seed=h_seed)
            del res["soc_history"]
            gen_results[h_key][name] = res
            logger.info(
                f"    {name:24s} | Reward: INR {res['reward']:7.1f} | Unmet: {res['unmet_load_kwh']:4.1f} kWh | "
                f"Import: {res['grid_import_kwh']:5.1f} kWh | Cost: INR {res['total_cost_inr']:6.1f}"
            )

    all_results["generalization_benchmarks"] = gen_results

    # Save to JSON
    out_file = os.path.join(PROJECT_ROOT, "outputs", "smart_dqn_benchmark_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.success(f"\nBenchmark complete! Results saved to {out_file}")
    return all_results


if __name__ == "__main__":
    run_benchmark()
