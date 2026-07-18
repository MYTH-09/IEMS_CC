"""
IEMS – Script 05: Rule-Based Controller Check
=================================================
Validates the rule-based baseline controller by running it through full
episodes and comparing its performance against two trivial baselines:

  - Random actions   (sanity floor — anything sensible should beat this)
  - Always idle       (no battery participation at all — everything
                        routes through solar-to-load-first + grid;
                        rule-based should still win by using the battery)

If rule-based doesn't outperform "always idle", the battery isn't
actually helping and something in the logic is wrong.

Usage (from project root with venv active):
    python scripts/05_rule_based_check.py
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController


# Reuse the same synthetic series generator as the environment check,
# so results are directly comparable to what we already validated.
def build_synthetic_series(n_days: int = 45, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="15min")
    hours = timestamps.hour + timestamps.minute / 60.0

    load = (
        0.3
        + 0.4 * np.exp(-((hours - 8) ** 2) / 4)
        + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
        + rng.normal(0, 0.05, n_steps)
    )
    load = np.clip(load, 0.05, None)

    solar = 3.0 * np.exp(-((hours - 12) ** 2) / 8)
    solar = np.where((hours < 6) | (hours > 18), 0.0, solar)
    solar = np.clip(solar + rng.normal(0, 0.02, n_steps), 0, None)

    return load.astype(np.float32), solar.astype(np.float32), timestamps


def run_episode(env: EnergyEnv, controller=None, fixed_action: int = None, seed: int = 42) -> dict:
    """
    Run one full episode using either a controller (select_action per step)
    or a fixed constant action every step. Returns summary stats.
    """
    obs, info = env.reset(seed=seed)
    if controller is not None:
        controller.reset()

    total_reward = 0.0
    terminated = truncated = False
    step_count = 0

    while not (terminated or truncated):
        if controller is not None:
            action = controller.select_action(env)
        else:
            action = fixed_action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    return {
        "total_reward_inr":       total_reward,
        "total_cost_inr":         info["episode_cost_inr"],
        "grid_import_kwh":        info["episode_grid_import_kwh"],
        "grid_export_kwh":        info["episode_grid_export_kwh"],
        "unmet_load_kwh":         info["episode_unmet_load_kwh"],
        "final_battery_soc":      info["battery_soc"],
        "steps":                  step_count,
    }


def print_summary(label: str, stats: dict) -> None:
    logger.info(f"\n  {label}")
    logger.info(f"  {'─'*45}")
    logger.info(f"    Total reward         : ₹{stats['total_reward_inr']:>10.2f}")
    logger.info(f"    Total cost           : ₹{stats['total_cost_inr']:>10.2f}")
    logger.info(f"    Grid import          : {stats['grid_import_kwh']:>10.2f} kWh")
    logger.info(f"    Grid export          : {stats['grid_export_kwh']:>10.2f} kWh")
    logger.info(f"    Unmet load           : {stats['unmet_load_kwh']:>10.2f} kWh")
    logger.info(f"    Final battery SOC    : {stats['final_battery_soc']:>10.1%}")


def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS Rule-Based Controller Check  —  START")
    logger.info("━" * 55)

    load_kw, solar_kw, timestamps = build_synthetic_series(n_days=45)

    # Add a realistic outage pattern: ~4 hrs/day, weighted toward evening
    # (matches the fallback statistical model's default assumptions)
    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])

    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid,
        episode_days=30, seed=42,
    )

    # ── Baseline 1: random actions ──────────────────────────────────────────
    rng_env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid, episode_days=30, seed=42,
    )
    obs, info = rng_env.reset(seed=42)
    rng = np.random.default_rng(0)
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(0, 4))
        obs, reward, terminated, truncated, info = rng_env.step(action)
        total_reward += reward
    random_stats = {
        "total_reward_inr": total_reward, "total_cost_inr": info["episode_cost_inr"],
        "grid_import_kwh": info["episode_grid_import_kwh"], "grid_export_kwh": info["episode_grid_export_kwh"],
        "unmet_load_kwh": info["episode_unmet_load_kwh"], "final_battery_soc": info["battery_soc"], "steps": 0,
    }
    print_summary("Baseline: RANDOM actions", random_stats)

    # ── Baseline 2: always idle (battery never participates) ──────────────
    idle_env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid, episode_days=30, seed=42,
    )
    idle_stats = run_episode(idle_env, fixed_action=0, seed=42)
    print_summary("Baseline: ALWAYS IDLE (no battery use)", idle_stats)

    # ── Rule-based controller ──────────────────────────────────────────────
    rb_env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid, episode_days=30, seed=42,
    )
    controller = RuleBasedController()
    rb_stats = run_episode(rb_env, controller=controller, seed=42)
    print_summary(f"RULE-BASED controller ({controller})", rb_stats)

    # ── Comparison ───────────────────────────────────────────────────────────
    logger.info("\n" + "━" * 55)
    logger.info("  COMPARISON")
    logger.info("━" * 55)
    logger.info(f"  Random        : ₹{random_stats['total_reward_inr']:>10.2f}")
    logger.info(f"  Always idle   : ₹{idle_stats['total_reward_inr']:>10.2f}")
    logger.info(f"  Rule-based    : ₹{rb_stats['total_reward_inr']:>10.2f}")

    assert rb_stats["total_reward_inr"] > random_stats["total_reward_inr"], \
        "Rule-based must outperform random actions"
    logger.success("  ✓ Rule-based outperforms random actions")

    assert rb_stats["total_reward_inr"] > idle_stats["total_reward_inr"], \
        "Rule-based must outperform never using the battery at all"
    logger.success("  ✓ Rule-based outperforms never using the battery")

    improvement_vs_idle = rb_stats["total_reward_inr"] - idle_stats["total_reward_inr"]
    logger.success(f"\n  Battery usage improved monthly reward by ₹{improvement_vs_idle:.2f} vs. no battery")

    logger.success("\n" + "━" * 55)
    logger.success("  ALL RULE-BASED CONTROLLER CHECKS PASSED ✓")
    logger.success("  Ready to serve as the DQN/MPC benchmark and")
    logger.success("  the sizing grid search controller.")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()