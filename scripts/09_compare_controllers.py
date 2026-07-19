"""
IEMS – Script 09: Three-Way Controller Comparison (§8.4)
=============================================================
Runs rule-based, DQN, and MPC controllers on the IDENTICAL held-out
household and hardware spec, and produces one clean comparison table.

This is the core experimental result of the project: given a fixed
system (same battery, same solar, same household), how much better does
intelligent control do than a simple fixed rule?

Uses the SAME held-out household (seed=999, load_scale=1.2) that both
scripts/07_train_dqn.py and scripts/08_mpc_check.py already evaluated
individually — this script just brings all three together in one place
so the numbers are directly comparable, and saves them to disk for the
eventual PDF/dashboard report (§10).

Requires models/dqn_policy.zip to already exist (run
`python scripts/07_train_dqn.py --full` first for a meaningful result —
an undertrained quick-smoke-test policy will make DQN look artificially
bad here, which is expected, not a bug).

Usage:
    python scripts/09_compare_controllers.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController
from src.controllers.dqn_controller import DQNController
from src.controllers.mpc_controller import MPCController


def build_household_series(n_days: int = 60, seed: int = 999, load_scale: float = 1.2):
    """Same held-out household used by scripts 07 and 08 — kept identical
    on purpose so all three controllers are compared on the same data."""
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


def build_env(load, solar, timestamps, grid) -> EnergyEnv:
    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )
    return EnergyEnv(
        load_kw=load, solar_kw=solar, timestamps=timestamps,
        battery_config=battery_config, grid=grid, episode_days=30, seed=999,
    )


def run_episode(env: EnergyEnv, controller, seed: int = 999) -> dict:
    obs, info = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    terminated = truncated = False
    t0 = time.time()
    while not (terminated or truncated):
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    elapsed = time.time() - t0

    return {
        "controller": controller.name,
        "total_reward_inr": round(total_reward, 2),
        "total_cost_inr": round(info["episode_cost_inr"], 2),
        "grid_import_kwh": round(info["episode_grid_import_kwh"], 2),
        "grid_export_kwh": round(info["episode_grid_export_kwh"], 2),
        "unmet_load_kwh": round(info["episode_unmet_load_kwh"], 2),
        "final_battery_soc": round(info["battery_soc"], 4),
        "battery_cycles": round(env.battery.total_cycles, 3),
        "runtime_sec": round(elapsed, 2),
    }


def main() -> None:
    logger.info("━" * 60)
    logger.info("  IEMS Three-Way Controller Comparison  —  START")
    logger.info("━" * 60)

    load, solar, timestamps = build_household_series(n_days=60, seed=999, load_scale=1.2)
    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])

    dqn_model_path = "models/dqn_policy.zip"
    if not os.path.exists(dqn_model_path):
        logger.error(
            f"'{dqn_model_path}' not found. Run "
            "`python scripts/07_train_dqn.py --full` first."
        )
        sys.exit(1)

    controllers = [
        ("Rule-Based", RuleBasedController()),
        ("DQN",        DQNController(model_path=dqn_model_path)),
        ("MPC",        MPCController(horizon_hours=24.0)),
    ]

    results = []
    for label, controller in controllers:
        logger.info(f"\nRunning {label}...")
        env = build_env(load, solar, timestamps, grid)
        stats = run_episode(env, controller, seed=999)
        results.append(stats)
        logger.info(
            f"  reward=₹{stats['total_reward_inr']:.2f}  "
            f"import={stats['grid_import_kwh']:.1f}kWh  "
            f"export={stats['grid_export_kwh']:.1f}kWh  "
            f"unmet={stats['unmet_load_kwh']:.1f}kWh  "
            f"runtime={stats['runtime_sec']:.1f}s"
        )

    # ── Comparison table ────────────────────────────────────────────────────
    df = pd.DataFrame(results).set_index("controller")
    logger.info("\n" + "─" * 90)
    logger.info("  COMPARISON TABLE")
    logger.info("─" * 90)
    logger.info(f"\n{df.to_string()}")

    # ── Relative improvement vs rule-based baseline ─────────────────────────
    baseline_reward = df.loc["rule_based", "total_reward_inr"]
    logger.info("\n" + "─" * 60)
    logger.info("  IMPROVEMENT OVER RULE-BASED BASELINE")
    logger.info("─" * 60)
    for name in df.index:
        if name == "rule_based":
            continue
        delta = df.loc[name, "total_reward_inr"] - baseline_reward
        pct = (delta / abs(baseline_reward) * 100) if baseline_reward != 0 else float("nan")
        logger.info(f"  {name:>12s}: ₹{delta:+.2f}  ({pct:+.1f}% relative to rule-based)")

    # ── Save results for the report/dashboard ──────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/controller_comparison.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "household_spec": {"n_days": 60, "seed": 999, "load_scale": 1.2},
                "battery_spec": {"capacity_kwh": 10.0, "max_rate_kw": 3.0, "rte": 0.90},
                "solar_spec": {"peak_kw": 3.0},
                "episode_days": 30,
                "results": results,
            },
            f, indent=2,
        )
    logger.success(f"\nResults saved → {output_path}")

    logger.success("\n" + "━" * 60)
    logger.success("  COMPARISON COMPLETE")
    logger.success("━" * 60)


if __name__ == "__main__":
    main()