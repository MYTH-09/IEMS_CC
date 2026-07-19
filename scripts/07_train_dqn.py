"""
IEMS – Script 07: Train and Evaluate DQN Controller
=======================================================
Trains a DQN policy and compares it against the rule-based baseline on
a held-out household. Two modes, controlled by `--quick`:

  Quick mode (default in this script as shipped): trains for a small
  number of timesteps (~5,000) purely to verify the full pipeline works
  end-to-end (env → training → save → load → inference → comparison).
  This does NOT produce a good policy — it's a wiring smoke test.

  Full training: set `total_timesteps=500_000` (the config default) for
  an actual policy worth evaluating. On CPU this takes roughly 30-90
  minutes depending on hardware; a GPU is not required for a network
  this small (net_arch=[256,256]) but will speed things up further.

Usage:
    python scripts/07_train_dqn.py            # quick smoke-test run
    python scripts/07_train_dqn.py --full      # full 500k-timestep training
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
from src.controllers.dqn_controller import train_dqn, DQNController


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic household generator — used for both training variety and eval
# ─────────────────────────────────────────────────────────────────────────────

def build_household_series(n_days: int = 60, seed: int = 42, load_scale: float = 1.0):
    """
    Build a synthetic load + solar series for one household. Varying
    `seed` and `load_scale` across calls gives the DQN a distribution of
    households to train on, rather than memorizing a single fixed one.
    """
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


def build_training_env(n_households: int = 6, n_days: int = 60) -> EnergyEnv:
    """
    Build a training environment with a POOL of diverse synthetic
    households, so DQN is forced to learn a transferable charge/discharge
    strategy rather than memorizing one household's specific rhythm.

    Diversity dimensions varied across the pool: random seed (different
    day-to-day noise pattern) and load_scale (different household size,
    0.6x-1.6x a moderate baseline). All households share the same
    battery_config and grid model — that's held constant here so the
    fix under test (multi-household rotation) is isolated; varying
    battery/grid too is a natural next extension once this is confirmed
    to help.
    """
    primary = build_household_series(n_days=n_days, seed=100, load_scale=1.0)
    primary_dict = {"load_kw": primary[0], "solar_kw": primary[1], "timestamps": primary[2]}

    load_scales = np.linspace(0.6, 1.6, n_households - 1)
    additional = []
    for i, scale in enumerate(load_scales):
        load, solar, ts = build_household_series(n_days=n_days, seed=200 + i, load_scale=float(scale))
        additional.append({"load_kw": load, "solar_kw": solar, "timestamps": ts})

    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])
    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )
    env = EnergyEnv(
        load_kw=primary_dict["load_kw"], solar_kw=primary_dict["solar_kw"],
        timestamps=primary_dict["timestamps"],
        battery_config=battery_config, grid=grid, episode_days=30, seed=42,
        additional_households=additional,
    )
    logger.info(f"Training pool: {len(env._household_pool)} households "
                f"(load_scale range ≈ 0.6x-1.6x)")
    return env


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


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full 500k-timestep training instead of the quick smoke test")
    args = parser.parse_args()

    total_timesteps = 500_000 if args.full else 5_000
    mode_label = "FULL TRAINING" if args.full else "QUICK SMOKE TEST"

    logger.info("━" * 55)
    logger.info(f"  IEMS DQN Controller — {mode_label}")
    logger.info("━" * 55)
    if not args.full:
        logger.warning(
            "  Running in quick mode (5,000 timesteps). This verifies the "
            "pipeline works but will NOT produce a competent policy. "
            "Run with --full for real training (500k timesteps, ~30-90 min)."
        )

    # ── Train ────────────────────────────────────────────────────────────────
    train_env = build_training_env(n_households=6, n_days=60)
    logger.info(f"Training environment: {train_env}")

    model = train_dqn(
        env=train_env,
        total_timesteps=total_timesteps,
        save_path="models/dqn_policy.zip",
        tensorboard_log=None,   # keep smoke test lightweight; enable for --full runs
        verbose=1,
        seed=42,
    )

    # ── Evaluate on a held-out household (different seed → different profile) ─
    logger.info("\n" + "─" * 55)
    logger.info("  Evaluating on held-out household (seed=999)...")
    logger.info("─" * 55)

    eval_load, eval_solar, eval_timestamps = build_household_series(n_days=60, seed=999, load_scale=1.2)
    eval_grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])
    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    dqn_env = EnergyEnv(
        load_kw=eval_load, solar_kw=eval_solar, timestamps=eval_timestamps,
        battery_config=battery_config, grid=eval_grid, episode_days=30, seed=999,
    )
    rb_env = EnergyEnv(
        load_kw=eval_load, solar_kw=eval_solar, timestamps=eval_timestamps,
        battery_config=battery_config, grid=eval_grid, episode_days=30, seed=999,
    )

    dqn_controller = DQNController(model_path="models/dqn_policy.zip")
    rb_controller  = RuleBasedController()

    dqn_stats = run_episode(dqn_env, dqn_controller, seed=999)
    rb_stats  = run_episode(rb_env, rb_controller, seed=999)

    logger.info(f"\n  DQN         : reward=₹{dqn_stats['total_reward_inr']:.2f}  "
                f"import={dqn_stats['grid_import_kwh']:.1f}kWh  "
                f"export={dqn_stats['grid_export_kwh']:.1f}kWh  "
                f"unmet={dqn_stats['unmet_load_kwh']:.1f}kWh")
    logger.info(f"  Rule-based  : reward=₹{rb_stats['total_reward_inr']:.2f}  "
                f"import={rb_stats['grid_import_kwh']:.1f}kWh  "
                f"export={rb_stats['grid_export_kwh']:.1f}kWh  "
                f"unmet={rb_stats['unmet_load_kwh']:.1f}kWh")

    if not args.full:
        logger.warning(
            "\n  Quick mode: DQN is undertrained (5k steps) and is NOT expected "
            "to beat rule-based yet. This run only confirms the pipeline "
            "(train → save → load → infer) works without error."
        )
    else:
        if dqn_stats["total_reward_inr"] > rb_stats["total_reward_inr"]:
            logger.success("\n  ✓ Trained DQN outperforms the rule-based baseline")
        else:
            logger.warning(
                "\n  DQN did not outperform rule-based — consider longer "
                "training, more household variety, or reward tuning."
            )

    logger.success("\n" + "━" * 55)
    logger.success("  DQN PIPELINE CHECK COMPLETE")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()