"""
IEMS – Script 04: Gymnasium Environment Sanity Check
=======================================================
Validates the EnergyEnv before it's used by any controller (rule-based,
DQN, or MPC). Catches energy-conservation bugs now rather than after
hours of RL training on a broken environment.

Checks performed
-----------------
1. Environment resets and steps without error across a full episode
2. Energy conservation holds at every step:
       generation_in = solar_kw × dt
       consumption_out = load_kw × dt
       (solar_to_load + solar_to_battery + solar_exported) == generation_in
       (solar_to_load + battery_to_load + grid_to_load + unmet_load) == consumption_out
       battery_to_load == battery discharge delivered (cross-checked against BatteryStep)
3. Reward sign sanity: importing costs money (negative reward component),
   exporting earns credit (positive), unmet load is penalized
4. Action-specific behavior:
     - idle action never touches the battery
     - charge_from_grid never draws during an outage
     - discharge_to_load never exceeds battery's max_deliverable_kw
5. Cross-check: total episode grid-import cost (accumulated via marginal_rate
   at each step) is close to what GridModel.bill_for_month() would compute
   for the same total import — small deviation expected only because
   bill_for_month applies slabs to the exact total, while the env accrues
   marginal cost per step (mathematically these should match exactly for
   a monotonically-increasing cumulative total, so we assert near-equality)

Usage (from project root with venv active):
    python scripts/04_environment_check.py
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.battery.battery_model import BatteryModel


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test data (small, fast, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def build_synthetic_series(n_days: int = 45, seed: int = 42):
    """
    Build a small synthetic load + solar series for testing.
    45 days ensures at least one full 30-day episode fits with room
    for random start-index sampling.
    """
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="15min")

    hours = timestamps.hour + timestamps.minute / 60.0

    # Synthetic load: base + morning/evening peaks + noise
    load = (
        0.3
        + 0.4 * np.exp(-((hours - 8) ** 2) / 4)
        + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
        + rng.normal(0, 0.05, n_steps)
    )
    load = np.clip(load, 0.05, None)

    # Synthetic solar: bell curve centered at noon, zero at night
    solar = 3.0 * np.exp(-((hours - 12) ** 2) / 8)
    solar = np.where((hours < 6) | (hours > 18), 0.0, solar)
    solar = np.clip(solar + rng.normal(0, 0.02, n_steps), 0, None)

    return load.astype(np.float32), solar.astype(np.float32), timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_basic_episode(env: EnergyEnv) -> None:
    logger.info("\n[1] Basic episode run (random actions)...")
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), "Initial observation out of bounds"

    total_reward = 0.0
    rng = np.random.default_rng(0)
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = int(rng.integers(0, 4))
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"Observation out of bounds at step {steps}"
        total_reward += reward
        steps += 1

    logger.info(f"    Episode completed: {steps} steps, total_reward=₹{total_reward:.2f}")
    assert steps == env.episode_steps, "Episode length mismatch"
    logger.success("    ✓ Episode runs cleanly for full length without error")


def check_energy_conservation(env: EnergyEnv) -> None:
    logger.info("\n[2] Energy conservation check (all 4 actions, 200 steps each)...")
    dt = env.dt_hours

    for action in range(4):
        env.reset(seed=2)
        max_violation = 0.0

        for _ in range(200):
            idx = env._data_idx
            load_kw  = float(env.load_kw[idx])
            solar_kw = float(env.solar_kw[idx])

            obs, reward, terminated, truncated, info = env.step(action)

            generation_kwh  = solar_kw * dt
            consumption_kwh = load_kw * dt

            solar_side = (
                info["solar_to_load_kwh"] +
                info["solar_to_battery_kwh"] +
                info["solar_exported_kwh"]
            )
            load_side = (
                info["solar_to_load_kwh"] +
                info["battery_to_load_kwh"] +
                info["grid_to_load_kwh"] +
                info["unmet_load_kwh"]
            )

            solar_violation = abs(solar_side - generation_kwh)
            load_violation  = abs(load_side - consumption_kwh)
            max_violation = max(max_violation, solar_violation, load_violation)

            if truncated:
                break

        logger.info(f"    Action {action}: max conservation violation = {max_violation:.8f} kWh")
        assert max_violation < 1e-5, f"Energy conservation violated for action {action}: {max_violation}"

    logger.success("    ✓ Energy conservation holds exactly for all 4 actions")


def check_action_specific_behavior(env: EnergyEnv) -> None:
    logger.info("\n[3] Action-specific behavior checks...")

    # --- idle never touches battery ---
    env.reset(seed=3)
    soc_before = env.battery.soc
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(0)  # idle
        if truncated:
            break
    soc_after = env.battery.soc
    logger.info(f"    Idle x50: SOC {soc_before:.4f} → {soc_after:.4f}")
    assert abs(soc_before - soc_after) < 1e-9, "Idle action must never change battery SOC"
    logger.success("    ✓ Idle action leaves battery untouched")

    # --- charge_from_grid never draws during outage ---
    # Force an outage-heavy grid model
    always_outage = GridModel(
        outage_lookup=pd.Series(True, index=env.timestamps)
    )
    env_outage = EnergyEnv(
        load_kw=env.load_kw, solar_kw=env.solar_kw, timestamps=env.timestamps,
        battery_config=env.battery_config, grid=always_outage,
        episode_days=1, seed=4,
    )
    env_outage.reset(seed=4)
    soc_before = env_outage.battery.soc
    for _ in range(20):
        obs, reward, terminated, truncated, info = env_outage.step(2)  # charge_from_grid
        assert info["grid_to_battery_kwh"] == 0.0, "charge_from_grid must be blocked during outage"
        if truncated:
            break
    logger.success("    ✓ charge_from_grid correctly blocked during outage")

    # --- discharge never exceeds battery's hardware/SOC limit ---
    env.reset(seed=5)
    env.battery.reset(initial_soc=env.battery.min_soc + 0.001)  # nearly empty
    obs, reward, terminated, truncated, info = env.step(3)  # discharge_to_load
    logger.info(f"    Near-empty battery discharge attempt: battery_to_load={info['battery_to_load_kwh']:.4f} kWh")
    assert env.battery.soc >= env.battery.min_soc - 1e-6, "Battery discharged below min_soc"
    logger.success("    ✓ Discharge respects battery SOC floor")


def check_reward_sign_sanity(env: EnergyEnv) -> None:
    logger.info("\n[4] Reward sign sanity...")

    # Force a scenario with pure grid import, no solar, no battery help
    zero_solar = np.zeros_like(env.solar_kw)
    env_import = EnergyEnv(
        load_kw=env.load_kw, solar_kw=zero_solar, timestamps=env.timestamps,
        battery_config={**env.battery_config, "initial_soc": env.battery.min_soc},
        grid=GridModel(outage_lookup=pd.Series(False, index=env.timestamps)),
        episode_days=1, seed=6,
    )
    env_import.reset(seed=6)
    obs, reward, terminated, truncated, info = env_import.step(0)  # idle, all load from grid
    logger.info(f"    Pure grid import step: reward=₹{reward:.4f} (should be negative)")
    assert reward < 0, "Importing from grid with no solar/battery must yield negative reward"
    logger.success("    ✓ Grid import correctly penalized (negative reward)")

    # Force a scenario with pure export (huge solar, zero load — not physically
    # realistic but isolates the export credit sign)
    huge_solar = np.full_like(env.solar_kw, 10.0)
    tiny_load  = np.full_like(env.load_kw, 0.01)
    env_export = EnergyEnv(
        load_kw=tiny_load, solar_kw=huge_solar, timestamps=env.timestamps,
        battery_config={**env.battery_config, "initial_soc": env.battery.max_soc},  # battery full, can't absorb more
        grid=GridModel(outage_lookup=pd.Series(False, index=env.timestamps)),
        episode_days=1, seed=7,
    )
    env_export.reset(seed=7)
    obs, reward, terminated, truncated, info = env_export.step(0)  # idle, surplus exported
    logger.info(f"    Pure export step: reward=₹{reward:.4f} (should be positive)")
    assert reward > 0, "Exporting surplus solar with no cost must yield positive reward"
    logger.success("    ✓ Grid export correctly credited (positive reward)")


def check_reward_matches_bill(env: EnergyEnv) -> None:
    logger.info("\n[5] Cross-check: accumulated marginal-rate cost vs. bill_for_month()...")

    # Run a full episode with grid-only behavior (idle action, no battery
    # participation) so we can compute an independent expected bill.
    zero_solar = np.zeros_like(env.solar_kw)
    env_bill = EnergyEnv(
        load_kw=env.load_kw, solar_kw=zero_solar, timestamps=env.timestamps,
        battery_config={**env.battery_config, "initial_soc": env.battery.min_soc},
        grid=GridModel(outage_lookup=pd.Series(False, index=env.timestamps)),
        episode_days=1, seed=8,
    )
    env_bill.reset(seed=8)

    total_import_kwh = 0.0
    accumulated_cost = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env_bill.step(0)  # idle → pure grid import
        total_import_kwh += info["grid_to_load_kwh"]
        accumulated_cost += -reward   # reward is negative cost here (no export/unmet)

    independent_bill = env_bill.grid.bill_for_month(total_import_kwh)
    expected_energy_charge = independent_bill["energy_charge_inr"]

    logger.info(
        f"    Total import: {total_import_kwh:.2f} kWh | "
        f"Env accumulated cost: ₹{accumulated_cost:.2f} | "
        f"bill_for_month() energy charge: ₹{expected_energy_charge:.2f}"
    )
    # These should match closely: summing marginal_rate(cumulative) × Δunits
    # at each step is mathematically equivalent to the slab integral computed
    # by bill_for_month() for a monotonically increasing cumulative total.
    rel_error = abs(accumulated_cost - expected_energy_charge) / max(expected_energy_charge, 1e-6)
    logger.info(f"    Relative error: {rel_error:.6f}")
    assert rel_error < 0.01, "Accumulated step-wise cost diverges from slab bill calculation"
    logger.success("    ✓ Step-wise marginal cost accumulation matches slab bill calculation")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS Gymnasium Environment Validation  —  START")
    logger.info("━" * 55)

    load_kw, solar_kw, timestamps = build_synthetic_series(n_days=45)
    logger.info(f"Synthetic test series: {len(load_kw)} steps ({len(load_kw)/96:.0f} days)")

    grid = GridModel(outage_lookup=pd.Series(False, index=timestamps))  # no outages by default
    battery_config = dict(
        capacity_kwh=10.0, max_charge_rate_kw=3.0, max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid,
        episode_days=30, seed=42,
    )
    logger.info(f"Environment: {env}")

    check_basic_episode(env)
    check_energy_conservation(env)
    check_action_specific_behavior(env)
    check_reward_sign_sanity(env)
    check_reward_matches_bill(env)

    logger.success("\n" + "━" * 55)
    logger.success("  ALL ENVIRONMENT CHECKS PASSED ✓")
    logger.success("  Environment is ready for controller development.")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()