"""
IEMS – Script: Train and Track Smart Forecast-Aware DQN Controller
===================================================================
Implements the architecture specified in SMART_DQN.md:
1. Lookahead weather & energy forecast observation (32 dimensions).
2. Diverse multi-city real smart meter data pool (Mathura, Bareilly, Bengaluru, Mumbai, etc.).
3. Realistic forecast error and uncertainty injection during training.
4. Future readiness & blackout anticipation reward design.
5. Real-time logging, TensorBoard tracking, and checkpointing.

Usage:
    # Full production training (e.g. 100,000 - 300,000 timesteps):
    python scripts/train_smart_dqn.py --steps 100000

    # Quick test run:
    python scripts/train_smart_dqn.py --quick
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.energy_env import EnergyEnv
from src.environment.forecast_provider import ForecastProvider
from src.grid.grid_model import GridModel
from src.controllers.safety_shield import SafetyShield
from src.controllers.dqn_controller import train_dqn, DQNController
from src.controllers.rule_based import RuleBasedController
from src.pipeline.stage_functions import _derive_monthly_outage_multipliers

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    _SB3_AVAILABLE = True
except Exception:
    DQN = None
    BaseCallback = object
    EvalCallback = None
    _SB3_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Real Dataset Pool Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_smart_training_pool(
    n_households: int = 24,
    n_days: int = 90,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Constructs a rich, diverse pool of training households using:
    - Real smart meter load traces from data/processed/meters_15min.parquet
    - Real multi-city solar generation data (Mathura, Bareilly, Bengaluru, Mumbai, Delhi, etc.)
    - Hardware variations: battery 3.0 to 20.0 kWh, panels 2.0 to 10.0 kWp
    - Injected weather anomalies (clear skies, monsoon rain spells, storm outages)
    """
    rng = np.random.default_rng(seed)
    n_steps = n_days * 96
    households: List[Dict[str, Any]] = []

    # 1. Load real meter IDs if available
    meters_path = "data/processed/meters_15min.parquet"
    real_meter_data = None
    available_meters = []
    if os.path.exists(meters_path):
        try:
            logger.info("Loading real smart meter data from data/processed/meters_15min.parquet...")
            mdf = pd.read_parquet(meters_path, columns=["meter_id", "power_kw", "city"])
            available_meters = list(mdf["meter_id"].unique())
            real_meter_data = mdf
            logger.info(f"Loaded real dataset with {len(available_meters)} unique meters.")
        except Exception as e:
            logger.warning(f"Could not load full meters parquet ({e}), will use synthetic blend.")

    # 2. Available solar datasets
    cities = ["Mathura", "Bareilly", "Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
    solar_caches = {}
    for city in cities:
        spath = f"data/solar/{city}_solar_15min.parquet"
        if os.path.exists(spath):
            try:
                sdf = pd.read_parquet(spath, columns=["solar_power_kw"])
                solar_caches[city] = sdf["solar_power_kw"].to_numpy(dtype=np.float32)
            except Exception:
                pass

    panel_options = [2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]
    battery_options = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]

    for i in range(n_households):
        h_seed = seed + i * 37
        h_rng = np.random.default_rng(h_seed)

        timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="15min")
        hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)

        # A. Load Series (extract from real meter or grounded synthetic profile)
        load_extracted = False
        load_kw = None
        if real_meter_data is not None and len(available_meters) > 0:
            m_id = available_meters[i % len(available_meters)]
            try:
                sub = real_meter_data[real_meter_data["meter_id"] == m_id]["power_kw"].to_numpy(dtype=np.float32)
                if len(sub) >= n_steps:
                    start_offset = int(h_rng.integers(0, max(1, len(sub) - n_steps)))
                    load_kw = sub[start_offset : start_offset + n_steps].copy()
                    # Add light variation and clip
                    scale = float(h_rng.uniform(0.7, 1.4))
                    load_kw = np.clip(load_kw * scale, 0.05, 12.0).astype(np.float32)
                    load_extracted = True
            except Exception:
                load_extracted = False

        if not load_extracted:
            # High-fidelity multi-modal synthetic household
            scale = float(h_rng.uniform(0.6, 1.8))
            base = 0.2 + 0.1 * h_rng.uniform(0.5, 1.5)
            morning_peak = 0.5 * np.exp(-((hours - float(h_rng.uniform(7.0, 9.0))) ** 2) / 3.5)
            evening_peak = 0.8 * np.exp(-((hours - float(h_rng.uniform(18.5, 21.0))) ** 2) / 4.0)
            weekend_factor = 1.0 + 0.2 * (timestamps.dayofweek >= 5)
            noise = h_rng.normal(0, 0.08, n_steps)
            load_kw = np.clip((base + morning_peak + evening_peak) * scale * weekend_factor + noise, 0.05, 15.0).astype(np.float32)

        # B. Solar Series (real city data + panel capacity)
        chosen_city = cities[i % len(cities)]
        chosen_panel_kw = panel_options[i % len(panel_options)]
        solar_kw = None

        if chosen_city in solar_caches and len(solar_caches[chosen_city]) >= n_steps:
            real_s = solar_caches[chosen_city]
            start_s = int(h_rng.integers(0, max(1, len(real_s) - n_steps)))
            solar_kw = (real_s[start_s : start_s + n_steps] * chosen_panel_kw).astype(np.float32)
        else:
            # Synthetic solar curve with cloud/rain episodes
            clear_sky = chosen_panel_kw * np.sin(np.pi * np.clip((hours - 6.0) / 12.0, 0.0, 1.0))
            clear_sky = np.where((hours < 6.0) | (hours > 18.0), 0.0, clear_sky)
            solar_kw = clear_sky.astype(np.float32)

        # C. Weather & Rain Storm Injection
        # Inject realistic rain & storm episodes (monsoon or storm spells):
        # During rain storms, solar collapses to 10-25% and outage risk spikes
        n_rain_events = int(h_rng.integers(2, 6))
        for _ in range(n_rain_events):
            event_start_day = int(h_rng.integers(5, n_days - 5))
            event_start_step = event_start_day * 96 + int(h_rng.integers(36, 68)) # afternoon/evening
            event_duration_steps = int(h_rng.integers(8, 24)) # 2 to 6 hours
            end_step = min(n_steps, event_start_step + event_duration_steps)
            # Attenuate solar severely
            solar_kw[event_start_step:end_step] *= float(h_rng.uniform(0.05, 0.25))

        # D. Battery configuration
        cap_kwh = battery_options[i % len(battery_options)]
        charge_rate = min(5.0, max(2.5, cap_kwh * 0.4))
        battery_config = {
            "capacity_kwh": cap_kwh,
            "max_charge_rate_kw": charge_rate,
            "max_discharge_rate_kw": charge_rate,
            "round_trip_efficiency": 0.90,
            "min_soc": 0.10,
            "max_soc": 0.95,
            "initial_soc": 0.50,
        }

        households.append({
            "load_kw": load_kw,
            "solar_kw": solar_kw,
            "timestamps": timestamps,
            "battery_config": battery_config,
            "city": chosen_city,
            "panel_kw": chosen_panel_kw,
        })

    logger.info(f"Built training pool with {len(households)} diverse multi-city households.")
    return households


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Tracking Callback
# ─────────────────────────────────────────────────────────────────────────────

class SmartDQNTrackingCallback(BaseCallback):
    """
    Logs training progress in real-time to both console, log file, and JSON:
    - Current timestep and percentage
    - Mean episodic reward
    - Grid import, export, and unmet load
    - Pre-charging & readiness statistics
    """

    def __init__(
        self,
        log_freq: int = 5000,
        log_file: str = "logs/dqn_smart_training.log",
        metrics_file: str = "outputs/dqn_smart_training_progress.json",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_file = log_file
        self.metrics_file = metrics_file
        self.start_time = None
        self.history: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== IEMS SMART DQN TRAINING SESSION STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            elapsed = time.time() - self.start_time
            steps = self.n_calls
            fps = int(steps / max(elapsed, 1e-3))

            # Query recent episode rewards from SB3 monitor/rollout if available
            ep_rewards = self.model.ep_info_buffer if hasattr(self.model, "ep_info_buffer") else []
            mean_ep_r = float(np.mean([ep["r"] for ep in ep_rewards])) if len(ep_rewards) > 0 else 0.0
            mean_ep_l = float(np.mean([ep["l"] for ep in ep_rewards])) if len(ep_rewards) > 0 else 0.0

            log_entry = {
                "step": steps,
                "elapsed_sec": round(elapsed, 1),
                "fps": fps,
                "mean_episode_reward": round(mean_ep_r, 2),
                "episodes_completed": len(ep_rewards),
            }
            self.history.append(log_entry)

            msg = (
                f"[Step {steps:7,d}] Elapsed: {elapsed/60:5.1f}m | FPS: {fps:4d} | "
                f"Mean Ep Reward: INR {mean_ep_r:7.2f} | Completed Eps: {len(ep_rewards)}"
            )
            logger.info(msg)

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | {msg}\n")

            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump({"history": self.history, "last_update": time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        final_msg = f"=== SMART DQN TRAINING FINISHED IN {elapsed/60:.2f} MINUTES ({self.n_calls:,} steps) ==="
        logger.success(final_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{final_msg}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Routine
# ─────────────────────────────────────────────────────────────────────────────

def train_smart_dqn(
    total_timesteps: int = 150_000,
    n_households: int = 24,
    save_path: str = "models/dqn_smart_policy.zip",
    best_model_dir: str = "models/checkpoints_smart",
    tensorboard_log: str = "logs/dqn_smart_tensorboard",
    seed: int = 42,
) -> Any:
    """
    Executes end-to-end training of the forecast-aware Smart DQN policy.
    """
    logger.info("━" * 60)
    logger.info("  IEMS SMART DQN CONTROLLER TRAINING")
    logger.info(f"  Target steps: {total_timesteps:,} | Households: {n_households}")
    logger.info(f"  Output Model: {save_path}")
    logger.info(f"  Checkpoints:  {best_model_dir}")
    logger.info("━" * 60)

    # 1. Build diverse pool
    pool = build_smart_training_pool(n_households=n_households, n_days=90, seed=seed)
    primary = pool[0]
    additional = pool[1:]

    grid = GridModel(
        fallback_outage_hours_per_day=4.0,
        outage_peak_hours=[10, 14, 18, 22],
        monthly_outage_multiplier=_derive_monthly_outage_multipliers("Mathura"),
    )

    # 2. Build training environment with forecast uncertainty simulation
    train_forecast_provider = ForecastProvider(training_mode=True, seed=seed)
    train_env = EnergyEnv(
        load_kw=primary["load_kw"],
        solar_kw=primary["solar_kw"],
        timestamps=primary["timestamps"],
        battery_config=primary["battery_config"],
        grid=grid,
        episode_days=30,
        seed=seed,
        additional_households=additional,
        use_forecast=True,
        forecast_provider=train_forecast_provider,
    )
    logger.info(f"Train Env Obs Space: {train_env.observation_space.shape} (Expected 32)")

    # 3. Build held-out evaluation environment (eval mode: clean forecast)
    eval_pool = build_smart_training_pool(n_households=4, n_days=60, seed=seed + 9999)
    eval_primary = eval_pool[0]
    eval_forecast_provider = ForecastProvider(training_mode=False, seed=seed + 9999)
    eval_env = EnergyEnv(
        load_kw=eval_primary["load_kw"],
        solar_kw=eval_primary["solar_kw"],
        timestamps=eval_primary["timestamps"],
        battery_config=eval_primary["battery_config"],
        grid=grid,
        episode_days=30,
        seed=seed + 9999,
        additional_households=eval_pool[1:],
        use_forecast=True,
        forecast_provider=eval_forecast_provider,
    )

    # 4. Setup Callbacks
    tracking_callback = SmartDQNTrackingCallback(
        log_freq=max(2000, total_timesteps // 50),
        log_file="logs/dqn_smart_training.log",
        metrics_file="outputs/dqn_smart_training_progress.json",
    )

    eval_callback = None
    if EvalCallback is not None:
        eval_freq = max(5000, total_timesteps // 30)
        os.makedirs(best_model_dir, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=best_model_dir,
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            render=False,
        )

    # 5. Hyperparameters tuned for 32-dim anticipatory control
    hyperparams = dict(
        learning_rate=0.0003,
        batch_size=128,
        buffer_size=150_000,
        gamma=0.99,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.04,
        target_update_interval=500,
        learning_starts=1500,
        train_freq=4,
        gradient_steps=2,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callbacks = [tracking_callback]
    if eval_callback is not None:
        callbacks.append(eval_callback)

    from stable_baselines3.common.callbacks import CallbackList
    combined_callback = CallbackList(callbacks)

    # 6. Train model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    model = train_dqn(
        env=train_env,
        total_timesteps=total_timesteps,
        hyperparams=hyperparams,
        save_path=save_path,
        tensorboard_log=tensorboard_log,
        verbose=0,
        seed=seed,
        callback=combined_callback,
    )

    # Copy to best checkpoint if eval_callback didn't trigger
    best_ckpt = os.path.join(best_model_dir, "best_model.zip")
    if model is not None and not os.path.exists(best_ckpt):
        model.save(best_ckpt)

    logger.success(f"✓ Smart DQN Training Complete! Policy saved -> {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train and track Smart Forecast-Aware DQN")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps (default: 100,000)")
    parser.add_argument("--quick", action="store_true", help="Quick run (5,000 steps) for smoke testing")
    parser.add_argument("--households", type=int, default=24, help="Number of households in pool (default: 24)")
    parser.add_argument("--output", type=str, default="models/dqn_smart_policy.zip", help="Path to save trained policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    steps = 5000 if args.quick else args.steps
    train_smart_dqn(
        total_timesteps=steps,
        n_households=args.households,
        save_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
