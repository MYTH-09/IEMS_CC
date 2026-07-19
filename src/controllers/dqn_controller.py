"""
IEMS – DQN Controller (§8.2)
===============================
Trains a Deep Q-Network on EnergyEnv using Stable-Baselines3, and wraps
the trained policy behind the same `BaseController` interface as the
rule-based controller, so both can be evaluated through one shared
harness (`scripts/07_train_controllers.py`).

Two distinct concerns live in this file:
  1. `train_dqn()` — trains a fresh policy and saves it to disk.
  2. `DQNController` — loads a saved policy and implements `select_action`,
     for use anywhere a controller is needed (evaluation, sizing, etc).

Why training uses a SEPARATE, wider household distribution than a single
fixed load/solar series
--------------------------------------------------------------------------
Training on one fixed household's year of data risks the policy
memorizing that household's specific pattern rather than learning a
transferable charge/discharge strategy. `train_dqn()` accepts a callable
`env_factory` that can return a *freshly sampled* environment (different
synthetic household, different season) on each `reset()`-triggering
episode, so the trained policy generalizes. For a first pass / sanity
run, a single fixed environment is also supported — see
`scripts/07_train_controllers.py` for both usage patterns.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from loguru import logger

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.controllers.base_controller import BaseController
from src.environment.energy_env import EnergyEnv


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DQN_HYPERPARAMS = dict(
    learning_rate=0.0003,
    batch_size=64,
    buffer_size=100_000,
    gamma=0.99,
    exploration_fraction=0.20,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    learning_starts=2000,
    train_freq=4,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def train_dqn(
    env: EnergyEnv,
    total_timesteps: int = 500_000,
    hyperparams: Optional[dict] = None,
    save_path: str = "models/dqn_policy.zip",
    tensorboard_log: Optional[str] = "logs/dqn_tensorboard",
    verbose: int = 1,
    seed: int = 42,
) -> DQN:
    """
    Train a DQN policy on the given environment and save it to disk.

    Parameters
    ----------
    env : EnergyEnv
        The (already constructed) environment to train on. For a single
        fixed household, pass one env instance directly. For better
        generalization across households, wrap env construction in a
        loop that periodically rebuilds it with a different synthetic
        profile (see the training script for the pattern).
    total_timesteps : int
        Total environment steps to train for. Config default: 500,000
        (≈ 174 episodes of 2880 steps each, at 30 days/episode).
    hyperparams : Optional[dict]
        Overrides for DEFAULT_DQN_HYPERPARAMS.
    save_path : str
        Where to save the trained policy (SB3 .zip format).
    tensorboard_log : Optional[str]
        Directory for TensorBoard logs. Set to None to disable.

    Returns
    -------
    The trained DQN model (also saved to `save_path`).
    """
    # Validate environment compliance before spending any training time
    check_env(env, warn=True)

    params = {**DEFAULT_DQN_HYPERPARAMS, **(hyperparams or {})}

    logger.info(f"Training DQN for {total_timesteps:,} timesteps...")
    logger.info(f"  Hyperparameters: {params}")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        **params,
    )

    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except ImportError:
        # tqdm/rich not installed (stable-baselines3[extra] not fully present) —
        # fall back to training without the progress bar rather than failing.
        logger.warning(
            "Progress bar dependencies (tqdm, rich) not found — "
            "training without progress bar. Install via "
            "`pip install stable-baselines3[extra]` to enable it."
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.success(f"DQN training complete. Policy saved → {save_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference — wraps a trained policy behind the BaseController interface
# ─────────────────────────────────────────────────────────────────────────────

class DQNController(BaseController):
    """
    Loads a trained DQN policy and exposes it through the shared
    `select_action(env) -> int` interface, so it can be evaluated
    identically to the rule-based controller.
    """

    name = "dqn"

    def __init__(self, model_path: str = "models/dqn_policy.zip", deterministic: bool = True):
        """
        Parameters
        ----------
        model_path : str
            Path to a saved DQN policy (.zip, from `train_dqn`).
        deterministic : bool
            If True, always pick the highest-Q action (used for
            evaluation). If False, sample from the exploration policy
            (not typically used outside training).
        """
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = DQN.load(model_path)
        logger.info(f"Loaded DQN policy from {model_path}")

    def select_action(self, env: EnergyEnv) -> int:
        obs = env._get_observation()
        action, _states = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)