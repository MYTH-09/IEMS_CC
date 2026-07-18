"""
IEMS – Energy Management Gymnasium Environment
=================================================
Ties household load, solar generation, battery storage, and the grid
model together into a single stepped simulation that a rule-based
controller, DQN agent, or MPC controller can all act on through one
shared interface.

Action space (Discrete(4))
---------------------------
    0 = idle
    1 = charge_from_solar    (battery charges from solar surplus only)
    2 = charge_from_grid     (battery charges by importing from grid)
    3 = discharge_to_load    (battery discharges to cover household load)

Observation space (Box, 9 features)
-------------------------------------
    [0] solar_kw                       (current solar generation, kW)
    [1] load_kw                        (current household load, kW)
    [2] battery_soc                    (0-1)
    [3] hour_sin                       (cyclical encoding of hour-of-day)
    [4] hour_cos
    [5] is_weekend                     (0 or 1)
    [6] cumulative_units_norm          (month-to-date grid import, normalized)
    [7] is_outage                      (0 or 1, current timestep)
    [8] month_sin                      (cyclical encoding of month, for seasonality)

Reward
------
Denominated in ₹ (rupees), using the grid's slab-aware marginal rate:

    reward = -(grid_import_kwh × grid.marginal_rate(cumulative_units_so_far))
             + (grid_export_kwh × grid.net_metering_rate)
             - (unmet_load_penalty_rate × unmet_load_kwh)

`unmet_load_kwh` occurs only during outages when neither solar nor battery
can cover the household's load — this is penalized to discourage the
agent from running the battery empty right before an anticipated outage.

Episode structure
------------------
One episode = `episode_steps` timesteps (default: 30 days × 96 slots/day
= 2880 steps, aligned with one billing month so `cumulative_units_norm`
resets naturally at each episode boundary). `reset()` picks a new random
start index into the load/solar series (with a fixed seed option for
reproducible evaluation) and re-initializes battery SOC and cumulative
units to zero.
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from src.battery.battery_model import BatteryModel
from src.grid.grid_model import GridModel


# ── Constants ─────────────────────────────────────────────────────────────────
SLOTS_PER_DAY        = 96
DEFAULT_EPISODE_DAYS = 30
DEFAULT_DT_HOURS      = 15.0 / 60.0
UNMET_LOAD_PENALTY_RATE_INR = 15.0   # ₹/kWh — deliberately > top tariff slab,
                                       # so the agent learns unmet load during
                                       # a foreseeable outage is worse than any
                                       # grid import cost it could have avoided


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class EnergyEnv(gym.Env):
    """
    Gymnasium environment for household solar + battery + grid simulation.

    Parameters
    ----------
    load_kw : np.ndarray
        Household load power (kW) time series, 15-min resolution.
    solar_kw : np.ndarray
        Solar generation power (kW) time series, same length/resolution
        and alignment as `load_kw`.
    timestamps : pd.DatetimeIndex
        Timestamps aligned to `load_kw`/`solar_kw` — used to derive hour,
        weekday, month features and to query the grid's outage model.
    battery_config : dict
        Kwargs forwarded to `BatteryModel(**battery_config)`.
    grid : GridModel
        A configured GridModel instance (tariff slabs, outage lookup, etc).
    episode_days : int
        Length of one episode in days (default 30 — one billing month).
    dt_hours : float
        Simulation timestep duration in hours (default 0.25 — 15 min).
    seed : Optional[int]
        Seed for reproducible episode start sampling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        load_kw: np.ndarray,
        solar_kw: np.ndarray,
        timestamps: pd.DatetimeIndex,
        battery_config: Optional[dict] = None,
        grid: Optional[GridModel] = None,
        episode_days: int = DEFAULT_EPISODE_DAYS,
        dt_hours: float = DEFAULT_DT_HOURS,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert len(load_kw) == len(solar_kw) == len(timestamps), \
            "load_kw, solar_kw, and timestamps must be the same length"

        self.load_kw    = np.asarray(load_kw, dtype=np.float32)
        self.solar_kw   = np.asarray(solar_kw, dtype=np.float32)
        self.timestamps = pd.DatetimeIndex(timestamps)
        self.dt_hours   = dt_hours

        self.episode_steps = int(episode_days * SLOTS_PER_DAY)
        assert len(self.load_kw) > self.episode_steps, (
            f"Data series ({len(self.load_kw)} steps) must be longer than "
            f"one episode ({self.episode_steps} steps) to allow random start sampling."
        )

        self.battery_config = battery_config or {}
        self.battery = BatteryModel(**self.battery_config)
        self.grid    = grid or GridModel()

        # Precompute normalization constants
        self._max_power_kw = max(
            float(self.load_kw.max()), float(self.solar_kw.max()), 1.0
        ) * 1.5   # headroom above observed max
        # Normalize cumulative units against a generous monthly cap (for observation scaling only)
        self._cum_units_norm_cap = 1000.0

        self._rng = np.random.default_rng(seed)

        # ── Gym spaces ───────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(4)

        obs_low  = np.array([0, 0, 0, -1, -1, 0, 0, 0, -1], dtype=np.float32)
        obs_high = np.array([1, 1, 1,  1,  1, 1, 1, 1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── Episode state ────────────────────────────────────────────────────
        self._start_idx: int = 0
        self._step_idx:  int = 0            # steps taken within current episode
        self._data_idx:  int = 0            # absolute index into load_kw/solar_kw
        self._cumulative_units_this_month: float = 0.0

        # Episode-level accumulators (useful for logging / evaluation, not required by Gym)
        self.episode_grid_import_kwh:  float = 0.0
        self.episode_grid_export_kwh:  float = 0.0
        self.episode_unmet_load_kwh:   float = 0.0
        self.episode_cost_inr:         float = 0.0

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random valid start index for this episode
        max_start = len(self.load_kw) - self.episode_steps - 1
        self._start_idx = int(self._rng.integers(0, max_start + 1))
        self._step_idx  = 0
        self._data_idx  = self._start_idx

        self.battery.reset(initial_soc=self.battery_config.get("initial_soc"))
        self._cumulative_units_this_month = 0.0

        self.episode_grid_import_kwh = 0.0
        self.episode_grid_export_kwh = 0.0
        self.episode_unmet_load_kwh  = 0.0
        self.episode_cost_inr        = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        ts        = self.timestamps[self._data_idx]
        load_kw   = float(self.load_kw[self._data_idx])
        solar_kw  = float(self.solar_kw[self._data_idx])
        is_outage = self.grid.is_outage(ts)

        flows = self._apply_action(action, load_kw, solar_kw, is_outage)

        # ── Reward ──────────────────────────────────────────────────────────
        rate = self.grid.marginal_rate(self._cumulative_units_this_month)
        import_cost   = flows["grid_import_kwh"] * rate
        export_credit = flows["grid_export_kwh"] * self.grid.net_metering_rate
        unmet_penalty = flows["unmet_load_kwh"] * UNMET_LOAD_PENALTY_RATE_INR

        reward = -import_cost + export_credit - unmet_penalty

        # ── Update running state ───────────────────────────────────────────
        self._cumulative_units_this_month += flows["grid_import_kwh"]
        self.episode_grid_import_kwh += flows["grid_import_kwh"]
        self.episode_grid_export_kwh += flows["grid_export_kwh"]
        self.episode_unmet_load_kwh  += flows["unmet_load_kwh"]
        self.episode_cost_inr        += (import_cost - export_credit + unmet_penalty)

        self._step_idx += 1
        self._data_idx += 1

        terminated = False   # no natural "failure" terminal state in this env
        truncated  = self._step_idx >= self.episode_steps

        obs  = self._get_observation()
        info = self._get_info()
        info.update(flows)
        info["marginal_rate_inr_per_kwh"] = rate
        info["is_outage"] = is_outage

        return obs, float(reward), terminated, truncated, info

    # ── Action logic ─────────────────────────────────────────────────────────

    def _apply_action(
        self,
        action: int,
        load_kw: float,
        solar_kw: float,
        is_outage: bool,
    ) -> dict:
        """
        Resolve one timestep's energy balance given the chosen action.

        Returns a dict of energy flows (all in kWh for this timestep):
            solar_to_load_kwh, solar_to_battery_kwh, solar_exported_kwh,
            battery_to_load_kwh, grid_to_battery_kwh, grid_to_load_kwh,
            grid_import_kwh, grid_export_kwh, unmet_load_kwh
        """
        dt = self.dt_hours
        remaining_load  = load_kw   # kW still needing to be served
        remaining_solar = solar_kw  # kW of solar still available to allocate

        solar_to_load_kwh    = 0.0
        solar_to_battery_kwh = 0.0
        solar_exported_kwh   = 0.0
        battery_to_load_kwh  = 0.0
        grid_to_battery_kwh  = 0.0
        grid_to_load_kwh     = 0.0
        unmet_load_kwh       = 0.0

        # ── Step 1: solar always serves load first (physically free priority,
        #    independent of the chosen action — matches real inverter behavior) ──
        direct = min(remaining_load, remaining_solar)
        solar_to_load_kwh = direct * dt
        remaining_load    -= direct
        remaining_solar    -= direct

        # ── Step 2: apply the chosen action to the battery ─────────────────
        if action == 1:   # charge_from_solar
            if remaining_solar > 0:
                bstep = self.battery.step(charge_kw=remaining_solar, dt_hours=dt)
                charged_kw = bstep.actual_charge_kw
                solar_to_battery_kwh = charged_kw * dt
                remaining_solar -= charged_kw
            # any leftover solar after charging is exported below

        elif action == 2:   # charge_from_grid
            if not is_outage:
                max_accept = self.battery.max_acceptable_kw(dt_hours=dt)
                bstep = self.battery.step(charge_kw=max_accept, dt_hours=dt)
                grid_to_battery_kwh = bstep.actual_charge_kw * dt
            # during an outage, grid is unavailable — action has no effect

        elif action == 3:   # discharge_to_load
            if remaining_load > 0:
                bstep = self.battery.step(discharge_kw=remaining_load, dt_hours=dt)
                delivered_kw = bstep.actual_discharge_kw
                battery_to_load_kwh = delivered_kw * dt
                remaining_load -= delivered_kw

        # action == 0 (idle): battery untouched, remaining_solar/load pass through

        # ── Step 3: resolve leftover solar (export or curtail if outage) ───
        if remaining_solar > 0:
            if not is_outage:
                solar_exported_kwh = remaining_solar * dt
            # during an outage, export is impossible (grid-tied inverters
            # cannot export without a live grid) — solar is curtailed,
            # not exported. (Off-grid/hybrid inverter islanding is a
            # possible future extension, not modeled here.)

        # ── Step 4: resolve leftover load (grid import or unmet) ───────────
        if remaining_load > 0:
            if not is_outage:
                grid_to_load_kwh = remaining_load * dt
            else:
                unmet_load_kwh = remaining_load * dt

        grid_import_kwh = grid_to_battery_kwh + grid_to_load_kwh
        grid_export_kwh = solar_exported_kwh

        return {
            "solar_to_load_kwh":    solar_to_load_kwh,
            "solar_to_battery_kwh": solar_to_battery_kwh,
            "solar_exported_kwh":   solar_exported_kwh,
            "battery_to_load_kwh":  battery_to_load_kwh,
            "grid_to_battery_kwh":  grid_to_battery_kwh,
            "grid_to_load_kwh":     grid_to_load_kwh,
            "grid_import_kwh":      grid_import_kwh,
            "grid_export_kwh":      grid_export_kwh,
            "unmet_load_kwh":       unmet_load_kwh,
        }

    # ── Observation / info ──────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        idx = min(self._data_idx, len(self.load_kw) - 1)
        ts  = self.timestamps[idx]

        solar_norm = np.clip(self.solar_kw[idx] / self._max_power_kw, 0, 1)
        load_norm  = np.clip(self.load_kw[idx]  / self._max_power_kw, 0, 1)

        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        month_sin = np.sin(2 * np.pi * ts.month / 12.0)

        is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0
        cum_units_norm = np.clip(
            self._cumulative_units_this_month / self._cum_units_norm_cap, 0, 1
        )
        is_outage = 1.0 if self.grid.is_outage(ts) else 0.0

        obs = np.array([
            solar_norm, load_norm, self.battery.soc,
            hour_sin, hour_cos, is_weekend,
            cum_units_norm, is_outage, month_sin,
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "timestamp": self.timestamps[min(self._data_idx, len(self.timestamps)-1)],
            "battery_soc": self.battery.soc,
            "cumulative_units_this_month": self._cumulative_units_this_month,
            "episode_grid_import_kwh": self.episode_grid_import_kwh,
            "episode_grid_export_kwh": self.episode_grid_export_kwh,
            "episode_unmet_load_kwh": self.episode_unmet_load_kwh,
            "episode_cost_inr": self.episode_cost_inr,
        }

    def __repr__(self) -> str:
        return (
            f"EnergyEnv(episode_steps={self.episode_steps}, "
            f"data_len={len(self.load_kw)}, "
            f"battery={self.battery.capacity_kwh}kWh, "
            f"grid={self.grid})"
        )