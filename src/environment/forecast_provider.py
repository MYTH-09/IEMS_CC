"""
IEMS – Weather and Energy Forecast Provider
============================================
Supplies the EnergyEnv and DQN agent with lookahead forecasts:
- Solar irradiance / PV generation forecast (1h, 3h, 6h)
- Rain probability (1h, 3h, 6h)
- Cloud cover (1h, 3h, 6h)
- Load demand forecast (3h, 6h)
- Outage probability (1h, 3h, 6h)
- Energy deficit and required SOC for anticipatory charging
- Forecast confidence

Supports:
- Realistic forecast error / uncertainty simulation during training
  (to prevent overfitting to perfect future knowledge, as required by SMART_DQN.md).
- Deterministic / operational lookahead for evaluation and live inference.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger


class ForecastProvider:
    """
    Computes multi-horizon weather, energy, and grid outage forecasts.

    Parameters
    ----------
    training_mode : bool
        If True, injects stochastic forecast noise (over/under estimation,
        false rain alerts, missed events) to train robust agents.
    noise_solar_std : float
        Relative standard deviation for solar forecast error during training.
    noise_rain_std : float
        Additive noise on rain probability during training.
    power_norm_kw : float
        Standard power normalization constant (default 15.0 kW).
    """

    def __init__(
        self,
        training_mode: bool = False,
        noise_solar_std: float = 0.15,
        noise_rain_std: float = 0.10,
        power_norm_kw: float = 15.0,
        seed: Optional[int] = None,
    ):
        self.training_mode = training_mode
        self.noise_solar_std = noise_solar_std
        self.noise_rain_std = noise_rain_std
        self.power_norm_kw = power_norm_kw
        self._rng = np.random.default_rng(seed)

    def set_mode(self, training: bool) -> None:
        """Switch between training mode (noisy forecasts) and eval mode."""
        self.training_mode = training

    def get_forecast(
        self,
        env,
        current_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute normalized forecast features for the current environment step.

        Horizons (at 15-min resolution):
        - 1 hour = 4 steps
        - 3 hours = 12 steps
        - 6 hours = 24 steps
        """
        idx = env._data_idx if current_idx is None else current_idx
        solar_kw = env.solar_kw
        load_kw = env.load_kw
        timestamps = env.timestamps
        grid = env.grid
        battery = env.battery

        n_total = len(solar_kw)
        max_idx = n_total - 1
        safe_idx = min(idx, max_idx)
        current_ts = timestamps[safe_idx]

        step_1h = 4
        step_3h = 12
        step_6h = 24

        # Extract future slices with edge padding
        def _get_slice(start_step: int, end_step: int, array: np.ndarray) -> np.ndarray:
            s = min(safe_idx + start_step, max_idx)
            e = min(safe_idx + end_step, max_idx)
            if s >= e:
                return np.full(end_step - start_step, array[max_idx], dtype=np.float32)
            segment = array[s:e]
            if len(segment) < (end_step - start_step):
                pad_len = (end_step - start_step) - len(segment)
                pad = np.full(pad_len, array[max_idx], dtype=np.float32)
                segment = np.concatenate([segment, pad])
            return segment

        future_solar_1h = _get_slice(0, step_1h, solar_kw)
        future_solar_3h = _get_slice(0, step_3h, solar_kw)
        future_solar_6h = _get_slice(0, step_6h, solar_kw)

        future_load_3h = _get_slice(0, step_3h, load_kw)
        future_load_6h = _get_slice(0, step_6h, load_kw)

        # ── 1. Solar generation forecasts ─────────────────────────────────────
        # Theoretical clear sky model for time of day to estimate cloud cover
        # STC estimate: peak around solar noon (12:00-13:00)
        future_hours = np.array([
            (timestamps[min(safe_idx + k, max_idx)].hour +
             timestamps[min(safe_idx + k, max_idx)].minute / 60.0)
            for k in range(step_6h)
        ])

        # Estimate clear-sky potential based on daylight hours (06:00 to 18:30)
        daylight_mask = (future_hours >= 6.0) & (future_hours <= 18.5)
        clear_sky_curve = np.zeros(step_6h, dtype=np.float32)
        clear_sky_curve[daylight_mask] = np.sin(
            np.pi * (future_hours[daylight_mask] - 6.0) / 12.5
        )
        # Scale to max capacity observed or default 5 kWp
        max_pot = max(float(np.max(solar_kw[:min(len(solar_kw), 960)])), 3.0)
        clear_sky_kw = clear_sky_curve * max_pot

        # Base physical future solar (mean over horizon)
        mean_solar_1h = float(np.mean(future_solar_1h))
        mean_solar_3h = float(np.mean(future_solar_3h))
        mean_solar_6h = float(np.mean(future_solar_6h))

        mean_load_3h = float(np.mean(future_load_3h))
        mean_load_6h = float(np.mean(future_load_6h))

        # ── 2. Cloud cover & Rain probability ─────────────────────────────────
        # Cloud cover is inferred from solar depression below clear sky during daylight
        def _calc_cloud_cover(actual_slice: np.ndarray, clear_slice: np.ndarray, hours_slice: np.ndarray) -> float:
            day_mask = (hours_slice >= 7.0) & (hours_slice <= 17.5)
            if not np.any(day_mask):
                # Nighttime: persistent cloud cover estimated from recent daytime or 0.1
                return 0.15
            clear_sub = clear_slice[day_mask]
            actual_sub = actual_slice[day_mask]
            # Ratio of suppression
            clear_total = np.sum(clear_sub)
            if clear_total < 1e-3:
                return 0.15
            suppression = 1.0 - (np.sum(actual_sub) / (clear_total + 1e-4))
            return float(np.clip(suppression, 0.0, 1.0))

        cloud_1h = _calc_cloud_cover(future_solar_1h, clear_sky_kw[:step_1h], future_hours[:step_1h])
        cloud_3h = _calc_cloud_cover(future_solar_3h, clear_sky_kw[:step_3h], future_hours[:step_3h])
        cloud_6h = _calc_cloud_cover(future_solar_6h, clear_sky_kw[:step_6h], future_hours[:step_6h])

        # Rain probability correlates with heavy cloud cover and season (monsoon months June-Sept in India)
        current_month = current_ts.month
        monsoon_boost = 0.25 if (6 <= current_month <= 9) else 0.05

        # If daytime solar drops severely during high clear sky, rain/storm likelihood is elevated
        rain_prob_1h = float(np.clip(cloud_1h * 0.75 + monsoon_boost, 0.0, 1.0))
        rain_prob_3h = float(np.clip(cloud_3h * 0.70 + monsoon_boost, 0.0, 1.0))
        rain_prob_6h = float(np.clip(cloud_6h * 0.65 + monsoon_boost, 0.0, 1.0))

        # ── 3. Outage probability ─────────────────────────────────────────────
        # Peak hours from grid configuration (e.g. 10, 14, 18, 22)
        peak_hours = getattr(grid, "_outage_peak_hours", [10, 14, 18, 22])

        def _calc_outage_prob(step_horizon: int, rain_p: float) -> float:
            probs = []
            for step_offset in range(step_horizon):
                step_idx = min(safe_idx + step_offset, max_idx)
                ts_step = timestamps[step_idx]
                h = int(ts_step.hour)
                # Baseline hazard
                h_prob = 0.75 if h in peak_hours else 0.15
                # Weather-induced grid cut risk (storms/rain frequently cause trips in Indian distribution)
                w_prob = h_prob + (rain_p * 0.25)
                probs.append(min(1.0, w_prob))
            # Probability of at least one outage in the window
            return float(np.clip(np.max(probs), 0.0, 1.0))

        outage_prob_1h = _calc_outage_prob(step_1h, rain_prob_1h)
        outage_prob_3h = _calc_outage_prob(step_3h, rain_prob_3h)
        outage_prob_6h = _calc_outage_prob(step_6h, rain_prob_6h)

        # ── 4. Training noise & uncertainty perturbation ──────────────────────
        confidence = 0.90
        if self.training_mode:
            # Overestimate or underestimate solar
            solar_noise_mult = float(self._rng.normal(1.0, self.noise_solar_std))
            solar_noise_mult = np.clip(solar_noise_mult, 0.65, 1.35)
            mean_solar_1h *= solar_noise_mult
            mean_solar_3h *= solar_noise_mult
            mean_solar_6h *= solar_noise_mult

            # Rain noise (occasional false alarm or missed forecast)
            rain_noise = float(self._rng.normal(0.0, self.noise_rain_std))
            rain_prob_1h = float(np.clip(rain_prob_1h + rain_noise, 0.0, 1.0))
            rain_prob_3h = float(np.clip(rain_prob_3h + rain_noise, 0.0, 1.0))
            rain_prob_6h = float(np.clip(rain_prob_6h + rain_noise, 0.0, 1.0))

            # Outage probability noise
            outage_noise = float(self._rng.normal(0.0, 0.08))
            outage_prob_1h = float(np.clip(outage_prob_1h + outage_noise, 0.0, 1.0))
            outage_prob_3h = float(np.clip(outage_prob_3h + outage_noise, 0.0, 1.0))
            outage_prob_6h = float(np.clip(outage_prob_6h + outage_noise, 0.0, 1.0))

            confidence = float(np.clip(self._rng.uniform(0.65, 0.95), 0.0, 1.0))

        # ── 5. Deficit calculation and Required SOC ───────────────────────────
        # Integrated energy over 3-hour horizon (12 steps of 0.25h)
        dt = env.dt_hours
        expected_solar_kwh_3h = mean_solar_3h * (step_3h * dt)
        expected_load_kwh_3h = mean_load_3h * (step_3h * dt)
        expected_deficit_kwh_3h = max(0.0, expected_load_kwh_3h - expected_solar_kwh_3h)

        battery_cap = max(float(battery.capacity_kwh), 1.0)
        # Required SOC to survive expected deficit during outage risk
        risk_weight = max(outage_prob_3h, rain_prob_3h * 0.8)
        soc_headroom = (expected_deficit_kwh_3h * risk_weight) / battery_cap
        required_soc = float(np.clip(float(battery.min_soc) + soc_headroom, float(battery.min_soc), float(battery.max_soc)))

        # Normalize outputs to [0, 1]
        p_norm = self.power_norm_kw

        return {
            "rain_probability_1h": float(np.clip(rain_prob_1h, 0.0, 1.0)),
            "rain_probability_3h": float(np.clip(rain_prob_3h, 0.0, 1.0)),
            "rain_probability_6h": float(np.clip(rain_prob_6h, 0.0, 1.0)),
            "cloud_cover_1h": float(np.clip(cloud_1h, 0.0, 1.0)),
            "cloud_cover_3h": float(np.clip(cloud_3h, 0.0, 1.0)),
            "cloud_cover_6h": float(np.clip(cloud_6h, 0.0, 1.0)),
            "forecast_solar_1h_norm": float(np.clip(mean_solar_1h / p_norm, 0.0, 1.0)),
            "forecast_solar_3h_norm": float(np.clip(mean_solar_3h / p_norm, 0.0, 1.0)),
            "forecast_solar_6h_norm": float(np.clip(mean_solar_6h / p_norm, 0.0, 1.0)),
            "forecast_load_3h_norm": float(np.clip(mean_load_3h / p_norm, 0.0, 1.0)),
            "forecast_load_6h_norm": float(np.clip(mean_load_6h / p_norm, 0.0, 1.0)),
            "outage_probability_1h": float(np.clip(outage_prob_1h, 0.0, 1.0)),
            "outage_probability_3h": float(np.clip(outage_prob_3h, 0.0, 1.0)),
            "outage_probability_6h": float(np.clip(outage_prob_6h, 0.0, 1.0)),
            "required_soc": float(required_soc),
            "forecast_deficit_3h_kwh": float(expected_deficit_kwh_3h),
            "forecast_confidence": float(confidence),
        }

    def get_forecast_vector(self, env, current_idx: Optional[int] = None) -> np.ndarray:
        """
        Returns a 15-dimensional float32 array of normalized forecast features:
        [0] rain_prob_1h
        [1] rain_prob_3h
        [2] rain_prob_6h
        [3] cloud_cover_1h
        [4] cloud_cover_3h
        [5] cloud_cover_6h
        [6] forecast_solar_1h_norm
        [7] forecast_solar_3h_norm
        [8] forecast_solar_6h_norm
        [9] forecast_load_3h_norm
        [10] forecast_load_6h_norm
        [11] outage_prob_1h
        [12] outage_prob_3h
        [13] outage_prob_6h
        [14] required_soc
        """
        fc = self.get_forecast(env, current_idx)
        return np.array([
            fc["rain_probability_1h"],
            fc["rain_probability_3h"],
            fc["rain_probability_6h"],
            fc["cloud_cover_1h"],
            fc["cloud_cover_3h"],
            fc["cloud_cover_6h"],
            fc["forecast_solar_1h_norm"],
            fc["forecast_solar_3h_norm"],
            fc["forecast_solar_6h_norm"],
            fc["forecast_load_3h_norm"],
            fc["forecast_load_6h_norm"],
            fc["outage_probability_1h"],
            fc["outage_probability_3h"],
            fc["outage_probability_6h"],
            fc["required_soc"],
        ], dtype=np.float32)

