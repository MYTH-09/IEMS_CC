"""
IEMS – Historical Forecast Provider (streaming demo mode)
===============================================================
Provides MPCController with a GENUINE next-24h forecast, sourced not
from a live weather API but from the real, already-cached NASA POWER
solar data (and real synthetic-but-grounded load data) — replayed
chronologically, one day at a time.

Why this is legitimate, not "cheating"
-------------------------------------------
The default simulation pipeline uses PERSISTENCE forecasting
("tomorrow = 24h ago") specifically to keep MPC's comparison against
DQN fair, since DQN never sees the future either (see mpc_controller.py's
module docstring). This module is for a DIFFERENT, explicitly separate
purpose: demonstrating what MPC looks like when given an ACCURATE
forecast — the realistic condition for a live deployment where a real
weather forecast arrives each morning. The data used here (NASA POWER
via pv_model.py) is REAL RECORDED weather for real historical dates —
using it as "the forecast" for those same dates is not fabricating
information, it's replaying history as if experienced day-by-day rather
than loaded all at once.

This is NOT used by, and does not change, any of the already-validated
comparisons (grid_search, nsga2_sizing, validate_candidates,
09_compare_controllers) — those keep using MPCController's default
persistence forecasting untouched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


class HistoricalForecastProvider:
    """
    Picklable callable matching MPCController's `forecast_provider`
    signature: (env, horizon_steps) -> (solar_forecast, load_forecast).

    Returns the ACTUAL real solar/load values for the next `horizon_steps`
    starting at the environment's current index — genuine lookahead into
    real historical data, standing in for "today's forecast, which turned
    out to be accurate" in a streaming/demo replay of a real year.
    """

    def __init__(self, label: str = "historical_replay"):
        self.label = label
        self._calls = 0

    def __call__(self, env, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        idx = env._data_idx
        max_idx = len(env.solar_kw)
        self._calls += 1

        end = idx + horizon_steps
        if end <= max_idx:
            solar_forecast = env.solar_kw[idx:end]
            load_forecast = env.load_kw[idx:end]
        else:
            # Ran past the end of the real data series — pad with the
            # last available real value rather than crash (can happen
            # near the final day of the cached year).
            avail = max_idx - idx
            pad = horizon_steps - avail
            solar_forecast = np.concatenate([
                env.solar_kw[idx:max_idx],
                np.full(pad, env.solar_kw[max_idx - 1] if max_idx > 0 else 0.0, dtype=np.float32),
            ])
            load_forecast = np.concatenate([
                env.load_kw[idx:max_idx],
                np.full(pad, env.load_kw[max_idx - 1] if max_idx > 0 else 0.0, dtype=np.float32),
            ])

        return solar_forecast.astype(np.float32), load_forecast.astype(np.float32)