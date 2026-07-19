"""
IEMS – MPC Controller (§8.3)
===============================
Model Predictive Control: at every timestep, solves a convex optimization
over a rolling look-ahead horizon (default 24h), then executes only the
first step's decision before re-solving at the next timestep (receding
horizon control).

Forecast method: PERSISTENCE (as specified in the build doc)
----------------------------------------------------------------
"Tomorrow looks like yesterday" — the forecast for the next H steps is
taken from the ACTUAL observed solar/load values exactly one day (96
steps) earlier in the series, not the true future. Using true future
values would give MPC an unfair perfect-foresight advantage over DQN
(which only ever sees the current state), making the three-way
comparison meaningless. Persistence is simple, realistic, and defensible.

Slab-tariff simplification (a deliberate, documented approximation)
----------------------------------------------------------------------
UP's slab tariff makes the true marginal rate a function of cumulative
monthly consumption — which is itself path-dependent on the very
decisions being optimized, making the exact problem non-convex. This
controller freezes the marginal rate at whatever `cumulative_units`
stands at the START of each horizon solve (not updated step-by-step
within the 24h window). This keeps the problem a clean, fast-solving
Linear Program. The approximation error is small in practice because a
single day rarely crosses a slab boundary — but it IS an approximation,
worth stating plainly rather than glossing over.

Action discretization
------------------------
The solver's continuous first-step decision is mapped onto the same
4-action interface DQN and rule-based use (idle / charge_from_solar /
charge_from_grid / discharge_to_load), so all three controllers can be
evaluated through the identical `EnergyEnv` action semantics — the
environment itself computes the exact physical flows for whichever
action is chosen, keeping all three controllers on equal footing rather
than letting MPC bypass the environment's own action-resolution logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cvxpy as cp
from loguru import logger

from src.controllers.base_controller import BaseController
from src.environment.energy_env import EnergyEnv


# Prefer ECOS (fast for small LPs) but fall back gracefully — newer cvxpy
# versions don't always bundle it, and forcing an unavailable solver would
# crash every solve. Any of these produce identical *correct* results for
# this problem since it's a plain LP; only speed differs.
# CLARABEL is preferred first: testing showed ECOS occasionally reports a
# spurious "infeasible" status on this problem (likely a warm-start +
# strict-tolerance interaction) even though the exact same problem solves
# cleanly with CLARABEL every time. Order otherwise falls back sensibly.
_PREFERRED_SOLVER_ORDER = ["CLARABEL", "ECOS", "HIGHS", "SCIPY", "SCS"]


def _pick_available_solver() -> Optional[str]:
    installed = cp.installed_solvers()
    for name in _PREFERRED_SOLVER_ORDER:
        if name in installed:
            return name
    return None   # let cvxpy auto-select as a last resort


class MPCController(BaseController):
    """
    Rolling-horizon MPC controller for battery dispatch.

    Parameters
    ----------
    horizon_hours : float
        Look-ahead window length (default 24h, per config `mpc.horizon_hours`).
    unmet_penalty_rate_inr : float
        ₹/kWh penalty for projected unmet load within the horizon — must
        match EnergyEnv's own penalty (see energy_env.py) so MPC's
        internal cost model agrees with what it's actually being scored on.
    action_threshold_kw : float
        Minimum solved kW below which a decision is treated as "idle"
        rather than triggering charge/discharge — avoids the controller
        selecting a nonzero action for numerically negligible amounts.
    """

    name = "mpc"

    def __init__(
        self,
        horizon_hours: float = 24.0,
        unmet_penalty_rate_inr: float = 15.0,
        action_threshold_kw: float = 0.05,
        solver: Optional[str] = None,
        dt_hours: float = 0.25,
    ):
        self.horizon_hours = horizon_hours
        self.unmet_penalty_rate_inr = unmet_penalty_rate_inr
        self.action_threshold_kw = action_threshold_kw
        self.solver = solver or _pick_available_solver()
        self.dt_hours = dt_hours
        self.horizon_steps = max(1, int(horizon_hours / dt_hours))

        logger.info(f"MPCController using solver: {self.solver}, horizon={self.horizon_steps} steps")

        self._problem = None
        self._battery_spec: Optional[tuple] = None   # (capacity, eta_c, eta_d, min_soc, max_soc, max_c, max_d)

    def reset(self) -> None:
        pass   # MPC is stateless across steps — each solve is self-contained

    # ─────────────────────────────────────────────────────────────────────

    def _build_problem(self, battery) -> None:
        """
        Build the cvxpy problem ONCE for a given battery spec, using
        Parameters for everything that changes step-to-step (forecasts,
        outage mask, current SOC, tariff rate). Rebuilding the full
        problem from scratch every step (the naive approach) costs
        ~400ms/step in Python-side canonicalization overhead alone —
        this pattern drops that to a few ms by reusing one compiled
        problem structure and only updating data between solves.
        """
        H = self.horizon_steps
        dt = self.dt_hours

        self._solar_param = cp.Parameter(H, nonneg=True)
        self._load_param  = cp.Parameter(H, nonneg=True)
        self._outage_param = cp.Parameter(H)   # 1.0 = outage, 0.0 = grid available
        self._soc0_param  = cp.Parameter()
        self._rate_param  = cp.Parameter(nonneg=True)

        self._charge    = cp.Variable(H, nonneg=True)
        self._discharge = cp.Variable(H, nonneg=True)
        self._grid_in   = cp.Variable(H, nonneg=True)
        self._grid_out  = cp.Variable(H, nonneg=True)
        self._unmet     = cp.Variable(H, nonneg=True)
        self._curtail   = cp.Variable(H, nonneg=True)
        self._soc       = cp.Variable(H + 1)

        # BIG_M safely exceeds any realistic residential grid flow (loads/solar
        # here are ~0.1-5 kW) while staying close enough to the problem's
        # actual scale to avoid the numerical ill-conditioning that a wildly
        # oversized big-M (e.g. 1000) can cause in LP solvers using strict
        # tolerances — this was the actual cause of rare spurious "infeasible"
        # solves observed during testing, not a true infeasibility.
        BIG_M = 50.0

        eta_c, eta_d = battery.eta_c, battery.eta_d
        capacity = battery.capacity_kwh
        min_soc, max_soc = battery.min_soc, battery.max_soc
        max_charge_kw, max_discharge_kw = battery.max_charge_rate_kw, battery.max_discharge_rate_kw

        constraints = [self._soc[0] == self._soc0_param]
        for t in range(H):
            constraints += [
                self._soc[t + 1] == self._soc[t]
                + (self._charge[t] * eta_c - self._discharge[t] / eta_d) * dt / capacity,
                self._soc[t + 1] >= min_soc,
                self._soc[t + 1] <= max_soc,
                self._charge[t] <= max_charge_kw,
                self._discharge[t] <= max_discharge_kw,
                self._grid_in[t]  <= BIG_M * (1 - self._outage_param[t]),
                self._grid_out[t] <= BIG_M * (1 - self._outage_param[t]),
                self._solar_param[t] + self._discharge[t] + self._grid_in[t] + self._unmet[t]
                == self._load_param[t] + self._charge[t] + self._grid_out[t] + self._curtail[t],
            ]

        export_rate = self._export_rate   # fixed for the whole simulation, set by caller before first build
        cost = (
            cp.sum(self._grid_in) * self._rate_param
            - cp.sum(self._grid_out) * export_rate
            + cp.sum(self._unmet) * self.unmet_penalty_rate_inr
        ) * dt

        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def _forecast(self, env: EnergyEnv, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Persistence forecast: solar/load for the next `horizon_steps` are
        taken from the ACTUAL values exactly one day (96 steps) earlier.
        Falls back to repeating the earliest available data if there
        isn't a full day of history yet (start of a household's series).
        """
        idx = env._data_idx
        lookback = 96   # one day at 15-min resolution

        source_start = idx - lookback
        if source_start < 0:
            solar_now = float(env.solar_kw[idx])
            load_now  = float(env.load_kw[idx])
            return (np.full(horizon_steps, solar_now, dtype=np.float32),
                    np.full(horizon_steps, load_now, dtype=np.float32))

        source_end = source_start + horizon_steps
        max_idx = len(env.solar_kw)

        if source_end <= max_idx:
            solar_forecast = env.solar_kw[source_start:source_end]
            load_forecast  = env.load_kw[source_start:source_end]
        else:
            avail = max_idx - source_start
            solar_forecast = np.concatenate([
                env.solar_kw[source_start:max_idx],
                np.full(horizon_steps - avail, env.solar_kw[max_idx - 1], dtype=np.float32),
            ])
            load_forecast = np.concatenate([
                env.load_kw[source_start:max_idx],
                np.full(horizon_steps - avail, env.load_kw[max_idx - 1], dtype=np.float32),
            ])

        return solar_forecast.astype(np.float32), load_forecast.astype(np.float32)

    def _forecast_outages(self, env: EnergyEnv, horizon_steps: int) -> np.ndarray:
        idx = env._data_idx
        outages = np.zeros(horizon_steps, dtype=np.float64)
        for h in range(horizon_steps):
            future_idx = idx + h
            if future_idx < len(env.timestamps):
                outages[h] = 1.0 if env.grid.is_outage(env.timestamps[future_idx]) else 0.0
            else:
                outages[h] = outages[h - 1] if h > 0 else 0.0
        return outages

    # ─────────────────────────────────────────────────────────────────────

    def select_action(self, env: EnergyEnv) -> int:
        battery = env.battery
        battery_spec = (
            battery.capacity_kwh, battery.eta_c, battery.eta_d,
            battery.min_soc, battery.max_soc,
            battery.max_charge_rate_kw, battery.max_discharge_rate_kw,
        )

        # (Re)build the problem only if this is the first call, or the
        # battery spec changed (e.g. controller reused across a sizing
        # sweep with different battery capacities). Otherwise reuse the
        # already-compiled problem — this is what makes repeated steps fast.
        if self._problem is None or battery_spec != self._battery_spec:
            self._export_rate = env.grid.net_metering_rate
            self._build_problem(battery)
            self._battery_spec = battery_spec

        solar_forecast, load_forecast = self._forecast(env, self.horizon_steps)
        outage_forecast = self._forecast_outages(env, self.horizon_steps)
        rate = env.grid.marginal_rate(env._cumulative_units_this_month)

        self._solar_param.value = solar_forecast
        self._load_param.value  = load_forecast
        self._outage_param.value = outage_forecast
        # Clip defensively: battery.soc should already be within
        # [min_soc, max_soc], but guard against tiny floating-point drift
        # that could otherwise trip the LP's strict boundary tolerance and
        # cause a spurious "infeasible" status on an otherwise fine problem.
        self._soc0_param.value  = float(np.clip(battery.soc, battery.min_soc, battery.max_soc))
        self._rate_param.value  = rate

        try:
            self._problem.solve(solver=self.solver, warm_start=True)
        except Exception as e:
            logger.warning(f"MPC solve failed ({e}) — falling back to idle action")
            return 0

        if self._problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(f"MPC solve status='{self._problem.status}' — falling back to idle action")
            return 0

        charge_0    = float(self._charge.value[0])
        discharge_0 = float(self._discharge.value[0])
        grid_in_0   = float(self._grid_in.value[0])

        if discharge_0 > self.action_threshold_kw:
            return 3   # discharge_to_load
        if charge_0 > self.action_threshold_kw:
            if grid_in_0 > self.action_threshold_kw:
                return 2   # charge_from_grid
            return 1       # charge_from_solar
        return 0            # idle