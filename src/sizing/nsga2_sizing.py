"""
IEMS – NSGA-II Multi-Objective Sizing (§9.2)
================================================
Advanced/stretch sizing mode. Where grid search (§9.1) picks from a fixed
discrete list of commercial sizes and ranks by one metric at a time,
NSGA-II searches CONTINUOUS panel/battery/tilt/azimuth combinations and
optimizes four objectives SIMULTANEOUSLY, returning a Pareto front — the
full set of non-dominated trade-offs — rather than one ranked answer.

Objectives (all converted to MINIMIZATION for pymoo, per its convention)
--------------------------------------------------------------------------
    1. CAPEX (₹)                    — minimize directly
    2. -NPV over 25 years (₹)       — minimize the negative (= maximize NPV)
    3. Payback period (years)       — minimize directly
    4. Unmet load (kWh/year)        — minimize directly (= maximize backup)

Why the fitness function still uses rule-based control
-----------------------------------------------------------
NSGA-II evaluates thousands of candidates (population × generations —
typically 1,000-20,000 individual simulations before convergence).
Running DQN training or even MPC's rolling-horizon solve that many times
is computationally infeasible. This is the SAME constraint that shaped
grid search (§9.1) — NSGA-II inherits it, just more acutely, since it
evaluates far more candidates than grid search's fixed 48. The Pareto
front this produces is therefore, like grid search's ranking, CONDITIONAL
on rule-based control — a lower-bound estimate, not a guarantee that
the ranking survives under smarter control. The same "spot-check the
top few finalists with DQN/MPC" validation step recommended for grid
search applies here too, scoped to a handful of points selected off the
front rather than the whole population.

Why tilt/azimuth need a swappable solar-curve function
------------------------------------------------------------
Unlike panel_kw and battery_kwh (which only rescale/resize an existing
solar curve), varying tilt and azimuth changes the SHAPE of solar
generation and requires re-running the pvlib simulation (§4.2) for each
candidate. This module takes a `solar_curve_fn(tilt_deg, azimuth_deg) ->
np.ndarray` callback rather than importing pv_model directly, so:
    - In real use, pass a function wrapping `simulate_pv()` against the
      already-cached NASA weather data (no network calls needed once
      Module 2a's pipeline has run once — see `scripts/02_solar_pipeline.py`).
    - In testing/demos without cached weather data, pass a lightweight
      synthetic stand-in (see `scripts/10_nsga2_sizing.py` for both).
Results are cached by rounding tilt/azimuth to reduce redundant re-solves
when nearby candidates repeat similar angles across generations.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from pymoo.core.problem import ElementwiseProblem, StarmapParallelization, LoopedElementwiseEvaluation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.grid.grid_model import GridModel
from src.sizing.grid_search import simulate_system_size
from src.report.financials import full_financial_analysis


# ─────────────────────────────────────────────────────────────────────────────
# Problem definition
# ─────────────────────────────────────────────────────────────────────────────

class SizingProblem(ElementwiseProblem):
    """
    pymoo problem wrapping the IEMS year-long simulation as a 4-variable,
    4-objective continuous optimization.

    Decision variables (all continuous)
    --------------------------------------
        x[0] = panel_kw      bounded by config sizing.panel_sizes_kw min/max
        x[1] = battery_kwh   bounded by config sizing.battery_sizes_kwh min/max
        x[2] = tilt_deg       bounded [0, 60]   (0=flat, 60=steep — realistic
                                                  rooftop range)
        x[3] = azimuth_deg    bounded [90, 270] (90=east, 180=south, 270=west)
    """

    def __init__(
        self,
        load_kw: np.ndarray,
        timestamps: pd.DatetimeIndex,
        grid: GridModel,
        solar_curve_fn: Callable[[float, float], np.ndarray],
        financial_config: dict,
        panel_kw_bounds: tuple[float, float] = (1.0, 10.0),
        battery_kwh_bounds: tuple[float, float] = (0.0, 20.0),
        tilt_bounds: tuple[float, float] = (0.0, 60.0),
        azimuth_bounds: tuple[float, float] = (90.0, 270.0),
        episode_days: int = 365,
        seed: int = 42,
        baseline_annual_cost_inr: float = 0.0,
        elementwise_runner=None,
    ):
        self.load_kw = load_kw
        self.timestamps = timestamps
        self.grid = grid
        self.solar_curve_fn = solar_curve_fn
        self.financial_config = financial_config
        self.episode_days = episode_days
        self.seed = seed
        self.baseline_annual_cost_inr = baseline_annual_cost_inr

        # Cache solar curves by rounded (tilt, azimuth) — nearby candidates
        # across generations often share similar angles, and re-running
        # pvlib for every single individual would be wasteful.
        #
        # ⚠ Parallelization note: when elementwise_runner distributes work
        # across worker PROCESSES (not threads), each worker gets its OWN
        # copy of this cache — it is NOT shared across processes. This means
        # some cache misses that a single-process run would have avoided
        # will recompute the same pvlib curve in different workers. This is
        # a real, accepted inefficiency, not a bug: correctness is unaffected
        # (each worker still computes the right curve), and the per-worker
        # cache still helps within that worker's share of the population
        # across generations. Full cross-process cache sharing would need a
        # multiprocessing.Manager dict, which adds IPC overhead that can
        # easily cost more than the redundant pvlib calls it saves for a
        # cache this cheap to populate.
        self._solar_cache: dict[tuple[float, float], np.ndarray] = {}

        xl = np.array([panel_kw_bounds[0], battery_kwh_bounds[0], tilt_bounds[0], azimuth_bounds[0]])
        xu = np.array([panel_kw_bounds[1], battery_kwh_bounds[1], tilt_bounds[1], azimuth_bounds[1]])

        super().__init__(
            n_var=4, n_obj=4, n_constr=0, xl=xl, xu=xu,
            elementwise_runner=elementwise_runner,
        )

    def _get_solar_curve(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        key = (round(tilt_deg, 1), round(azimuth_deg, 1))
        if key not in self._solar_cache:
            self._solar_cache[key] = self.solar_curve_fn(*key)
        return self._solar_cache[key]

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        panel_kw, battery_kwh, tilt_deg, azimuth_deg = x

        solar_kw_per_kwp = self._get_solar_curve(tilt_deg, azimuth_deg)

        result = simulate_system_size(
            load_kw=self.load_kw,
            solar_kw_per_kwp=solar_kw_per_kwp,
            timestamps=self.timestamps,
            panel_kw=panel_kw,
            battery_kwh=battery_kwh,
            grid=self.grid,
            episode_days=self.episode_days,
            seed=self.seed,
        )

        annual_savings_inr = self.baseline_annual_cost_inr - result["annual_cost_inr"]

        financials = full_financial_analysis(
            panel_kw=panel_kw, battery_kwh=battery_kwh,
            year1_savings_inr=max(annual_savings_inr, 0.0),
            config=self.financial_config,
        )

        # pymoo minimizes everything — negate NPV since we want to maximize it.
        # NPV/payback can be NaN/inf for hopeless candidates (e.g. huge CAPEX,
        # tiny savings) — replace with large finite penalties so NSGA-II can
        # still rank and steer away from them instead of crashing.
        npv = financials.npv_25yr_inr
        payback = financials.payback_years
        payback = payback if np.isfinite(payback) else 1e6

        out["F"] = [
            financials.total_capex_inr,
            -npv,
            payback,
            result["annual_unmet_load_kwh"],
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_nsga2(
    load_kw: np.ndarray,
    timestamps: pd.DatetimeIndex,
    grid: GridModel,
    solar_curve_fn: Callable[[float, float], np.ndarray],
    financial_config: dict,
    baseline_annual_cost_inr: float,
    pop_size: int = 40,
    n_generations: int = 50,
    episode_days: int = 365,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run NSGA-II and return the Pareto front as a DataFrame.

    Parameters
    ----------
    pop_size, n_generations : Population and generation count. Each
        generation evaluates `pop_size` candidates, so total simulations
        ≈ pop_size × n_generations. Default 40×50=2000 is a reasonable
        real-run size; reduce both for a fast smoke test (e.g. 8×3=24).
    n_workers : Optional[int]
        Number of worker PROCESSES to evaluate candidates in parallel.
        - None (default): auto-detect via os.cpu_count(), capped at 8
          (diminishing returns beyond that for a problem this size, and
          avoids accidentally oversubscribing a shared machine).
        - 1: disables parallelization entirely — runs serially in the
          main process. Useful for debugging (cleaner tracebacks) or on
          single-core environments.
        - N: use exactly N worker processes.

        Fitness evaluations here are fully independent (each candidate's
        year-long simulation doesn't touch any other candidate's state),
        so this gives close to linear speedup — e.g. an 8-core machine
        should turn a ~2-hour serial run into roughly 15-25 minutes,
        not a full 8x due to process startup and IPC overhead, but
        substantial. This does NOT change the search itself or the
        quality of the resulting Pareto front — it only changes how
        fast the identical computation completes.

    Returns
    -------
    pd.DataFrame — one row per Pareto-optimal (non-dominated) solution,
    columns: panel_kw, battery_kwh, tilt_deg, azimuth_deg, total_capex_inr,
    npv_25yr_inr, payback_years, annual_unmet_load_kwh.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    logger.info(
        f"Running NSGA-II: pop_size={pop_size}, generations={n_generations} "
        f"(~{pop_size * n_generations} total simulations), n_workers={n_workers}"
    )

    pool = None
    # ⚠ pymoo does NOT treat elementwise_runner=None as "use the default
    # serial evaluator" — it stores whatever is passed and calls it
    # directly, so None causes a hard "'NoneType' object is not callable"
    # crash at evaluation time. Must explicitly supply pymoo's own serial
    # runner class when parallelization is off, not just omit a runner.
    elementwise_runner = LoopedElementwiseEvaluation()

    try:
        if n_workers > 1:
            # 'fork' (Linux/macOS default) copies the parent process's memory
            # into each worker, so closures like solar_curve_fn and large
            # captured objects (weather DataFrames) don't need to be
            # re-pickled at pool creation — only the per-task arguments are
            # pickled across the pipe on each dispatch, which is cheap here
            # (a handful of floats per candidate).
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            pool = ctx.Pool(n_workers)
            elementwise_runner = StarmapParallelization(pool.starmap)
            logger.info(f"Parallel evaluation enabled: {n_workers} worker processes")
        else:
            logger.info("Parallel evaluation disabled (n_workers=1) — running serially")

        problem = SizingProblem(
            load_kw=load_kw, timestamps=timestamps, grid=grid,
            solar_curve_fn=solar_curve_fn, financial_config=financial_config,
            episode_days=episode_days, seed=seed,
            baseline_annual_cost_inr=baseline_annual_cost_inr,
            elementwise_runner=elementwise_runner,
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", n_generations)

        result = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    finally:
        # Always clean up the pool, even if minimize() raises — leaving
        # worker processes running after an error is a real resource leak,
        # not just untidy.
        if pool is not None:
            pool.close()
            pool.join()

    logger.success(f"NSGA-II complete: {len(result.X)} non-dominated solutions found")

    # The optimizer's own per-generation evaluations compute annual_savings_inr
    # internally (inside SizingProblem._evaluate, to build financials) but
    # never persist it to the output — only the 4 pymoo objectives survive
    # into result.F. Rather than have every downstream caller guess or
    # approximate this from NPV (e.g. app.py previously used a crude
    # `npv/15` placeholder), re-simulate just the small set of FINAL front
    # points once each — cheap, since the front is typically ≤pop_size
    # points, not the thousands evaluated during the search itself.
    logger.info(f"Computing annual_savings_inr for {len(result.X)} front points...")
    solar_cache: dict[tuple, np.ndarray] = {}
    rows = []
    for x, f in zip(result.X, result.F):
        panel_kw, battery_kwh, tilt_deg, azimuth_deg = x
        key = (round(tilt_deg, 1), round(azimuth_deg, 1))
        if key not in solar_cache:
            solar_cache[key] = solar_curve_fn(*key)
        sim_result = simulate_system_size(
            load_kw=load_kw, solar_kw_per_kwp=solar_cache[key], timestamps=timestamps,
            panel_kw=panel_kw, battery_kwh=battery_kwh, grid=grid,
            episode_days=episode_days, seed=seed,
        )
        annual_savings_inr = baseline_annual_cost_inr - sim_result["annual_cost_inr"]

        rows.append({
            "panel_kw": round(panel_kw, 2),
            "battery_kwh": round(battery_kwh, 2),
            "tilt_deg": round(tilt_deg, 1),
            "azimuth_deg": round(azimuth_deg, 1),
            "total_capex_inr": round(f[0], 2),
            "npv_25yr_inr": round(-f[1], 2),
            "payback_years": round(f[2], 2),
            "annual_unmet_load_kwh": round(f[3], 2),
            "annual_savings_inr": round(annual_savings_inr, 2),
        })

    df = pd.DataFrame(rows).sort_values("npv_25yr_inr", ascending=False).reset_index(drop=True)
    return df