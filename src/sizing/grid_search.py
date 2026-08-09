"""
IEMS – Grid Search Sizing (§9.1)
===================================
Brute-force sweep across commercially realistic panel + battery size
combinations. For each combination, simulates a full year using the
rule-based controller (deliberately NOT DQN/MPC — see the build spec's
§6 dependency resolution: sizing must stay fast and independent of RL
training), then scores each combination financially.

Why grid search here and not a smarter search algorithm
-----------------------------------------------------------
With 8 panel sizes × 6 battery sizes = 48 combinations, each a cheap
vectorized year-long simulation (seconds), brute force finds the exact
best answer in the discrete space in under ~2 minutes. Real hardware
only comes in fixed commercial sizes anyway, so a discrete sweep matches
what's actually purchasable — no rounding a continuous "optimum" down to
something a customer can buy. See §9.2 in the build spec for when a
genetic algorithm (NSGA-II) is actually justified instead (continuous
sizing + multiple simultaneous objectives).

Output
------
Does NOT collapse to one "optimal" system — returns the full scored
table plus three named recommendations (fastest payback / best 25yr NPV /
best outage backup), since "optimal" depends on what the customer values.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController
from src.report.financials import full_financial_analysis, FinancialSummary


# ─────────────────────────────────────────────────────────────────────────────
# Simulation for one candidate system size
# ─────────────────────────────────────────────────────────────────────────────

def simulate_system_size(
    load_kw: np.ndarray,
    solar_kw_per_kwp: np.ndarray,
    timestamps: pd.DatetimeIndex,
    panel_kw: float,
    battery_kwh: float,
    grid: GridModel,
    episode_days: int = 365,
    seed: int = 42,
) -> dict:
    """
    Run one full year of simulation for a single (panel_kw, battery_kwh)
    combination using the rule-based controller.

    Parameters
    ----------
    load_kw : np.ndarray
        Household load time series (kW), 15-min resolution, 1 year+ long.
    solar_kw_per_kwp : np.ndarray
        Solar generation PER INSTALLED kWp (kW/kWp) — i.e. the output of
        a 1 kWp reference system. Scaled by `panel_kw` inside this function.
    timestamps : pd.DatetimeIndex
        Aligned to both series.
    panel_kw, battery_kwh : Candidate system size being evaluated.
    grid : GridModel
        Shared grid model (tariffs + outage calendar) for this household.
    episode_days : int
        Simulation length — use 365 for a full annual sizing evaluation
        (longer than the standard 30-day RL training episode).

    Returns
    -------
    dict with keys: panel_kw, battery_kwh, annual_grid_import_kwh,
    annual_grid_export_kwh, annual_unmet_load_kwh, annual_cost_inr,
    outage_steps_covered. Savings vs. baseline is computed by the caller
    (`run_grid_search`), which has both the baseline and this result in
    scope simultaneously.
    """
    solar_kw = solar_kw_per_kwp * panel_kw

    battery_config = dict(
        capacity_kwh=max(battery_kwh, 0.01),   # avoid zero-capacity edge case
        max_charge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
        max_discharge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
        round_trip_efficiency=0.90,
        min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid,
        episode_days=episode_days, seed=seed,
    )
    controller = RuleBasedController()

    obs, info = env.reset(seed=seed)
    controller.reset()

    outage_steps_covered = 0
    terminated = truncated = False
    step_info = info   # fallback in case the loop body never executes

    while not (terminated or truncated):
        if battery_kwh <= 0.01:
            # No battery installed — never attempt battery actions
            action = 0
        else:
            action = controller.select_action(env)

        idx_before = env._data_idx
        was_outage = env.grid.is_outage(env.timestamps[idx_before])

        obs, reward, terminated, truncated, step_info = env.step(action)

        if was_outage and step_info["unmet_load_kwh"] == 0.0 and step_info["battery_to_load_kwh"] > 0:
            outage_steps_covered += 1

    return {
        "panel_kw": panel_kw,
        "battery_kwh": battery_kwh,
        "annual_grid_import_kwh": step_info["episode_grid_import_kwh"],
        "annual_grid_export_kwh": step_info["episode_grid_export_kwh"],
        "annual_unmet_load_kwh": step_info["episode_unmet_load_kwh"],
        "annual_cost_inr": step_info["episode_cost_inr"],
        "outage_steps_covered": outage_steps_covered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

def _score_one_combo(
    panel_kw: float,
    battery_kwh: float,
    load_kw: np.ndarray,
    solar_kw_per_kwp: np.ndarray,
    timestamps: pd.DatetimeIndex,
    grid: GridModel,
    baseline_annual_cost_inr: float,
    financial_config: dict,
    seed: int,
) -> dict:
    """
    Score exactly one (panel_kw, battery_kwh) combination end-to-end —
    simulation + financials. Kept as a plain MODULE-LEVEL function (not a
    closure/nested function) specifically so it can be pickled and
    dispatched to worker processes by ProcessPoolExecutor — the same
    constraint that shaped nsga2_sizing.py's picklable-class requirement
    for solar curve functions. All arguments here are plain data (numpy
    arrays, a DatetimeIndex, a GridModel, dicts, floats), which pickle
    fine on their own; only the *function itself* needed to not be a
    closure.
    """
    result = simulate_system_size(
        load_kw, solar_kw_per_kwp, timestamps,
        panel_kw=panel_kw, battery_kwh=battery_kwh, grid=grid,
        episode_days=365, seed=seed,
    )

    annual_savings_inr = baseline_annual_cost_inr - result["annual_cost_inr"]

    financials = full_financial_analysis(
        panel_kw=panel_kw, battery_kwh=battery_kwh,
        year1_savings_inr=max(annual_savings_inr, 0.0),
        config=financial_config,
    )

    return {
        "panel_kw": panel_kw,
        "battery_kwh": battery_kwh,
        "annual_grid_import_kwh": result["annual_grid_import_kwh"],
        "annual_grid_export_kwh": result["annual_grid_export_kwh"],
        "annual_unmet_load_kwh": result["annual_unmet_load_kwh"],
        "outage_steps_covered": result["outage_steps_covered"],
        "annual_savings_inr": round(annual_savings_inr, 2),
        "total_capex_inr": financials.total_capex_inr,
        "subsidy_amount_inr": financials.subsidy_amount_inr,
        "payback_years": financials.payback_years,
        "npv_25yr_inr": financials.npv_25yr_inr,
        "irr_pct": financials.irr_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    load_kw: np.ndarray,
    solar_kw_per_kwp: np.ndarray,
    timestamps: pd.DatetimeIndex,
    grid: GridModel,
    panel_sizes_kw: list[float],
    battery_sizes_kwh: list[float],
    financial_config: dict,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sweep all (panel_kw, battery_kwh) combinations and return a scored
    DataFrame, one row per combination, including full financial metrics.

    Parameters
    ----------
    load_kw, solar_kw_per_kwp, timestamps : See `simulate_system_size`.
    grid : GridModel
    panel_sizes_kw, battery_sizes_kwh : Search grid (from config `sizing:` section).
    financial_config : dict
        The `sizing:` section of `default_config.yaml` (cost/subsidy/
        discount-rate parameters, forwarded to `full_financial_analysis`).
    n_workers : Optional[int]
        Number of worker PROCESSES to score combinations in parallel.
        - None (default): auto-detect via os.cpu_count(), capped at 8.
        - 1: disables parallelization — runs serially (easier debugging).
        - N: use exactly N worker processes.
        Each combination's year-long simulation is fully independent of
        every other's, so this gives close to linear speedup — the same
        pattern already used for NSGA-II (src/sizing/nsga2_sizing.py).

    Returns
    -------
    pd.DataFrame sorted by NPV descending, with columns:
        panel_kw, battery_kwh, annual_grid_import_kwh, annual_grid_export_kwh,
        annual_unmet_load_kwh, outage_steps_covered, annual_savings_inr,
        total_capex_inr, payback_years, npv_25yr_inr, irr_pct
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    combos = [(p, b) for p in panel_sizes_kw for b in battery_sizes_kwh]
    total = len(combos)
    logger.info(
        f"Running grid search: {len(panel_sizes_kw)} panel sizes × "
        f"{len(battery_sizes_kwh)} battery sizes = {total} combinations, "
        f"n_workers={n_workers}"
    )

    # ── Step 1: compute the no-solar, no-battery baseline once ────────────
    logger.info("  Computing no-solar baseline (panel=0, battery=0)...")
    baseline_result = simulate_system_size(
        load_kw, solar_kw_per_kwp, timestamps,
        panel_kw=0.0, battery_kwh=0.0, grid=grid,
        episode_days=365, seed=seed,
    )
    baseline_annual_import_kwh = baseline_result["annual_grid_import_kwh"]
    baseline_annual_cost_inr   = baseline_result["annual_cost_inr"]
    logger.info(
        f"  Baseline: {baseline_annual_import_kwh:.0f} kWh/year import, "
        f"₹{baseline_annual_cost_inr:,.0f}/year cost"
    )

    # ── Step 2: sweep all combinations ──────────────────────────────────────
    rows = []
    count = 0

    if n_workers > 1:
        # ProcessPoolExecutor's default start method pickles the target
        # function + its arguments for dispatch to each worker — this is
        # why _score_one_combo is a module-level function, not a closure
        # (closures can't be pickled; see nsga2_sizing.py for the same
        # lesson learned the hard way there).
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _score_one_combo, panel_kw, battery_kwh,
                    load_kw, solar_kw_per_kwp, timestamps, grid,
                    baseline_annual_cost_inr, financial_config, seed,
                ): (panel_kw, battery_kwh)
                for panel_kw, battery_kwh in combos
            }
            for future in as_completed(futures):
                panel_kw, battery_kwh = futures[future]
                row = future.result()
                rows.append(row)
                count += 1
                logger.debug(
                    f"  [{count}/{total}] {panel_kw}kW + {battery_kwh}kWh → "
                    f"savings=₹{row['annual_savings_inr']:,.0f}/yr, "
                    f"payback={row['payback_years']:.1f}yr, "
                    f"NPV25=₹{row['npv_25yr_inr']:,.0f}"
                )
    else:
        for panel_kw, battery_kwh in combos:
            count += 1
            row = _score_one_combo(
                panel_kw, battery_kwh, load_kw, solar_kw_per_kwp, timestamps,
                grid, baseline_annual_cost_inr, financial_config, seed,
            )
            rows.append(row)
            logger.debug(
                f"  [{count}/{total}] {panel_kw}kW + {battery_kwh}kWh → "
                f"savings=₹{row['annual_savings_inr']:,.0f}/yr, "
                f"payback={row['payback_years']:.1f}yr, "
                f"NPV25=₹{row['npv_25yr_inr']:,.0f}"
            )

    df = pd.DataFrame(rows).sort_values("npv_25yr_inr", ascending=False).reset_index(drop=True)
    logger.success(f"Grid search complete: {len(df)} combinations scored")
    return df


def top_recommendations(df: pd.DataFrame, n: int = 3) -> dict:
    """
    Extract top candidates by three different priorities, since there's
    no single "optimal" answer — a customer might prioritize any of these.

    Works with BOTH grid_search's output schema (has outage_steps_covered)
    AND nsga2_sizing's output schema (has annual_unmet_load_kwh only) —
    the NSGA-II Pareto front never had an outage_steps_covered column,
    so calling this with "thorough" depth's DataFrame previously raised
    KeyError: 'outage_steps_covered'.

    Returns
    -------
    dict with keys 'fastest_payback', 'best_npv', 'best_backup', each a
    pd.DataFrame of the top `n` rows for that criterion.
    """
    # Exclude systems that never pay back (inf) from the payback ranking
    payback_candidates = df[np.isfinite(df["payback_years"])]

    if "annual_unmet_load_kwh" in df.columns:
        # Lower unmet load = better backup coverage. Present in both
        # grid_search's and nsga2_sizing's output — the reliable choice.
        best_backup = df.nsmallest(n, "annual_unmet_load_kwh")
    elif "outage_steps_covered" in df.columns:
        # Fallback for any caller whose DataFrame only has this older
        # grid_search-specific column (higher = better here).
        best_backup = df.nlargest(n, "outage_steps_covered")
    else:
        raise KeyError(
            "top_recommendations(): DataFrame has neither "
            "'annual_unmet_load_kwh' nor 'outage_steps_covered' — "
            "cannot determine a backup-coverage ranking."
        )

    return {
        "fastest_payback": payback_candidates.nsmallest(n, "payback_years"),
        "best_npv":         df.nlargest(n, "npv_25yr_inr"),
        "best_backup":      best_backup,
    }