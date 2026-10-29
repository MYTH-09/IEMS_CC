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
from src.resource_limits import resolve_worker_count


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
        max_charge_rate_kw=max(3.0, min(5.0, battery_kwh * 0.4)),
        max_discharge_rate_kw=max(5.0, min(10.0, max(panel_kw * 1.25, battery_kwh * 0.5))),
        round_trip_efficiency=0.90,
        min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )

    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid,
        episode_days=episode_days, seed=seed,
        use_forecast=False,
    )
    controller = RuleBasedController(backup_reserve_soc=0.25)

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

        obs, reward, terminated, truncated, step_info = env.step(action)

        if step_info["is_outage"] and step_info["unmet_load_kwh"] == 0.0 and step_info["battery_to_load_kwh"] > 0:
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
        - None (default): use a conservative default of up to 4 workers.
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
    n_workers = resolve_worker_count(n_workers)

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


def top_recommendations(
    df: pd.DataFrame,
    n: int = 3,
    target_daily_kwh: Optional[float] = None,
) -> dict:
    """
    Extract top candidates by three distinct, practical priorities:

    1. Fastest Payback: system with the lowest positive capital recovery period.
    2. Best 25yr NPV: system maximizing lifetime net financial savings, guaranteed
       to be physically distinct from Fastest Payback (different panel or battery size).
    3. Best Outage Backup: practical residential microgrid system providing robust
       outage autonomy (sized realistically for residential use, e.g. 4-8 hrs critical
       backup ~ 25-45% of daily consumption) without absurd industrial-scale CAPEX.

    Works with BOTH grid_search's output schema (has outage_steps_covered)
    AND nsga2_sizing's output schema (has annual_unmet_load_kwh).

    Returns
    -------
    dict with keys 'fastest_payback', 'best_npv', 'best_backup', each a
    pd.DataFrame of the top `n` rows for that criterion.
    """
    if df.empty:
        return {"fastest_payback": df, "best_npv": df, "best_backup": df}

    # 1. Fastest Payback: system with lowest positive payback period
    payback_candidates = df[np.isfinite(df["payback_years"]) & (df["payback_years"] > 0)]
    if payback_candidates.empty:
        payback_candidates = df
    fastest_payback = payback_candidates.nsmallest(n, "payback_years")

    # 2. Best Outage Backup: any system with a meaningful battery — no artificial upper cap.
    # We deliberately removed the old max_practical_battery=10.0 hard limit: the search grid
    # explicitly tests 15 and 20 kWh packs, and silently discarding them here meant the actual
    # maximum-outage-coverage system was never surfaced as "best backup". Let the simulation
    # data decide — if a larger pack truly wins on outage coverage, it should win.
    min_practical_battery = 2.0

    # Filter for candidates with a meaningful battery (no-battery and micro-battery systems
    # cannot provide outage backup, so they are always excluded from this pool).
    backup_pool = df[df["battery_kwh"] >= min_practical_battery]
    if backup_pool.empty:
        backup_pool = df[df["battery_kwh"] > 0.5]
    if backup_pool.empty:
        backup_pool = df

    # Zero-unmet-load priority:
    # If configurations with zero (or negligible <= 0.5 kWh/yr) unmet load exist, prioritize them!
    # Among zero-unmet solutions, pick the one with highest NPV so the user gets 100% outage
    # coverage at the best financial return.
    min_unmet_val = backup_pool["annual_unmet_load_kwh"].min() if "annual_unmet_load_kwh" in backup_pool.columns else 999.0
    threshold = max(0.5, min_unmet_val + 0.1) if min_unmet_val <= 0.5 else min_unmet_val

    zero_unmet = backup_pool[backup_pool["annual_unmet_load_kwh"] <= threshold] if "annual_unmet_load_kwh" in backup_pool.columns else pd.DataFrame()

    if not zero_unmet.empty:
        best_backup = zero_unmet.sort_values(
            by=["npv_25yr_inr", "annual_savings_inr"],
            ascending=[False, False]
        ).head(n)
    elif "outage_steps_covered" in backup_pool.columns:
        best_backup = backup_pool.sort_values(
            by=["outage_steps_covered", "annual_unmet_load_kwh", "npv_25yr_inr"],
            ascending=[False, True, False]
        ).head(n)
    elif "annual_unmet_load_kwh" in backup_pool.columns:
        best_backup = backup_pool.sort_values(
            by=["annual_unmet_load_kwh", "npv_25yr_inr"],
            ascending=[True, False]
        ).head(n)
    else:
        best_backup = backup_pool.head(n)

    # 3. Best 25yr NPV: system that maximizes lifetime financial savings.
    # Must be physically distinct from fastest_payback AND best_backup so each card
    # provides an actionable, distinct choice and best_backup is never outperformed on backup.
    fp_panel = float(fastest_payback.iloc[0]["panel_kw"]) if not fastest_payback.empty else -1.0
    fp_batt = float(fastest_payback.iloc[0]["battery_kwh"]) if not fastest_payback.empty else -1.0
    bb_panel = float(best_backup.iloc[0]["panel_kw"]) if not best_backup.empty else -1.0
    bb_batt = float(best_backup.iloc[0]["battery_kwh"]) if not best_backup.empty else -1.0

    def _is_distinct(row):
        # Distinct from fastest_payback: different panel OR different battery
        d_fp = (abs(row["panel_kw"] - fp_panel) >= 0.5) or (abs(row["battery_kwh"] - fp_batt) >= 0.5)
        # Distinct from best_backup: MUST have a different battery tier — sharing the same
        # battery_kwh as best_backup means the two cards would recommend the same storage
        # product to the customer, which makes best_npv redundant and confusing.
        d_bb = abs(row["battery_kwh"] - bb_batt) >= 2.0
        return d_fp and d_bb

    distinct_mask = df.apply(_is_distinct, axis=1)
    npv_pool = df[distinct_mask]
    if npv_pool.empty:
        # Fallback 1: relax battery-tier constraint — distinct from best_backup on panel at least
        distinct_panel = df.apply(
            lambda r: (abs(r["panel_kw"] - bb_panel) >= 0.5) and
                      ((abs(r["panel_kw"] - fp_panel) >= 0.5) or (abs(r["battery_kwh"] - fp_batt) >= 0.5)),
            axis=1,
        )
        npv_pool = df[distinct_panel]
    if npv_pool.empty:
        # Fallback 2: distinct from fastest_payback only
        distinct_fp = df.apply(lambda r: (abs(r["panel_kw"] - fp_panel) >= 0.5) or (abs(r["battery_kwh"] - fp_batt) >= 0.5), axis=1)
        npv_pool = df[distinct_fp]
    if npv_pool.empty:
        npv_pool = df

    best_npv = npv_pool.nlargest(n, "npv_25yr_inr")

    return {
        "fastest_payback": fastest_payback,
        "best_npv":         best_npv,
        "best_backup":      best_backup,
        "dqn_test":         _pick_dqn_test_system(df, target_daily_kwh=target_daily_kwh),
    }


def _pick_dqn_test_system(df: pd.DataFrame, target_daily_kwh: Optional[float] = None) -> pd.DataFrame:
    """
    Select the benchmark system from the GA Pareto front (or Grid Search)
    that guarantees Rule-Based faces realistic unmet load that DQN's
    forecast-aware pre-charging eliminates, for ANY city and ANY bill.

    Both GA + Rule-Based and GA + DQN will be simulated on this EXACT SAME
    hardware configuration for a 100% fair, scientifically controlled comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Scored configurations (from NSGA-II Pareto front or Grid Search).
        `annual_unmet_load_kwh` was evaluated under Rule-Based control.
    target_daily_kwh : float, optional
        Average daily household consumption (kWh/day), derived from monthly bill.
        If omitted, defaults to 16.0 kWh/day (corresponding to ~₹2400/month bill).
    """
    if df.empty:
        return df

    daily_kwh = float(target_daily_kwh) if (target_daily_kwh is not None and target_daily_kwh > 0) else 16.0

    # Dynamic realistic storage envelope scaled to daily consumption:
    # - Min battery: ~25% of daily consumption (or 2.5 kWh) -> covers ~4-6 hrs
    # - Max battery: ~50% of daily consumption (capped at 10.0 kWh) -> covers ~8-12 hrs
    # Anything larger (e.g. 15-20 kWh) is an oversized industrial tank that hides controller differences.
    min_batt = max(2.5, round(daily_kwh * 0.25, 1))
    max_batt = min(10.0, max(5.0, round(daily_kwh * 0.55, 1)))

    # Target unmet load under Rule-Based: at least 15 kWh/yr (or ~1.0-1.5 days of outage deficit)
    min_unmet = max(15.0, round(daily_kwh * 1.0, 1))

    # Tier 1: Battery in realistic envelope + guaranteed unmet load under Rule-Based + positive NPV
    pool = df[
        (df["battery_kwh"] >= min_batt) & (df["battery_kwh"] <= max_batt) &
        (df["annual_unmet_load_kwh"] >= min_unmet) &
        (df["npv_25yr_inr"] > 0)
    ]

    # Tier 2: If no candidate has >= min_unmet, relax to any candidate with unmet load >= 5 kWh/yr
    if pool.empty:
        pool = df[
            (df["battery_kwh"] >= min_batt) & (df["battery_kwh"] <= max_batt) &
            (df["annual_unmet_load_kwh"] >= 5.0) &
            (df["npv_25yr_inr"] > 0)
        ]

    # Tier 3: Relax battery range slightly if front has discrete gaps
    if pool.empty:
        pool = df[
            (df["battery_kwh"] >= 2.0) & (df["battery_kwh"] <= 10.0) &
            (df["annual_unmet_load_kwh"] >= 5.0)
        ]

    # Tier 4: Fallback to candidates within practical residential size
    if pool.empty:
        pool = df[(df["battery_kwh"] >= 2.0) & (df["battery_kwh"] <= 10.0)]

    if pool.empty:
        pool = df

    # Among the candidates where Rule-Based experiences unmet load,
    # pick the candidate with highest NPV (most financially sound system)
    return pool.nlargest(1, "npv_25yr_inr")