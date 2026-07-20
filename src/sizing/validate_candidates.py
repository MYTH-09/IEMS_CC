"""
IEMS – Sizing Candidate Validation (spot-check step)
=======================================================
Grid search (§9.1) and NSGA-II (§9.2) both rank candidates using
RULE-BASED control internally, purely because evaluating thousands of
candidates under DQN training or MPC's rolling-horizon solve is
computationally infeasible. That means their rankings are CONDITIONAL on
rule-based control being a reasonable proxy — not proven to hold under
smarter control.

This module closes that gap the cheap way: instead of re-running the
full search under DQN/MPC, take a SMALL, representative handful of
already-found candidates (e.g. top 3 off a Pareto front) and re-simulate
just those under rule-based, MPC, and DQN. If the ranking survives, the
sizing search's conclusion is validated. If it flips, that's a genuinely
important finding, not a failure.

Honest constraints, stated plainly rather than glossed over
----------------------------------------------------------------
1. MPC needs NO per-candidate training — its rolling-horizon LP is
   parameterized directly by battery capacity/rate at solve time (see
   MPCController._build_problem's battery_spec check), so it validates
   ANY candidate's hardware spec correctly, out of the box.

2. DQN is different: `models/dqn_policy.zip` was trained on ONE FIXED
   hardware spec (10kWh battery, 3kW charge/discharge — see
   scripts/07_train_dqn.py). A candidate with a very different battery
   size (e.g. 1.5kWh, or 15kWh) is hardware the policy never
   experienced during training. Running it anyway is not meaningless —
   the underlying "charge on solar surplus, discharge to cover deficit,
   pre-charge before outages" logic may partially transfer — but it is
   NOT a fair, apples-to-apples test of DQN's true potential on that
   specific hardware the way MPC's result is. This module runs DQN
   anyway (skipping it entirely would hide information), but explicitly
   FLAGS every candidate whose battery_kwh deviates significantly from
   the trained spec, so the caveat travels with the number instead of
   being silently forgotten.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.report.financials import full_financial_analysis


# Battery spec the shipped DQN policy (models/dqn_policy.zip) was trained
# on — see scripts/07_train_dqn.py's build_training_env(). Used only to
# compute how far a candidate's hardware deviates from that training spec,
# for the fairness flag described in the module docstring.
DQN_TRAINED_BATTERY_KWH = 10.0
DQN_TRAINED_MAX_RATE_KW = 3.0
DQN_MISMATCH_WARN_THRESHOLD = 0.35   # fractional deviation before flagging


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────────────

def select_representative_candidates(pareto_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Pick a small, genuinely representative spread off a Pareto front —
    not just the top N by one metric, which would just re-test near-
    identical points clustered at one end of the trade-off curve.

    Selection (for n=3, extends reasonably for other n):
        1. Best NPV overall — the front's single strongest financial pick.
        2. Best backup coverage (lowest annual_unmet_load_kwh) among
           candidates with NPV in the upper half of the front — avoids
           picking a technically-zero-unmet-load candidate that's
           financially uncompetitive.
        3. Most "balanced" — closest to the front's centroid in
           normalized (CAPEX, -NPV, payback, unmet_load) space, i.e. a
           genuine middle-ground option rather than an extreme.

    Each criterion walks its OWN full ranked list (not just the single
    best via idxmax/idxmin) to find the first candidate not already
    selected — this matters because it's entirely possible for two
    criteria to agree on the same top candidate (e.g. the best-NPV
    candidate also happens to have the lowest unmet load within the
    competitive subset). Stopping at "best, or nothing" in that case
    silently under-fills the selection; walking down to the next-best
    distinct candidate for that criterion is what actually guarantees
    `n` genuinely different candidates when the front has enough spread.
    """
    df = pareto_df.copy().reset_index(drop=True)
    picks: list[int] = []

    def add_first_available(ranked_indices) -> bool:
        for idx in ranked_indices:
            if idx not in picks:
                picks.append(idx)
                return True
        return False   # every candidate in this ranking is already picked

    reasons = []

    # 1. Best NPV
    npv_ranked = df["npv_25yr_inr"].sort_values(ascending=False).index
    if add_first_available(npv_ranked):
        reasons.append("best_npv")

    if n >= 2:
        npv_median = df["npv_25yr_inr"].median()
        competitive = df[df["npv_25yr_inr"] >= npv_median]
        backup_ranked = competitive["annual_unmet_load_kwh"].sort_values(ascending=True).index
        if add_first_available(backup_ranked):
            reasons.append("best_backup")

    if n >= 3:
        cols = ["total_capex_inr", "npv_25yr_inr", "payback_years", "annual_unmet_load_kwh"]
        norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min() + 1e-9)
        norm["npv_25yr_inr"] = 1 - norm["npv_25yr_inr"]
        centroid = norm.mean()
        dist_ranked = (((norm - centroid) ** 2).sum(axis=1) ** 0.5).sort_values().index
        if add_first_available(dist_ranked):
            reasons.append("most_balanced")

    # Backfill (rare): if n > 3 or the front is small enough that even the
    # above still falls short of min(n, len(df)), fill remaining slots by
    # NPV rank so the function always returns as many candidates as the
    # front can actually support.
    if len(picks) < min(n, len(df)):
        fallback_ranked = df["npv_25yr_inr"].sort_values(ascending=False).index
        while len(picks) < min(n, len(df)):
            if not add_first_available(fallback_ranked):
                break
            reasons.append("fallback_by_npv")

    if len(picks) < n:
        logger.warning(
            f"Requested n={n} but the Pareto front only supports {len(picks)} "
            f"distinct candidates (front has {len(df)} points total)"
        )

    selected = df.loc[picks].copy()
    selected["selection_reason"] = reasons[: len(selected)]

    logger.info(f"Selected {len(selected)} representative candidates from {len(df)}-point Pareto front")
    return selected.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-candidate simulation
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_battery_config(battery_kwh: float) -> dict:
    """
    Mirror grid_search.simulate_system_size's battery config logic exactly,
    so re-simulating a candidate here uses the SAME hardware assumptions
    it was originally scored under — only the CONTROLLER changes between
    the original sizing run and this validation.
    """
    return dict(
        capacity_kwh=max(battery_kwh, 0.01),
        max_charge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
        max_discharge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
        round_trip_efficiency=0.90,
        min_soc=0.10, max_soc=0.95, initial_soc=0.50,
    )


def dqn_fairness_flag(battery_kwh: float) -> Optional[str]:
    """Returns a warning string if this candidate's battery deviates
    significantly from what the shipped DQN policy was trained on."""
    kwh_dev = abs(battery_kwh - DQN_TRAINED_BATTERY_KWH) / DQN_TRAINED_BATTERY_KWH
    if kwh_dev > DQN_MISMATCH_WARN_THRESHOLD:
        return (
            f"battery_kwh={battery_kwh:.1f} deviates {kwh_dev:.0%} from the "
            f"{DQN_TRAINED_BATTERY_KWH}kWh spec DQN was trained on — "
            "result may not reflect DQN's true potential on this hardware"
        )
    return None


def simulate_candidate(
    load_kw: np.ndarray,
    solar_kw_per_kwp: np.ndarray,
    timestamps: pd.DatetimeIndex,
    panel_kw: float,
    battery_kwh: float,
    grid: GridModel,
    controller,
    episode_days: int,
    seed: int = 42,
) -> dict:
    """Run one candidate under one controller for `episode_days`, returning
    the same shape of result grid_search.simulate_system_size produces."""
    solar_kw = solar_kw_per_kwp * panel_kw
    battery_config = build_candidate_battery_config(battery_kwh)

    env = EnergyEnv(
        load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
        battery_config=battery_config, grid=grid,
        episode_days=episode_days, seed=seed,
    )

    obs, info = env.reset(seed=seed)
    controller.reset()
    terminated = truncated = False
    step_info = info

    while not (terminated or truncated):
        if battery_kwh <= 0.01:
            action = 0
        else:
            action = controller.select_action(env)
        obs, reward, terminated, truncated, step_info = env.step(action)

    return {
        "annual_grid_import_kwh": step_info["episode_grid_import_kwh"] * (365 / episode_days),
        "annual_grid_export_kwh": step_info["episode_grid_export_kwh"] * (365 / episode_days),
        "annual_unmet_load_kwh": step_info["episode_unmet_load_kwh"] * (365 / episode_days),
        "annual_cost_inr": step_info["episode_cost_inr"] * (365 / episode_days),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full validation run
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    candidates_df: pd.DataFrame,
    load_kw: np.ndarray,
    solar_curve_fn,
    timestamps: pd.DatetimeIndex,
    grid: GridModel,
    financial_config: dict,
    baseline_annual_cost_inr: float,
    dqn_model_path: Optional[str] = None,
    episode_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Re-simulate each candidate under rule-based, MPC, and (if a policy
    path is given) DQN. Recomputes CAPEX/NPV/payback per controller per
    candidate, so the original (rule-based) ranking can be directly
    compared against MPC- and DQN-driven rankings.

    Note on `episode_days`: shorter than the full 365-day window used
    for the original sizing search — this is a SPOT-CHECK, not a full
    re-run, and MPC's per-step LP solve makes a full year meaningfully
    slower per candidate. Results are annualized (scaled by 365/episode_days)
    for direct comparability with the original Pareto front numbers, at
    the cost of not capturing full seasonal variation — acceptable for
    confirming whether a ranking direction holds, not for a final number.
    """
    controllers = [("rule_based", RuleBasedController()), ("mpc", MPCController(horizon_hours=24.0))]

    if dqn_model_path:
        from src.controllers.dqn_controller import DQNController
        controllers.append(("dqn", DQNController(model_path=dqn_model_path)))
    else:
        logger.warning("No dqn_model_path given — skipping DQN in this validation run")

    # Solar curve is cached per (tilt, azimuth) since multiple candidates
    # may share close angles, avoiding redundant pvlib re-simulation.
    solar_cache: dict[tuple, np.ndarray] = {}

    def get_solar(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        key = (round(tilt_deg, 1), round(azimuth_deg, 1))
        if key not in solar_cache:
            solar_cache[key] = solar_curve_fn(*key)
        return solar_cache[key]

    rows = []
    for _, cand in candidates_df.iterrows():
        panel_kw, battery_kwh = cand["panel_kw"], cand["battery_kwh"]
        tilt_deg, azimuth_deg = cand["tilt_deg"], cand["azimuth_deg"]
        reason = cand.get("selection_reason", "")

        logger.info(
            f"\n── Candidate ({reason}): panel={panel_kw:.2f}kW, battery={battery_kwh:.2f}kWh, "
            f"tilt={tilt_deg:.1f}°, azimuth={azimuth_deg:.1f}° ──"
        )

        solar_per_kwp = get_solar(tilt_deg, azimuth_deg)
        flag = dqn_fairness_flag(battery_kwh)

        for ctrl_name, controller in controllers:
            if ctrl_name == "dqn" and flag:
                logger.warning(f"  [DQN fairness flag] {flag}")

            result = simulate_candidate(
                load_kw=load_kw, solar_kw_per_kwp=solar_per_kwp, timestamps=timestamps,
                panel_kw=panel_kw, battery_kwh=battery_kwh, grid=grid,
                controller=controller, episode_days=episode_days, seed=seed,
            )

            annual_savings_inr = baseline_annual_cost_inr - result["annual_cost_inr"]
            financials = full_financial_analysis(
                panel_kw=panel_kw, battery_kwh=battery_kwh,
                year1_savings_inr=max(annual_savings_inr, 0.0),
                config=financial_config,
            )

            logger.info(
                f"  {ctrl_name:>10s}: savings=₹{annual_savings_inr:>9,.0f}/yr  "
                f"NPV25=₹{financials.npv_25yr_inr:>10,.0f}  "
                f"payback={financials.payback_years:>5.1f}yr  "
                f"unmet={result['annual_unmet_load_kwh']:>7.1f}kWh"
            )

            rows.append({
                "candidate": reason,
                "panel_kw": panel_kw, "battery_kwh": battery_kwh,
                "tilt_deg": tilt_deg, "azimuth_deg": azimuth_deg,
                "controller": ctrl_name,
                "dqn_fairness_flag": flag if ctrl_name == "dqn" else None,
                "annual_savings_inr": round(annual_savings_inr, 2),
                "npv_25yr_inr": round(financials.npv_25yr_inr, 2),
                "payback_years": round(financials.payback_years, 2),
                "annual_unmet_load_kwh": round(result["annual_unmet_load_kwh"], 2),
            })

    return pd.DataFrame(rows)


def diagnose_ranking_disagreement(
    results_df: pd.DataFrame,
    margin_threshold_pct: float = 5.0,
) -> dict:
    """
    Determine whether controllers genuinely disagree on candidate ranking,
    or whether apparent rank flips are just near-ties within noise.

    Why a plain rank comparison isn't enough
    -------------------------------------------
    A candidate flipping from rank 2 to rank 3 by a ₹500 NPV margin out of
    ₹300,000 is not a meaningful disagreement — it's noise. A rank 1→3
    flip with a ₹100,000 gap is a real one. Treating every rank change as
    equally alarming (the original, simpler check) can't tell these apart.

    Method
    ------
    For every PAIR of candidates, compare their relative order (which one
    has higher NPV) under each controller. If the order is the same under
    every controller, that pair is not in dispute. If it differs, compute
    the NPV gap (%) between the two candidates under EACH controller
    involved — if the largest such gap is still below
    `margin_threshold_pct`, the flip is a "near-tie" (both controllers
    essentially see these two candidates as equivalent, so which one
    nominally ranks higher is not a meaningful signal). If the gap
    exceeds the threshold under any controller, it's a genuine
    disagreement.

    Returns
    -------
    dict with keys:
        overall_verdict : 'stable' | 'near_tie_only' | 'real_disagreement'
        pairwise_flips  : list of dicts describing every pair whose order
                           differs across controllers, with per-controller
                           gap percentages and a severity label
        pivot           : candidate × controller NPV table (for reuse)
    """
    pivot = results_df.pivot_table(index="candidate", columns="controller", values="npv_25yr_inr")
    controllers = list(pivot.columns)
    candidates = list(pivot.index)

    pairwise_flips = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            orders = {}
            gaps = {}
            for ctrl in controllers:
                v1, v2 = pivot.loc[c1, ctrl], pivot.loc[c2, ctrl]
                orders[ctrl] = v1 > v2   # True: c1 ranks better than c2 under this controller
                denom = max(abs(v1), abs(v2), 1e-6)
                gaps[ctrl] = abs(v1 - v2) / denom * 100

            if len(set(orders.values())) > 1:   # order genuinely differs across controllers
                max_gap = max(gaps.values())
                severity = "real_disagreement" if max_gap >= margin_threshold_pct else "near_tie"
                pairwise_flips.append({
                    "candidate_a": c1, "candidate_b": c2,
                    "orders": orders,
                    "gaps_pct": {k: round(v, 2) for k, v in gaps.items()},
                    "max_gap_pct": round(max_gap, 2),
                    "severity": severity,
                })

    if not pairwise_flips:
        overall_verdict = "stable"
    elif all(f["severity"] == "near_tie" for f in pairwise_flips):
        overall_verdict = "near_tie_only"
    else:
        overall_verdict = "real_disagreement"

    return {"overall_verdict": overall_verdict, "pairwise_flips": pairwise_flips, "pivot": pivot}


def resolve_recommendation(
    diagnosis: dict,
    tie_break_controller: str = "mpc",
) -> dict:
    """
    Produce ONE defensible bottom-line recommendation, regardless of
    whether controllers agree — while never hiding a real disagreement
    when one exists.

    Resolution policy
    ---------------------
    - 'stable' or 'near_tie_only': all controllers effectively agree (or
      differ only within noise) — the recommendation is unambiguous.
      HIGH / MEDIUM confidence respectively.
    - 'real_disagreement': controllers genuinely disagree on which
      candidate is best. A single recommendation is STILL produced (a
      report needs one headline number), using `tie_break_controller`
      as the deciding reference — MPC by default, since it solves an
      explicit constrained optimization rather than following a fixed
      heuristic (rule-based) or an approximate learned policy (DQN),
      making it the closest thing this project has to a rigorous
      ceiling estimate. Confidence is explicitly marked LOW, and the
      full disagreement detail travels with the result rather than
      being silently discarded — this is Option A (transparency) and
      Option B (MPC tie-break) combined, not one replacing the other.

    Returns
    -------
    dict with keys: recommended_candidate, recommended_by_controller,
    confidence, verdict, all_controller_rankings, disagreement_details.
    """
    pivot = diagnosis["pivot"]
    if tie_break_controller not in pivot.columns:
        raise ValueError(
            f"tie_break_controller='{tie_break_controller}' not present in results "
            f"(available: {list(pivot.columns)})"
        )

    ranked = pivot[tie_break_controller].sort_values(ascending=False)
    recommended_candidate = ranked.index[0]

    confidence_map = {"stable": "high", "near_tie_only": "medium", "real_disagreement": "low"}

    return {
        "recommended_candidate": recommended_candidate,
        "recommended_by_controller": tie_break_controller,
        "confidence": confidence_map[diagnosis["overall_verdict"]],
        "verdict": diagnosis["overall_verdict"],
        "all_controller_rankings": {
            ctrl: list(pivot[ctrl].sort_values(ascending=False).index) for ctrl in pivot.columns
        },
        "disagreement_details": diagnosis["pairwise_flips"] if diagnosis["overall_verdict"] != "stable" else [],
    }


def check_ranking_stability(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each controller, rank candidates by NPV, then compare rankings
    across controllers. Returns a wide DataFrame: rows=candidates,
    columns=controller ranks, so a flipped ranking is visible at a glance.

    Kept as a simple, quick-glance display table — use
    `diagnose_ranking_disagreement()` for the margin-aware verdict of
    whether a flip actually matters, and `resolve_recommendation()` for
    a single defensible bottom-line answer.
    """
    pivot = results_df.pivot_table(index="candidate", columns="controller", values="npv_25yr_inr")
    ranks = pivot.rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"{c}_rank" for c in ranks.columns]
    return pd.concat([pivot, ranks], axis=1)