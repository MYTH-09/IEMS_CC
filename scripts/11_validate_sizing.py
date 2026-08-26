"""
IEMS – Script 11: Validate Top Sizing Candidates Under DQN/MPC
===================================================================
Takes the Pareto front from scripts/10_nsga2_sizing.py (or a grid search
CSV with the same columns), selects a small representative spread of
candidates, and re-simulates each under rule-based, MPC, and DQN — to
check whether the original rule-based-driven ranking survives smarter
control, or flips.

Usage:
    python scripts/11_validate_sizing.py --city Mathura
    python scripts/11_validate_sizing.py --city Mathura --pareto-csv outputs/nsga2_pareto_front.csv --n 3
    python scripts/11_validate_sizing.py --city Mathura --skip-dqn   # MPC-only, no DQN fairness caveat needed
"""

import os
import sys
import argparse

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid.grid_model import GridModel
from src.sizing.grid_search import simulate_system_size
from src.sizing.validate_candidates import (
    select_representative_candidates, run_validation, check_ranking_stability,
)

# Reuse the exact fixed classes from script 10 (avoids duplicating the
# picklable-solar-curve logic and the calendar-alignment fix).
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location("nsga2_script", os.path.join(os.path.dirname(__file__), "10_nsga2_sizing.py"))
_nsga2_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nsga2_script)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Mathura", help="City name (e.g. Bengaluru, Chennai, Hyderabad, Mysuru, Mathura, Bareilly)")
    parser.add_argument("--pareto-csv", default="outputs/nsga2_pareto_front.csv")
    parser.add_argument("--n", type=int, default=3, help="Number of representative candidates to validate")
    parser.add_argument("--episode-days", type=int, default=180, help="Spot-check window (shorter than the full 365-day sizing search — see module docstring)")
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--dqn-model-path", default="models/dqn_policy.zip")
    args = parser.parse_args()

    logger.info("━" * 60)
    logger.info("  IEMS Sizing Candidate Validation  —  START")
    logger.info("━" * 60)

    if not os.path.exists(args.pareto_csv):
        logger.error(f"'{args.pareto_csv}' not found. Run scripts/10_nsga2_sizing.py first.")
        sys.exit(1)

    pareto_df = pd.read_csv(args.pareto_csv)
    candidates = select_representative_candidates(pareto_df, n=args.n)
    logger.info(f"\n{candidates.to_string(index=False)}")

    from src.solar.locations import get_city
    from src.solar.pv_model import get_or_generate_solar_15min

    city_info = get_city(args.city)
    if city_info is None:
        logger.error(f"Unknown city '{args.city}'. Please choose a valid Indian city.")
        sys.exit(1)

    city_name = city_info.name
    coords = dict(lat=city_info.lat, lon=city_info.lon, altitude_m=city_info.altitude_m)

    # Ensure weather data is ready
    get_or_generate_solar_15min(city_name)

    from src.solar.nasa_power import load_cached
    weather_hourly = load_cached(city_name, cache_dir="data/solar")
    weather_start = weather_hourly["timestamp"].min()
    weather_span_days = (weather_hourly["timestamp"].max() - weather_start).days

    n_days = min(400, weather_span_days - 5)
    load_kw, timestamps = _nsga2_script.build_test_household(n_days=n_days, start_date=str(weather_start.date()))
    solar_curve_fn = _nsga2_script.RealSolarCurve(city_name, timestamps_target=timestamps, **coords)

    grid = GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])

    financial_config = dict(
        panel_cost_per_kw=45000.0, battery_cost_per_kwh=15000.0, balance_of_system_cost=10000.0,
        subsidy_rate_up_to_2kw=0.40, subsidy_rate_2_to_3kw=0.20, subsidy_max_kw=3.0,
        analysis_period_years=25, discount_rate=0.08,
        electricity_inflation_rate=0.05, panel_degradation_rate=0.005,
        maintenance_cost_annual_inr=2000.0,
    )

    logger.info("\nComputing no-solar baseline...")
    import numpy as np
    zero_solar = np.zeros_like(load_kw)
    baseline = simulate_system_size(
        load_kw=load_kw, solar_kw_per_kwp=zero_solar, timestamps=timestamps,
        panel_kw=0.0, battery_kwh=0.0, grid=grid, episode_days=args.episode_days, seed=42,
    )
    baseline_annual_cost_inr = baseline["annual_cost_inr"] * (365 / args.episode_days)
    logger.info(f"Baseline annual cost: ₹{baseline_annual_cost_inr:,.0f}")

    dqn_path = None if args.skip_dqn else args.dqn_model_path
    if dqn_path and not os.path.exists(dqn_path):
        logger.warning(f"'{dqn_path}' not found — proceeding without DQN")
        dqn_path = None

    results_df = run_validation(
        candidates_df=candidates, load_kw=load_kw, solar_curve_fn=solar_curve_fn,
        timestamps=timestamps, grid=grid, financial_config=financial_config,
        baseline_annual_cost_inr=baseline_annual_cost_inr,
        dqn_model_path=dqn_path, episode_days=args.episode_days, seed=42,
    )

    logger.info("\n" + "─" * 100)
    logger.info("  FULL VALIDATION RESULTS")
    logger.info("─" * 100)
    logger.info(f"\n{results_df.to_string(index=False)}")

    stability = check_ranking_stability(results_df)
    logger.info("\n" + "─" * 60)
    logger.info("  RANKING COMPARISON (rank 1 = best NPV under that controller)")
    logger.info("─" * 60)
    logger.info(f"\n{stability.to_string()}")

    from src.sizing.validate_candidates import diagnose_ranking_disagreement, resolve_recommendation

    diagnosis = diagnose_ranking_disagreement(results_df, margin_threshold_pct=5.0)
    recommendation = resolve_recommendation(diagnosis, tie_break_controller="mpc")

    logger.info("\n" + "─" * 60)
    logger.info("  DISAGREEMENT DIAGNOSIS")
    logger.info("─" * 60)
    if diagnosis["overall_verdict"] == "stable":
        logger.success("Verdict: STABLE — every controller agrees on the exact same ranking.")
    elif diagnosis["overall_verdict"] == "near_tie_only":
        logger.info("Verdict: NEAR-TIE ONLY — rank order differs slightly between some "
                     "controllers, but only among candidates whose NPV gap is small "
                     "(<5%) — not a meaningful disagreement.")
        for f in diagnosis["pairwise_flips"]:
            logger.info(f"  {f['candidate_a']} vs {f['candidate_b']}: "
                         f"max gap {f['max_gap_pct']:.1f}% (near-tie)")
    else:
        logger.warning("Verdict: REAL DISAGREEMENT — controllers meaningfully disagree "
                        "on candidate ranking. This is a genuine finding, not a pipeline error.")
        for f in diagnosis["pairwise_flips"]:
            logger.warning(f"  {f['candidate_a']} vs {f['candidate_b']}: "
                            f"max gap {f['max_gap_pct']:.1f}% ({f['severity']}) — "
                            f"order under each controller: {f['orders']}")

    logger.info("\n" + "─" * 60)
    logger.info("  FINAL RECOMMENDATION")
    logger.info("─" * 60)
    logger.info(f"  Recommended candidate : {recommendation['recommended_candidate']}")
    logger.info(f"  Decided by            : {recommendation['recommended_by_controller']} "
                f"(tie-break reference — see module docstring for why MPC is the default)")
    logger.info(f"  Confidence            : {recommendation['confidence'].upper()}")
    if recommendation["confidence"] == "low":
        logger.warning(
            "  ⚠ LOW confidence means controllers genuinely disagreed on the best "
            "candidate. This recommendation reflects MPC's ranking specifically — "
            "review 'all_controller_rankings' in the saved JSON before presenting "
            "this to a customer as a single confident answer."
        )

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/candidate_validation.csv", index=False)

    import json
    with open("outputs/candidate_validation_recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2, default=str)

    logger.success("\nResults saved → outputs/candidate_validation.csv")
    logger.success("Recommendation saved → outputs/candidate_validation_recommendation.json")

    logger.success("\n" + "━" * 60)
    logger.success("  VALIDATION COMPLETE")
    logger.success("━" * 60)


if __name__ == "__main__":
    main()