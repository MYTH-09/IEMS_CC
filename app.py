"""
IEMS – §10 Backend API
=========================
Exposes the real sizing pipeline (grid search + NSGA-II) as JSON endpoints
for the contractor UI (iems_contractor_ui.html) to call directly, instead
of that page's hardcoded demo data.

Why FastAPI + the existing custom HTML instead of Streamlit
-----------------------------------------------------------------
The build spec originally named Streamlit for §10. The contractor UI we
designed (custom fonts, exact layout, Chart.js) can't be reproduced in
Streamlit without fighting the framework the entire way. This backend
instead exposes the Python pipeline as plain JSON endpoints; the HTML
frontend stays exactly as designed and just fetches real numbers instead
of demo ones.

Requires (from the user's machine, not this sandbox)
----------------------------------------------------------
- `data/solar/{city}_solar_15min.parquet` — from scripts/02_solar_pipeline.py
- `models/load_profiles.parquet` + `models/profile_kmeans.pkl` — from
  scripts/01_data_pipeline.py
Both must already exist; this API does not fetch NASA data or retrain
anything live — it only reuses what the earlier pipeline scripts produced.

Run with:
    uvicorn app:app --reload --port 8000
Then open iems_contractor_ui.html directly in a browser (it fetches
http://localhost:8000/api/size).
"""

from __future__ import annotations

import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.grid.grid_model import GridModel
from src.data.profile_generator import load_model as load_profile_model, generate_profile
from src.solar.pv_model import load_solar_data, DEFAULT_SYSTEM
from src.sizing.grid_search import run_grid_search, top_recommendations, simulate_system_size
from src.sizing.validate_candidates import (
    run_validation, diagnose_ranking_disagreement, resolve_recommendation,
)
from src.report.financials import full_financial_analysis


def compute_baseline_annual_cost(
    load_kw: np.ndarray, timestamps: pd.DatetimeIndex, grid: GridModel, seed: int = 42,
) -> float:
    """
    The household's real annual grid cost with NO solar/battery at all —
    the reference point every "savings" figure is measured against.

    ⚠ Bug fixed here: an earlier version computed this by calling
    `run_grid_search(panel_sizes_kw=[0.0], battery_sizes_kwh=[0.0], ...)`.
    That's a self-referential trap — inside run_grid_search, the ONLY
    candidate being swept (panel=0, battery=0) is compared against
    run_grid_search's OWN internal baseline, which is that exact same
    zero-system. Comparing a value to itself always yields zero savings,
    so `baseline_cost` silently became 0 every time, corrupting every
    downstream "savings vs. no solar" number (confirmed in a real run —
    validation savings/NPV all went negative, e.g. -₹80,418/yr, because
    `annual_savings = 0 - actual_cost` for every real candidate).

    This function sidesteps the trap entirely by calling
    `simulate_system_size` directly for the zero-system and reading its
    `annual_cost_inr` — never routed through any "savings" calculation.
    """
    result = simulate_system_size(
        load_kw=load_kw,
        solar_kw_per_kwp=np.zeros_like(load_kw),
        timestamps=timestamps, panel_kw=0.0, battery_kwh=0.0,
        grid=grid, episode_days=365, seed=seed,
    )
    return result["annual_cost_inr"]


_PROCESSED_DATA_PATH = "data/processed/meters_15min.parquet"
_outage_params_cache: dict[str, dict] = {}   # module-level cache — this file is a few MB and city
                                              # outage rates don't change between requests, so we
                                              # only want to load/aggregate it once per process, not
                                              # once per HTTP request.


def derive_outage_params_from_real_data(city: str) -> dict:
    """
    Derive real, measured outage statistics for `city` from Module 1's
    processed smart-meter data, instead of using a generic guessed
    average (the previous behavior — GridModel was always constructed
    with a hardcoded `fallback_outage_hours_per_day=4.0` regardless of
    which city was selected, silently discarding the actual outage
    patterns Module 1 was built specifically to detect).

    Falls back to the generic UP-wide estimate (4.0 hrs/day, evening-
    weighted) ONLY if the processed data file isn't present — so the API
    still works on a fresh machine before scripts/01_data_pipeline.py has
    been run, but prefers real data whenever it's available.

    Returns
    -------
    dict with keys: fallback_outage_hours_per_day, outage_peak_hours,
    source ('measured' or 'generic_fallback').
    """
    if city in _outage_params_cache:
        return _outage_params_cache[city]

    generic = {
        "fallback_outage_hours_per_day": 4.0,
        "outage_peak_hours": [10, 14, 18, 22],
        "source": "generic_fallback",
    }

    if not os.path.exists(_PROCESSED_DATA_PATH):
        logger.warning(
            f"'{_PROCESSED_DATA_PATH}' not found — using generic outage estimate "
            f"for {city}, not real measured data. Run scripts/01_data_pipeline.py "
            "to enable real outage grounding."
        )
        _outage_params_cache[city] = generic
        return generic

    try:
        df = pd.read_parquet(_PROCESSED_DATA_PATH, columns=["city", "hour", "is_outage"])
        city_df = df[df["city"] == city]
        if len(city_df) == 0:
            logger.warning(f"No processed data rows found for city='{city}' — using generic estimate")
            _outage_params_cache[city] = generic
            return generic

        # Real average daily outage hours: fraction of 15-min slots that
        # are outages, scaled to a 24-hour day.
        outage_fraction = city_df["is_outage"].mean()
        real_hours_per_day = outage_fraction * 24.0

        # Real peak outage hours: the 4 hours of day with the highest
        # measured outage frequency for this city (replaces the generic
        # [10,14,18,22] guess with what actually happened).
        hourly_rate = city_df.groupby("hour")["is_outage"].mean().sort_values(ascending=False)
        real_peak_hours = hourly_rate.head(4).index.tolist()

        result = {
            "fallback_outage_hours_per_day": round(float(real_hours_per_day), 2),
            "outage_peak_hours": real_peak_hours,
            "source": "measured",
        }
        logger.info(
            f"Derived REAL outage params for {city} from measured data: "
            f"{result['fallback_outage_hours_per_day']} hrs/day, "
            f"peak hours={result['outage_peak_hours']}"
        )
        _outage_params_cache[city] = result
        return result

    except Exception as e:
        logger.warning(f"Failed to derive real outage params for {city} ({e}) — using generic estimate")
        _outage_params_cache[city] = generic
        return generic


class FixedSolarCurve:
    """
    Picklable stand-in solar-curve callable (tilt/azimuth ignored, always
    returns the pre-computed per-kWp curve for this city). Used for BOTH
    the "thorough" NSGA-II path and the validation step below.

    ⚠ Documented simplification, not full rigor: a true tilt/azimuth-aware
    curve would re-run pvlib per candidate angle (see
    scripts/10_nsga2_sizing.py's RealSolarCurve for the real version this
    stands in for). Implemented as a class — not a lambda/closure —
    specifically because closures cannot be pickled, and both
    run_grid_search's and run_nsga2's multiprocessing paths dispatch this
    callable to worker processes.
    """
    def __init__(self, solar_kw_per_kwp: np.ndarray):
        self.solar_kw_per_kwp = solar_kw_per_kwp

    def __call__(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        return self.solar_kw_per_kwp


app = FastAPI(title="IEMS Sizing API")

# Local-dev CORS: the HTML frontend is opened as a plain file (file://) or
# served separately, not from this same origin — allow it through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from pipeline_routes import router as pipeline_router
app.include_router(pipeline_router)

from ga_explainer_routes import router as ga_explainer_router
app.include_router(ga_explainer_router)

from src.solar.locations import get_city, list_cities, get_all_city_coords_dict
from src.solar.pv_model import load_solar_data, DEFAULT_SYSTEM, get_or_generate_solar_15min, is_city_solar_cached


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class SizingRequest(BaseModel):
    monthly_bill_inr: float
    city: str = "Mathura"               # Supports any major Indian city name or alias (e.g. Bengaluru, Chennai, Hyderabad, Mysuru, Mumbai, Delhi, etc.)
    budget_cap_inr: Optional[float] = None
    subsidy_eligible: bool = True
    depth: Literal["quick", "thorough"] = "quick"
    avg_rate_inr_per_kwh: float = 5.0   # rough rate to convert bill → units for the initial profile target
    n_workers: Optional[int] = None     # None = auto-detect all available CPU cores
    validate_candidates_flag: bool = True   # run the real rule-based/MPC/DQN spot-check (§8.4/validate_candidates.py)


CITY_COORDS = get_all_city_coords_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_household_and_solar(city: str, monthly_bill_inr: float, avg_rate: float):
    """
    Build a synthetic household load (Module 1) aligned to the SAME
    calendar dates as the city's cached solar output (Module 2a).
    Automatically fetches and generates 15-minute solar data for any
    supported Indian city if not already cached.
    """
    city_info = get_city(city)
    canonical_city = city_info.name if city_info is not None else city

    try:
        solar_df = get_or_generate_solar_15min(canonical_city)
    except Exception as exc:
        solar_path = f"data/solar/{canonical_city}_solar_15min.parquet"
        if os.path.exists(solar_path):
            solar_df = load_solar_data(solar_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Could not load or fetch solar data for city '{city}' ({exc}).",
            )

    solar_kw_per_kwp = (solar_df["solar_power_kw"] / DEFAULT_SYSTEM["capacity_kw"]).to_numpy(dtype=np.float32)
    timestamps = pd.DatetimeIndex(solar_df["timestamp"])

    daily_kwh_target = (monthly_bill_inr / avg_rate) / 30.0

    try:
        profiles, kmeans = load_profile_model("models")
        n_days = min(400, (timestamps.max() - timestamps.min()).days - 5)
        # generate_profile tiles a fixed-length season profile; here we need
        # it aligned 1:1 with `timestamps` length, so we tile/truncate to match.
        raw_profile = generate_profile(
            profiles, kmeans, target_daily_kwh=daily_kwh_target,
            season="summer", is_weekend=False, n_days=n_days,
        )
        reps = int(np.ceil(len(timestamps) / len(raw_profile)))
        load_kw = np.tile(raw_profile, reps)[: len(timestamps)] * 4.0   # kWh/slot -> kW
    except FileNotFoundError:
        logger.warning("No trained profile model found — falling back to a synthetic load shape")
        hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
        shape = 0.3 + 0.4 * np.exp(-((hours - 8) ** 2) / 4) + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
        load_kw = (shape * (daily_kwh_target / (shape.sum() / (len(shape) / 96)))).astype(np.float32)

    return load_kw.astype(np.float32), solar_kw_per_kwp, timestamps


def build_financial_config(subsidy_eligible: bool) -> dict:
    cfg = dict(
        panel_cost_per_kw=45000.0, battery_cost_per_kwh=15000.0, balance_of_system_cost=10000.0,
        subsidy_rate_up_to_2kw=0.40 if subsidy_eligible else 0.0,
        subsidy_rate_2_to_3kw=0.20 if subsidy_eligible else 0.0,
        subsidy_max_kw=3.0,
        analysis_period_years=25, discount_rate=0.08,
        electricity_inflation_rate=0.05, panel_degradation_rate=0.005,
        maintenance_cost_annual_inr=2000.0,
    )
    return cfg


def compute_total_outage_load_kwh(load_kw: np.ndarray, timestamps: pd.DatetimeIndex, grid: GridModel) -> float:
    """
    Total household load (kWh) that occurs during grid outage windows,
    independent of any particular system size — this is the denominator
    for a meaningful backup-coverage percentage. Computed once per
    household/city, reused across all three candidates.
    """
    dt_hours = 0.25
    outage_mask = np.array([grid.is_outage(ts) for ts in timestamps])
    return float((load_kw[outage_mask] * dt_hours).sum())


def row_to_candidate_json(row: pd.Series, financial_config: dict, total_outage_load_kwh: float) -> dict:
    """Shape the output to match exactly what iems_contractor_ui.html's
    `candidates` object expects — see the frontend's JS for the schema."""
    financials = full_financial_analysis(
        panel_kw=row["panel_kw"], battery_kwh=row["battery_kwh"],
        year1_savings_inr=max(row["annual_savings_inr"], 0.0),
        config=financial_config,
    )
    if total_outage_load_kwh > 1e-6:
        backup_pct = 100 * max(0.0, 1 - row["annual_unmet_load_kwh"] / total_outage_load_kwh)
    else:
        backup_pct = 100.0   # no outage exposure recorded for this household/city at all

    return {
        "panel": round(row["panel_kw"], 2),
        "battery": round(row["battery_kwh"], 2),
        "capex": round(financials.total_capex_inr),
        "subsidy": round(financials.subsidy_amount_inr),
        "net": round(financials.total_capex_inr - financials.subsidy_amount_inr),
        "npv": round(financials.npv_25yr_inr),
        "payback": round(financials.payback_years, 1),
        "irr": round(financials.irr_pct, 1) if financials.irr_pct == financials.irr_pct else 0,  # NaN check
        "monthly": round(row["annual_savings_inr"] / 12),
        "annual": round(row["annual_savings_inr"]),
        "lifetime": f"{financials.npv_25yr_inr/100000:.1f}L",
        "tiltDeg": round(row.get("tilt_deg", 27.0), 1),
        "azimuth": round(row.get("azimuth_deg", 180.0), 1),
        "backupPct": round(backup_pct, 1),
        "outageDesc": f"{row['annual_unmet_load_kwh']:.0f} kWh of load went unmet out of "
                       f"{total_outage_load_kwh:.0f} kWh exposed to outages this year.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/size")
def size_system(req: SizingRequest) -> dict:
    city_info = get_city(req.city)
    city_name = city_info.name if city_info is not None else req.city

    coords = CITY_COORDS.get(city_name, {"lat": 27.4924, "lon": 77.6737, "altitude_m": 182.0})
    outage_params = derive_outage_params_from_real_data(city_name)
    grid = GridModel(
        fallback_outage_hours_per_day=outage_params["fallback_outage_hours_per_day"],
        outage_peak_hours=outage_params["outage_peak_hours"],
    )
    financial_config = build_financial_config(req.subsidy_eligible)

    load_kw, solar_kw_per_kwp, timestamps = build_household_and_solar(
        city_name, req.monthly_bill_inr, req.avg_rate_inr_per_kwh,
    )
    solar_curve = FixedSolarCurve(solar_kw_per_kwp)

    workers = req.n_workers if req.n_workers is not None else (os.cpu_count() or 1)

    if req.depth == "quick":
        panel_sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        battery_sizes = [0.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        df = run_grid_search(
            load_kw=load_kw, solar_kw_per_kwp=solar_kw_per_kwp, timestamps=timestamps,
            grid=grid, panel_sizes_kw=panel_sizes, battery_sizes_kwh=battery_sizes,
            financial_config=financial_config, seed=42, n_workers=workers,
        )
        # The no-solar baseline determines "savings vs no solar" for every
        # candidate — computed directly, not via a self-referential
        # run_grid_search call (see compute_baseline_annual_cost docstring
        # for the bug that approach had).
        baseline_cost = compute_baseline_annual_cost(load_kw, timestamps, grid, seed=42)
    else:
        # "Thorough" (NSGA-II) is a 20min-2hr job at full rigor — not
        # something an HTTP request should block on. A production version
        # of this endpoint should kick this off as a background job
        # (Celery/RQ) and let the frontend poll for completion. For this
        # capstone's interactive demo, we run a deliberately small pop/gen
        # NSGA-II instead so the request still returns in a reasonable
        # time — this is a documented simplification, not the real
        # "thorough" depth's actual rigor (see scripts/10_nsga2_sizing.py
        # for the real pop=40/gen=50 version this stands in for).
        from src.sizing.nsga2_sizing import run_nsga2
        baseline_cost = compute_baseline_annual_cost(load_kw, timestamps, grid, seed=42)
        # pareto = run_nsga2(
        #     load_kw=load_kw, timestamps=timestamps, grid=grid,
        #     solar_curve_fn=solar_curve,   # picklable — safe for n_workers > 1, unlike the old lambda
        #     financial_config=financial_config, baseline_annual_cost_inr=baseline_cost,
        #     pop_size=12, n_generations=8, episode_days=180, seed=42, n_workers=workers,
        # )
        pareto = run_nsga2(
            load_kw=load_kw, timestamps=timestamps, grid=grid,
            solar_curve_fn=solar_curve,
            financial_config=financial_config, baseline_annual_cost_inr=baseline_cost,
            pop_size=40, n_generations=50, episode_days=365, seed=42, n_workers=workers,
        )
        # run_nsga2 now computes annual_savings_inr directly (re-simulates
        # each final front point once) — no placeholder/approximation needed.
        df = pareto

    if req.budget_cap_inr:
        df = df[df["total_capex_inr"] <= req.budget_cap_inr]
        if len(df) == 0:
            raise HTTPException(400, "No system fits within the given budget cap.")

    daily_kwh = req.monthly_bill_inr / req.avg_rate_inr_per_kwh / 30.0
    recs = top_recommendations(df, n=1, target_daily_kwh=daily_kwh)
    total_outage_load_kwh = compute_total_outage_load_kwh(load_kw, timestamps, grid)

    result = {
        "payback": row_to_candidate_json(recs["fastest_payback"].iloc[0], financial_config, total_outage_load_kwh),
        "npv":     row_to_candidate_json(recs["best_npv"].iloc[0], financial_config, total_outage_load_kwh),
        "backup":  row_to_candidate_json(recs["best_backup"].iloc[0], financial_config, total_outage_load_kwh),
        "search_meta": {
            "depth": req.depth,
            "n_configurations": len(df),
            "outage_data_source": outage_params["source"],
            "outage_hours_per_day": outage_params["fallback_outage_hours_per_day"],
        },
    }

    # ── Real validation: spot-check the 3 chosen candidates under ─────────
    # rule-based, MPC, and DQN (if a trained policy exists) — this is what
    # actually backs the report's "checked three ways" confidence claim.
    # Without this block, that claim would be static HTML text with no
    # computation behind it (a real gap caught and fixed here, not present
    # by design).
    if req.validate_candidates_flag:
        dqn_path = "models/checkpoints/best_model.zip"
        dqn_path = dqn_path if os.path.exists(dqn_path) else None

        candidates_df = pd.DataFrame([
            {"panel_kw": recs["fastest_payback"].iloc[0]["panel_kw"],
             "battery_kwh": recs["fastest_payback"].iloc[0]["battery_kwh"],
             "tilt_deg": recs["fastest_payback"].iloc[0].get("tilt_deg", 27.0),
             "azimuth_deg": recs["fastest_payback"].iloc[0].get("azimuth_deg", 180.0),
             "selection_reason": "payback"},
            {"panel_kw": recs["best_npv"].iloc[0]["panel_kw"],
             "battery_kwh": recs["best_npv"].iloc[0]["battery_kwh"],
             "tilt_deg": recs["best_npv"].iloc[0].get("tilt_deg", 27.0),
             "azimuth_deg": recs["best_npv"].iloc[0].get("azimuth_deg", 180.0),
             "selection_reason": "npv"},
            {"panel_kw": recs["best_backup"].iloc[0]["panel_kw"],
             "battery_kwh": recs["best_backup"].iloc[0]["battery_kwh"],
             "tilt_deg": recs["best_backup"].iloc[0].get("tilt_deg", 27.0),
             "azimuth_deg": recs["best_backup"].iloc[0].get("azimuth_deg", 180.0),
             "selection_reason": "backup"},
        ])

        try:
            validation_df = run_validation(
                candidates_df=candidates_df, load_kw=load_kw, solar_curve_fn=solar_curve,
                timestamps=timestamps, grid=grid, financial_config=financial_config,
                baseline_annual_cost_inr=baseline_cost,
                dqn_model_path=dqn_path, episode_days=60, seed=42,
            )
            diagnosis = diagnose_ranking_disagreement(validation_df, margin_threshold_pct=5.0)
            recommendation = resolve_recommendation(diagnosis, tie_break_controller="mpc")

            result["validation"] = {
                "verdict": diagnosis["overall_verdict"],
                "confidence": recommendation["confidence"],
                "recommended_candidate": recommendation["recommended_candidate"],
                "controllers_checked": ["rule_based", "mpc"] + (["dqn"] if dqn_path else []),
            }
        except Exception as e:
            logger.warning(f"Validation step failed, omitting from response: {e}")
            result["validation"] = {
                "verdict": "not_run", "confidence": "unknown",
                "recommended_candidate": None, "controllers_checked": [],
            }
    else:
        result["validation"] = {
            "verdict": "skipped", "confidence": "unknown",
            "recommended_candidate": None, "controllers_checked": [],
        }

    return result


@app.get("/")
def root():
    return {
        "message": "IEMS Sizing API",
        "health": "/api/health",
        "docs": "/docs",
        "size": "/api/size",
    }


@app.get("/api/cities")
def get_cities_list():
    """Return all supported Indian cities with their coordinates and cached status."""
    cities = list_cities()
    return {
        "count": len(cities),
        "cities": [
            {
                "name": c.name,
                "state": c.state,
                "display_name": c.display_name,
                "lat": c.lat,
                "lon": c.lon,
                "altitude_m": c.altitude_m,
                "optimal_tilt": c.optimal_tilt,
                "is_cached": is_city_solar_cached(c.name),
            }
            for c in cities
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon():
    return {"status": "no_favicon"}