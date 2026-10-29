"""
IEMS – GA Explainer API ("how is NSGA-II actually deciding")
================================================================
Standalone, ADDITIVE endpoints built specifically to answer one question
for the capstone defense: not just "does the GA sizing work" (already
shown by the ga tab in pipeline_demo.html) but "how does it decide, and
can we watch it happen." This file is intentionally separate from
pipeline_demo.html's 8-stage pipeline.

What this does NOT touch
-------------------------
This module does not modify nsga2_sizing.py, stage_functions.py,
orchestrator.py, grid_search.py, or app.py's existing /api/size endpoint.
It only IMPORTS `SizingProblem` (read-only reuse of the exact same
4-variable/4-objective problem definition the real GA sizing stage
uses) and runs its OWN independent NSGA-II job with its own in-memory
job store — zero shared state with orchestrator.py's `_JOBS`. If this
job crashes or runs slowly, nothing about the main pipeline demo or the
contractor sizing tool is affected; the only integration point is two
lines added to app.py (see bottom of this docstring).

Why a SEPARATE run instead of reusing the real GA stage's job
-----------------------------------------------------------------
run_nsga2() in nsga2_sizing.py only returns the FINAL Pareto front — it
was never built to expose intermediate generations, and adding a
callback parameter to it would mean touching a file the main pipeline
depends on. Instead, this module builds its own NSGA2 algorithm object
from the same SizingProblem class and passes pymoo's own `callback`
mechanism (pymoo.core.callback.Callback) into `minimize()` directly.
This was verified against pymoo's real API before writing this file:
after NSGA2's survival step, every individual in `algorithm.pop` already
carries a `rank` attribute (0 = non-dominated / Pareto-optimal within
that generation, 1 = dominated only by rank-0 individuals, etc) — the
callback simply reads X, F, and rank off the real population each
generation. Nothing here is estimated or recomputed after the fact.

Compute cost — stated honestly, not hidden
----------------------------------------------
Each generation evaluates `pop_size` candidates, each a real
`episode_days`-long rule-based simulation (see grid_search.simulate_system_size).
Defaults here (pop=24, gen=15, episode_days=60) trade full-year fidelity
for a run that finishes in a few minutes on a multi-core machine — the
same kind of documented simplification this project already makes
elsewhere (e.g. validate_candidates.py's shorter spot-check episodes).
This endpoint runs LIVE against real simulation code every time it's
called — nothing here is cached or pre-computed. Because it shares the
same server process as the rest of the app, a heavy run here will
compete for CPU with anything else running concurrently on the same
machine (e.g. a simultaneous pipeline_demo.html run) — it just cannot
corrupt or break that other work, since there is no shared state.

Baseline cost convention
-------------------------
Matches stage_functions.py's stage_ga exactly: the no-solar baseline is
computed at a full 365-day episode regardless of `episode_days`, while
each candidate is evaluated at the shorter `episode_days` — the same
already-existing simplification carried over from nsga2_sizing.py's own
`baseline_annual_cost_inr` convention, not something newly introduced
here.

Integration (add exactly these two lines to app.py, nothing else):

    from ga_explainer_routes import router as ga_explainer_router
    app.include_router(ga_explainer_router)

Endpoints
---------
POST /api/ga-explainer/start
    Body: {"monthly_bill_inr": 2400, "city": "Mathura", "pop_size": 24,
           "n_generations": 15, "episode_days": 60, "n_workers": null}
    Returns: {"job_id": "..."} immediately — does not block on the run.

GET /api/ga-explainer/status/{job_id}
    Returns: {"status": "running"|"done"|"failed", "current_generation": N,
              "total_generations": N, "error": "..."|null}
    Safe to poll every 1-2s — current_generation updates live as the
    callback fires, so the frontend can show real progress mid-run.

GET /api/ga-explainer/result/{job_id}
    Once status is "done": the full generation-by-generation history,
    the final non-dominated front, and a real crossover/mutation
    walkthrough on two real parents from that front. 409 if not ready.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import StarmapParallelization, LoopedElementwiseEvaluation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.grid.grid_model import GridModel
from src.sizing.grid_search import simulate_system_size
from src.sizing.nsga2_sizing import SizingProblem
from src.resource_limits import resolve_worker_count


router = APIRouter(prefix="/api/ga-explainer", tags=["ga-explainer"])


# ─────────────────────────────────────────────────────────────────────────────
# Household + solar loading — same real-data-with-labeled-synthetic-fallback
# pattern used in stage_functions.py / app.py, duplicated in miniature here
# ONLY so this module has zero import dependency on either of those files.
# ─────────────────────────────────────────────────────────────────────────────

class _FixedSolarCurve:
    """Picklable stand-in solar-curve callable (tilt/azimuth ignored) — same
    documented simplification as app.py's FixedSolarCurve / stage_functions.py's
    _FixedSolarCurve. Must stay module-level (not a closure) so multiprocessing
    can pickle it for worker dispatch."""

    def __init__(self, solar_kw_per_kwp: np.ndarray):
        self.solar_kw_per_kwp = solar_kw_per_kwp

    def __call__(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        return self.solar_kw_per_kwp


def _build_grid(city: str) -> GridModel:
    return GridModel(fallback_outage_hours_per_day=4.0, outage_peak_hours=[10, 14, 18, 22])


def _load_household_and_solar(city: str, monthly_bill_inr: float):
    """Returns (load_kw, solar_kw_per_kwp, timestamps, source_labels)."""
    solar_path = f"data/solar/{city}_solar_15min.parquet"
    if os.path.exists(solar_path):
        solar_df = pd.read_parquet(solar_path)
        timestamps = pd.DatetimeIndex(solar_df["timestamp"])
        solar_kw_per_kwp = (solar_df["solar_power_kw"] / 3.0).to_numpy(dtype=np.float32)
        solar_source = "real_nasa_pvlib"
    else:
        timestamps = pd.date_range("2024-01-01", periods=400 * 96, freq="15min")
        hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
        solar_kw_per_kwp = np.clip(1.0 * np.exp(-((hours - 12) ** 2) / 8), 0, None).astype(np.float32)
        solar_kw_per_kwp = np.where((hours < 6) | (hours > 18), 0.0, solar_kw_per_kwp).astype(np.float32)
        solar_source = "synthetic_fallback"

    daily_kwh_target = (monthly_bill_inr / 5.0) / 30.0
    try:
        from src.data.profile_generator import generate_profile, load_model
        profiles, kmeans = load_model("models")
        n_days = min(400, (timestamps.max() - timestamps.min()).days - 5)
        raw = generate_profile(profiles, kmeans, target_daily_kwh=daily_kwh_target,
                                season="summer", is_weekend=False, n_days=n_days)
        reps = int(np.ceil(len(timestamps) / len(raw)))
        load_kw = (np.tile(raw, reps)[: len(timestamps)] * 4.0).astype(np.float32)
        load_source = "real_module1_archetypes"
    except FileNotFoundError:
        hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy(dtype=float)
        shape = 0.3 + 0.4 * np.exp(-((hours - 8) ** 2) / 4) + 0.6 * np.exp(-((hours - 19) ** 2) / 4)
        load_kw = (shape * (daily_kwh_target / (shape.sum() / (len(shape) / 96)))).astype(np.float32)
        load_source = "synthetic_fallback"

    n = min(len(load_kw), len(solar_kw_per_kwp))
    return load_kw[:n], solar_kw_per_kwp[:n], timestamps[:n], {"load": load_source, "solar": solar_source}


# ─────────────────────────────────────────────────────────────────────────────
# The actual "show the decisions" mechanism: a pymoo Callback
# ─────────────────────────────────────────────────────────────────────────────

class GenerationRecorder(Callback):
    """Fires once per generation, AFTER NSGA2's survival step (non-dominated
    sort + crowding distance) has already assigned each individual a `rank`.
    Every number recorded here is read directly off the live algorithm
    state — nothing is recomputed or approximated for display purposes."""

    def __init__(self):
        super().__init__()
        self.history: list[dict] = []

    def notify(self, algorithm) -> None:
        pop = algorithm.pop
        X, F, rank = pop.get("X"), pop.get("F"), pop.get("rank")
        records = []
        for i in range(len(X)):
            panel_kw, battery_kwh, tilt_deg, azimuth_deg = X[i]
            capex, neg_npv, payback, unmet = F[i]
            records.append({
                "panel_kw": round(float(panel_kw), 2),
                "battery_kwh": round(float(battery_kwh), 2),
                "tilt_deg": round(float(tilt_deg), 1),
                "azimuth_deg": round(float(azimuth_deg), 1),
                "capex_inr": round(float(capex)),
                "npv_25yr_inr": round(float(-neg_npv)),
                "payback_years": round(float(payback), 2) if payback < 1e5 else None,
                "unmet_load_kwh": round(float(unmet), 1),
                "rank": int(rank[i]) if rank is not None else 0,
            })
        self.history.append({"generation": int(algorithm.n_gen), "population": records})


# ─────────────────────────────────────────────────────────────────────────────
# Job store — separate, in-memory, intentionally NOT shared with
# orchestrator.py's _JOBS (zero coupling with the main pipeline demo)
# ─────────────────────────────────────────────────────────────────────────────

_JOBS: dict[str, dict] = {}


class GAExplainerStartRequest(BaseModel):
    monthly_bill_inr: float = 2400.0
    city: str = "Mathura"
    pop_size: int = 24
    n_generations: int = 15
    episode_days: int = 60
    n_workers: Optional[int] = None


def _run_ga_explainer(job_id: str, req: GAExplainerStartRequest) -> None:
    job = _JOBS[job_id]
    try:
        load_kw, solar_per_kwp, timestamps, sources = _load_household_and_solar(
            req.city, req.monthly_bill_inr
        )
        grid = _build_grid(req.city)

        # Baseline at a full 365-day episode regardless of episode_days —
        # matches stage_functions.py's stage_ga convention exactly (see
        # module docstring: "Baseline cost convention").
        baseline = simulate_system_size(
            load_kw, np.zeros_like(load_kw), timestamps,
            panel_kw=0.0, battery_kwh=0.0, grid=grid, episode_days=365, seed=42,
        )
        baseline_cost = baseline["annual_cost_inr"]

        financial_config = dict(
            panel_cost_per_kw=45000.0, battery_cost_per_kwh=15000.0, balance_of_system_cost=10000.0,
            subsidy_rate_up_to_2kw=0.40, subsidy_rate_2_to_3kw=0.20, subsidy_max_kw=3.0,
            analysis_period_years=25, discount_rate=0.08,
            electricity_inflation_rate=0.05, panel_degradation_rate=0.005,
            maintenance_cost_annual_inr=2000.0,
        )

        n_workers = resolve_worker_count(req.n_workers)
        pool = None
        elementwise_runner = LoopedElementwiseEvaluation()
        if n_workers > 1:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            pool = ctx.Pool(n_workers)
            elementwise_runner = StarmapParallelization(pool.starmap)

        try:
            problem = SizingProblem(
                load_kw=load_kw, timestamps=timestamps, grid=grid,
                solar_curve_fn=_FixedSolarCurve(solar_per_kwp), financial_config=financial_config,
                episode_days=req.episode_days, seed=42,
                baseline_annual_cost_inr=baseline_cost,
                elementwise_runner=elementwise_runner,
            )
            algorithm = NSGA2(
                pop_size=req.pop_size, sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                eliminate_duplicates=True,
            )
            recorder = GenerationRecorder()
            job["recorder"] = recorder  # status endpoint reads this live for progress

            minimize(
                problem, algorithm, get_termination("n_gen", req.n_generations),
                seed=42, callback=recorder, verbose=True,
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if not recorder.history:
            raise RuntimeError("NSGA-II produced no generations — check pop_size/n_generations.")

        # ── Manual SBX crossover + Polynomial mutation walkthrough ─────────
        # Pulled from the final generation's rank-0 (non-dominated) front,
        # on two real parents — same as before. Previously this called
        # pymoo's SBX/PM operator classes via .do(), which only returns the
        # final child array: it does NOT expose the random draw `u`, the
        # resulting spread factor (beta for SBX / delta for PM), or which
        # formula branch fired for each gene — exactly the "show the actual
        # math" detail requested. Rather than monkey-patch pymoo's internals
        # to extract those (fragile, version-dependent), the standard SBX
        # (Deb & Agrawal, 1995) and Polynomial Mutation (Deb & Goyal, 1996)
        # equations pymoo's own operators are built on are computed here
        # explicitly, one gene at a time, with a real random draw for each
        # gene — same algorithm, just computed transparently instead of
        # through an opaque black box, so every intermediate number can be
        # shown to the user.
        last_gen = recorder.history[-1]["population"]
        rank0 = [r for r in last_gen if r["rank"] == 0]
        pool_for_demo = rank0 if len(rank0) >= 2 else last_gen
        parent_a, parent_b = pool_for_demo[0], pool_for_demo[min(1, len(pool_for_demo) - 1)]

        # (name, unit, lower bound, upper bound) — matches SizingProblem's
        # own variable bounds exactly (see nsga2_sizing.py's SizingProblem
        # defaults), so the mutation step's (xu - xl) range is the real
        # search-space range, not an arbitrary stand-in.
        DEMO_VARIABLES = [
            ("panel_kw", "kW", 1.0, 10.0),
            ("battery_kwh", "kWh", 0.0, 20.0),
            ("tilt_deg", "°", 0.0, 60.0),
            ("azimuth_deg", "°", 90.0, 270.0),
        ]
        SBX_ETA = 15
        PM_ETA = 20
        demo_rng = np.random.default_rng()

        def _sbx_one_gene(p1: float, p2: float, xl: float, xu: float, eta: float) -> dict:
            """Standard (unbounded) SBX equation for one gene, producing one
            child. Returns every intermediate value used to compute it."""
            u = float(demo_rng.random())
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                branch = "u ≤ 0.5"
                child = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                branch = "u > 0.5"
                child = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
            child = float(np.clip(child, xl, xu))
            return {"u": round(u, 4), "beta": round(beta, 4), "branch": branch, "value": child}

        def _pm_one_gene(x: float, xl: float, xu: float, eta: float) -> dict:
            """Standard polynomial mutation equation for one gene."""
            u = float(demo_rng.random())
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                branch = "u < 0.5"
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                branch = "u ≥ 0.5"
            child = x + delta * (xu - xl)
            child = float(np.clip(child, xl, xu))
            return {"u": round(u, 4), "delta": round(delta, 4), "branch": branch, "value": child}

        variables_demo = []
        for name, unit, xl, xu in DEMO_VARIABLES:
            p1, p2 = float(parent_a[name]), float(parent_b[name])
            cx = _sbx_one_gene(p1, p2, xl, xu, SBX_ETA)
            mut = _pm_one_gene(cx["value"], xl, xu, PM_ETA)
            variables_demo.append({
                "name": name, "unit": unit, "bounds": [xl, xu],
                "parent_a": round(p1, 2), "parent_b": round(p2, 2),
                "crossover": {**cx, "value": round(cx["value"], 2)},
                "mutation": {**mut, "value": round(mut["value"], 2)},
            })

        crossover_demo = {
            "parent_a": parent_a, "parent_b": parent_b,
            "variables": variables_demo,
            "sbx_eta": SBX_ETA, "pm_eta": PM_ETA,
        }

        job["result"] = {
            "generations": recorder.history,
            "final_front": rank0,
            "crossover_demo": crossover_demo,
            "baseline_annual_cost_inr": round(baseline_cost),
            "data_sources": sources,
            "meta": {
                "pop_size": req.pop_size, "n_generations": req.n_generations,
                "episode_days": req.episode_days, "city": req.city,
                "monthly_bill_inr": req.monthly_bill_inr,
            },
        }
        job["status"] = "done"

    except Exception as e:
        logger.exception(f"[ga-explainer {job_id}] failed")
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["finished_at"] = time.time()


@router.post("/start")
async def start_ga_explainer(req: GAExplainerStartRequest) -> dict:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "running", "result": None, "error": None, "recorder": None,
        "started_at": time.time(), "finished_at": None, "req": req.dict(),
    }
    # Fire-and-forget in a background thread — this is a long-running,
    # CPU-bound job (real simulations), so it must not block the event
    # loop the way an `await` in this coroutine would. Same asyncio.to_thread
    # pattern stage_functions.py already uses for its own heavy stages.
    asyncio.create_task(asyncio.to_thread(_run_ga_explainer, job_id, req))
    logger.info(f"GA explainer job started: {job_id} (pop={req.pop_size}, gen={req.n_generations})")
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def ga_explainer_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    recorder = job.get("recorder")
    current_gen = len(recorder.history) if recorder else 0
    return {
        "status": job["status"],
        "current_generation": current_gen,
        "total_generations": job["req"]["n_generations"],
        "error": job.get("error"),
    }


@router.get("/result/{job_id}")
async def ga_explainer_result(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not ready (status: {job['status']}).")
    return job["result"]