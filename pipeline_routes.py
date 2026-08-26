"""
IEMS – Pipeline Demo API Routes
===================================
Exposes src/pipeline/orchestrator.py as HTTP endpoints for the tabbed
demo frontend (7 tabs shown in the UI: synthetic_load, grid_search, ga,
rule_based_gs, rule_based_ga, dqn, outage_proof — the last displayed as
"Final Report". solar_generation still runs as a backend stage since
grid_search/ga depend on it, but isn't shown as its own tab. mpc_normal
and mpc_weather stages are commented out in stage_functions.py and are
not run at all — see the notes there and in orchestrator.py's
STAGE_DEPENDENCIES for how to re-enable them).

Designed to be ADDITIVE to the existing app.py — does not replace or
modify the existing /api/size endpoint or anything else already there.

Integration (add these two lines to app.py, nothing else):

    from pipeline_routes import router as pipeline_router
    app.include_router(pipeline_router)

Endpoints
---------
POST /api/pipeline/start
    Body: {"monthly_bill_inr": 2400, "city": "Mathura", "subsidy_eligible": true,
           "n_workers": null, "ga_pop_size": 12, "ga_n_gen": 8}
    Returns: {"job_id": "..."}
    Starts the 9-stage background pipeline immediately; returns right
    away (does NOT wait for anything to finish).

GET /api/pipeline/status/{job_id}
    Returns per-stage status: pending / running / done / failed / blocked
    (no result payloads — lightweight, safe to poll every 1-2 seconds).

GET /api/pipeline/result/{job_id}/{stage_name}
    Returns one stage's actual result payload (chart data, tables, etc.)
    once that stage's status is "done". 404 if not ready yet — the
    frontend should keep polling /status and only call this once a
    stage shows "done".
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.pipeline.orchestrator import (
    create_job, run_pipeline_job, get_job_status, get_stage_result, STAGE_DEPENDENCIES,
)
from src.pipeline.stage_functions import STAGE_FUNCTIONS


router = APIRouter(prefix="/api/pipeline", tags=["pipeline-demo"])


class PipelineStartRequest(BaseModel):
    monthly_bill_inr: float
    city: str
    subsidy_eligible: bool = True
    n_workers: Optional[int] = None
    ga_pop_size: int = 12
    ga_n_gen: int = 8


def _build_financial_config(subsidy_eligible: bool) -> dict:
    """
    Self-contained here (not imported from app.py) so this router has no
    hard dependency on app.py's internal structure — keeps it a genuinely
    drop-in file that works even if app.py's helper names change later.
    Mirrors the same subsidy logic used by the existing /api/size endpoint.
    """
    return dict(
        panel_cost_per_kw=45000.0, battery_cost_per_kwh=15000.0, balance_of_system_cost=10000.0,
        subsidy_rate_up_to_2kw=0.40 if subsidy_eligible else 0.0,
        subsidy_rate_2_to_3kw=0.20 if subsidy_eligible else 0.0,
        subsidy_max_kw=3.0,
        analysis_period_years=25, discount_rate=0.08,
        electricity_inflation_rate=0.05, panel_degradation_rate=0.005,
        maintenance_cost_annual_inr=2000.0,
    )


@router.post("/start")
async def start_pipeline(req: PipelineStartRequest) -> dict:
    from src.solar.locations import get_city
    city_info = get_city(req.city)
    city_name = city_info.name if city_info is not None else req.city

    params = {
        "monthly_bill_inr": req.monthly_bill_inr,
        "city": city_name,
        "financial_config": _build_financial_config(req.subsidy_eligible),
        "n_workers": req.n_workers,
        "ga_pop_size": req.ga_pop_size,
        "ga_n_gen": req.ga_n_gen,
    }
    job_id = create_job(params)

    # Fire-and-forget: do NOT await this here, or the HTTP response would
    # block until the entire 9-stage pipeline finishes (defeating the
    # whole point of background execution). asyncio.create_task lets it
    # run independently while this endpoint returns immediately.
    asyncio.create_task(run_pipeline_job(job_id, STAGE_FUNCTIONS))

    logger.info(f"Pipeline job started: {job_id} (city={req.city}, bill=₹{req.monthly_bill_inr})")
    return {"job_id": job_id, "stages": list(STAGE_DEPENDENCIES.keys())}


@router.get("/status/{job_id}")
async def pipeline_status(job_id: str) -> dict:
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return status


@router.get("/result/{job_id}/{stage_name}")
async def pipeline_stage_result(job_id: str, stage_name: str) -> dict:
    if stage_name not in STAGE_DEPENDENCIES:
        raise HTTPException(400, f"Unknown stage_name: {stage_name}. Valid: {list(STAGE_DEPENDENCIES.keys())}")

    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")

    stage_status = status["stages"][stage_name]["status"]
    if stage_status != "done":
        raise HTTPException(
            409,  # 409 Conflict: the resource exists but isn't in a usable state yet
            f"Stage '{stage_name}' is not ready (status: {stage_status}). Poll /status first.",
        )

    result = get_stage_result(job_id, stage_name)
    if result is None:
        raise HTTPException(500, f"Stage '{stage_name}' marked done but has no result — internal error")

    # Strip large/non-JSON-serializable working arrays (e.g. raw load_kw /
    # solar_kw_per_kwp numpy arrays that early stages keep around for
    # downstream stages to consume) — the frontend only needs chart/table
    # data, not multi-thousand-element raw arrays.
    NON_SERIALIZABLE_KEYS = {"load_kw", "timestamps", "solar_kw_per_kwp"}
    clean_result = {k: v for k, v in result.items() if k not in NON_SERIALIZABLE_KEYS}
    return clean_result