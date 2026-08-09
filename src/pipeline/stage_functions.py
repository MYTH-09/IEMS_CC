"""
IEMS – Pipeline Demo Stage Functions
========================================
The actual async stage implementations the orchestrator calls, wiring
each of the 9 demo tabs to real project modules. Each function has the
signature `(params, get_upstream) -> dict`, where `get_upstream(name)`
returns an already-completed upstream stage's result dict.

Heavy synchronous work (grid search, NSGA-II, MPC's per-step solves) is
offloaded to a thread via `asyncio.to_thread` so it doesn't block the
event loop — without this, one slow stage would freeze the status-polling
endpoint for every other concurrent user, not just this job.

Chart payloads are pre-shaped here (not left for the frontend to compute)
so each stage's `result` is directly what its tab needs to render —
keeps the frontend as a thin rendering layer over real, already-processed
numbers.
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.grid.grid_model import GridModel
from src.environment.energy_env import EnergyEnv
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.sizing.grid_search import run_grid_search, top_recommendations, simulate_system_size
from src.report.financials import full_financial_analysis


class _FixedSolarCurve:
    """
    Picklable stand-in solar-curve callable for stage_ga's NSGA-II run
    (tilt/azimuth ignored — always returns the same pre-computed per-kWp
    curve, same simplification used elsewhere in this project for demo
    speed, e.g. app.py's FixedSolarCurve).

    ⚠ MUST be a module-level class, not nested inside a function — this
    was a real, confirmed bug: multiprocessing (used by run_nsga2's
    n_workers) pickles callables by reference (module + qualified name)
    to send them to worker processes. A class defined inside stage_ga's
    local _run() has no such importable path and fails with
    "Can't pickle local object '...'" — this is the exact same lesson
    already learned once in scripts/10_nsga2_sizing.py (see
    SyntheticSolarCurve/RealSolarCurve there), reintroduced here because
    this file was written without cross-checking that earlier fix.

    Also note: the array is stored as a real __init__ argument, not
    captured via closure — a closure over an enclosing function's local
    variable has the same pickling problem as a nested class, even if
    the class itself were somehow made "module-level" some other way.
    """
    def __init__(self, solar_kw_per_kwp: np.ndarray):
        self.solar_kw_per_kwp = solar_kw_per_kwp

    def __call__(self, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        return self.solar_kw_per_kwp

try:
    from src.controllers.historical_forecast_provider import HistoricalForecastProvider
except ImportError:
    HistoricalForecastProvider = None

try:
    from src.controllers.dqn_controller import DQNController
except ImportError:
    DQNController = None


_PROCESSED_DATA_PATH = "data/processed/meters_15min.parquet"
_DEMO_EPISODE_DAYS = 30   # keep each controller's demo run short (fast tab loads); a
                          # different, longer window backs the headline validated numbers


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _derive_monthly_outage_multipliers(city: str) -> dict[int, float]:
    if not os.path.exists(_PROCESSED_DATA_PATH):
        return {}
    df = pd.read_parquet(_PROCESSED_DATA_PATH, columns=["city", "timestamp", "is_outage"])
    city_df = df[df["city"] == city]
    if len(city_df) == 0:
        return {}
    city_df = city_df.copy()
    city_df["month"] = pd.to_datetime(city_df["timestamp"]).dt.month
    overall_rate = city_df["is_outage"].mean()
    if overall_rate <= 0:
        return {}
    monthly_rate = city_df.groupby("month")["is_outage"].mean()
    return {int(k): round(float(v / overall_rate), 3) for k, v in monthly_rate.items()}


def _build_grid(city: str) -> GridModel:
    return GridModel(
        fallback_outage_hours_per_day=4.0,
        outage_peak_hours=[10, 14, 18, 22],
        monthly_outage_multiplier=_derive_monthly_outage_multipliers(city),
    )


def _day_shape_chart(load_kw: np.ndarray, timestamps: pd.DatetimeIndex, n_days_avg: int = 30) -> dict:
    """Average 96-slot daily shape over the first n_days_avg days — the
    chart synthetic_load and solar_generation both use."""
    slots = np.zeros(96)
    counts = np.zeros(96)
    for i in range(min(len(load_kw), n_days_avg * 96)):
        slot = i % 96
        slots[slot] += load_kw[i]
        counts[slot] += 1
    avg = np.divide(slots, counts, out=np.zeros_like(slots), where=counts > 0)
    return {"hour_labels": [f"{h:02d}:00" for h in range(24) for _ in range(4)][::4],
            "values_kw": avg[::4].round(3).tolist()}  # downsample to hourly for a cleaner chart


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Synthetic Load
# ─────────────────────────────────────────────────────────────────────────────

async def stage_synthetic_load(params: dict, get_upstream: Callable) -> dict:
    from src.data.profile_generator import load_model, generate_profile

    def _run():
        daily_kwh = params["monthly_bill_inr"] / 5.0 / 30.0
        try:
            profiles, kmeans = load_model("models")
            raw = generate_profile(profiles, kmeans, target_daily_kwh=daily_kwh,
                                    season="summer", is_weekend=False, n_days=400)
            source = "real_module1_archetypes"
        except FileNotFoundError:
            rng = np.random.default_rng(42)
            hours = np.arange(96) * 15 / 60.0
            shape = 0.3 + 0.4*np.exp(-((hours-8)**2)/4) + 0.6*np.exp(-((hours-19)**2)/4)
            shape = shape / shape.sum() * daily_kwh
            raw = np.tile(shape, 400).astype(np.float32)
            source = "synthetic_fallback"

        load_kw = (raw * 4.0).astype(np.float32)  # kWh/slot -> kW
        timestamps = pd.date_range("2024-01-01", periods=len(load_kw), freq="15min")
        chart = _day_shape_chart(load_kw, timestamps)
        return {
            "load_kw": load_kw, "timestamps": timestamps,   # consumed by downstream stages
            "chart": chart, "source": source,
            "daily_kwh_target": round(daily_kwh, 2),
            "peak_kw": round(float(load_kw.max()), 2),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Solar Generation
# ─────────────────────────────────────────────────────────────────────────────

async def stage_solar_generation(params: dict, get_upstream: Callable) -> dict:
    def _run():
        solar_path = f"data/solar/{params['city']}_solar_15min.parquet"
        if os.path.exists(solar_path):
            df = pd.read_parquet(solar_path)
            solar_kw = df["solar_power_kw"].to_numpy(dtype=np.float32)
            timestamps = pd.DatetimeIndex(df["timestamp"])
            source = "real_nasa_pvlib"
        else:
            timestamps = pd.date_range("2024-01-01", periods=400*96, freq="15min")
            hours = (timestamps.hour + timestamps.minute/60.0).to_numpy(dtype=float)
            solar_kw = np.clip(3.0*np.exp(-((hours-12)**2)/8), 0, None).astype(np.float32)
            solar_kw = np.where((hours < 6) | (hours > 18), 0.0, solar_kw).astype(np.float32)
            source = "synthetic_fallback"

        monthly = pd.DataFrame({"timestamp": timestamps, "solar_kw": solar_kw})
        monthly["month"] = monthly["timestamp"].dt.month
        monthly_kwh = (monthly.groupby("month")["solar_kw"].sum() / 4.0).round(1)  # kW*0.25h -> kWh

        return {
            "solar_kw_per_kwp": solar_kw / 3.0, "timestamps": timestamps,   # per-kWp for downstream scaling
            "chart": {"months": [f"{m:02d}" for m in monthly_kwh.index], "kwh": monthly_kwh.tolist()},
            "source": source,
            "annual_yield_kwh_3kwp": round(float(solar_kw.sum() / 4.0 / (len(solar_kw)/(365*96))), 0),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Grid Search
# ─────────────────────────────────────────────────────────────────────────────

async def stage_grid_search(params: dict, get_upstream: Callable) -> dict:
    load = get_upstream("synthetic_load")
    solar = get_upstream("solar_generation")

    def _run():
        n = min(len(load["load_kw"]), len(solar["solar_kw_per_kwp"]))
        load_kw = load["load_kw"][:n]
        solar_kw_per_kwp = solar["solar_kw_per_kwp"][:n]
        timestamps = load["timestamps"][:n]

        grid = _build_grid(params["city"])
        panel_sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        battery_sizes = [0.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        financial_config = params["financial_config"]

        df = run_grid_search(load_kw, solar_kw_per_kwp, timestamps, grid,
                              panel_sizes, battery_sizes, financial_config,
                              seed=42, n_workers=params.get("n_workers"))
        recs = top_recommendations(df, n=1)
        best_npv_row = recs["best_npv"].iloc[0]

        return {
            "table": df.round(2).to_dict(orient="records"),
            "chart": {"panel_kw": df["panel_kw"].tolist(), "battery_kwh": df["battery_kwh"].tolist(),
                      "npv": df["npv_25yr_inr"].tolist()},
            "recommended": {"panel_kw": float(best_npv_row["panel_kw"]), "battery_kwh": float(best_npv_row["battery_kwh"])},
            "n_configurations": len(df),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — GA (NSGA-II)
# ─────────────────────────────────────────────────────────────────────────────

async def stage_ga(params: dict, get_upstream: Callable) -> dict:
    load = get_upstream("synthetic_load")
    solar = get_upstream("solar_generation")

    def _run():
        from src.sizing.nsga2_sizing import run_nsga2

        n = min(len(load["load_kw"]), len(solar["solar_kw_per_kwp"]))
        load_kw = load["load_kw"][:n]
        timestamps = load["timestamps"][:n]
        solar_per_kwp = solar["solar_kw_per_kwp"][:n]

        grid = _build_grid(params["city"])
        baseline = simulate_system_size(load_kw, np.zeros_like(load_kw), timestamps,
                                          panel_kw=0.0, battery_kwh=0.0, grid=grid,
                                          episode_days=365, seed=42)
        baseline_cost = baseline["annual_cost_inr"]

        pareto = run_nsga2(load_kw=load_kw, timestamps=timestamps, grid=grid,
                            solar_curve_fn=_FixedSolarCurve(solar_per_kwp), financial_config=params["financial_config"],
                            baseline_annual_cost_inr=baseline_cost,
                            pop_size=params.get("ga_pop_size", 12), n_generations=params.get("ga_n_gen", 8),
                            episode_days=180, seed=42, n_workers=params.get("n_workers"))

        best = pareto.loc[pareto["npv_25yr_inr"].idxmax()]
        return {
            "table": pareto.round(2).to_dict(orient="records"),
            "chart": {"capex": pareto["total_capex_inr"].tolist(), "npv": pareto["npv_25yr_inr"].tolist(),
                      "unmet_kwh": pareto["annual_unmet_load_kwh"].tolist()},
            "recommended": {"panel_kw": float(best["panel_kw"]), "battery_kwh": float(best["battery_kwh"]),
                             "tilt_deg": float(best["tilt_deg"]), "azimuth_deg": float(best["azimuth_deg"])},
            "n_configurations": len(pareto),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Controller stages (shared runner)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_recommended_system(get_upstream: Callable) -> dict:
    """GA's recommendation takes priority (more thorough search); falls
    back to grid search's if GA failed/was blocked."""
    ga = get_upstream("ga")
    if ga is not None:
        return ga["recommended"]
    gs = get_upstream("grid_search")
    return gs["recommended"]


def _run_controller_demo(load, solar, params, controller, panel_kw, battery_kwh):
    n = min(len(load["load_kw"]), len(solar["solar_kw_per_kwp"]))
    load_kw = load["load_kw"][:n]
    solar_kw = solar["solar_kw_per_kwp"][:n] * panel_kw
    timestamps = load["timestamps"][:n]

    grid = _build_grid(params["city"])
    battery_config = dict(capacity_kwh=max(battery_kwh, 0.01),
                           max_charge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
                           max_discharge_rate_kw=min(3.0, max(battery_kwh, 0.01)),
                           round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50)
    env = EnergyEnv(load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
                     battery_config=battery_config, grid=grid, episode_days=_DEMO_EPISODE_DAYS, seed=42)
    obs, info = env.reset(seed=42)
    controller.reset()

    soc_trace, hour_trace = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.select_action(env)
        idx_before = env._data_idx
        obs, reward, terminated, truncated, info = env.step(action)
        if idx_before < 96:   # capture first day's SOC trace for the chart
            soc_trace.append(round(env.battery.soc * 100, 1))
            hour_trace.append(timestamps[idx_before].strftime("%H:%M"))

    return {
        "chart": {"hours": hour_trace, "soc_pct": soc_trace},
        "total_cost_inr": round(info["episode_cost_inr"], 2),
        "grid_import_kwh": round(info["episode_grid_import_kwh"], 1),
        "unmet_load_kwh": round(info["episode_unmet_load_kwh"], 1),
        "panel_kw": panel_kw, "battery_kwh": battery_kwh,
    }


async def stage_rule_based(params: dict, get_upstream: Callable) -> dict:
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = _pick_recommended_system(get_upstream)

    def _run():
        return _run_controller_demo(load, solar, params, RuleBasedController(),
                                      rec["panel_kw"], rec["battery_kwh"])
    return await asyncio.to_thread(_run)


async def stage_mpc_normal(params: dict, get_upstream: Callable) -> dict:
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = _pick_recommended_system(get_upstream)

    def _run():
        return _run_controller_demo(load, solar, params, MPCController(horizon_hours=24.0),
                                      rec["panel_kw"], rec["battery_kwh"])
    return await asyncio.to_thread(_run)


async def stage_mpc_weather(params: dict, get_upstream: Callable) -> dict:
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = _pick_recommended_system(get_upstream)

    def _run():
        provider = HistoricalForecastProvider() if HistoricalForecastProvider else None
        return _run_controller_demo(load, solar, params, MPCController(horizon_hours=24.0, forecast_provider=provider),
                                      rec["panel_kw"], rec["battery_kwh"])
    return await asyncio.to_thread(_run)


async def stage_dqn(params: dict, get_upstream: Callable) -> dict:
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = _pick_recommended_system(get_upstream)

    def _run():
        model_path = "models/dqn_policy.zip"
        if DQNController is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"'{model_path}' not found — run scripts/07_train_dqn.py --full first")
        return _run_controller_demo(load, solar, params, DQNController(model_path=model_path),
                                      rec["panel_kw"], rec["battery_kwh"])
    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 9 — Outage Protection Proof
# ─────────────────────────────────────────────────────────────────────────────

async def stage_outage_proof(params: dict, get_upstream: Callable) -> dict:
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")

    def _run():
        controller_results = {
            "rule_based": get_upstream("rule_based"), "mpc_normal": get_upstream("mpc_normal"),
            "mpc_weather": get_upstream("mpc_weather"), "dqn": get_upstream("dqn"),
        }
        # Filter out any controller that failed (get_upstream returns None
        # for a stage that never completed successfully — e.g. DQN with no
        # trained model file). This is not an edge case: outage_proof's
        # any_of dependency is SPECIFICALLY designed to still run when only
        # some controllers succeeded (see STAGE_DEPENDENCIES's comment) —
        # skipping this filter crashes with 'NoneType' object is not
        # subscriptable the moment that documented fallback actually fires,
        # which it did in real testing (DQN failing on a fresh machine).
        available = {name: r for name, r in controller_results.items() if r is not None}
        if not available:
            raise RuntimeError(
                "outage_proof: no controller results available — "
                "rule_based, mpc_normal, mpc_weather, and dqn all failed or were blocked."
            )

        # Pick whichever AVAILABLE controller achieved the lowest unmet load
        best_name, best = min(available.items(), key=lambda kv: kv[1]["unmet_load_kwh"])

        # "Without solar" baseline: panel=0, battery=0, rule-based (no controller choice
        # matters when there's no battery/solar — the household is 100% grid-dependent)
        grid = _build_grid(params["city"])
        n = min(len(load["load_kw"]), len(solar["solar_kw_per_kwp"]))
        no_solar_env = EnergyEnv(
            load_kw=load["load_kw"][:n], solar_kw=np.zeros(n, dtype=np.float32),
            timestamps=load["timestamps"][:n],
            battery_config=dict(capacity_kwh=0.01, max_charge_rate_kw=0.01, max_discharge_rate_kw=0.01,
                                 round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50),
            grid=grid, episode_days=_DEMO_EPISODE_DAYS, seed=42,
        )
        obs, info = no_solar_env.reset(seed=42)
        rb = RuleBasedController(); rb.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = no_solar_env.step(0)

        return {
            "best_controller": best_name,
            "with_solar": {"unmet_load_kwh": best["unmet_load_kwh"], "cost_inr": best["total_cost_inr"]},
            "without_solar": {"unmet_load_kwh": round(info["episode_unmet_load_kwh"], 1),
                               "cost_inr": round(info["episode_cost_inr"], 2)},
        }

    return await asyncio.to_thread(_run)


STAGE_FUNCTIONS = {
    "synthetic_load": stage_synthetic_load,
    "solar_generation": stage_solar_generation,
    "grid_search": stage_grid_search,
    "ga": stage_ga,
    "rule_based": stage_rule_based,
    "mpc_normal": stage_mpc_normal,
    "mpc_weather": stage_mpc_weather,
    "dqn": stage_dqn,
    "outage_proof": stage_outage_proof,
}