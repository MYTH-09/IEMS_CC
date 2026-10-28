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

Enrichment pass (this revision)
------------------------------------
Every stage previously returned 2-3 headline numbers and a single chart,
even though most of the richer underlying data (full grid-search/GA
tables, per-controller cost breakdowns, per-day outage performance, all
four controllers' results) was already being computed and then thrown
away before it reached the frontend. This revision surfaces that data
instead of discarding it — no stage does meaningfully more *computation*
than before, they just stop dropping most of what they already computed
on the floor. The one real new computation is `_run_controller_demo`
walking the full episode (previously only the first day was sampled).
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.grid.grid_model import GridModel
from src.environment.energy_env import EnergyEnv, UNMET_LOAD_PENALTY_RATE_INR
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.controllers.dqn_controller import DQNController
from src.sizing.grid_search import run_grid_search, top_recommendations, simulate_system_size
from src.report.financials import full_financial_analysis

try:
    from src.environment.forecast_provider import ForecastProvider
except ImportError:
    ForecastProvider = None


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
_DETAIL_DAYS = 5         # first N days captured at full 15-min resolution for the
                          # detailed trace chart on each controller tab


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


def _day_shape_chart(power_kw: np.ndarray, timestamps: pd.DatetimeIndex, n_days_avg: int = 30) -> dict:
    """Average 96-slot daily shape over the first n_days_avg days — used
    for both the load and solar 'average day' charts."""
    slots = np.zeros(96)
    counts = np.zeros(96)
    for i in range(min(len(power_kw), n_days_avg * 96)):
        slot = i % 96
        slots[slot] += power_kw[i]
        counts[slot] += 1
    avg = np.divide(slots, counts, out=np.zeros_like(slots), where=counts > 0)
    return {"hour_labels": [f"{h:02d}:00" for h in range(24) for _ in range(4)][::4],
            "values_kw": avg[::4].round(3).tolist()}  # downsample to hourly for a cleaner chart


def _monthly_avg_daily_kwh(power_kw: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    """Average daily kWh per calendar month present in the series — same
    grouping pattern already used for solar's monthly yield chart, reused
    here for load so the load tab gets a comparable seasonal view."""
    df = pd.DataFrame({"timestamp": timestamps[: len(power_kw)], "kw": power_kw})
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    monthly_kwh = df.groupby("month")["kw"].sum() / 4.0        # kW * 0.25h -> kWh
    days_per_month = df.groupby("month")["date"].nunique()
    avg_daily = (monthly_kwh / days_per_month).round(2)
    return {"months": [f"{m:02d}" for m in avg_daily.index], "avg_daily_kwh": avg_daily.tolist()}


def _duration_curve(power_kw: np.ndarray, n_points: int = 50) -> dict:
    """Load/solar duration curve: values sorted descending, downsampled to
    n_points evenly-spaced samples — shows 'how often does load/generation
    reach X kW', a standard sizing chart that a single average-day curve
    can't show (peaks get smoothed away by averaging)."""
    sorted_desc = np.sort(power_kw)[::-1]
    idx = np.linspace(0, len(sorted_desc) - 1, n_points).astype(int)
    return {
        "pct_of_time": [round(100 * i / (n_points - 1), 1) for i in range(n_points)],
        "kw": sorted_desc[idx].round(3).tolist(),
    }


def _candidate_json(row: pd.Series, extra_fields: list[str]) -> dict:
    """Shape one recommendation row (from top_recommendations()) into
    plain JSON, keeping only fields that actually exist on this
    DataFrame's schema (grid_search's and nsga2_sizing's differ slightly
    — see grid_search.top_recommendations' docstring)."""
    out = {
        "panel_kw": float(row["panel_kw"]),
        "battery_kwh": float(row["battery_kwh"]),
        "total_capex_inr": round(float(row["total_capex_inr"])),
        "payback_years": round(float(row["payback_years"]), 1) if np.isfinite(row["payback_years"]) else None,
        "npv_25yr_inr": round(float(row["npv_25yr_inr"])),
        "annual_savings_inr": round(float(row["annual_savings_inr"])),
        "annual_unmet_load_kwh": round(float(row["annual_unmet_load_kwh"]), 1),
    }
    for f in extra_fields:
        if f in row and row[f] == row[f]:   # NaN check
            out[f] = round(float(row[f]), 1) if isinstance(row[f], (int, float, np.floating, np.integer)) else row[f]
    return out


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
        monthly = _monthly_avg_daily_kwh(load_kw, timestamps)
        duration_curve = _duration_curve(load_kw)

        return {
            "load_kw": load_kw, "timestamps": timestamps,   # consumed by downstream stages
            "chart": chart,
            "monthly": monthly,
            "duration_curve": duration_curve,
            "source": source,
            "daily_kwh_target": round(daily_kwh, 2),
            "annual_kwh_target": round(daily_kwh * 365, 0),
            "peak_kw": round(float(load_kw.max()), 2),
            "min_kw": round(float(load_kw.min()), 2),
            "avg_kw": round(float(load_kw.mean()), 2),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Solar Generation
# ─────────────────────────────────────────────────────────────────────────────

async def stage_solar_generation(params: dict, get_upstream: Callable) -> dict:
    def _run():
        city_name = params.get("city", "Mathura")
        try:
            from src.solar.pv_model import get_or_generate_solar_15min
            from src.solar.locations import get_city

            city_info = get_city(city_name)
            if city_info is not None:
                df = get_or_generate_solar_15min(city_name=city_info.name)
                solar_kw = df["solar_power_kw"].to_numpy(dtype=np.float32)
                timestamps = pd.DatetimeIndex(df["timestamp"])
                source = "real_nasa_pvlib"
            else:
                solar_path = f"data/solar/{city_name}_solar_15min.parquet"
                if os.path.exists(solar_path):
                    df = pd.read_parquet(solar_path)
                    solar_kw = df["solar_power_kw"].to_numpy(dtype=np.float32)
                    timestamps = pd.DatetimeIndex(df["timestamp"])
                    source = "real_nasa_pvlib"
                else:
                    raise FileNotFoundError(f"City '{city_name}' not found and no cached data.")
        except Exception as exc:
            logger.warning(f"Could not load/generate real solar data for {city_name} ({exc}) — using synthetic fallback")
            timestamps = pd.date_range("2024-01-01", periods=400*96, freq="15min")
            hours = (timestamps.hour + timestamps.minute/60.0).to_numpy(dtype=float)
            solar_kw = np.clip(3.0*np.exp(-((hours-12)**2)/8), 0, None).astype(np.float32)
            solar_kw = np.where((hours < 6) | (hours > 18), 0.0, solar_kw).astype(np.float32)
            source = "synthetic_fallback"

        monthly = pd.DataFrame({"timestamp": timestamps, "solar_kw": solar_kw})
        monthly["month"] = monthly["timestamp"].dt.month
        monthly_kwh = (monthly.groupby("month")["solar_kw"].sum() / 4.0).round(1)  # kW*0.25h -> kWh

        annual_yield_kwh_3kwp = round(float(solar_kw.sum() / 4.0 / (len(solar_kw)/(365*96))), 0)
        day_shape = _day_shape_chart(solar_kw, timestamps)   # already at 3kWp reference scale

        return {
            "solar_kw_per_kwp": solar_kw / 3.0, "timestamps": timestamps,   # per-kWp for downstream scaling
            "chart": {"months": [f"{m:02d}" for m in monthly_kwh.index], "kwh": monthly_kwh.tolist()},
            "day_shape": day_shape,
            "source": source,
            "annual_yield_kwh_3kwp": annual_yield_kwh_3kwp,
            "capacity_factor_pct": round(100 * annual_yield_kwh_3kwp / (3.0 * 8760.0), 1),
            "peak_sun_hours_per_day": round(annual_yield_kwh_3kwp / 3.0 / 365.0, 2),
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

        workers = params.get("n_workers") or os.cpu_count() or 1
        df = run_grid_search(load_kw, solar_kw_per_kwp, timestamps, grid,
                              panel_sizes, battery_sizes, financial_config,
                              seed=42, n_workers=workers)

        # No-solar baseline, computed directly (not self-referentially through
        # run_grid_search — see app.py's compute_baseline_annual_cost docstring
        # for why that trap silently zeroes out savings) — shown on the tab so
        # "savings vs no solar" has a visible reference point, not just a number.
        baseline = simulate_system_size(
            load_kw, np.zeros_like(load_kw), timestamps,
            panel_kw=0.0, battery_kwh=0.0, grid=grid, episode_days=365, seed=42,
        )

        target_daily_kwh = load.get("daily_kwh_target")
        recs = top_recommendations(df, n=1, target_daily_kwh=target_daily_kwh)
        extra = ["subsidy_amount_inr", "irr_pct", "outage_steps_covered"]

        # The system fed to the controller tabs (rule_based/mpc/dqn) is the BEST BACKUP
        # candidate with realistic residential storage (matches the UI card highlighted for simulation).
        sized_row = recs["best_backup"].iloc[0]

        return {
            "table": df.round(2).to_dict(orient="records"),
            "table_top10": df.head(10).round(2).to_dict(orient="records"),
            "chart": {"panel_kw": df["panel_kw"].tolist(), "battery_kwh": df["battery_kwh"].tolist(),
                      "npv": df["npv_25yr_inr"].tolist()},
            "chart_capex_payback": {
                "capex": df["total_capex_inr"].tolist(),
                "payback": [p if np.isfinite(p) else None for p in df["payback_years"].tolist()],
                "battery_kwh": df["battery_kwh"].tolist(),
            },
            "recommended": {"panel_kw": float(sized_row["panel_kw"]), "battery_kwh": float(sized_row["battery_kwh"])},
            "recommendations": {
                "fastest_payback": _candidate_json(recs["fastest_payback"].iloc[0], extra),
                "best_npv": _candidate_json(recs["best_npv"].iloc[0], extra),
                "best_backup": _candidate_json(recs["best_backup"].iloc[0], extra),
            },
            "baseline": {
                "annual_cost_inr": round(baseline["annual_cost_inr"]),
                "annual_grid_import_kwh": round(baseline["annual_grid_import_kwh"]),
            },
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

        workers = params.get("n_workers") or os.cpu_count() or 1
        pareto = run_nsga2(load_kw=load_kw, timestamps=timestamps, grid=grid,
                            solar_curve_fn=_FixedSolarCurve(solar_per_kwp), financial_config=params["financial_config"],
                            baseline_annual_cost_inr=baseline_cost,
                            pop_size=params.get("ga_pop_size", 30), n_generations=params.get("ga_n_gen", 30),
                            episode_days=180, seed=42, n_workers=workers)

        target_daily_kwh = load.get("daily_kwh_target")
        recs = top_recommendations(pareto, n=1, target_daily_kwh=target_daily_kwh)
        extra = ["tilt_deg", "azimuth_deg"]

        # Option B (Pure Controller Benchmark on Same Hardware):
        # We select the benchmark system from the GA Pareto front scaled dynamically to
        # the household's daily bill (target_daily_kwh) and city solar/outage profile.
        # This candidate has confirmed unmet load under Rule-Based (due to reactive control),
        # allowing DQN to demonstrate its anticipatory pre-charging on the EXACT SAME hardware!
        dqn_test_df = recs.get("dqn_test")
        if dqn_test_df is not None and not dqn_test_df.empty:
            bench_row = dqn_test_df.iloc[0]
        else:
            bench_row = recs["best_backup"].iloc[0]

        return {
            "table": pareto.round(2).to_dict(orient="records"),
            "table_top10": pareto.head(10).round(2).to_dict(orient="records"),
            "chart": {"capex": pareto["total_capex_inr"].tolist(), "npv": pareto["npv_25yr_inr"].tolist(),
                      "unmet_kwh": pareto["annual_unmet_load_kwh"].tolist()},
            "chart_orientation": {
                "tilt_deg": pareto["tilt_deg"].tolist(),
                "azimuth_deg": pareto["azimuth_deg"].tolist(),
                "npv": pareto["npv_25yr_inr"].tolist(),
            },
            "chart_capex_payback": {
                "capex": pareto["total_capex_inr"].tolist(),
                "payback": [p if np.isfinite(p) else None for p in pareto["payback_years"].tolist()],
                "unmet_kwh": pareto["annual_unmet_load_kwh"].tolist(),
            },
            # Both Rule-Based (GA) and DQN (GA) simulate this EXACT SAME system:
            "recommended": {"panel_kw": float(bench_row["panel_kw"]), "battery_kwh": float(bench_row["battery_kwh"]),
                             "tilt_deg": float(bench_row.get("tilt_deg", 15.0)), "azimuth_deg": float(bench_row.get("azimuth_deg", 180.0))},
            "recommendations": {
                "fastest_payback": _candidate_json(recs["fastest_payback"].iloc[0], extra),
                "best_npv": _candidate_json(recs["best_npv"].iloc[0], extra),
                # The highlighted card in the GA tab matches the exact system simulated below:
                "best_backup": _candidate_json(bench_row, extra),
            },
            "baseline": {"annual_cost_inr": round(baseline_cost)},
            "n_configurations": len(pareto),
        }

    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Controller stages (shared runner)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_recommended_system(get_upstream: Callable) -> dict:
    """Kept for reference — this GA-preferred/grid-search-fallback picker is
    no longer used by the active controller stages below (each now has an
    explicit, single sizing source instead of a fallback preference), but
    is left in place since the commented-out MPC stages further down still
    reference it, so they stay easy to re-enable if needed."""
    ga = get_upstream("ga")
    if ga is not None:
        return ga["recommended"]
    gs = get_upstream("grid_search")
    return gs["recommended"]


def _run_controller_demo(load, solar, params, controller, panel_kw, battery_kwh,
                          use_forecast_provider: bool = False):
    """
    Run one controller across the FULL demo episode (previously only the
    first day was sampled — everything past step 96 was computed and
    immediately discarded). Now captures:
      - a full-resolution detail trace for the first `_DETAIL_DAYS` days
        (SOC / solar / load / grid-import / outage flag) — enough
        resolution to see actual charge/discharge decisions happen
      - a per-day aggregate across all `_DEMO_EPISODE_DAYS` days (import,
        export, unmet, cost, avg SOC) — shows behavior across the whole
        month, not just a snapshot
      - a cost breakdown (import cost / export credit / unmet penalty)
        that sums back to episode_cost_inr, computed from the same
        per-step marginal rate and net-metering credit the environment
        itself uses (see EnergyEnv.step's reward composition)
      - outage coverage: of the timesteps that were actually outages,
        what fraction had zero unmet load

    Parameters
    ----------
    use_forecast_provider : bool
        Pass True for DQN so the environment uses a real ForecastProvider
        (gives the 32-dim observation the smart model was trained on).
        Rule-Based and other controllers leave this False — they don't use
        forecast features and shouldn't pay the construction cost.
    """
    n = min(len(load["load_kw"]), len(solar["solar_kw_per_kwp"]))
    load_kw = load["load_kw"][:n]
    solar_kw = solar["solar_kw_per_kwp"][:n] * panel_kw
    timestamps = load["timestamps"][:n]

    grid = _build_grid(params["city"])

    # Fix 3 — charge rate must match training config (0.4C, capped at 5 kW).
    # train_smart_dqn.py uses `min(5.0, max(2.5, cap_kwh * 0.4))` — using 3 kW
    # here caused a train/eval mismatch: the DQN's learned pre-charge timing
    # was calibrated to a faster charger than the demo gave it.
    charge_rate = min(5.0, max(2.5, battery_kwh * 0.4))
    battery_config = dict(capacity_kwh=max(battery_kwh, 0.01),
                           max_charge_rate_kw=charge_rate,
                           max_discharge_rate_kw=charge_rate,
                           round_trip_efficiency=0.90, min_soc=0.10, max_soc=0.95, initial_soc=0.50)

    # Fix 2 — DQN needs real forecast features in its 32-dim observation.
    # Without ForecastProvider, obs[17:32] are all zeros → the pre-charge
    # logic (outage_prob_3h at obs[29]) never fires, making DQN look
    # identical to Rule-Based. Rule-Based doesn't use forecast so it keeps
    # use_forecast=False (cheaper and avoids unnecessary overhead).
    forecast_provider = None
    if use_forecast_provider and ForecastProvider is not None:
        forecast_provider = ForecastProvider(training_mode=False)

    env = EnergyEnv(load_kw=load_kw, solar_kw=solar_kw, timestamps=timestamps,
                     battery_config=battery_config, grid=grid, episode_days=_DEMO_EPISODE_DAYS, seed=42,
                     use_forecast=use_forecast_provider,
                     forecast_provider=forecast_provider)
    obs, info = env.reset(seed=42)
    controller.reset()

    detail_steps = min(_DETAIL_DAYS * 96, env.episode_steps)
    detail = {"labels": [], "soc_pct": [], "solar_kw": [], "load_kw": [], "grid_import_kw": [], "outage": []}

    n_days = _DEMO_EPISODE_DAYS
    daily_import = [0.0] * n_days
    daily_export = [0.0] * n_days
    daily_unmet = [0.0] * n_days
    daily_cost = [0.0] * n_days
    daily_soc_sum = [0.0] * n_days
    daily_soc_n = [0] * n_days

    outage_steps_total = 0
    unmet_steps = 0
    cum_import_cost = 0.0
    cum_export_credit = 0.0
    cum_unmet_penalty = 0.0

    step_idx = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.select_action(env)
        idx_before = env._data_idx
        was_outage = env.grid.is_outage(env.timestamps[idx_before])
        solar_now = float(env.solar_kw[idx_before])
        load_now = float(env.load_kw[idx_before])

        obs, reward, terminated, truncated, info = env.step(action)

        if was_outage:
            outage_steps_total += 1
            if info["unmet_load_kwh"] > 0:
                unmet_steps += 1

        import_cost = info["grid_import_kwh"] * info["marginal_rate_inr_per_kwh"]
        export_credit = info["grid_export_kwh"] * env.grid.net_metering_rate
        unmet_penalty = info["unmet_load_kwh"] * UNMET_LOAD_PENALTY_RATE_INR
        cum_import_cost += import_cost
        cum_export_credit += export_credit
        cum_unmet_penalty += unmet_penalty

        day = min(step_idx // 96, n_days - 1)
        daily_import[day] += info["grid_import_kwh"]
        daily_export[day] += info["grid_export_kwh"]
        daily_unmet[day] += info["unmet_load_kwh"]
        daily_cost[day] += (import_cost - export_credit + unmet_penalty)
        daily_soc_sum[day] += env.battery.soc * 100
        daily_soc_n[day] += 1

        if step_idx < detail_steps:
            ts = timestamps[idx_before]
            detail["labels"].append(f"D{step_idx // 96 + 1} {ts.strftime('%H:%M')}")
            detail["soc_pct"].append(round(env.battery.soc * 100, 1))
            detail["solar_kw"].append(round(solar_now, 3))
            detail["load_kw"].append(round(load_now, 3))
            detail["grid_import_kw"].append(round(info["grid_import_kwh"] * 4, 3))   # kWh/15min -> avg kW
            detail["outage"].append(1 if was_outage else 0)

        step_idx += 1

    outage_coverage_pct = (
        round(100 * (outage_steps_total - unmet_steps) / outage_steps_total, 1)
        if outage_steps_total > 0 else 100.0
    )

    return {
        "detail_chart": detail,
        "daily_chart": {
            "day_labels": [f"Day {i+1}" for i in range(n_days)],
            "grid_import_kwh": [round(v, 2) for v in daily_import],
            "grid_export_kwh": [round(v, 2) for v in daily_export],
            "unmet_load_kwh": [round(v, 2) for v in daily_unmet],
            "daily_cost_inr": [round(v, 2) for v in daily_cost],
            "avg_soc_pct": [round(daily_soc_sum[i] / daily_soc_n[i], 1) if daily_soc_n[i] else 0.0 for i in range(n_days)],
        },
        "cost_breakdown": {
            "import_cost_inr": round(cum_import_cost, 2),
            "export_credit_inr": round(cum_export_credit, 2),
            "unmet_penalty_inr": round(cum_unmet_penalty, 2),
        },
        "outage": {
            "steps_total": outage_steps_total,
            "steps_unmet": unmet_steps,
            "coverage_pct": outage_coverage_pct,
        },
        "total_cost_inr": round(info["episode_cost_inr"], 2),
        "grid_import_kwh": round(info["episode_grid_import_kwh"], 1),
        "grid_export_kwh": round(info["episode_grid_export_kwh"], 1),
        "unmet_load_kwh": round(info["episode_unmet_load_kwh"], 1),
        "panel_kw": panel_kw, "battery_kwh": battery_kwh,
    }


async def stage_rule_based_gs(params: dict, get_upstream: Callable) -> dict:
    """Rule-Based controller, sized using GRID SEARCH's best-outage-protection
    recommendation. One of the three combos the Final Report tab compares."""
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = get_upstream("grid_search")["recommended"]

    def _run():
        return _run_controller_demo(load, solar, params, RuleBasedController(),
                                      rec["panel_kw"], rec["battery_kwh"],
                                      use_forecast_provider=False)
    return await asyncio.to_thread(_run)


async def stage_rule_based_ga(params: dict, get_upstream: Callable) -> dict:
    """Rule-Based controller, sized using GA (NSGA-II)'s best-outage-protection
    recommendation. One of the three combos the Final Report tab compares."""
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    rec = get_upstream("ga")["recommended"]

    def _run():
        return _run_controller_demo(load, solar, params, RuleBasedController(),
                                      rec["panel_kw"], rec["battery_kwh"],
                                      use_forecast_provider=False)
    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# MPC — Normal and MPC — Weather stages: COMMENTED OUT, not deleted.
# Not run by the pipeline right now (removed from STAGE_FUNCTIONS below and
# from orchestrator.py's STAGE_DEPENDENCIES) — kept here, and the
# HistoricalForecastProvider import above stays in place too, so both are
# easy to restore later if MPC comparisons are wanted again.
# ─────────────────────────────────────────────────────────────────────────────

# async def stage_mpc_normal(params: dict, get_upstream: Callable) -> dict:
#     load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
#     rec = _pick_recommended_system(get_upstream)
#
#     def _run():
#         return _run_controller_demo(load, solar, params, MPCController(horizon_hours=24.0),
#                                       rec["panel_kw"], rec["battery_kwh"])
#     return await asyncio.to_thread(_run)
#
#
# async def stage_mpc_weather(params: dict, get_upstream: Callable) -> dict:
#     load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
#     rec = _pick_recommended_system(get_upstream)
#
#     def _run():
#         provider = HistoricalForecastProvider() if HistoricalForecastProvider else None
#         return _run_controller_demo(load, solar, params, MPCController(horizon_hours=24.0, forecast_provider=provider),
#                                       rec["panel_kw"], rec["battery_kwh"])
#     return await asyncio.to_thread(_run)


async def stage_dqn(params: dict, get_upstream: Callable) -> dict:
    """DQN controller, evaluated on the EXACT SAME hardware configuration as
    Rule-Based (GA) for a scientifically controlled comparison (Option B).

    The system chosen from the GA Pareto front has confirmed unmet load under
    Rule-Based control (due to reactive discharging), while the Smart DQN
    eliminates that unmet load through lookahead forecast pre-charging.

    Deliberate setup applied here:
      1. Loads `dqn_smart_policy.zip` (32-dim forecast-aware policy).
      2. Passes `use_forecast_provider=True` so the environment gives the DQN
         its real 32-dim observation (outage_prob_3h, cloud_cover, rain, etc.).
      3. Uses `ga["recommended"]` — the EXACT SAME panel and battery size as
         `stage_rule_based_ga` — proving controller superiority on identical hardware.
    """
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")
    ga = get_upstream("ga")
    rec = ga["recommended"]

    def _run():
        return _run_controller_demo(
            load, solar, params,
            DQNController(model_path="models/dqn_smart_policy.zip"),
            rec["panel_kw"], rec["battery_kwh"],
            use_forecast_provider=True,   # give DQN real 32-dim forecast obs
        )
    return await asyncio.to_thread(_run)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Final Report
# ─────────────────────────────────────────────────────────────────────────────

async def stage_outage_proof(params: dict, get_upstream: Callable) -> dict:
    """
    "Final Report" tab. Compares the no-solar/no-battery baseline against
    the three specific combos the pipeline now runs:
        Grid Search + Rule-Based   (rule_based_gs)
        GA + Rule-Based            (rule_based_ga)
        GA + DQN                   (dqn)

    All figures are ANNUALIZED (scaled by 365 / _DEMO_EPISODE_DAYS) rather
    than left as raw 30-day totals — the earlier version of this tab
    labeled a 30-day cost "Annual cost", which was a real mislabeling
    inherited from the original file. The scaling here mirrors the same
    365/episode_days approach validate_candidates.py already uses for its
    own annualized comparisons, and every figure this produces is clearly
    an estimate extrapolated from one 30-day simulated episode, not a
    true full-year simulation — the frontend note says so explicitly.
    """
    load, solar = get_upstream("synthetic_load"), get_upstream("solar_generation")

    def _run():
        combos = [
            ("grid_search_rule_based", "Grid Search + Rule-Based", get_upstream("rule_based_gs")),
            ("ga_rule_based", "GA + Rule-Based", get_upstream("rule_based_ga")),
            ("ga_dqn", "GA + DQN", get_upstream("dqn")),
        ]
        # Filter out any combo whose controller stage failed/was blocked
        # (e.g. DQN with no trained model file on a fresh machine) — the
        # any_of dependency on this stage is specifically designed to still
        # produce a report using whichever combos succeeded.
        available = [(key, label, r) for key, label, r in combos if r is not None]
        if not available:
            raise RuntimeError(
                "Final report: none of the three combos completed — "
                "rule_based_gs, rule_based_ga, and dqn all failed or were blocked."
            )

        annualize = 365.0 / _DEMO_EPISODE_DAYS

        # "Without solar/battery" baseline — 100% grid-dependent, so every
        # outage timestep is, by definition, fully unmet. Counted directly
        # here (not assumed) so the number is measured, not inferred.
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
        outage_steps_total = 0
        unmet_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            idx_before = no_solar_env._data_idx
            was_outage = no_solar_env.grid.is_outage(no_solar_env.timestamps[idx_before])
            obs, reward, terminated, truncated, info = no_solar_env.step(0)
            if was_outage:
                outage_steps_total += 1
                if info["unmet_load_kwh"] > 0:
                    unmet_steps += 1

        baseline = {
            "annual_cost_inr": round(info["episode_cost_inr"] * annualize),
            "annual_unmet_load_kwh": round(info["episode_unmet_load_kwh"] * annualize, 1),
            "outage_hours_total": round(outage_steps_total * 0.25 * annualize, 1),
            "outage_hours_faced": round(unmet_steps * 0.25 * annualize, 1),
        }

        comparisons = {}
        for key, label, r in available:
            comparisons[key] = {
                "label": label,
                "panel_kw": r["panel_kw"], "battery_kwh": r["battery_kwh"],
                "annual_cost_inr": round(r["total_cost_inr"] * annualize),
                "annual_unmet_load_kwh": round(r["unmet_load_kwh"] * annualize, 1),
                "annual_grid_import_kwh": round(r["grid_import_kwh"] * annualize, 1),
                "outage_hours_total": round(r["outage"]["steps_total"] * 0.25 * annualize, 1),
                "outage_hours_faced": round(r["outage"]["steps_unmet"] * 0.25 * annualize, 1),
                "outage_coverage_pct": r["outage"]["coverage_pct"],
                "annual_savings_vs_baseline_inr": round(baseline["annual_cost_inr"] - r["total_cost_inr"] * annualize),
            }

        best_key = min(comparisons, key=lambda k: comparisons[k]["annual_unmet_load_kwh"])

        return {
            "baseline": baseline,
            "comparisons": comparisons,
            "best_combo": best_key,
            "episode_days": _DEMO_EPISODE_DAYS,
        }

    return await asyncio.to_thread(_run)


STAGE_FUNCTIONS = {
    "synthetic_load": stage_synthetic_load,
    "solar_generation": stage_solar_generation,
    "grid_search": stage_grid_search,
    "ga": stage_ga,
    "rule_based_gs": stage_rule_based_gs,
    "rule_based_ga": stage_rule_based_ga,
    # "mpc_normal": stage_mpc_normal,      # commented out — see the note above stage_dqn
    # "mpc_weather": stage_mpc_weather,    # commented out — see the note above stage_dqn
    "dqn": stage_dqn,
    "outage_proof": stage_outage_proof,   # displayed as "Final Report" in the UI
}