# IEMS — Intelligent Energy Management Simulator
## Architecture, Pipeline & Module Reference

---

## 1. Project Overview

IEMS is a Python-based simulation and optimization platform for solar + battery systems targeting Indian residential households in Uttar Pradesh (Mathura & Bareilly). It ingests real CEEW smart-meter data, fetches NASA solar weather data, runs physics-based simulations, applies three kinds of energy controllers (Rule-Based, MPC, DQN), optimizes system sizing (Grid Search + NSGA-II), and exposes everything via a FastAPI backend serving two HTML frontends.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                     │
│  CEEW Smart-Meter CSVs (Mathura/Bareilly)   NASA POWER API (solar irr.) │
└────────────────────┬────────────────────────────────┬────────────────────┘
                     │                                │
                     ▼                                ▼
┌────────────────────────────┐        ┌──────────────────────────────┐
│  MODULE 1: Data Pipeline   │        │  MODULE 2: Solar Pipeline     │
│  src/data/                 │        │  src/solar/                   │
│  ├── loader.py             │        │  ├── nasa_power.py            │
│  ├── preprocessor.py       │        │  └── pv_model.py             │
│  └── profile_generator.py  │        └──────────────┬───────────────┘
└────────────────┬───────────┘                       │
                 │                                   │
                 └──────────────┬────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    CORE SIMULATION LAYER                                 │
│                                                                          │
│   ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │
│   │ src/battery/     │  │ src/grid/        │  │ src/environment/      │  │
│   │ battery_model.py │  │ grid_model.py    │  │ energy_env.py         │  │
│   │ (LFP physics)    │  │ (tariff+outage)  │  │ (Gymnasium env)       │  │
│   └──────────────────┘  └─────────────────┘  └───────────────────────┘  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      ▼             ▼             ▼
            ┌──────────────┐ ┌──────────┐ ┌─────────────┐
            │ Rule-Based   │ │   MPC    │ │     DQN     │
            │ Controller   │ │Controller│ │  Controller │
            └──────┬───────┘ └────┬─────┘ └──────┬──────┘
                   └──────────────┼───────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        SIZING LAYER                                      │
│   ┌───────────────────────┐    ┌────────────────────────────────────┐   │
│   │  src/sizing/          │    │  src/sizing/validate_candidates.py │   │
│   │  grid_search.py       │    │  (spot-check top candidates under  │   │
│   │  nsga2_sizing.py      │    │   Rule-Based, MPC, DQN)            │   │
│   └───────────────────────┘    └────────────────────────────────────┘   │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   REPORT / FINANCIAL LAYER                               │
│                      src/report/financials.py                            │
│         (CAPEX, NPV, IRR, Payback, PM Surya Ghar subsidy)                │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
          ┌──────────────┐  ┌────────────┐  ┌──────────────────┐
          │  FastAPI     │  │ Pipeline   │  │  GA Explainer    │
          │  app.py      │  │  Routes    │  │  Routes          │
          │ /api/size    │  │ pipeline_  │  │  ga_explainer_   │
          │              │  │ routes.py  │  │  routes.py       │
          └──────┬───────┘  └─────┬──────┘  └──────┬───────────┘
                 │                │                 │
       ┌─────────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
       │ Contractor UI  │  │ Pipeline    │  │ GA Explainer │
       │ (sizing form)  │  │ Demo UI     │  │ HTML         │
       └────────────────┘  └─────────────┘  └─────────────┘
```

---

## 3. Complete Pipeline Flow

### Offline Pipeline (Scripts — run once)

```
scripts/01_data_pipeline.py
  └── Loads CEEW CSVs → preprocesses → clusters profiles
      Outputs: data/processed/meters_15min.parquet
               models/load_profiles.parquet
               models/profile_kmeans.pkl

scripts/02_solar_pipeline.py
  └── Fetches NASA POWER data → pvlib simulation
      Outputs: data/solar/Mathura_solar_15min.parquet
               data/solar/Bareilly_solar_15min.parquet

scripts/03_battery_grid_check.py   → Validates battery + grid models
scripts/04_environment_check.py    → Validates EnergyEnv (Gymnasium)
scripts/05_rule_based_check.py     → Evaluates rule-based controller
scripts/06_sizing_pipeline.py      → Runs grid search sizing
scripts/07_train_dqn.py            → Trains DQN policy (500k timesteps)
                                     Output: models/dqn_policy.zip
scripts/08_mpc_check.py            → Validates MPC controller
scripts/09_compare_controllers.py  → 3-way Rule/MPC/DQN comparison
scripts/10_nsga2_sizing.py         → Full NSGA-II sizing (pop=40, gen=50)
scripts/11_validate_sizing.py      → Cross-validates sizing candidates
scripts/12_live_deployment_demo.py → End-to-end demo
```

### Live Pipeline (API — per request)

```
POST /api/size  (iems_contractor_ui.html)
  1. build_household_and_solar()    ← profile_generator + pv_model
  2. GridModel (outage params)      ← derived from real meter data
  3. run_grid_search()   OR         ← depth="quick"  (48 combos)
     run_nsga2()                    ← depth="thorough" (Pareto front)
  4. top_recommendations()          ← fastest_payback / best_npv / best_backup
  5. run_validation()               ← spot-check under Rule/MPC/DQN
  6. full_financial_analysis()      ← CAPEX, NPV, IRR, payback
  → JSON response → Contractor UI
```

### Demo Pipeline (pipeline_demo.html — background stages)

```
POST /api/pipeline/start
  Stage 1: synthetic_load      ← profile_generator (no deps)
  Stage 2: solar_generation    ← pv_model parquet (no deps)
  Stage 3: grid_search         ← depends: [1, 2]
  Stage 4: ga (NSGA-II)        ← depends: [1, 2]
  Stage 5: rule_based_gs       ← depends: [3]  (Grid Search + Rule-Based)
  Stage 6: rule_based_ga       ← depends: [4]  (GA + Rule-Based)
  Stage 7: dqn                 ← depends: [4]  (GA + DQN)
  Stage 8: outage_proof        ← depends any_of: [5, 6, 7]  (Final Report)

GET /api/pipeline/status/{job_id}   ← poll per-stage status
GET /api/pipeline/result/{job_id}/{stage}  ← fetch chart/table data
```

---

## 4. Module Reference

### 4.1 `src/data/loader.py`
Loads 4 raw CEEW smart-meter CSV files (Mathura 2019/2020, Bareilly 2019/2020). Uses float32 + category dtypes for ~75% RAM reduction (~600 MB vs ~2 GB). Auto-derives city from filename. Exposes `load_single_file()`, `load_all_files()`, `save_parquet()`, `load_parquet()`.

### 4.2 `src/data/preprocessor.py`
5-step pipeline executed via `run_pipeline(df)`:
1. **validate()** — drops negative energy, clamps voltage outliers [50V–350V]
2. **add_power_kw()** — derives kW from kWh per 3-min slot
3. **resample_to_15min()** — 3-min → 15-min aggregation (sum energy, mean electrical)
4. **detect_outages()** — flags zero-energy daytime slots as outages; assigns run IDs
5. **add_time_features()** — hour, weekday, season (Indian meteorological), slot_of_day

Also provides `daily_summary()` for per-meter per-day aggregation.

### 4.3 `src/data/profile_generator.py`
Extracts representative load archetypes and synthesises profiles for new customers:
- `extract_profiles()` — 96-slot normalised daily shapes per meter × season × day-type
- `cluster_profiles()` — K-Means (K=5) on L2-normalised shape vectors → household archetypes (ultra-low / low / medium / high / very-high consumption)
- `generate_profile()` — picks closest cluster to target daily kWh, scales, tiles N days, adds 8% Gaussian noise
- `save_model()` / `load_model()` — persist fitted profiles + KMeans to `models/`

### 4.4 `src/solar/nasa_power.py`
REST client for NASA POWER API (community=RE). Fetches hourly GHI/DNI/DHI + temperature + wind per year. Converts UTC → IST (Asia/Kolkata). Year-level parquet caching avoids redundant API calls. Strict glob pattern prevents mixing raw hourly weather with processed 15-min PV output.

### 4.5 `src/solar/pv_model.py`
Physics chain using pvlib:
1. Solar position (apparent zenith/azimuth) via `Location.get_solarposition()`
2. Plane-of-Array irradiance via Perez model (`get_total_irradiance`)
3. Cell temperature via Faiman model
4. DC power via PVWatts equations (temperature derating)
5. AC power via inverter efficiency + system losses
6. Resample hourly → 15-min (forward-fill irradiance, linear interp temperature)

All pvlib inputs converted to numpy arrays before computation to avoid pandas index-alignment shape explosions. Default system: 3 kWp, tilt=27°, azimuth=180° (south), losses=14%.

### 4.6 `src/battery/battery_model.py`
Physics-based LFP battery model. Round-trip efficiency split symmetrically: η_charge = η_discharge = √RTE ≈ 0.9487 for RTE=0.90. Enforces SOC bounds [min_soc=10%, max_soc=95%]. Tracks cumulative cycles and State of Health (linear degradation to 80% over cycle_life=3000 cycles). Exposes `step(charge_kw, discharge_kw)` returning `BatteryStep` dataclass with all energy flows.

### 4.7 `src/grid/grid_model.py`
Models UP electricity grid connection:
- **Slab tariff billing** — DVVNL/PVVNL 2023-24 residential rates: ₹3.35→₹6.00/kWh across 5 slabs. `marginal_rate(cumulative_units)` returns the next-unit rate for per-step reward shaping. `bill_for_month(total_units)` computes full slab bill for reporting.
- **Net metering** — ₹2.50/kWh credit for exported solar
- **Outage model** — uses real detected outage calendar (`outage_lookup` Series) when available; falls back to deterministic statistical model keyed by hash(date, hour) for reproducibility. Supports monthly seasonal multipliers derived from real meter data.
- Static helpers: `build_outage_lookup()`, `build_representative_outage_profile()`

### 4.8 `src/environment/energy_env.py`
Gymnasium `EnergyEnv` tying all physical models together:

**Action space** — Discrete(4): idle / charge_from_solar / charge_from_grid / discharge_to_load

**Observation space** — Box(9): solar_kw_norm, load_kw_norm, battery_soc, hour_sin, hour_cos, is_weekend, cumulative_units_norm, is_outage, month_sin

**Reward** — `-(grid_import_kwh × marginal_rate) + (export_kwh × net_metering_rate) - (unmet_kwh × ₹15/kWh penalty)`

**Energy balance per step**: solar → load first (always), then action applied to battery, then remaining solar exported, remaining load from grid or unmet if outage.

**Multi-household pool** — `additional_households` parameter forces DQN to learn transferable policies across households rather than memorizing one rhythm. Fixed normalization constant (15 kW) ensures observation scale is portable across households.

### 4.9 `src/controllers/base_controller.py`
Abstract base defining `select_action(env) -> int` and `reset()` interface shared by all three controllers.

### 4.10 `src/controllers/rule_based.py`
Deterministic priority-logic controller (no training required):
1. Solar surplus → charge battery (action 1)
2. Solar deficit + battery available → discharge to load (action 3)
3. Otherwise → idle (action 0, grid covers remainder)

Never selects charge_from_grid (action 2) — that capability is reserved to show learned controllers outperforming this baseline by pre-charging ahead of anticipated outages.

### 4.11 `src/controllers/mpc_controller.py`
Model Predictive Control via cvxpy Linear Program over 24h rolling horizon:
- **Forecast** — persistence: tomorrow = yesterday's actuals (96-step lookback), avoids giving MPC perfect-foresight advantage over DQN
- **Problem** — minimizes (grid_import × rate - export × net_metering_rate + unmet × ₹15 penalty) × dt, subject to SOC dynamics, power limits, outage-masked grid flows
- **Solver** — CLARABEL preferred (ECOS fallback), problem compiled once per battery spec and reused via Parameters for fast repeated solves
- **Slab-tariff** — rate frozen at horizon start (acceptable approximation since single day rarely crosses slab boundary)
- **Action mapping** — continuous first-step decision mapped to discrete 4-action interface

### 4.12 `src/controllers/dqn_controller.py`
Stable-Baselines3 DQN with MlpPolicy (2 × 256 hidden layers):
- `train_dqn()` — trains for 500k timesteps, saves to `models/dqn_policy.zip`. Hyperparams: lr=3e-4, batch=64, buffer=100k, γ=0.99, exploration 100%→5% over first 20% of training.
- `DQNController` — loads trained policy, implements `select_action()` via deterministic greedy prediction.

### 4.13 `src/controllers/historical_forecast_provider.py`
Alternative forecast provider for MPC using historical averages instead of persistence. Plugged in via `forecast_provider` kwarg on `MPCController`.

### 4.14 `src/sizing/grid_search.py`
Brute-force sweep over 8 panel sizes × 6 battery sizes = 48 combinations:
- Each combo: full 365-day simulation under rule-based controller
- Parallel evaluation via `ProcessPoolExecutor` (auto-detects CPUs, capped at 8)
- `_score_one_combo()` is module-level (not closure) for pickling to worker processes
- `top_recommendations()` returns three named picks: fastest payback / best 25yr NPV / best backup coverage (lowest annual unmet load)
- Baseline computed directly via `simulate_system_size(panel=0, battery=0)` — never self-referentially through `run_grid_search` which would silently zero out all savings

### 4.15 `src/sizing/nsga2_sizing.py`
NSGA-II multi-objective sizing via pymoo:
- **Decision variables** — panel_kw [1–10], battery_kwh [0–20], tilt_deg [0–60], azimuth_deg [90–270] (continuous)
- **Objectives** — minimize {CAPEX, -NPV_25yr, payback_years, annual_unmet_load_kwh}
- **Population** — default pop=40, generations=50 (~2000 total evaluations)
- **Parallelization** — fork-based multiprocessing with `StarmapParallelization`
- **Solar cache** — per-worker dict keyed by rounded (tilt, azimuth) to avoid redundant pvlib calls
- **Solar curve** — `FixedSolarCurve` picklable class (not lambda/closure) for multiprocessing compatibility
- Final Pareto front re-simulated to compute `annual_savings_inr` (not approximated from NPV)

### 4.16 `src/sizing/validate_candidates.py`
Spot-check module that re-simulates top sizing candidates under all three controllers:
- `select_representative_candidates()` — picks best-NPV, best-backup (in competitive NPV half), and most-balanced (closest to Pareto centroid)
- `run_validation()` — simulates each candidate under Rule-Based, MPC, and (optionally) DQN for `episode_days` (default 180), annualizes results
- `diagnose_ranking_disagreement()` — pairwise flip detection with margin-aware severity (real disagreement vs near-tie)
- `resolve_recommendation()` — single bottom-line recommendation using MPC as tie-breaker, with explicit confidence level (high/medium/low)
- DQN fairness flag — warns when candidate battery_kwh deviates >35% from training spec

### 4.17 `src/report/financials.py`
Pure financial functions (no I/O, easily unit-testable):
- `capex()` — gross panel cost, PM Surya Ghar Muft Bijli Yojana subsidy (40% ≤2kWp, 20% 2–3kWp), battery + BoS costs
- `project_annual_savings()` — 25-year projection accounting for tariff inflation (5%/yr) and panel degradation (0.5%/yr)
- `npv()` — discounted at 8% (conservative Indian cost of capital)
- `irr()` — Newton-Raphson on NPV function
- `payback_period()` — fractional year via linear interpolation
- `full_financial_analysis()` — convenience wrapper returning `FinancialSummary` dataclass

### 4.18 `src/pipeline/orchestrator.py`
Async background job scheduler for the demo pipeline:
- In-memory job store (`_JOBS` dict) — no Redis/Celery needed at demo scale
- `STAGE_DEPENDENCIES` — DAG with AND semantics (plain list) and OR semantics (`{any_of: [...]}`)
- `run_pipeline_job()` — asyncio loop launching stages concurrently as dependencies are satisfied, reaping finished tasks, propagating failures as "blocked" to dependents
- `_transitive_dependents()` — propagates failure/block downstream immediately

### 4.19 `src/pipeline/stage_functions.py`
Async stage implementations called by the orchestrator. Each follows `(params, get_upstream) -> dict`:
- Heavy sync work offloaded to thread via `asyncio.to_thread()` to not block status-polling
- `stage_synthetic_load` — generates 400-day household load from profile model or synthetic fallback
- `stage_solar_generation` — loads cached 15-min PV parquet or synthetic fallback
- `stage_grid_search` — runs 48-combo sweep, picks best-backup candidate for controller tabs
- `stage_ga` — runs NSGA-II (pop=12, gen=8 for demo speed), picks best-backup candidate
- `stage_rule_based_gs/ga` — runs RuleBasedController on Grid Search / GA sized system
- `stage_dqn` — runs DQNController on GA-sized system
- `stage_outage_proof` — compares all three combos vs no-solar baseline, annualizes to 365 days
- `_run_controller_demo()` — full episode capture: detail trace (first 5 days at 15-min), daily aggregates, cost breakdown, outage coverage %

### 4.20 `app.py` — FastAPI Backend
Main application combining three routers:
- **`/api/size`** (POST) — contractor sizing endpoint. Accepts `SizingRequest` (city, monthly bill, depth, budget cap, subsidy). Returns three candidate JSONs (payback/NPV/backup) + validation result.
- **`/api/health`** (GET) — health check
- **`/api/pipeline/*`** — pipeline demo routes (via `pipeline_routes.py`)
- **`/api/ga-explainer/*`** — GA explainer routes (via `ga_explainer_routes.py`)

Key helpers in `app.py`:
- `derive_outage_params_from_real_data()` — derives real outage hours/peak-hours from processed meter data per city; cached module-level
- `compute_baseline_annual_cost()` — directly calls `simulate_system_size(panel=0, battery=0)` to avoid self-referential baseline trap
- `FixedSolarCurve` — picklable class wrapping pre-computed per-kWp solar curve (safe for multiprocessing)
- `build_household_and_solar()` — aligns synthetic load to solar calendar dates

### 4.21 `pipeline_routes.py`
FastAPI router (`/api/pipeline`):
- `POST /start` — creates job, fires `asyncio.create_task(run_pipeline_job(...))`, returns `job_id` immediately
- `GET /status/{job_id}` — lightweight per-stage status (no result payloads), safe to poll every 1–2s
- `GET /result/{job_id}/{stage_name}` — returns one stage's chart/table payload once status = "done"; strips non-serializable numpy arrays

### 4.22 `ga_explainer_routes.py`
Routes serving the `ga_explainer.html` interactive explainer for the NSGA-II genetic algorithm.

---

## 5. Data Flow Summary

```
CEEW CSVs
  └─ loader → preprocessor → profile_generator
                                    │
                                    ▼
                              KMeans archetypes
                              (models/load_profiles.parquet
                               models/profile_kmeans.pkl)

NASA POWER API
  └─ nasa_power → pv_model (pvlib physics)
                       │
                       ▼
                 data/solar/{city}_solar_15min.parquet

Runtime:
  monthly_bill_inr
    └─ generate_profile() → load_kw [35040 steps, 15-min]
    └─ load_solar_data()  → solar_kw_per_kwp [35040 steps]
    └─ GridModel(outage)  → tariff + outage calendar

  run_grid_search(load, solar, grid) → scored DataFrame
    OR
  run_nsga2(load, solar, grid)       → Pareto front DataFrame

  top_recommendations() → {payback, npv, backup} candidates

  run_validation(candidates, controllers) → ranking stability

  full_financial_analysis() → {CAPEX, NPV, IRR, payback, subsidy}
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| Rule-based controller drives sizing search | DQN/MPC evaluation per candidate is infeasible at 48–2000 iterations |
| Fixed observation normalization (15 kW) | Enables DQN policy portability across households; per-household normalization destroys transfer |
| Persistence forecast for MPC | Avoids giving MPC perfect-foresight advantage over DQN in the 3-way comparison |
| Module-level picklable classes for solar curves | Closures cannot be pickled by multiprocessing; confirmed bug in scripts then fixed |
| Baseline computed directly (not via grid_search) | Self-referential baseline silently zeroed all savings (confirmed production bug, fixed) |
| `any_of` dependency semantics in orchestrator | Final Report stage should run even if DQN fails (no trained model on fresh machine) |
| Slab tariff marginal_rate vs bill_for_month | Per-step reward shaping needs next-unit rate; monthly bill is for reporting only |

---

## 7. Directory Structure

```
iems/
├── app.py                        # FastAPI entry point + /api/size endpoint
├── pipeline_routes.py            # /api/pipeline/* routes
├── ga_explainer_routes.py        # /api/ga-explainer/* routes
├── iems_contractor_ui.html       # Contractor-facing sizing UI
├── pipeline_demo.html            # 7-tab pipeline demo UI
├── ga_explainer.html             # NSGA-II interactive explainer
├── configs/
│   └── default_config.yaml       # All system parameters (solar, battery, grid, DQN, MPC, sizing)
├── scripts/
│   ├── 01_data_pipeline.py       # Data ingestion + profile clustering
│   ├── 02_solar_pipeline.py      # NASA fetch + pvlib simulation
│   ├── 03_battery_grid_check.py  # Model validation
│   ├── 04_environment_check.py   # Gymnasium env check
│   ├── 05_rule_based_check.py    # Rule-based evaluation
│   ├── 06_sizing_pipeline.py     # Grid search sizing
│   ├── 07_train_dqn.py           # DQN training (500k steps)
│   ├── 08_mpc_check.py           # MPC validation
│   ├── 09_compare_controllers.py # 3-way controller comparison
│   ├── 10_nsga2_sizing.py        # Full NSGA-II sizing
│   ├── 11_validate_sizing.py     # Candidate cross-validation
│   └── 12_live_deployment_demo.py# End-to-end demo
├── src/
│   ├── data/
│   │   ├── loader.py             # CEEW CSV loader
│   │   ├── preprocessor.py       # 5-step preprocessing pipeline
│   │   └── profile_generator.py  # Load archetype clustering + synthesis
│   ├── solar/
│   │   ├── nasa_power.py         # NASA POWER API client + caching
│   │   └── pv_model.py           # pvlib physics chain → 15-min AC output
│   ├── battery/
│   │   └── battery_model.py      # LFP physics (SOC, RTE, degradation)
│   ├── grid/
│   │   └── grid_model.py         # UP slab tariffs, net metering, outage model
│   ├── environment/
│   │   └── energy_env.py         # Gymnasium EnergyEnv (4 actions, 9 obs)
│   ├── controllers/
│   │   ├── base_controller.py    # Abstract interface
│   │   ├── rule_based.py         # Priority-logic baseline
│   │   ├── mpc_controller.py     # cvxpy LP rolling-horizon MPC
│   │   ├── dqn_controller.py     # Stable-Baselines3 DQN train + infer
│   │   └── historical_forecast_provider.py  # Historical-avg MPC forecast
│   ├── sizing/
│   │   ├── grid_search.py        # 48-combo brute-force sweep
│   │   ├── nsga2_sizing.py       # pymoo NSGA-II Pareto optimization
│   │   └── validate_candidates.py# Cross-controller candidate validation
│   ├── report/
│   │   └── financials.py         # CAPEX, NPV, IRR, payback, subsidy
│   └── pipeline/
│       ├── orchestrator.py       # Async DAG job scheduler
│       └── stage_functions.py    # Per-stage async implementations
├── data/
│   ├── raw/                      # CEEW CSVs (not in repo)
│   ├── processed/                # meters_15min.parquet
│   └── solar/                    # {city}_solar_15min.parquet + raw weather cache
├── models/
│   ├── load_profiles.parquet     # Fitted load archetypes
│   ├── profile_kmeans.pkl        # Fitted KMeans model
│   └── dqn_policy.zip            # Trained DQN policy (SB3 format)
├── outputs/                      # Script outputs and plots
├── logs/                         # Loguru logs + DQN TensorBoard
└── requirements.txt
```

---

## 8. Tech Stack

| Layer | Libraries |
|---|---|
| API | FastAPI, uvicorn, pydantic |
| Physics | pvlib, numpy |
| ML / RL | stable-baselines3, gymnasium, scikit-learn |
| Optimization | pymoo (NSGA-II), cvxpy (MPC LP) |
| Data | pandas, pyarrow (parquet), requests |
| Logging | loguru |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
