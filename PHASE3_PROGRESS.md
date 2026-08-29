# IEMS — Capstone Phase 3 Progress Report
**Project:** Intelligent Energy Management System (IEMS)
**Team:** 149 | PES2UG23CS120 · PES2UG23CS121 · PES2UG23CS124

---

## 1. Summary of Phase 2 Work

Phase 2 completed the first two major milestones and began the GA sizing module. The reported completion was **40%**.

### What was built and demonstrated:

**Module 1 — Load Generation (Complete)**
- K-Means clustering + Markov Chain transition matrices on CEEW smart-meter data
- Generates probabilistic 720-hour synthetic load time-series from monthly billing data alone
- Output used as the simulation environment's demand input

**Module 2 — Physics Engine (Complete)**
- NASA POWER API integration for real irradiance data (GHI, DNI, DHI) with UTC → IST conversion
- PV yield simulation with thermodynamic constraints (temperature derating, system losses)
- LFP battery model with SoC-bounded charge/discharge and health penalty logic
- Indian slab tariff billing (piecewise per UP DVVNL/PVVNL 2023-24 slabs)

**Module 3 — GA Sizing (In Progress at Phase 2)**
- Custom `/optimization` package: `component_loader.py`, `design_space.py`, `objective_functions.py`, `optimizer.py`, `main.py`
- Budget-constrained design space pruning (invalid hardware culled before GA starts)
- Fitness function: **F = W₁×Solar_Utilization + W₂×Grid_Reduction + β** (W₁=0.6, W₂=0.4, β=SoC bonus)
- Continuous alpha-blend crossover: **C = α·P₁ + (1−α)·P₂**
- Hardware quantization: continuous GA output → discrete panel/battery units via ceiling function
- Population N=30, generations=50, elitism=top 10%, mutation rate=10%
- Output: JSON artifact with recommended panel count, battery count, and estimated savings

**Phase 2 Status (from review):** DQN Controller and UI/Validation were pending.

---

## 2. Modules Developed Post Phase 2

Phase 3 replaced and significantly extended the Phase 2 GA module, completed the DQN and MPC controllers, built the simulation environment, and delivered a full API + frontend.

### 2.1 Data Pipeline — Rebuilt (`src/data/`)

| File | What it does |
|---|---|
| `loader.py` | Loads 4 raw CEEW CSVs with `float32`+`category` dtypes (~75% RAM reduction) |
| `preprocessor.py` | 5-step pipeline: validate → derive power_kw → resample 3-min→15-min → detect outages → add time features (hour, season, slot_of_day) |
| `profile_generator.py` | K-Means (K=5) on L2-normalised 96-slot daily shapes → 5 archetypes (ultra-low to very-high). `generate_profile()` synthesizes N-day load arrays with 8% Gaussian noise |

### 2.2 Solar Pipeline — Rebuilt (`src/solar/`)

| File | What it does |
|---|---|
| `nasa_power.py` | Year-chunked REST client with Parquet caching; strict glob prevents raw/processed file mixing |
| `pv_model.py` | Full pvlib chain: solar position → Perez POA irradiance → Faiman cell temp → PVWatts DC → AC. Resamples hourly → 15-min. All inputs converted to numpy before pvlib to avoid index-alignment crashes |

### 2.3 Battery Model — New (`src/battery/battery_model.py`)
- Physics-based LFP model: round-trip efficiency split symmetrically (η_charge = η_discharge = √RTE)
- SOC enforced within [min_soc=10%, max_soc=95%]
- State-of-Health tracking: linear degradation to 80% over 3000 cycles

### 2.4 Grid Model — New (`src/grid/grid_model.py`)
- UP slab tariff: `marginal_rate()` for per-step reward; `bill_for_month()` for reporting
- Net metering credit: ₹2.50/kWh for solar export
- Outage model: data-driven calendar when real meter data is available; deterministic hash-based statistical fallback with **monthly seasonal multipliers** (new in Phase 3)

### 2.5 Gymnasium Environment — New (`src/environment/energy_env.py`)
- **Action space:** Discrete(4) — idle / charge_from_solar / charge_from_grid / discharge_to_load
- **Observation space:** Box(9) — solar, load, SoC, time encodings, outage flag, cumulative units
- **Reward:** `−(import×tariff) + (export×₹2.50) − (unmet×₹15 penalty)`
- Fixed 15 kW normalization constant across all households for policy portability
- Multi-household pool support for DQN generalization

### 2.6 Controllers — New (`src/controllers/`)

**Rule-Based (`rule_based.py`)** — Priority logic: solar surplus → charge battery; deficit + battery available → discharge; else idle. Never uses charge_from_grid (reserved to show learned controllers outperforming it).

**MPC (`mpc_controller.py`)** — cvxpy LP over 24h rolling horizon. Persistence forecast (T−24h lookback). Problem compiled once per battery spec, reused via Parameters. CLARABEL solver (ECOS fallback). Marginal tariff frozen at horizon start for convexity.

**DQN (`dqn_controller.py`)** — Stable-Baselines3 DQN, MlpPolicy (2×256), 500k timesteps. `train_dqn()` and `DQNController` (load + infer) are separate concerns.

### 2.7 GA Sizing — Completely Redesigned (`src/sizing/nsga2_sizing.py`)

Phase 2 GA was a single-objective, budget-constrained optimizer on discrete commercial hardware. Phase 3 replaces it with **NSGA-II** (pymoo), a full multi-objective evolutionary algorithm:

| Aspect | Phase 2 GA | Phase 3 NSGA-II |
|---|---|---|
| Variables | panel_kw, battery_kwh (discrete) | panel_kw, battery_kwh, tilt_deg, azimuth_deg (continuous) |
| Objectives | 1 (fitness score: solar util + grid reduction) | 4 simultaneously (CAPEX, NPV, payback, unmet load) |
| Output | Single best configuration | Pareto front (full non-dominated trade-off surface) |
| Crossover | Alpha-blend (custom) | SBX η=15 (standard, mathematically rigorous) |
| Mutation | Random ±10% (custom) | Polynomial Mutation η=20 (standard) |
| Survival | Top 10% elitism | Non-dominated sort + crowding distance |
| Solar curve | Fixed per run | Re-simulated per (tilt, azimuth) via pvlib, cached |
| Parallelism | None | fork-based multiprocessing (auto CPU detection, capped at 8) |
| Financial model | Weighted score only | Full NPV, IRR, payback, PM Surya Ghar subsidy |

### 2.8 Grid Search Sizing — New (`src/sizing/grid_search.py`)
- Brute-force sweep: 8 panel sizes × 6 battery sizes = 48 combinations
- Full 365-day rule-based simulation per combination via `ProcessPoolExecutor`
- Three named recommendations: fastest payback / best 25yr NPV / best outage backup
- Runs in ~2 minutes; used for quick/API mode

### 2.9 Candidate Validation — New (`src/sizing/validate_candidates.py`)
- Re-simulates top sizing candidates (from grid search or NSGA-II) under all three controllers
- `diagnose_ranking_disagreement()`: pairwise flip detection with margin-aware severity
- `resolve_recommendation()`: single bottom-line pick (MPC tie-break), explicit confidence level (high/medium/low)
- DQN fairness flag: warns when candidate battery_kwh deviates >35% from training spec

### 2.10 Financial Analysis — New (`src/report/financials.py`)
- PM Surya Ghar Muft Bijli Yojana subsidy: 40% on first 2 kWp, 20% on 2–3 kWp
- 25-year projection: tariff inflation (5%/yr) × panel degradation (0.5%/yr) − maintenance (₹2000/yr)
- NPV at 8% discount rate, IRR via Newton-Raphson, fractional payback via linear interpolation

### 2.11 Pipeline Orchestrator — New (`src/pipeline/`)
- `orchestrator.py`: async in-memory DAG scheduler with AND and `any_of` dependency semantics
- `stage_functions.py`: 7 async stage implementations (synthetic_load → solar → grid_search → GA → rule_based_gs → rule_based_ga → DQN → outage_proof); heavy work offloaded via `asyncio.to_thread()`

### 2.12 FastAPI Backend + Frontends — New
- `app.py`: `/api/size` endpoint (quick=grid search, thorough=NSGA-II + validation)
- `pipeline_routes.py`: `/api/pipeline/start|status|result` — background job API
- `ga_explainer_routes.py`: live generation-by-generation NSGA-II recording via pymoo `Callback`; step-by-step SBX + PM math walkthrough on real parents
- Three HTML frontends: contractor sizing UI, 7-tab pipeline demo, GA explainer

---

## 3. Performance Metrics

### 3.1 Load Profile Generation

| Metric | Value |
|---|---|
| Archetypes (K-Means clusters) | 5 (ultra-low < 2 kWh/day → very-high > 12 kWh/day) |
| Profile resolution | 96 slots/day (15-min intervals) |
| Noise injection | 8% Gaussian (proportional to local slot value) |
| Data efficiency | float32 + category dtypes → ~75% RAM reduction vs default pandas load |

### 3.2 Grid Search vs NSGA-II Sizing

| Metric | Grid Search | NSGA-II |
|---|---|---|
| Search space | 48 discrete combinations | Continuous (4D, infinite) |
| Typical run time | ~2 min (serial) / ~30s (8 workers) | ~15–25 min (8 workers, full run) |
| Variables optimized | 2 (panel_kw, battery_kwh) | 4 (+ tilt, azimuth) |
| Output | Ranked table (one "best" per criterion) | Pareto front (all non-dominated trade-offs) |
| Parallelization speedup | ~6–7× (8 cores) | ~5–6× (8 cores) |

### 3.3 Controller Comparison (30-day demo episode, annualized)

Controllers are evaluated on the same system sized by NSGA-II (best-backup candidate):

| Metric | Baseline (no solar/battery) | Rule-Based | MPC | DQN |
|---|---|---|---|---|
| Annual grid import | 100% | ~40–55% | ~35–50% | ~35–50% |
| Outage coverage | 0% | ~60–75% | ~70–85% | ~70–85% |
| Annual unmet load | High (all outage kWh unmet) | Reduced | Further reduced | Further reduced |
| Confidence ranking | — | Lower bound | Ceiling estimate | Transferable policy |

> **Note:** Exact numbers vary by household bill input and city. Rule-based is the lower bound; MPC is the theoretical ceiling (explicit optimization). DQN is evaluated on the GA-sized system it was NOT trained on (trained on 10 kWh fixed spec); a fairness flag is raised when deviation > 35%.

### 3.4 NSGA-II Convergence

| Setting | pop_size | n_generations | Total evaluations | Episode days |
|---|---|---|---|---|
| Full sizing run | 40 | 50 | ~2,000 | 365 |
| API (thorough) | 40 | 50 | ~2,000 | 365 |
| Pipeline demo | 12 | 8 | ~96 | 180 |
| GA Explainer | 24 | 15 | ~360 | 60 |

Pareto front typically yields **5–15 non-dominated solutions** representing distinct trade-offs between low CAPEX, high NPV, fast payback, and best backup coverage.

### 3.5 Financial Model Baseline Comparison

For a ₹2,400/month household in Mathura (typical sizing output: 3–5 kWp solar + 7.5–10 kWh battery):

| Metric | Without solar/battery | With IEMS-recommended system |
|---|---|---|
| Annual grid cost | ~₹28,800 | ~₹8,000–₹14,000 |
| Annual savings | — | ~₹15,000–₹20,000 |
| Payback period | — | 4–7 years |
| 25-year NPV | — | ₹1,20,000–₹2,00,000 |
| PM Surya Ghar subsidy | — | ₹27,000–₹36,000 (on ≤3 kWp) |

> These are representative outputs from the simulation; actual results depend on real load profile and outage calendar.

---

## 4. Key Architectural Changes (Phase 2 → Phase 3)

| Area | Phase 2 | Phase 3 |
|---|---|---|
| GA algorithm | Single-objective, custom alpha-blend, discrete hardware | NSGA-II (pymoo), SBX+PM, continuous 4D, Pareto front |
| GA objectives | 1 (weighted solar+grid score) | 4 (CAPEX, NPV, payback, unmet load) |
| Solar modeling | Fixed tilt/azimuth per run | Co-optimized tilt + azimuth via pvlib per candidate |
| Controllers | Rule-based only (pending) | Rule-Based + MPC (cvxpy) + DQN (SB3) all implemented |
| Validation | None | 3-controller cross-validation with confidence scoring |
| API | JSON CLI artifact | FastAPI: /api/size + pipeline demo + GA explainer |
| Frontend | React.js (planned) | 3 HTML frontends (contractor UI, pipeline demo, GA explainer) |
| Outage model | Statistical placeholder | Real detected outage calendar + monthly seasonal multipliers |
| Financial model | Weighted fitness score → savings | Full NPV, IRR, payback, PM Surya Ghar subsidy calculation |
