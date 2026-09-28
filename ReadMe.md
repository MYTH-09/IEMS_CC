# IEMS — Intelligent Energy Management Simulator

A decision-support platform that helps solar contractors size and validate residential solar + battery systems for Indian households — grounded in real smart-meter data, real satellite irradiance data, and real grid outage patterns, rather than generic industry averages.

This README is written as a complete rebuild guide: if the codebase were lost, this document plus the referenced data sources should be enough to reconstruct the system's architecture and reasoning from scratch.

---

## 1. The Problem This Solves

Existing solar calculators (e.g. manufacturer lead-gen tools) estimate savings from a single flat electricity rate and a generic "peak sun hours" constant — no battery sizing, no outage modeling, no location-specific physics. For India specifically, two things make this inadequate:

1. **Electricity billing is slab-based**, not flat-rate — the cost of the next unit depends on how much has already been consumed that month.
2. **Grid outages (load-shedding) are common and location-specific** — a battery's real value depends heavily on how often and when the grid actually fails, which no generic tool accounts for.

IEMS addresses both by simulating a full year of a specific household's energy behavior at 15-minute resolution, using real historical data wherever possible, and by testing whether its sizing recommendation actually holds up under genuinely intelligent battery control — not just a simplified proxy.

---

## 2. Architecture — End to End

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                       │
│  Real CEEW smart-meter data (Mathura + Bareilly, 2019-2021)       │
│    → cleaned, resampled to 15-min, outages detected               │
│    → clustered into 5 household archetypes                        │
│    → synthetic year generated for any target daily kWh            │
│                                                                     │
│  NASA POWER API (satellite irradiance, cached locally)             │
│    → pvlib PVWatts physics chain → 15-min solar generation curve   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHYSICS LAYER                                                    │
│  Battery model — SOC tracking, round-trip efficiency, degradation  │
│  Grid model — real UP DVVNL slab tariffs + real/statistical outage │
│               patterns per city                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SIMULATION ENVIRONMENT (Gymnasium-compatible)                    │
│  At every 15-min step: observes solar/load/SOC/outage/tariff state,│
│  accepts a control action, updates battery + grid, returns reward  │
│  denominated in ₹ (slab-tariff-aware)                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  CONTROLLERS (shared interface — interchangeable)                  │
│  Rule-based  — fixed priority logic (search baseline)               │
│  MPC         — cvxpy rolling 24h-horizon optimization                │
│  DQN         — Stable-Baselines3, trained across varied households   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SIZING                                                            │
│  Grid search  — 48 fixed commercial panel/battery combos,           │
│                 ranked 3 ways (payback / NPV / backup)               │
│  NSGA-II      — continuous multi-objective search (panel, battery,   │
│                 tilt, azimuth) → Pareto front                        │
│  Both use rule-based control internally (cost-tractability),        │
│  then top candidates are spot-checked under MPC + DQN                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION                                                        │
│  Re-simulates the top 2-3 sizing candidates under rule-based, MPC,  │
│  and DQN. Checks whether the ranking survives smarter control.      │
│  Produces a genuine confidence label (high/medium/low), not a       │
│  hardcoded claim.                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                                 │
│  FastAPI backend (app.py) — exposes the pipeline as a JSON API       │
│  Custom HTML/JS frontend — contractor-facing input + report UI       │
│  (Not Streamlit — the original build spec named Streamlit, but the   │
│   designed UI needed exact layout/typography control Streamlit       │
│   can't provide without fighting the framework.)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Why Certain Design Decisions Were Made

### Why simulation instead of a formula
A flat "kWp × peak sun hours" estimate can't capture panel temperature derating, seasonal irradiance variation, a specific household's actual usage timing, or when a specific city's grid actually fails. Simulating a real year at 15-minute resolution captures all of these; a formula cannot.

### Why the sizing search uses rule-based control internally, not DQN or MPC
Grid search evaluates 48 candidates; NSGA-II evaluates 500–2000+. Training a DQN or running MPC's per-step optimization that many times is computationally prohibitive. Rule-based control is cheap and deterministic, making the search itself tractable. This is a stated, deliberate simplification — not an oversight — and its output should be read as a **lower-bound estimate**: a smarter controller should only do as well or better on the same hardware, never worse.

### Why a separate validation step exists
Because the above simplification means grid search's *ranking between candidates* is unverified — it's possible a smaller, cheaper battery under MPC/DQN control outperforms a larger battery under rule-based control. The validation step (`validate_candidates.py`) re-simulates only the top 2-3 candidates under all three controllers and checks whether the ranking survives. If it does, the recommendation is trustworthy. If controllers disagree, the system produces a single recommendation anyway (using MPC as the tie-breaker, since it solves an explicit optimization rather than following a heuristic or an approximate learned policy) but labels the confidence as **low**, with full disagreement detail preserved rather than hidden.

### Why DQN and MPC are compared as a headline result
This is the project's primary technical contribution: given an already-chosen, fixed hardware spec, how much does control *quality* matter? Real validated result on one household: MPC improved annual reward by +284% over rule-based; DQN improved it by +23%. MPC's 24-hour lookahead lets it anticipate outages; DQN only ever sees the current instant and relies on learned pattern-matching from training. MPC pays for this advantage in compute — roughly 80–90ms per decision vs. DQN's ~1ms — a real, honestly-stated tradeoff for a live deployment.

### Why UP slab tariffs specifically, not time-of-use
Most published home-energy-management research assumes time-of-use pricing and reliable grid supply — both wrong assumptions for Uttar Pradesh. The reward function calls `GridModel.marginal_rate(cumulative_units_this_month)` at every step, which correctly reflects that the cost of the next unit depends on month-to-date consumption, not time of day. This was validated by checking that accumulated step-wise marginal cost matches an independently-computed slab bill for the same import trajectory (confirmed to within 0.01% relative error).

---

## 4. Module Reference

### Module 1 — Data Pipeline (`src/data/`)
- `loader.py` — loads the 4 raw CEEW smart-meter CSVs (Mathura 2019–2020, Bareilly 2020–2021), renames columns, tags city/year from filename.
- `preprocessor.py` — validates data, resamples 3-min → 15-min, **detects outages** (daytime near-zero energy, or nighttime near-zero energy + low voltage), adds time features (season, weekday, etc.).
- `profile_generator.py` — extracts normalized 96-slot daily load shapes per household, clusters into 5 archetypes via K-Means, and generates a synthetic year for any target daily kWh by matching to the closest archetype and scaling.

**Real validated result:** ~21.5% of measured time slots are outages (~970 hrs/meter/year) — this is a genuine, expected finding for these UP districts, not a data bug.

### Module 2a — Solar Generation (`src/solar/`)
- `nasa_power.py` — fetches hourly GHI/DNI/DHI/temperature/wind from NASA POWER, caches per city per year as parquet (zero repeat API calls once cached).
- `pv_model.py` — runs pvlib's full physics chain (solar position → POA irradiance via Perez model → Faiman cell temperature → PVWatts DC/AC power) to produce 15-minute AC power output.

**Real validated result:** Mathura and Bareilly both land at 17.3–17.6% capacity factor for a 3kWp system — within the expected 16–22% range for Indian rooftop solar.

### Module 3a — Battery Model (`src/battery/battery_model.py`)
SOC tracking with round-trip efficiency split equally between charge/discharge directions (`η = √RTE`), hardware rate limits, min/max SOC bounds, and a linear degradation model.

**Real validated result:** starting from empty, charging 2kW then discharging 2kW for 1 hour each yields exactly 10% round-trip loss for a configured 90% RTE — confirmed to the decimal.

### Module 3b — Grid Model (`src/grid/grid_model.py`)
Real UP DVVNL/PVVNL slab tariffs (₹3.35 → ₹6.00/kWh across 5 slabs), net metering credit, and outage awareness. `is_outage(timestamp)` uses a real per-timestamp lookup built from Module 1's detected outages when available, falling back to a deterministic statistical model (same timestamp always returns the same result) for dates outside the historical range.

### Module 3c — Simulation Environment (`src/environment/energy_env.py`)
Gymnasium-compatible environment tying load + solar + battery + grid together. 4 discrete actions (idle / charge-from-solar / charge-from-grid / discharge-to-load). Reward is denominated in real ₹, using the grid's slab-aware marginal rate. Supports a **household pool** — training rotates through multiple synthetic households with a **fixed observation normalization constant** (critical: an earlier version normalized against each household's own max value, which meant a trained policy's learned weights became meaningless when evaluated on a different household — fixed by using one constant, 15kW, across every household).

**Real validated result:** energy conservation holds to exactly zero violation across all 4 actions over a full episode; accumulated step-wise reward matches an independently-computed slab bill to within 0.01% relative error.

### Module 3d — Controllers (`src/controllers/`)
- `rule_based.py` — fixed priority: solar→load, solar→battery, battery→load, else grid. Never uses charge-from-grid (reserved as a capability only smarter controllers can discover is sometimes worth using).
- `mpc_controller.py` — cvxpy rolling 24-hour horizon LP, re-solved every step (persistence forecasting — "tomorrow looks like yesterday," to keep the comparison with DQN fair, since DQN also never sees true future data). Includes a **curtailment variable** for excess solar that can't be stored or exported during an outage — an earlier version lacked this and produced genuine (not just numerical) infeasibility at specific timesteps.
- `dqn_controller.py` — Stable-Baselines3 DQN, trained across a rotating pool of synthetic households (500k timesteps) so the policy generalizes rather than memorizing one household's rhythm.

### Module 2b — Sizing (`src/sizing/`)
- `grid_search.py` — sweeps 8 panel sizes × 6 battery sizes = 48 fixed commercial combinations, scores each via full-year simulation + `financials.py`, returns three ranked views (fastest payback / best NPV / best backup) rather than one forced answer. Supports multiprocessing (`n_workers`) since each combination's simulation is fully independent.
- `nsga2_sizing.py` — NSGA-II multi-objective search over continuous panel kW, battery kWh, tilt, azimuth, optimizing CAPEX/NPV/payback/unmet-load simultaneously, producing a Pareto front. Also supports multiprocessing (closures had to be replaced with picklable classes for this to work — a real bug caught and fixed during development).
- `validate_candidates.py` — the spot-check layer described in §3 above. Includes margin-aware disagreement detection (a rank flip only matters if the underlying value gap exceeds a threshold — otherwise it's a near-tie, not a real disagreement) and a defined resolution policy that always produces one recommendation with an honest confidence label.

### Module 4 — Financials (`src/report/financials.py`)
CAPEX with PM Surya Ghar Muft Bijli Yojana subsidy (40% up to 2kW, 20% for 2–3kW, applied to panel cost only), 25-year NPV with tariff inflation and panel degradation, IRR via Newton-Raphson, payback period.

**Known gap:** the scheme's 300 free monthly units for adopters is not yet modeled — current savings figures are a slight underestimate, not overestimate.

### Application Layer (`app.py` + frontend)
FastAPI backend exposing `/api/size` — takes a customer's monthly bill, city, budget cap, subsidy eligibility, and search depth, and returns three sized candidates plus a real validation-backed confidence label. Grid outage parameters are derived from **real measured data** per city (`derive_outage_params_from_real_data`), not a generic hardcoded guess — this was a real gap caught and fixed during development (the app was silently using the same 4.0 hrs/day assumption for every city regardless of what was actually measured).

Frontend is a single-file HTML/JS interface (not Streamlit — see §3), contractor-facing, with tabbed candidate comparison, an interactive Chart.js visualization, and PDF export as the only report-sharing option (WhatsApp/email deliberately excluded for a B2B tool).

---

## 5. Known Limitations — Stated Honestly

These are real, current gaps, not hidden:

1. **"Thorough" sizing depth in the live API is a deliberately shrunk NSGA-II** (smaller population/generations, shorter episode length) compared to the full, validated version run via the standalone script — because a live HTTP request shouldn't block for the 1.5–2 hours a full run takes. A production version should run this as an async background job.
2. **Shading is not modeled at all.** No tool in this space (including commercial competitors) does this well without full 3D roof geometry; a scoped future extension would ask for obstruction height + distance + direction per side of the roof and compute geometric shading using the sun-position data already computed by pvlib.
3. **Only Mathura and Bareilly have real measured outage data.** Any other location falls back to a statistical approximation.
4. **DQN was trained on one fixed hardware spec** (10kWh battery, 3kW rate). Validating a candidate with a very different battery size is flagged with an explicit fairness warning, since the policy never experienced that hardware during training.
5. **Roof area is not currently a hard constraint** on sizing — a system could be recommended that wouldn't physically fit.
6. **Sizing's ranking is fundamentally proxy-based** (rule-based control internally) — the validation layer confirms whether it holds, but does not eliminate the underlying dependency.

---

## 6. Setup — Rebuilding From Scratch

### Prerequisites
- Python 3.10+
- The 4 raw CEEW smart-meter CSVs (Kaggle: `pythonafroz/electricity-smart-meter-data-from-india`), placed in `data/raw/`

### Pipeline order (each step depends on the previous)
```bash
pip install -r requirements.txt

python scripts/01_data_pipeline.py       # → models/load_profiles.parquet, profile_kmeans.pkl
python scripts/02_solar_pipeline.py      # → data/solar/{city}_solar_15min.parquet (NASA POWER, cached)
python scripts/03_battery_grid_check.py  # validates battery + grid physics
python scripts/04_environment_check.py   # validates the Gym environment
python scripts/05_rule_based_check.py    # validates the baseline controller
python scripts/06_sizing_pipeline.py     # grid search demo run
python scripts/07_train_dqn.py --full    # trains DQN (~6 min on modern CPU)
python scripts/08_mpc_check.py           # validates MPC controller
python scripts/09_compare_controllers.py # 3-way comparison, headline result
python scripts/10_nsga2_sizing.py --real-solar --city Mathura --full
python scripts/11_validate_sizing.py --city Mathura
```

### Running the live application
```bash
uvicorn app:app --reload --port 8000
```
Then open the frontend HTML file directly in a browser (no build step — it's a single self-contained file that calls the API via `fetch`).

---

## 7. Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy, scikit-learn |
| Solar physics | pvlib, NASA POWER API |
| RL environment | Gymnasium |
| DQN | Stable-Baselines3, PyTorch |
| MPC | cvxpy (CLARABEL solver) |
| Multi-objective sizing | pymoo (NSGA-II) |
| Backend | FastAPI, uvicorn |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
| Financial modeling | Custom (NPV/IRR/payback in `financials.py`) |

---

## 8. Headline Validated Results (One Real Household, Mathura)

- Capacity factor: 17.3% (within expected 16–22% range for Indian rooftop solar)
- Battery round-trip loss: exactly 10% for configured 90% RTE
- Reward-vs-bill consistency: 0.01% relative error
- **MPC vs. rule-based:** +283.9% net reward improvement
- **DQN vs. rule-based:** +23.0% net reward improvement
- **Sizing validation:** all three controllers (rule-based, MPC, DQN) agreed on the same candidate ranking for a tested household — high confidence

### Phase 3 DQN Generalization Follow-up

After fixing the MPC oracle-forecast leak, adding persistence-forecast features to the DQN observation, and retuning the DQN training budget and hyperparameters, DQN outperformed MPC across three held-out unseen households: average improvement was +20.2% versus rule-based, compared with MPC's +3.7%. This used the best-checkpoint model selected by EvalCallback rather than the regressed final 1.5M-step model. Caveat: the checkpoint was still selected against the seed=999 household that overlaps with `09_compare_controllers.py`'s reporting household; a cleaner retrain with a separate validation household is documented follow-up work and has not yet been done.