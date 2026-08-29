# How the Genetic Algorithm (NSGA-II) Works in IEMS

---

## 1. Why a Genetic Algorithm?

The basic sizing approach is a **Grid Search** (`src/sizing/grid_search.py`) that sweeps 48 fixed combinations of commercial panel and battery sizes (8 × 6). It is fast, exact, and maps to purchasable hardware — but it has two hard limitations:

| Grid Search Limitation | NSGA-II Solution |
|---|---|
| Discrete panel/battery sizes only | Continuous search over any value in a real range |
| Cannot vary panel tilt or azimuth | Optimizes all four variables simultaneously |
| Ranks by one metric at a time | Returns the full Pareto-optimal trade-off surface |
| 48 evaluations total | Thousands of evaluations, converging on non-dominated solutions |

NSGA-II (**Non-dominated Sorting Genetic Algorithm II**, Deb et al., 2002) is the algorithm of choice because the sizing problem has **four competing objectives** — lower cost, higher NPV, faster payback, and better outage coverage — that cannot all be maximized simultaneously. NSGA-II produces a **Pareto front**: every solution on it is the best possible trade-off for its position; no other solution is better on every objective at once.

---

## 2. The Four Decision Variables

Every candidate system in NSGA-II is a point in this **4-dimensional continuous search space**:

```
x = [panel_kw,  battery_kwh,  tilt_deg,   azimuth_deg]
     [1.0–10.0]  [0.0–20.0]   [0.0–60.0]  [90.0–270.0]
```

| Variable | Range | Physical Meaning |
|---|---|---|
| `panel_kw` | 1–10 kWp | Installed solar panel capacity |
| `battery_kwh` | 0–20 kWh | Battery storage capacity |
| `tilt_deg` | 0°–60° | Panel inclination from horizontal (optimal ≈ latitude ≈ 27° for Mathura) |
| `azimuth_deg` | 90°–270° | Panel facing direction (90=East, 180=South, 270=West) |

The grid search only varies panel_kw and battery_kwh on a fixed grid. NSGA-II varies all four **continuously** — meaning the optimizer can discover, for example, that a slightly west-facing panel at 31° tilt with 7.3 kWh battery outperforms every grid-search candidate for a specific objective combination.

---

## 3. The Four Objectives (Fitness Functions)

pymoo minimizes everything, so NPV is negated:

```python
# From src/sizing/nsga2_sizing.py — SizingProblem._evaluate()
out["F"] = [
    financials.total_capex_inr,   # Objective 1: minimize CAPEX (₹)
    -npv,                          # Objective 2: minimize -NPV = maximize NPV (₹)
    payback,                       # Objective 3: minimize payback period (years)
    result["annual_unmet_load_kwh"], # Objective 4: minimize unmet load (kWh/yr = maximize backup)
]
```

Every time NSGA-II evaluates one candidate `x`, it:
1. Calls `solar_curve_fn(tilt_deg, azimuth_deg)` → gets a 35,040-point (15-min × 1yr) solar generation curve
2. Runs `simulate_system_size()` → a full year-long rule-based simulation
3. Computes `full_financial_analysis()` → CAPEX, NPV, IRR, payback
4. Returns the 4 objective values

This is expensive (~1–2 seconds per candidate), which is why the fitness function uses the **rule-based controller** (not MPC or DQN). Evaluating MPC or training DQN per candidate across thousands of evaluations is computationally infeasible.

---

## 4. The NSGA-II Algorithm — Step by Step

### 4.1 Initialization

```
Population = FloatRandomSampling()
             → pop_size random candidates uniformly sampled within bounds
             → default: pop_size = 40 (full run) or 12 (demo)
```

Each individual is a 4-element vector `[panel_kw, battery_kwh, tilt_deg, azimuth_deg]` randomly drawn within the bounds. All 40 candidates are **evaluated** (fitness computed) before evolution begins.

### 4.2 Main Loop — Per Generation

Each generation repeats this cycle:

```
┌─────────────────────────────────────────────────────────────┐
│  1. SELECTION          Randomly pick parent pairs           │
│  2. CROSSOVER (SBX)    Combine parents → offspring          │
│  3. MUTATION (PM)      Perturb offspring                    │
│  4. EVALUATION         Simulate each offspring (fitness)    │
│  5. SURVIVAL           Non-dominated sort + crowding dist   │
│  6. NEXT GENERATION    Keep best pop_size individuals       │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Crossover — Simulated Binary Crossover (SBX)

SBX mimics single-point binary crossover in continuous space. For each gene (variable) of two parents P₁ and P₂, it produces one child C using:

```
Draw u ~ Uniform(0, 1)

If u ≤ 0.5:
    β = (2u)^(1/(η+1))
    C = 0.5 × [(1 + β)×P₁ + (1 - β)×P₂]

If u > 0.5:
    β = (1 / (2(1-u)))^(1/(η+1))
    C = 0.5 × [(1 - β)×P₁ + (1 + β)×P₂]

C = clip(C, lower_bound, upper_bound)
```

**Parameters used in IEMS:**
- `eta = 15` (distribution index — higher = children stay closer to parents)
- `prob = 0.9` (crossover probability per pair)

**Example (from the GA Explainer real walkthrough):**

```
Gene: panel_kw (bounds: [1.0, 10.0])
  Parent A = 3.2 kWp,  Parent B = 7.8 kWp
  Draw u = 0.34  → u ≤ 0.5 branch
  β = (2 × 0.34)^(1/16) = 0.68^0.0625 = 0.9763
  Child = 0.5 × [(1 + 0.9763)×3.2 + (1 - 0.9763)×7.8]
        = 0.5 × [1.9763×3.2 + 0.0237×7.8]
        = 0.5 × [6.324 + 0.185] = 3.25 kWp

Gene: tilt_deg (bounds: [0.0, 60.0])
  Parent A = 22.0°,  Parent B = 35.0°
  Draw u = 0.67  → u > 0.5 branch
  β = (1/(2×0.33))^(1/16) = 1.515^0.0625 = 1.026
  Child = 0.5 × [(1 - 1.026)×22 + (1 + 1.026)×35]
        = 0.5 × [-0.572 + 70.91] = 35.17°  → clipped to [0, 60] → 35.2°
```

The key property: when β < 1, the child lies **between** the parents; when β > 1, it **extrapolates beyond** them. This allows exploration beyond the current population's range while mostly exploiting the region between good parents.

### 4.4 Mutation — Polynomial Mutation (PM)

PM perturbs a gene slightly. For a gene value `x` with bounds `[xl, xu]`:

```
Draw u ~ Uniform(0, 1)

If u < 0.5:
    δ = (2u)^(1/(η+1)) - 1          ← δ ∈ [-1, 0)
Else:
    δ = 1 - (2(1-u))^(1/(η+1))      ← δ ∈ [0, 1)

Child = x + δ × (xu - xl)
Child = clip(Child, xl, xu)
```

**Parameters used in IEMS:**
- `eta = 20` (distribution index — higher = mutation stays close to parent)

**Example:**

```
Gene: battery_kwh  (x = 10.0 kWh, bounds: [0, 20])
  Draw u = 0.31  → u < 0.5 branch
  δ = (2 × 0.31)^(1/21) - 1 = 0.62^0.0476 - 1 = 0.977 - 1 = -0.023
  Child = 10.0 + (-0.023) × (20 - 0) = 10.0 - 0.46 = 9.54 kWh

Gene: azimuth_deg  (x = 185.0°, bounds: [90, 270])
  Draw u = 0.72  → u ≥ 0.5 branch
  δ = 1 - (2 × 0.28)^(1/21) = 1 - 0.56^0.0476 = 1 - 0.971 = 0.029
  Child = 185.0 + 0.029 × (270 - 90) = 185.0 + 5.22 = 190.2°
```

The mutation is deliberately **small** (high `eta`) to refine candidates rather than randomly jump around the search space.

### 4.5 Non-Dominated Sorting (The Core of NSGA-II)

After evaluation, the combined parent + offspring pool (size = 2 × pop_size) is sorted into **Pareto ranks**:

```
Rank 0 (Pareto front): solutions NOT dominated by any other
        → no other solution is better on ALL four objectives simultaneously

Rank 1: dominated only by rank-0 solutions
Rank 2: dominated only by rank-0 or rank-1 solutions
...
```

**Domination rule:** Candidate A dominates candidate B if:
- A is **at least as good** as B on every objective, AND
- A is **strictly better** than B on at least one objective

```
Example (two objectives: CAPEX and unmet_load_kwh):

Candidate A: CAPEX=₹80,000  unmet=200 kWh
Candidate B: CAPEX=₹90,000  unmet=250 kWh   ← A dominates B (cheaper AND less unmet)
Candidate C: CAPEX=₹70,000  unmet=350 kWh   ← Neither A nor C dominates the other
                                               (C is cheaper but A has better backup)
→ A and C are both on the Pareto front (rank 0)
→ B is rank 1
```

### 4.6 Crowding Distance (Diversity Preservation)

Within each rank, individuals are ranked by **crowding distance** — a measure of how isolated a solution is from its neighbors on the Pareto front:

```
For each objective:
    Sort solutions by that objective
    Boundary solutions → distance = ∞
    Interior solutions → distance += (next_value - prev_value) / (max - min)

Total crowding distance = sum across all objectives
```

When selecting which solutions to carry to the next generation (if a rank partially fits), those with **larger crowding distance** (more isolated = more diverse) are preferred. This prevents the algorithm from clustering all solutions at one spot on the front and missing the rest of the trade-off curve.

### 4.7 Survival Selection

The next generation is assembled as follows:

```
Pool = current_population (size=pop_size) + offspring (size=pop_size)
     = 2 × pop_size candidates

Sort pool by (rank ASC, crowding_distance DESC)

Keep top pop_size individuals
```

This guarantees **elitism** — good solutions from the previous generation are never discarded in favor of worse offspring.

---

## 5. Solar Curve Caching

Since varying `tilt_deg` and `azimuth_deg` requires re-running the pvlib simulation (different angles → different POA irradiance → different AC output), and thousands of candidates are evaluated, a per-worker cache avoids redundant pvlib calls:

```python
# src/sizing/nsga2_sizing.py — SizingProblem
self._solar_cache: dict[tuple[float, float], np.ndarray] = {}

def _get_solar_curve(self, tilt_deg, azimuth_deg):
    key = (round(tilt_deg, 1), round(azimuth_deg, 1))
    if key not in self._solar_cache:
        self._solar_cache[key] = self.solar_curve_fn(*key)
    return self._solar_cache[key]
```

**Important caveat:** When parallel multiprocessing is used (`n_workers > 1`), each worker process has its **own copy** of this cache — it is not shared across workers. Cache misses are higher than in serial mode, but correctness is unaffected and per-worker caching still avoids re-running the same tilt/azimuth within one worker's share of the population.

---

## 6. Parallel Evaluation

Each candidate's simulation is fully independent, enabling near-linear parallelism:

```python
# src/sizing/nsga2_sizing.py — run_nsga2()
if n_workers > 1:
    ctx = mp.get_context("fork")   # fork: copies parent's memory → no re-pickling of large arrays
    pool = ctx.Pool(n_workers)
    elementwise_runner = StarmapParallelization(pool.starmap)
else:
    elementwise_runner = LoopedElementwiseEvaluation()  # serial

problem = SizingProblem(..., elementwise_runner=elementwise_runner)
```

- **fork** context (Linux/macOS) copies the parent process memory into each worker → the large solar arrays and GridModel are already present in worker memory without pickling overhead
- Each worker receives only the tiny per-candidate `x` vector to evaluate
- `FixedSolarCurve` and `SizingProblem` are implemented as **module-level classes** (not closures or nested classes) specifically because Python's pickle module cannot serialize closures for dispatch to worker processes — this was a confirmed bug that was fixed

---

## 7. The Pickling Constraint (A Confirmed Bug and Its Fix)

A critical implementation detail: **any callable passed to worker processes must be picklable**.

```python
# WRONG — closure cannot be pickled:
solar_kw_data = ...
solar_curve_fn = lambda tilt, az: solar_kw_data   # ← AttributeError at pickle time

# WRONG — nested class cannot be pickled:
def run_nsga2(...):
    class LocalSolarCurve:  # ← "Can't pickle local object" at dispatch
        ...

# CORRECT — module-level class, stores data as attribute:
class FixedSolarCurve:           # defined at module level in app.py / stage_functions.py
    def __init__(self, solar_kw_per_kwp):
        self.solar_kw_per_kwp = solar_kw_per_kwp
    def __call__(self, tilt_deg, azimuth_deg):
        return self.solar_kw_per_kwp
```

This same lesson was learned independently in `scripts/10_nsga2_sizing.py` (`SyntheticSolarCurve`, `RealSolarCurve`), `src/sizing/nsga2_sizing.py`, `app.py`, `stage_functions.py`, and `ga_explainer_routes.py` — every place the solar curve callable crosses a process boundary.

---

## 8. Final Pareto Front Processing

After `n_generations` complete, pymoo's `result.X` and `result.F` contain the final non-dominated solutions. These are **re-simulated once** to compute `annual_savings_inr` (which NSGA-II's internal evaluations computed but did not persist to the result):

```python
# src/sizing/nsga2_sizing.py — run_nsga2()
for x, f in zip(result.X, result.F):
    panel_kw, battery_kwh, tilt_deg, azimuth_deg = x
    sim_result = simulate_system_size(...)
    annual_savings_inr = baseline_annual_cost_inr - sim_result["annual_cost_inr"]
    rows.append({
        "panel_kw": ..., "battery_kwh": ..., "tilt_deg": ..., "azimuth_deg": ...,
        "total_capex_inr": f[0],
        "npv_25yr_inr": -f[1],        # un-negate
        "payback_years": f[2],
        "annual_unmet_load_kwh": f[3],
        "annual_savings_inr": annual_savings_inr,
    })
```

The Pareto front is returned as a DataFrame sorted by NPV descending. `top_recommendations()` then selects three named picks (fastest payback / best NPV / best backup) for the contractor UI and pipeline demo.

---

## 9. NSGA-II Settings Across Contexts

| Context | pop_size | n_generations | episode_days | Total evals | Purpose |
|---|---|---|---|---|---|
| `scripts/10_nsga2_sizing.py --full` | 40 | 50 | 365 | ~2,000 | Real sizing decision |
| `scripts/10_nsga2_sizing.py` (quick) | 8 | 3 | 365 | 24 | Wiring smoke test |
| `/api/size` (thorough depth) | 40 | 50 | 365 | ~2,000 | Contractor API |
| Pipeline demo (`stage_ga`) | 12 | 8 | 180 | ~96 | Demo pipeline (fast) |
| GA Explainer (`/api/ga-explainer/start`) | 24 | 15 | 60 | ~360 | Visual explainer |

---

## 10. GA Explainer — Watching NSGA-II in Real Time

The `ga_explainer_routes.py` module adds a separate `/api/ga-explainer` API that runs an **independent NSGA-II job** with live generation-by-generation recording, powering the `ga_explainer.html` UI.

### How it works:

```python
# ga_explainer_routes.py — GenerationRecorder (pymoo Callback)
class GenerationRecorder(Callback):
    def notify(self, algorithm):
        pop = algorithm.pop
        X, F, rank = pop.get("X"), pop.get("F"), pop.get("rank")
        # Records every individual's variables + objectives + Pareto rank
        # after NSGA-II's survival step assigns ranks
        self.history.append({"generation": algorithm.n_gen, "population": records})
```

The callback fires **after each generation's survival step**, so `rank` attributes are real (assigned by NSGA-II's non-dominated sort), not post-hoc estimates. The status endpoint exposes `current_generation` live so the frontend can show a progress bar.

### Crossover + Mutation Walkthrough:

The result also includes a **step-by-step crossover and mutation demo** on two real parents from the final Pareto front:

```python
# ga_explainer_routes.py — _sbx_one_gene()
def _sbx_one_gene(p1, p2, xl, xu, eta):
    u = rng.random()
    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        child = 0.5 * ((1 + beta)*p1 + (1 - beta)*p2)
    else:
        beta = (1.0 / (2.0 * (1 - u))) ** (1.0 / (eta + 1.0))
        child = 0.5 * ((1 - beta)*p1 + (1 + beta)*p2)
    return {"u": u, "beta": beta, "branch": ..., "value": child}

# _pm_one_gene() similarly returns u, delta, branch, value
```

Every intermediate value (`u`, `β` or `δ`, which branch fired, final child value) is returned so the frontend can display the exact math applied to each of the four variables — not a summary or approximation, the real computation.

---

## 11. End-to-End Flow Diagram

```
                     INPUTS
          ┌──────────────────────────────┐
          │ load_kw  (35,040 steps)      │
          │ solar_curve_fn(tilt, az)     │
          │ GridModel (tariffs+outages)  │
          │ financial_config             │
          │ baseline_annual_cost_inr     │
          └──────────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Generation 0       │
              │  Random population  │
              │  (pop_size=40)      │
              └──────────┬──────────┘
                         │  Evaluate all individuals:
                         │  simulate_system_size() × pop_size
                         ▼
              ┌─────────────────────────────────────────┐
              │  FOR each generation (1 … n_generations): │
              │                                           │
              │  1. Tournament selection of parent pairs  │
              │  2. SBX crossover  (η=15, prob=0.9)      │
              │     → new panel_kw, battery_kwh,          │
              │       tilt_deg, azimuth_deg per child     │
              │  3. PM mutation    (η=20)                 │
              │  4. Evaluate offspring (simulate year)    │
              │  5. Merge parents + offspring             │
              │  6. Non-dominated sort → assign ranks     │
              │  7. Crowding distance within each rank    │
              │  8. Keep best pop_size by (rank↑, CD↓)   │
              │  9. GenerationRecorder.notify() fires     │
              └──────────────────────┬──────────────────┘
                                     │
                         ▼ after n_generations
              ┌─────────────────────────────────────────┐
              │  Final rank-0 individuals = Pareto front │
              │  Re-simulate each → annual_savings_inr   │
              │  Return DataFrame sorted by NPV↓         │
              └──────────────────────┬──────────────────┘
                                     │
                         ▼
              ┌─────────────────────────────────────────┐
              │  top_recommendations()                   │
              │  ├── fastest_payback (min payback_years) │
              │  ├── best_npv       (max npv_25yr_inr)  │
              │  └── best_backup    (min unmet_load_kwh) │
              └─────────────────────────────────────────┘
```

---

## 12. Why Rule-Based Control for Fitness Evaluation?

NSGA-II evaluates **thousands of candidates** (pop_size × n_generations). Each evaluation is one year-long simulation. The three controller options and their cost per evaluation:

| Controller | Cost per candidate | Feasible for NSGA-II? |
|---|---|---|
| Rule-Based | ~0.5–2 sec | ✅ Yes |
| MPC (24h horizon LP) | ~30–120 sec | ❌ No (would take days) |
| DQN (training) | Hours | ❌ Impossible |

The rule-based controller is used for the same reason as in the grid search: it is fast, deterministic, and reproducible. The ranking it produces is a **lower bound** — a system ranked highly under rule-based control will perform even better under MPC or DQN. The `validate_candidates.py` spot-check step re-evaluates the top 3 candidates under all three controllers to confirm the ranking survives smarter control.

---

## 13. The Validated Sizing Result

After NSGA-II completes and `top_recommendations()` picks the three candidates, `validate_candidates.py` re-simulates them:

```
Top 3 NSGA-II candidates (payback / NPV / backup)
         │
         ├─ Rule-Based (180-day episode, annualized)
         ├─ MPC (24h horizon, CLARABEL LP solver)
         └─ DQN (loaded from models/dqn_policy.zip, if available)
                 │
                 ▼
    diagnose_ranking_disagreement()
    → "stable" / "near_tie_only" / "real_disagreement"

    resolve_recommendation()
    → single candidate (MPC tie-break), with HIGH / MEDIUM / LOW confidence
```

This is what the `/api/size` endpoint returns in the `validation` field of its JSON response — the claim that sizing is "checked three ways" is backed by real computation, not static text.
