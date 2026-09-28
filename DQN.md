# IEMS: Previous Problems and Resolutions

This document records the major implementation problems identified during development of the Intelligent Energy Management Simulator (IEMS), the changes made to resolve them, and the limitations that still remain.

## Project Scope

IEMS sizes and validates residential solar-plus-battery systems for Indian households. It combines:

- Real smart-meter load data from Mathura and Bareilly
- NASA POWER solar irradiance data and `pvlib` solar simulation
- Battery and grid models
- Rule-based, MPC, and DQN controllers
- Grid-search and NSGA-II system sizing
- A FastAPI backend and contractor-facing HTML interfaces

The system operates at 15-minute resolution and accounts for slab-based electricity tariffs and grid outages.

## Problems Resolved

### 1. Household observation normalization was inconsistent

**Previous problem:** Each household normalized observations using its own maximum value. A DQN trained on one household could therefore interpret the same normalized value differently when evaluated on another household. The learned policy did not generalize reliably.

**Resolution:** A single fixed observation normalization constant is now used across the household pool. This keeps the observation meaning consistent during training and evaluation.

**Result:** Policies can be evaluated across different household load scales without changing the interpretation of their inputs.

### 2. MPC had an unfair future-information advantage

**Previous problem:** MPC could use an oracle-like future forecast while DQN only observed the current state. This made controller comparisons unfair because the controllers were not given equivalent information.

**Resolution:** MPC now uses a persistence forecast, where recent or historical patterns are carried forward. Persistence-forecast features were also added to the DQN observation so both controllers operate with deployment-realistic information.

**Result:** The comparison between MPC and DQN is now based on a more consistent information set.

### 3. MPC became infeasible during excess solar and outages

**Previous problem:** At some timesteps, excess solar could not be stored, exported, or consumed while the grid was unavailable. The optimization problem had no feasible way to dispose of that energy.

**Resolution:** A curtailment variable was added to the MPC optimization model. Excess solar can now be curtailed when storage, load, and export options are unavailable.

**Result:** MPC remains feasible during outage and battery-capacity edge cases.

### 4. DQN evaluation used a regressed final model

**Previous problem:** The final model after a long training run was not always the best model observed during training. Using only the final checkpoint could report weaker controller performance.

**Resolution:** Evaluation now uses the best checkpoint selected by `EvalCallback`, rather than automatically using the final training state.

**Result:** Reported DQN results represent the strongest validated checkpoint from the training run.

### 5. DQN training diversity was too narrow

**Previous problem:** The initial household pool mainly varied load magnitude. Load shape, solar behavior, hardware, and outage conditions were largely shared, allowing the policy to overfit the training distribution.

**Resolution:** Training and validation were extended to use multiple synthetic households and held-out households. The DQN was retrained with persistence information and revised training settings.

**Result:** In the Phase 3 evaluation, DQN outperformed MPC across three held-out households, with an average improvement of 20.2% over rule-based control versus 3.7% for MPC.

**Qualification:** The checkpoint-selection household overlaps with one reporting household in the comparison workflow. A separate validation household should be used for a fully independent final benchmark.

### 6. Outage assumptions were generic across cities

**Previous problem:** The application used the same outage assumption for every city, including a fixed daily outage estimate. This did not represent the measured differences between locations.

**Resolution:** For cities with available smart-meter data, outage parameters are derived from the measured data using `derive_outage_params_from_real_data`. Locations without measured data use a deterministic statistical fallback.

**Result:** Mathura and Bareilly use location-specific outage evidence, while unsupported locations still produce repeatable estimates instead of failing silently.

### 7. NSGA-II multiprocessing failed with local closures

**Previous problem:** The NSGA-II implementation used local closures that could not be serialized by multiprocessing. Parallel sizing therefore failed when multiple workers were requested.

**Resolution:** The multiprocessing callbacks were replaced with picklable classes and compatible worker interfaces.

**Result:** NSGA-II can run with multiprocessing without the previous serialization failure.

### 8. Reward calculations did not have sufficient physics checks

**Previous problem:** Battery transitions, energy conservation, and slab-based billing needed independent verification before controller and sizing results could be trusted.

**Resolution:** Dedicated validation scripts now check:

- Battery round-trip efficiency and state-of-charge transitions
- Grid tariff and outage behavior
- Energy conservation across all environment actions
- Agreement between accumulated step rewards and independent slab-bill calculations
- Rule-based controller behavior
- MPC feasibility
- Controller comparison and sizing validation

**Result:** The environment conserves energy with zero measured violation in the validation episode, and reward-to-bill agreement is within 0.01% relative error.

### 9. Sizing recommendations were not independently validated

**Previous problem:** Grid search and NSGA-II rank candidates using inexpensive rule-based control. That ranking could change under a more capable controller.

**Resolution:** The top candidates are re-simulated under rule-based, MPC, and DQN control. The validation layer detects meaningful ranking disagreements, applies a defined resolution policy, and returns a confidence label.

**Result:** The API reports a recommendation together with high, medium, or low confidence instead of presenting an unverified ranking as certain.

### 10. Live pipeline requests could block the web interface

**Previous problem:** Running all pipeline stages directly inside an HTTP request would make the API wait for a long computation and leave the frontend unresponsive.

**Resolution:** The pipeline API starts a background job and immediately returns a `job_id`. The frontend can poll status and retrieve completed stage results separately.

**Result:** The pipeline endpoints support asynchronous progress reporting through:

- `POST /api/pipeline/start`
- `GET /api/pipeline/status/{job_id}`
- `GET /api/pipeline/result/{job_id}/{stage_name}`

## Validated Results

The following results are currently documented from the validation workflow:

- Mathura and Bareilly solar capacity factor: approximately 17.3% to 17.6%
- Battery test: configured 90% round-trip efficiency produces exactly 10% loss in the validation scenario
- Reward and independent slab-bill agreement: within 0.01% relative error
- Earlier controller comparison: MPC improved annual reward by 283.9% over rule-based control; DQN improved it by 23.0%
- Phase 3 held-out evaluation: DQN averaged 20.2% improvement over rule-based control; MPC averaged 3.7%
- Sizing validation compares the top candidates across all three controllers and reports confidence

## Current Limitations

These items are not presented as resolved:

1. The live API uses a reduced NSGA-II configuration so that an HTTP request does not run the full long experiment. A production deployment should use a persistent background worker and job queue.
2. Roof shading is not modeled.
3. Real measured outage data is currently available only for Mathura and Bareilly.
4. DQN training uses a fixed hardware configuration in the existing validated workflow. Hardware variation should be added together with battery parameters in the observation.
5. Roof area is not yet enforced as a hard sizing constraint.
6. Sizing still uses rule-based control as a computationally practical proxy before validation.
7. The 300 free monthly units associated with the relevant subsidy scheme are not yet modeled in the financial calculations.
8. A fully independent DQN benchmark should use a validation household that is separate from both checkpoint selection and headline comparison reporting.

## How to Run

Create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pymoo
```

Run the validation and pipeline scripts in order when rebuilding the full workflow:

```powershell
python scripts/01_data_pipeline.py
python scripts/02_solar_pipeline.py --city Mathura
python scripts/03_battery_grid_check.py
python scripts/04_environment_check.py
python scripts/05_rule_based_check.py
python scripts/06_sizing_pipeline.py
python scripts/07_train_dqn.py --full
python scripts/08_mpc_check.py
python scripts/09_compare_controllers.py
python scripts/10_nsga2_sizing.py --real-solar --city Mathura --full
python scripts/11_validate_sizing.py --city Mathura
```

Start the API with:

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

Serve the frontend from another terminal:

```powershell
python -m http.server 5500
```

Then open `http://localhost:5500/pipeline_demo.html` or `http://localhost:5500/iems_contractor_ui.html`.

## Related Documentation

- `ReadMe.md`: full architecture, design decisions, module reference, and validated results
- `How_TO_RUN.md`: quick setup and application startup commands
- `PIPELINE_EXPLANATION_GUIDE.md`: explanation of the pipeline stages and frontend flow
