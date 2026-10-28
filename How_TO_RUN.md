# IEMS — How to Run & Verify

This guide provides exact, step-by-step instructions to set up, run, train, and evaluate the **Intelligent Energy Management Simulator (IEMS)** with the Smart Forecast-Aware DQN controller and multi-objective sizing pipeline.

---

## 1. Environment Setup

### Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/MYTH-09/IEMS_CC.git
cd IEMS_CC

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
pip install pymoo
```

---

## 2. Solar Data Pipeline (NASA POWER & pvlib)

Real satellite solar irradiance for 11 major Indian cities is already pre-cached in `data/solar/` (Mathura, Bareilly, Kolkata, Delhi, Bengaluru, Chennai, Mumbai, Hyderabad, Hubballi, Mangaluru, Mysuru).

To fetch or verify additional cities:
```bash
# List all 50+ supported Indian cities:
python scripts/02_solar_pipeline.py --list

# Fetch and simulate solar data for specific cities:
python scripts/02_solar_pipeline.py --city Bengaluru
python scripts/02_solar_pipeline.py --cities Kolkata Delhi Mumbai Chennai
```

---

## 3. Running the Web Application

### Step 1: Start the FastAPI Backend
```bash
uvicorn app:app --reload --port 8000
```
Backend API will be running at: `http://localhost:8000`  
Interactive API Docs: `http://localhost:8000/docs`

### Step 2: Open the Interactive Pipeline Demo
You can open `pipeline_demo.html` directly in your browser or run a simple local web server:
```bash
python -m http.server 5500
```
Then visit: `http://localhost:5500/pipeline_demo.html`

### Step 3: Run Full Pipeline
1. Enter **Monthly bill (₹)** (e.g. ₹3,000, ₹4,000).
2. Select **City** (e.g. Mumbai, Kolkata, Delhi, Mathura).
3. Select **Subsidy** (Yes / No).
4. Click **Run full pipeline →** and watch all stages compute live in ~35 seconds:
   - Synthetic Load Profiler (400 days from UP smart-meter archetypes)
   - Solar Generation (NASA POWER satellite + pvlib physics)
   - Grid Search Sizing (48 commercial configurations evaluated over 365 days)
   - Genetic Algorithm (NSGA-II continuous Pareto optimization)
   - Inverter Dispatch (Rule-Based vs. Smart Forecast-Aware DQN)
   - Final Report (Annualized financial and outage resilience comparison)

---

## 4. Smart DQN Training & Evaluation

### Train Smart DQN (Forecast-Aware 32-dim policy)
Trains the Deep Q-Network on multi-horizon weather, cloud cover, and grid hazard forecasts:
```bash
python scripts/train_smart_dqn.py --timesteps 500000 --save models/dqn_smart_policy.zip
```

### Evaluate Smart DQN Benchmark
Evaluates the trained Smart DQN against Rule-Based and Legacy controllers under blackout stress scenarios:
```bash
python scripts/evaluate_smart_dqn.py
```

---

## 5. Key Architecture Highlights

* **32-Dimensional State Space:** Combines 17 physical microgrid features with 15 multi-horizon rolling forecast signals (1h, 3h, 6h cloud cover, rain probability, outage hazard, and target battery reserve $SOC_{\text{req}}$).
* **Action Safety Shield:** Enforces strict physical safety constraints preventing battery damage, inverter clipping violations, and parasitic grid charging during non-risk hours.
* **Continuous Pareto Frontier:** NSGA-II optimizes 4 variables simultaneously (Panel kW, Battery kWh, Tilt, Azimuth) to generate 3 distinct practical choices (Fastest Payback, Best 25yr NPV, Best Outage Protection).
