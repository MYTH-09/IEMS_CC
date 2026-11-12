# Intelligent Energy Management System: Multi-Objective Sizing and Forecast-Aware Deep Reinforcement Learning for Blackout-Resilient Residential Microgrids

**Basava Darshan M** (PES2UG23CS118)  
**Bhanuteja Manchal** (PES2UG23CS119)  
**Bharath** (PES2UG23CS121)  
**Bhargav S** (PES2UG23CS124)  

*Department of Computer Science and Engineering, PES University, Bengaluru, India*  
**Project Guide:** Dr. Sharavana K  
**Course & Project ID:** UE23CS441A – Capstone Project Phase 3 | Project ID: 149  

---

## Abstract
Residential rooftop Solar Photovoltaic (PV) systems integrated with Battery Energy Storage Systems (BESS) are critical to decentralized grid decarbonization. However, in emerging distribution grids subject to unscheduled load shedding and monsoon-induced outages, standard energy management strategies fail. Conventional rule-based controllers operate myopically, exhausting stored battery energy during evening peak tariffs and leaving households defenseless when grid failures occur. Conversely, classical Model Predictive Control (MPC) entails prohibitive computational overhead on embedded edge microcontrollers. 

This paper presents an end-to-end **Intelligent Energy Management System (IEMS)** that addresses both static capacity planning and dynamic operational dispatch. First, a **Non-dominated Sorting Genetic Algorithm (NSGA-II)** optimizes continuous four-dimensional hardware configurations (PV capacity, battery capacity, panel tilt, and azimuth) across 35,040 annual timesteps, establishing a Pareto front that balances capital expenditure, grid dependency, and unmet load. Second, a **Smart Forecast-Aware Deep Q-Network (DQN)** controller is formulated over a 32-dimensional continuous state space. The state representation incorporates 1-hour, 3-hour, and 6-hour lookahead forecasts of solar irradiance, cloud cover, monsoon precipitation probability, and utility feeder outage hazards, alongside an analytical metric for required State of Charge ($\text{SoC}_{\text{req}}$). 

To ensure safe and resilient dispatch, an asymmetric reward function penalizes unserved load at $-₹100.00/\text{kWh}$ (the Value of Lost Load) while rewarding anticipatory pre-charging ahead of high-risk outage windows, backed by an external deterministic **Safety Shield** that enforces electrochemical and anti-islanding invariants. Evaluated on real 15-minute smart meter traces across diverse Indian cities under a strictly controlled hardware benchmark (identical 7.5 kWh BESS and 4.5 kWp PV), the proposed Smart DQN achieves a **96.0% reduction in unserved blackout energy** (reducing annual unmet load from 45.2 kWh to 1.8 kWh) and saves ₹1,087/year in electricity costs compared to standard heuristic dispatch. Embedded deployment on a Raspberry Pi 4 edge gateway demonstrates sub-millisecond ($0.8\text{ ms}$) ONNX inference and reliable RS-485 Modbus RTU inverter actuation, verifying real-world feasibility.

**Keywords:** Residential Microgrids, Deep Q-Networks (DQN), Reinforcement Learning, NSGA-II, Multi-Objective Sizing, Blackout Resilience, Weather Forecasting, Modbus RTU, Edge Computing.

---

## I. Introduction

Decentralized renewable energy systems, primarily rooftop Solar Photovoltaic (PV) installations paired with Lithium Iron Phosphate (LiFePO4) Battery Energy Storage Systems (BESS), represent the cornerstone of clean residential electrification. In rapidly expanding economies such as India, initiatives like the *PM Surya Ghar: Muft Bijli Yojana* have accelerated residential rooftop solar adoption through capital subsidies. However, distribution infrastructure in developing regions faces acute operational challenges: transmission congestion, peak load shedding, and frequent weather-induced distribution feeder faults during monsoon seasons.

In such environments, a residential microgrid must fulfill two competing objectives:
1. **Economic Arbitrage:** Maximize self-consumption of free solar generation and shift battery discharging to peak tariff periods (e.g., BESCOM time-of-day tariff slabs) to minimize electricity bills.
2. **Blackout Resilience:** Retain adequate stored electrical energy to bridge unpredictable grid interruptions, ensuring critical household appliances remain powered.

Existing residential energy management systems (EMS) fail to reconcile these objectives. Commercial hybrid inverters predominantly rely on **heuristic rule-based controllers** (e.g., self-consumption priority). These controllers are inherently **myopic**; they observe only instantaneous power balances ($P_{\text{solar}}$ and $P_{\text{load}}$). During evening peak hours (18:00 to 21:00), when grid power is most expensive, the rule-based controller discharges the battery down to its minimum depth-of-discharge (e.g., 10% State of Charge) to avoid grid purchases. If an unscheduled blackout or monsoon-induced line trip occurs at 21:00, the battery is completely depleted, plunging the household into darkness.

Advanced control techniques, such as **Model Predictive Control (MPC)** using Mixed-Integer Linear Programming (MILP), can theoretically optimize multi-hour dispatch. However, MPC requires solving complex optimization matrices online every 15 minutes. On cost-effective residential edge hardware (e.g., Raspberry Pi or ESP32 microcontrollers), commercial solvers introduce latency (2 to 8 seconds), high memory utilization, and failure modes during solver timeouts or network drops.

### Research Contributions
To overcome these limitations, this paper develops a two-stage Intelligent Energy Management System (IEMS) combining evolutionary hardware optimization with deep reinforcement learning:
1. **Multi-Objective Evolutionary Hardware Sizing:** We implement an NSGA-II framework optimizing a continuous 4D chromosome ($\text{PV kWp}, \text{Battery kWh}, \text{Tilt}^\circ, \text{Azimuth}^\circ$) across 35,040 annual 15-minute simulation steps, resolving Pareto trade-offs between CapEx, grid dependency, and outage vulnerability.
2. **32-Dimensional Lookahead State Formulation:** We formulate an MDP observation vector combining 17 base physical/tariff/temporal features with 15 multi-horizon lookahead features (1h, 3h, 6h forecast horizons of solar output, cloud cover, rain probability, feeder outage hazard, and an analytical required SoC headroom metric $\text{SoC}_{\text{req}}$).
3. **Asymmetric Reward & Resilience Engineering:** We design an asymmetric reward structure reflecting the true Value of Lost Load ($-₹100/\text{kWh}$ unserved penalty vs. ₹7/kWh import cost), teaching the DQN agent to proactively pre-charge from the grid prior to impending storms.
4. **Deterministic Action Safety Shield:** We implement a hard physical assertion wrapper that overrides neural outputs to guarantee electrochemical LiFePO4 operating bounds ($\text{SoC} \in [10\%, 100\%]$) and anti-islanding isolation.
5. **Pure Controller Benchmark & Hardware Edge Validation:** On an identical hardware configuration (7.5 kWh BESS), we prove that our Smart DQN achieves a **96.0% reduction in unserved outage energy** over rule-based control, and demonstrate embedded deployment on a Raspberry Pi 4 with $0.8\text{ ms}$ ONNX inference communicating via RS-485 Modbus RTU.

---

## II. Related Work

### A. Sizing Optimization in Photovoltaic-Battery Microgrids
Microgrid capacity sizing determines long-term return on investment (ROI). Classical techniques utilize discrete Grid Search sweeps over pre-defined component catalogs. While auditable, discrete sweeps fail to capture non-linear degradation, localized azimuth orientations, and roof tilt constraints. Recent studies apply genetic algorithms (GA) and Particle Swarm Optimization (PSO). Deb et al. established NSGA-II as the benchmark for multi-objective optimization due to its fast non-dominated sorting and crowding distance metric. In this work, NSGA-II is coupled with custom PVLib transposition algorithms and Faiman thermal models to simulate annual irradiance variations across diverse Indian climates.

### B. Microgrid Dispatch: Rule-Based vs. Advanced Control
Standard commercial hybrid inverters (e.g., SMA, Deye, Growatt, Victron) deploy heuristic threshold state machines. While reliable and computationally trivial ($< 0.01\text{ ms}$), their myopic nature prevents adaptation to dynamic tariffs or upcoming grid outages. 

Model Predictive Control (MPC) has been explored extensively for microgrid scheduling. MPC formulates finite-horizon optimizations using linear or mixed-integer programming. However, MPC's performance depends heavily on forecast precision; unexpected weather volatility causes significant performance degradation unless complex stochastic or robust MPC variants are deployed, which exponentially increases computational complexity beyond the capabilities of embedded residential gateways.

### C. Reinforcement Learning for Energy Management
Reinforcement Learning (RL) circumvents MPC's online computational bottleneck by shifting computation to an offline training phase. Once trained, deep neural networks execute forward inferences in milliseconds. Prior literature has applied Q-learning, Deep Q-Networks (DQN), and continuous policy gradient methods (PPO, DDPG, SAC) to home energy management. 

However, continuous policy gradient methods are sample-inefficient and struggle with inverter contactor dynamics, which operate across discrete operational states (idle, charge, pre-charge, discharge). Furthermore, existing RL microgrid literature overwhelmingly optimizes solely for electricity cost under the assumption of an uninterrupted 100% reliable utility grid—an assumption that completely fails in developing nations. Our work bridges this gap by explicitly integrating multi-horizon storm and outage forecasting with an asymmetric Value of Lost Load penalty.

---

## III. System Architecture & Mathematical Modeling

The physical architecture of the residential microgrid modeled in IEMS is shown below:

```
               +-------------------------------------------------+
               |             ROOFTOP SOLAR PV ARRAY              |
               +-------------------------------------------------+
                                        | (DC Power P_pv)
                                        v
+---------------+             +-------------------+             +---------------+
| UTILITY GRID  |<----------->|  HYBRID INVERTER  |<----------->| LiFePO4 BESS  |
|  (Distribution| (Bidirectional   (DC/AC Bus)    | (Bidirectional (Storage E_batt|
|    Feeder)    |   AC Power) +-------------------+   DC Power) |  SoC ∈ [0.1,1])
+---------------+                       |                       +---------------+
                                        | (AC Output)
                                        v
                              +-------------------+
                              | HOUSEHOLD LOADS   |
                              | (Critical & Flex) |
                              +-------------------+
```

### A. Solar PV Physics Modeling
Solar generation is modeled using hourly satellite irradiance data (Global Horizontal Irradiance $G_{\text{ghi}}$, Direct Normal Irradiance $D_{\text{dni}}$, and Diffuse Horizontal Irradiance $D_{\text{dhi}}$) obtained from the National Solar Radiation Database (NSRDB) and PVGIS, synthesized to 15-minute intervals. 

Plane-of-Array (POA) irradiance $I_{\text{poa}}$ is computed using the Perez diffuse transposition model as a function of surface tilt $\beta$ and azimuth $\gamma$:
$$I_{\text{poa}} = I_{\text{beam}} + I_{\text{diffuse, Perez}} + I_{\text{ground}}$$

Cell temperature $T_{\text{cell}}$ is evaluated using the empirical Faiman thermal model:
$$T_{\text{cell}} = T_{\text{ambient}} + \frac{I_{\text{poa}}}{u_0 + u_1 \cdot v_{\text{wind}}}$$
where $u_0 = 25.0\text{ W/m}^2\text{K}$ and $u_1 = 6.84\text{ W/m}^2\text{K}/(\text{m/s})$.

DC power generation $P_{\text{dc}}$ accounts for temperature-induced efficiency loss:
$$P_{\text{dc}} = P_{\text{stc}} \cdot \left(\frac{I_{\text{poa}}}{1000}\right) \cdot \left[1 + \gamma_{\text{temp}} (T_{\text{cell}} - 25^\circ\text{C})\right]$$
where $\gamma_{\text{temp}} = -0.0035/^\circ\text{C}$ for monocrystalline silicon. AC inverter output $P_{\text{pv}}$ is subject to inverter clipping at rated capacity $P_{\text{inv,max}}$ with conversion efficiency $\eta_{\text{inv}} = 0.96$:
$$P_{\text{pv}} = \min\left(P_{\text{inv,max}}, P_{\text{dc}} \cdot \eta_{\text{inv}}\right)$$

### B. LiFePO4 Battery Electrochemical Dynamics
The battery state of charge $\text{SoC}_t \in [\text{SoC}_{\min}, \text{SoC}_{\max}]$ represents the normalized energy reservoir:
$$\text{SoC}_t = \frac{E_{\text{batt}}(t)}{C_{\text{nominal}}}$$
where $\text{SoC}_{\min} = 0.10$ (10% reserve to prevent copper shunting and cathode degradation) and $\text{SoC}_{\max} = 1.00$.

The discrete-time energy transition equation over interval $\Delta t = 0.25\text{ hours}$ (15 minutes) is:
$$E_{\text{batt}}(t+1) = E_{\text{batt}}(t) + \left( P_{\text{chg}}(t) \cdot \eta_{\text{chg}} - \frac{P_{\text{dis}}(t)}{\eta_{\text{dis}}} \right) \Delta t$$
where internal charging efficiency $\eta_{\text{chg}} = 0.9487$ and discharging efficiency $\eta_{\text{dis}} = 0.9487$, yielding a roundtrip efficiency of:
$$\eta_{\text{rt}} = \eta_{\text{chg}} \cdot \eta_{\text{dis}} = 0.90 \quad (90.0\%)$$

Charging and discharging power rates are constrained by maximum C-rate limits ($C_{\text{rate}} = 0.5\text{C}$ continuous):
$$0 \le P_{\text{chg}}(t) \le \min\left(P_{\text{chg,max}}, \frac{(\text{SoC}_{\max} - \text{SoC}_t) C_{\text{nominal}}}{\eta_{\text{chg}} \Delta t}\right)$$
$$0 \le P_{\text{dis}}(t) \le \min\left(P_{\text{dis,max}}, \frac{(\text{SoC}_t - \text{SoC}_{\min}) C_{\text{nominal}} \cdot \eta_{\text{dis}}}{\Delta t}\right)$$

### C. Physical Energy Conservation & Bus Priority
At each timestep, local solar generation physically routes to household load via direct bus coupling before any battery or grid interaction:
$$P_{\text{direct}}(t) = \min\left(P_{\text{load}}(t), P_{\text{pv}}(t)\right)$$
The remaining residual power balance $\Delta P(t)$ is:
$$\Delta P(t) = P_{\text{pv}}(t) - P_{\text{load}}(t)$$

If $\Delta P(t) > 0$ (Generation Surplus):
- Battery charges up to allowable C-rate.
- Leftover excess solar is exported to the grid if active ($\text{Grid}_{\text{status}} = 1$), yielding feed-in export credits.
- If grid is down ($\text{Grid}_{\text{status}} = 0$), excess solar is curtailed to avoid inverter overvoltage.

If $\Delta P(t) < 0$ (Generation Deficit):
- Deficit is supplied by battery discharge subject to available SoC.
- If the grid is active, any remaining shortfall is imported from the utility grid at current tariff slab rates.
- If the grid is down, remaining shortfall cannot be served, producing **Unmet Load (Blackout)**:
  $$E_{\text{unmet}}(t) = \left(| \Delta P(t) | - P_{\text{dis}}(t)\right) \Delta t$$

---

## IV. Multi-Objective Sizing via NSGA-II

Before operational control begins, the static capacity of the microgrid must be optimized. We formulate sizing as a multi-objective optimization problem solved using the Non-dominated Sorting Genetic Algorithm II (NSGA-II).

```
                 NSGA-II CHROMOSOME STRUCTURE (4-DIMENSIONAL)
┌───────────────────────┬────────────────────────┬─────────────┬──────────────┐
│ PV Capacity (kWp)     │ Battery Capacity (kWh) │ Tilt Angle  │ Azimuth      │
│ Range: [1.0, 15.0]    │ Range: [0.0, 30.0]     │ Range: [0°, 45°]  Range: [90°, 270°]
└───────────────────────┴────────────────────────┴─────────────┴──────────────┘
```

### A. Mathematical Objective Formulation
The three competing objective functions evaluated across 35,040 annual steps are:

1. **Minimize Net Capital Expenditure ($f_1$):**
   $$f_1 = C_{\text{pv}} \cdot P_{\text{pv,cap}} + C_{\text{batt}} \cdot C_{\text{nominal}} + C_{\text{bos}} - \text{Subsidy}_{\text{PM\_Surya}}$$
   where $C_{\text{pv}} = ₹45,000/\text{kWp}$, $C_{\text{batt}} = ₹18,000/\text{kWh}$, and the Indian Central Government subsidy deducts up to ₹78,000 for systems $\ge 3\text{ kWp}$.

2. **Minimize Grid Import Dependency ($f_2$):**
   $$f_2 = \frac{\sum_{t=1}^{35040} E_{\text{grid\_import}}(t)}{\sum_{t=1}^{35040} E_{\text{load}}(t)} \times 100\%$$

3. **Minimize Annual Unserved Outage Load ($f_3$):**
   $$f_3 = \sum_{t=1}^{35040} E_{\text{unmet}}(t) \quad (\text{kWh/year})$$

### B. Genetic Mechanics & Convergence
The population ($N = 50$, Generations $= 40$) evolves through:
- **Simulated Binary Crossover (SBX):** Generates offspring chromosomes with distribution index $\eta_c = 15$.
- **Polynomial Mutation (PM):** Perturbs parameters with mutation probability $p_m = 0.25$ and $\eta_m = 20$.
- **Fast Non-dominated Sorting & Crowding Distance:** Preserves Pareto diversity across generations without fitness sharing parameters.

---

## V. Smart Forecast-Aware Deep Q-Network (DQN)

### A. MDP Formulation & Discrete Action Space
The dynamic dispatch problem is formulated as an infinite-horizon MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$. The discrete action space $\mathcal{A} = \{0, 1, 2, 3\}$ directly corresponds to commercial inverter operational modes:

```
┌─────────┬──────────────────────┬────────────────────────────────────────────────────────┐
│ Action  │ Command Name         │ Inverter Operational Mode                              │
├─────────┼──────────────────────┼────────────────────────────────────────────────────────┤
│    0    │ idle                 │ Inverter open. Battery isolated. Solar/load bypass.    │
│    1    │ charge_from_solar    │ Surplus PV routed to battery; leftover exported.       │
│    2    │ charge_from_grid     │ Proactive AC grid pre-charging to establish reserve.   │
│    3    │ discharge_to_load    │ Battery delivers power to home to offset peak tariff.  │
└─────────┴──────────────────────┴────────────────────────────────────────────────────────┘
```

### B. 32-Dimensional State Representation
At each 15-minute timestep $t$, the environment constructs state vector $s_t \in \mathbb{R}^{32}$:

```
                                  THE 32-DIMENSIONAL DQN OBSERVATION VECTOR
 ┌─────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
 │               BASE OPERATIONAL FEATURES (Indices 0–16)          │           LOOKAHEAD FORECAST FEATURES (Indices 17–31)            │
 ├────┬─────────────────────────────┬────┬─────────────────────────┼────┬─────────────────────────────┬────┬──────────────────────────┤
 │Idx │ Feature Name                │Idx │ Feature Name            │Idx │ Feature Name                │Idx │ Feature Name             │
 ├────┼─────────────────────────────┼────┼─────────────────────────┼────┼─────────────────────────────┼────┼──────────────────────────┤
 │ 0  │ solar_norm                  │ 9  │ hour_cos                │ 17 │ rain_probability_1h         │ 25 │ forecast_solar_6h_norm   │
 │ 1  │ load_norm                   │ 10 │ is_weekend              │ 18 │ rain_probability_3h         │ 26 │ forecast_load_3h_norm    │
 │ 2  │ net_power_norm              │ 11 │ cum_units_norm          │ 19 │ rain_probability_6h         │ 27 │ forecast_load_6h_norm    │
 │ 3  │ battery_soc                 │ 12 │ is_outage               │ 20 │ cloud_cover_1h              │ 28 │ outage_probability_1h    │
 │ 4  │ battery_capacity_norm       │ 13 │ outage_hazard           │ 21 │ cloud_cover_3h              │ 29 │ outage_probability_3h    │
 │ 5  │ autonomy_hours_norm         │ 14 │ month_sin               │ 22 │ cloud_cover_6h              │ 30 │ outage_probability_6h    │
 │ 6  │ marginal_rate_norm          │ 15 │ yesterday_solar_norm    │ 23 │ forecast_solar_1h_norm      │ 31 │ required_soc             │
 │ 7  │ rate_ratio                  │ 16 │ yesterday_load_norm     │ 24 │ forecast_solar_3h_norm      │    │                          │
 └────┴─────────────────────────────┴────┴─────────────────────────┴────┴─────────────────────────────┴────┴──────────────────────────┘
```

#### Analytical Required SoC Headroom
A central contribution of our state space is feature index 31: **$\text{SoC}_{\text{req}}$**. It dynamically computes the battery reserve required to survive an upcoming outage based on rolling 3-hour forecasts:
$$E_{\text{deficit,3h}} = \max\left(0, \int_{t}^{t+3\text{h}} \left[P_{\text{load}}(\tau) - P_{\text{solar}}(\tau)\right] d\tau\right)$$
$$\text{Risk}_{\text{outage}} = \max\left(\text{OutageProb}_{\text{3h}}, 0.8 \cdot \text{RainProb}_{\text{3h}}\right)$$
$$\text{SoC}_{\text{req}} = \text{clip}\left(\text{SoC}_{\min} + \frac{E_{\text{deficit,3h}} \cdot \text{Risk}_{\text{outage}}}{C_{\text{nominal}}}, \text{SoC}_{\min}, \text{SoC}_{\max}\right)$$

### C. Asymmetric Reward Engineering
The scalar reward $R_t$ aligns the agent's Q-values directly with bill savings and outage survival:
$$R_t = R_{\text{economic}} + R_{\text{incentives}}$$

$$R_{\text{economic}} = - \left(E_{\text{import}} \cdot \text{Rate}_{\text{import}}\right) + \left(E_{\text{export}} \cdot \text{Rate}_{\text{export}}\right) - \left(100.0 \cdot E_{\text{unmet}}\right)$$

Where:
- $\text{Rate}_{\text{import}}$ reflects tiered BESCOM slabs (₹4.50 to ₹7.50/kWh).
- $\text{Rate}_{\text{export}}$ reflects net-metering feed-in credit (₹3.00/kWh).
- Unserved load penalty is set to **$-₹100.00/\text{kWh}$**, representing the empirical Value of Lost Load (VoLL).

To guide policy exploration, structured shadow incentives are added:
$$R_{\text{incentives}} = \begin{cases}
+5.0 \cdot E_{\text{grid}\to\text{batt}}, & \text{if Action } 2 \text{ and } \text{Risk} \ge 0.45 \text{ and } \text{SoC} < \text{SoC}_{\text{req}} \\
-4.0 \cdot E_{\text{grid}\to\text{batt}}, & \text{if Action } 2 \text{ and clear weather (Parasitic import)} \\
-4.0, & \text{if Action } 0 \text{ during solar surplus with battery headroom} \\
+15.0 \cdot E_{\text{batt}\to\text{load}}, & \text{if Action } 3 \text{ during active grid outage}
\end{cases}$$

### D. Neural Network Topology & Training Stabilization
The Q-network is implemented as a Multi-Layer Perceptron (MLP) parameterized by weights $\theta$:
$$\text{Input } s_t \in \mathbb{R}^{32} \longrightarrow \text{Linear}(256) \to \text{ReLU} \longrightarrow \text{Linear}(256) \to \text{ReLU} \longrightarrow \text{Linear}(4) \to Q(s, a; \theta)$$

```
Input: State Vector s ∈ ℝ³²
           │
           ▼
    ┌──────────────┐
    │ Dense(256)   │  ── ReLU Activation
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Dense(256)   │  ── ReLU Activation
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Output Dense │  ── Linear (No Activation)
    └──────────────┘
           │
           ▼
Output: Q-Values for 4 Actions: [Q(s, a=0), Q(s, a=1), Q(s, a=2), Q(s, a=3)]
```

#### Replay Buffer & Target Network
To break time-series autocorrelation, transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ are stored in a ring buffer $\mathcal{D}$ of capacity $|\mathcal{D}| = 300,000$. Mini-batches of size $B = 128$ are sampled uniformly. Target values $y_t$ are generated via a frozen target network $\theta^-$ synchronized every 500 gradient steps:
$$y_t = r_t + \gamma (1 - d_t) \max_{a' \in \mathcal{A}} Q(s_{t+1}, a'; \theta^-)$$
with discount factor $\gamma = 0.995$ to sustain multi-hour planning horizons.

Weights are updated using the Adam optimizer ($\eta = 3 \times 10^{-4}$) on the smooth $L_1$ Huber loss:
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \text{Huber}\left(y_i - Q(s_i, a_i; \theta)\right)$$

#### Stochastic Training Pool & Forecast Noise Injection
The agent is trained across a diverse pool of 24 real smart meter traces (sampled from Mathura, Bareilly, Bengaluru, Mumbai, Chennai, and Hyderabad) spanning diverse load profiles. To prevent overfitting to perfect simulation forecasts, Gaussian noise is injected into training observations:
$$P_{\text{solar,fc}} \leftarrow P_{\text{solar,fc}} \cdot \text{clip}\left(\mathcal{N}(1.0, 0.15), 0.65, 1.35\right)$$
$$\text{RainProb} \leftarrow \text{clip}\left(\text{RainProb} + \mathcal{N}(0.0, 0.10), 0.0, 1.0\right)$$

### E. Deterministic Safety Shield
Before dispatching actions to physical inverter hardware, the chosen action $a_t = \arg\max Q(s_t, a)$ passes through an external deterministic assertion layer:

```python
class SafetyShield:
    def shield_action(self, env, chosen_action: int) -> int:
        soc = env.battery.soc
        is_outage = env.is_outage

        # Rule 1: Under-voltage cutoff (Overridden to Idle)
        if chosen_action == 3 and soc <= env.battery.min_soc + 0.005:
            return 0  
        # Rule 2: Over-charge cutoff (Overridden to Idle)
        if chosen_action in (1, 2) and soc >= env.battery.max_soc - 0.005:
            return 0  
        # Rule 3: Anti-Islanding Protection
        if chosen_action == 2 and is_outage:
            return 0  
        return chosen_action
```

---

## VI. Experimental Setup & Results

### A. Controlled Benchmark Methodology
To isolate the algorithmic contribution of the controller from hardware capacity influences, we conduct a **Pure Controller Benchmark**:
- **Hardware Configuration (Fixed):** $4.5\text{ kWp}$ Monocrystalline PV array, $7.5\text{ kWh}$ LiFePO4 BESS, $5.0\text{ kW}$ Hybrid Inverter.
- **Simulation Duration:** 35,040 consecutive timesteps (1 full year at 15-minute resolution).
- **Location:** Bengaluru, Karnataka, India (BESCOM residential tariff slabs, NSRDB satellite irradiance).
- **Outage Model:** Calibrated using historical BESCOM feeder interruption logs with elevated hazard during monsoon months (June to September).

### B. Quantitative Performance Comparison
Table I summarizes the annual benchmark results between standard Rule-Based dispatch and the proposed Smart Forecast-Aware DQN:

#### TABLE I: Annual Benchmark Performance (Identical 7.5 kWh Hardware)
| Performance Metric | Standard Rule-Based | Smart Forecast DQN | Relative Improvement |
|---|---|---|---|
| **Annual Unserved Load (Blackout)** | **45.2 kWh/year** | **1.8 kWh/year** | **96.0% Mitigation** |
| **Annual Blackout Duration** | 42.5 hours/year | 1.7 hours/year | **96.0% Reduction** |
| **Grid Electricity Cost** | ₹8,420 / year | ₹7,333 / year | **₹1,087 / year Savings** |
| **Solar Self-Consumption Rate** | 82.4% | 88.1% | + 5.7% Improvement |
| **Battery Equivalent Full Cycles** | 284 cycles/year | 268 cycles/year | Extended Cell Lifetime |
| **Worst-Case Single Outage Deficit** | 3.85 kWh | 0.12 kWh | 96.9% Peak Protection |

```
                ANNUAL UNSERVED BLACKOUT ENERGY (kWh/year)
       50 ┌────────────────────────────────────────────────────────┐
          │  ████████████████████████████████████ 45.2 kWh         │
       40 │  ████████████████████████████████████                  │
          │  ████████████████████████████████████  Rule-Based      │
       30 │  ████████████████████████████████████  (Myopic)        │
          │  ████████████████████████████████████                  │
       20 │  ████████████████████████████████████                  │
          │  ████████████████████████████████████                  │
       10 │  ████████████████████████████████████                  │
          │  █ 1.8 kWh  Smart DQN (96.0% Mitigation)               │
        0 └────────────────────────────────────────────────────────┘
```

### C. Analysis of Outage Mitigation Behavior
Figure 1 highlights the operational divergence during a representative monsoon storm event (17:00 to 21:00):
1. **Rule-Based Controller:** At 17:00, solar generation drops to $0.4\text{ kW}$ while load increases to $1.8\text{ kW}$. Observing a net deficit, the rule-based controller greedily discharges the battery to avoid peak tariffs. By 19:00, battery SoC hits the $10\%$ cutoff. When the storm causes a utility line fault at 19:15, the battery is depleted, causing $3.85\text{ kWh}$ of unserved load.
2. **Smart Forecast DQN:** At 16:30, the 32-dim observation detects $\text{RainProb}_{\text{3h}} = 0.85$ and $\text{OutageProb}_{\text{3h}} = 0.80$, yielding $\text{SoC}_{\text{req}} = 0.65$. The agent selects Action 2 (Grid Pre-charge), investing ₹21 in grid power to raise battery SoC to $75\%$ before the line trips. When the blackout occurs at 19:15, the agent transitions to Action 3, powering the home seamlessly with zero unserved energy.

### D. Embedded Edge Gateway Implementation
The trained PyTorch policy was exported to an optimized **ONNX Runtime** model and deployed on a **Raspberry Pi 4 Model B** (Quad-core ARM Cortex-A72 @ 1.5 GHz, 4GB RAM). 

Experimental execution profiling confirms:
- **Inference Latency:** Mean forward-pass time of **$0.82\text{ ms}$** (maximum $1.15\text{ ms}$), well within the 15-minute dispatch window ($900,000\text{ ms}$).
- **Hardware Communication:** Interface with a hybrid inverter achieved over an RS-485 to USB converter using the Modbus RTU protocol (Baud rate: 9600 bps, 8 data bits, no parity, 1 stop bit).
- **Modbus Registers:** Register `0x0021` (Grid Charge Enable), Register `0x0024` (Max Discharge Current), and Register `0x0030` (BMS SoC Telemetry).
- **Fail-Safe Redundancy:** If network communication with weather APIs fails, the edge runtime automatically falls back to an embedded deterministic policy (`_TrainedQPolicy`) with zero execution downtime.

---

## VII. Discussion & Viva Examination Insights

### A. Theoretical Justifications for Committee Review
1. **Discrete vs. Continuous RL:** Continuous algorithms (e.g., PPO, SAC) require continuous action space mapping, introducing continuous-to-discrete quantization errors that induce contactor chattering. DQN directly mirrors physical inverter firmware states.
2. **Value of Lost Load Selection:** Retail electricity tariffs in India range from ₹4.50 to ₹7.50/kWh, whereas diesel generator backup power costs ₹25.00 to ₹35.00/kWh. The $-₹100.00/\text{kWh}$ unserved penalty accurately forces the agent to trade off small upfront grid purchases for catastrophic outage avoidance.
3. **Generalization to Unseen Climates:** By training on a multi-household pool perturbed with stochastic forecast noise, the policy generalizes across non-training cities without policy retraining.

---

## VIII. Conclusion & Future Work

This paper presented an Intelligent Energy Management System (IEMS) combining multi-objective NSGA-II capacity sizing with a forecast-aware Deep Q-Network for residential microgrids. By integrating 1h, 3h, and 6h weather and outage forecasts into a 32-dimensional state vector and penalizing unserved load at $-₹100/\text{kWh}$, the DQN learns proactive pre-charging behaviors. 

Across an annual 35,040-step simulation on identical 7.5 kWh BESS hardware, the Smart DQN achieved a **96.0% reduction in unserved blackout energy** and saved ₹1,087/year in electricity bills. Physical safety is verified via an external deterministic safety shield, and edge hardware feasibility is validated on a Raspberry Pi 4 running ONNX inference in $0.8\text{ ms}$ over RS-485 Modbus RTU.

Future work will expand this framework to multi-agent peer-to-peer (P2P) transactive energy trading within distribution transformer microgrids and incorporate Vehicle-to-Home (V2H) bidirectional electric vehicle battery storage.

---

## References

1. V. Mnih, K. Kavukcuoglu, D. Silver, et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.
2. K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2, pp. 182–197, 2002.
3. Central Electricity Authority (CEA), "Technical Standards for Connectivity of the Distributed Generation Resources Regulations," Ministry of Power, Government of India, 2019.
4. M. Faiman, "Assessing the outdoor operating temperature of photovoltaic modules," *Progress in Photovoltaics: Research and Applications*, vol. 16, no. 4, pp. 307–315, 2008.
5. F. Dobbe, D. Arnold, S. Liu, et al., "Learning to control in power systems: Design and field evaluation," *IEEE Transactions on Control Systems Technology*, vol. 28, no. 6, pp. 2427–2441, 2020.
6. A. P. Dobos, "PVWatts Version 5 Manual," National Renewable Energy Laboratory (NREL) Technical Report NREL/TP-6A20-62641, 2014.
7. R. Perez, P. Ineichen, R. Seals, J. Michalsky, and R. Stewart, "Modeling daylight availability and irradiance components from direct and global irradiance," *Solar Energy*, vol. 44, no. 5, pp. 271–289, 1990.
8. BESCOM, "Retail Electricity Tariff Schedules for Low Tension Domestic Consumers (LT-2)," Bangalore Electricity Supply Company Limited, Karnataka, 2024.
9. Ministry of New and Renewable Energy (MNRE), "Operational Guidelines for PM Surya Ghar: Muft Bijli Yojana," Government of India, Feb. 2024.
10. S. R. Barker, A. Mishra, D. Irwin, E. Cecchet, and P. Shenoy, "SmartCap: Flattening Peak Electricity Demand in Smart Homes," in *Proc. IEEE International Conference on Pervasive Computing and Communications (PerCom)*, 2012, pp. 67–75.
