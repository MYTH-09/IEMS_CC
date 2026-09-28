
git clone https://github.com/MYTH-09/IEMS_CC.git

-- Create a virtual env --
```
python3 -m venv venv
```
```
source venv/bin/activate
```

-- Installing modules --
```
pip install -r requirements.txt
pip install pymoo
```

-- Solar Data Pipeline (NASA POWER) --
```
# List all 50+ supported Indian cities (Bengaluru, Chennai, Hyderabad, Mysuru, Mumbai, Delhi, etc.):
python scripts/02_solar_pipeline.py --list

# Fetch and simulate solar data for specific cities:
python scripts/02_solar_pipeline.py --city Bengaluru
python scripts/02_solar_pipeline.py --cities Bengaluru Chennai Hyderabad Mysuru Mumbai Delhi

# Fetch for all Indian cities:
python scripts/02_solar_pipeline.py --all
```

-- starting backend -- http://127.0.0.1:8000
```
uvicorn app:app --reload --port 8000
# or on Windows:
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

-- starting frontend -- 
```
Open "pipeline_demo.html" or "iems_contractor_ui.html" via Live Server / browser:
python -m http.server 5500
```
http://localhost:5500/pipeline_demo.html

All major Indian cities (Bengaluru, Chennai, Hyderabad, Mysuru, Delhi, Mumbai, Kolkata, etc.) are available in the City dropdown!


Updated todo list

I’ll ground this in the repository’s actual environment and training setup, then compute the discount horizon precisely. I’ll also verify the installed SB3 behavior rather than relying only on general DQN conventions, since the Double/Dueling/PER question is implementation-specific.

Read , lines 35 to 145

Read , lines 80 to 180

Read , lines 35 to 100

Read 

The local setup matches the concern: all six households share the same 10 kWh/3 kW battery and the same outage process, while the solar shape is also identical apart from tiny noise. I’m verifying SB3’s actual constructor defaults and then I’ll rank interventions by leverage, including whether they change the policy’s input/action contract.

Ran terminal command: import stable_baselines3 as sb3; import inspect; from stable_baselines3 import DQN; print('sb3', sb3.__version__); print(inspect.signature(DQN.__init__)); print(inspect.getsource(DQN.train)[:5000])

**Pool Diversity**

The current pool is narrow. In `07_train_dqn.py:48-89`:

- All households use the same daily load shape.
- Only `load_scale` varies from `0.6x` to `1.6x`.
- Solar has the same magnitude and daily shape for every household, with only small noise differences.
- All households use the same 10 kWh battery, 3 kW charge/discharge limits, efficiencies, and SOC limits.
- All use the same `GridModel`.
- Because timestamps are identical and the fallback outage model is timestamp-based, the outage schedule is effectively identical too.

This tests whether a policy can handle different load magnitudes. It does not test generalization across household behavior, solar systems, batteries, or grid conditions.

**Minimum useful expansion**

1. **Add genuinely different household and solar profiles first**

   Increase from 6 to at least **24–32 households**, with variation in:

   - Morning/evening usage patterns
   - Base load and peak load
   - Weekday/weekend behavior
   - Solar capacity and generation shape
   - Random weather/cloud variability

   More copies of the same synthetic profile will not help much. The current 60-day series is long enough mechanically, but it is not behaviorally diverse. Extending it to 90–180 days is useful only if it introduces seasonal or weather variation.

2. **Add hardware variation together with hardware observations**

   Randomize battery capacity, charge/discharge rate, efficiency, and SOC limits. For example:

   - Capacity: 5–20 kWh
   - Charge/discharge rate: approximately `0.25C–1C`
   - Efficiency: 0.85–0.98
   - Different usable SOC ranges

   This requires adding battery parameters to the observation. Otherwise, the same observation can require different actions for two batteries. Without those features, hardware randomization may reduce performance rather than improve generalization.

3. **Randomize outage behavior and expose anticipation information**

   The current observation contains only the present `is_outage` bit. Varying outage parameters without giving the DQN outage history, forecast, or probability will make the task partially observable.

   A practical minimum is to add:

   - Recent outage history
   - Expected outage probability over the next few hours, or
   - A deployment-realistic outage forecast

   If the DQN is being compared against MPC, future outage information must be made equally available to both controllers. Otherwise MPC retains an information advantage.

**Discount Horizon**

The environment uses 15-minute steps:

$$
dt = 0.25\text{ hours}
$$

With $\gamma=0.99$:

$$
H_{\text{effective}} \approx \frac{1}{1-\gamma}
= \frac{1}{0.01}
= 100\text{ steps}
$$

Therefore:

$$
100 \times 0.25 = 25\text{ hours}
$$

The exponential time constant gives essentially the same result:

$$
-\frac{1}{\ln(0.99)} \times 0.25 \approx 24.9\text{ hours}
$$

A reward exactly 24 hours ahead still has weight:

$$
0.99^{96} \approx 0.382
$$

So `gamma=0.99` is reasonably aligned with the 24-hour outage-avoidance problem. It does not need to change as a first fix.

A small experiment with `gamma=0.995` is worthwhile if pre-charging decisions need longer credit assignment:

- Effective horizon: approximately 50 hours
- 24-hour reward weight: approximately $0.619$

However, increasing gamma will not compensate for missing future load, solar, or outage information. The observation mismatch is more important.

**Prioritized Changes**

| Priority | Change | Type | Expected impact | Retraining |
|---|---|---|---|---|
| 1 | Add a 24-hour forecast vector for load and solar, plus future outage information available at deployment | Observation-space | Very high; directly reduces MPC’s look-ahead advantage | From scratch |
| 2 | Expand the household distribution with 24–32 genuinely different load/solar profiles | Training process | Very high; reduces profile overfitting | From scratch |
| 3 | Increase training from 500k to at least 2–5 million steps and balance household sampling | Training process | High; 500k gives only about 173 complete episodes total | From scratch |
| 4 | Add battery capacity, rate, efficiency, and SOC limits to both environment randomization and observations | Observation-space + training process | High when deployment hardware varies | From scratch |
| 5 | Randomize outage schedules and add outage history/forecast features | Observation-space + training process | High for outage avoidance | From scratch |
| 6 | Test `exploration_fraction=0.30–0.50`, keeping final epsilon around `0.05` | Hyperparameter | Medium; improves coverage of rare household/outage situations | Retrain |
| 7 | Test `gamma=0.995` alongside `0.99` | Hyperparameter | Medium or low; potentially helps multi-hour pre-charging | Retrain |
| 8 | Replace the four coarse actions with continuous battery power or more granular actions | Action-space/model change | High ceiling, but high implementation effort | From scratch |
| 9 | Reduce or retune the network after distribution changes | Hyperparameter/model | Medium; `256x256` is large for 9 inputs and 4 actions but not the primary problem | Retrain |

The highest-value first experiment would be:

- 32 diverse households
- Randomized outage schedules
- A 24-hour persistence forecast and outage forecast in the observation
- 2 million or more training steps
- Compare `gamma=0.99` and `gamma=0.995`

**SB3 DQN Features**

The direct `stable_baselines3.DQN` used in `dqn_controller.py:74-96` is vanilla DQN. SB3’s standard DQN does **not** enable these by default:

- Double DQN: **No**
- Dueling networks: **No**
- Prioritized experience replay: **No**

The replay buffer is standard uniform replay. The existing `policy_kwargs=dict(net_arch=[256, 256])` changes only the MLP size; it does not activate dueling architecture.

`sb3-contrib` provides `QRDQN`, which replaces scalar Q-values with a distributional quantile representation. It can be useful for noisy rewards and risk-sensitive behavior, but it is not automatically a complete Double-Dueling-PER implementation. Getting all three would generally require:

- A custom Double-DQN target calculation
- A custom dueling Q-network or policy
- A prioritized replay buffer, including importance-sampling corrections
- Careful validation against SB3’s training loop

For this environment, these upgrades are probably **lower priority** than fixing the data and information gaps. A 9-dimensional observation and 4-action space is small; the dominant limitations are narrow training diversity, missing future information, coarse actions, and limited trajectory exposure. QRDQN or dueling DQN may provide incremental stability improvements, but they are unlikely to explain or close a `+23%` versus `+284%` gap by themselves.

Created 4 todos