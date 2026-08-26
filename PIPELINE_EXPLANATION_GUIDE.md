# IEMS Energy Simulation & Pipeline Guide

This guide explains how the Intelligent Energy Management Simulator (IEMS) works in simple, clear terms without complex math formulas or special symbols.

---

## 1. Energy Calculation (Bill to Daily & Monthly Units)

### What is a "Unit" of Electricity?
- 1 Unit = 1 Kilowatt-hour (kWh).
- 1 kWh is the amount of electrical energy used by a 1,000-Watt appliance (like an electric iron or geyser) running for exactly 1 hour.

### How Monthly and Daily Usage is Calculated
When a customer enters their monthly electricity bill (e.g., Rs 5,000 / month):

1. **Average Electricity Price**:
   In India, domestic electricity costs around Rs 5.00 per unit (kWh).

2. **Monthly Energy (Units/month)**:
   - Formula: Monthly Bill divided by Rs 5.00
   - Example: Rs 5,000 / 5.00 = 1,000 kWh (units) per month.

3. **Daily Energy (Units/day)**:
   - Formula: Monthly Energy divided by 30 days
   - Example: 1,000 / 30 = 33.33 kWh per day.

4. **Annual Energy (Units/year)**:
   - Formula: Daily Energy multiplied by 365 days
   - Example: 33.33 x 365 = 12,167 kWh per year.

5. **Average Power Demand**:
   - Formula: Daily Energy divided by 24 hours
   - Example: 33.33 / 24 = 1.39 kW average continuous load.

---

## 2. Explanation of the 7 Web Tabs & Pipeline Topics

---

### Tab 1: Synthetic Load Profile

#### Intention & Purpose
Most homes only know their monthly electricity bill, not their exact second-by-second electricity consumption. This stage converts that single bill amount into a full 365-day (15-minute resolution) electricity usage curve based on real Indian household smart-meter patterns.

#### Key Terms
- **Peak Load (kW)**: The highest amount of power drawn at any single moment (e.g., 3.38 kW in the morning when heaters and appliances run).
- **Average Load (kW)**: The average power level required across the entire day (1.39 kW).
- **Min Load (kW)**: The background "always-on" power (refrigerators, standby devices, 0.33 kW).
- **15-minute slot**: The day is split into 96 intervals of 15 minutes each to track changes accurately.

#### Web Output & Graphs
1. **Average Daily Consumption Shape**:
   - Shows the 24-hour daily curve with two peaks: a morning peak (~9:00 AM) and an evening peak (~8:30 PM).
   - Use for Homeowner: Shows when power is used and why solar alone cannot meet night peaks without batteries.
   - Use for Developer: Validates that total daily energy equals 33.33 kWh.
2. **Load Duration Curve (LDC)**:
   - Shows how often the home reaches high power levels.
   - Use: Proves that peak power (3.38 kW) only happens for 5% of the day, so inverters should not be oversized unnecessarily.
3. **Seasonal Shape (Monthly Averages)**:
   - Shows variation across months (summer vs monsoon vs winter).

---

### Tab 2: Solar Irradiance & Generation (NASA POWER + pvlib)

#### Intention & Purpose
Determines how much solar electricity can actually be generated on a rooftop in a specific Indian city (Bengaluru, Chennai, Delhi, Mumbai, Hyderabad, Mysuru, etc.) using official satellite weather data from NASA.

#### Key Terms
- **GHI (Global Horizontal Irradiance)**: Total solar sunlight hitting flat ground.
- **POA (Plane of Array Irradiance)**: Total sunlight falling on the tilted solar panel surface.
- **Optimal Tilt Angle**: The best panel angle (approx 12 deg to 28 deg in India) facing true South.
- **Capacity Factor (CF)**: The actual annual energy produced compared to if the panels ran at maximum capacity 24 hours a day (typically 17% to 18% in India).
- **Peak Sun Hours (PSH)**: Equivalent hours of full bright sunshine per day (4.2 to 4.4 hours/day).

---

### Tab 3: Grid Search Sizing (48 Commercial Sizes)

#### Intention & Purpose
Simulates 48 standard, commercially available hardware combinations of panels (1 to 10 kW) and batteries (0 to 20 kWh) against the home's 1-year load to find the best solutions for different budgets.

#### Three Recommended Hardware Packages
1. **Fastest Payback**:
   - Focus: Smallest upfront investment and fastest break-even.
   - Sizing: 2.0 kW solar with 0 kWh battery.
   - Result: Breaks even in ~2.7 years with a 40.5% return (IRR).
2. **Best 25-Year NPV (Maximum Profit)**:
   - Focus: Generating the highest total financial savings over 25 years.
   - Sizing: 10 kW solar with 20 kWh battery.
   - Result: Earns over Rs 8,43,000 in net lifetime savings.
3. **Best Outage Protection**:
   - Focus: Maximum power security during grid blackouts.
   - Sizing: 10 kW solar with 20 kWh battery.
   - Result: Keeps lights and appliances running through 99.5%+ of all power cuts.

#### Key Terms
- **CAPEX (Capital Expenditure)**: Total equipment and installation cost.
- **PM Surya Ghar Subsidy**: Central government solar subsidy (up to Rs 78,000) that reduces net cost.
- **Payback Period**: Number of years until electricity bill savings pay back the net installation cost.
- **NPV (Net Present Value)**: Total net profit earned over 25 years after subtracting maintenance and inflation.
- **IRR (Internal Rate of Return)**: Annual percentage return on investment compared to a fixed deposit.

#### Web Output & Graphs
- **Panel Size vs. NPV Bubble Chart**: Shows how bigger solar arrays increase profit, with bubble size representing battery storage.
- **CAPEX vs. Payback**: Shows how small systems pay back quickly, while larger systems take ~6.5 years but give complete power backup.

---

### Tab 4: Genetic Algorithm (GA / NSGA-II)

#### Intention & Purpose
Instead of testing only fixed commercial steps, the Genetic Algorithm uses evolutionary computing to test continuous variables (e.g., 3.42 kW panels, 7.8 kWh battery, 22.5 deg tilt) to find custom configurations tailored to unique rooftop limits.

#### Key Terms
- **Multi-Objective Optimization**: Finding solutions that optimize 3 conflicting goals at the same time:
  1. Minimize Upfront Cost (CAPEX)
  2. Maximize Lifetime Profit (NPV)
  3. Minimize Power Cuts (Unmet Load)
- **Pareto Frontier**: The line of optimal solutions where you cannot improve savings without spending more on hardware.

#### Web Output & Graphs
- **3D Pareto Front**: Shows the curve of optimal systems.
- **Generations Plot**: Shows the algorithm evolving towards optimal balance over multiple generations.

---

### Tab 5: Rule-Based (Grid Search)

#### Intention & Purpose
Simulates standard solar inverter logic on the Grid-Search recommended hardware package across 30 days of real operation.

#### How Rule-Based Logic Works
1. Solar directly powers the home first.
2. Extra solar charges the battery (up to 95% full).
3. Any remaining extra solar is exported to the grid for bill credits.
4. When sun goes down, battery discharges (down to 10% empty) to power the home.
5. Grid power is only imported if the battery is empty.
6. During a power cut, grid is disconnected and the battery powers essential appliances.

#### Web Output & Graphs
- **Battery State of Charge (SoC %)**: Shows daily battery cycling between 10% and 95%.
- **Power Flow Breakdown**: Shows how solar, battery, and grid share the load.

---

### Tab 6: Rule-Based (GA) & DQN (GA)

#### Intention & Purpose
Compares conventional rule-based inverter control against Deep Q-Network (Reinforcement Learning / AI) control on custom GA-sized hardware.

#### Key Terms & Differences
- **DQN (Deep Q-Network)**: An intelligent controller that learns to look ahead.
- **Pre-charging for Outages**: Unlike simple inverters, the DQN controller anticipates evening power cuts and ensures the battery is charged before blackouts occur.
- **Tariff Arbitrage**: Preserves battery energy for peak evening electricity slab rates to save extra money on bills.

#### Web Output & Graphs
- Side-by-side comparison of battery State of Charge and grid import reduction.

---

### Tab 7: Final Report (Outage Proof)

#### Intention & Purpose
Summarizes all simulation results into a clear, bankable contractor proposal for the homeowner.

#### Web Output & Deliverables
1. Three clean choice cards (Fastest Payback, Best 25-Yr Return, Best Outage Protection).
2. Exact bill savings per month and per year.
3. Detailed breakdown of Equipment Cost, Government Subsidy, and Net Cost.
4. Outage Coverage Score: Verified percentage of blackout hours completely bridged by solar and battery storage.

---

## 3. End-to-End Workflow Summary

```
Step 1: User enters Monthly Bill (e.g. Rs 5,000) and selects City (e.g. Bengaluru).
   │
   ▼
Step 2: Load Profiler converts the bill into a 365-day 15-minute consumption pattern.
   │
   ▼
Step 3: NASA POWER provides satellite solar data; pvlib simulates solar production.
   │
   ▼
Step 4: Grid Search tests 48 sizes; Genetic Algorithm finds optimal custom sizes.
   │
   ▼
Step 5: Inverter Dispatch simulation (Rule-Based & DQN) tests battery behavior and power cuts.
   │
   ▼
Step 6: Final Report generates the bankable proposal with costs, subsidies, savings, and backup scores.
```
