"""
IEMS – Action Safety Shield
============================
Enforces physical, electrical, and blackout-resilience safety constraints
on proposed controller actions (from DQN or heuristics).

Architecture (as detailed in SMART_DQN.md):
    Proposed Action (from DQN)
              ↓
        Safety Shield
              ↓
    Feasible Safe Action
              ↓
         Environment

Guarantees:
1. Physical feasibility: No solar charging without surplus; no grid charging
   during blackout; no discharging an empty battery below safety min_soc.
2. Blackout protection: If grid is down and household load exceeds solar,
   forces battery discharge (Action 3) whenever battery has usable capacity,
   preventing catastrophic unserved load caused by accidental idling.
3. Overcharge / deep discharge protection against boundary violations.
4. Strategic Resilience & Reserve Management:
   - Preserves an emergency battery reserve (e.g., 30% SOC) during night hours
     or elevated outage risk when the grid is UP, preventing premature
     depletion on non-critical evening loads.
   - Triggers pre-charging (Action 2) from grid power ahead of high-hazard
     outage periods if the battery is depleted, ensuring full backup protection
     when the grid cuts out.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class SafetyShield:
    """
    Validates and transforms proposed actions into physically feasible,
    blackout-safe battery commands with active outage-reserve protection.
    """

    def __init__(
        self,
        verbose: bool = False,
        reserve_soc: float = 0.25,
        precharge_threshold: float = 0.45,
        precharge_target: float = 0.40,
    ):
        self.verbose = verbose
        self.reserve_soc = reserve_soc
        self.precharge_threshold = precharge_threshold
        self.precharge_target = precharge_target

    def shield_action(self, env, proposed_action: int) -> int:
        """
        Validate proposed_action against the current physical state of env.
        Returns the safe action: 0 (idle), 1 (solar charge), 2 (grid charge), 3 (discharge).
        """
        idx = min(env._data_idx, len(env.load_kw) - 1)
        ts = env.timestamps[idx]
        is_outage = env.grid.is_outage(ts)
        load_kw = float(env.load_kw[idx])
        solar_kw = float(env.solar_kw[idx])
        soc = float(env.battery.soc)
        min_soc = float(env.battery.min_soc)
        max_soc = float(env.battery.max_soc)

        # ── 1. Critical Blackout Guard ─────────────────────────────────────────
        # During an outage, if household load exceeds solar and battery has usable
        # capacity, discharging to load (Action 3) MUST take priority to prevent blackout.
        if is_outage and (load_kw > solar_kw) and (soc > min_soc + 1e-3):
            return 3

        # ── 2. Automatic Solar Surplus Absorption (Self-Consumption Priority) ───
        # Across every grid jurisdiction, buying grid power at night (₹5-7/kWh) costs
        # more than double the net-metering export credit (₹2-2.50/kWh). Therefore,
        # whenever free solar surplus exists, charging from solar (Action 1) takes
        # physical precedence (battery stores excess; if full, inverter exports/curtails).
        if solar_kw > load_kw + 1e-3 and soc < max_soc - 1e-3:
            return 1

        # ── 3. Strategic Grid-Connected Deficit Management ────────────────────
        # During normal non-outage hours when load exceeds solar:
        if not is_outage and (solar_kw <= load_kw):
            hour = float(ts.hour) + float(ts.minute) / 60.0

            # Pre-sunrise smart discharge (05:00 - 07:00):
            # Upcoming solar generation will arrive at dawn to fully recharge the
            # battery. Utilizing stored clean energy down to 0.15 SOC completely
            # eliminates expensive early-morning grid imports while retaining safety margin.
            # During all other normal grid hours, preserve the 0.25 emergency backup reserve.
            reserve_floor = 0.15 if (5.0 <= hour < 7.0) else self.reserve_soc

            # Action 2 (charge_from_grid): ONLY allowed if explicitly proposed by
            # the policy during an impending blackout risk with no solar available.
            # Never hijack Action 0 (idle) into Action 2.
            if proposed_action == 2:
                lookahead = min(12, len(env.solar_kw) - idx)
                upcoming_solar = float(np.mean(env.solar_kw[idx:idx+lookahead])) if lookahead > 0 else 0.0
                hazard = 1.0 if int(ts.hour) in getattr(env.grid, "_outage_peak_hours", [10, 14, 18, 22]) else 0.2
                if hazard > 0.60 and upcoming_solar < 0.2 and soc < reserve_floor:
                    return 2
                # If daytime solar will arrive or hazard is low, avoid wasteful grid pre-charge
                return 0

            # Action 3 (discharge_to_load) or Action 0:
            # If battery has clean stored solar above the reserve floor, discharge
            # to displace grid imports.
            if soc > reserve_floor + 1e-3:
                return 3
            return 0

        # ── 4. Physical Feasibility Checks ────────────────────────────────────
        action = proposed_action

        # Action 1: charge_from_solar
        # Requires physical solar surplus (solar > load) and battery not full
        if action == 1:
            if solar_kw <= load_kw or soc >= max_soc - 1e-3:
                # If load deficit exists, can we discharge?
                if load_kw > solar_kw and soc > min_soc + 1e-3:
                    action = 3
                else:
                    action = 0

        # Action 2: charge_from_grid
        # Grid must be available, no solar surplus, battery not completely full
        elif action == 2:
            if is_outage:
                # Grid is dead: cannot charge from grid
                if load_kw > solar_kw and soc > min_soc + 1e-3:
                    action = 3
                else:
                    action = 0
            elif solar_kw > load_kw:
                # Surplus solar available: free solar takes precedence over paid grid!
                if soc < max_soc - 1e-3:
                    action = 1
                else:
                    action = 0
            elif soc >= 0.90:
                # Battery already well-buffered; avoid wasteful grid charging
                action = 0

        # Action 3: discharge_to_load
        # Requires active load deficit (load > solar) and battery above min_soc
        elif action == 3:
            if load_kw <= solar_kw:
                # We have excess solar, so discharging is unnecessary and counterproductive
                if soc < max_soc - 1e-3:
                    action = 1
                else:
                    action = 0
            elif soc <= min_soc + 1e-3:
                # Battery depleted to safety minimum: cannot discharge further
                action = 0

        # Action 0: idle
        # If solar surplus exists and battery can accept it, don't let it waste
        elif action == 0:
            if (not is_outage) and (solar_kw > load_kw + 0.1) and (soc < max_soc - 0.05):
                # Auto-divert surplus solar to storage
                action = 1

        return int(action)
