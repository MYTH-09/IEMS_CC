"""
IEMS – Rule-Based Baseline Controller
========================================
A simple, fully explainable priority-logic controller. Serves two roles
in the project:

1. The benchmark that DQN and MPC are compared against — the core
   "does intelligent control actually beat a sensible fixed rule?"
   experiment (§8.4 of the build spec).
2. The controller used during the sizing grid search (§9.1) — kept
   deliberately simple and fast so sizing doesn't depend on RL training.

Priority logic:
----------------
1. If solar generation exceeds household load right now:
       → charge the battery with the surplus (action 1: charge_from_solar)
2. Else if solar is insufficient AND the battery has energy above floor:
       → discharge the battery to cover the deficit (action 3: discharge_to_load)
       During normal grid hours, preserves an emergency reserve (default 25% SOC)
       so blackouts never strike with an empty battery.
       During an active outage, discharges all the way to min_soc (10%) to keep
       the home powered.
3. Otherwise (no surplus, no usable battery charge):
       → idle (action 0) — the environment routes the remaining deficit to grid
4. This baseline NEVER selects charge_from_grid (action 2) — that
   capability exists in the environment specifically so learned controllers
   (DQN) can demonstrate smart forecast-aware pre-charging.
"""

from __future__ import annotations

from src.controllers.base_controller import BaseController
from src.environment.energy_env import EnergyEnv


class RuleBasedController(BaseController):
    """
    Fixed priority-logic controller: solar -> load, then solar -> battery,
    then battery -> load, then (implicitly, via the environment) grid -> load.

    Maintains an active 25% backup reserve during grid-connected hours to
    protect against load shedding, discharging fully down to min_soc (10%)
    during active power outages.
    """

    name = "rule_based"

    def __init__(self, backup_reserve_soc: float = 0.25):
        self.backup_reserve_soc = backup_reserve_soc

    def reset(self):
        pass

    def select_action(self, env: EnergyEnv) -> int:
        idx = env._data_idx
        solar_kw = float(env.solar_kw[idx])
        load_kw  = float(env.load_kw[idx])
        soc      = env.battery.soc
        min_soc  = env.battery.min_soc
        is_outage = env.grid.is_outage(env.timestamps[idx])

        surplus_kw = solar_kw - load_kw

        if surplus_kw > 1e-9:
            # Solar covers the load with some left over — store it.
            return 1   # charge_from_solar

        # During grid-connected hours, preserve backup reserve for blackouts.
        # During an active outage, allow discharging all the way down to min_soc.
        discharge_floor = min_soc if is_outage else max(min_soc, self.backup_reserve_soc)

        if surplus_kw < -1e-9 and soc > discharge_floor + 1e-9:
            # Solar can't cover the load, but battery has usable charge above floor.
            return 3   # discharge_to_load

        # No surplus to store, and no usable battery charge to draw on —
        # let the environment route the deficit to the grid.
        return 0   # idle
