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

Priority logic (exact, as specified — do not add hidden heuristics)
----------------------------------------------------------------------
1. If solar generation exceeds household load right now:
       → charge the battery with the surplus (action 1: charge_from_solar)
2. Else if solar is insufficient AND the battery has energy above min_soc:
       → discharge the battery to cover the deficit (action 3: discharge_to_load)
3. Otherwise (no surplus, no usable battery charge):
       → idle (action 0) — the environment's own energy-balance logic
         then routes the remaining deficit to the grid automatically
4. This baseline NEVER selects charge_from_grid (action 2) — that
   capability exists in the environment specifically so a learned
   controller (DQN/MPC) can be shown to outperform this baseline by
   discovering when pre-charging ahead of an anticipated outage is
   worth the grid cost. If the rule-based controller used it too,
   that comparison would be muddied.
"""

from __future__ import annotations

from src.controllers.base_controller import BaseController
from src.environment.energy_env import EnergyEnv


class RuleBasedController(BaseController):
    """
    Fixed priority-logic controller: solar → load, then solar → battery,
    then battery → load, then (implicitly, via the environment) grid → load.
    """

    name = "rule_based"

    def select_action(self, env: EnergyEnv) -> int:
        idx = env._data_idx
        solar_kw = float(env.solar_kw[idx])
        load_kw  = float(env.load_kw[idx])
        soc      = env.battery.soc
        min_soc  = env.battery.min_soc

        surplus_kw = solar_kw - load_kw

        if surplus_kw > 1e-9:
            # Solar covers the load with some left over — store it.
            return 1   # charge_from_solar

        if surplus_kw < -1e-9 and soc > min_soc + 1e-9:
            # Solar can't cover the load, but the battery has usable charge.
            return 3   # discharge_to_load

        # No surplus to store, and no usable battery charge to draw on —
        # let the environment route the deficit to the grid.
        return 0   # idle