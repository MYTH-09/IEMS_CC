"""
IEMS – Controller Base Interface
===================================
Defines the shared interface every controller (rule-based, DQN, MPC) must
implement, so `scripts/06_train_controllers.py` can evaluate all three
against the same environment without special-casing any of them.

Why controllers receive the live `EnergyEnv` instance rather than just an
observation vector
--------------------------------------------------------------------------
DQN operates naturally on the environment's normalized `Box` observation
(that's what the neural net was trained on). But MPC needs forward-looking
raw solar/load arrays for its forecast horizon, and the rule-based
controller reads raw, un-normalized values for its threshold logic —
neither of those is recoverable cleanly from the normalized observation
vector alone. Giving every controller the environment object itself lets
each one pull whatever representation of state it actually needs, while
still sharing one `select_action(env) -> int` call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.environment.energy_env import EnergyEnv


class BaseController(ABC):
    """
    Common interface for all IEMS battery controllers.

    Subclasses must implement `select_action`, which is called once per
    simulation timestep, immediately before `env.step(action)`.
    """

    name: str = "base"

    @abstractmethod
    def select_action(self, env: EnergyEnv) -> int:
        """
        Decide the action for the current timestep.

        Parameters
        ----------
        env : EnergyEnv
            The live environment, already advanced to the current timestep
            (i.e. `env._data_idx` points at the timestep about to be acted
            on). Controllers may read `env.solar_kw`, `env.load_kw`,
            `env.battery`, `env.grid`, `env._data_idx`, etc. directly.

        Returns
        -------
        int — one of {0: idle, 1: charge_from_solar, 2: charge_from_grid,
        3: discharge_to_load}, matching `EnergyEnv.action_space`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional hook called at the start of each episode (after
        `env.reset()`), for controllers that carry internal state across
        an episode (e.g. MPC's forecast cache). Rule-based and DQN
        controllers typically don't need this.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"