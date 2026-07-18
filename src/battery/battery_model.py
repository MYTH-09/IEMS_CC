"""
IEMS – Battery Storage Model
=============================
Physics-based model for a residential LFP (Lithium Iron Phosphate) battery.

Efficiency model
----------------
Round-trip efficiency (RTE) is split equally between charge and discharge:
    η_charge   = √RTE     (e.g. √0.90 ≈ 0.9487)
    η_discharge = √RTE

Charging (power P_c kW drawn from source):
    energy_stored  = P_c × η_charge × dt          (kWh added to battery)
    ΔSOC           = energy_stored / capacity_kwh

Discharging (power P_d kW delivered to load):
    energy_from_bat = P_d × dt / η_discharge       (kWh drawn from battery)
    ΔSOC            = -energy_from_bat / capacity_kwh

This ensures the round-trip loss is exactly (1 - RTE):
    charge 1 kWh → store η_c kWh → deliver η_c × η_d = RTE kWh to load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_DT_HOURS_DEFAULT = 15.0 / 60.0


@dataclass
class BatteryStep:
    soc_before: float
    soc_after: float
    actual_charge_kw: float
    actual_discharge_kw: float
    energy_stored_kwh: float
    energy_from_battery_kwh: float
    charge_curtailed_kw: float
    discharge_unmet_kw: float

    @property
    def is_charging(self) -> bool:
        return self.actual_charge_kw > 1e-6

    @property
    def is_discharging(self) -> bool:
        return self.actual_discharge_kw > 1e-6

    @property
    def is_idle(self) -> bool:
        return not self.is_charging and not self.is_discharging


class BatteryModel:
    def __init__(
        self,
        capacity_kwh:           float = 10.0,
        max_charge_rate_kw:     float = 3.0,
        max_discharge_rate_kw:  float = 3.0,
        round_trip_efficiency:  float = 0.90,
        min_soc:                float = 0.10,
        max_soc:                float = 0.95,
        initial_soc:            float = 0.50,
        cycle_life:             int   = 3000,
    ):
        assert 0 < capacity_kwh
        assert 0 < max_charge_rate_kw
        assert 0 < max_discharge_rate_kw
        assert 0 < round_trip_efficiency <= 1.0
        assert 0 <= min_soc < max_soc <= 1.0
        assert min_soc <= initial_soc <= max_soc

        self.capacity_kwh          = capacity_kwh
        self.max_charge_rate_kw    = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.round_trip_efficiency = round_trip_efficiency
        self.min_soc               = min_soc
        self.max_soc               = max_soc
        self.initial_soc           = initial_soc
        self.cycle_life            = cycle_life

        self.eta_c = float(np.sqrt(round_trip_efficiency))
        self.eta_d = float(np.sqrt(round_trip_efficiency))

        self._soc:                     float = initial_soc
        self._cumulative_charged_kwh:  float = 0.0
        self._cumulative_discharged_kwh: float = 0.0
        self._total_cycles:            float = 0.0

    @property
    def soc(self) -> float:
        return self._soc

    @property
    def energy_kwh(self) -> float:
        return self._soc * self.capacity_kwh

    @property
    def headroom_kwh(self) -> float:
        return (self.max_soc - self._soc) * self.capacity_kwh

    @property
    def available_kwh(self) -> float:
        return (self._soc - self.min_soc) * self.capacity_kwh

    @property
    def total_cycles(self) -> float:
        return self._total_cycles

    @property
    def state_of_health(self) -> float:
        degradation = (self._total_cycles / self.cycle_life) * 0.20
        return max(0.80, 1.0 - degradation)

    def reset(self, initial_soc: Optional[float] = None) -> None:
        self._soc                      = initial_soc if initial_soc is not None else self.initial_soc
        self._cumulative_charged_kwh   = 0.0
        self._cumulative_discharged_kwh = 0.0
        self._total_cycles             = 0.0

    def step(
        self,
        charge_kw:    float = 0.0,
        discharge_kw: float = 0.0,
        dt_hours:     float = _DT_HOURS_DEFAULT,
    ) -> BatteryStep:
        assert charge_kw >= 0
        assert discharge_kw >= 0
        assert dt_hours > 0

        soc_before = self._soc

        if charge_kw > 0 and discharge_kw > 0:
            discharge_kw = 0.0

        if charge_kw > 0:
            return self._charge(charge_kw, dt_hours, soc_before)
        elif discharge_kw > 0:
            return self._discharge(discharge_kw, dt_hours, soc_before)
        else:
            return BatteryStep(
                soc_before=soc_before, soc_after=soc_before,
                actual_charge_kw=0.0, actual_discharge_kw=0.0,
                energy_stored_kwh=0.0, energy_from_battery_kwh=0.0,
                charge_curtailed_kw=0.0, discharge_unmet_kw=0.0,
            )

    def _charge(self, requested_kw: float, dt_h: float, soc_before: float) -> BatteryStep:
        kw = min(requested_kw, self.max_charge_rate_kw)
        headroom = (self.max_soc - soc_before) * self.capacity_kwh
        kw = min(kw, headroom / (self.eta_c * dt_h + 1e-9))
        kw = max(kw, 0.0)

        energy_stored = kw * self.eta_c * dt_h

        soc_after = np.clip(
            soc_before + energy_stored / self.capacity_kwh,
            self.min_soc, self.max_soc,
        )
        self._soc = float(soc_after)
        self._cumulative_charged_kwh += energy_stored

        return BatteryStep(
            soc_before=soc_before,
            soc_after=float(soc_after),
            actual_charge_kw=kw,
            actual_discharge_kw=0.0,
            energy_stored_kwh=energy_stored,
            energy_from_battery_kwh=0.0,
            charge_curtailed_kw=max(0.0, requested_kw - kw),
            discharge_unmet_kw=0.0,
        )

    def _discharge(self, requested_kw: float, dt_h: float, soc_before: float) -> BatteryStep:
        kw = min(requested_kw, self.max_discharge_rate_kw)
        available = (soc_before - self.min_soc) * self.capacity_kwh
        kw = min(kw, available * self.eta_d / (dt_h + 1e-9))
        kw = max(kw, 0.0)

        energy_from_bat = kw * dt_h / self.eta_d

        soc_after = np.clip(
            soc_before - energy_from_bat / self.capacity_kwh,
            self.min_soc, self.max_soc,
        )
        self._soc = float(soc_after)
        self._cumulative_discharged_kwh += kw * dt_h
        self._total_cycles += energy_from_bat / (2.0 * self.capacity_kwh)

        return BatteryStep(
            soc_before=soc_before,
            soc_after=float(soc_after),
            actual_charge_kw=0.0,
            actual_discharge_kw=kw,
            energy_stored_kwh=0.0,
            energy_from_battery_kwh=energy_from_bat,
            charge_curtailed_kw=0.0,
            discharge_unmet_kw=max(0.0, requested_kw - kw),
        )

    def max_deliverable_kw(self, dt_hours: float = _DT_HOURS_DEFAULT) -> float:
        available = (self._soc - self.min_soc) * self.capacity_kwh
        return min(self.max_discharge_rate_kw, available * self.eta_d / dt_hours)

    def max_acceptable_kw(self, dt_hours: float = _DT_HOURS_DEFAULT) -> float:
        headroom = (self.max_soc - self._soc) * self.capacity_kwh
        return min(self.max_charge_rate_kw, headroom / (self.eta_c * dt_hours + 1e-9))

    def __repr__(self) -> str:
        return (
            f"BatteryModel("
            f"capacity={self.capacity_kwh} kWh, "
            f"SOC={self._soc:.1%}, "
            f"energy={self.energy_kwh:.2f} kWh, "
            f"cycles={self._total_cycles:.1f}, "
            f"SoH={self.state_of_health:.1%})"
        )