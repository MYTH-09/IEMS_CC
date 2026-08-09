"""
IEMS – Grid Model
==================
Models the household's connection to the UP electricity grid:

1. Slab-based tariff billing (DVVNL/PVVNL residential rates)
2. Net metering credit for exported solar energy
3. Grid outage awareness (from real detected outage patterns, Module 1)

Why slab tariffs matter for the control policy
------------------------------------------------
Unlike time-of-use (ToU) tariffs common in Western RL energy papers, UP
residential billing charges a *higher rate for every unit* once cumulative
monthly consumption crosses a threshold. This means the "cost" of importing
1 more kWh from the grid depends on how much has already been consumed
*this month* — not on the time of day. A per-timestep reward function must
account for this cumulative context, or it will optimize for the wrong
thing (e.g. treating all grid units as equally expensive/cheap).

This module exposes both:
- `marginal_rate(cumulative_units)` — the ₹/kWh rate for the *next* unit,
  given how much has already been consumed this billing month. This is
  what a per-step reward function should use.
- `bill_for_month(total_units)` — the full slab-computed bill for a month,
  used for final reporting/validation, not for step-wise reward shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Tariff slab definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TariffSlab:
    """One slab: applies `rate` (₹/kWh) to units up to `units_upto`."""
    units_upto: float   # Upper bound of this slab (kWh/month); use float('inf') for last slab
    rate: float          # ₹/kWh within this slab


# Default UP residential tariff (DVVNL/PVVNL, 2023-24) — mirrors default_config.yaml
DEFAULT_TARIFF_SLABS: list[TariffSlab] = [
    TariffSlab(100,        3.35),
    TariffSlab(150,        3.85),
    TariffSlab(300,        5.00),
    TariffSlab(500,        5.50),
    TariffSlab(float("inf"), 6.00),
]

DEFAULT_FIXED_CHARGE_INR   = 125.0   # ₹/month, independent of usage
DEFAULT_NET_METERING_RATE  = 2.50    # ₹/kWh credited for exported solar


# ─────────────────────────────────────────────────────────────────────────────
# Grid Model
# ─────────────────────────────────────────────────────────────────────────────

class GridModel:
    """
    Models grid billing and availability for one household.

    Usage
    -----
        grid = GridModel(outage_lookup=outage_series)

        # Per-timestep, during simulation:
        rate = grid.marginal_rate(cumulative_units_this_month=42.3)
        cost = import_kwh * rate

        credit = export_kwh * grid.net_metering_rate

        is_out = grid.is_outage(timestamp)

        # End of month, for reporting:
        bill = grid.bill_for_month(total_units=310)
    """

    def __init__(
        self,
        tariff_slabs: Optional[list[TariffSlab]] = None,
        fixed_charge_inr: float = DEFAULT_FIXED_CHARGE_INR,
        net_metering_rate: float = DEFAULT_NET_METERING_RATE,
        outage_lookup: Optional[pd.Series] = None,
        fallback_outage_hours_per_day: float = 4.0,
        outage_peak_hours: Optional[list[int]] = None,
        monthly_outage_multiplier: Optional[dict[int, float]] = None,   # ADD THIS LINE
    ):
        """
        Parameters
        ----------
        tariff_slabs : list[TariffSlab]
            Ascending slab boundaries with associated ₹/kWh rates.
            Defaults to UP DVVNL/PVVNL residential slabs.
        fixed_charge_inr : float
            Monthly fixed charge (₹), independent of consumption.
        net_metering_rate : float
            ₹/kWh credited for solar energy exported to the grid.
        outage_lookup : Optional[pd.Series]
            Boolean series indexed by timestamp (from Module 1's
            `is_outage` column), True = grid unavailable at that timestamp.
            If None, falls back to a statistical outage model.
        fallback_outage_hours_per_day : float
            Used only when `outage_lookup` is not provided — average daily
            outage hours to sample synthetically.
        outage_peak_hours : Optional[list[int]]
            Used only in fallback mode — hours more likely to have outages.
        """
        self.tariff_slabs = tariff_slabs or DEFAULT_TARIFF_SLABS
        self.fixed_charge_inr = fixed_charge_inr
        self.net_metering_rate = net_metering_rate

        self._outage_lookup = outage_lookup
        self._fallback_outage_hours_per_day = fallback_outage_hours_per_day
        self._outage_peak_hours = outage_peak_hours or [10, 14, 18, 22]
        self._monthly_outage_multiplier = monthly_outage_multiplier or {}   # ADD THIS LINE

        self._validate_slabs()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_slabs(self) -> None:
        prev_upto = 0.0
        for slab in self.tariff_slabs:
            assert slab.units_upto > prev_upto, "Slab boundaries must be strictly increasing"
            assert slab.rate > 0, "Slab rate must be positive"
            prev_upto = slab.units_upto
        assert self.tariff_slabs[-1].units_upto == float("inf"), \
            "Last slab must have units_upto = inf to cover unlimited consumption"

    # ── Tariff logic ─────────────────────────────────────────────────────────

    def marginal_rate(self, cumulative_units_this_month: float) -> float:
        """
        Return the ₹/kWh rate that applies to the *next* unit of grid import,
        given how much has already been consumed this billing month.

        This is the rate a per-timestep reward function should use — it
        correctly reflects slab-based pricing without needing the reward
        function to know the full month's consumption in advance.

        Parameters
        ----------
        cumulative_units_this_month : float
            Total kWh imported from the grid so far this billing month
            (before this timestep's import).

        Returns
        -------
        float — ₹/kWh marginal rate.
        """
        assert cumulative_units_this_month >= 0, "cumulative_units_this_month must be ≥ 0"

        for slab in self.tariff_slabs:
            if cumulative_units_this_month < slab.units_upto:
                return slab.rate
        # Should never reach here since last slab is inf, but be safe:
        return self.tariff_slabs[-1].rate

    def bill_for_month(self, total_units: float) -> dict:
        """
        Compute the full slab-based bill for a month of total_units kWh.
        Used for end-of-month reporting and validation — NOT for per-step
        reward shaping (use `marginal_rate` for that).

        Returns
        -------
        dict with keys:
            total_units, energy_charge_inr, fixed_charge_inr, total_bill_inr,
            slab_breakdown (list of per-slab kWh and cost)
        """
        assert total_units >= 0, "total_units must be ≥ 0"

        remaining = total_units
        prev_upto = 0.0
        energy_charge = 0.0
        breakdown = []

        for slab in self.tariff_slabs:
            slab_capacity = slab.units_upto - prev_upto
            units_in_slab = min(remaining, slab_capacity)
            if units_in_slab <= 0:
                break
            cost = units_in_slab * slab.rate
            energy_charge += cost
            breakdown.append({
                "slab_upto": slab.units_upto,
                "rate": slab.rate,
                "units": units_in_slab,
                "cost_inr": round(cost, 2),
            })
            remaining -= units_in_slab
            prev_upto = slab.units_upto
            if remaining <= 0:
                break

        total_bill = energy_charge + self.fixed_charge_inr

        return {
            "total_units": total_units,
            "energy_charge_inr": round(energy_charge, 2),
            "fixed_charge_inr": self.fixed_charge_inr,
            "total_bill_inr": round(total_bill, 2),
            "slab_breakdown": breakdown,
        }

    # ── Outage logic ─────────────────────────────────────────────────────────

    def is_outage(self, timestamp: pd.Timestamp) -> bool:
        """
        Return True if the grid is unavailable at `timestamp`.

        Uses the real detected outage calendar (`outage_lookup`) when
        available. Falls back to a statistical model otherwise — deterministic
        given the same timestamp, so simulations remain reproducible.
        """
        if self._outage_lookup is not None:
            try:
                return bool(self._outage_lookup.loc[timestamp])
            except KeyError:
                # Timestamp not in lookup (e.g. synthetic date beyond data range)
                return self._fallback_is_outage(timestamp)
        return self._fallback_is_outage(timestamp)

    def _fallback_is_outage(self, timestamp: pd.Timestamp) -> bool:
        """
        Deterministic statistical outage fallback, used when no real outage
        data is available for the requested timestamp (e.g. synthetic future
        dates in a multi-year projection).

        Uses a hash of (date, hour) to deterministically decide outage status,
        so repeated calls for the same timestamp always agree, while still
        approximating the configured average daily outage hours.
        """
        hour = timestamp.hour
        month = timestamp.month                                                    # ADD
        base_prob = self._fallback_outage_hours_per_day / 24.0
        peak_boost = 2.0 if hour in self._outage_peak_hours else 1.0
        month_mult = self._monthly_outage_multiplier.get(month, 1.0)                # ADD
        prob = min(base_prob * peak_boost * month_mult, 1.0)                        # CHANGED
        seed_val = hash((timestamp.date(), hour)) % (2**32)
        rng = np.random.default_rng(seed_val)
        return bool(rng.random() < prob)

    # ── Outage lookup construction (helper for pipeline scripts) ─────────────

    @staticmethod
    def build_outage_lookup(processed_df: pd.DataFrame, meter_id: str) -> pd.Series:
        """
        Build an outage lookup Series for a single meter from the processed
        15-min DataFrame (output of preprocessor.run_pipeline()).

        Parameters
        ----------
        processed_df : pd.DataFrame
            Must contain columns: meter_id, timestamp, is_outage.
        meter_id : str
            Which meter's outage calendar to extract.

        Returns
        -------
        pd.Series
            Boolean series indexed by timestamp, True = outage.
        """
        subset = processed_df[processed_df["meter_id"] == meter_id]
        if len(subset) == 0:
            raise ValueError(f"No data found for meter_id='{meter_id}'")

        lookup = subset.set_index("timestamp")["is_outage"]
        logger.info(
            f"Built outage lookup for {meter_id}: {len(lookup):,} timestamps, "
            f"{lookup.sum():,} outage slots ({100*lookup.mean():.1f}%)"
        )
        return lookup

    @staticmethod
    def build_representative_outage_profile(
        processed_df: pd.DataFrame,
        city: str,
    ) -> pd.Series:
        """
        Build a city-wide representative outage profile by averaging outage
        occurrence across all meters in that city, keyed by (month, hour) —
        useful for projecting outage behavior onto synthetic future years
        that fall outside the real dataset's date range.

        Returns
        -------
        pd.Series
            MultiIndex (month, hour) → outage probability [0, 1].
        """
        subset = processed_df[processed_df["city"] == city].copy()
        subset["month"] = subset["timestamp"].dt.month
        subset["hour"]  = subset["timestamp"].dt.hour

        profile = (
            subset.groupby(["month", "hour"])["is_outage"]
            .mean()
            .rename("outage_probability")
        )
        logger.info(
            f"Built representative outage profile for {city}: "
            f"{len(profile)} (month, hour) buckets, "
            f"avg probability={profile.mean():.3f}"
        )
        return profile

    def __repr__(self) -> str:
        n_slabs = len(self.tariff_slabs)
        has_real_outages = self._outage_lookup is not None
        return (
            f"GridModel(slabs={n_slabs}, "
            f"fixed_charge=₹{self.fixed_charge_inr}/mo, "
            f"net_metering=₹{self.net_metering_rate}/kWh, "
            f"outage_source={'real_data' if has_real_outages else 'statistical_fallback'})"
        )