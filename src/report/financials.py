"""
IEMS – Financial Calculations
================================
Pure functions for techno-economic analysis: system cost (with PM Surya
Ghar subsidy applied), net present value, internal rate of return, and
payback period. Used by both the sizing grid search (§9.1) and the final
PDF report (§10).

These are intentionally side-effect-free — no file I/O, no config
loading — so they're trivial to unit test and reuse across modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CAPEX (with PM Surya Ghar Muft Bijli Yojana subsidy)
# ─────────────────────────────────────────────────────────────────────────────

def capex(
    panel_kw: float,
    battery_kwh: float,
    panel_cost_per_kw: float = 45000.0,
    battery_cost_per_kwh: float = 15000.0,
    balance_of_system_cost: float = 10000.0,
    subsidy_rate_up_to_2kw: float = 0.40,
    subsidy_rate_2_to_3kw: float = 0.20,
    subsidy_max_kw: float = 3.0,
) -> dict:
    """
    Compute total system CAPEX after the PM Surya Ghar Muft Bijli Yojana
    subsidy.

    Subsidy structure (as of the 2024 scheme, mirrored from config):
        - 40% subsidy on panel cost for the first 2 kWp
        - 20% subsidy on panel cost for the portion between 2-3 kWp
        - No subsidy beyond 3 kWp
        - Battery and balance-of-system costs are NOT subsidized

    Parameters
    ----------
    panel_kw, battery_kwh : System size being costed.
    panel_cost_per_kw, battery_cost_per_kwh, balance_of_system_cost : ₹ unit costs.
    subsidy_rate_up_to_2kw, subsidy_rate_2_to_3kw, subsidy_max_kw : Subsidy structure.

    Returns
    -------
    dict with keys: panel_cost_gross, subsidy_amount, panel_cost_net,
    battery_cost, bos_cost, total_capex.
    """
    assert panel_kw >= 0 and battery_kwh >= 0, "Sizes must be non-negative"

    panel_cost_gross = panel_kw * panel_cost_per_kw

    # Subsidy applies only to panel cost, tiered by capacity:
    # first 2kW at subsidy_rate_up_to_2kw, next portion (up to subsidy_max_kw) at subsidy_rate_2_to_3kw
    tier1_kw = min(panel_kw, 2.0)
    tier2_kw = max(0.0, min(panel_kw, subsidy_max_kw) - 2.0)

    subsidy_amount = (
        tier1_kw * panel_cost_per_kw * subsidy_rate_up_to_2kw +
        tier2_kw * panel_cost_per_kw * subsidy_rate_2_to_3kw
    )

    panel_cost_net = panel_cost_gross - subsidy_amount
    battery_cost   = battery_kwh * battery_cost_per_kwh
    bos_cost       = balance_of_system_cost

    total_capex = panel_cost_net + battery_cost + bos_cost

    return {
        "panel_cost_gross": round(panel_cost_gross, 2),
        "subsidy_amount":   round(subsidy_amount, 2),
        "panel_cost_net":   round(panel_cost_net, 2),
        "battery_cost":     round(battery_cost, 2),
        "bos_cost":         round(bos_cost, 2),
        "total_capex":      round(total_capex, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NPV / IRR / Payback
# ─────────────────────────────────────────────────────────────────────────────

def project_annual_savings(
    year1_savings_inr: float,
    years: int,
    tariff_inflation_rate: float = 0.05,
    panel_degradation_rate: float = 0.005,
    maintenance_cost_annual_inr: float = 2000.0,
) -> np.ndarray:
    """
    Project annual net savings across the system lifetime.

    Two competing effects:
        - Grid tariffs rise over time (tariff_inflation_rate), which
          *increases* the value of each kWh the system offsets.
        - Panel output degrades over time (panel_degradation_rate), which
          *decreases* the kWh actually offset.
        - Maintenance cost is a flat annual deduction.

    This is a simplification (assumes the two rate effects apply
    multiplicatively to a fixed year-1 baseline) — adequate for a
    contractor-facing estimate, not a substitute for detailed year-by-year
    simulation.

    Returns
    -------
    np.ndarray of length `years` — net savings for each year (₹).
    """
    years_arr = np.arange(1, years + 1)
    tariff_growth      = (1 + tariff_inflation_rate) ** years_arr
    panel_output_factor = (1 - panel_degradation_rate) ** years_arr

    gross_savings = year1_savings_inr * tariff_growth * panel_output_factor
    net_savings   = gross_savings - maintenance_cost_annual_inr

    return net_savings


def npv(
    annual_savings: np.ndarray,
    initial_capex: float,
    discount_rate: float = 0.08,
) -> float:
    """
    Net present value of a system over its projected savings lifetime.

    NPV = -CAPEX + Σ (savings_t / (1 + r)^t)  for t = 1..years
    """
    years_arr = np.arange(1, len(annual_savings) + 1)
    discount_factors = 1.0 / (1 + discount_rate) ** years_arr
    pv_savings = float(np.sum(annual_savings * discount_factors))
    return pv_savings - initial_capex


def irr(cashflows: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> float:
    """
    Internal rate of return via Newton-Raphson on the NPV function.

    Parameters
    ----------
    cashflows : np.ndarray
        cashflows[0] should be the initial CAPEX as a NEGATIVE number,
        followed by each year's net savings as positive numbers.

    Returns
    -------
    float — IRR as a decimal (e.g. 0.12 = 12%). Returns np.nan if no
    solution converges (e.g. all cashflows are negative).
    """
    if np.all(cashflows <= 0):
        return float("nan")

    rate = 0.1   # initial guess
    years_arr = np.arange(len(cashflows))

    for _ in range(max_iter):
        denom = (1 + rate) ** years_arr
        npv_val  = np.sum(cashflows / denom)
        d_npv    = np.sum(-years_arr * cashflows / (denom * (1 + rate)))

        if abs(d_npv) < 1e-12:
            break

        new_rate = rate - npv_val / d_npv
        if abs(new_rate - rate) < tol:
            return float(new_rate)
        rate = new_rate

        if rate <= -0.99:   # guard against divergence
            return float("nan")

    return float(rate)


def payback_period(
    initial_capex: float,
    annual_savings: np.ndarray,
) -> float:
    """
    Simple payback period: the year (fractional) at which cumulative
    savings first equals or exceeds CAPEX.

    Returns
    -------
    float — years to payback. Returns np.inf if payback never occurs
    within the projected period.
    """
    cumulative = np.cumsum(annual_savings)
    payback_year_idx = np.searchsorted(cumulative, initial_capex)

    if payback_year_idx >= len(cumulative):
        return float("inf")

    # Linear interpolation within the payback year for a fractional result
    year_start_cum = cumulative[payback_year_idx - 1] if payback_year_idx > 0 else 0.0
    year_savings   = annual_savings[payback_year_idx]
    fraction       = (initial_capex - year_start_cum) / year_savings if year_savings > 0 else 0.0

    return float(payback_year_idx + fraction)


@dataclass
class FinancialSummary:
    """Bundled result of a full financial analysis for one system size."""
    panel_kw: float
    battery_kwh: float
    total_capex_inr: float
    subsidy_amount_inr: float
    year1_savings_inr: float
    payback_years: float
    npv_25yr_inr: float
    irr_pct: float

    def __repr__(self) -> str:
        return (
            f"FinancialSummary(panel={self.panel_kw}kW, battery={self.battery_kwh}kWh, "
            f"CAPEX=₹{self.total_capex_inr:,.0f}, payback={self.payback_years:.1f}yr, "
            f"NPV25={self.npv_25yr_inr:,.0f}, IRR={self.irr_pct:.1f}%)"
        )


def full_financial_analysis(
    panel_kw: float,
    battery_kwh: float,
    year1_savings_inr: float,
    config: dict,
) -> FinancialSummary:
    """
    Convenience wrapper: run the full CAPEX → savings projection → NPV/IRR/
    payback pipeline for one system size, using values pulled from the
    project config's `sizing:` section.

    Parameters
    ----------
    panel_kw, battery_kwh : System size.
    year1_savings_inr : First-year net savings (from simulating this
        system size through a full year with the rule-based controller —
        see `src/sizing/grid_search.py`).
    config : dict
        The `sizing:` section of `default_config.yaml`.
    """
    capex_result = capex(
        panel_kw, battery_kwh,
        panel_cost_per_kw       = config["panel_cost_per_kw"],
        battery_cost_per_kwh    = config["battery_cost_per_kwh"],
        balance_of_system_cost  = config["balance_of_system_cost"],
        subsidy_rate_up_to_2kw  = config["subsidy_rate_up_to_2kw"],
        subsidy_rate_2_to_3kw   = config["subsidy_rate_2_to_3kw"],
        subsidy_max_kw          = config["subsidy_max_kw"],
    )

    years = config["analysis_period_years"]
    annual_savings = project_annual_savings(
        year1_savings_inr,
        years=years,
        tariff_inflation_rate       = config["electricity_inflation_rate"],
        panel_degradation_rate      = config["panel_degradation_rate"],
        maintenance_cost_annual_inr = config["maintenance_cost_annual_inr"],
    )

    npv_result = npv(
        annual_savings, capex_result["total_capex"],
        discount_rate=config["discount_rate"],
    )

    cashflows = np.concatenate([[-capex_result["total_capex"]], annual_savings])
    irr_result = irr(cashflows)

    payback = payback_period(capex_result["total_capex"], annual_savings)

    return FinancialSummary(
        panel_kw=panel_kw,
        battery_kwh=battery_kwh,
        total_capex_inr=capex_result["total_capex"],
        subsidy_amount_inr=capex_result["subsidy_amount"],
        year1_savings_inr=year1_savings_inr,
        payback_years=payback,
        npv_25yr_inr=npv_result,
        irr_pct=irr_result * 100 if not np.isnan(irr_result) else float("nan"),
    )