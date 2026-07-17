"""
IEMS – Script 03: Battery + Grid Model Sanity Check
=====================================================
Validates the battery physics model and grid tariff model in isolation,
before they're wired into the Gymnasium simulation environment.

Checks performed
-----------------
Battery:
  1. Charging increases SOC correctly, respecting round-trip efficiency
  2. Discharging decreases SOC correctly
  3. Hardware rate limits are enforced (can't exceed max_charge/discharge_kw)
  4. SOC bounds are enforced (never below min_soc or above max_soc)
  5. Round-trip efficiency: charge X kWh in, less than X kWh comes back out

Grid:
  1. Marginal rate increases correctly as cumulative units cross slab boundaries
  2. Full month bill calculation matches manual slab arithmetic
  3. Outage lookup returns correct values for known timestamps
  4. Fallback outage model produces deterministic results

Usage (from project root with venv active):
    python scripts/03_battery_grid_check.py
"""

import os
import sys

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.battery.battery_model import BatteryModel
from src.grid.grid_model import GridModel, TariffSlab


# ─────────────────────────────────────────────────────────────────────────────
# Battery checks
# ─────────────────────────────────────────────────────────────────────────────

def check_battery() -> None:
    logger.info("\n" + "═" * 55)
    logger.info("  BATTERY MODEL — Sanity Checks")
    logger.info("═" * 55)

    battery = BatteryModel(
        capacity_kwh=10.0,
        max_charge_rate_kw=3.0,
        max_discharge_rate_kw=3.0,
        round_trip_efficiency=0.90,
        min_soc=0.10,
        max_soc=0.95,
        initial_soc=0.50,
    )
    logger.info(f"Initial: {battery}")

    # --- Test 1: basic charge ---
    step = battery.step(charge_kw=2.0, dt_hours=0.25)
    expected_soc_delta = (2.0 * battery.eta_c * 0.25) / 10.0
    logger.info(
        f"[1] Charge 2kW/15min: SOC {step.soc_before:.4f} → {step.soc_after:.4f} "
        f"(Δ={step.soc_after-step.soc_before:.4f}, expected≈{expected_soc_delta:.4f})"
    )
    assert abs((step.soc_after - step.soc_before) - expected_soc_delta) < 1e-6, "SOC delta mismatch on charge"
    logger.success("    ✓ Charge SOC delta matches efficiency-adjusted calculation")

    # --- Test 2: basic discharge ---
    battery.reset()
    step = battery.step(discharge_kw=1.5, dt_hours=0.25)
    expected_soc_delta = -(1.5 * 0.25 / battery.eta_d) / 10.0
    logger.info(
        f"[2] Discharge 1.5kW/15min: SOC {step.soc_before:.4f} → {step.soc_after:.4f} "
        f"(Δ={step.soc_after-step.soc_before:.4f}, expected≈{expected_soc_delta:.4f})"
    )
    assert abs((step.soc_after - step.soc_before) - expected_soc_delta) < 1e-6, "SOC delta mismatch on discharge"
    logger.success("    ✓ Discharge SOC delta matches efficiency-adjusted calculation")

    # --- Test 3: hardware rate limit enforced ---
    battery.reset()
    step = battery.step(charge_kw=10.0, dt_hours=0.25)   # request way above max
    logger.info(f"[3] Request 10kW charge (max=3kW): actual={step.actual_charge_kw:.2f}kW")
    assert step.actual_charge_kw <= battery.max_charge_rate_kw + 1e-9, "Charge rate limit violated"
    assert step.charge_curtailed_kw > 0, "Curtailment should be reported"
    logger.success(f"    ✓ Charge clamped to {step.actual_charge_kw:.2f}kW, curtailed={step.charge_curtailed_kw:.2f}kW")

    # --- Test 4: SOC upper bound enforced ---
    battery.reset(initial_soc=0.94)   # near max_soc=0.95
    step = battery.step(charge_kw=3.0, dt_hours=1.0)     # would overshoot without clamping
    logger.info(f"[4] Near-full battery (SOC=0.94) forced charge: SOC → {step.soc_after:.4f}")
    assert step.soc_after <= battery.max_soc + 1e-9, "SOC exceeded max_soc bound"
    logger.success(f"    ✓ SOC correctly clamped at max_soc={battery.max_soc}")

    # --- Test 5: SOC lower bound enforced ---
    battery.reset(initial_soc=0.11)   # near min_soc=0.10
    step = battery.step(discharge_kw=3.0, dt_hours=1.0)  # would overshoot without clamping
    logger.info(f"[5] Near-empty battery (SOC=0.11) forced discharge: SOC → {step.soc_after:.4f}")
    assert step.soc_after >= battery.min_soc - 1e-9, "SOC violated min_soc bound"
    logger.success(f"    ✓ SOC correctly clamped at min_soc={battery.min_soc}")

    # --- Test 6: round-trip efficiency loss ---
    # Start near-empty so this cycle's charge is isolated from any pre-existing
    # stored energy (comparing against a mixed reservoir would be an invalid
    # apples-to-oranges check, since the battery can't distinguish "old" vs
    # "new" energy in storage).
    battery.reset(initial_soc=battery.min_soc)
    charge_step = battery.step(charge_kw=2.0, dt_hours=1.0)
    # Discharge exactly this cycle's worth back down toward min_soc
    discharge_step = battery.step(discharge_kw=2.0, dt_hours=1.0)

    energy_in  = charge_step.actual_charge_kw * 1.0     # kWh drawn FROM SOURCE (pre-efficiency)
    energy_out = discharge_step.actual_discharge_kw * 1.0  # kWh delivered TO LOAD (post-efficiency)
    round_trip_loss_pct = 100 * (1 - energy_out / energy_in)

    logger.info(
        f"[6] Round-trip: {energy_in:.3f}kWh from source → {energy_out:.3f}kWh to load "
        f"(loss={round_trip_loss_pct:.1f}%, expected≈{100*(1-battery.round_trip_efficiency):.1f}%)"
    )
    assert energy_out < energy_in, "Round-trip should lose energy, not gain it"
    assert abs(round_trip_loss_pct - 100*(1-battery.round_trip_efficiency)) < 1.0, \
        "Round-trip loss doesn't match configured RTE within tolerance"
    logger.success(f"    ✓ Round-trip loss ({round_trip_loss_pct:.1f}%) matches configured RTE={battery.round_trip_efficiency:.0%}")

    logger.success("\n  ALL BATTERY CHECKS PASSED ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
# Grid checks
# ─────────────────────────────────────────────────────────────────────────────

def check_grid() -> None:
    logger.info("═" * 55)
    logger.info("  GRID MODEL — Sanity Checks")
    logger.info("═" * 55)

    grid = GridModel()
    logger.info(f"Initial: {grid}")

    # --- Test 1: marginal rate at zero consumption ---
    rate = grid.marginal_rate(0)
    logger.info(f"[1] Marginal rate at 0 units: ₹{rate}/kWh")
    assert rate == 3.35, f"Expected first slab rate 3.35, got {rate}"
    logger.success("    ✓ First slab rate correct")

    # --- Test 2: marginal rate just past a slab boundary ---
    rate = grid.marginal_rate(100.5)
    logger.info(f"[2] Marginal rate at 100.5 units (just past 100): ₹{rate}/kWh")
    assert rate == 3.85, f"Expected second slab rate 3.85, got {rate}"
    logger.success("    ✓ Slab transition correct")

    # --- Test 3: marginal rate deep in the highest slab ---
    rate = grid.marginal_rate(1000)
    logger.info(f"[3] Marginal rate at 1000 units (top slab): ₹{rate}/kWh")
    assert rate == 6.00, f"Expected top slab rate 6.00, got {rate}"
    logger.success("    ✓ Top (unbounded) slab correct")

    # --- Test 4: full month bill — manual cross-check ---
    # 310 units: 100@3.35 + 50@3.85 + 150@5.00 + 10@5.50 = 335+192.5+750+55 = 1332.5
    bill = grid.bill_for_month(310)
    expected_energy = 100*3.35 + 50*3.85 + 150*5.00 + 10*5.50
    logger.info(f"[4] Bill for 310 units: energy=₹{bill['energy_charge_inr']}, total=₹{bill['total_bill_inr']}")
    logger.info(f"    Manual calc expected energy charge: ₹{expected_energy:.2f}")
    assert abs(bill["energy_charge_inr"] - expected_energy) < 0.01, "Bill calculation mismatch"
    assert abs(bill["total_bill_inr"] - (expected_energy + grid.fixed_charge_inr)) < 0.01
    logger.success("    ✓ Slab-based bill matches manual calculation")

    # --- Test 5: outage lookup with real data ---
    dummy_index = pd.date_range("2024-01-01", periods=96, freq="15min")
    dummy_outages = pd.Series([True]*10 + [False]*86, index=dummy_index)
    grid_with_data = GridModel(outage_lookup=dummy_outages)
    ts_outage    = dummy_index[5]    # within the True block
    ts_no_outage = dummy_index[50]   # within the False block
    logger.info(
        f"[5] Real outage lookup: ts={ts_outage} → {grid_with_data.is_outage(ts_outage)}, "
        f"ts={ts_no_outage} → {grid_with_data.is_outage(ts_no_outage)}"
    )
    assert grid_with_data.is_outage(ts_outage) == True
    assert grid_with_data.is_outage(ts_no_outage) == False
    logger.success("    ✓ Real outage lookup returns correct values")

    # --- Test 6: fallback outage model is deterministic ---
    ts = pd.Timestamp("2025-06-15 14:00:00")
    result_1 = grid.is_outage(ts)
    result_2 = grid.is_outage(ts)
    logger.info(f"[6] Fallback outage determinism: call1={result_1}, call2={result_2}")
    assert result_1 == result_2, "Fallback outage model must be deterministic for same timestamp"
    logger.success("    ✓ Fallback outage model is deterministic")

    # --- Test 7: fallback outage rate roughly matches configured average ---
    test_dates = pd.date_range("2025-01-01", periods=30, freq="D")
    test_hours = range(24)
    outage_count = sum(
        grid.is_outage(pd.Timestamp(d) + pd.Timedelta(hours=h))
        for d in test_dates for h in test_hours
    )
    total_slots = len(test_dates) * 24
    observed_hours_per_day = 24 * outage_count / total_slots
    logger.info(
        f"[7] Fallback outage rate over 30 days: {observed_hours_per_day:.1f} hrs/day "
        f"(configured target={grid._fallback_outage_hours_per_day} hrs/day)"
    )
    # Loose tolerance — this is a statistical approximation, not exact
    assert 1.0 <= observed_hours_per_day <= 8.0, "Fallback outage rate wildly off target"
    logger.success("    ✓ Fallback outage rate roughly matches configured target")

    logger.success("\n  ALL GRID CHECKS PASSED ✓\n")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("━" * 55)
    logger.info("  IEMS Battery + Grid Model Validation  —  START")
    logger.info("━" * 55)

    check_battery()
    check_grid()

    logger.success("━" * 55)
    logger.success("  ALL CHECKS PASSED — models are ready for the")
    logger.success("  Gymnasium environment.")
    logger.success("━" * 55)


if __name__ == "__main__":
    main()