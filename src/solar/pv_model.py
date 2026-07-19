"""
IEMS – PV Generation Model
===========================
Converts NASA POWER hourly irradiance data into 15-min AC power output
for a configurable solar PV system using pvlib's PVWatts model.

Physics chain
-------------
1. Solar position       → zenith, azimuth at each timestamp
2. POA irradiance       → plane-of-array GHI from GHI/DNI/DHI + tilt/azimuth
3. Cell temperature     → Faiman model (POA + ambient temp + wind)
4. DC power (PVWatts)   → accounts for temperature derating
5. AC power             → apply system losses + inverter efficiency
6. Resample hourly→15min (forward-fill irradiance, linear interp for met)

PVWatts equations (NREL)
------------------------
P_dc = GHI_poa × (1 - losses) × η_ref × (1 + γ × (T_cell - 25))
                         ↑ pdc0 / (1000 W/m² × area)
P_ac = P_dc × η_inverter

References
----------
[1] Dobos, A. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641.
[2] Faiman, D. (2008). Assessing the outdoor operating temperature of
    photovoltaic modules. Progress in Photovoltaics.
[3] pvlib documentation: https://pvlib-python.readthedocs.io
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
from pvlib.temperature import faiman
from pvlib.pvsystem import pvwatts_dc
from pvlib.inverter import pvwatts as pvwatts_ac   # moved from pvsystem in pvlib ≥ 0.9
from loguru import logger
from typing import Optional


# ── Default system parameters (configurable via default_config.yaml) ──────────
DEFAULT_SYSTEM = {
    "capacity_kw":           3.0,    # Nameplate DC capacity (kWp)
    "tilt_deg":              27.0,   # Panel tilt angle (≈ latitude for max annual)
    "azimuth_deg":           180.0,  # 180° = true south
    "gamma_pdc":             -0.004, # Temperature coefficient of power (%/°C)
    "system_losses":         0.14,   # Total system losses (wiring, soiling, shading)
    "inverter_efficiency":   0.96,   # Inverter peak efficiency
    "altitude_m":            182.0,  # Site altitude (affects air mass)
}


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────


def simulate_pv(
    weather_hourly: pd.DataFrame,
    lat: float,
    lon: float,
    capacity_kw: float          = DEFAULT_SYSTEM["capacity_kw"],
    tilt_deg: float             = DEFAULT_SYSTEM["tilt_deg"],
    azimuth_deg: float          = DEFAULT_SYSTEM["azimuth_deg"],
    gamma_pdc: float            = DEFAULT_SYSTEM["gamma_pdc"],
    system_losses: float        = DEFAULT_SYSTEM["system_losses"],
    inverter_efficiency: float  = DEFAULT_SYSTEM["inverter_efficiency"],
    altitude_m: float           = DEFAULT_SYSTEM["altitude_m"],
) -> pd.DataFrame:
    """
    Simulate AC power output for a PV system at 15-min IST resolution.

    Parameters
    ----------
    weather_hourly : pd.DataFrame
        Output of nasa_power.fetch_solar_data() — hourly IST data with
        columns: timestamp, ghi, dni, dhi, temp_air, wind_speed.
    lat, lon       : Site coordinates (decimal degrees, WGS84).
    capacity_kw    : Nameplate DC capacity in kWp.
    tilt_deg       : Panel tilt from horizontal (degrees).
    azimuth_deg    : Panel azimuth (180 = south, 270 = west).
    gamma_pdc      : Temperature coefficient of power (negative, /°C).
    system_losses  : Fractional losses from wiring, soiling, etc. (0–1).
    inverter_efficiency : Inverter peak efficiency (0–1).
    altitude_m     : Site elevation above sea level (metres).

    Returns
    -------
    pd.DataFrame
        15-min IST rows, columns:
            timestamp       – 15-min IST datetimes
            solar_power_kw  – AC output (kW), clipped to [0, capacity_kw]
            solar_energy_kwh– Energy in the 15-min slot (kWh) = power × 0.25
            ghi, poa_global – Irradiance values (W/m²) for reference
            cell_temp_c     – Estimated cell temperature (°C)

    Notes
    -----
    IMPORTANT: all pvlib inputs are converted to plain numpy arrays before
    any computation. This avoids pandas index-alignment issues that arise
    when solar_pos (DatetimeIndex) and weather data (RangeIndex) are mixed
    in arithmetic — which causes silent shape explosions (29328 vs 14664).
    """
    logger.info(
        f"Simulating PV: {capacity_kw} kWp | "
        f"tilt={tilt_deg}° | azimuth={azimuth_deg}° | "
        f"lat={lat}, lon={lon}"
    )

    # ── Step 0: clean input ────────────────────────────────────────────────
    df = weather_hourly.copy().sort_values("timestamp").reset_index(drop=True)

    # Timezone-aware index required by pvlib for solar position accuracy
    times_ist = pd.DatetimeIndex(df["timestamp"]).tz_localize("Asia/Kolkata")

    # ── Step 1: Solar position  →  numpy arrays immediately ───────────────
    site = Location(
        latitude=lat,
        longitude=lon,
        tz="Asia/Kolkata",
        altitude=altitude_m,
    )
    solar_pos = site.get_solarposition(times_ist)

    # Convert to numpy — this is the critical fix.
    # solar_pos has a DatetimeIndex; df columns have RangeIndex.
    # Mixing them in pvlib internal arithmetic causes pandas to outer-join
    # the two indices → shape doubles from N to 2N with all NaN.
    zenith_arr  = solar_pos["apparent_zenith"].to_numpy(dtype=float)
    azimuth_arr = solar_pos["azimuth"].to_numpy(dtype=float)

    ghi_arr       = df["ghi"].fillna(0.0).to_numpy(dtype=float)
    dni_arr       = df["dni"].fillna(0.0).to_numpy(dtype=float)
    dhi_arr       = df["dhi"].fillna(0.0).to_numpy(dtype=float)
    temp_air_arr  = df["temp_air"].fillna(25.0).to_numpy(dtype=float)
    wind_arr      = df["wind_speed"].fillna(1.0).to_numpy(dtype=float)

    # ── Step 2: Extraterrestrial irradiance + air mass (both numpy) ───────
    dni_extra = pvlib.irradiance.get_extra_radiation(times_ist).to_numpy(dtype=float)
    airmass   = pvlib.atmosphere.get_relative_airmass(zenith_arr)   # already numpy

    # ── Step 3: Plane-of-Array irradiance (Perez model) ───────────────────
    poa = get_total_irradiance(
        surface_tilt    = tilt_deg,
        surface_azimuth = azimuth_deg,
        solar_zenith    = zenith_arr,
        solar_azimuth   = azimuth_arr,
        dni             = dni_arr,
        ghi             = ghi_arr,
        dhi             = dhi_arr,
        model           = "perez",
        dni_extra       = dni_extra,
        airmass         = airmass,
    )
    # poa is a dict or DataFrame; extract and convert to numpy
    poa_global = np.clip(np.asarray(poa["poa_global"], dtype=float), 0.0, None)

    # ── Step 4: Cell temperature (Faiman model) ───────────────────────────
    cell_temp = faiman(
        poa_global  = poa_global,
        temp_air    = temp_air_arr,
        wind_speed  = wind_arr,
    )
    # faiman returns a numpy array when given numpy inputs
    cell_temp = np.asarray(cell_temp, dtype=float)

    # ── Step 5: DC power (PVWatts) ────────────────────────────────────────
    pdc0_w = capacity_kw * 1000.0   # Rated DC power at STC (W)

    p_dc = pvwatts_dc(
        g_poa_effective = poa_global,
        temp_cell       = cell_temp,
        pdc0            = pdc0_w,
        gamma_pdc       = gamma_pdc,
    )
    p_dc = np.asarray(p_dc, dtype=float)

    # ── Step 6: AC power (inverter losses) ────────────────────────────────
    pdc0_inv = pdc0_w * (1.0 - system_losses)   # de-rate for system losses

    p_ac_w = pvwatts_ac(
        pdc         = p_dc,
        pdc0        = pdc0_inv,
        eta_inv_nom = inverter_efficiency,
    )
    p_ac_kw = np.clip(np.asarray(p_ac_w, dtype=float) / 1000.0, 0.0, capacity_kw)

    # ── Step 7: Assemble hourly DataFrame (plain timestamps, no tz) ───────
    result_hourly = pd.DataFrame({
        "timestamp":      df["timestamp"].values,           # naive IST
        "solar_power_kw": p_ac_kw.astype("float32"),
        "ghi":            ghi_arr.astype("float32"),
        "poa_global":     poa_global.astype("float32"),
        "cell_temp_c":    cell_temp.astype("float32"),
    })

    # ── Step 8: Resample hourly → 15-min ──────────────────────────────────
    result_15min = _resample_to_15min(result_hourly)
    result_15min["solar_energy_kwh"] = (
        result_15min["solar_power_kw"] / 4.0
    ).astype("float32")

    total_kwh = float(result_15min["solar_energy_kwh"].sum())
    logger.success(
        f"PV simulation complete: {len(result_15min):,} 15-min rows | "
        f"Total yield = {total_kwh:,.0f} kWh"
    )
    return result_15min


def _resample_to_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample hourly solar data to 15-min intervals.

    Irradiance / power  : forward-fill within each hour (step function —
                          more physically correct than linear interp for
                          hourly averages).
    Temperature         : linear interpolation (smooth variation expected).
    """
    # Defensive: guard against duplicate timestamps reaching this point
    # from any upstream source (this is a second safety net — the primary
    # fix is in nasa_power.load_cached()'s stricter file-matching pattern).
    if df_hourly["timestamp"].duplicated().any():
        n_dupes = df_hourly["timestamp"].duplicated().sum()
        logger.warning(f"_resample_to_15min: dropping {n_dupes} duplicate timestamp rows before resampling")
        df_hourly = df_hourly.drop_duplicates(subset="timestamp", keep="last")

    df = df_hourly.set_index("timestamp")

    # Create a 15-min target index spanning the same period
    start = df.index.min()
    end   = df.index.max() + pd.Timedelta("45min")  # complete last hour
    idx_15min = pd.date_range(start=start, end=end, freq="15min")

    # Reindex to 15-min, then fill
    df = df.reindex(df.index.union(idx_15min)).sort_index()

    # Forward-fill power/irradiance (hold hourly value for all 4 sub-slots)
    for col in ["solar_power_kw", "ghi", "poa_global"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Linear interpolate temperature (smoother physically)
    if "cell_temp_c" in df.columns:
        df["cell_temp_c"] = df["cell_temp_c"].interpolate("linear")

    # Keep only the 15-min grid
    df = df.reindex(idx_15min).reset_index().rename(columns={"index": "timestamp"})

    # Clip negative residuals
    df["solar_power_kw"] = df["solar_power_kw"].clip(lower=0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def annual_yield_kwh(solar_15min: pd.DataFrame) -> float:
    """Return total energy yield (kWh) from a 15-min solar DataFrame."""
    return float(solar_15min["solar_energy_kwh"].sum())


def capacity_factor(solar_15min: pd.DataFrame, capacity_kw: float) -> float:
    """
    Capacity factor = actual energy / theoretical maximum energy.
    For Indian rooftop solar, typical CF is 0.16–0.22.
    """
    max_possible = capacity_kw * len(solar_15min) * (15 / 60)
    return annual_yield_kwh(solar_15min) / max_possible if max_possible > 0 else 0.0


def save_solar_data(df: pd.DataFrame, path: str) -> None:
    """Save 15-min solar DataFrame to parquet."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"Solar data saved → {path}  ({size_mb:.1f} MB)")


def load_solar_data(path: str) -> pd.DataFrame:
    """Load a saved 15-min solar parquet file."""
    if not __import__("os").path.exists(path):
        raise FileNotFoundError(
            f"Solar data not found: '{path}'. "
            "Run scripts/02_solar_pipeline.py first."
        )
    df = pd.read_parquet(path)
    logger.info(f"Solar data loaded ← {path}  ({len(df):,} rows)")
    return df