"""
IEMS – Data Preprocessor
========================
Transforms the raw combined DataFrame into a clean, 15-minute interval
dataset ready for the simulation environment and profile generator.

Pipeline
--------
1. validate()           – type safety, drop physically impossible rows
2. add_power_kw()       – derive instantaneous power from energy per interval
3. resample_to_15min()  – 3-min → 15-min (sum energy, mean electrical params)
4. detect_outages()     – flag grid power-cut slots
5. add_time_features()  – hour, weekday, season, slot_of_day, etc.

All public functions are pure (return new DataFrames, don't mutate input).
Call `run_pipeline(df)` to execute all steps in order.
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────
RAW_INTERVAL_MIN    = 3      # Source data resolution
SIM_INTERVAL_MIN    = 15     # Target simulation resolution
SLOTS_PER_DAY       = 24 * 60 // SIM_INTERVAL_MIN   # 96 slots per day

# Outage detection thresholds
# A 15-min slot is flagged as outage when resampled energy is below this
# (5 × 3-min readings summed; 0.001 kWh = minimum meter resolution)
OUTAGE_ENERGY_THRESH_KWH = 0.001
# How many consecutive outage slots before we confirm (avoid single noise hits)
OUTAGE_MIN_SLOTS    = 1      # Even a single 15-min zero is a cut in Indian grid context
# Daytime window: 06:00-22:00. Zeros here are almost certainly outages.
OUTAGE_DAYTIME_START = 6
OUTAGE_DAYTIME_END   = 22
# At night (22:00-06:00) a zero might be genuine zero standby; we flag anyway
# but the simulation will treat nighttime outages differently.

# Voltage sanity bounds (sensor error outside this range)
VOLTAGE_MIN = 50.0    # V — genuine brownout (not data error)
VOLTAGE_MAX = 350.0   # V — above this is a sensor fault

# Indian meteorological seasons (month → season)
_SEASON_MAP: dict[int, str] = {
    1: "winter",      2: "winter",     12: "winter",
    3: "summer",      4: "summer",      5: "summer",
    6: "monsoon",     7: "monsoon",     8: "monsoon",   9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Validate
# ─────────────────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce types, drop physically impossible rows, clamp sensor outliers.

    Checks performed
    ----------------
    * timestamp is datetime64
    * energy_kwh ≥ 0  (negative energy is a sensor fault)
    * voltage_v  clamped to [VOLTAGE_MIN, VOLTAGE_MAX]
    * current_a  ≥ 0
    """
    df = df.copy()

    # --- Timestamp ---
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Energy: drop negative rows ---
    neg_mask = df["energy_kwh"] < 0
    if neg_mask.any():
        logger.warning(f"Dropping {neg_mask.sum():,} rows with negative energy_kwh")
        df = df[~neg_mask].copy()

    # --- Voltage: clamp outliers (don't drop — keeps timeline intact) ---
    bad_v = (df["voltage_v"] < VOLTAGE_MIN) | (df["voltage_v"] > VOLTAGE_MAX)
    if bad_v.any():
        logger.debug(f"Clamping {bad_v.sum():,} voltage outliers to [{VOLTAGE_MIN}, {VOLTAGE_MAX}]")
        df["voltage_v"] = df["voltage_v"].clip(VOLTAGE_MIN, VOLTAGE_MAX)

    # --- Current: clamp negatives ---
    df["current_a"] = df["current_a"].clip(lower=0)

    logger.info(f"Validate: {len(df):,} rows retained")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Derive power_kw
# ─────────────────────────────────────────────────────────────────────────────

def add_power_kw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive average power (kW) from energy per 3-min interval.

    Formula:  power_kW = energy_kWh / (interval_min / 60)
              power_kW = energy_kWh × (60 / 3)
              power_kW = energy_kWh × 20

    Example:  0.017 kWh in 3 min → 0.017 × 20 = 0.34 kW = 340 W  ✓
    """
    df = df.copy()
    df["power_kw"] = (df["energy_kwh"] * (60.0 / RAW_INTERVAL_MIN)).astype("float32")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Resample 3-min → 15-min
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate each meter from 3-min to 15-min intervals.

    Aggregation rules
    -----------------
    * energy_kwh  → sum   (total energy in the 15-min window)
    * voltage_v   → mean
    * current_a   → mean
    * frequency_hz→ mean
    * power_kw    → re-derived from resampled energy (more accurate than mean)
    * city        → first (constant per meter)
    * data_year   → first (constant per meter)

    The resulting power_kw is the average power over the 15-min window:
        power_kw = energy_kwh / (15/60) = energy_kwh × 4
    """
    logger.info("Resampling 3-min → 15-min ...")

    df = df.set_index("timestamp")

    agg: dict = {
        "energy_kwh":   "sum",
        "voltage_v":    "mean",
        "current_a":    "mean",
        "frequency_hz": "mean",
        "city":         "first",
        "data_year":    "first",
    }

    resampled = (
        df.groupby("meter_id", observed=True)
          .resample(f"{SIM_INTERVAL_MIN}min")
          .agg(agg)
          .reset_index()
    )

    # Re-derive power from the 15-min energy sum
    resampled["power_kw"] = (
        resampled["energy_kwh"] * (60.0 / SIM_INTERVAL_MIN)
    ).astype("float32")

    resampled["energy_kwh"] = resampled["energy_kwh"].astype("float32")

    # Re-derive city from meter_id prefix (BR=Bareilly, MH=Mathura).
    # The "first" aggregation above leaves NaN for gap-filled slots created
    # by resample() — this fixes that cleanly without a forward-fill loop.
    _CITY_PREFIX = {"BR": "Bareilly", "MH": "Mathura"}
    resampled["city"] = (
        resampled["meter_id"].astype(str).str[:2].map(_CITY_PREFIX)
    )
    resampled["city"] = pd.Categorical(resampled["city"])

    # Also forward-fill data_year (same gap-fill issue for integer column)
    resampled["data_year"] = (
        resampled.groupby("meter_id", observed=True)["data_year"]
        .transform(lambda s: s.ffill().bfill())
        .astype("int16")
    )

    logger.info(
        f"Resampled: {len(resampled):,} rows  "
        f"(ratio {len(df)/len(resampled):.1f}×  ≈5× expected)"
    )
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Detect outages
# ─────────────────────────────────────────────────────────────────────────────

def detect_outages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag 15-min slots where the grid was unavailable (load-shedding / fault).

    Heuristic
    ---------
    Primary rule (confident):
        energy_kwh < OUTAGE_ENERGY_THRESH_KWH during daytime (06:00–22:00)
        → Indian grid load-shedding rarely causes all-night outages; daytime
          zeros are almost always a cut.

    Secondary rule (conservative):
        energy_kwh < OUTAGE_ENERGY_THRESH_KWH at night AND voltage_v < 150 V
        → Only flag nighttime zeros when voltage confirms no supply.
          (Nighttime standby ≈ 0.001 kWh/15min which passes the threshold,
           so genuine standby won't be mislabelled.)

    Adds columns
    ------------
    is_outage (bool)
        True → no grid power in this slot.
    outage_run_id (int32)
        Monotonically increasing integer; each contiguous outage block per
        meter gets a unique ID. Useful for computing outage duration.
    """
    logger.info("Detecting outages ...")
    df = df.copy()

    hour         = df["timestamp"].dt.hour
    daytime      = (hour >= OUTAGE_DAYTIME_START) & (hour < OUTAGE_DAYTIME_END)
    zero_energy  = df["energy_kwh"] < OUTAGE_ENERGY_THRESH_KWH
    low_voltage  = df["voltage_v"]  < 150.0   # V — definitive brownout

    # Combine rules
    df["is_outage"] = (
        (zero_energy & daytime)           |   # daytime zero → outage
        (zero_energy & ~daytime & low_voltage)  # nighttime zero + low voltage → outage
    )

    # Assign run IDs per meter so duration can be computed downstream
    run_id = np.zeros(len(df), dtype="int32")
    current_id = 0
    for _, grp in df.groupby("meter_id", observed=True, sort=False):
        idx = grp.index
        prev_outage = False
        for i in idx:
            if df.at[i, "is_outage"]:
                if not prev_outage:
                    current_id += 1
                run_id[i] = current_id
                prev_outage = True
            else:
                prev_outage = False
    df["outage_run_id"] = run_id

    # Summary statistics
    n_outage  = df["is_outage"].sum()
    n_total   = len(df)
    n_meters  = df["meter_id"].nunique()
    avg_hrs   = (n_outage * SIM_INTERVAL_MIN / 60) / n_meters / df["data_year"].nunique()
    logger.info(
        f"Outage slots: {n_outage:,}/{n_total:,}  "
        f"({100*n_outage/n_total:.1f}%)  "
        f"≈ {avg_hrs:.1f} hrs/meter/year"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Time features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal feature columns consumed by the RL environment and profiler.

    New columns
    -----------
    hour (int8)          : 0–23
    minute (int8)        : 0, 15, 30, 45
    day_of_week (int8)   : 0=Mon … 6=Sun
    is_weekend (bool)    : Sat/Sun
    month (int8)         : 1–12
    day_of_year (int16)  : 1–366
    slot_of_day (int8)   : 0–95  (which 15-min slot within the day)
    season (category)    : winter | summer | monsoon | post_monsoon
    """
    df  = df.copy()
    ts  = df["timestamp"]

    df["hour"]        = ts.dt.hour.astype("int8")
    df["minute"]      = ts.dt.minute.astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["is_weekend"]  = (df["day_of_week"] >= 5)
    df["month"]       = ts.dt.month.astype("int8")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")
    df["slot_of_day"] = (df["hour"] * 4 + df["minute"] // 15).astype("int8")
    df["season"]      = pd.Categorical(df["month"].map(_SEASON_MAP))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all preprocessing steps in order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame from loader.load_all_files().

    Returns
    -------
    pd.DataFrame
        Clean, 15-min interval DataFrame with full feature set.
    """
    logger.info("=" * 55)
    logger.info("IEMS Preprocessing Pipeline — START")
    logger.info("=" * 55)

    df = validate(df)
    df = add_power_kw(df)
    df = resample_to_15min(df)
    df = detect_outages(df)
    df = add_time_features(df)

    logger.success("=" * 55)
    logger.success(f"Pipeline complete  →  shape: {df.shape}")
    logger.success(
        f"Date range : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    )
    logger.success(f"Meters     : {df['meter_id'].nunique()}")
    logger.success(f"Cities     : {df['city'].unique().tolist()}")
    logger.success("=" * 55)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-meter, per-day aggregate statistics.

    Returns columns:
        meter_id, city, date, total_kwh, avg_power_kw, peak_power_kw,
        avg_voltage_v, outage_hours, supply_hours
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    summary = (
        df.groupby(["meter_id", "city", "date"], observed=True)
          .agg(
              total_kwh      = ("energy_kwh",  "sum"),
              avg_power_kw   = ("power_kw",    "mean"),
              peak_power_kw  = ("power_kw",    "max"),
              avg_voltage_v  = ("voltage_v",   "mean"),
              outage_slots   = ("is_outage",   "sum"),
              total_slots    = ("is_outage",   "count"),
          )
          .reset_index()
    )

    summary["outage_hours"] = summary["outage_slots"] * SIM_INTERVAL_MIN / 60
    summary["supply_hours"] = (
        (summary["total_slots"] - summary["outage_slots"]) * SIM_INTERVAL_MIN / 60
    )
    return summary