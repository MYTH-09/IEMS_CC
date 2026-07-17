"""
IEMS – Load Profile Generator
==============================
Extracts representative load profiles from the processed meter data,
clusters them into household archetypes, and synthesises realistic
15-min load profiles for new (unseen) customers.

This feeds directly into the Gymnasium simulation environment: when a
contractor inputs a customer's estimated monthly consumption, this module
generates the load time-series used during simulation.

Workflow
--------
1. extract_profiles()     → 96-slot daily profiles per meter × season × weekday/weekend
2. cluster_profiles()     → K-Means clusters (household archetypes)
3. generate_profile()     → Synthetic profile for a given daily kWh target
4. save / load            → Persist fitted model to disk

Cluster labels (defaults, K=5)
-------------------------------
0 – Ultra-low  (<2 kWh/day)  : single-room, basic appliances
1 – Low        (2–4 kWh/day) : 1BHK, fans + lights + phone
2 – Medium     (4–7 kWh/day) : 2BHK, fridge + TV + cooler
3 – High       (7–12 kWh/day): 3BHK, AC + washing machine
4 – Very-high  (>12 kWh/day) : large household / small commercial
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ── Constants ─────────────────────────────────────────────────────────────────
SLOTS_PER_DAY   = 96          # 15-min intervals in 24 hours
DEFAULT_K       = 5           # Number of household archetypes
DEFAULT_SEED    = 42
NOISE_STD_FRAC  = 0.08        # Gaussian noise as fraction of profile value (8 %)
MIN_DAYS_PER_METER = 30       # Minimum days a meter must have to be included


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Extract profiles
# ─────────────────────────────────────────────────────────────────────────────

def extract_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average normalised 96-slot daily load profiles per meter,
    grouped by season and day-type (weekday / weekend).

    Normalisation: each profile is divided by its daily sum so that shape
    (temporal distribution) is independent of consumption level.
    This allows clustering on *how* a household uses electricity, not *how much*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocessor.run_pipeline() — must contain:
        timestamp, energy_kwh, meter_id, city, season,
        slot_of_day, is_weekend, is_outage

    Returns
    -------
    pd.DataFrame
        One row per (meter_id, city, season, is_weekend) combination.
        Columns include:
            - slot_0 … slot_95    : normalised average energy fraction per slot
            - daily_kwh           : mean daily consumption (absolute scale)
            - peak_kw             : mean daily peak power
            - n_days              : number of days averaged
    """
    logger.info("Extracting load profiles ...")

    # Exclude outage slots from profile (we model outages separately)
    df_clean = df[~df["is_outage"]].copy()
    df_clean["date"] = df_clean["timestamp"].dt.date

    # --- per-meter, per-day stats for filtering ---
    meter_day_counts = (
        df_clean.groupby("meter_id", observed=True)["date"]
        .nunique()
        .rename("n_days")
    )
    valid_meters = meter_day_counts[meter_day_counts >= MIN_DAYS_PER_METER].index
    logger.info(
        f"  Meters with ≥ {MIN_DAYS_PER_METER} days: "
        f"{len(valid_meters)}/{df['meter_id'].nunique()}"
    )
    df_clean = df_clean[df_clean["meter_id"].isin(valid_meters)]

    # --- pivot: rows = (meter, city, season, is_weekend, date), cols = slots ---
    pivot = (
        df_clean
        .groupby(
            ["meter_id", "city", "season", "is_weekend", "date", "slot_of_day"],
            observed=True,
        )["energy_kwh"]
        .sum()
        .unstack("slot_of_day", fill_value=0.0)  # → columns 0-95
    )
    pivot.columns = [f"slot_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Add daily total (absolute scale reference)
    slot_cols = [f"slot_{i}" for i in range(SLOTS_PER_DAY)]
    # Handle missing slots (meter had no data for some slots that day)
    for col in slot_cols:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot["daily_kwh"] = pivot[slot_cols].sum(axis=1)

    # Normalise each day's profile (shape only; avoid div-by-zero on zero days)
    eps = 1e-6
    pivot_vals    = pivot[slot_cols].values
    daily_totals  = pivot["daily_kwh"].values[:, np.newaxis]
    pivot[slot_cols] = np.where(
        daily_totals > eps,
        pivot_vals / (daily_totals + eps),
        pivot_vals,
    )

    # --- Average across days per (meter, city, season, is_weekend) ---
    profiles = (
        pivot.groupby(["meter_id", "city", "season", "is_weekend"], observed=True)
             .agg(
                 **{col: (col, "mean") for col in slot_cols},
                 daily_kwh = ("daily_kwh", "mean"),
                 n_days    = ("daily_kwh", "count"),
             )
             .reset_index()
    )

    # Peak power (kW) = max slot energy × 4  (energy_kwh × 60/15)
    max_slot_energy = profiles[slot_cols].max(axis=1)
    profiles["peak_kw"] = (max_slot_energy * profiles["daily_kwh"] * 4).astype("float32")

    logger.info(f"Profiles extracted: {len(profiles)} (meter × season × day-type combos)")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Cluster into archetypes
# ─────────────────────────────────────────────────────────────────────────────

def cluster_profiles(
    profiles: pd.DataFrame,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, KMeans]:
    """
    K-Means cluster the normalised load profiles into `k` archetypes.

    The shape vectors (slot_0 … slot_95) are L2-normalised before clustering
    so that cluster membership is driven purely by temporal shape, not scale.

    Parameters
    ----------
    profiles : pd.DataFrame
        Output of extract_profiles().
    k : int
        Number of clusters (household archetypes).

    Returns
    -------
    (profiles_with_labels, fitted_kmeans)
        profiles_with_labels : original DataFrame + 'cluster' column (int)
        fitted_kmeans        : fitted sklearn KMeans (saved for inference)
    """
    logger.info(f"Clustering profiles (K={k}) ...")
    slot_cols = [f"slot_{i}" for i in range(SLOTS_PER_DAY)]

    X = profiles[slot_cols].values.astype("float64")
    X_norm = normalize(X, norm="l2")          # L2-norm for shape clustering

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    profiles = profiles.copy()
    profiles["cluster"] = labels.astype("int8")

    # Log cluster summary
    summary = (
        profiles.groupby("cluster")["daily_kwh"]
        .agg(["mean", "min", "max", "count"])
        .round(2)
    )
    logger.info(f"Cluster summary (daily kWh):\n{summary.to_string()}")

    return profiles, kmeans


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Generate synthetic profile for a new household
# ─────────────────────────────────────────────────────────────────────────────

def generate_profile(
    profiles: pd.DataFrame,
    kmeans: KMeans,
    target_daily_kwh: float,
    season: str = "summer",
    is_weekend: bool = False,
    n_days: int = 365,
    noise_std_frac: float = NOISE_STD_FRAC,
    seed: Optional[int] = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate a synthetic load profile for a new household.

    Given a target average daily consumption, this function:
      1. Picks the best-matching cluster (closest mean daily kWh).
      2. Retrieves the cluster's normalised shape vector.
      3. Scales it to `target_daily_kwh`.
      4. Tiles it across `n_days` and adds Gaussian noise.

    Parameters
    ----------
    profiles          : pd.DataFrame — output of cluster_profiles()
    kmeans            : fitted KMeans model
    target_daily_kwh  : float — customer's expected daily consumption (kWh)
    season            : str — 'summer' | 'winter' | 'monsoon' | 'post_monsoon'
    is_weekend        : bool — generate a weekday or weekend profile
    n_days            : int — number of simulation days to generate
    noise_std_frac    : float — Gaussian noise std as fraction of signal (0.08=8%)
    seed              : Optional[int] — for reproducibility

    Returns
    -------
    np.ndarray, shape (n_days × 96,)
        Energy per 15-min slot in kWh. Array is contiguous in time:
        [day0_slot0, day0_slot1, …, day0_slot95, day1_slot0, …]

    Usage
    -----
    profile = generate_profile(profiles, kmeans, target_daily_kwh=5.0,
                               season='summer', n_days=365)
    # profile.shape → (35040,)   # 365 × 96
    """
    rng = np.random.default_rng(seed)
    slot_cols = [f"slot_{i}" for i in range(SLOTS_PER_DAY)]

    # Filter to matching season and day-type
    mask = (profiles["season"] == season) & (profiles["is_weekend"] == is_weekend)
    subset = profiles[mask]

    if len(subset) == 0:
        logger.warning(
            f"No profiles found for season='{season}', is_weekend={is_weekend}. "
            "Falling back to all-season average."
        )
        subset = profiles[profiles["is_weekend"] == is_weekend]
    if len(subset) == 0:
        subset = profiles   # last resort: use everything

    # Find cluster whose mean daily_kwh is closest to target
    cluster_means = subset.groupby("cluster")["daily_kwh"].mean()
    best_cluster  = (cluster_means - target_daily_kwh).abs().idxmin()

    # Get the mean normalised shape for that cluster
    cluster_rows  = subset[subset["cluster"] == best_cluster]
    shape_vector  = cluster_rows[slot_cols].mean().values   # (96,)

    # Scale shape to target_daily_kwh
    # shape_vector sums to ~1 (normalised), so scale by target
    shape_sum = shape_vector.sum()
    if shape_sum > 1e-6:
        shape_vector = shape_vector / shape_sum    # re-normalise to exactly 1
    scaled_day = shape_vector * target_daily_kwh   # kWh per 15-min slot

    # Tile across n_days
    profile = np.tile(scaled_day, n_days)          # (n_days × 96,)

    # Add Gaussian noise (proportional to local value, floor at 0)
    noise = rng.normal(loc=0.0, scale=noise_std_frac * (scaled_day.mean() + 1e-6), size=profile.shape)
    profile = np.maximum(profile + noise, 0.0).astype("float32")

    logger.debug(
        f"Generated {n_days}-day profile | cluster={best_cluster} | "
        f"target={target_daily_kwh:.2f} kWh/day | "
        f"actual mean={profile.sum()/n_days:.2f} kWh/day"
    )
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(profiles: pd.DataFrame, kmeans: KMeans, out_dir: str = "models") -> None:
    """Save fitted profiles and KMeans model to disk."""
    os.makedirs(out_dir, exist_ok=True)
    profiles.to_parquet(os.path.join(out_dir, "load_profiles.parquet"), index=False)
    with open(os.path.join(out_dir, "profile_kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    logger.success(f"Profile model saved → {out_dir}/")


def load_model(model_dir: str = "models") -> tuple[pd.DataFrame, KMeans]:
    """Load fitted profiles and KMeans model from disk."""
    profiles_path = os.path.join(model_dir, "load_profiles.parquet")
    kmeans_path   = os.path.join(model_dir, "profile_kmeans.pkl")

    if not (os.path.exists(profiles_path) and os.path.exists(kmeans_path)):
        raise FileNotFoundError(
            f"Profile model not found in '{model_dir}/'. "
            "Run scripts/01_data_pipeline.py first."
        )

    profiles = pd.read_parquet(profiles_path)
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    logger.info(f"Profile model loaded ← {model_dir}/  ({len(profiles)} profiles, K={kmeans.n_clusters})")
    return profiles, kmeans
