"""
IEMS – Script 01: Data Pipeline
================================
Run this once to:
  1. Load all 4 raw CSVs from data/raw/
  2. Preprocess (validate, resample to 15-min, detect outages, add features)
  3. Save processed data to data/processed/meters_15min.parquet
  4. Extract and cluster load profiles (K=5 archetypes)
  5. Save profile model to models/

Usage (from project root with venv active):
    python scripts/01_data_pipeline.py

Expected runtime:  5–10 minutes (dominated by CSV loading & resampling)
Disk output:
    data/processed/meters_15min.parquet   (~80–100 MB)
    models/load_profiles.parquet          (~1 MB)
    models/profile_kmeans.pkl             (~5 KB)
"""

import os
import sys
import time

from loguru import logger

# Allow running from project root without pip install -e .
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader           import load_all_files, save_parquet
from src.data.preprocessor     import run_pipeline, daily_summary
from src.data.profile_generator import (
    extract_profiles,
    cluster_profiles,
    save_model,
    generate_profile,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR         = "data/raw"
PROCESSED_PATH  = "data/processed/meters_15min.parquet"
SUMMARY_PATH    = "data/processed/daily_summary.parquet"
MODEL_DIR       = "models"


def main() -> None:
    t0 = time.time()
    logger.info("━" * 55)
    logger.info("  IEMS Data Pipeline  —  START")
    logger.info("━" * 55)

    # ── 1. Load raw CSVs ─────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading raw CSV files ...")
    raw_df = load_all_files(RAW_DIR)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    logger.info("\n[2/5] Running preprocessing pipeline ...")
    processed_df = run_pipeline(raw_df)
    del raw_df   # free ~600 MB RAM

    # ── 3. Save processed data ───────────────────────────────────────────────
    logger.info("\n[3/5] Saving processed data ...")
    save_parquet(processed_df, PROCESSED_PATH)

    # Also save a lightweight daily summary (useful for dashboard)
    summary = daily_summary(processed_df)
    save_parquet(summary, SUMMARY_PATH)
    logger.info(f"Daily summary saved → {SUMMARY_PATH}  ({len(summary):,} rows)")

    # ── 4. Extract & cluster load profiles ──────────────────────────────────
    logger.info("\n[4/5] Extracting and clustering load profiles ...")
    profiles = extract_profiles(processed_df)
    profiles, kmeans = cluster_profiles(profiles, k=5)

    # ── 5. Save profile model ────────────────────────────────────────────────
    logger.info("\n[5/5] Saving profile model ...")
    save_model(profiles, kmeans, out_dir=MODEL_DIR)

    # ── Quick smoke-test: generate a sample profile ──────────────────────────
    logger.info("\n── Smoke test: generating a sample profile ──")
    test_profile = generate_profile(
        profiles, kmeans,
        target_daily_kwh=5.0,
        season="summer",
        is_weekend=False,
        n_days=7,
    )
    logger.info(
        f"  Sample 7-day profile: shape={test_profile.shape}  "
        f"mean={test_profile.mean()*4:.3f} kW  "
        f"peak={test_profile.max()*4:.3f} kW"
    )

    elapsed = time.time() - t0
    logger.success("\n" + "━" * 55)
    logger.success(f"  Pipeline COMPLETE in {elapsed/60:.1f} min")
    logger.success("━" * 55)
    logger.success("\n  Outputs:")
    logger.success(f"    {PROCESSED_PATH}")
    logger.success(f"    {SUMMARY_PATH}")
    logger.success(f"    {MODEL_DIR}/load_profiles.parquet")
    logger.success(f"    {MODEL_DIR}/profile_kmeans.pkl")
    logger.success("\n  Next: run scripts/02_solar_pipeline.py")


if __name__ == "__main__":
    main()
