"""
IEMS - Data Exploration Script
Run this ONCE to understand your raw CSV structure.
Usage:
    python explore_data.py

Output will tell us exactly what's in the files so we can
build the correct data loader.
"""

import pandas as pd
import os
import glob

RAW_DATA_DIR = "data/raw"

def explore_file(filepath: str) -> None:
    filename = os.path.basename(filepath)
    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    # Load just 5 rows first to check structure
    try:
        df = pd.read_csv(filepath, nrows=5)
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return

    print(f"\n  Shape (rows x cols):   {df.shape}")
    print(f"  Columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns):
        print(f"    [{i:2d}]  {col!r}")

    print(f"\n  Data types:")
    for col, dtype in df.dtypes.items():
        print(f"         {col!r:<40} {dtype}")

    print(f"\n  First 3 rows (raw):")
    print(df.head(3).to_string(index=False))

    # Load full file for basic stats
    print(f"\n  Loading full file for stats...")
    try:
        df_full = pd.read_csv(filepath, low_memory=False)
        print(f"  Total rows:            {len(df_full):,}")
        print(f"  Null counts per column:")
        for col in df_full.columns:
            nulls = df_full[col].isna().sum()
            pct   = 100 * nulls / len(df_full)
            if nulls > 0:
                print(f"         {col!r:<40} {nulls:,} nulls ({pct:.1f}%)")

        # Identify likely timestamp column
        print(f"\n  Unique values (first 5) per column:")
        for col in df_full.columns[:10]:   # cap at first 10 cols to keep output manageable
            uniq = df_full[col].dropna().unique()
            sample = uniq[:5].tolist()
            total_uniq = len(uniq)
            print(f"         {col!r:<40} {total_uniq} unique | sample: {sample}")

    except Exception as e:
        print(f"  ERROR on full load: {e}")


def main():
    csv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in '{RAW_DATA_DIR}/'.")
        print("Make sure you're running this from the project root (iems/).")
        return

    print(f"\nFound {len(csv_files)} CSV files in '{RAW_DATA_DIR}/':")
    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f):<55} {size_mb:.1f} MB")

    for filepath in csv_files:
        explore_file(filepath)

    print("\n" + "=" * 65)
    print("Exploration complete. Share this output to build the loader.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()