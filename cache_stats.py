"""
Fetch FanGraphs batting leaders via pybaseball and cache to CSV.
Run locally before deploying to Streamlit Cloud (pybaseball can't connect there).

Usage: python cache_stats.py
"""

import os
import pandas as pd
from pybaseball import batting_stats, cache
from datetime import datetime

cache.enable()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "batting_stats_cache.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    year = datetime.now().year

    # Try current year first, fall back to prior years
    for y in [year, year - 1, year - 2]:
        print(f"Fetching {y} batting leaders (min 50 PA)...")
        try:
            df = batting_stats(y, qual=50)
        except Exception as e:
            print(f"  Failed for {y}: {e}")
            continue

        if df is not None and len(df) >= 50:
            df.to_csv(OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {OUT_PATH} (season {y})")
            return

        print(f"  Only {len(df) if df is not None else 0} rows for {y}, trying earlier year...")

    print("ERROR: Could not fetch sufficient batting data for any recent season.")


if __name__ == "__main__":
    main()
