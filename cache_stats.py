"""
Fetch batting leaders and cache to CSV.
Run locally before deploying to Streamlit Cloud.

Tries FanGraphs via pybaseball first, falls back to MLB Stats API.

Usage: python cache_stats.py
"""

import os
import requests
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "batting_stats_cache.csv")

MIN_PA = 100


def _fetch_fangraphs(year: int) -> pd.DataFrame | None:
    """Fetch from FanGraphs via pybaseball."""
    try:
        from pybaseball import batting_stats, cache
        cache.enable()
        df = batting_stats(year, qual=MIN_PA)
        if df is not None and len(df) >= 50:
            # Normalise column subset to our standard schema
            keep = [c for c in ["Name","Team","G","PA","AB","H","2B","3B","HR","R","RBI","SB",
                                 "BB%","K%","AVG","OBP","SLG","wOBA","ISO","BABIP"] if c in df.columns]
            return df[keep]
    except Exception as e:
        print(f"  FanGraphs failed for {year}: {e}")
    return None


def _fetch_mlb_api(year: int) -> pd.DataFrame | None:
    """Fetch from MLB Stats API (free, no key)."""
    url = (
        f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
        f"&season={year}&sportId=1&limit=1000&offset=0&gameType=R"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  MLB API failed for {year}: {e}")
        return None

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})

        pa = int(stat.get("plateAppearances", 0))
        if pa < MIN_PA:
            continue

        ab = int(stat.get("atBats", 0))
        h = int(stat.get("hits", 0))
        doubles = int(stat.get("doubles", 0))
        triples = int(stat.get("triples", 0))
        hr = int(stat.get("homeRuns", 0))
        bb = int(stat.get("baseOnBalls", 0))
        hbp = int(stat.get("hitByPitch", 0))
        sf = int(stat.get("sacFlies", 0))
        so = int(stat.get("strikeOuts", 0))
        ibb = int(stat.get("intentionalWalks", 0))

        avg = h / ab if ab else 0
        obp = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) else 0
        singles = h - doubles - triples - hr
        slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / ab if ab else 0
        iso = slg - avg
        bb_pct = bb / pa * 100 if pa else 0
        k_pct = so / pa * 100 if pa else 0

        # wOBA (2024 linear weights)
        woba_num = 0.690 * bb + 0.722 * hbp + 0.878 * singles + 1.242 * doubles + 1.568 * triples + 2.004 * hr
        woba_den = ab + bb - ibb + sf + hbp
        woba = woba_num / woba_den if woba_den else 0

        babip_den = ab - so - hr + sf
        babip = (h - hr) / babip_den if babip_den else 0

        rows.append({
            "Name": player.get("fullName", ""),
            "Team": team.get("abbreviation", ""),
            "G": int(stat.get("gamesPlayed", 0)),
            "PA": pa, "AB": ab, "H": h,
            "2B": doubles, "3B": triples, "HR": hr,
            "R": int(stat.get("runs", 0)),
            "RBI": int(stat.get("rbi", 0)),
            "SB": int(stat.get("stolenBases", 0)),
            "BB%": round(bb_pct, 1), "K%": round(k_pct, 1),
            "AVG": round(avg, 3), "OBP": round(obp, 3), "SLG": round(slg, 3),
            "wOBA": round(woba, 3), "ISO": round(iso, 3), "BABIP": round(babip, 3),
        })

    if len(rows) >= 50:
        return pd.DataFrame(rows).sort_values("PA", ascending=False).reset_index(drop=True)
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    year = datetime.now().year

    for y in [year, year - 1, year - 2]:
        print(f"Fetching {y} batting leaders (min {MIN_PA} PA)...")

        # Try FanGraphs first (richer data)
        df = _fetch_fangraphs(y)
        if df is not None and len(df) >= 50:
            df.to_csv(OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {OUT_PATH} (FanGraphs {y})")
            return

        # Fallback: MLB Stats API
        print(f"  Trying MLB Stats API for {y}...")
        df = _fetch_mlb_api(y)
        if df is not None and len(df) >= 50:
            df.to_csv(OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {OUT_PATH} (MLB API {y})")
            return

        print(f"  Insufficient data for {y}, trying earlier year...")

    print("ERROR: Could not fetch sufficient batting data for any recent season.")


if __name__ == "__main__":
    main()
