"""
Fetch batting + pitching leaders and cache to CSV.
Run locally before deploying to Streamlit Cloud.

Tries FanGraphs via pybaseball first, falls back to MLB Stats API.

Usage: python cache_stats.py
"""

import os
import requests
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATTING_OUT_PATH = os.path.join(DATA_DIR, "batting_stats_cache.csv")
PITCHING_OUT_PATH = os.path.join(DATA_DIR, "pitching_stats_cache.csv")

MIN_PA = 100
MIN_IP = 10


def _fetch_fangraphs(year: int) -> pd.DataFrame | None:
    """Fetch from FanGraphs via pybaseball."""
    try:
        from pybaseball import batting_stats, cache
        cache.enable()
        df = batting_stats(year, qual=MIN_PA)
        if df is not None and len(df) >= 50:
            # Normalise column subset to our standard schema (include Statcast expected stats)
            keep = [c for c in ["Name","Team","G","PA","AB","H","2B","3B","HR","R","RBI","SB",
                                 "BB%","K%","AVG","OBP","SLG","wOBA","ISO","BABIP",
                                 "xBA","xSLG","xwOBA","Barrel%","HardHit%"] if c in df.columns]
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


def _fetch_pitching_fangraphs(year: int) -> pd.DataFrame | None:
    """Fetch pitching leaders from FanGraphs via pybaseball."""
    try:
        from pybaseball import pitching_stats, cache
        cache.enable()
        df = pitching_stats(year, qual=MIN_IP)
        if df is not None and len(df) >= 20:
            keep = [c for c in ["Name","Team","G","GS","IP","ERA","FIP","K/9","BB/9",
                                 "WHIP","K%","BB%","HR/9","W","L","SV"] if c in df.columns]
            return df[keep]
    except Exception as e:
        print(f"  FanGraphs pitching failed for {year}: {e}")
    return None


def _fetch_pitching_mlb_api(year: int) -> pd.DataFrame | None:
    """Fetch pitching leaders from MLB Stats API (free, no key)."""
    url = (
        f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching"
        f"&season={year}&sportId=1&limit=1000&offset=0&gameType=R"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  MLB API pitching failed for {year}: {e}")
        return None

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})

        ip_str = str(stat.get("inningsPitched", "0"))
        try:
            parts = ip_str.split(".")
            ip = int(parts[0]) + int(parts[1]) / 3 if len(parts) > 1 else float(parts[0])
        except (ValueError, IndexError):
            ip = 0.0
        if ip < MIN_IP:
            continue

        era = float(stat.get("era", "0") or 0)
        so = int(stat.get("strikeOuts", 0))
        bb = int(stat.get("baseOnBalls", 0))
        h = int(stat.get("hits", 0))
        hr = int(stat.get("homeRuns", 0))
        bf = int(stat.get("battersFaced", 0)) or 1

        k9 = so / ip * 9 if ip > 0 else 0
        bb9 = bb / ip * 9 if ip > 0 else 0
        hr9 = hr / ip * 9 if ip > 0 else 0
        whip = (bb + h) / ip if ip > 0 else 0
        k_pct = so / bf * 100 if bf > 0 else 0
        bb_pct = bb / bf * 100 if bf > 0 else 0

        # FIP calculation: (13*HR + 3*BB - 2*K) / IP + cFIP (~3.10)
        fip = (13 * hr + 3 * bb - 2 * so) / ip + 3.10 if ip > 0 else 0

        rows.append({
            "Name": player.get("fullName", ""),
            "Team": team.get("abbreviation", ""),
            "G": int(stat.get("gamesPlayed", 0)),
            "GS": int(stat.get("gamesStarted", 0)),
            "IP": round(ip, 1),
            "ERA": round(era, 2),
            "FIP": round(fip, 2),
            "K/9": round(k9, 1),
            "BB/9": round(bb9, 1),
            "WHIP": round(whip, 2),
            "K%": round(k_pct, 1),
            "BB%": round(bb_pct, 1),
            "HR/9": round(hr9, 1),
        })

    if len(rows) >= 20:
        return pd.DataFrame(rows).sort_values("IP", ascending=False).reset_index(drop=True)
    return None


def _fetch_statcast_expected(year: int) -> pd.DataFrame | None:
    """Fetch Statcast expected stats (xBA, xSLG, xwOBA, Barrel%) via pybaseball."""
    try:
        from pybaseball import statcast_batter_expected_stats, cache
        cache.enable()
        df = statcast_batter_expected_stats(year, min_pa=MIN_PA)
        if df is not None and not df.empty:
            # Map to columns we need
            cols = {}
            for orig, target in [("last_name, first_name", "Name"), ("pa", "PA"),
                                  ("est_ba", "xBA"), ("est_slg", "xSLG"),
                                  ("est_woba", "xwOBA"), ("brl_percent", "Barrel%")]:
                if orig in df.columns:
                    cols[orig] = target
            if cols:
                df = df.rename(columns=cols)
            keep = [c for c in ["Name", "xBA", "xSLG", "xwOBA", "Barrel%"] if c in df.columns]
            if keep:
                return df[keep]
    except Exception as e:
        print(f"  Statcast expected stats failed for {year}: {e}")
    return None


def _cache_batting(year: int) -> bool:
    """Cache batting stats with Statcast expected stats merged in. Returns True on success."""
    for y in [year, year - 1, year - 2]:
        print(f"Fetching {y} batting leaders (min {MIN_PA} PA)...")
        df = _fetch_fangraphs(y)
        if df is not None and len(df) >= 50:
            # Try to merge Statcast expected stats if not already in FanGraphs data
            if "xBA" not in df.columns:
                print(f"  Fetching Statcast expected stats for {y}...")
                sc = _fetch_statcast_expected(y)
                if sc is not None and "Name" in sc.columns:
                    df = df.merge(sc, on="Name", how="left", suffixes=("", "_sc"))
            df.to_csv(BATTING_OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {BATTING_OUT_PATH} (FanGraphs {y})")
            return True
        print(f"  Trying MLB Stats API for {y}...")
        df = _fetch_mlb_api(y)
        if df is not None and len(df) >= 50:
            # Try to merge Statcast expected stats
            print(f"  Fetching Statcast expected stats for {y}...")
            sc = _fetch_statcast_expected(y)
            if sc is not None and "Name" in sc.columns:
                df = df.merge(sc, on="Name", how="left", suffixes=("", "_sc"))
            df.to_csv(BATTING_OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {BATTING_OUT_PATH} (MLB API {y})")
            return True
        print(f"  Insufficient data for {y}, trying earlier year...")
    print("ERROR: Could not fetch sufficient batting data for any recent season.")
    return False


def _cache_pitching(year: int) -> bool:
    """Cache pitching stats. Returns True on success."""
    for y in [year, year - 1, year - 2]:
        print(f"Fetching {y} pitching leaders (min {MIN_IP} IP)...")
        df = _fetch_pitching_fangraphs(y)
        if df is not None and len(df) >= 20:
            df.to_csv(PITCHING_OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {PITCHING_OUT_PATH} (FanGraphs {y})")
            return True
        print(f"  Trying MLB Stats API for {y}...")
        df = _fetch_pitching_mlb_api(y)
        if df is not None and len(df) >= 20:
            df.to_csv(PITCHING_OUT_PATH, index=False)
            print(f"Saved {len(df)} rows to {PITCHING_OUT_PATH} (MLB API {y})")
            return True
        print(f"  Insufficient data for {y}, trying earlier year...")
    print("ERROR: Could not fetch sufficient pitching data for any recent season.")
    return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    year = datetime.now().year
    _cache_batting(year)
    _cache_pitching(year)


if __name__ == "__main__":
    main()
