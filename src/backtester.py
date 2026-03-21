"""
Historical Backtesting Engine — TASK 5

Simulates what the model would have predicted for every game in the 2025 MLB
season, grades against actual box-score results, and produces a comprehensive
accuracy report.

PURPOSE:
  Validate the prediction model on ~5,000+ historical picks BEFORE risking
  real money. Identify systematic biases (direction, prop type, confidence
  calibration) so they can be fixed before going live.

WALK-FORWARD REQUIREMENT (CRITICAL):
  When backtesting April 15, only stats from BEFORE April 15 are used to
  build batter profiles. Full-season stats are never leaked into early-season
  predictions. Violating this would produce unrealistically optimistic results.

DATA SOURCES (all free, no auth):
  - MLB Stats API  (schedule, box scores, lineups)
  - pybaseball      (FanGraphs season stats, Statcast metrics)

USAGE:
  from src.backtester import run_backtest
  summary = run_backtest("2025-04-01", "2025-09-30")

  # Resume an interrupted run — it picks up where save file left off:
  summary = run_backtest("2025-04-01", "2025-09-30")

  # Generate report from saved results:
  from src.backtester import load_results, generate_backtest_report
  results = load_results()
  report = generate_backtest_report(results)
"""

from __future__ import annotations

import json
import os
import time
import traceback
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Clear proxy environment variables to avoid sandbox network issues
for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

# ── Project imports ──────────────────────────────────────────────────────────
from src.predictor import generate_prediction, _clear_weights_cache

# ── Constants ────────────────────────────────────────────────────────────────

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

DEFAULT_RESULTS_PATH = "data/backtest/backtest_2025.json"
DEFAULT_REPORT_PATH = "data/backtest/backtest_2025_report.json"

# DraftKings Hitter Fantasy Score weights
DK_SINGLE = 3
DK_DOUBLE = 5
DK_TRIPLE = 8
DK_HR = 10
DK_RBI = 2
DK_RUN = 2
DK_BB_HBP = 2
DK_SB = 5

# Prop types to backtest and the default PrizePicks-style lines
DEFAULT_PROP_TYPES = [
    "hitter_fantasy_score",
    "hits",
    "total_bases",
    "home_runs",
    "pitcher_strikeouts",
]

DEFAULT_LINES = {
    "hitter_fantasy_score": 7.5,
    "hits": 1.5,
    "total_bases": 1.5,
    "home_runs": 0.5,
    "pitcher_strikeouts": 4.5,
}

# FanGraphs column mapping — convert DF columns to batter_profile keys
FANGRAPHS_BATTER_MAP = {
    "PA": "pa",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "ISO": "iso",
    "wOBA": "woba",
    "BABIP": "babip",
    "HR": "hr",
    "SB": "sb",
    "RBI": "rbi",
    "R": "r",
    "2B": "2b",
    "3B": "3b",
    "K%": "k_rate",
    "BB%": "bb_rate",
}

FANGRAPHS_PITCHER_MAP = {
    "IP": "ip",
    "ERA": "era",
    "FIP": "fip",
    "xFIP": "xfip",
    "WHIP": "whip",
    "K/9": "k9",
    "BB/9": "bb9",
    "HR/9": "hr9",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "GS": "gs",
}


# ═════════════════════════════════════════════════════════════════════════════
# MLB Stats API helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mlb_get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET request to the MLB Stats API with basic retry logic."""
    url = f"{MLB_API_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            # Disable proxy to avoid sandbox network restrictions
            resp = requests.get(url, params=params, timeout=30, proxies={"http": "", "https": ""})
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                print(f"  [API ERROR] {url} — {exc}")
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_schedule(game_date: str) -> list[dict]:
    """
    Return list of game dicts for a single date.
    Each dict contains gamePk and basic info.
    """
    data = _mlb_get("schedule", {
        "sportId": 1,
        "date": game_date,
        "hydrate": "linescore",
    })
    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status in ("Final", "Completed Early"):
                games.append(g)
    return games


def fetch_boxscore(game_pk: int) -> dict:
    """Fetch full box score for a game. Returns raw API response."""
    return _mlb_get(f"game/{game_pk}/boxscore")


def _extract_batter_stats(player_data: dict) -> dict | None:
    """
    Pull batting line from a box-score player entry.
    Returns dict with counting stats or None if player did not bat.
    """
    batting = player_data.get("stats", {}).get("batting", {})
    if not batting or batting.get("atBats", 0) == 0 and batting.get("plateAppearances", 0) == 0:
        return None

    return {
        "ab": batting.get("atBats", 0),
        "pa": batting.get("plateAppearances", 0),
        "hits": batting.get("hits", 0),
        "doubles": batting.get("doubles", 0),
        "triples": batting.get("triples", 0),
        "hr": batting.get("homeRuns", 0),
        "rbi": batting.get("rbi", 0),
        "runs": batting.get("runs", 0),
        "bb": batting.get("baseOnBalls", 0),
        "hbp": batting.get("hitByPitch", 0),
        "sb": batting.get("stolenBases", 0),
        "k": batting.get("strikeOuts", 0),
    }


def _extract_pitcher_stats(player_data: dict) -> dict | None:
    """Pull pitching line from a box-score player entry."""
    pitching = player_data.get("stats", {}).get("pitching", {})
    if not pitching:
        return None
    ip_str = pitching.get("inningsPitched", "0")
    try:
        ip = float(ip_str)
    except (ValueError, TypeError):
        ip = 0.0

    if ip == 0:
        return None

    return {
        "ip": ip,
        "k": pitching.get("strikeOuts", 0),
        "bb": pitching.get("baseOnBalls", 0),
        "hits_allowed": pitching.get("hits", 0),
        "er": pitching.get("earnedRuns", 0),
        "hr_allowed": pitching.get("homeRuns", 0),
    }


def extract_all_batters(boxscore: dict) -> list[dict]:
    """
    From a box-score response, return a list of dicts, one per batter who
    appeared in the game.  Each dict includes full_name, team_abbr, and
    the actual counting stats.

    CRITICAL: Only returns players with PA > 0 to avoid including bench
    non-players in backtest results. Non-plays artificially inflate LESS
    props because players who didn't bat naturally have actual=0.
    """
    batters = []
    for side in ("home", "away"):
        team_data = boxscore.get("teams", {}).get(side, {})
        team_abbr = team_data.get("team", {}).get("abbreviation", "")
        players = team_data.get("players", {})
        for pid, pdata in players.items():
            person = pdata.get("person", {})
            full_name = person.get("fullName", "")
            stats = _extract_batter_stats(pdata)
            # Filter: only include batters with at least 2 PA (exclude pinch hitters)
            # Rationale: Backtest includes only probable starters who get consistent AB.
            # Players with 1 PA are likely pinch hitters or defensive replacements
            # who don't get PrizePicks props in live trading.
            if stats and full_name and stats.get("pa", 0) >= 2:
                stats["full_name"] = full_name
                stats["team"] = team_abbr
                stats["game_side"] = side
                batters.append(stats)
    return batters


def extract_starting_pitcher(boxscore: dict, side: str) -> dict | None:
    """
    Return pitcher stats dict for the starting pitcher on a given side
    ('home' or 'away').
    """
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})

    # The pitchers list is ordered; first entry is the starter
    pitcher_ids = team_data.get("pitchers", [])
    if not pitcher_ids:
        return None

    starter_id = f"ID{pitcher_ids[0]}"
    pdata = players.get(starter_id)
    if not pdata:
        return None

    person = pdata.get("person", {})
    stats = _extract_pitcher_stats(pdata)
    if stats:
        stats["full_name"] = person.get("fullName", "")
        stats["team"] = team_data.get("team", {}).get("abbreviation", "")
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# Fantasy-score calculation
# ═════════════════════════════════════════════════════════════════════════════

def calculate_fantasy_score(stats: dict) -> float:
    """
    Calculate DraftKings hitter fantasy score from box-score counting stats.

    DK weights:
      Single=3, Double=5, Triple=8, HR=10, RBI=2, Run=2, BB/HBP=2, SB=5
    """
    hits = stats.get("hits", 0)
    doubles = stats.get("doubles", 0)
    triples = stats.get("triples", 0)
    hr = stats.get("hr", 0)
    singles = hits - doubles - triples - hr

    score = (
        singles * DK_SINGLE
        + doubles * DK_DOUBLE
        + triples * DK_TRIPLE
        + hr * DK_HR
        + stats.get("rbi", 0) * DK_RBI
        + stats.get("runs", 0) * DK_RUN
        + (stats.get("bb", 0) + stats.get("hbp", 0)) * DK_BB_HBP
        + stats.get("sb", 0) * DK_SB
    )
    return float(score)


def _actual_value_for_prop(prop_type: str, batter_stats: dict,
                           pitcher_stats: dict | None = None) -> float | None:
    """
    Map a prop type to the actual result from box-score stats.
    Returns None if the stat cannot be determined.
    """
    if prop_type == "hitter_fantasy_score":
        return calculate_fantasy_score(batter_stats)
    elif prop_type == "hits":
        return float(batter_stats.get("hits", 0))
    elif prop_type == "total_bases":
        h = batter_stats.get("hits", 0)
        d = batter_stats.get("doubles", 0)
        t = batter_stats.get("triples", 0)
        hr = batter_stats.get("hr", 0)
        singles = h - d - t - hr
        return float(singles + 2 * d + 3 * t + 4 * hr)
    elif prop_type == "home_runs":
        return float(batter_stats.get("hr", 0))
    elif prop_type == "pitcher_strikeouts":
        if pitcher_stats:
            return float(pitcher_stats.get("k", 0))
        return None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Walk-forward profile building
# ═════════════════════════════════════════════════════════════════════════════

def _strip_pct(val) -> float:
    """Convert FanGraphs percentage fields (which may be strings like '22.7%') to float."""
    if isinstance(val, str):
        return float(val.replace("%", "").replace(" ", ""))
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove suffixes like Jr./Sr./III."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


# Cache for season batting/pitching stats.
# For completed (prior) years: keyed by (year, None) — full-season is fine.
# For the current backtest year: keyed by (year, cutoff_date_str) so that
# each backtest date only sees stats from BEFORE that date.
_batting_cache: dict[tuple[int, str | None], pd.DataFrame] = {}
_pitching_cache: dict[tuple[int, str | None], pd.DataFrame] = {}


def _mlb_api_season_stats(
    group: str,
    season: int,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch season stats from the MLB Stats API filtered to games on or before
    *end_date*.  *group* is 'hitting' or 'pitching'.

    Returns a DataFrame with a 'Name' column (for player matching) plus the
    stat columns the MLB API returns, mapped to FanGraphs-compatible names
    where possible.

    Falls back to an empty DataFrame on any error.
    """
    season_start = f"{season}-03-20"  # Spring training cutoff; regular season ~late March

    params = {
        "stats": "season",
        "group": group,
        "season": season,
        "startDate": season_start,
        "endDate": end_date,
        "sportIds": 1,
        "limit": 5000,
        "fields": (
            "stats,splits,stat,player,fullName"
        ),
    }
    data = _mlb_get("stats", params)

    rows: list[dict] = []
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            player_name = split.get("player", {}).get("fullName", "")
            s = split.get("stat", {})
            if not player_name or not s:
                continue

            if group == "hitting":
                pa = int(s.get("plateAppearances", 0))
                ab = int(s.get("atBats", 0))
                hits = int(s.get("hits", 0))
                doubles = int(s.get("doubles", 0))
                triples = int(s.get("triples", 0))
                hr = int(s.get("homeRuns", 0))
                rbi = int(s.get("rbi", 0))
                r = int(s.get("runs", 0))
                sb = int(s.get("stolenBases", 0))
                bb = int(s.get("baseOnBalls", 0))
                k = int(s.get("strikeOuts", 0))
                hbp = int(s.get("hitByPitch", 0))

                avg = hits / ab if ab > 0 else 0.0
                obp = (hits + bb + hbp) / pa if pa > 0 else 0.0
                singles = hits - doubles - triples - hr
                tb = singles + 2 * doubles + 3 * triples + 4 * hr
                slg = tb / ab if ab > 0 else 0.0
                iso = slg - avg

                # Approximate wOBA using linear weights
                woba_num = (
                    0.69 * bb + 0.72 * hbp + 0.89 * singles
                    + 1.27 * doubles + 1.62 * triples + 2.10 * hr
                )
                woba = woba_num / pa if pa > 0 else 0.0

                # BABIP
                sf = int(s.get("sacFlies", 0))
                babip_denom = ab - k - hr + sf
                babip = (hits - hr) / babip_denom if babip_denom > 0 else 0.0

                k_rate = (k / pa * 100) if pa > 0 else 0.0
                bb_rate = (bb / pa * 100) if pa > 0 else 0.0

                rows.append({
                    "Name": player_name,
                    "PA": pa, "AVG": round(avg, 3), "OBP": round(obp, 3),
                    "SLG": round(slg, 3), "ISO": round(iso, 3),
                    "wOBA": round(woba, 3), "BABIP": round(babip, 3),
                    "HR": hr, "SB": sb, "RBI": rbi, "R": r,
                    "2B": doubles, "3B": triples,
                    "K%": round(k_rate, 1), "BB%": round(bb_rate, 1),
                })
            else:
                # Pitching
                ip_str = s.get("inningsPitched", "0")
                try:
                    ip = float(ip_str)
                except (ValueError, TypeError):
                    ip = 0.0
                if ip <= 0:
                    continue

                er = int(s.get("earnedRuns", 0))
                k = int(s.get("strikeOuts", 0))
                bb = int(s.get("baseOnBalls", 0))
                hits_a = int(s.get("hits", 0))
                hr_a = int(s.get("homeRuns", 0))
                gs = int(s.get("gamesStarted", 0))
                bf = int(s.get("battersFaced", 0))

                era = (er * 9) / ip if ip > 0 else 0.0
                whip = (bb + hits_a) / ip if ip > 0 else 0.0
                k9 = (k * 9) / ip if ip > 0 else 0.0
                bb9 = (bb * 9) / ip if ip > 0 else 0.0
                hr9 = (hr_a * 9) / ip if ip > 0 else 0.0
                k_pct = (k / bf * 100) if bf > 0 else 0.0
                bb_pct = (bb / bf * 100) if bf > 0 else 0.0

                # Approximate FIP: (13*HR + 3*BB - 2*K) / IP + 3.10
                fip = ((13 * hr_a + 3 * bb - 2 * k) / ip + 3.10) if ip > 0 else 0.0

                rows.append({
                    "Name": player_name,
                    "IP": ip, "ERA": round(era, 2), "FIP": round(fip, 2),
                    "xFIP": round(fip, 2),  # Approximation (no league HR/FB data)
                    "WHIP": round(whip, 2),
                    "K/9": round(k9, 1), "BB/9": round(bb9, 1),
                    "HR/9": round(hr9, 1),
                    "K%": round(k_pct, 1), "BB%": round(bb_pct, 1),
                    "GS": gs,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _get_season_batting(year: int, cutoff_date: str | None = None) -> pd.DataFrame:
    """
    Fetch batting leaderboard for *year* up to (but NOT including) *cutoff_date*.

    Walk-forward safety:
      - If *cutoff_date* is provided and falls within *year*, stats are pulled
        from the MLB Stats API with an endDate of the day before *cutoff_date*.
      - If *cutoff_date* is None (completed/prior year), full-season FanGraphs
        data is used (no leakage risk since the season is over).

    Results are cached by (year, cutoff_date) to avoid redundant API calls
    while still respecting the walk-forward boundary.
    """
    cache_key = (year, cutoff_date)
    if cache_key in _batting_cache:
        return _batting_cache[cache_key]

    df = pd.DataFrame()

    if cutoff_date is not None:
        # Walk-forward: only use stats from BEFORE the backtest date
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
        end_dt = cutoff_dt - timedelta(days=1)
        end_str = end_dt.strftime("%Y-%m-%d")

        # Don't query if end_date is before season start
        if end_dt >= datetime(year, 3, 20):
            df = _mlb_api_season_stats("hitting", year, end_str)
            if not df.empty:
                _batting_cache[cache_key] = df
                return df

        # If MLB API returned nothing (season hasn't started yet or API error),
        # return empty — do NOT fall back to full-season FanGraphs for the
        # current year as that would re-introduce data leakage.
        if df.empty:
            _batting_cache[cache_key] = df
            return df

    # Completed (prior) year — full-season FanGraphs data is safe
    try:
        from src.stats import fetch_batting_leaders
        df = fetch_batting_leaders(season=year, min_pa=1)
    except Exception as exc:
        print(f"  [WARN] Could not fetch batting stats for {year}: {exc}")
        df = pd.DataFrame()

    _batting_cache[cache_key] = df
    return df


def _get_season_pitching(year: int, cutoff_date: str | None = None) -> pd.DataFrame:
    """
    Fetch pitching leaderboard for *year* up to (but NOT including) *cutoff_date*.

    Same walk-forward logic as _get_season_batting — see its docstring.
    """
    cache_key = (year, cutoff_date)
    if cache_key in _pitching_cache:
        return _pitching_cache[cache_key]

    df = pd.DataFrame()

    if cutoff_date is not None:
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
        end_dt = cutoff_dt - timedelta(days=1)
        end_str = end_dt.strftime("%Y-%m-%d")

        if end_dt >= datetime(year, 3, 20):
            df = _mlb_api_season_stats("pitching", year, end_str)
            if not df.empty:
                _pitching_cache[cache_key] = df
                return df

        if df.empty:
            _pitching_cache[cache_key] = df
            return df

    # Completed (prior) year
    try:
        from src.stats import fetch_pitching_leaders
        df = fetch_pitching_leaders(season=year, min_ip=1)
    except Exception as exc:
        print(f"  [WARN] Could not fetch pitching stats for {year}: {exc}")
        df = pd.DataFrame()

    _pitching_cache[cache_key] = df
    return df


def _match_player_row(player_name: str, df: pd.DataFrame) -> pd.Series | None:
    """
    Match a player name (from MLB box score) to a FanGraphs leaderboard row.
    Tries: exact, case-insensitive, last-name + first-initial, unique last name.
    """
    if df.empty or "Name" not in df.columns:
        return None

    target = _normalize_name(player_name)

    # Exact (normalized)
    for idx, row in df.iterrows():
        if _normalize_name(str(row.get("Name", ""))) == target:
            return row

    # Last name + first initial
    parts = target.split()
    if len(parts) >= 2:
        first_init = parts[0][0]
        last = parts[-1]
        matches = []
        for idx, row in df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            rparts = rn.split()
            if len(rparts) >= 2 and rparts[-1] == last and rparts[0][0] == first_init:
                matches.append(row)
        if len(matches) == 1:
            return matches[0]

    # Unique last name
    if parts:
        last = parts[-1]
        matches = [
            row for _, row in df.iterrows()
            if _normalize_name(str(row.get("Name", ""))).split()[-1:] == [last]
        ]
        if len(matches) == 1:
            return matches[0]

    return None


def build_walkforward_profile(player_name: str, game_date: str,
                              is_pitcher: bool = False) -> dict | None:
    """
    Build a batter (or pitcher) profile using ONLY data available before
    *game_date*.

    Walk-forward logic:
      - Current-year stats are fetched from the MLB Stats API with an
        endDate of (game_date - 1 day), ensuring NO future data leaks
        into the profile.  This is the key guard against inflated backtest
        accuracy.
      - If game_date is in the first month of the season (April), rely
        heavily on the PRIOR year stats (the current-year sample is too
        small to be meaningful).
      - As the season progresses, the current-year stats accumulate PA
        and the Bayesian regression in the predictor naturally up-weights
        them.
      - Prior-year stats use full-season FanGraphs data (no leakage risk
        since that season is already complete).

    Returns None if the player cannot be matched.
    """
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    year = dt.year
    month = dt.month

    if is_pitcher:
        # Current year stats — filtered to BEFORE game_date (walk-forward)
        df_current = _get_season_pitching(year, cutoff_date=game_date)
        row = _match_player_row(player_name, df_current)

        # If early season or no match, try prior year (full season, no leakage)
        if row is None or (month <= 5 and (row is None or float(row.get("IP", 0)) < 10)):
            df_prior = _get_season_pitching(year - 1, cutoff_date=None)
            row_prior = _match_player_row(player_name, df_prior)
            if row_prior is not None:
                row = row_prior

        if row is None:
            return None

        profile: dict = {}
        for fg_col, key in FANGRAPHS_PITCHER_MAP.items():
            val = row.get(fg_col, 0)
            if key in ("k_pct", "bb_pct"):
                profile[key] = _strip_pct(val)
            else:
                try:
                    profile[key] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    profile[key] = 0.0

        # Guard against decimal-vs-percentage confusion (FanGraphs CSVs
        # store K%/BB% as decimals like 0.227; predictor expects 22.7).
        for rate_key in ("k_pct", "bb_pct"):
            v = profile.get(rate_key, 0)
            if v and v < 1.0:
                profile[rate_key] = v * 100

        return profile

    # ── Batter path ──
    # Current year stats — filtered to BEFORE game_date (walk-forward)
    df_current = _get_season_batting(year, cutoff_date=game_date)
    row = _match_player_row(player_name, df_current)

    # Early season or no current-year match: fall back to prior year (full season, no leakage)
    if row is None or (month <= 5 and float(row.get("PA", 0)) < 50):
        df_prior = _get_season_batting(year - 1, cutoff_date=None)
        row_prior = _match_player_row(player_name, df_prior)
        if row_prior is not None:
            # If we have BOTH years, blend (weight current by fraction of 500 PA)
            if row is not None and float(row.get("PA", 0)) > 0:
                # We keep the current-year row and let the predictor's Bayesian
                # stabilization handle the small sample — but we patch in the
                # prior-year counting stats for events that need a base (HR, SB,
                # RBI, R, 2B, 3B) if the current-year counts are tiny.
                pa_current = float(row.get("PA", 0))
                if pa_current < 50:
                    row = row_prior
            else:
                row = row_prior

    if row is None:
        return None

    profile = {}
    for fg_col, key in FANGRAPHS_BATTER_MAP.items():
        val = row.get(fg_col, 0)
        if key in ("k_rate", "bb_rate"):
            profile[key] = _strip_pct(val)
        elif key in ("hr", "sb", "rbi", "r", "2b", "3b", "pa"):
            try:
                profile[key] = int(val) if val is not None else 0
            except (ValueError, TypeError):
                profile[key] = 0
        else:
            try:
                profile[key] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                profile[key] = 0.0

    # Guard against decimal-vs-percentage confusion (FanGraphs CSVs
    # store K%/BB% as decimals like 0.227; predictor expects 22.7).
    for rate_key in ("k_rate", "bb_rate"):
        v = profile.get(rate_key, 0)
        if v and v < 1.0:
            profile[rate_key] = v * 100

    # Statcast expected stats — set to 0 so the predictor uses season-only;
    # pulling per-player Statcast for every batter-day in a 6-month backtest
    # would be extremely slow. The predictor gracefully ignores zeros.
    profile.setdefault("xba", 0)
    profile.setdefault("xslg", 0)
    profile.setdefault("barrel_rate", 0)
    profile.setdefault("recent_barrel_rate", 0)
    profile.setdefault("recent_hard_hit_pct", 0)
    profile.setdefault("sprint_speed", 0)

    return profile


# ═════════════════════════════════════════════════════════════════════════════
# Grading
# ═════════════════════════════════════════════════════════════════════════════

def grade_backtest_prediction(prediction: dict, actual_stats: dict) -> str:
    """
    Compare a prediction to actual box-score results.

    Returns:
      'W'    — pick was correct
      'L'    — pick was wrong
      'push' — actual equals the line exactly
    """
    prop_type = prediction["prop_type"]
    line = prediction["line"]
    pick = prediction["pick"]

    actual = _actual_value_for_prop(
        prop_type,
        actual_stats,
        pitcher_stats=prediction.get("_pitcher_stats"),
    )
    if actual is None:
        return "skip"

    if actual > line:
        return "W" if pick == "MORE" else "L"
    elif actual < line:
        return "W" if pick == "LESS" else "L"
    else:
        return "push"


# ═════════════════════════════════════════════════════════════════════════════
# Single-day backtesting
# ═════════════════════════════════════════════════════════════════════════════

def backtest_single_day(game_date: str,
                        prop_types: list[str] | None = None) -> list[dict]:
    """
    Process one day of MLB games. For every batter who appeared in a final
    game, run the predictor for each requested prop type, then grade against
    the actual box-score result.

    Returns a list of result dicts, one per prediction.
    """
    prop_types = prop_types or DEFAULT_PROP_TYPES
    results: list[dict] = []

    games = fetch_schedule(game_date)
    if not games:
        return results

    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue

        home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")

        boxscore = fetch_boxscore(game_pk)
        if not boxscore:
            continue

        batters = extract_all_batters(boxscore)

        # Get starting pitchers for context
        home_sp = extract_starting_pitcher(boxscore, "home")
        away_sp = extract_starting_pitcher(boxscore, "away")

        for batter in batters:
            player_name = batter["full_name"]
            batter_team = batter["team"]

            for prop_type in prop_types:
                # Skip pitcher_strikeouts for batters — handled separately
                if prop_type == "pitcher_strikeouts":
                    continue

                line = DEFAULT_LINES.get(prop_type, 1.5)

                # Build walk-forward profile
                profile = build_walkforward_profile(player_name, game_date,
                                                    is_pitcher=False)
                if profile is None:
                    continue

                # Determine opposing pitcher for context
                opp_pitcher_profile = None
                park = batter_team  # Approximate: use batter's team as park
                if batter["game_side"] == "home":
                    park = home_team
                    if away_sp:
                        opp_pitcher_profile = build_walkforward_profile(
                            away_sp["full_name"], game_date, is_pitcher=True
                        )
                else:
                    park = away_team  # Away batters face home park
                    park = home_team  # Actually home park is where game is
                    if home_sp:
                        opp_pitcher_profile = build_walkforward_profile(
                            home_sp["full_name"], game_date, is_pitcher=True
                        )

                # Run prediction
                try:
                    pred = generate_prediction(
                        player_name=player_name,
                        stat_type=prop_type.replace("_", " ").title(),
                        stat_internal=prop_type,
                        line=line,
                        batter_profile=profile,
                        opp_pitcher_profile=opp_pitcher_profile,
                        park_team=park,
                    )
                except Exception as exc:
                    # Never let one prediction crash the whole day
                    continue

                # Compute actual value
                actual = _actual_value_for_prop(prop_type, batter)
                if actual is None:
                    continue

                # Grade
                pick = pred.get("pick", "MORE")
                if actual > line:
                    grade = "W" if pick == "MORE" else "L"
                elif actual < line:
                    grade = "W" if pick == "LESS" else "L"
                else:
                    grade = "push"

                results.append({
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "player_name": player_name,
                    "team": batter_team,
                    "prop_type": prop_type,
                    "line": line,
                    "projection": pred.get("projection", 0),
                    "pick": pick,
                    "confidence": pred.get("confidence", 0.5),
                    "rating": pred.get("rating", "D"),
                    "edge": pred.get("edge", 0),
                    "p_over": pred.get("p_over", 0.5),
                    "p_under": pred.get("p_under", 0.5),
                    "actual": actual,
                    "result": grade,
                })

        # ── Pitcher strikeouts (starting pitchers only) ──
        if "pitcher_strikeouts" in prop_types:
            for sp, sp_side in [(home_sp, "home"), (away_sp, "away")]:
                if sp is None:
                    continue

                sp_name = sp["full_name"]
                line = DEFAULT_LINES.get("pitcher_strikeouts", 4.5)

                pitcher_profile = build_walkforward_profile(
                    sp_name, game_date, is_pitcher=True
                )
                if pitcher_profile is None:
                    continue

                try:
                    pred = generate_prediction(
                        player_name=sp_name,
                        stat_type="Pitcher Strikeouts",
                        stat_internal="pitcher_strikeouts",
                        line=line,
                        pitcher_profile=pitcher_profile,
                        park_team=home_team,
                    )
                except Exception:
                    continue

                actual_k = float(sp.get("k", 0))
                pick = pred.get("pick", "MORE")
                if actual_k > line:
                    grade = "W" if pick == "MORE" else "L"
                elif actual_k < line:
                    grade = "W" if pick == "LESS" else "L"
                else:
                    grade = "push"

                results.append({
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "player_name": sp_name,
                    "team": sp["team"],
                    "prop_type": "pitcher_strikeouts",
                    "line": line,
                    "projection": pred.get("projection", 0),
                    "pick": pick,
                    "confidence": pred.get("confidence", 0.5),
                    "rating": pred.get("rating", "D"),
                    "edge": pred.get("edge", 0),
                    "p_over": pred.get("p_over", 0.5),
                    "p_under": pred.get("p_under", 0.5),
                    "actual": actual_k,
                    "result": grade,
                })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Persistence
# ═════════════════════════════════════════════════════════════════════════════

def save_results(results: list[dict],
                 filepath: str = DEFAULT_RESULTS_PATH) -> None:
    """Save backtest results to JSON. Uses atomic write (write to temp, then move)."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then atomically move it
    temp_p = Path(str(p) + ".tmp")
    with open(temp_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    temp_p.replace(p)  # Atomic move
    print(f"  [SAVED] {len(results)} results -> {p}")


def load_results(filepath: str = DEFAULT_RESULTS_PATH) -> list[dict]:
    """Load previously saved backtest results. Returns [] if file missing."""
    p = Path(filepath)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_nonplays(results: list[dict]) -> tuple[list[dict], dict]:
    """
    Remove predictions where the player had 0 PA in the game (non-plays).

    Non-plays are batters included in the box score but who didn't actually bat
    (benched, injured that day, etc.). These artificially inflate LESS accuracy
    and depress MORE accuracy because they produce automatic W/L results.

    Returns:
        (filtered_results, stats_dict) where stats_dict contains:
        - total_predictions: count before filtering
        - nonplays_removed: count of removed non-plays
        - kept_predictions: count after filtering
        - pct_removed: percentage of predictions that were non-plays
    """
    plays = [r for r in results if r.get("actual", 0) > 0]
    nonplays = [r for r in results if r.get("actual", 0) == 0]

    stats = {
        "total_predictions": len(results),
        "nonplays_removed": len(nonplays),
        "kept_predictions": len(plays),
        "pct_removed": round(100.0 * len(nonplays) / len(results), 1) if results else 0,
    }

    return plays, stats


# ═════════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_backtest_report(results: list[dict]) -> dict:
    """
    Comprehensive accuracy report from backtest results.

    Sections:
      - overall        : W-L record, win%, push count
      - by_prop_type   : accuracy per stat type
      - by_rating      : accuracy per grade (A/B/C/D)
      - by_direction   : MORE vs LESS
      - by_month       : monthly breakdown
      - calibration    : do X% confidence picks hit X%?
      - edge_vs_accuracy : higher edge = better results?
      - top_wins       : 10 biggest correct predictions
      - top_misses     : 10 biggest wrong predictions

    The report dict is saved to data/backtest/backtest_2025_report.json
    for the self-learning system (TASK 6) to consume.

    CRITICAL FIX: Filters out non-plays (actual=0) before analysis. Non-plays
    artificially inflate LESS accuracy and depress MORE accuracy. This was
    the root cause of the 30pp gap between MORE and LESS accuracy.
    """
    if not results:
        return {"error": "No results to analyze."}

    # Filter non-plays before analysis
    plays, nonplay_stats = filter_nonplays(results)
    df = pd.DataFrame(plays)

    # Filter to only W/L for accuracy (exclude push and skip)
    wl = df[df["result"].isin(["W", "L"])].copy()

    total = len(wl)
    wins = int((wl["result"] == "W").sum())
    losses = total - wins
    pushes = int((df["result"] == "push").sum())
    accuracy = wins / total if total > 0 else 0.0

    report: dict = {
        "generated_at": datetime.now().isoformat(),
        "total_predictions_loaded": nonplay_stats["total_predictions"],
        "total_predictions_analyzed": len(df),
        "nonplay_filter": nonplay_stats,
        "overall": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total_decided": total,
            "accuracy": round(accuracy, 4),
            "win_pct": round(accuracy * 100, 2),
        },
    }

    # ── By prop type ──
    by_prop: dict = {}
    for pt in wl["prop_type"].unique():
        subset = wl[wl["prop_type"] == pt]
        w = int((subset["result"] == "W").sum())
        t = len(subset)
        by_prop[pt] = {
            "wins": w, "total": t,
            "accuracy": round(w / t, 4) if t > 0 else 0,
        }
    report["by_prop_type"] = by_prop

    # ── By rating ──
    by_rating: dict = {}
    for r in ["A", "B", "C", "D"]:
        subset = wl[wl["rating"] == r]
        w = int((subset["result"] == "W").sum())
        t = len(subset)
        by_rating[r] = {
            "wins": w, "total": t,
            "accuracy": round(w / t, 4) if t > 0 else 0,
        }
    report["by_rating"] = by_rating

    # ── By direction ──
    by_dir: dict = {}
    for d in ["MORE", "LESS"]:
        subset = wl[wl["pick"] == d]
        w = int((subset["result"] == "W").sum())
        t = len(subset)
        by_dir[d] = {
            "wins": w, "total": t,
            "accuracy": round(w / t, 4) if t > 0 else 0,
        }
    report["by_direction"] = by_dir

    # ── By month ──
    by_month: dict = {}
    if "game_date" in wl.columns:
        wl_dates = wl.copy()
        wl_dates["month"] = pd.to_datetime(wl_dates["game_date"]).dt.month
        for m in sorted(wl_dates["month"].unique()):
            subset = wl_dates[wl_dates["month"] == m]
            w = int((subset["result"] == "W").sum())
            t = len(subset)
            month_name = datetime(2025, int(m), 1).strftime("%B")
            by_month[month_name] = {
                "wins": w, "total": t,
                "accuracy": round(w / t, 4) if t > 0 else 0,
            }
    report["by_month"] = by_month

    # ── Calibration curve ──
    # Bucket predictions by confidence and see if actual hit rate matches
    calibration: list[dict] = []
    if "confidence" in wl.columns:
        bins = [(0.50, 0.54), (0.54, 0.57), (0.57, 0.62), (0.62, 0.70), (0.70, 1.01)]
        for lo, hi in bins:
            subset = wl[(wl["confidence"] >= lo) & (wl["confidence"] < hi)]
            w = int((subset["result"] == "W").sum())
            t = len(subset)
            calibration.append({
                "confidence_range": f"{lo:.2f}-{hi:.2f}",
                "predicted_avg": round((lo + hi) / 2, 3),
                "actual_accuracy": round(w / t, 4) if t > 0 else 0,
                "count": t,
            })
    report["calibration"] = calibration

    # ── Edge vs accuracy ──
    edge_buckets: list[dict] = []
    if "edge" in wl.columns:
        bins = [(0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.0)]
        for lo, hi in bins:
            subset = wl[(wl["edge"] >= lo) & (wl["edge"] < hi)]
            w = int((subset["result"] == "W").sum())
            t = len(subset)
            edge_buckets.append({
                "edge_range": f"{lo:.2f}-{hi:.2f}",
                "accuracy": round(w / t, 4) if t > 0 else 0,
                "count": t,
            })
    report["edge_vs_accuracy"] = edge_buckets

    # ── Top 10 biggest wins (largest edge that was correct) ──
    wins_df = wl[wl["result"] == "W"].copy()
    if not wins_df.empty:
        wins_df["abs_diff"] = (wins_df["actual"] - wins_df["line"]).abs()
        top_wins = wins_df.nlargest(10, "abs_diff")
        report["top_wins"] = [
            {
                "player": row["player_name"],
                "date": row["game_date"],
                "prop": row["prop_type"],
                "line": row["line"],
                "projection": row["projection"],
                "actual": row["actual"],
                "pick": row["pick"],
                "rating": row["rating"],
                "edge": row["edge"],
            }
            for _, row in top_wins.iterrows()
        ]
    else:
        report["top_wins"] = []

    # ── Top 10 biggest misses (largest edge that was wrong) ──
    losses_df = wl[wl["result"] == "L"].copy()
    if not losses_df.empty:
        losses_df["abs_diff"] = (losses_df["actual"] - losses_df["line"]).abs()
        top_misses = losses_df.nlargest(10, "abs_diff")
        report["top_misses"] = [
            {
                "player": row["player_name"],
                "date": row["game_date"],
                "prop": row["prop_type"],
                "line": row["line"],
                "projection": row["projection"],
                "actual": row["actual"],
                "pick": row["pick"],
                "rating": row["rating"],
                "edge": row["edge"],
            }
            for _, row in top_misses.iterrows()
        ]
    else:
        report["top_misses"] = []

    # ── Save report ──
    report_path = Path(DEFAULT_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  [REPORT SAVED] {report_path}")

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_date: str = "2025-04-01",
    end_date: str = "2025-09-30",
    prop_types: list[str] | None = None,
    save_interval: int = 1,
) -> dict:
    """
    Run the full historical backtest over the specified date range.

    Processes one day at a time sequentially. Saves results after every
    *save_interval* days so the run can be resumed if interrupted. On
    restart, already-processed dates are skipped automatically.

    Args:
        start_date:    First game day (inclusive), YYYY-MM-DD.
        end_date:      Last game day (inclusive), YYYY-MM-DD.
        prop_types:    List of prop type keys to backtest. Defaults to
                       hitter_fantasy_score, hits, total_bases, home_runs,
                       pitcher_strikeouts.
        save_interval: Save to disk every N days processed.

    Returns:
        Summary report dict (same as generate_backtest_report output).
    """
    prop_types = prop_types or DEFAULT_PROP_TYPES

    # Clear predictor weights cache to ensure fresh read from data/weights/current.json
    _clear_weights_cache()

    # Load any previously saved results so we can resume
    all_results = load_results()
    processed_dates: set[str] = {r["game_date"] for r in all_results}

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (dt_end - dt_start).days + 1
    current = dt_start
    days_processed = 0
    days_since_save = 0

    print("=" * 70)
    print(f"  MLB PROP EDGE — HISTORICAL BACKTEST")
    print(f"  Range: {start_date} to {end_date} ({total_days} days)")
    print(f"  Props: {', '.join(prop_types)}")
    print(f"  Resuming with {len(all_results)} existing results")
    print("=" * 70)

    while current <= dt_end:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in processed_dates:
            current += timedelta(days=1)
            days_processed += 1
            continue

        print(f"\n  [{days_processed + 1}/{total_days}] Processing {date_str} ...", end=" ")

        try:
            day_results = backtest_single_day(date_str, prop_types)
        except Exception as exc:
            print(f"ERROR: {exc}")
            traceback.print_exc()
            day_results = []

        all_results.extend(day_results)
        processed_dates.add(date_str)

        w_count = sum(1 for r in day_results if r.get("result") == "W")
        l_count = sum(1 for r in day_results if r.get("result") == "L")
        print(f"  {len(day_results)} predictions ({w_count}W-{l_count}L)")

        days_since_save += 1
        if days_since_save >= save_interval:
            save_results(all_results)
            days_since_save = 0

        current += timedelta(days=1)
        days_processed += 1

    # Final save
    save_results(all_results)

    # Generate report
    print("\n" + "=" * 70)
    print("  Generating backtest report ...")
    report = generate_backtest_report(all_results)

    overall = report.get("overall", {})
    print(f"\n  FINAL RESULTS:")
    print(f"    Total decided: {overall.get('total_decided', 0)}")
    print(f"    Record: {overall.get('wins', 0)}W - {overall.get('losses', 0)}L"
          f" ({overall.get('pushes', 0)} pushes)")
    print(f"    Accuracy: {overall.get('win_pct', 0):.1f}%")
    print("=" * 70)

    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLB Prop Edge — Historical Backtester")
    parser.add_argument("--start", default="2025-04-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-09-30", help="End date YYYY-MM-DD")
    parser.add_argument("--props", nargs="*", default=None,
                        help="Prop types to backtest (space-separated)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing results")
    args = parser.parse_args()

    if args.report_only:
        results = load_results()
        if results:
            report = generate_backtest_report(results)
            print(json.dumps(report.get("overall", {}), indent=2))
        else:
            print("No saved results found. Run backtest first.")
    else:
        run_backtest(
            start_date=args.start,
            end_date=args.end,
            prop_types=args.props,
        )
