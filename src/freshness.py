"""
Data Freshness & Validation Module
NEVER trust cached or stale data silently. Every data pull gets timestamped.
If data is older than the threshold, the user gets warned.

This exists because of one rule: DON'T LIE TO THE USER ABOUT DATA AGE.
If we're showing last week's Statcast data, the user needs to KNOW that.
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from pybaseball import cache as pb_cache
    PYBASEBALL_OK = True
except ImportError:
    PYBASEBALL_OK = False


FRESHNESS_LOG = Path("data/freshness.json")

# Maximum age (in minutes) before data is flagged as stale
STALENESS_THRESHOLDS = {
    "prizepicks_lines": 30,      # Lines can move fast — 30 min max
    "sharp_odds": 30,            # Same — odds move
    "statcast_batter": 360,      # 6 hours — Statcast updates after games
    "statcast_pitcher": 360,
    "fangraphs_batting": 720,    # 12 hours — season stats update overnight
    "fangraphs_pitching": 720,
    "weather": 120,              # 2 hours — forecasts update
    "schedule": 120,             # 2 hours — lineups confirmed ~90 min pre-game
    "umpire": 240,               # 4 hours — assigned day-of
    "bvp_matchup": 1440,         # 24 hours — historical, doesn't change fast
}


def record_data_pull(source: str, details: str = ""):
    """Record when a data source was last pulled."""
    FRESHNESS_LOG.parent.mkdir(parents=True, exist_ok=True)

    log = _load_log()
    log[source] = {
        "timestamp": datetime.now().isoformat(),
        "unix": time.time(),
        "details": details,
    }
    _save_log(log)


def check_freshness(source: str) -> dict:
    """
    Check if a data source is fresh enough.
    
    Returns:
        dict with: is_fresh, age_minutes, last_pull, warning (if stale)
    """
    log = _load_log()
    entry = log.get(source)

    if not entry:
        return {
            "is_fresh": False,
            "age_minutes": None,
            "last_pull": None,
            "warning": f"⚠️ {source}: NEVER PULLED — no data available",
            "status": "never_pulled",
        }

    last_pull = datetime.fromisoformat(entry["timestamp"])
    age = datetime.now() - last_pull
    age_minutes = age.total_seconds() / 60

    threshold = STALENESS_THRESHOLDS.get(source, 360)
    is_fresh = age_minutes <= threshold

    if is_fresh:
        return {
            "is_fresh": True,
            "age_minutes": round(age_minutes, 1),
            "last_pull": last_pull.strftime("%I:%M %p"),
            "warning": None,
            "status": "fresh",
        }

    # Build warning based on how stale
    if age_minutes > 1440:
        severity = "🔴 CRITICAL"
        msg = f"{source}: Data is {age_minutes/1440:.1f} DAYS old"
    elif age_minutes > 360:
        severity = "🟡 WARNING"
        msg = f"{source}: Data is {age_minutes/60:.1f} hours old"
    else:
        severity = "⚠️ STALE"
        msg = f"{source}: Data is {age_minutes:.0f} min old (threshold: {threshold} min)"

    return {
        "is_fresh": False,
        "age_minutes": round(age_minutes, 1),
        "last_pull": last_pull.strftime("%I:%M %p"),
        "warning": f"{severity} — {msg}",
        "status": "stale",
    }


def get_all_freshness() -> dict:
    """Get freshness status for all tracked data sources."""
    results = {}
    for source in STALENESS_THRESHOLDS:
        results[source] = check_freshness(source)
    return results


def get_freshness_summary() -> str:
    """One-line summary for the UI header."""
    all_fresh = get_all_freshness()
    stale = [k for k, v in all_fresh.items() if not v["is_fresh"] and v["status"] != "never_pulled"]
    never = [k for k, v in all_fresh.items() if v["status"] == "never_pulled"]
    fresh = [k for k, v in all_fresh.items() if v["is_fresh"]]

    if stale:
        return f"⚠️ {len(stale)} stale source(s): {', '.join(stale)}"
    elif never:
        # Only warn about sources we'd expect to have
        important_never = [n for n in never if n in ("prizepicks_lines", "sharp_odds")]
        if important_never:
            return f"🔵 Waiting for first data pull"
        return f"🟢 Core data fresh ({len(fresh)} sources)"
    else:
        return f"🟢 All data fresh ({len(fresh)} sources)"


def clear_pybaseball_cache():
    """
    Force clear pybaseball's disk cache.
    
    pybaseball caches Statcast data to ~/.pybaseball/cache/
    This is GOOD for performance but BAD if you don't realize
    you're looking at yesterday's data.
    
    Call this daily or when you suspect stale data.
    """
    if not PYBASEBALL_OK:
        return {"cleared": False, "reason": "pybaseball not installed"}

    try:
        cache_dir = Path.home() / ".pybaseball" / "cache"
        if cache_dir.exists():
            file_count = 0
            for f in cache_dir.glob("*"):
                if f.is_file():
                    # Only clear files older than 6 hours
                    age = time.time() - f.stat().st_mtime
                    if age > 6 * 3600:
                        f.unlink()
                        file_count += 1
            return {"cleared": True, "files_removed": file_count}
        return {"cleared": True, "files_removed": 0, "note": "No cache directory found"}
    except Exception as e:
        return {"cleared": False, "reason": str(e)}


def force_clear_all_cache():
    """Nuclear option: clear ALL pybaseball cache regardless of age."""
    if not PYBASEBALL_OK:
        return {"cleared": False}

    try:
        cache_dir = Path.home() / ".pybaseball" / "cache"
        if cache_dir.exists():
            count = 0
            for f in cache_dir.glob("*"):
                if f.is_file():
                    f.unlink()
                    count += 1
            return {"cleared": True, "files_removed": count}
        return {"cleared": True, "files_removed": 0}
    except Exception as e:
        return {"cleared": False, "reason": str(e)}


def validate_season_data(data_year: int = None) -> dict:
    """
    Check if we're pulling current season data or stale prior-season data.
    This is the EXACT bug that killed the previous tool.
    """
    current_year = datetime.now().year
    current_month = datetime.now().month

    # During offseason (Nov-Feb), prior year data is expected
    if current_month in (11, 12, 1, 2):
        expected_year = current_year if current_month >= 11 else current_year - 1
        in_season = False
    else:
        expected_year = current_year
        in_season = True

    if data_year is None:
        return {
            "expected_year": expected_year,
            "in_season": in_season,
            "warning": None,
        }

    if data_year == expected_year:
        return {
            "valid": True,
            "data_year": data_year,
            "expected_year": expected_year,
            "warning": None,
        }
    elif data_year < expected_year:
        return {
            "valid": False,
            "data_year": data_year,
            "expected_year": expected_year,
            "warning": f"🔴 WRONG SEASON: Data is from {data_year} but current season is {expected_year}. "
                       f"This is EXACTLY the bug that breaks prop predictions. Clear cache and re-pull.",
        }
    else:
        return {
            "valid": False,
            "data_year": data_year,
            "expected_year": expected_year,
            "warning": f"⚠️ Data year {data_year} is in the future. Something is wrong.",
        }


def validate_game_date(game_date_str: str) -> dict:
    """
    Verify that game/line data is actually from today, not yesterday.
    """
    try:
        game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return {"valid": False, "warning": "⚠️ Could not parse game date"}

    today = datetime.now().date()
    game_day = game_date.date()

    if game_day == today:
        return {"valid": True, "warning": None}
    elif game_day < today:
        days_old = (today - game_day).days
        return {
            "valid": False,
            "warning": f"🔴 STALE LINES: This game was {days_old} day(s) ago. You may be looking at old data.",
        }
    else:
        # Future game — normal for upcoming schedule
        days_ahead = (game_day - today).days
        return {
            "valid": True if days_ahead <= 3 else False,
            "warning": f"Game is {days_ahead} day(s) from now" if days_ahead > 1 else None,
        }


def startup_checks() -> list:
    """
    Run on app startup. Returns list of warnings to display.
    """
    warnings = []

    # Check season
    sv = validate_season_data()
    if not sv.get("in_season"):
        warnings.append("ℹ️ MLB is in the offseason. Limited data available.")

    # Check pybaseball cache age
    if PYBASEBALL_OK:
        cache_dir = Path.home() / ".pybaseball" / "cache"
        if cache_dir.exists():
            files = list(cache_dir.glob("*"))
            if files:
                oldest = min(f.stat().st_mtime for f in files if f.is_file())
                age_hours = (time.time() - oldest) / 3600
                if age_hours > 24:
                    warnings.append(
                        f"⚠️ pybaseball cache has files {age_hours:.0f} hours old. "
                        f"Consider clearing cache for fresh data."
                    )

    # Check data freshness
    freshness = get_all_freshness()
    critical_stale = [
        k for k, v in freshness.items()
        if not v["is_fresh"] and v["status"] == "stale" and k in ("prizepicks_lines", "sharp_odds")
    ]
    if critical_stale:
        warnings.append(f"⚠️ Stale data detected: {', '.join(critical_stale)}. Refresh recommended.")

    return warnings


def _load_log() -> dict:
    if FRESHNESS_LOG.exists():
        with open(FRESHNESS_LOG) as f:
            return json.load(f)
    return {}


def _save_log(log: dict):
    FRESHNESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FRESHNESS_LOG, "w") as f:
        json.dump(log, f, indent=2)
