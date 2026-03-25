"""
Recent Trends Module
Short-term momentum detection for batters and pitchers.

Compares recent performance windows to detect hot/cold streaks.
Research says: hot hand exists in MLB (Green & Zwiebel 2017, 2M at-bats)
but is hard to exploit above what good projections capture.

We use it as a TIEBREAKER, not a primary signal. When two picks look
equally good on fundamentals, recent form can nudge the decision.

Weight in final model: ~5-8% max. Never the primary driver.
"""

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from pybaseball import cache, playerid_lookup, statcast_batter, statcast_pitcher
    cache.enable()
    PYBASEBALL_OK = True
except ImportError:
    PYBASEBALL_OK = False


@lru_cache(maxsize=2048)
def _lookup_batter_mlbam_id(player_name: str) -> int | None:
    """Resolve an MLBAM player id from a display name for trend lookups."""
    if not PYBASEBALL_OK or not player_name:
        return None

    try:
        parts = str(player_name).strip().split()
        if len(parts) < 2:
            return None
        result = playerid_lookup(parts[-1], parts[0])
        if result.empty:
            return None
        row = result.sort_values("key_mlbam", ascending=False).iloc[0]
        mlbam_id = row.get("key_mlbam")
        if pd.isna(mlbam_id):
            return None
        return int(mlbam_id)
    except Exception:
        return None


def get_batter_trend(player_id: int | str | None, short_days: int = 7,
                      long_days: int = 21) -> dict:
    """
    Compare batter's recent short window vs longer baseline.

    Returns trend multiplier and supporting stats.
    Hot streak: short-term AVG/SLG significantly above long-term baseline.
    Cold streak: opposite.

    Uses Statcast pitch-level data for accuracy.
    """
    if isinstance(player_id, str):
        player_id = _lookup_batter_mlbam_id(player_id)

    if not PYBASEBALL_OK or not player_id:
        return _neutral_trend()

    end = datetime.now()
    start_long = end - timedelta(days=long_days)
    start_short = end - timedelta(days=short_days)

    try:
        data = statcast_batter(
            start_long.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        if data.empty:
            return _neutral_trend()
    except Exception:
        return _neutral_trend()

    # Filter to at-bat-ending events
    if "events" not in data.columns:
        return _neutral_trend()

    events = data.dropna(subset=["events"]).copy()
    events["game_date"] = pd.to_datetime(events["game_date"], errors="coerce")

    if events.empty:
        return _neutral_trend()

    # Split into long and short windows
    short_start = pd.Timestamp(start_short)
    long_events = events.copy()
    short_events = events[events["game_date"] >= short_start]

    # Calculate AVG and SLG for each window
    long_stats = _calc_batting_stats(long_events)
    short_stats = _calc_batting_stats(short_events)

    if long_stats["ab"] < 15 or short_stats["ab"] < 5:
        return _neutral_trend()

    # Trend multiplier: how much better/worse is short-term vs long-term
    avg_ratio = short_stats["avg"] / long_stats["avg"] if long_stats["avg"] > 0 else 1.0
    slg_ratio = short_stats["slg"] / long_stats["slg"] if long_stats["slg"] > 0 else 1.0

    # Blend AVG and SLG trends (SLG captures power trending too)
    raw_trend = avg_ratio * 0.6 + slg_ratio * 0.4

    # Cap the multiplier to prevent overreaction (±10% max)
    trend_mult = max(0.90, min(1.10, raw_trend))

    # Recent Statcast quality (last 7 days)
    short_batted = short_events[short_events.get("type", pd.Series()) == "X"] if "type" in short_events.columns else pd.DataFrame()
    recent_ev = float(short_batted["launch_speed"].mean()) if not short_batted.empty and "launch_speed" in short_batted.columns else 0.0
    recent_hh = float((short_batted["launch_speed"] >= 95).mean() * 100) if not short_batted.empty and "launch_speed" in short_batted.columns else 0.0

    # Determine trend label
    if trend_mult >= 1.06:
        label = "🔥 Hot streak"
    elif trend_mult >= 1.03:
        label = "↗️ Trending up"
    elif trend_mult <= 0.94:
        label = "❄️ Cold streak"
    elif trend_mult <= 0.97:
        label = "↘️ Trending down"
    else:
        label = "➖ Stable"

    return {
        "has_data": True,
        "trend_multiplier": round(trend_mult, 4),
        "label": label,
        "short_avg": round(short_stats["avg"], 3),
        "long_avg": round(long_stats["avg"], 3),
        "short_slg": round(short_stats["slg"], 3),
        "long_slg": round(long_stats["slg"], 3),
        "short_ab": short_stats["ab"],
        "long_ab": long_stats["ab"],
        "short_hits": short_stats["hits"],
        "short_hr": short_stats["hr"],
        "recent_exit_velo": round(recent_ev, 1),
        "recent_hard_hit_pct": round(recent_hh, 1),
        "short_days": short_days,
        "long_days": long_days,
    }


def get_pitcher_trend(player_id: int, short_days: int = 14,
                       long_days: int = 45) -> dict:
    """
    Compare pitcher's recent form to season baseline.

    Uses longer windows for pitchers since they pitch every 5 days.
    Short = last 2-3 starts. Long = last 7-9 starts.

    Key signals: K rate trending, velocity changes, CSW% changes.
    Velocity decline between starts is a real predictive signal
    (FiveThirtyEight Hidden Markov Model research).
    """
    if not PYBASEBALL_OK or not player_id:
        return _neutral_pitcher_trend()

    end = datetime.now()
    start_long = end - timedelta(days=long_days)
    start_short = end - timedelta(days=short_days)

    try:
        data = statcast_pitcher(
            start_long.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        if data.empty:
            return _neutral_pitcher_trend()
    except Exception:
        return _neutral_pitcher_trend()

    data["game_date"] = pd.to_datetime(data.get("game_date", pd.Series()), errors="coerce")
    short_data = data[data["game_date"] >= pd.Timestamp(start_short)]

    # Pitch counts
    long_pitches = len(data)
    short_pitches = len(short_data)

    if long_pitches < 100 or short_pitches < 30:
        return _neutral_pitcher_trend()

    # K-rate trend (CSW% and SwStr%)
    long_csw = _calc_csw(data)
    short_csw = _calc_csw(short_data)
    csw_trend = short_csw / long_csw if long_csw > 0 else 1.0

    # Velocity trend (fastball velo)
    long_velo = _calc_fb_velo(data)
    short_velo = _calc_fb_velo(short_data)
    velo_trend = short_velo / long_velo if long_velo > 0 else 1.0

    # K-rate trend feeds K prop multiplier
    # Velo decline is a red flag for all pitcher props
    k_trend_mult = max(0.90, min(1.10, csw_trend))
    overall_trend = max(0.92, min(1.08, csw_trend * 0.6 + velo_trend * 0.4))

    if overall_trend >= 1.04:
        label = "🔥 Pitcher trending sharp"
    elif overall_trend >= 1.02:
        label = "↗️ Slightly improving"
    elif overall_trend <= 0.96:
        label = "❄️ Pitcher struggling"
    elif overall_trend <= 0.98:
        label = "↘️ Slight decline"
    else:
        label = "➖ Stable form"

    # Velo-specific warning
    velo_warning = None
    if long_velo > 0 and short_velo > 0:
        velo_drop = long_velo - short_velo
        if velo_drop >= 1.5:
            velo_warning = f"⚠️ Velocity drop: {long_velo:.1f} → {short_velo:.1f} mph (-{velo_drop:.1f})"
        elif velo_drop <= -1.0:
            velo_warning = f"✅ Velocity up: {long_velo:.1f} → {short_velo:.1f} mph (+{abs(velo_drop):.1f})"

    return {
        "has_data": True,
        "k_trend_multiplier": round(k_trend_mult, 4),
        "overall_trend_multiplier": round(overall_trend, 4),
        "label": label,
        "short_csw": round(short_csw, 1),
        "long_csw": round(long_csw, 1),
        "short_velo": round(short_velo, 1) if short_velo else 0.0,
        "long_velo": round(long_velo, 1) if long_velo else 0.0,
        "velo_warning": velo_warning,
        "short_pitches": short_pitches,
        "long_pitches": long_pitches,
    }


def _calc_batting_stats(events_df: pd.DataFrame) -> dict:
    """Calculate basic batting stats from event-level data."""
    if events_df.empty or "events" not in events_df.columns:
        return {"ab": 0, "hits": 0, "hr": 0, "avg": 0.0, "slg": 0.0}

    hit_events = {"single", "double", "triple", "home_run"}
    non_ab_events = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt",
                     "sac_fly_double_play", "catcher_interf"}

    total = len(events_df)
    non_ab = len(events_df[events_df["events"].isin(non_ab_events)])
    ab = total - non_ab

    hits = len(events_df[events_df["events"].isin(hit_events)])
    singles = len(events_df[events_df["events"] == "single"])
    doubles = len(events_df[events_df["events"] == "double"])
    triples = len(events_df[events_df["events"] == "triple"])
    hr = len(events_df[events_df["events"] == "home_run"])

    avg = hits / ab if ab > 0 else 0.0
    tb = singles + doubles * 2 + triples * 3 + hr * 4
    slg = tb / ab if ab > 0 else 0.0

    return {"ab": ab, "hits": hits, "hr": hr, "avg": avg, "slg": slg, "tb": tb}


def _calc_csw(pitch_data: pd.DataFrame) -> float:
    """Calculate CSW% from pitch-level data."""
    if pitch_data.empty or "description" not in pitch_data.columns:
        return 0.0
    total = len(pitch_data)
    csw = pitch_data["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"
    ]).sum()
    return csw / total * 100 if total > 0 else 0.0


def _calc_fb_velo(pitch_data: pd.DataFrame) -> float:
    """Calculate average fastball velocity."""
    if pitch_data.empty:
        return 0.0
    if "pitch_type" in pitch_data.columns and "release_speed" in pitch_data.columns:
        fb = pitch_data[pitch_data["pitch_type"].isin(["FF", "SI", "FC"])]
        return float(fb["release_speed"].mean()) if not fb.empty else 0.0
    return 0.0


def _neutral_trend() -> dict:
    return {
        "has_data": False,
        "trend_multiplier": 1.0,
        "label": "➖ No trend data",
    }


def _neutral_pitcher_trend() -> dict:
    return {
        "has_data": False,
        "k_trend_multiplier": 1.0,
        "overall_trend_multiplier": 1.0,
        "label": "➖ No trend data",
    }
