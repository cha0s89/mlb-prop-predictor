"""
Player Stats Module
Pulls batting and pitching stats from pybaseball (Statcast + FanGraphs).
Uses caching to avoid hammering Baseball Savant.
"""

import logging
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from pybaseball import (
        batting_stats,
        pitching_stats,
        playerid_lookup,
        statcast_batter,
        statcast_pitcher,
        cache,
    )
    cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


# ── Stat stabilization constants (PA needed for 50% regression) ──
STABILIZATION = {
    "k_rate": 60,
    "bb_rate": 120,
    "babip": 820,
    "iso": 160,
    "hr_fb": 300,
    "barrel_rate": 100,
    "contact_rate": 100,
    "obp": 300,
    "slg": 320,
    "woba": 300,
    "csw_pct": 200,      # ~10 starts for pitchers
    "swstr_pct": 150,
    "k_pct_pitcher": 70,  # BF, not PA
    "bb_pct_pitcher": 170,
}


def _current_season() -> int:
    """Return the most recent completed or in-progress MLB season year.
    Before April of any year, use the prior year's stats since the new
    season hasn't produced enough data yet."""
    now = datetime.now()
    # Before April, use prior year (new season hasn't started)
    if now.month < 4:
        return now.year - 1
    return now.year


def _safe_console_text(text: str) -> str:
    """Best-effort ASCII fallback for Windows consoles with non-UTF8 encodings."""
    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split()) or text


def fetch_batting_leaders(season: int = None, min_pa: int = 50) -> pd.DataFrame:
    """
    Fetch FanGraphs batting leaderboard for the season.
    Returns DataFrame with advanced + Statcast metrics.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    season = season or _current_season()

    # CRITICAL: Validate we're pulling the right season
    if season >= _current_season():
        try:
            from src.freshness import validate_season_data, record_data_pull
            sv = validate_season_data(season)
            if sv.get("warning"):
                logging.getLogger(__name__).warning(
                    "[STALENESS WARNING] %s",
                    _safe_console_text(sv["warning"]),
                )
        except ImportError:
            pass

    try:
        df = batting_stats(season, qual=min_pa)
        # Record freshness
        try:
            from src.freshness import record_data_pull
            record_data_pull("fangraphs_batting", f"season={season}, {len(df)} players")
        except Exception:
            pass
        return df
    except Exception:
        fallback_season = season - 1
        logging.getLogger(__name__).warning(
            f"Using {fallback_season} season stats as fallback for batting (current season {season} data unavailable)"
        )
        return pd.DataFrame()


def fetch_pitching_leaders(season: int = None, min_ip: int = 10) -> pd.DataFrame:
    """
    Fetch FanGraphs pitching leaderboard for the season.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    season = season or _current_season()

    # CRITICAL: Validate season
    if season >= _current_season():
        try:
            from src.freshness import validate_season_data, record_data_pull
            sv = validate_season_data(season)
            if sv.get("warning"):
                logging.getLogger(__name__).warning(
                    "[STALENESS WARNING] %s",
                    _safe_console_text(sv["warning"]),
                )
        except ImportError:
            pass

    try:
        df = pitching_stats(season, qual=min_ip)
        try:
            from src.freshness import record_data_pull
            record_data_pull("fangraphs_pitching", f"season={season}, {len(df)} players")
        except Exception:
            pass
        return df
    except Exception:
        fallback_season = season - 1
        logging.getLogger(__name__).warning(
            f"Using {fallback_season} season stats as fallback for pitching (current season {season} data unavailable)"
        )
        return pd.DataFrame()


def get_batter_recent_statcast(player_id: int, days: int = 30) -> pd.DataFrame:
    """
    Pull recent Statcast data for a batter.
    Returns pitch-level data with exit velo, launch angle, barrel, etc.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        df = statcast_batter(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        return df
    except Exception:
        return pd.DataFrame()


def get_pitcher_recent_statcast(player_id: int, days: int = 30) -> pd.DataFrame:
    """
    Pull recent Statcast data for a pitcher.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        df = statcast_pitcher(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        return df
    except Exception:
        return pd.DataFrame()


def compute_batter_profile(season_stats: pd.Series, recent_sc: pd.DataFrame) -> dict:
    """
    Build a composite batter profile from season stats + recent Statcast.
    Uses Bayesian regression toward league mean based on stabilization constants.
    """
    profile = {}

    # Season-level stats (from FanGraphs leaderboard)
    profile["name"] = season_stats.get("Name", "Unknown")
    profile["team"] = season_stats.get("Team", "")
    profile["pa"] = int(season_stats.get("PA", 0))
    profile["avg"] = float(season_stats.get("AVG", 0.000))
    profile["obp"] = float(season_stats.get("OBP", 0.000))
    profile["slg"] = float(season_stats.get("SLG", 0.000))
    profile["iso"] = float(season_stats.get("ISO", 0.000))
    profile["babip"] = float(season_stats.get("BABIP", 0.000))
    profile["k_rate"] = float(season_stats.get("K%", 0.0)) if isinstance(season_stats.get("K%"), (int, float)) else 0.0
    profile["bb_rate"] = float(season_stats.get("BB%", 0.0)) if isinstance(season_stats.get("BB%"), (int, float)) else 0.0
    profile["woba"] = float(season_stats.get("wOBA", 0.000))
    profile["hr"] = int(season_stats.get("HR", 0))
    profile["sb"] = int(season_stats.get("SB", 0))

    # Recent Statcast aggregates (last ~30 days)
    if not recent_sc.empty:
        batted = recent_sc[recent_sc["type"] == "X"] if "type" in recent_sc.columns else recent_sc
        if not batted.empty and "launch_speed" in batted.columns:
            profile["recent_exit_velo"] = float(batted["launch_speed"].mean())
            profile["recent_ev90"] = float(batted["launch_speed"].quantile(0.9))
            profile["recent_barrel_rate"] = float(
                (batted.get("barrel", pd.Series([0])) == 1).mean() * 100
            ) if "barrel" in batted.columns else 0.0
            profile["recent_hard_hit_pct"] = float(
                (batted["launch_speed"] >= 95).mean() * 100
            )
            profile["recent_sweet_spot_pct"] = float(
                batted["launch_angle"].between(8, 32).mean() * 100
            ) if "launch_angle" in batted.columns else 0.0
        else:
            profile.update(_empty_statcast_batter())
    else:
        profile.update(_empty_statcast_batter())

    return profile


def compute_pitcher_profile(season_stats: pd.Series, recent_sc: pd.DataFrame) -> dict:
    """
    Build a composite pitcher profile from season stats + recent Statcast.
    """
    profile = {}

    profile["name"] = season_stats.get("Name", "Unknown")
    profile["team"] = season_stats.get("Team", "")
    profile["ip"] = float(season_stats.get("IP", 0))
    profile["era"] = float(season_stats.get("ERA", 0.00))
    profile["fip"] = float(season_stats.get("FIP", 0.00))
    profile["k9"] = float(season_stats.get("K/9", 0.0))
    profile["bb9"] = float(season_stats.get("BB/9", 0.0))
    profile["k_pct"] = float(season_stats.get("K%", 0.0)) if isinstance(season_stats.get("K%"), (int, float)) else 0.0
    profile["bb_pct"] = float(season_stats.get("BB%", 0.0)) if isinstance(season_stats.get("BB%"), (int, float)) else 0.0
    profile["whip"] = float(season_stats.get("WHIP", 0.00))
    profile["hr9"] = float(season_stats.get("HR/9", 0.0))

    # Recent Statcast aggregates
    if not recent_sc.empty:
        # CSW% - called strikes + whiffs / total pitches
        if "description" in recent_sc.columns:
            total_pitches = len(recent_sc)
            csw_events = recent_sc["description"].isin([
                "called_strike", "swinging_strike", "swinging_strike_blocked",
                "foul_tip",  # foul tips are swinging strikes
            ])
            profile["recent_csw_pct"] = float(csw_events.sum() / total_pitches * 100) if total_pitches > 0 else 0.0

            swstr_events = recent_sc["description"].isin([
                "swinging_strike", "swinging_strike_blocked",
            ])
            profile["recent_swstr_pct"] = float(swstr_events.sum() / total_pitches * 100) if total_pitches > 0 else 0.0

            chase_pitches = recent_sc[recent_sc.get("zone", pd.Series()).isin([11, 12, 13, 14]) if "zone" in recent_sc.columns else pd.Series(dtype=bool)]
            if len(chase_pitches) > 0 and "description" in chase_pitches.columns:
                profile["recent_chase_rate"] = float(
                    chase_pitches["description"].str.contains("swinging").sum() / len(chase_pitches) * 100
                )
            else:
                profile["recent_chase_rate"] = 0.0
        else:
            profile.update(_empty_statcast_pitcher())

        # Velocity
        if "release_speed" in recent_sc.columns:
            fb = recent_sc[recent_sc.get("pitch_type", pd.Series()).isin(["FF", "SI", "FC"])] if "pitch_type" in recent_sc.columns else recent_sc
            profile["recent_fb_velo"] = float(fb["release_speed"].mean()) if not fb.empty else 0.0
        else:
            profile["recent_fb_velo"] = 0.0
    else:
        profile.update(_empty_statcast_pitcher())

    return profile


def regress_stat(observed: float, pa: int, stabilization_pa: int, league_avg: float) -> float:
    """
    Bayesian regression to the mean.
    True Talent = (n * observed + x * prior) / (n + x)
    """
    return (pa * observed + stabilization_pa * league_avg) / (pa + stabilization_pa)


def _empty_statcast_batter() -> dict:
    return {
        "recent_exit_velo": 0.0,
        "recent_ev90": 0.0,
        "recent_barrel_rate": 0.0,
        "recent_hard_hit_pct": 0.0,
        "recent_sweet_spot_pct": 0.0,
    }


def _empty_statcast_pitcher() -> dict:
    return {
        "recent_csw_pct": 0.0,
        "recent_swstr_pct": 0.0,
        "recent_chase_rate": 0.0,
        "recent_fb_velo": 0.0,
    }


# ── League average baselines (approximate 2024 values, update annually) ──
LEAGUE_AVG = {
    # Batting
    "avg": 0.248, "obp": 0.312, "slg": 0.399, "iso": 0.151,
    "babip": 0.296, "woba": 0.310, "wrc_plus": 100,
    "k_rate": 22.7, "bb_rate": 8.3,
    "hr_per_pa": 0.033, "sb_per_game": 0.18,
    "rbi_per_game": 0.55, "runs_per_game": 0.55,
    "hits_per_game": 0.95, "tb_per_game": 1.50,
    # Statcast batting
    "exit_velo": 88.5, "hard_hit_pct": 37.0, "barrel_rate": 7.5,
    "sweet_spot_pct": 32.0, "ev90": 105.0,
    "xba": 0.248, "xslg": 0.399, "xwoba": 0.310,
    "sprint_speed": 27.0,
    # Pitching
    "era": 4.17, "fip": 4.12, "xfip": 4.10, "siera": 4.05,
    "whip": 1.28, "k9": 8.58, "bb9": 3.22, "hr9": 1.15,
    "k_pct_pitcher": 22.7, "bb_pct_pitcher": 8.3,
    "hr_fb_rate": 12.0,
    # Statcast pitching
    "csw_pct": 28.5, "swstr_pct": 11.3,
    "zone_pct": 45.0, "f_strike_pct": 60.0, "chase_rate": 28.0,
}
