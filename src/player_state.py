"""
Player State Detection Module

Detects breakout, slump, and fatigue states using real Statcast signals —
NOT simple rolling averages. This module classifies players into states
that serve as an uncertainty/tail overlay on the base projection.

States for hitters:
  - HEATING_UP: xwOBA trending up, barrel% increasing, contact quality rising
  - COLD_UNLUCKY: low BABIP but good xwOBA (regression candidate — buy low)
  - COLD_REAL: low xwOBA + low contact quality (genuine slump)
  - CHANGED_APPROACH: K%/BB% shifted materially (swing decision change)
  - NORMAL: no significant state change detected

States for pitchers:
  - SHARPENING: velo up, CSW% up, whiff rate improving
  - FATIGUED: velo declining, pitch count trending up, efficiency dropping
  - LOSING_STUFF: velo/movement declining without workload explanation
  - COMMAND_CHANGE: zone%/first-strike% shifted materially
  - NORMAL: no significant state change detected

Design principles (per Sierra's audit):
  - Use state-change SIGNALS, not rolling averages
  - Output as overlay/uncertainty modifier, not replacement for base projection
  - All features are deltas (short window vs long baseline)
  - No hand-tuned multipliers — just state classification + confidence adjustment

Data: pybaseball Statcast (cached), computed at board-build time.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pybaseball import statcast_batter, statcast_pitcher, cache
    cache.enable()
    PYBASEBALL_OK = True
except ImportError:
    PYBASEBALL_OK = False


# ─────────────────────────────────────────────
# HITTER STATE DETECTION
# ─────────────────────────────────────────────

def detect_hitter_state(player_id: int, short_days: int = 14,
                         long_days: int = 45) -> dict:
    """
    Detect a hitter's current performance state from Statcast data.

    Compares short window (14 days) to long baseline (45 days) across
    multiple quality-of-contact and plate discipline metrics.

    Args:
        player_id: MLBAM player ID
        short_days: Recent window (default 14 days — ~10-12 games)
        long_days: Baseline window (default 45 days — ~35-40 games)

    Returns:
        Dict with state, confidence_adjustment, features, and explanation.
    """
    result = _neutral_hitter_state()

    if not PYBASEBALL_OK or not player_id:
        return result

    end = datetime.now()
    start_long = end - timedelta(days=long_days)
    start_short = end - timedelta(days=short_days)

    try:
        data = statcast_batter(
            start_long.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        if data.empty or len(data) < 50:
            return result
    except Exception:
        return result

    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    long_data = data.copy()
    short_data = data[data["game_date"] >= pd.Timestamp(start_short)]

    if len(short_data) < 20:
        return result

    # Compute features (deltas between short and long windows)
    features = {}

    # 1. xwOBA delta (expected weighted on-base average)
    long_xwoba = _mean_col(long_data, "estimated_woba_using_speedangle")
    short_xwoba = _mean_col(short_data, "estimated_woba_using_speedangle")
    if long_xwoba and short_xwoba:
        features["xwoba_delta"] = round(short_xwoba - long_xwoba, 3)

    # 2. Barrel rate delta
    long_barrel = _barrel_rate(long_data)
    short_barrel = _barrel_rate(short_data)
    if long_barrel is not None and short_barrel is not None:
        features["barrel_rate_delta"] = round(short_barrel - long_barrel, 1)

    # 3. Hard hit rate delta (95+ mph exit velo)
    long_hh = _hard_hit_rate(long_data)
    short_hh = _hard_hit_rate(short_data)
    if long_hh is not None and short_hh is not None:
        features["hard_hit_delta"] = round(short_hh - long_hh, 1)

    # 4. K% delta (plate discipline)
    long_k = _k_rate(long_data)
    short_k = _k_rate(short_data)
    if long_k is not None and short_k is not None:
        features["k_rate_delta"] = round(short_k - long_k, 1)

    # 5. BB% delta
    long_bb = _bb_rate(long_data)
    short_bb = _bb_rate(short_data)
    if long_bb is not None and short_bb is not None:
        features["bb_rate_delta"] = round(short_bb - long_bb, 1)

    # 6. BABIP vs xBA gap (luck indicator)
    short_babip = _babip(short_data)
    short_xba = _mean_col(short_data, "estimated_ba_using_speedangle")
    if short_babip is not None and short_xba is not None and short_xba > 0:
        features["babip_xba_gap"] = round(short_babip - short_xba, 3)

    # 7. Chase rate delta (swing at pitches outside zone)
    long_chase = _chase_rate(long_data)
    short_chase = _chase_rate(short_data)
    if long_chase is not None and short_chase is not None:
        features["chase_rate_delta"] = round(short_chase - long_chase, 1)

    result["features"] = features
    result["has_data"] = True

    # Classify state based on feature constellation
    xwd = features.get("xwoba_delta", 0)
    brd = features.get("barrel_rate_delta", 0)
    hhd = features.get("hard_hit_delta", 0)
    krd = features.get("k_rate_delta", 0)
    bxg = features.get("babip_xba_gap", 0)

    if xwd >= 0.025 and (brd >= 2.0 or hhd >= 3.0):
        result["state"] = "heating_up"
        result["confidence_adjustment"] = 1.04  # +4% confidence boost
        result["explanation"] = (
            f"xwOBA up {xwd:+.3f}, barrel% {brd:+.1f}pp — "
            "quality of contact improving"
        )
    elif xwd >= 0.010 and bxg < -0.030:
        result["state"] = "cold_unlucky"
        result["confidence_adjustment"] = 1.02  # Buy low — regression coming
        result["explanation"] = (
            f"xwOBA up {xwd:+.3f} but BABIP-xBA gap {bxg:+.3f} — "
            "unlucky, expect regression up"
        )
    elif xwd <= -0.030 and (brd <= -2.0 or hhd <= -3.0):
        result["state"] = "cold_real"
        result["confidence_adjustment"] = 0.94  # Genuine slump
        result["explanation"] = (
            f"xwOBA down {xwd:+.3f}, barrel% {brd:+.1f}pp — "
            "real decline in contact quality"
        )
    elif abs(krd) >= 5.0 or abs(features.get("bb_rate_delta", 0)) >= 3.0:
        result["state"] = "changed_approach"
        result["confidence_adjustment"] = 0.97  # Increase uncertainty
        result["explanation"] = (
            f"K% delta {krd:+.1f}pp, BB% delta {features.get('bb_rate_delta', 0):+.1f}pp — "
            "plate approach has shifted"
        )
    else:
        result["state"] = "normal"

    return result


# ─────────────────────────────────────────────
# PITCHER STATE DETECTION
# ─────────────────────────────────────────────

def detect_pitcher_state(player_id: int, short_days: int = 21,
                          long_days: int = 60) -> dict:
    """
    Detect a pitcher's current state from Statcast pitch data.

    Uses longer windows than hitters because pitchers pitch every 5 days.
    Short = last 3-4 starts. Long = last 10-12 starts.

    Returns:
        Dict with state, confidence_adjustment, features, and explanation.
    """
    result = _neutral_pitcher_state()

    if not PYBASEBALL_OK or not player_id:
        return result

    end = datetime.now()
    start_long = end - timedelta(days=long_days)
    start_short = end - timedelta(days=short_days)

    try:
        data = statcast_pitcher(
            start_long.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            player_id,
        )
        if data.empty or len(data) < 100:
            return result
    except Exception:
        return result

    data["game_date"] = pd.to_datetime(data.get("game_date", pd.Series()), errors="coerce")
    long_data = data.copy()
    short_data = data[data["game_date"] >= pd.Timestamp(start_short)]

    if len(short_data) < 40:
        return result

    features = {}

    # 1. Fastball velocity delta
    long_velo = _fb_velo(long_data)
    short_velo = _fb_velo(short_data)
    if long_velo and short_velo:
        features["velo_delta"] = round(short_velo - long_velo, 1)

    # 2. CSW% delta (called strike + whiff rate)
    long_csw = _csw_rate(long_data)
    short_csw = _csw_rate(short_data)
    if long_csw and short_csw:
        features["csw_delta"] = round(short_csw - long_csw, 1)

    # 3. Whiff rate delta
    long_whiff = _whiff_rate(long_data)
    short_whiff = _whiff_rate(short_data)
    if long_whiff is not None and short_whiff is not None:
        features["whiff_delta"] = round(short_whiff - long_whiff, 1)

    # 4. Zone% delta (throwing in the zone)
    long_zone = _zone_rate(long_data)
    short_zone = _zone_rate(short_data)
    if long_zone is not None and short_zone is not None:
        features["zone_delta"] = round(short_zone - long_zone, 1)

    # 5. First-pitch strike% delta
    long_fstrike = _first_strike_rate(long_data)
    short_fstrike = _first_strike_rate(short_data)
    if long_fstrike is not None and short_fstrike is not None:
        features["first_strike_delta"] = round(short_fstrike - long_fstrike, 1)

    result["features"] = features
    result["has_data"] = True

    # Classify state
    vd = features.get("velo_delta", 0)
    cswd = features.get("csw_delta", 0)
    wd = features.get("whiff_delta", 0)
    zd = features.get("zone_delta", 0)

    if (cswd >= 2.0 or wd >= 3.0) and vd >= -0.3:
        result["state"] = "sharpening"
        result["confidence_adjustment"] = 1.04
        result["explanation"] = (
            f"CSW% {cswd:+.1f}pp, whiff% {wd:+.1f}pp — "
            "stuff is improving"
        )
    elif vd <= -1.0 and cswd <= -1.0:
        result["state"] = "fatigued"
        result["confidence_adjustment"] = 0.93
        result["explanation"] = (
            f"Velo {vd:+.1f} mph, CSW% {cswd:+.1f}pp — "
            "possible fatigue or declining stuff"
        )
    elif vd <= -1.5:
        result["state"] = "losing_stuff"
        result["confidence_adjustment"] = 0.90
        result["explanation"] = (
            f"Velo dropped {vd:+.1f} mph — "
            "significant velocity decline"
        )
    elif abs(zd) >= 5.0 or abs(features.get("first_strike_delta", 0)) >= 5.0:
        result["state"] = "command_change"
        result["confidence_adjustment"] = 0.96
        result["explanation"] = (
            f"Zone% {zd:+.1f}pp, 1st strike% "
            f"{features.get('first_strike_delta', 0):+.1f}pp — "
            "command profile shifted"
        )
    else:
        result["state"] = "normal"

    return result


# ─────────────────────────────────────────────
# STATCAST FEATURE HELPERS
# ─────────────────────────────────────────────

def _mean_col(df: pd.DataFrame, col: str) -> Optional[float]:
    """Mean of a column, ignoring NaN. Returns None if column missing or empty."""
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) >= 5 else None


def _barrel_rate(df: pd.DataFrame) -> Optional[float]:
    """Barrel% from batted ball data."""
    if "barrel" not in df.columns or "type" not in df.columns:
        return None
    batted = df[df["type"] == "X"]
    if len(batted) < 10:
        return None
    return float((batted["barrel"] == 1).mean() * 100)


def _hard_hit_rate(df: pd.DataFrame) -> Optional[float]:
    """Hard hit% (95+ mph exit velo)."""
    if "launch_speed" not in df.columns or "type" not in df.columns:
        return None
    batted = df[df["type"] == "X"].dropna(subset=["launch_speed"])
    if len(batted) < 10:
        return None
    return float((batted["launch_speed"] >= 95).mean() * 100)


def _k_rate(df: pd.DataFrame) -> Optional[float]:
    """K% from event-level data."""
    if "events" not in df.columns:
        return None
    events = df.dropna(subset=["events"])
    if len(events) < 15:
        return None
    ks = events["events"].isin(["strikeout", "strikeout_double_play"]).sum()
    return float(ks / len(events) * 100)


def _bb_rate(df: pd.DataFrame) -> Optional[float]:
    """BB% from event-level data."""
    if "events" not in df.columns:
        return None
    events = df.dropna(subset=["events"])
    if len(events) < 15:
        return None
    bbs = events["events"].isin(["walk", "hit_by_pitch"]).sum()
    return float(bbs / len(events) * 100)


def _babip(df: pd.DataFrame) -> Optional[float]:
    """BABIP from event-level data."""
    if "events" not in df.columns:
        return None
    events = df.dropna(subset=["events"])
    hits = events["events"].isin(["single", "double", "triple"]).sum()
    hr = (events["events"] == "home_run").sum()
    ks = events["events"].isin(["strikeout", "strikeout_double_play"]).sum()
    sf = (events["events"] == "sac_fly").sum()
    ab_equiv = len(events) - events["events"].isin(
        ["walk", "hit_by_pitch", "sac_bunt", "catcher_interf"]
    ).sum()
    denom = ab_equiv - ks - hr + sf
    if denom <= 0:
        return None
    return float(hits / denom)


def _chase_rate(df: pd.DataFrame) -> Optional[float]:
    """Chase rate: swing% at pitches outside the zone."""
    if "zone" not in df.columns or "description" not in df.columns:
        return None
    outside = df[df["zone"].isin([11, 12, 13, 14])]
    if len(outside) < 20:
        return None
    swings = outside["description"].str.contains(
        "swing|foul|hit_into", case=False, na=False
    ).sum()
    return float(swings / len(outside) * 100)


def _fb_velo(df: pd.DataFrame) -> Optional[float]:
    """Average fastball velocity."""
    if "pitch_type" not in df.columns or "release_speed" not in df.columns:
        return None
    fb = df[df["pitch_type"].isin(["FF", "SI", "FC"])]
    velos = pd.to_numeric(fb["release_speed"], errors="coerce").dropna()
    return float(velos.mean()) if len(velos) >= 15 else None


def _csw_rate(df: pd.DataFrame) -> Optional[float]:
    """CSW% (called strikes + whiffs / total pitches)."""
    if "description" not in df.columns:
        return None
    total = len(df)
    if total < 50:
        return None
    csw = df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"
    ]).sum()
    return float(csw / total * 100)


def _whiff_rate(df: pd.DataFrame) -> Optional[float]:
    """Whiff rate: swinging strikes / swings."""
    if "description" not in df.columns:
        return None
    swings = df["description"].str.contains(
        "swing|foul", case=False, na=False
    ).sum()
    whiffs = df["description"].str.contains(
        "swinging_strike", case=False, na=False
    ).sum()
    return float(whiffs / swings * 100) if swings >= 20 else None


def _zone_rate(df: pd.DataFrame) -> Optional[float]:
    """Zone%: pitches in the strike zone."""
    if "zone" not in df.columns:
        return None
    total = len(df)
    if total < 50:
        return None
    in_zone = df["zone"].isin(range(1, 10)).sum()
    return float(in_zone / total * 100)


def _first_strike_rate(df: pd.DataFrame) -> Optional[float]:
    """First-pitch strike%."""
    if "balls" not in df.columns or "strikes" not in df.columns:
        return None
    first_pitches = df[(df["balls"] == 0) & (df["strikes"] == 0)]
    if len(first_pitches) < 20:
        return None
    strikes = first_pitches["description"].isin([
        "called_strike", "swinging_strike", "foul", "foul_tip",
        "swinging_strike_blocked", "hit_into_play",
    ]).sum()
    return float(strikes / len(first_pitches) * 100)


# ─────────────────────────────────────────────
# NEUTRAL STATE DEFAULTS
# ─────────────────────────────────────────────

def _neutral_hitter_state() -> dict:
    return {
        "state": "normal",
        "has_data": False,
        "confidence_adjustment": 1.0,
        "features": {},
        "explanation": "",
    }


def _neutral_pitcher_state() -> dict:
    return {
        "state": "normal",
        "has_data": False,
        "confidence_adjustment": 1.0,
        "features": {},
        "explanation": "",
    }
