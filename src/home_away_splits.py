"""Per-player home/away split adjustments.

SEPARATE from park factors. Park factors adjust for the venue environment.
This module adjusts for each player's personal performance difference when
playing at home vs. on the road (routine, comfort, crowd familiarity, etc.).

Data source: MLB Stats API homeAndAway split for current season.
Bayesian regression toward 1.0 (neutral) when sample is small (< 80 PA equiv).
Output capped at [0.90, 1.10] to prevent overreaction.

Graceful fallback to 1.0 if data unavailable or insufficient.
"""

from __future__ import annotations

import math
import requests
from datetime import date
from functools import lru_cache
from typing import Optional

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Bayesian prior weight — regress toward neutral when sample is small
_PRIOR_PA = 80.0

# Hard cap on any adjustment
_CAP_LOW = 0.90
_CAP_HIGH = 1.10

# Batter props where home/away splits apply
_BATTER_PROPS = {
    "hits", "total_bases", "home_runs", "rbis", "runs",
    "singles", "doubles", "batter_strikeouts", "walks",
    "hitter_fantasy_score", "hits_runs_rbis",
}

# Pitcher props where home/away splits apply
_PITCHER_PROPS = {
    "pitcher_strikeouts", "earned_runs", "walks_allowed", "hits_allowed",
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(n) else n


def _bayesian_blend(raw_mult: float, sample_pa: float) -> float:
    """Blend raw multiplier toward 1.0 based on sample size, then cap."""
    if sample_pa <= 0:
        return 1.0
    blended = (sample_pa * raw_mult + _PRIOR_PA * 1.0) / (sample_pa + _PRIOR_PA)
    return max(_CAP_LOW, min(_CAP_HIGH, blended))


def _rate_mult(
    home_count: float,
    away_count: float,
    home_denom: float,
    away_denom: float,
    is_home: bool,
) -> tuple[float, float]:
    """
    Compute (raw_multiplier, sample_denom) for home or away game.

    raw_mult = side_rate / overall_rate, where overall is PA-weighted average.
    sample_denom is passed to _bayesian_blend for regression.
    """
    if home_denom <= 0 or away_denom <= 0:
        return 1.0, 0.0
    total_count = home_count + away_count
    total_denom = home_denom + away_denom
    if total_count <= 0 or total_denom <= 0:
        return 1.0, 0.0

    overall_rate = total_count / total_denom
    if overall_rate <= 0:
        return 1.0, 0.0

    if is_home:
        side_rate = home_count / home_denom if home_denom > 0 else 0.0
        sample = home_denom
    else:
        side_rate = away_count / away_denom if away_denom > 0 else 0.0
        sample = away_denom

    return side_rate / overall_rate, sample


# ─── MLB Stats API ────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _fetch_splits(player_id: int, season: int, group: str) -> dict:
    """Fetch home/away splits from MLB Stats API. Results cached in memory.

    Returns {"home": {...stat fields...}, "away": {...stat fields...}}
    or {} on failure.
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{player_id}/stats",
            params={
                "stats": "homeAndAway",
                "group": group,
                "season": season,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    home_stat: dict = {}
    away_stat: dict = {}
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            code = split.get("split", {}).get("code", "")
            if code == "H":
                home_stat = split.get("stat", {})
            elif code == "A":
                away_stat = split.get("stat", {})

    if not home_stat and not away_stat:
        return {}
    return {"home": home_stat, "away": away_stat}


# ─── batter multiplier ────────────────────────────────────────────────────────

def _batter_mult(home: dict, away: dict, prop_type: str, is_home: bool) -> float:
    """Return Bayesian-blended multiplier for a batter prop."""
    home_pa = _safe_float(home.get("plateAppearances") or home.get("atBats", 0))
    away_pa = _safe_float(away.get("plateAppearances") or away.get("atBats", 0))
    home_ab = _safe_float(home.get("atBats", 0))
    away_ab = _safe_float(away.get("atBats", 0))

    # Need at least 10 PA total for any signal
    if home_pa + away_pa < 10:
        return 1.0

    if prop_type in ("hits", "rbis", "hitter_fantasy_score", "hits_runs_rbis"):
        h_hits = _safe_float(home.get("hits", 0))
        a_hits = _safe_float(away.get("hits", 0))
        raw, sample = _rate_mult(h_hits, a_hits, home_ab, away_ab, is_home)

    elif prop_type == "total_bases":
        # Use SLG * AB as proxy for total bases
        h_slg = _safe_float(home.get("slg", 0))
        a_slg = _safe_float(away.get("slg", 0))
        h_tb = h_slg * home_ab
        a_tb = a_slg * away_ab
        raw, sample = _rate_mult(h_tb, a_tb, home_ab, away_ab, is_home)

    elif prop_type == "home_runs":
        h_hr = _safe_float(home.get("homeRuns", 0))
        a_hr = _safe_float(away.get("homeRuns", 0))
        raw, sample = _rate_mult(h_hr, a_hr, home_ab, away_ab, is_home)

    elif prop_type == "singles":
        h_s = max(
            _safe_float(home.get("hits", 0))
            - _safe_float(home.get("homeRuns", 0))
            - _safe_float(home.get("doubles", 0))
            - _safe_float(home.get("triples", 0)),
            0.0,
        )
        a_s = max(
            _safe_float(away.get("hits", 0))
            - _safe_float(away.get("homeRuns", 0))
            - _safe_float(away.get("doubles", 0))
            - _safe_float(away.get("triples", 0)),
            0.0,
        )
        raw, sample = _rate_mult(h_s, a_s, home_ab, away_ab, is_home)

    elif prop_type == "doubles":
        h_d = _safe_float(home.get("doubles", 0))
        a_d = _safe_float(away.get("doubles", 0))
        raw, sample = _rate_mult(h_d, a_d, home_ab, away_ab, is_home)

    elif prop_type == "batter_strikeouts":
        h_k = _safe_float(home.get("strikeOuts", 0))
        a_k = _safe_float(away.get("strikeOuts", 0))
        raw, sample = _rate_mult(h_k, a_k, home_ab, away_ab, is_home)

    elif prop_type == "walks":
        h_bb = _safe_float(home.get("baseOnBalls", 0))
        a_bb = _safe_float(away.get("baseOnBalls", 0))
        raw, sample = _rate_mult(h_bb, a_bb, home_pa, away_pa, is_home)

    elif prop_type == "runs":
        h_r = _safe_float(home.get("runs", 0))
        a_r = _safe_float(away.get("runs", 0))
        raw, sample = _rate_mult(h_r, a_r, home_pa, away_pa, is_home)

    else:
        return 1.0

    if raw == 1.0 and sample == 0.0:
        return 1.0
    return _bayesian_blend(raw, sample)


# ─── pitcher multiplier ───────────────────────────────────────────────────────

def _pitcher_mult(home: dict, away: dict, prop_type: str, is_home: bool) -> float:
    """Return Bayesian-blended multiplier for a pitcher prop."""
    home_ip = _safe_float(home.get("inningsPitched", 0))
    away_ip = _safe_float(away.get("inningsPitched", 0))

    if home_ip + away_ip < 5:
        return 1.0

    # Convert IP to approximate PA-equivalent for Bayesian weight (≈4.3 BF/IP)
    BF_PER_IP = 4.3
    home_pa_equiv = home_ip * BF_PER_IP
    away_pa_equiv = away_ip * BF_PER_IP

    if prop_type == "pitcher_strikeouts":
        h_k = _safe_float(home.get("strikeOuts", 0))
        a_k = _safe_float(away.get("strikeOuts", 0))
        raw, sample_ip = _rate_mult(h_k, a_k, home_ip, away_ip, is_home)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "earned_runs":
        h_er = _safe_float(home.get("earnedRuns", 0))
        a_er = _safe_float(away.get("earnedRuns", 0))
        raw, sample_ip = _rate_mult(h_er, a_er, home_ip, away_ip, is_home)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "walks_allowed":
        h_bb = _safe_float(home.get("baseOnBalls", 0))
        a_bb = _safe_float(away.get("baseOnBalls", 0))
        raw, sample_ip = _rate_mult(h_bb, a_bb, home_ip, away_ip, is_home)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "hits_allowed":
        h_h = _safe_float(home.get("hits", 0))
        a_h = _safe_float(away.get("hits", 0))
        raw, sample_ip = _rate_mult(h_h, a_h, home_ip, away_ip, is_home)
        sample = sample_ip * BF_PER_IP

    else:
        return 1.0

    if raw == 1.0 and sample == 0.0:
        return 1.0
    return _bayesian_blend(raw, sample)


# ─── public API ───────────────────────────────────────────────────────────────

def get_home_away_split_multiplier(
    player_id: int,
    is_home: bool,
    prop_type: str,
    is_pitcher: bool = False,
    season: Optional[int] = None,
) -> float:
    """Return a projection multiplier based on the player's home/away split.

    This is SEPARATE from park factors. Park factors adjust for the venue.
    This adjusts for the player's personal comfort/routine advantage at home
    vs. on the road — independent of which park they play in.

    Args:
        player_id: MLB MLBAM player ID.
        is_home: True if the player is playing at home today.
        prop_type: Internal stat name (e.g. "hits", "pitcher_strikeouts").
        is_pitcher: True for pitcher props.
        season: Season year (defaults to current calendar year).

    Returns:
        float in [0.90, 1.10]. Returns 1.0 on any error or insufficient data.
    """
    if not player_id or player_id <= 0:
        return 1.0

    # Only apply to known prop types
    if is_pitcher and prop_type not in _PITCHER_PROPS:
        return 1.0
    if not is_pitcher and prop_type not in _BATTER_PROPS:
        return 1.0

    if season is None:
        season = date.today().year

    try:
        group = "pitching" if is_pitcher else "hitting"
        splits = _fetch_splits(player_id, season, group)
        if not splits:
            return 1.0

        home_stat = splits.get("home", {})
        away_stat = splits.get("away", {})
        if not home_stat and not away_stat:
            return 1.0

        if is_pitcher:
            return _pitcher_mult(home_stat, away_stat, prop_type, is_home)
        else:
            return _batter_mult(home_stat, away_stat, prop_type, is_home)

    except Exception:
        return 1.0


def clear_splits_cache() -> None:
    """Clear the in-memory splits cache (useful for testing or daily refresh)."""
    _fetch_splits.cache_clear()
