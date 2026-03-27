"""Per-player day/night split adjustments.

~15-20% of MLB players show statistically reliable day/night splits
(30-50 pt OBP difference). Wrigley Field afternoon shadows are notorious.

Data source: MLB Stats API dayNight split for current season.
Bayesian regression toward 1.0 when sample is small (< 80 PA equiv).
Output capped at [0.92, 1.08] to prevent overreaction.

Special: Wrigley Field (CHC home) afternoon games apply an additional
shadow suppression factor to K props.

Graceful fallback to 1.0 if data unavailable or insufficient.
"""

from __future__ import annotations

import math
import requests
from datetime import date, datetime
from functools import lru_cache
from typing import Optional

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Bayesian prior weight — regress toward neutral when sample is small
_PRIOR_PA = 80.0

# Hard cap on any adjustment (tighter than home/away because day/night is noisier)
_CAP_LOW = 0.92
_CAP_HIGH = 1.08

# Wrigley Field afternoon shadow adjustments for K props
# Batters struggle to pick up the ball → more strikeouts; pitcher K lines slightly suppressed
_WRIGLEY_SHADOW_BATTER_K = 1.03
_WRIGLEY_SHADOW_PITCHER_K = 0.97

# CHC team abbreviations treated as "Wrigley home"
_WRIGLEY_TEAMS = {"CHC", "CHN", "CUBS"}

# Batter props where day/night splits apply
_BATTER_PROPS = {
    "hits", "total_bases", "home_runs", "rbis", "runs",
    "singles", "doubles", "batter_strikeouts", "walks",
    "hitter_fantasy_score", "hits_runs_rbis",
}

# Pitcher props where day/night splits apply
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
    day_count: float,
    night_count: float,
    day_denom: float,
    night_denom: float,
    is_day: bool,
) -> tuple[float, float]:
    """
    Compute (raw_multiplier, sample_denom) for a day or night game.

    raw_mult = side_rate / overall_rate (PA-weighted average).
    sample_denom is passed to _bayesian_blend for regression.
    """
    if day_denom <= 0 or night_denom <= 0:
        return 1.0, 0.0
    total_count = day_count + night_count
    total_denom = day_denom + night_denom
    if total_count <= 0 or total_denom <= 0:
        return 1.0, 0.0

    overall_rate = total_count / total_denom
    if overall_rate <= 0:
        return 1.0, 0.0

    if is_day:
        side_rate = day_count / day_denom if day_denom > 0 else 0.0
        sample = day_denom
    else:
        side_rate = night_count / night_denom if night_denom > 0 else 0.0
        sample = night_denom

    return side_rate / overall_rate, sample


# ─── MLB Stats API ────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _fetch_splits(player_id: int, season: int, group: str) -> dict:
    """Fetch day/night splits from MLB Stats API. Results cached in memory.

    Returns {"day": {...stat fields...}, "night": {...stat fields...}}
    or {} on failure.
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{player_id}/stats",
            params={
                "stats": "dayNight",
                "group": group,
                "season": season,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    day_stat: dict = {}
    night_stat: dict = {}
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            code = split.get("split", {}).get("code", "")
            if code == "D":
                day_stat = split.get("stat", {})
            elif code == "N":
                night_stat = split.get("stat", {})

    if not day_stat and not night_stat:
        return {}
    return {"day": day_stat, "night": night_stat}


# ─── day/night determination ─────────────────────────────────────────────────

def is_day_game(start_time_utc: str) -> bool:
    """Return True if the game starts before 5 PM Eastern time.

    Uses a simple UTC → Eastern approximation:
    UTC-4 during DST (March–November), UTC-5 otherwise.
    Falls back to False (night game) on parse errors.
    """
    if not start_time_utc:
        return False
    try:
        iso = start_time_utc.strip()
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        dt_utc = datetime.fromisoformat(iso)
        month = dt_utc.month
        et_offset = -4 if 3 <= month <= 11 else -5
        hour_et = (dt_utc.hour + et_offset) % 24
        return hour_et < 17  # before 5 PM ET
    except Exception:
        return False


# ─── batter multiplier ────────────────────────────────────────────────────────

def _batter_mult(day: dict, night: dict, prop_type: str, is_day: bool) -> float:
    """Return Bayesian-blended multiplier for a batter prop."""
    day_pa = _safe_float(day.get("plateAppearances") or day.get("atBats", 0))
    night_pa = _safe_float(night.get("plateAppearances") or night.get("atBats", 0))
    day_ab = _safe_float(day.get("atBats", 0))
    night_ab = _safe_float(night.get("atBats", 0))

    # Need at least 10 PA total for any signal
    if day_pa + night_pa < 10:
        return 1.0

    if prop_type in ("hits", "rbis", "hitter_fantasy_score", "hits_runs_rbis"):
        d_hits = _safe_float(day.get("hits", 0))
        n_hits = _safe_float(night.get("hits", 0))
        raw, sample = _rate_mult(d_hits, n_hits, day_ab, night_ab, is_day)

    elif prop_type == "total_bases":
        d_slg = _safe_float(day.get("slg", 0))
        n_slg = _safe_float(night.get("slg", 0))
        d_tb = d_slg * day_ab
        n_tb = n_slg * night_ab
        raw, sample = _rate_mult(d_tb, n_tb, day_ab, night_ab, is_day)

    elif prop_type == "home_runs":
        d_hr = _safe_float(day.get("homeRuns", 0))
        n_hr = _safe_float(night.get("homeRuns", 0))
        raw, sample = _rate_mult(d_hr, n_hr, day_ab, night_ab, is_day)

    elif prop_type == "singles":
        d_s = max(
            _safe_float(day.get("hits", 0))
            - _safe_float(day.get("homeRuns", 0))
            - _safe_float(day.get("doubles", 0))
            - _safe_float(day.get("triples", 0)),
            0.0,
        )
        n_s = max(
            _safe_float(night.get("hits", 0))
            - _safe_float(night.get("homeRuns", 0))
            - _safe_float(night.get("doubles", 0))
            - _safe_float(night.get("triples", 0)),
            0.0,
        )
        raw, sample = _rate_mult(d_s, n_s, day_ab, night_ab, is_day)

    elif prop_type == "doubles":
        d_d = _safe_float(day.get("doubles", 0))
        n_d = _safe_float(night.get("doubles", 0))
        raw, sample = _rate_mult(d_d, n_d, day_ab, night_ab, is_day)

    elif prop_type == "batter_strikeouts":
        d_k = _safe_float(day.get("strikeOuts", 0))
        n_k = _safe_float(night.get("strikeOuts", 0))
        raw, sample = _rate_mult(d_k, n_k, day_ab, night_ab, is_day)

    elif prop_type == "walks":
        d_bb = _safe_float(day.get("baseOnBalls", 0))
        n_bb = _safe_float(night.get("baseOnBalls", 0))
        raw, sample = _rate_mult(d_bb, n_bb, day_pa, night_pa, is_day)

    elif prop_type == "runs":
        d_r = _safe_float(day.get("runs", 0))
        n_r = _safe_float(night.get("runs", 0))
        raw, sample = _rate_mult(d_r, n_r, day_pa, night_pa, is_day)

    else:
        return 1.0

    if raw == 1.0 and sample == 0.0:
        return 1.0
    return _bayesian_blend(raw, sample)


# ─── pitcher multiplier ───────────────────────────────────────────────────────

def _pitcher_mult(day: dict, night: dict, prop_type: str, is_day: bool) -> float:
    """Return Bayesian-blended multiplier for a pitcher prop."""
    day_ip = _safe_float(day.get("inningsPitched", 0))
    night_ip = _safe_float(night.get("inningsPitched", 0))

    if day_ip + night_ip < 5:
        return 1.0

    # Convert IP to approximate PA-equivalent for Bayesian weight (≈4.3 BF/IP)
    BF_PER_IP = 4.3

    if prop_type == "pitcher_strikeouts":
        d_k = _safe_float(day.get("strikeOuts", 0))
        n_k = _safe_float(night.get("strikeOuts", 0))
        raw, sample_ip = _rate_mult(d_k, n_k, day_ip, night_ip, is_day)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "earned_runs":
        d_er = _safe_float(day.get("earnedRuns", 0))
        n_er = _safe_float(night.get("earnedRuns", 0))
        raw, sample_ip = _rate_mult(d_er, n_er, day_ip, night_ip, is_day)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "walks_allowed":
        d_bb = _safe_float(day.get("baseOnBalls", 0))
        n_bb = _safe_float(night.get("baseOnBalls", 0))
        raw, sample_ip = _rate_mult(d_bb, n_bb, day_ip, night_ip, is_day)
        sample = sample_ip * BF_PER_IP

    elif prop_type == "hits_allowed":
        d_h = _safe_float(day.get("hits", 0))
        n_h = _safe_float(night.get("hits", 0))
        raw, sample_ip = _rate_mult(d_h, n_h, day_ip, night_ip, is_day)
        sample = sample_ip * BF_PER_IP

    else:
        return 1.0

    if raw == 1.0 and sample == 0.0:
        return 1.0
    return _bayesian_blend(raw, sample)


# ─── Wrigley shadow ───────────────────────────────────────────────────────────

def get_wrigley_shadow_mult(
    prop_type: str,
    is_pitcher: bool,
    park_team: Optional[str],
    is_day: bool,
) -> float:
    """Return Wrigley Field afternoon shadow factor for K props.

    Wrigley's mid-afternoon sun/shadow makes it harder for batters to pick
    up the ball, inflating batter Ks and modestly suppressing pitcher K lines
    (batters are more contact-averse, game pace changes).

    Returns 1.0 if conditions don't apply.
    """
    if not is_day:
        return 1.0
    if not park_team or park_team.upper() not in _WRIGLEY_TEAMS:
        return 1.0
    if is_pitcher and prop_type == "pitcher_strikeouts":
        return _WRIGLEY_SHADOW_PITCHER_K
    if not is_pitcher and prop_type == "batter_strikeouts":
        return _WRIGLEY_SHADOW_BATTER_K
    return 1.0


# ─── public API ───────────────────────────────────────────────────────────────

def get_day_night_split_multiplier(
    player_id: int,
    is_day: bool,
    prop_type: str,
    is_pitcher: bool = False,
    season: Optional[int] = None,
) -> float:
    """Return a projection multiplier based on the player's day/night split.

    Args:
        player_id: MLB MLBAM player ID.
        is_day: True if the game starts before 5 PM Eastern time.
        prop_type: Internal stat name (e.g. "hits", "pitcher_strikeouts").
        is_pitcher: True for pitcher props.
        season: Season year (defaults to current calendar year).

    Returns:
        float in [0.92, 1.08]. Returns 1.0 on any error or insufficient data.
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

        day_stat = splits.get("day", {})
        night_stat = splits.get("night", {})
        if not day_stat and not night_stat:
            return 1.0

        if is_pitcher:
            return _pitcher_mult(day_stat, night_stat, prop_type, is_day)
        else:
            return _batter_mult(day_stat, night_stat, prop_type, is_day)

    except Exception:
        return 1.0


def clear_splits_cache() -> None:
    """Clear the in-memory splits cache (useful for testing or daily refresh)."""
    _fetch_splits.cache_clear()
