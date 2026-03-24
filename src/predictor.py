"""
Prediction Engine v2 — Full Sabermetric Integration
Every prop type. Every relevant stat. Proper stabilization and weighting.

PROP TYPES COVERED:
  Pitcher: Strikeouts, Outs Recorded, Earned Runs, Walks, Hits Allowed
  Batter:  Hits, Total Bases, Home Runs, RBIs, Runs, Stolen Bases,
           Batter Strikeouts, Walks, Singles, Doubles, H+R+RBI combo

KEY INPUTS PER PROJECTION:
  1. Season stats (regressed via Bayesian stabilization)
  2. Statcast quality metrics (xBA, xSLG, barrel rate, CSW%, EV90, etc.)
  3. BvP matchup history (batter vs. specific pitcher)
  4. Platoon splits (L/R handedness advantage)
  5. Park factors (general, HR-specific, K-specific)
  6. Weather adjustments (temp, wind, humidity)
  7. Umpire tendencies (K-rate impact)
  8. Lineup position (PA opportunity adjustment)
  9. Opposing quality (pitcher FIP for batters, lineup wOBA for pitchers)

WEIGHTING PHILOSOPHY (from research):
  - Sharp book edge is primary (handled in sharp_odds.py)
  - This engine is the SECONDARY confirmation layer
  - Bayesian regression pulls small samples toward league average
  - Statcast expected stats are descriptive, not purely predictive
    (used as one signal among many, not the sole driver)
  - BvP data is powerful but requires 10+ PA to be meaningful
"""

import json
import os
import numpy as np
from datetime import date, datetime
from scipy import stats as sp_stats  # kept for potential future use
from typing import Optional

from src import distributions
from src.weather import get_stat_specific_weather_adjustment


# ═══════════════════════════════════════════════════════
# LEARNED WEIGHTS (loaded from data/weights/current.json)
# ═══════════════════════════════════════════════════════

_WEIGHTS_CACHE = {}
_CALIBRATION_CACHE = {}

DEFAULT_CALIBRATION_BLEND_WEIGHTS = {
    "hits": 0.0,                    # Theoretical optimal (72.5%)
    "total_bases": 0.90,            # 56.1% → 65.8% with empirical + floors
    "pitcher_strikeouts": 0.0,
    "hitter_fantasy_score": 1.0,    # 58.0% → 61.7% with full empirical
}
DEFAULT_CONFIDENCE_SHRINKAGE = 0.70

DIST_DEFAULTS = {
    "pitcher_strikeouts": ("betabinom", 2.2),
    "hits": ("negbin", 2.2),
    "total_bases": ("negbin", 2.5),
    "stolen_bases": ("negbin", 2.5),
    "hitter_fantasy_score": ("gamma", 4.0),
    "earned_runs": ("negbin", 2.2),
    "batter_strikeouts": ("negbin", 1.4),
    "walks_allowed": ("negbin", 1.6),
    "hits_allowed": ("negbin", 1.5),
    "rbis": ("negbin", 1.8),
    "runs": ("negbin", 1.7),
    "walks": ("negbin", 1.5),
    "singles": ("negbin", 1.3),
    "doubles": ("negbin", 1.6),
    "pitching_outs": ("normal", 1.3),
    "hits_runs_rbis": ("gamma", 2.8),
    "home_runs": ("negbin", 2.0),
}

TAIL_SIGNAL_DEFAULTS = {
    "label_thresholds": {
        "breakout_medium": 0.10,
        "breakout_high": 0.20,
        "dud_medium": 0.20,
        "dud_high": 0.35,
    },
    "prop_thresholds": {
        "hits": {"good_over": 3, "bad_under": 0},
        "total_bases": {"good_over": 4, "bad_under": 0},
        "home_runs": {"good_over": 2, "bad_under": 0},
        "rbis": {"good_over": 3, "bad_under": 0},
        "runs": {"good_over": 2, "bad_under": 0},
        "stolen_bases": {"good_over": 2, "bad_under": 0},
        "walks": {"good_over": 2, "bad_under": 0},
        "singles": {"good_over": 2, "bad_under": 0},
        "doubles": {"good_over": 2, "bad_under": 0},
        "hitter_fantasy_score": {"good_over": 14, "bad_under": 3},
        "hits_runs_rbis": {"good_over": 4, "bad_under": 0},
        "pitcher_strikeouts": {"good_over": 8, "bad_under": 4},
        "pitching_outs": {"good_over": 21, "bad_under": 15},
        "earned_runs": {"good_under": 1, "bad_over": 4},
        "walks_allowed": {"good_under": 1, "bad_over": 3},
        "hits_allowed": {"good_under": 4, "bad_over": 8},
        "batter_strikeouts": {"good_under": 0, "bad_over": 3},
    },
}


def _clear_weights_cache() -> None:
    """Clear the weights cache so next _load_weights() reads fresh from disk."""
    _WEIGHTS_CACHE.clear()
    _CALIBRATION_CACHE.clear()


def _merge_weight_overrides(base: dict, override: dict) -> dict:
    """Recursively merge a runtime override onto the active weight set."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_weight_overrides(base[key], value)
        else:
            base[key] = value
    return base


def _load_weights() -> dict:
    """Load learned weights from current.json plus any runtime override."""
    if _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "data", "weights")
    weights_path = os.path.join(weights_dir, "current.json")
    runtime_override_path = os.path.join(weights_dir, "runtime_override.json")
    try:
        with open(weights_path, encoding="utf-8") as f:
            w = json.load(f)
        if os.path.exists(runtime_override_path):
            try:
                with open(runtime_override_path, encoding="utf-8") as f:
                    runtime_override = json.load(f)
                w = _merge_weight_overrides(w, runtime_override)
            except (OSError, json.JSONDecodeError):
                pass
        _WEIGHTS_CACHE.update(w)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return _WEIGHTS_CACHE


def _load_calibration() -> dict:
    """Load empirical calibration tables from calibration_v015.json. Cached."""
    if _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE
    cal_path = os.path.join(os.path.dirname(__file__), "..", "data", "weights", "calibration_v015.json")
    try:
        with open(cal_path, encoding="utf-8") as f:
            c = json.load(f)
        _CALIBRATION_CACHE.update(c)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return _CALIBRATION_CACHE


def _get_calibration_blend_weights(weights: Optional[dict] = None) -> dict:
    """Return configured empirical blend weights with safe defaults."""
    weights = weights or _load_weights()
    configured = weights.get("calibration_blend_weights")
    if not configured:
        configured = weights.get("metadata", {}).get("calibration_blend_weights", {})

    merged = dict(DEFAULT_CALIBRATION_BLEND_WEIGHTS)
    if isinstance(configured, dict):
        for key, value in configured.items():
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
    return merged


def _get_confidence_shrinkage(prop_type: str, weights: Optional[dict] = None) -> float:
    """Return the configured confidence shrinkage for this prop."""
    weights = weights or _load_weights()
    raw = weights.get("confidence_shrinkage", DEFAULT_CONFIDENCE_SHRINKAGE)
    if isinstance(raw, dict):
        raw = raw.get(prop_type, raw.get("default", DEFAULT_CONFIDENCE_SHRINKAGE))
    try:
        shrinkage = float(raw)
    except (TypeError, ValueError):
        shrinkage = DEFAULT_CONFIDENCE_SHRINKAGE
    return max(0.30, min(1.10, shrinkage))


def get_distribution_config(prop_type: str, weights: Optional[dict] = None) -> dict:
    """Return the active distribution config for a prop."""
    weights = weights or _load_weights()
    dist_params = weights.get("distribution_params", {})
    variance_ratios = weights.get("variance_ratios", {})
    prop_dist = dist_params.get(prop_type, {})
    dist_type = prop_dist.get("type", "negbin")

    if not prop_dist and prop_type in DIST_DEFAULTS:
        dist_type, default_vr = DIST_DEFAULTS[prop_type]
    else:
        default_vr = 1.5

    return {
        "dist_type": dist_type,
        "var_ratio": prop_dist.get("vr", variance_ratios.get(prop_type, default_vr)),
        "phi": prop_dist.get("phi", 25),
    }


def _get_tail_signal_config(weights: Optional[dict] = None) -> dict:
    """Return breakout/dud thresholds and label cutoffs."""
    weights = weights or _load_weights()
    cfg = weights.get("tail_signal_config", {}) or {}
    merged = {
        "label_thresholds": dict(TAIL_SIGNAL_DEFAULTS["label_thresholds"]),
        "prop_thresholds": dict(TAIL_SIGNAL_DEFAULTS["prop_thresholds"]),
        "label_thresholds_by_prop": {},
    }
    if isinstance(cfg.get("label_thresholds"), dict):
        merged["label_thresholds"].update(cfg["label_thresholds"])
    if isinstance(cfg.get("label_thresholds_by_prop"), dict):
        for prop, prop_cfg in cfg["label_thresholds_by_prop"].items():
            if isinstance(prop_cfg, dict):
                merged["label_thresholds_by_prop"][prop] = dict(prop_cfg)
    if isinstance(cfg.get("prop_thresholds"), dict):
        for prop, prop_cfg in cfg["prop_thresholds"].items():
            base = dict(merged["prop_thresholds"].get(prop, {}))
            if isinstance(prop_cfg, dict):
                base.update(prop_cfg)
            merged["prop_thresholds"][prop] = base
    return merged


def _label_tail_probability(prob: float, medium_cutoff: float, high_cutoff: float) -> str:
    """Bucket a tail probability into Low/Medium/High."""
    if prob >= high_cutoff:
        return "High"
    if prob >= medium_cutoff:
        return "Medium"
    return "Low"


def _empirical_probability(projection: float, prop_type: str, line: float = 0.0) -> Optional[dict]:
    """
    Look up empirical P(over)/P(under) from calibration tables.
    Uses linear interpolation between adjacent bins.

    IMPORTANT: Calibration tables are built against specific lines (e.g., PK=4.5,
    hits=1.5). If the current line doesn't match, calibration is skipped to avoid
    applying P(over line_A) when evaluating against line_B.

    Returns dict with p_over, p_under, n, is_dead_zone or None if no data.
    """
    cal = _load_calibration()
    if prop_type not in cal:
        return None

    # Safety check: only apply calibration when line matches the calibration line.
    # Calibration P(over) was measured against a specific line; using it for a
    # different line would produce incorrect probabilities.
    cal_line = cal[prop_type].get("line", 0)
    if line > 0 and cal_line > 0 and abs(line - cal_line) > 0.01:
        return None  # Line mismatch — fall back to theoretical only

    points = cal[prop_type].get("points", [])
    if not points:
        return None

    # Find the two adjacent bins for interpolation
    # Points are sorted by proj_mid ascending
    proj = projection

    # Below the lowest calibration point
    if proj <= points[0]["proj_mid"]:
        pt = points[0]
        if pt["n"] < 30:
            return None
        p_over = pt["p_over"]
        p_under = pt["p_under"]
        n = pt["n"]
    # Above the highest calibration point
    elif proj >= points[-1]["proj_mid"]:
        pt = points[-1]
        if pt["n"] < 30:
            return None
        p_over = pt["p_over"]
        p_under = pt["p_under"]
        n = pt["n"]
    else:
        # Find bracketing points and interpolate
        lo_pt = None
        hi_pt = None
        for i in range(len(points) - 1):
            if points[i]["proj_mid"] <= proj <= points[i + 1]["proj_mid"]:
                lo_pt = points[i]
                hi_pt = points[i + 1]
                break

        if lo_pt is None or hi_pt is None:
            return None

        # Skip if either bracket has too few observations
        if lo_pt["n"] < 30 or hi_pt["n"] < 30:
            return None

        # Linear interpolation
        span = hi_pt["proj_mid"] - lo_pt["proj_mid"]
        if span <= 0:
            return None
        t = (proj - lo_pt["proj_mid"]) / span
        p_over = lo_pt["p_over"] * (1 - t) + hi_pt["p_over"] * t
        p_under = lo_pt["p_under"] * (1 - t) + hi_pt["p_under"] * t
        n = min(lo_pt["n"], hi_pt["n"])

    # Dead zone: neither direction has meaningful edge (best direction < 54%)
    best_dir = max(p_over, p_under)
    is_dead_zone = best_dir < 0.54

    return {
        "p_over": p_over,
        "p_under": p_under,
        "n": n,
        "is_dead_zone": is_dead_zone,
    }


# ═══════════════════════════════════════════════════════
# LEAGUE AVERAGES (2024 season — update annually)
# ═══════════════════════════════════════════════════════
LG = {
    # Batting
    "avg": 0.248, "obp": 0.312, "slg": 0.399, "iso": 0.151,
    "babip": 0.296, "woba": 0.310, "wrc_plus": 100,
    "k_rate": 22.7, "bb_rate": 8.3,
    "hr_per_pa": 0.033, "sb_per_game": 0.18,
    "rbi_per_game": 0.49, "runs_per_game": 0.49,
    "hits_per_game": 0.93, "tb_per_game": 1.50,
    # Statcast batting
    "exit_velo": 88.5, "hard_hit_pct": 37.0, "barrel_rate": 7.5,
    "sweet_spot_pct": 32.0, "ev90": 105.0,
    "xba": 0.248, "xslg": 0.399, "xwoba": 0.310,
    "sprint_speed": 27.0,  # ft/s
    # Pitching
    "era": 4.17, "fip": 4.12, "xfip": 4.10, "siera": 4.05,
    "whip": 1.28, "k9": 8.58, "bb9": 3.22, "hr9": 1.15,
    "k_pct_p": 22.7, "bb_pct_p": 8.3, "hr_fb_rate": 12.0,
    # Statcast pitching
    "csw_pct": 28.5, "swstr_pct": 11.3,
    "zone_pct": 45.0, "f_strike_pct": 60.0,
    "chase_rate": 28.0,
    # Game context
    "pa_per_lineup_spot": {1: 4.8, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.3,
                           6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8},
    "bf_per_ip": 4.3,
    "avg_ip_starter": 5.5,
}

# ═══════════════════════════════════════════════════════
# STABILIZATION CONSTANTS (PA/BF for 50% regression to mean)
# From FanGraphs / Russell Carleton research
# ═══════════════════════════════════════════════════════
STAB = {
    # Batter (PA)
    "k_rate": 60, "bb_rate": 120, "hbp_rate": 240,
    "babip": 820, "avg": 500, "obp": 300, "slg": 320, "iso": 160,
    "woba": 300, "hr_fb": 300, "gb_rate": 80, "fb_rate": 80,
    "ld_rate": 600, "barrel_rate": 100, "contact_rate": 100,
    "sprint_speed": 50,  # essentially stable (physical trait)
    # Pitcher (BF)
    "k_pct_p": 70, "bb_pct_p": 170, "hr_fb_p": 300,
    "babip_p": 2000, "csw_pct": 200, "swstr_pct": 150,
    "era": 600, "fip": 200, "gb_rate_p": 100,
    "f_strike_pct": 100, "zone_pct": 100,
}


# ═══════════════════════════════════════════════════════
# PARK FACTORS (100 = neutral)
# ═══════════════════════════════════════════════════════
PARK = {  # General runs
    "COL": 116, "ARI": 106, "TEX": 105, "CIN": 104, "BOS": 103,
    "PHI": 102, "ATL": 101, "CHC": 101, "MIL": 101, "LAD": 100,
    "MIN": 100, "NYY": 100, "CLE": 100, "HOU": 100, "BAL": 100,
    "DET": 99, "KC": 99, "STL": 99, "PIT": 98, "CWS": 98,
    "WSH": 98, "TOR": 98, "LAA": 97, "NYM": 97, "SEA": 96,
    "SD": 96, "SF": 95, "TB": 96, "MIA": 95, "OAK": 97,
}
PARK_HR = {  # HR-specific (more extreme)
    "COL": 130, "NYY": 115, "CIN": 112, "TEX": 108, "ARI": 107,
    "PHI": 106, "BAL": 105, "BOS": 103, "CHC": 104, "ATL": 102,
    "MIL": 102, "LAD": 100, "MIN": 100, "HOU": 99, "DET": 98,
    "CLE": 97, "CWS": 97, "OAK": 97, "KC": 95, "STL": 98,
    "PIT": 94, "WSH": 99, "TOR": 98, "LAA": 96, "TB": 96,
    "NYM": 92, "SEA": 93, "SD": 91, "MIA": 90, "SF": 88,
}
PARK_K = {  # K-rate
    "SD": 103, "SF": 102, "SEA": 102, "MIA": 102, "PHI": 101,
    "ARI": 101, "MIL": 101, "LAD": 101, "CLE": 101, "HOU": 101,
    "TOR": 101, "LAA": 101, "NYM": 101, "TB": 101,
    "ATL": 100, "CHC": 100, "DET": 100, "KC": 100, "STL": 100,
    "PIT": 100, "CWS": 100, "WSH": 100, "BAL": 100, "MIN": 100,
    "NYY": 100, "OAK": 100, "TEX": 99, "CIN": 100, "BOS": 98,
    "COL": 96,
}
PARK_SB = {  # Stolen base factors (turf, foul territory, altitude)
    "TB": 105, "TOR": 104, "ARI": 103, "HOU": 102, "MIA": 102,
    "COL": 101, "KC": 101, "MIN": 101, "TEX": 100, "ATL": 100,
    "BAL": 100, "BOS": 100, "CHC": 100, "CIN": 100, "CLE": 100,
    "CWS": 100, "DET": 100, "LAA": 100, "LAD": 100, "MIL": 100,
    "NYM": 100, "NYY": 100, "OAK": 100, "PHI": 100, "PIT": 100,
    "SD": 100, "SF": 99, "SEA": 100, "STL": 100, "WSH": 100,
}

# Also export for backward compatibility
PARK_FACTORS = PARK
PARK_FACTORS_HR = PARK_HR
PARK_FACTORS_K = PARK_K


def _regress(observed, sample_n, stab_n, league_avg):
    """Bayesian regression: blend observed with league average based on sample size."""
    return (sample_n * observed + stab_n * league_avg) / (sample_n + stab_n)


# ═══════════════════════════════════════════════════════
# LOG5 MATCHUP ADJUSTMENT (Bill James)
# ═══════════════════════════════════════════════════════

def log5_rate(pitcher_rate: float, batter_rate: float, league_rate: float) -> float:
    """Log5 matchup adjustment for any binary rate stat.

    Combines pitcher skill, batter skill, and league context to produce
    a matchup-specific rate that's more accurate than either alone.

    Example: Pitcher with 28% K rate vs lineup with 26% K rate (league 22%):
      log5 = (0.28 * 0.26 / 0.22) / (0.28 * 0.26 / 0.22 + 0.72 * 0.74 / 0.78)
           ≈ 0.326 → higher K rate than either individually

    Args:
        pitcher_rate: Pitcher's rate (decimal, e.g. 0.25 for 25% K rate)
        batter_rate: Batter/team rate (decimal, e.g. 0.22 for 22% K rate)
        league_rate: League average rate (decimal, e.g. 0.22)

    Returns:
        Matchup-adjusted rate (decimal)
    """
    # Guard against division by zero or extreme values
    league_rate = max(0.01, min(0.99, league_rate))
    pitcher_rate = max(0.01, min(0.99, pitcher_rate))
    batter_rate = max(0.01, min(0.99, batter_rate))

    num = pitcher_rate * batter_rate / league_rate
    den = num + (1 - pitcher_rate) * (1 - batter_rate) / (1 - league_rate)
    if den <= 0:
        return league_rate
    return num / den


# ═══════════════════════════════════════════════════════
# ABS CHALLENGE SYSTEM ADJUSTMENT (2026 Rule Change)
# ═══════════════════════════════════════════════════════

def abs_adjustment_factor(prop_type: str, game_date: date = None) -> float:
    """Temporary ABS challenge system adjustment for 2026 season.

    The ABS (Automated Ball-Strike) challenge system changes umpire behavior:
    - Umps call a wider zone pre-challenge → fewer Ks, more walks early
    - Effect decays as pitchers/hitters adapt over ~50 games (~7 weeks)

    Args:
        prop_type: Internal stat type (e.g. 'pitcher_strikeouts')
        game_date: Date of the game (defaults to today)

    Returns:
        Multiplier to apply to projection (e.g. 0.98 for -2% K rate)
    """
    if game_date is None:
        game_date = date.today()

    # Check if ABS adjustment is enabled in weights
    weights = _load_weights()
    if not weights.get("abs_adjustment_enabled", True):
        return 1.0

    opening_day = date(2026, 3, 27)
    days_since = (game_date - opening_day).days
    if days_since < 0:
        return 1.0  # Pre-season, no adjustment

    # Decay factor: full effect day 1, zero effect by day 50
    decay = max(0.0, 1.0 - days_since / 50)

    adjustments = {
        "pitcher_strikeouts": 1.0 - 0.02 * decay,   # -2% K rate initially
        "batter_strikeouts": 1.0 - 0.02 * decay,     # -2% K rate initially
        "walks_allowed": 1.0 + 0.015 * decay,         # +1.5% BB rate initially
        "walks": 1.0 + 0.015 * decay,                 # +1.5% BB rate for batters too
        "pitching_outs": 1.0 - 0.01 * decay,          # Slightly fewer outs (more walks)
    }
    return adjustments.get(prop_type, 1.0)


def _park(team, factors_dict):
    """Get park factor as multiplier (1.0 = neutral)."""
    return factors_dict.get(team, 100) / 100.0


def _lineup_pa(order_pos):
    """Expected PA based on lineup position (1-9)."""
    return LG["pa_per_lineup_spot"].get(order_pos, 4.2)


def estimate_plate_appearances(
    lineup_pos: int = None,
    team_runs_per_game: float = None,
    blowout_risk: float = 0.0,
    pinch_hit_risk: float = 0.0,
) -> dict:
    """Opportunity-first PA estimation for hitters.

    Models PA distribution (not just mean) based on:
    - Lineup position (1-9): top of order gets ~1 more PA than bottom
    - Team run environment: high-scoring teams cycle lineup more
    - Blowout risk: lopsided games → starters pulled early
    - Pinch-hit risk: platoon players / late-inning substitution

    Returns dict with mean_pa, std_pa, and expected_ab.

    Source: Research Reports 3 & 4 — opportunity-first architecture
    """
    # Base PA from lineup slot
    base_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2

    # Team run environment adjustment
    # High-scoring teams turn over the lineup more → more PA at top
    if team_runs_per_game and team_runs_per_game > 0:
        lg_rpg = 4.5  # league average runs per game
        run_env_adj = (team_runs_per_game / lg_rpg - 1) * 0.12
        base_pa *= (1 + run_env_adj)

    # Blowout risk reduces PA (starters pulled in lopsided games)
    if blowout_risk > 0:
        base_pa *= (1 - blowout_risk * 0.15)

    # Pinch-hit risk (platoon players may not get full game)
    if pinch_hit_risk > 0:
        base_pa *= (1 - pinch_hit_risk)

    # PA standard deviation: ~0.8-1.0 for most lineup spots
    # Top of order has slightly higher variance (more extras/walkoff PAs)
    if lineup_pos and lineup_pos <= 3:
        std_pa = 1.0
    elif lineup_pos and lineup_pos >= 7:
        std_pa = 0.75
    else:
        std_pa = 0.85

    # Expected AB: PA minus walks/HBP (~12% of PA for average hitter)
    exp_ab = base_pa * 0.88

    return {
        "mean_pa": round(base_pa, 2),
        "std_pa": round(std_pa, 2),
        "expected_ab": round(exp_ab, 2),
        "lineup_pos": lineup_pos,
    }


RUNS_LINEUP_MULTIPLIERS = {
    1: 1.18,
    2: 1.12,
    3: 1.04,
    4: 1.00,
    5: 0.96,
    6: 0.91,
    7: 0.85,
    8: 0.79,
    9: 0.75,
}

RBI_LINEUP_MULTIPLIERS = {
    1: 0.72,
    2: 0.86,
    3: 1.16,
    4: 1.24,
    5: 1.14,
    6: 0.99,
    7: 0.86,
    8: 0.75,
    9: 0.69,
}


def _estimate_batter_games(b: dict) -> float:
    """Estimate games played from the profile, preferring explicit season totals."""
    games = float(b.get("g", 0) or 0)
    if games > 0:
        return games
    pa = float(b.get("pa", 0) or 0)
    if pa > 0:
        return max(pa / 4.2, 1.0)
    return 1.0


def _estimate_team_runs_per_game(
    b: dict,
    opp_p: Optional[dict] = None,
    park: Optional[str] = None,
    wx: Optional[dict] = None,
    platoon: Optional[dict] = None,
    team_lineup_context: Optional[dict] = None,
) -> float:
    """Estimate the batter's team run environment for PA-sensitive props."""
    team_rpg = 4.5

    woba = b.get("woba", LG["woba"])
    xwoba = b.get("xwoba", 0)
    wrc_plus = b.get("wrc_plus", 0)

    if woba > 0:
        team_rpg *= 1 + (woba / LG["woba"] - 1) * 0.10
    if xwoba and xwoba > 0:
        team_rpg *= 1 + (xwoba / LG["xwoba"] - 1) * 0.06
    if wrc_plus and wrc_plus > 0:
        team_rpg *= 1 + (wrc_plus / 100.0 - 1) * 0.14

    if opp_p:
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_whip = opp_p.get("whip", LG["whip"])
        opp_quality = (opp_fip / LG["fip"] * 0.65) + (opp_whip / LG["whip"] * 0.35)
        team_rpg *= 1 + (opp_quality - 1) * 0.65

    if platoon and platoon.get("adjustment"):
        team_rpg *= 1 + (platoon["adjustment"] - 1) * 0.20

    if park:
        team_rpg *= 1 + (_park(park, PARK) - 1) * 0.85

    if wx:
        weather_mult = get_stat_specific_weather_adjustment(wx, "runs")
        team_rpg *= 1 + (weather_mult - 1) * 0.75

    if team_lineup_context and team_lineup_context.get("has_data"):
        lineup_woba = team_lineup_context.get("avg_woba", LG["woba"])
        top5_woba = team_lineup_context.get("top5_woba", lineup_woba)
        depth_woba = team_lineup_context.get("lineup_depth_woba", lineup_woba)
        team_rpg *= 1 + (lineup_woba / LG["woba"] - 1) * 0.16
        team_rpg *= 1 + (top5_woba / LG["woba"] - 1) * 0.12
        team_rpg *= 1 + (depth_woba / LG["woba"] - 1) * 0.08

    return max(2.8, min(team_rpg, 6.8))


def _context_multiplier(value: float, baseline: float, weight: float,
                        low: float = 0.8, high: float = 1.25) -> float:
    """Translate a context metric into a bounded multiplicative adjustment."""
    if not value or not baseline:
        return 1.0
    mult = 1 + (value / baseline - 1) * weight
    return max(low, min(mult, high))


def estimate_batters_faced(
    pitcher_ip: float = 5.5,
    pitcher_whip: float = None,
    early_season_discount: float = 1.0,
    opposing_lineup_woba: float = None,
) -> dict:
    """Opportunity-first BF estimation for pitchers.

    Models BF distribution based on:
    - Expected IP (from pitcher projections + early-season discount)
    - WHIP (more baserunners = more BF per inning)
    - Opposing lineup quality (better lineups → more BF)

    Returns dict with mean_bf, std_bf.

    Source: Research Reports 3 & 4 — opportunity-first architecture
    """
    # Base BF from expected IP
    bf_per_ip = LG["bf_per_ip"]  # ~4.3 league average

    # WHIP adjustment: higher WHIP → more BF per IP
    if pitcher_whip and pitcher_whip > 0:
        whip_adj = pitcher_whip / LG["whip"]
        bf_per_ip *= (0.7 + 0.3 * whip_adj)  # Partial adjustment

    # Opposing lineup quality
    if opposing_lineup_woba and opposing_lineup_woba > 0:
        lineup_adj = opposing_lineup_woba / LG["woba"]
        bf_per_ip *= (0.85 + 0.15 * lineup_adj)

    effective_ip = pitcher_ip * early_season_discount
    mean_bf = effective_ip * bf_per_ip

    # BF standard deviation: ~3-5 for starters (wide range of outcomes)
    std_bf = max(2.5, mean_bf * 0.18)

    return {
        "mean_bf": round(mean_bf, 1),
        "std_bf": round(std_bf, 1),
        "effective_ip": round(effective_ip, 2),
        "bf_per_ip": round(bf_per_ip, 2),
    }


def _ensure_pct(val, lg_default=None):
    """Convert decimal rates (0.227) to percentages (22.7).

    FanGraphs CSVs store K%/BB% as decimals (0.227) but the predictor
    math expects percentages (22.7).  Values already in percentage form
    (>= 1.0) are returned unchanged.  A league-average *lg_default* is
    returned when *val* is falsy (0, None, etc.).
    """
    if not val and lg_default is not None:
        return lg_default
    if val is not None and val < 1.0:
        return val * 100
    return val


# ═══════════════════════════════════════════════════════
# PITCHER PROJECTIONS
# ═══════════════════════════════════════════════════════


def tto_k_rate_decay(base_k_rate: float, expected_bf: float) -> float:
    """Apply Times Through Order penalty to K rate.

    Research: K% drops by ~3% absolute on the third time through the order.
    This translates to roughly -0.15 K% per batter faced beyond 18 BF
    (the point where TTO3 begins for most lineups).

    The effect is significant — a 28% K rate pitcher drops to ~25% on TTO3.
    Not accounting for this causes systematic over-projection of Ks for
    pitchers expected to go deep into games.

    Args:
        base_k_rate: Matchup-adjusted K rate (decimal, e.g. 0.25)
        expected_bf: Expected batters faced

    Returns:
        Adjusted K rate accounting for TTO decay
    """
    if expected_bf <= 18:
        return base_k_rate  # No TTO3 penalty if not reaching 3rd time

    # BF beyond 18 (start of TTO3)
    tto3_bf = expected_bf - 18

    # Each TTO3 batter has ~3% lower K rate than base
    # Weight by fraction of total BF in TTO3
    tto3_fraction = tto3_bf / expected_bf
    k_rate_drop = 0.03 * tto3_fraction  # Blended drop across all BF

    return max(base_k_rate * 0.70, base_k_rate - k_rate_drop)  # Floor at 70% of base


def _early_season_ip_discount(game_date: date = None) -> float:
    """
    Early-season workload discount for pitcher IP projections.

    Teams limit starter pitch counts in the first few weeks of the season,
    typically capping at 75-85 pitches on Opening Day and ramping up over
    the first ~5 weeks. This creates a systematic bias where full-season
    IP/start projections overshoot actual early-season outings.

    Research-based ramp-up schedule (approximate):
      Opening Day (week 0):   ~80 pitches → ~4.5-5.0 IP → discount 15%
      Week 1-2:               ~85-90 pitches → ~5.0-5.5 IP → discount 10%
      Week 3-4:               ~90-95 pitches → ~5.5-6.0 IP → discount 5%
      Week 5+:                Full workload → no discount

    Returns a multiplier (0.85 to 1.0) to apply to projected IP/outs.
    """
    if game_date is None:
        game_date = date.today()

    # MLB Opening Day is typically late March / early April.
    # Use April 1 as a rough anchor — actual opening day varies by year.
    year = game_date.year
    # Opening Day 2026: March 27. Adjust for other years as needed.
    season_start = date(year, 3, 27)

    if game_date < season_start:
        # Pre-Opening Day: moderate discount (less aggressive, spring games still happen)
        return 0.90

    days_into_season = (game_date - season_start).days

    if days_into_season <= 7:
        # Opening week: ~15% discount
        return 0.85
    elif days_into_season <= 14:
        # Week 2: ~10% discount
        return 0.90
    elif days_into_season <= 21:
        # Week 3: ~7% discount
        return 0.93
    elif days_into_season <= 35:
        # Week 4-5: ~4% discount, almost ramped up
        return 0.96
    else:
        # Week 6+: full workload
        return 1.0


def project_pitcher_strikeouts(p, bvp=None, platoon=None, ump=None,
                                opp_k_rate=None, park=None, wx=None,
                                expected_ip=None, opp_lineup_context=None):
    """
    PITCHER STRIKEOUTS — strongest signal prop type.

    Primary inputs (weighted by research):
      K%/K9 (30%) — season rate, regressed
      CSW% + SwStr% (20%) — Statcast pitch quality
      Opposing lineup K% (20%) — who they're facing matters enormously
      Umpire tendency (10%) — high-K umps add 0.5-1.0 K per pitcher
      BvP aggregate K rate (10%) — historical matchup
      Park K factor (5%) — slight effect
      Weather (5%) — cold = grip issues = fewer Ks
    """
    # K% — accept k_pct (already percentage), k_rate (decimal 0-1), or k9 (per 9 IP)
    _raw_k = p.get("k_pct", 0)
    if not _raw_k and p.get("k_rate", 0):
        # k_rate is decimal (e.g. 0.30 = 30%) — convert to percentage
        _kr = p["k_rate"]
        _raw_k = _kr * 100 if _kr < 1 else _kr
    k_pct = _ensure_pct(_raw_k, lg_default=None) or (p.get("k9", p.get("k_per_9", LG["k9"])) / (LG["bf_per_ip"] * 9) * 100)

    # BF for regression sample — use season total IP, or estimate from per-start × games
    _season_ip = p.get("ip", 0)
    if not _season_ip and p.get("ip_per_start", 0):
        _season_ip = p["ip_per_start"] * max(p.get("gs", 10), 1)
    bf_est = _season_ip * LG["bf_per_ip"] if _season_ip > 0 else 0
    csw = p.get("recent_csw_pct", 0)
    swstr = p.get("recent_swstr_pct", 0)

    # Regress K%
    reg_k = _regress(k_pct, bf_est, STAB["k_pct_p"], LG["k_pct_p"])

    # CSW quality adjustment (r=0.87 with K%)
    if csw > 0:
        csw_delta = (csw - LG["csw_pct"]) * 1.2  # +1 CSW% ≈ +1.2 K%
        reg_k = reg_k * 0.70 + (reg_k + csw_delta) * 0.30
    if swstr > 0:
        swstr_delta = (swstr - LG["swstr_pct"]) * 1.5
        reg_k = reg_k * 0.85 + (reg_k + swstr_delta) * 0.15

    lineup_k_rate = None
    lineup_woba = None
    if opp_lineup_context and opp_lineup_context.get("has_data"):
        lineup_k_rate = opp_lineup_context.get("top6_k_rate") or opp_lineup_context.get("avg_k_rate")
        lineup_woba = opp_lineup_context.get("top5_woba") or opp_lineup_context.get("avg_woba")
        if lineup_k_rate:
            if opp_k_rate and opp_k_rate > 0:
                opp_k_rate = opp_k_rate * 0.55 + float(lineup_k_rate) * 0.45
            else:
                opp_k_rate = float(lineup_k_rate)

    # v018: Log5 matchup adjustment for opposing lineup K% (replaces simple ratio)
    # Team K rates range 19%-27% — huge impact on pitcher K projections
    if opp_k_rate and opp_k_rate > 0:
        pitcher_k_dec = reg_k / 100.0
        opp_k_dec = opp_k_rate / 100.0 if opp_k_rate > 1 else opp_k_rate
        lg_k_dec = LG["k_rate"] / 100.0
        matchup_k = log5_rate(pitcher_k_dec, opp_k_dec, lg_k_dec)
        reg_k = matchup_k * 100.0

    # BvP aggregate K adjustment
    if bvp and bvp.get("has_data") and bvp.get("total_pa", 0) >= 10:
        bvp_k = bvp.get("agg_k_rate", LG["k_rate"])
        bvp_adj = bvp_k / LG["k_rate"]
        reg_k *= (1 + (bvp_adj - 1) * 0.25)  # 25% weight on BvP

    # v018 Task 3A: Opportunity-first BF estimation
    # Use centralized estimate_batters_faced() for consistent pitcher opportunity modeling
    if expected_ip is None:
        ip = p.get("ip", 0)
        # Prefer ip_per_start if available (direct per-game average)
        if p.get("ip_per_start", 0) > 0:
            expected_ip = p["ip_per_start"]
        elif ip > 10:
            starts = max(p.get("gs", ip / 5.5), 1)
            expected_ip = min(ip / starts, 7.0)
        else:
            expected_ip = LG["avg_ip_starter"]
        expected_ip = max(4.5, min(7.0, expected_ip))  # Allow up to 7.0 for aces
    bf_est_result = estimate_batters_faced(
        pitcher_ip=expected_ip,
        pitcher_whip=p.get("whip"),
        early_season_discount=_early_season_ip_discount(),
        opposing_lineup_woba=lineup_woba,
    )
    exp_bf = bf_est_result["mean_bf"]
    expected_ip = bf_est_result["effective_ip"]

    # v018 Task 3A: TTO penalty — K rate drops ~3% on third time through order
    k_rate_tto = tto_k_rate_decay(reg_k / 100.0, exp_bf)

    # Raw projection (using TTO-adjusted rate)
    proj = exp_bf * k_rate_tto

    # Park K factor
    if park: proj *= _park(park, PARK_K)

    # Umpire adjustment (+/- 0.5-1.0 K)
    if ump and ump.get("known"):
        proj += ump.get("k_adjustment", 0)

    # Weather
    if wx:
        proj *= get_stat_specific_weather_adjustment(wx, "pitcher_strikeouts")

    # Platoon (if entire lineup skews one hand)
    if platoon and platoon.get("k_adjustment"):
        proj *= platoon["k_adjustment"]

    # v018: Removed proportional dampening. Instead apply hard cap at 12 Ks to prevent
    # outliers while allowing elite pitchers to project more accurately.
    proj = min(proj, 12.0)

    # v018: ABS Challenge System adjustment (2026 rule change)
    proj *= abs_adjustment_factor("pitcher_strikeouts")

    mu = max(proj, 0.5)

    # Beta-Binomial distribution parameters for strikeout projections
    # Use stabilized K-rate and precision based on pitcher reliability
    raw_k_rate = reg_k / 100.0  # Convert percentage to decimal
    sample_bf = p.get("gs", 10) * 25  # Approximation: ~25 BF/start
    stabilized_k_rate = distributions.bayesian_stabilize(
        raw_k_rate, LG["k_pct_p"] / 100.0, sample_bf, "pitcher_k_rate"
    )
    precision = distributions.pitcher_k_precision(p.get("gs", 10), k_rate_stability=0.56)
    alpha, beta = distributions.betabinom_params(stabilized_k_rate, precision=precision)

    bb_mean, bb_var = distributions.betabinom_mean_var(int(exp_bf), alpha, beta)

    result = {
        "projection": round(mu, 2), "mu": mu, "regressed_k_pct": round(reg_k, 1),
        "expected_ip": round(expected_ip, 1), "expected_bf": round(exp_bf, 1),
        "opp_lineup_k_rate": round(opp_k_rate, 2) if opp_k_rate else None,
        "opp_lineup_woba": round(lineup_woba, 3) if lineup_woba else None,
        # Beta-Binomial distribution info
        "bb_alpha": round(alpha, 3), "bb_beta": round(beta, 3),
        "bb_mean": round(bb_mean, 2), "bb_variance": round(bb_var, 2),
    }
    return result


def project_pitcher_outs(p, park=None, wx=None):
    """
    PITCHER OUTS RECORDED — sensitive to pitch count and bullpen usage.
    Outs = IP × 3. A starter going 6 IP = 18 outs.

    Key inputs: historical IP/start, pitch efficiency (pitches/PA),
    BB rate (walks extend innings), game script tendency.
    """
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bb_pct = _ensure_pct(p.get("bb_pct"), lg_default=LG["bb_pct_p"])
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    # Regress BB% (high BB = shorter outings)
    reg_bb = _regress(bb_pct, bf_est, STAB["bb_pct_p"], LG["bb_pct_p"])

    # Average IP per start
    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(6.5, avg_ip))  # BUGFIX: 8.0 was too high, causing projections like 25+ outs

    # BB% adjustment: high walk rate = fewer outs (more pitches burned)
    bb_adj = 1.0 - (reg_bb - LG["bb_pct_p"]) / LG["bb_pct_p"] * 0.15

    proj_ip = avg_ip * bb_adj
    # v018 Task 3A: Use centralized BF estimation for consistent opportunity modeling
    bf_result = estimate_batters_faced(
        pitcher_ip=proj_ip,
        pitcher_whip=p.get("whip"),
        early_season_discount=_early_season_ip_discount(),
    )
    proj_outs = bf_result["effective_ip"] * 3

    if park: proj_outs *= (1 + (_park(park, PARK) - 1) * -0.1)  # Hitter parks = fewer outs
    if wx:
        proj_outs *= get_stat_specific_weather_adjustment(wx, "pitching_outs")

    # v018: ABS Challenge System adjustment (more walks → fewer outs)
    proj_outs *= abs_adjustment_factor("pitching_outs")

    mu = max(proj_outs, 9.0)
    return {"projection": round(mu, 1), "mu": mu, "avg_ip_start": round(avg_ip, 1),
            "regressed_bb_pct": round(reg_bb, 1)}


def project_pitcher_earned_runs(p, park=None, wx=None, opp_woba=None):
    """
    PITCHER EARNED RUNS — use FIP/xFIP over ERA (more predictive).

    Key: xFIP (fielding-independent, normalized HR/FB) is the best
    predictor of future run prevention. ERA is noisy.
    """
    era = p.get("era", LG["era"])
    fip = p.get("fip", era)
    xfip = p.get("xfip", fip)
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    # Use FIP/xFIP blend, regressed
    # xFIP is most stable, ERA is noisiest
    pitching_rate = era * 0.15 + fip * 0.35 + xfip * 0.50
    reg_rate = _regress(pitching_rate, bf_est, STAB["fip"], LG["fip"])

    # v018 Task 3A: Use centralized BF estimation
    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(6.5, avg_ip))
    bf_result = estimate_batters_faced(
        pitcher_ip=avg_ip,
        pitcher_whip=p.get("whip"),
        early_season_discount=_early_season_ip_discount(),
        opposing_lineup_woba=opp_woba,
    )
    effective_ip = bf_result["effective_ip"]

    # ER projection = (rate / 9) * expected IP
    proj_er = (reg_rate / 9.0) * effective_ip

    # Opposing lineup quality
    if opp_woba and opp_woba > 0:
        opp_adj = opp_woba / LG["woba"]
        proj_er *= (1 + (opp_adj - 1) * 0.5)

    # Park factor (hitter parks = more ER)
    if park: proj_er *= _park(park, PARK)

    # Weather (warm = more offense = more ER)
    if wx:
        proj_er *= get_stat_specific_weather_adjustment(wx, "earned_runs")

    mu = max(proj_er, 0.5)

    # Negative Binomial distribution parameters for earned runs
    # MLB ER overdispersion ratio ~2.0-2.5
    nb_n, nb_p = distributions.negbinom_params(mu, overdispersion=2.2)

    return {"projection": round(mu, 2), "mu": mu, "blended_rate": round(reg_rate, 2),
            "avg_ip": round(avg_ip, 1),
            "nb_n": round(nb_n, 4), "nb_p": round(nb_p, 4)}


def project_pitcher_walks(p, park=None, ump=None):
    """PITCHER WALKS ALLOWED — BB% is the key driver."""
    bb_pct = _ensure_pct(p.get("bb_pct"), lg_default=LG["bb_pct_p"])
    bb9 = p.get("bb9", LG["bb9"])
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    reg_bb = _regress(bb_pct, bf_est, STAB["bb_pct_p"], LG["bb_pct_p"])

    # Zone% and F-Strike% adjustments (pitchers who throw strikes walk fewer)
    zone = p.get("zone_pct", 0)
    if zone > 0:
        zone_adj = (zone - LG["zone_pct"]) / LG["zone_pct"] * -0.2
        reg_bb *= (1 + zone_adj)

    # v018 Task 3A: Use centralized BF estimation
    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(6.5, avg_ip))
    bf_result = estimate_batters_faced(
        pitcher_ip=avg_ip,
        pitcher_whip=p.get("whip"),
        early_season_discount=_early_season_ip_discount(),
    )
    exp_bf = bf_result["mean_bf"]

    proj = exp_bf * (reg_bb / 100)

    # Umpire with tight zone = more walks
    if ump and ump.get("known"):
        k_adj = ump.get("k_adjustment", 0)
        proj -= k_adj * 0.3  # Inverse: high-K ump = fewer walks

    # v018: ABS Challenge System adjustment
    proj *= abs_adjustment_factor("walks_allowed")

    mu = max(proj, 0.5)
    return {"projection": round(mu, 2), "mu": mu, "regressed_bb_pct": round(reg_bb, 1)}


def project_pitcher_hits_allowed(p, park=None, wx=None, opp_avg=None):
    """PITCHER HITS ALLOWED — WHIP, BABIP, K rate drive this."""
    whip = p.get("whip", LG["whip"])
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0
    bb9 = p.get("bb9", LG["bb9"])

    # Hits per 9 = (WHIP - BB/9) * 9... approximate
    h9 = (whip * 9) - (bb9 if bb9 > 0 else LG["bb9"])
    h9 = max(h9, 5.0)

    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(6.5, avg_ip))  # BUGFIX: 7.5 was too high for hits_allowed projections

    proj = (h9 / 9.0) * avg_ip

    if opp_avg and opp_avg > 0:
        proj *= (opp_avg / LG["avg"])

    if park: proj *= _park(park, PARK)
    if wx:
        proj *= get_stat_specific_weather_adjustment(wx, "hits_allowed")

    mu = max(proj, 2.0)
    return {"projection": round(mu, 2), "mu": mu}


# ═══════════════════════════════════════════════════════
# BATTER PROJECTIONS
# ═══════════════════════════════════════════════════════

def project_batter_hits(b, opp_p=None, bvp=None, platoon=None,
                         park=None, wx=None, lineup_pos=None):
    """
    BATTER HITS — most predictable batting prop.

    Key inputs:
      AVG/xBA (25%) — season rate + Statcast expected, regressed
      K% (20%) — low K = more balls in play = more hit opportunities
      Contact quality (15%) — hard hit%, barrel rate, EV90
      BvP matchup (15%) — head-to-head history
      Opposing pitcher (10%) — WHIP, K%, quality
      Platoon (8%) — handedness advantage
      Park + Weather (7%) — slight effects
    """
    avg = b.get("avg", LG["avg"])
    xba = b.get("xba", 0)
    pa = b.get("pa", 0)
    k_rate = _ensure_pct(b.get("k_rate"), lg_default=LG["k_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    ev90 = b.get("recent_ev90", LG["ev90"])
    babip = b.get("babip", LG["babip"])

    # Regress AVG — v015: use reduced stabilization for batters with 200+ PA
    # (more PA = more signal, less regression needed)
    stab_avg = STAB["avg"]
    if pa >= 200:
        stab_avg = int(stab_avg * 0.75)  # 375 instead of 500 for established batters
    reg_avg = _regress(avg, pa, stab_avg, LG["avg"])

    # xBA blend — v018: increased from 40% to 50% weight. xBA is a better predictor
    # of future hitting than raw AVG because it strips out BABIP luck. Backtest shows
    # model under-projects hits by 0.18 consistently; stronger Statcast weight helps.
    if xba > 0:
        reg_avg = reg_avg * 0.50 + xba * 0.50

    # K% adjustment: low K% = more balls in play
    reg_k = _regress(k_rate, pa, STAB["k_rate"], LG["k_rate"])
    k_adj = 1.0 + (LG["k_rate"] - reg_k) / LG["k_rate"] * 0.25  # v018: 0.15 → 0.25
    reg_avg *= k_adj

    # Contact quality: hard hit rate, EV90 — v018: increased from 0.14 to 0.22
    if hard_hit > 0:
        hh_adj = (hard_hit - LG["hard_hit_pct"]) / LG["hard_hit_pct"] * 0.22
        reg_avg *= (1 + hh_adj)

    # BABIP regression check (if BABIP is way above/below xBA, regression coming)
    if babip > 0 and xba > 0:
        babip_delta = babip - (xba + 0.050)  # BABIP is typically ~50 pts above xBA
        if abs(babip_delta) > 0.030:
            reg_avg *= (1 - babip_delta * 0.15)  # Pull toward expected

    # BvP matchup
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_avg = bvp.get("avg", LG["avg"])
        bvp_weight = min(bvp["pa"] / 50, 0.30)  # Max 30% weight at 50+ PA
        reg_avg = reg_avg * (1 - bvp_weight) + bvp_avg * bvp_weight

    # v018: Log5 matchup adjustment for opposing pitcher
    # Uses pitcher's hit-allowed rate vs batter AVG with league context.
    # IMPORTANT: Both Log5 inputs must be rates in the SAME direction (hit rates).
    if opp_p:
        opp_whip = opp_p.get("whip", LG["whip"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_bb9 = opp_p.get("bb9", LG["bb9"])
        # Derive pitcher's hit-allowed rate from WHIP:
        #   H/IP = WHIP - BB/IP;  h_rate_per_BF = H/IP / BF_per_IP
        # Average pitcher: WHIP 1.28, BB9 3.22 → H/IP=0.92, h_rate=0.92/4.3≈.214
        opp_h_per_ip = max(0.5, opp_whip - opp_bb9 / 9.0)
        opp_h_rate = min(0.35, max(0.15, opp_h_per_ip / LG["bf_per_ip"]))
        matchup_avg = log5_rate(
            opp_h_rate,   # pitcher hit-ALLOWED rate (same scale as batter AVG)
            reg_avg,       # batter hit rate
            LG["avg"]      # league AVG
        )
        # Blend: 60% Log5, 40% old quality method for stability
        opp_quality = (opp_whip / LG["whip"] * 0.6 + opp_fip / LG["fip"] * 0.4)
        old_adj = reg_avg * (1 + (opp_quality - 1) * 0.35)
        reg_avg = matchup_avg * 0.6 + old_adj * 0.4

    # Platoon
    if platoon and platoon.get("adjustment"):
        reg_avg *= platoon["adjustment"]

    # Park + weather
    if park: reg_avg *= (1 + (_park(park, PARK) - 1) * 0.25)
    if wx:
        reg_avg *= get_stat_specific_weather_adjustment(wx, "hits")

    # v018 Task 3A: Opportunity-first PA estimation
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    # Use batter's actual BB rate for AB conversion (not league average 12%)
    bb_rate_pct = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    exp_ab = exp_pa * (1 - bb_rate_pct / 100)

    mu = max(exp_ab * reg_avg, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_avg": round(reg_avg, 3),
            "expected_pa": round(exp_pa, 1), "expected_ab": round(exp_ab, 1)}


def project_batter_singles(b, opp_p=None, bvp=None, platoon=None,
                           park=None, wx=None, lineup_pos=None):
    """BATTER SINGLES — contact/batted-ball prop, not just a hits proxy."""
    avg = b.get("avg", LG["avg"])
    xba = b.get("xba", 0)
    xslg = b.get("xslg", 0)
    pa = b.get("pa", 0)
    bb_rate_pct = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    k_rate = _ensure_pct(b.get("k_rate"), lg_default=LG["k_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    babip = b.get("babip", LG["babip"])
    ab = b.get("ab", 0) or (pa * (1 - bb_rate_pct / 100) if pa > 0 else 0)
    hits = b.get("h", 0) or (avg * ab if pa > 0 else 0)
    hr = b.get("hr", 0)
    doubles = b.get("2b", 0)
    triples = b.get("3b", 0)

    singles = max(hits - hr - doubles - triples, 0)
    single_rate = singles / pa if pa > 0 else 0.135
    reg_single = _regress(single_rate, pa, STAB["avg"], 0.135)

    if xba > 0 and xslg > 0:
        x_iso = max(xslg - xba, 0.0)
        x_hr_per_pa = max(x_iso * 0.22, 0.0)
        x_trp_per_pa = 0.004
        x_dbl_per_pa = max((xslg - xba - 3 * x_hr_per_pa) / 2, 0.015)
        x_single_per_pa = max(xba - x_hr_per_pa - x_dbl_per_pa - x_trp_per_pa, 0.05)
        reg_single = reg_single * 0.65 + x_single_per_pa * 0.35

    reg_k = _regress(k_rate, pa, STAB["k_rate"], LG["k_rate"])
    reg_single *= 1 + (LG["k_rate"] - reg_k) / LG["k_rate"] * 0.18

    if hard_hit > 0:
        reg_single *= 1 + (hard_hit - LG["hard_hit_pct"]) / LG["hard_hit_pct"] * 0.08

    if babip > 0 and xba > 0:
        babip_delta = babip - (xba + 0.050)
        if abs(babip_delta) > 0.030:
            reg_single *= (1 - babip_delta * 0.12)

    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_avg = bvp.get("avg", LG["avg"])
        bvp_weight = min(bvp["pa"] / 60, 0.20)
        reg_single = reg_single * (1 - bvp_weight) + bvp_avg * bvp_weight

    if opp_p:
        opp_whip = opp_p.get("whip", LG["whip"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_quality = (opp_whip / LG["whip"] * 0.65 + opp_fip / LG["fip"] * 0.35)
        reg_single *= 1 + (opp_quality - 1) * 0.22

    if platoon and platoon.get("adjustment"):
        reg_single *= 1 + (platoon["adjustment"] - 1) * 0.45

    if park:
        reg_single *= 1 + (_park(park, PARK) - 1) * 0.20
    if wx:
        reg_single *= get_stat_specific_weather_adjustment(wx, "hits")

    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    exp_ab = exp_pa * (1 - bb_rate_pct / 100)

    mu = max(exp_ab * reg_single, 0.05)
    return {
        "projection": round(mu, 2),
        "mu": mu,
        "regressed_single_rate": round(reg_single, 3),
        "expected_pa": round(exp_pa, 1),
        "expected_ab": round(exp_ab, 1),
    }


def project_batter_doubles(b, opp_p=None, bvp=None, platoon=None,
                           park=None, wx=None, lineup_pos=None):
    """BATTER DOUBLES — extra-base contact, modeled directly from doubles rate."""
    xba = b.get("xba", 0)
    xslg = b.get("xslg", 0)
    pa = b.get("pa", 0)
    doubles = b.get("2b", 0)
    bb_rate_pct = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    barrel = b.get("recent_barrel_rate", LG["barrel_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    ev90 = b.get("recent_ev90", LG["ev90"])

    double_rate = doubles / pa if pa > 0 else 0.045
    reg_dbl = _regress(double_rate, pa, 220, 0.045)

    if xba > 0 and xslg > 0:
        x_iso = max(xslg - xba, 0.0)
        x_hr_per_pa = max(x_iso * 0.22, 0.0)
        x_dbl_per_pa = max((xslg - xba - 3 * x_hr_per_pa) / 2, 0.015)
        reg_dbl = reg_dbl * 0.60 + x_dbl_per_pa * 0.40

    if barrel > 0:
        reg_dbl *= 1 + (barrel - LG["barrel_rate"]) / LG["barrel_rate"] * 0.12
    if hard_hit > 0:
        reg_dbl *= 1 + (hard_hit - LG["hard_hit_pct"]) / LG["hard_hit_pct"] * 0.10
    if ev90 > 0:
        reg_dbl *= 1 + (ev90 - LG["ev90"]) / LG["ev90"] * 0.06

    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_slg = bvp.get("slg", LG["slg"])
        reg_dbl *= 1 + (bvp_slg / LG["slg"] - 1) * min(bvp["pa"] / 80, 0.14)

    if opp_p:
        opp_whip = opp_p.get("whip", LG["whip"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_quality = (opp_whip / LG["whip"] * 0.55 + opp_fip / LG["fip"] * 0.45)
        reg_dbl *= 1 + (opp_quality - 1) * 0.18

    if platoon and platoon.get("adjustment"):
        reg_dbl *= 1 + (platoon["adjustment"] - 1) * 0.35

    if park:
        reg_dbl *= 1 + (_park(park, PARK) - 1) * 0.15
    if wx:
        reg_dbl *= 1 + (get_stat_specific_weather_adjustment(wx, "total_bases") - 1) * 0.35

    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    exp_ab = exp_pa * (1 - bb_rate_pct / 100)

    mu = max(exp_ab * reg_dbl, 0.01)
    return {
        "projection": round(mu, 2),
        "mu": mu,
        "regressed_double_rate": round(reg_dbl, 3),
        "expected_pa": round(exp_pa, 1),
        "expected_ab": round(exp_ab, 1),
    }


def project_batter_total_bases(b, opp_p=None, bvp=None, platoon=None,
                                 park=None, wx=None, lineup_pos=None):
    """
    BATTER TOTAL BASES — SLG-driven with power metrics.

    Key: Barrel rate is the single best Statcast predictor for TB.
    85.8% of HRs are barrels, 50.2% of barrels become HRs.
    """
    slg = b.get("slg", LG["slg"])
    xslg = b.get("xslg", 0)
    iso = b.get("iso", LG["iso"])
    pa = b.get("pa", 0)
    barrel = b.get("recent_barrel_rate", LG["barrel_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    ev90 = b.get("recent_ev90", LG["ev90"])

    # Regress SLG — v015: reduced stabilization for established batters
    stab_slg = STAB["slg"]
    if pa >= 200:
        stab_slg = int(stab_slg * 0.75)  # 240 instead of 320
    reg_slg = _regress(slg, pa, stab_slg, LG["slg"])

    # xSLG blend — v015: increased from 30% to 42%. xSLG captures true power
    # better than SLG (strips BABIP luck on singles). Model under-projects TB by
    # 0.32 consistently; stronger Statcast signal improves discrimination.
    if xslg > 0:
        reg_slg = reg_slg * 0.58 + xslg * 0.42

    # Barrel rate (strongest power predictor) — v015: increased from 0.18 to 0.22
    if barrel > 0:
        barrel_adj = (barrel - LG["barrel_rate"]) / LG["barrel_rate"] * 0.22
        reg_slg *= (1 + barrel_adj)

    # EV90 (90th percentile exit velo — more stable than max EV) — v015: 0.08 → 0.11
    if ev90 > 0:
        ev_adj = (ev90 - LG["ev90"]) / LG["ev90"] * 0.11
        reg_slg *= (1 + ev_adj)

    # ISO regression: if ISO is way above xSLG-xBA, regression likely
    if iso > 0 and xslg > 0:
        x_iso = xslg - (b.get("xba", LG["xba"]))
        if x_iso > 0:
            iso_delta = iso - x_iso
            if abs(iso_delta) > 0.030:
                reg_slg *= (1 - iso_delta * 0.10)

    # BvP matchup
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_slg = bvp.get("slg", LG["slg"])
        bvp_w = min(bvp["pa"] / 50, 0.25)
        reg_slg = reg_slg * (1 - bvp_w) + bvp_slg * bvp_w

    # Opposing pitcher
    if opp_p:
        opp_hr9 = opp_p.get("hr9", LG["hr9"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_quality = (opp_fip / LG["fip"] * 0.6 + opp_hr9 / LG["hr9"] * 0.4)
        reg_slg *= (1 + (opp_quality - 1) * 0.30)

    # Platoon
    if platoon and platoon.get("adjustment"):
        reg_slg *= platoon["adjustment"]

    # Park (blend general + HR)
    if park:
        pk = _park(park, PARK) * 0.55 + _park(park, PARK_HR) * 0.45
        reg_slg *= (1 + (pk - 1) * 0.40)

    # Weather (HR mult matters more for TB)
    if wx:
        reg_slg *= get_stat_specific_weather_adjustment(wx, "total_bases")

    # v018 Task 3A: Opportunity-first PA estimation
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    bb_rate_pct = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    exp_ab = exp_pa * (1 - bb_rate_pct / 100)

    mu = max(exp_ab * reg_slg, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_slg": round(reg_slg, 3),
            "expected_pa": round(exp_pa, 1), "expected_ab": round(exp_ab, 1)}


def project_batter_home_runs(b, opp_p=None, bvp=None, platoon=None,
                               park=None, wx=None, lineup_pos=None):
    """
    BATTER HOME RUNS — REDESIGNED AS BINOMIAL MODEL (P(1+ HR in game))

    Previous version used: HR/PA rate * expected AB = continuous projection (0.14)
    But the line is 0.5, and actual HRs are discrete (0, 1, 2, ...)
    So we need: P(at least 1 HR in expected PA) = 1 - (1 - HR_rate)^PA

    This gives a probability (0-1) that directly compares to the 0.5 line.
    Even Judge only homers in ~26% of games. The line is usually 0.5.
    """
    hr = b.get("hr", 0)
    pa = b.get("pa", 0)
    iso = b.get("iso", LG["iso"])
    barrel = b.get("recent_barrel_rate", LG["barrel_rate"])

    # Base HR rate (per PA)
    hr_rate = hr / pa if pa > 0 else LG["hr_per_pa"]
    reg_hr_rate = _regress(hr_rate, pa, 300, LG["hr_per_pa"])

    # Barrel rate is THE predictor of HR ability
    if barrel > 0:
        barrel_adj = barrel / LG["barrel_rate"]
        reg_hr_rate *= (1 + (barrel_adj - 1) * 0.35)

    # ISO confirmation
    reg_iso = _regress(iso, pa, STAB["iso"], LG["iso"])
    iso_adj = reg_iso / LG["iso"]
    reg_hr_rate *= (1 + (iso_adj - 1) * 0.15)

    # BvP HR history
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 10:
        bvp_hr_rate = bvp["home_runs"] / bvp["pa"] if bvp["pa"] > 0 else 0
        if bvp_hr_rate > 0:
            bvp_w = min(bvp["pa"] / 60, 0.20)
            reg_hr_rate = reg_hr_rate * (1 - bvp_w) + bvp_hr_rate * bvp_w

    # Opposing pitcher HR tendency
    if opp_p:
        opp_hr9 = opp_p.get("hr9", LG["hr9"])
        reg_hr_rate *= (opp_hr9 / LG["hr9"])

    # Platoon (ISO is ~14% higher in favorable platoon)
    if platoon and platoon.get("favorable"):
        reg_hr_rate *= 1.14
    elif platoon and platoon.get("favorable") is False:
        reg_hr_rate *= 0.86

    # Park HR factor (biggest effect of any prop)
    if park: reg_hr_rate *= _park(park, PARK_HR)

    # Weather (temp is huge for HR — 2% per degree C above 72F)
    if wx:
        reg_hr_rate *= get_stat_specific_weather_adjustment(wx, "home_runs")

    # v018 Task 3A: Opportunity-first PA estimation
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]

    # CORRECTED APPROACH: Expected HR count = PA × HR rate
    # The distribution layer will handle converting this into P(over/under line)
    rate_clamped = max(min(reg_hr_rate, 0.20), 0.001)
    mu = exp_pa * rate_clamped

    return {"projection": round(mu, 3), "mu": mu, "expected_count": round(mu, 4)}


def project_batter_rbis(b, opp_p=None, bvp=None, platoon=None,
                          park=None, wx=None, lineup_pos=None,
                          lineup_context=None):
    """
    BATTER RBIs — heavily dependent on lineup context.

    RBIs are NOT purely a player skill stat — they depend on who's on base.
    Lineup position is critical: cleanup hitter behind 3 high-OBP guys
    has way more RBI opportunities than a #8 hitter.

    Key: wOBA (weights all offensive events) + lineup position + team run environment.
    """
    woba = b.get("woba", LG["woba"])
    slg = b.get("slg", LG["slg"])
    xwoba = b.get("xwoba", 0)
    wrc_plus = b.get("wrc_plus", 0)
    pa = b.get("pa", 0)
    hr = b.get("hr", 0)
    rbi = b.get("rbi", 0)
    games = _estimate_batter_games(b)

    reg_woba = _regress(woba, pa, STAB["woba"], LG["woba"])
    reg_slg = _regress(slg, pa, STAB["slg"], LG["slg"])
    if xwoba and xwoba > 0:
        reg_woba = reg_woba * 0.70 + xwoba * 0.30

    team_runs_pg = _estimate_team_runs_per_game(
        b,
        opp_p=opp_p,
        park=park,
        wx=wx,
        platoon=platoon,
        team_lineup_context=lineup_context,
    )
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos, team_runs_per_game=team_runs_pg)
    exp_pa = pa_result["mean_pa"]

    observed_rbi_per_game = rbi / games if (rbi and games > 0) else LG["rbi_per_game"]
    reg_rbi_per_game = _regress(observed_rbi_per_game, pa, 320, LG["rbi_per_game"])
    proj = reg_rbi_per_game * (exp_pa / 4.2)

    proj *= RBI_LINEUP_MULTIPLIERS.get(lineup_pos, 1.0)
    proj *= 1 + (reg_woba / LG["woba"] - 1) * 0.42
    proj *= 1 + (reg_slg / LG["slg"] - 1) * 0.28

    if wrc_plus and wrc_plus > 0:
        proj *= 1 + (wrc_plus / 100.0 - 1) * 0.16

    # Power hitters drive in more runs (HR = guaranteed RBI)
    hr_rate = hr / pa if pa > 0 else LG["hr_per_pa"]
    proj *= 1 + (hr_rate / LG["hr_per_pa"] - 1) * 0.14

    # BvP
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 10:
        bvp_slg = bvp.get("slg", LG["slg"])
        bvp_adj = bvp_slg / LG["slg"]
        proj *= (1 + (bvp_adj - 1) * 0.12)

    if platoon and platoon.get("adjustment"):
        proj *= 1 + (platoon["adjustment"] - 1) * 0.55

    proj *= 1 + (team_runs_pg / 4.5 - 1) * 0.75

    lineup_support_mult = 1.0
    if lineup_context and lineup_context.get("has_data"):
        lineup_support_mult *= _context_multiplier(lineup_context.get("ahead_obp", LG["obp"]), LG["obp"], 0.36)
        lineup_support_mult *= _context_multiplier(lineup_context.get("ahead_woba", LG["woba"]), LG["woba"], 0.18)
        lineup_support_mult *= _context_multiplier(lineup_context.get("ahead_bb_rate", LG["bb_rate"]), LG["bb_rate"], 0.06)
        lineup_support_mult *= _context_multiplier(lineup_context.get("team_avg_woba", LG["woba"]), LG["woba"], 0.12)
        lineup_support_mult = max(0.82, min(lineup_support_mult, 1.30))
        proj *= lineup_support_mult

    mu = max(proj, 0.1)
    return {
        "projection": round(mu, 2),
        "mu": mu,
        "regressed_woba": round(reg_woba, 3),
        "regressed_slg": round(reg_slg, 3),
        "expected_pa": round(exp_pa, 2),
        "team_runs_per_game": round(team_runs_pg, 2),
        "lineup_support_mult": round(lineup_support_mult, 3),
        "ahead_obp": lineup_context.get("ahead_obp") if lineup_context else None,
        "ahead_woba": lineup_context.get("ahead_woba") if lineup_context else None,
        "team_avg_woba": lineup_context.get("team_avg_woba") if lineup_context else None,
    }


def project_batter_runs(b, opp_p=None, bvp=None, platoon=None,
                          park=None, wx=None, lineup_pos=None,
                          lineup_context=None):
    """
    BATTER RUNS SCORED — OBP-driven + lineup position + sprint speed.

    Leadoff hitters score more runs. Fast guys score more.
    OBP = getting on base = opportunity to score.
    """
    obp = b.get("obp", LG["obp"])
    woba = b.get("woba", LG["woba"])
    xwoba = b.get("xwoba", 0)
    wrc_plus = b.get("wrc_plus", 0)
    pa = b.get("pa", 0)
    sprint = b.get("sprint_speed", LG["sprint_speed"])
    runs = b.get("r", 0)
    games = _estimate_batter_games(b)

    reg_obp = _regress(obp, pa, STAB["obp"], LG["obp"])
    reg_woba = _regress(woba, pa, STAB["woba"], LG["woba"])
    if xwoba and xwoba > 0:
        reg_woba = reg_woba * 0.72 + xwoba * 0.28

    team_runs_pg = _estimate_team_runs_per_game(
        b,
        opp_p=opp_p,
        park=park,
        wx=wx,
        platoon=platoon,
        team_lineup_context=lineup_context,
    )
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos, team_runs_per_game=team_runs_pg)
    exp_pa = pa_result["mean_pa"]

    observed_runs_per_game = runs / games if (runs and games > 0) else LG["runs_per_game"]
    reg_runs_per_game = _regress(observed_runs_per_game, pa, 280, LG["runs_per_game"])
    proj = reg_runs_per_game * (exp_pa / 4.2)

    proj *= RUNS_LINEUP_MULTIPLIERS.get(lineup_pos, 1.0)
    proj *= 1 + (reg_obp / LG["obp"] - 1) * 0.58
    proj *= 1 + (reg_woba / LG["woba"] - 1) * 0.18

    if wrc_plus and wrc_plus > 0:
        proj *= 1 + (wrc_plus / 100.0 - 1) * 0.12

    # Sprint speed: fast players score from 1st on doubles, score on sac flies
    if sprint > 0:
        speed_adj = (sprint - LG["sprint_speed"]) / LG["sprint_speed"] * 0.12
        proj *= (1 + speed_adj)

    # Better run environments matter more for runs than for other batter props.
    proj *= 1 + (team_runs_pg / 4.5 - 1) * 0.65

    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 10:
        bvp_obp = bvp.get("obp", 0)
        if bvp_obp and bvp_obp > 0:
            proj *= 1 + (bvp_obp / LG["obp"] - 1) * 0.10

    if platoon and platoon.get("adjustment"):
        proj *= 1 + (platoon["adjustment"] - 1) * 0.45

    lineup_support_mult = 1.0
    if lineup_context and lineup_context.get("has_data"):
        lineup_support_mult *= _context_multiplier(lineup_context.get("behind_woba", LG["woba"]), LG["woba"], 0.30)
        lineup_support_mult *= _context_multiplier(lineup_context.get("behind_slg", LG["slg"]), LG["slg"], 0.22)
        lineup_support_mult *= _context_multiplier(lineup_context.get("team_avg_woba", LG["woba"]), LG["woba"], 0.10)
        behind_k_rate = lineup_context.get("behind_k_rate")
        if behind_k_rate:
            behind_k_mult = 1 + (LG["k_rate"] / max(float(behind_k_rate), 1.0) - 1) * 0.10
            lineup_support_mult *= max(0.88, min(behind_k_mult, 1.12))
        lineup_support_mult = max(0.82, min(lineup_support_mult, 1.28))
        proj *= lineup_support_mult

    mu = max(proj, 0.1)
    return {
        "projection": round(mu, 2),
        "mu": mu,
        "regressed_obp": round(reg_obp, 3),
        "regressed_woba": round(reg_woba, 3),
        "expected_pa": round(exp_pa, 2),
        "team_runs_per_game": round(team_runs_pg, 2),
        "lineup_support_mult": round(lineup_support_mult, 3),
        "behind_woba": lineup_context.get("behind_woba") if lineup_context else None,
        "behind_slg": lineup_context.get("behind_slg") if lineup_context else None,
        "behind_k_rate": lineup_context.get("behind_k_rate") if lineup_context else None,
        "team_avg_woba": lineup_context.get("team_avg_woba") if lineup_context else None,
    }


def project_batter_stolen_bases(b, park=None):
    """
    BATTER STOLEN BASES — sprint speed + opportunity + catcher.

    SB is a rare event (most lines are 0.5). Very binary.
    Sprint speed is essentially a physical trait — stabilizes fast.
    """
    sb = b.get("sb", 0)
    pa = b.get("pa", 0)
    sprint = b.get("sprint_speed", LG["sprint_speed"])
    obp = b.get("obp", LG["obp"])

    # SB rate per game
    games = pa / 4.2 if pa > 0 else 1
    sb_rate = sb / games if games > 0 else LG["sb_per_game"]
    reg_sb = _regress(sb_rate, pa, 200, LG["sb_per_game"])

    # Sprint speed is the dominant factor
    if sprint > 0:
        speed_factor = (sprint - 25.0) / (30.0 - 25.0)  # Normalize: 25=slow, 30=elite
        speed_factor = max(0, min(1.5, speed_factor))
        reg_sb *= (0.5 + speed_factor)

    # OBP: can't steal if you're not on base
    reg_obp = _regress(obp, pa, STAB["obp"], LG["obp"])
    reg_sb *= (reg_obp / LG["obp"])

    if park: reg_sb *= _park(park, PARK_SB)

    mu = max(reg_sb, 0.01)
    return {"projection": round(mu, 3), "mu": mu, "sprint_speed": sprint}


def project_batter_strikeouts(b, opp_p=None, bvp=None, platoon=None,
                                park=None, ump=None, lineup_pos=None):
    """
    BATTER STRIKEOUTS — K% is the fastest-stabilizing batter stat (60 PA).

    High-K batters against high-K pitchers with a big-zone umpire = K over city.
    """
    k_rate = _ensure_pct(b.get("k_rate"), lg_default=LG["k_rate"])
    pa = b.get("pa", 0)
    contact = b.get("contact_rate", 0)

    reg_k = _regress(k_rate, pa, STAB["k_rate"], LG["k_rate"])

    # Contact rate adjustment (if available from Statcast)
    if contact > 0:
        contact_adj = (100 - contact) / (100 - (100 - LG["k_rate"]))
        reg_k = reg_k * 0.75 + (reg_k * contact_adj) * 0.25

    # Opposing pitcher K ability
    if opp_p:
        opp_k = _ensure_pct(opp_p.get("k_pct"), lg_default=LG["k_pct_p"])
        opp_csw = opp_p.get("recent_csw_pct", LG["csw_pct"])
        opp_k_quality = (opp_k / LG["k_pct_p"] * 0.6 +
                          (opp_csw / LG["csw_pct"] if opp_csw > 0 else 1.0) * 0.4)
        reg_k *= opp_k_quality

    # BvP K rate
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_k = bvp.get("k_rate", LG["k_rate"])
        bvp_w = min(bvp["pa"] / 50, 0.25)
        reg_k = reg_k * (1 - bvp_w) + bvp_k * bvp_w

    # Platoon (same-hand = more Ks)
    if platoon:
        k_plat = platoon.get("k_adjustment", 1.0)
        reg_k *= k_plat

    # Umpire (big zone = more Ks for batters too)
    if ump and ump.get("known"):
        k_ump_adj = ump.get("k_adjustment", 0) * 0.15  # Scaled for individual batter
        reg_k *= (1 + k_ump_adj / 5)

    # v018 Task 3A: Opportunity-first PA estimation
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    proj = exp_pa * (reg_k / 100)

    if park: proj *= _park(park, PARK_K)

    # v018: ABS Challenge System adjustment
    proj *= abs_adjustment_factor("batter_strikeouts")

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_k_rate": round(reg_k, 1),
            "expected_pa": round(exp_pa, 1)}


def project_batter_walks(b, opp_p=None, ump=None, lineup_pos=None):
    """BATTER WALKS — BB% + opposing pitcher BB% + umpire zone."""
    bb_rate = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    pa = b.get("pa", 0)

    reg_bb = _regress(bb_rate, pa, STAB["bb_rate"], LG["bb_rate"])

    if opp_p:
        opp_bb = _ensure_pct(opp_p.get("bb_pct"), lg_default=LG["bb_pct_p"])
        reg_bb *= (opp_bb / LG["bb_pct_p"])

    # Tight-zone ump = more walks
    if ump and ump.get("known"):
        k_adj = ump.get("k_adjustment", 0)
        reg_bb *= (1 - k_adj * 0.05)  # High-K ump = fewer walks

    # v018 Task 3A: Opportunity-first PA estimation
    pa_result = estimate_plate_appearances(lineup_pos=lineup_pos)
    exp_pa = pa_result["mean_pa"]
    proj = exp_pa * (reg_bb / 100)

    # v018: ABS Challenge System adjustment
    proj *= abs_adjustment_factor("walks")

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_bb_rate": round(reg_bb, 1)}


def project_hitter_fantasy_score(b, opp_p=None, bvp=None, platoon=None,
                                 park=None, wx=None, lineup_pos=None,
                                 lineup_context=None):
    """
    HITTER FANTASY SCORE — DraftKings scoring system.

    DK Scoring Weights:
      Single = 3, Double = 5, Triple = 8, Home Run = 10,
      RBI = 2, Run = 2, Walk/HBP = 2, Stolen Base = 5

    League average: ~7.7 fantasy pts/game (4.2 PA x ~1.83 pts/PA).
    Line is usually 7.5. Players range from ~4.5 (weak) to ~12+ (elite).

    Approach: Calculate per-PA rates for each scoring event from season stats,
    regress via Bayesian stabilization, blend with Statcast expected stats,
    apply contextual multipliers, multiply by expected PA.
    """
    pa = b.get("pa", 0)
    avg = b.get("avg", LG["avg"])
    obp = b.get("obp", LG["obp"])
    slg = b.get("slg", LG["slg"])
    iso = b.get("iso", LG["iso"])
    bb_rate = _ensure_pct(b.get("bb_rate"), lg_default=LG["bb_rate"])
    k_rate = _ensure_pct(b.get("k_rate"), lg_default=LG["k_rate"])
    hr = b.get("hr", 0)
    sb = b.get("sb", 0)
    rbi = b.get("rbi", 0)
    r = b.get("r", 0)
    doubles = b.get("2b", 0)
    triples = b.get("3b", 0)
    sprint = b.get("sprint_speed", LG["sprint_speed"])

    # Statcast expected stats
    xba = b.get("xba", 0)
    xslg = b.get("xslg", 0)
    barrel = b.get("recent_barrel_rate", b.get("barrel_rate", LG["barrel_rate"]))
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])

    # ── Per-PA rates for each scoring event ──
    games_est = pa / 4.2 if pa > 0 else 1

    # Hit types from season stats — derive from rate stats when counting unavailable
    ab_est = pa * (1 - bb_rate / 100) if pa > 0 else 0
    hits = avg * ab_est if pa > 0 else 0

    # HR: prefer counting stat, fall back to hr_rate × PA, then ISO estimate
    hr_count = hr
    if not hr_count and pa > 0:
        _hr_rate = b.get("hr_rate", 0) or b.get("hr_fb", 0)
        if _hr_rate and _hr_rate < 1:
            hr_count = _hr_rate * pa
        elif iso > 0:
            hr_count = iso * 0.22 * pa  # ISO to HR approximation

    dbl_count = doubles
    if not dbl_count and pa > 0:
        # Estimate doubles from ISO and SLG: doubles ≈ (SLG - AVG - 3×HR/AB) / 2 × AB
        _hr_per_ab = hr_count / ab_est if ab_est > 0 else 0
        dbl_count = max((slg - avg - 3 * _hr_per_ab) / 2 * ab_est, 0) if ab_est > 0 else 0

    trp_count = triples
    singles_count = max(hits - hr_count - dbl_count - trp_count, 0) if pa > 0 else 0

    # Per-PA rates
    hr_per_pa = hr_count / pa if pa > 0 else LG["hr_per_pa"]
    dbl_per_pa = dbl_count / pa if pa > 0 else 0.045  # ~4.5% league avg
    trp_per_pa = trp_count / pa if pa > 0 else 0.004  # ~0.4% league avg
    single_per_pa = singles_count / pa if pa > 0 else 0.135  # ~13.5% league avg
    bb_per_pa = bb_rate / 100 if bb_rate > 0 else LG["bb_rate"] / 100

    # RBI/R/SB: derive from rate stats when counting unavailable
    sb_per_game = sb / games_est if (sb and games_est > 0) else LG["sb_per_game"]
    rbi_per_game = rbi / games_est if (rbi and games_est > 0) else LG["rbi_per_game"]
    r_per_game = r / games_est if (r and games_est > 0) else LG["runs_per_game"]

    # ── Bayesian stabilization ──
    reg_hr = _regress(hr_per_pa, pa, 300, LG["hr_per_pa"])
    reg_dbl = _regress(dbl_per_pa, pa, 200, 0.045)
    reg_trp = _regress(trp_per_pa, pa, 400, 0.004)
    reg_single = _regress(single_per_pa, pa, STAB["avg"], 0.135)
    reg_bb = _regress(bb_per_pa, pa, STAB["bb_rate"], LG["bb_rate"] / 100)
    reg_sb = _regress(sb_per_game, pa, 200, LG["sb_per_game"])
    reg_rbi = _regress(rbi_per_game, pa, 200, LG["rbi_per_game"])
    reg_r = _regress(r_per_game, pa, 200, LG["runs_per_game"])

    # ── Statcast blend — v015: increased from 30% to 40% weight ──
    # xBA/xSLG strip BABIP luck and provide true-talent hitting rates.
    # Backtest shows FS under-projects by 1.57 pts; stronger Statcast helps.
    if xba > 0 and xslg > 0:
        x_iso = xslg - xba
        x_hr_per_pa = x_iso * 0.22  # ISO to HR approximation
        x_single_per_pa = xba - x_hr_per_pa - reg_dbl - reg_trp
        x_single_per_pa = max(x_single_per_pa, 0.05)

        reg_hr = reg_hr * 0.60 + x_hr_per_pa * 0.40
        reg_single = reg_single * 0.60 + x_single_per_pa * 0.40

    # Barrel rate boost for power events — v015: 0.25 → 0.30
    if barrel > 0:
        barrel_adj = barrel / LG["barrel_rate"]
        reg_hr *= (1 + (barrel_adj - 1) * 0.30)
        reg_dbl *= (1 + (barrel_adj - 1) * 0.12)

    # ── Contextual multipliers ──
    context_mult = 1.0

    # Opposing pitcher quality
    if opp_p:
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_whip = opp_p.get("whip", LG["whip"])
        # Bad pitchers (high FIP) = more fantasy points
        opp_adj = (opp_fip / LG["fip"] * 0.6 + opp_whip / LG["whip"] * 0.4)
        context_mult *= (1 + (opp_adj - 1) * 0.35)

    # BvP matchup
    bvp_mult = 1.0
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_slg = bvp.get("slg", LG["slg"])
        bvp_avg = bvp.get("avg", LG["avg"])
        bvp_quality = (bvp_slg / LG["slg"] * 0.6 + bvp_avg / LG["avg"] * 0.4)
        bvp_weight = min(bvp["pa"] / 50, 0.25)
        bvp_mult = 1 + (bvp_quality - 1) * bvp_weight

    # Platoon advantage
    platoon_mult = 1.0
    if platoon and platoon.get("adjustment"):
        platoon_mult = platoon["adjustment"]

    # Park factor (blend general + HR)
    park_mult = 1.0
    if park:
        pk = _park(park, PARK) * 0.55 + _park(park, PARK_HR) * 0.45
        park_mult = 1 + (pk - 1) * 0.35

    # Weather
    wx_mult = 1.0
    if wx:
        wx_mult = get_stat_specific_weather_adjustment(wx, "hitter_fantasy_score")

    total_mult = context_mult * bvp_mult * platoon_mult * park_mult * wx_mult

    # ── Calculate expected PA ──
    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2

    lineup_support_mult = 1.0
    if lineup_context and lineup_context.get("has_data"):
        lineup_support_mult *= _context_multiplier(lineup_context.get("ahead_obp", LG["obp"]), LG["obp"], 0.12)
        lineup_support_mult *= _context_multiplier(lineup_context.get("behind_woba", LG["woba"]), LG["woba"], 0.12)
        lineup_support_mult *= _context_multiplier(lineup_context.get("team_avg_woba", LG["woba"]), LG["woba"], 0.08)
        lineup_support_mult = max(0.88, min(lineup_support_mult, 1.18))
        total_mult *= lineup_support_mult

    # ── Fantasy points per PA ──
    # DK scoring: 1B=3, 2B=5, 3B=8, HR=10, RBI=2, R=2, BB/HBP=2, SB=5
    fantasy_per_pa = (
        reg_single * 3 +     # Singles
        reg_dbl * 5 +         # Doubles
        reg_trp * 8 +         # Triples
        reg_hr * 10 +         # Home Runs
        reg_bb * 2            # Walks/HBP
    ) * total_mult

    # Per-game events (RBI, Runs, SB) — not per PA
    fantasy_per_game = (
        reg_rbi * 2 +         # RBIs
        reg_r * 2 +           # Runs
        reg_sb * 5            # Stolen Bases
    ) * total_mult

    # Total projection = per-PA component * expected PA + per-game component
    mu = max(fantasy_per_pa * exp_pa + fantasy_per_game, 2.0)

    return {
        "projection": round(mu, 2), "mu": mu,
        "fantasy_per_pa": round(fantasy_per_pa, 3),
        "expected_pa": round(exp_pa, 1),
        "context_mult": round(total_mult, 3),
        "lineup_support_mult": round(lineup_support_mult, 3),
        "ahead_obp": lineup_context.get("ahead_obp") if lineup_context else None,
        "behind_woba": lineup_context.get("behind_woba") if lineup_context else None,
        "team_avg_woba": lineup_context.get("team_avg_woba") if lineup_context else None,
    }


def project_hits_runs_rbis(b, opp_p=None, bvp=None, platoon=None,
                            park=None, wx=None, ump=None, lineup_pos=None,
                            lineup_context=None):
    """HITS + RUNS + RBIs combo — sum of individual projections."""
    hits = project_batter_hits(b, opp_p, bvp, platoon, park, wx, lineup_pos)
    runs = project_batter_runs(b, opp_p, bvp, platoon, park, wx, lineup_pos, lineup_context)
    rbis = project_batter_rbis(b, opp_p, bvp, platoon, park, wx, lineup_pos, lineup_context)

    mu = hits["mu"] + runs["mu"] + rbis["mu"]
    return {
        "projection": round(mu, 2),
        "mu": mu,
        "hits_proj": hits["projection"],
        "runs_proj": runs["projection"],
        "rbis_proj": rbis["projection"],
        "team_runs_per_game": max(runs.get("team_runs_per_game", 0.0), rbis.get("team_runs_per_game", 0.0)),
        "ahead_obp": rbis.get("ahead_obp"),
        "behind_woba": runs.get("behind_woba"),
        "team_avg_woba": max(runs.get("team_avg_woba") or 0.0, rbis.get("team_avg_woba") or 0.0),
    }


# ═══════════════════════════════════════════════════════
# PROBABILITY & DISTRIBUTION
# ═══════════════════════════════════════════════════════

def _rating_from_confidence(confidence: float, is_dead_zone: bool = False) -> str:
    """Map resolved confidence to the display grade."""
    if is_dead_zone:
        return "D"
    if confidence >= 0.66:
        return "A"
    if confidence >= 0.60:
        return "B"
    if confidence >= 0.55:
        return "C"
    return "D"


def _win_prob_from_confidence(confidence: float, p_push: float) -> float:
    """Convert resolved confidence back to outright win probability."""
    non_push_mass = max(0.0, 1.0 - p_push)
    return confidence * non_push_mass


def calculate_tail_metrics(projection: float, line: float, prop_type: str,
                           proj_result: Optional[dict] = None,
                           weights_override: Optional[dict] = None,
                           dist_type_override: Optional[str] = None) -> dict:
    """Return percentile and breakout/dud metrics for the projected outcome."""
    mu = max(float(projection), 0.01)
    weights = weights_override or _load_weights()
    cfg = get_distribution_config(prop_type, weights)
    dist_type = dist_type_override or cfg["dist_type"]
    var_ratio = cfg["var_ratio"]
    phi = cfg["phi"]

    proj_result = proj_result or {}
    bb_alpha = proj_result.get("bb_alpha")
    bb_beta = proj_result.get("bb_beta")
    expected_bf = proj_result.get("expected_bf")
    n_batters = int(expected_bf) if expected_bf else None

    p10 = distributions.distribution_quantile(
        0.10, mu, dist_type, var_ratio=var_ratio, phi=phi,
        n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
    )
    p50 = distributions.distribution_quantile(
        0.50, mu, dist_type, var_ratio=var_ratio, phi=phi,
        n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
    )
    p90 = distributions.distribution_quantile(
        0.90, mu, dist_type, var_ratio=var_ratio, phi=phi,
        n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
    )

    tail_cfg = _get_tail_signal_config(weights)
    label_cfg = dict(tail_cfg["label_thresholds"])
    label_cfg.update(tail_cfg.get("label_thresholds_by_prop", {}).get(prop_type, {}))
    prop_cfg = tail_cfg["prop_thresholds"].get(prop_type, {})

    breakout_target = None
    breakout_prob = None
    dud_target = None
    dud_prob = None

    if "good_over" in prop_cfg:
        breakout_target = prop_cfg["good_over"]
        breakout_prob = distributions.prob_at_least(
            breakout_target, mu, dist_type, var_ratio=var_ratio, phi=phi,
            n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
        )
    elif "good_under" in prop_cfg:
        breakout_target = prop_cfg["good_under"]
        breakout_prob = distributions.prob_at_most(
            breakout_target, mu, dist_type, var_ratio=var_ratio, phi=phi,
            n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
        )

    if "bad_under" in prop_cfg:
        dud_target = prop_cfg["bad_under"]
        dud_prob = distributions.prob_at_most(
            dud_target, mu, dist_type, var_ratio=var_ratio, phi=phi,
            n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
        )
    elif "bad_over" in prop_cfg:
        dud_target = prop_cfg["bad_over"]
        dud_prob = distributions.prob_at_least(
            dud_target, mu, dist_type, var_ratio=var_ratio, phi=phi,
            n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta,
        )

    breakout_prob = max(0.0, min(1.0, breakout_prob if breakout_prob is not None else 0.0))
    dud_prob = max(0.0, min(1.0, dud_prob if dud_prob is not None else 0.0))

    return {
        "p10": round(float(p10), 2),
        "p50": round(float(p50), 2),
        "p90": round(float(p90), 2),
        "breakout_target": breakout_target,
        "breakout_prob": round(float(breakout_prob), 4),
        "breakout_watch": _label_tail_probability(
            breakout_prob,
            label_cfg["breakout_medium"],
            label_cfg["breakout_high"],
        ),
        "dud_target": dud_target,
        "dud_prob": round(float(dud_prob), 4),
        "dud_risk": _label_tail_probability(
            dud_prob,
            label_cfg["dud_medium"],
            label_cfg["dud_high"],
        ),
    }


def calculate_over_under_probability(projection, line, prop_type, proj_result=None,
                                     weights_override: Optional[dict] = None,
                                     dist_type_override: Optional[str] = None):
    """
    P(over) and P(under) using prop-appropriate distributions.

    Distribution selection (via distributions.compute_probabilities):
      Negative Binomial: all count props (hits, Ks, walks, runs, SB, ER, home_runs, etc.)
      Beta-Binomial: pitcher strikeouts (bounded by batters faced)
      Gamma: continuous-ish scores (fantasy score)
      Normal: high-mean continuous props (pitching outs, H+R+RBI)

    Args:
        projection: The mean projection value
        line: The prop line
        prop_type: Type of prop (e.g., "pitcher_strikeouts")
        proj_result: Optional dict with projection details (e.g., BB params for strikeouts)
    """

    mu = max(projection, 0.01)

    # Load weights once (needed for both BB and non-BB paths)
    weights = weights_override or _load_weights()

    # ── Compute probabilities via centralized distributions module ──────
    # distributions.compute_probabilities() is the SINGLE source of truth for
    # all CDF/PMF calculations. It reads distribution type from weights file
    # (distribution_params) and routes to the correct distribution.
    dist_cfg = get_distribution_config(prop_type, weights)
    dist_type = dist_type_override or dist_cfg["dist_type"]
    var_ratio = dist_cfg["var_ratio"]
    phi = dist_cfg["phi"]

    # Build distribution-specific params
    bb_alpha = None
    bb_beta = None
    n_batters = None
    if dist_type == "betabinom" and proj_result is not None:
        bb_alpha = proj_result.get("bb_alpha")
        bb_beta = proj_result.get("bb_beta")
        n_batters = int(proj_result.get("expected_bf", 18)) if proj_result.get("expected_bf") else None

    # For earned runs with pre-fit NB params, use them directly
    if (prop_type == "earned_runs" and proj_result is not None and
            "nb_n" in proj_result and "nb_p" in proj_result):
        nb_n = proj_result.get("nb_n")
        nb_p = proj_result.get("nb_p")
        if nb_n > 0 and 0 < nb_p < 1:
            prob_result_dist = {
                "p_over": distributions.prob_over_negbinom(line, nb_n, nb_p),
                "p_under": distributions.prob_under_negbinom(line, nb_n, nb_p),
                "p_push": distributions.prob_push_negbin_mu(line, mu, var_ratio)
                          if line == int(line) else 0.0,
            }
        else:
            prob_result_dist = distributions.compute_probabilities(
                line, mu, dist_type, var_ratio=var_ratio, phi=phi,
                n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta)
    else:
        prob_result_dist = distributions.compute_probabilities(
            line, mu, dist_type, var_ratio=var_ratio, phi=phi,
            n_batters=n_batters, bb_alpha=bb_alpha, bb_beta=bb_beta)

    p_over = prob_result_dist["p_over"]
    p_under = prob_result_dist["p_under"]
    p_push = prob_result_dist["p_push"]

    # Keep a single probability contract everywhere:
    # p_over + p_under + p_push == 1.0
    total = p_over + p_under + p_push
    if total > 0:
        p_over /= total
        p_under /= total
        p_push /= total
    else:
        p_over = 0.5
        p_under = 0.5
        p_push = 0.0

    non_push_mass = p_over + p_under
    if non_push_mass > 0:
        resolved_over = p_over / non_push_mass
        resolved_under = p_under / non_push_mass
    else:
        resolved_over = 0.5
        resolved_under = 0.5

    pick = "MORE" if resolved_over >= resolved_under else "LESS"

    # ── v016: Empirical calibration blend ──────────────────
    # Per-prop blend of theoretical distribution probability with empirical
    # observed rates from 128K+ backtest predictions. Weights optimized via
    # grid search: hits/PK use pure theoretical, TB/FS use heavy empirical.
    is_dead_zone = False
    blend_weights = _get_calibration_blend_weights(weights)
    base_emp_weight = blend_weights.get(prop_type, 0.0)
    if base_emp_weight > 0:
        emp = _empirical_probability(mu, prop_type, line=line)
        if emp is not None:
            is_dead_zone = emp["is_dead_zone"]
            # Reduce weight slightly for low-N bins (< 100 observations)
            emp_weight = base_emp_weight if emp["n"] >= 100 else base_emp_weight * 0.8
            theo_weight = 1.0 - emp_weight
            resolved_over = emp["p_over"] * emp_weight + resolved_over * theo_weight
            resolved_under = emp["p_under"] * emp_weight + resolved_under * theo_weight
            total = resolved_over + resolved_under
            if total > 0:
                resolved_over /= total
                resolved_under /= total

    pick = "MORE" if resolved_over >= resolved_under else "LESS"

    # Confidence is the resolved (non-push) probability that the chosen side wins.
    raw_prob = max(resolved_over, resolved_under)

    # ── MODEL UNCERTAINTY DISCOUNT ─────────────────────────────
    # Raw CDF probability assumes the projection is perfect truth.
    # In reality, projections have ~0.3-0.5 standard error.
    # Backtest shows A-grade picks hit 55% vs 70%+ raw probability —
    # a ~15pp overconfidence gap. Apply a calibration curve that
    # compresses extreme probabilities toward 50%.
    #
    # Formula: calibrated = 0.50 + (raw - 0.50) × shrinkage
    # shrinkage < 1.0 pulls everything toward the coinflip baseline.
    # 0.70 shrinkage: raw 80% → calibrated 71%, raw 75% → 67.5%, raw 60% → 57%
    model_uncertainty_shrinkage = _get_confidence_shrinkage(prop_type, weights)
    confidence = 0.50 + (raw_prob - 0.50) * model_uncertainty_shrinkage
    confidence = max(0.50, min(confidence, 1.0))

    edge = round(abs(confidence - 0.50), 4)
    win_prob = _win_prob_from_confidence(confidence, p_push)
    rating = _rating_from_confidence(confidence, is_dead_zone=is_dead_zone)
    tail_metrics = calculate_tail_metrics(
        mu,
        line,
        prop_type,
        proj_result=proj_result,
        weights_override=weights,
        dist_type_override=dist_type_override,
    )


    return {
        "p_over": round(p_over, 4), "p_under": round(p_under, 4),
        "p_push": round(p_push, 4),
        "pick": pick, "confidence": round(confidence, 4),
        "win_prob": round(win_prob, 4),
        "edge": round(edge, 4), "rating": rating,
        "projection": round(mu, 2), "line": line,
        **tail_metrics,
    }


# ═══════════════════════════════════════════════════════
# MASTER ROUTER
# ═══════════════════════════════════════════════════════

def generate_prediction(player_name, stat_type, stat_internal, line,
                         batter_profile=None, pitcher_profile=None,
                         opp_pitcher_profile=None, opp_team_k_rate=None,
                         bvp=None, platoon=None, ump=None,
                         park_team=None, weather=None, lineup_pos=None,
                         batter_lineup_context=None, opp_lineup_context=None):
    """
    Master prediction router. Picks the right projection function
    based on prop type and feeds in all available context.
    """
    b = batter_profile or {}
    p = pitcher_profile or {}
    opp = opp_pitcher_profile or {}

    # Track whether we have real player data vs league-average fallback
    _has_player_data = bool(b or p)

    # Route to correct projection
    route = {
        "pitcher_strikeouts": lambda: project_pitcher_strikeouts(
            p, bvp=bvp, platoon=platoon, ump=ump,
            opp_k_rate=opp_team_k_rate, park=park_team, wx=weather,
            opp_lineup_context=opp_lineup_context),
        "pitching_outs": lambda: project_pitcher_outs(p, park=park_team, wx=weather),
        "earned_runs": lambda: project_pitcher_earned_runs(
            p, park=park_team, wx=weather, opp_woba=opp.get("woba")),
        "walks_allowed": lambda: project_pitcher_walks(p, park=park_team, ump=ump),
        "hits_allowed": lambda: project_pitcher_hits_allowed(
            p, park=park_team, wx=weather),
        "hits": lambda: project_batter_hits(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "total_bases": lambda: project_batter_total_bases(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "home_runs": lambda: project_batter_home_runs(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "rbis": lambda: project_batter_rbis(
            b, opp, bvp, platoon, park_team, weather, lineup_pos,
            batter_lineup_context),
        "runs": lambda: project_batter_runs(
            b, opp, bvp, platoon, park_team, weather, lineup_pos,
            batter_lineup_context),
        "stolen_bases": lambda: project_batter_stolen_bases(b, park_team),
        "batter_strikeouts": lambda: project_batter_strikeouts(
            b, opp, bvp, platoon, park_team, ump, lineup_pos),
        "walks": lambda: project_batter_walks(b, opp, ump, lineup_pos),
        "hitter_fantasy_score": lambda: project_hitter_fantasy_score(
            b, opp, bvp, platoon, park_team, weather, lineup_pos,
            batter_lineup_context),
        "hits_runs_rbis": lambda: project_hits_runs_rbis(
            b, opp, bvp, platoon, park_team, weather, ump, lineup_pos,
            batter_lineup_context),
        "singles": lambda: project_batter_singles(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "doubles": lambda: project_batter_doubles(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
    }

    proj_fn = route.get(stat_internal)
    if proj_fn:
        proj_result = proj_fn()
    elif b:
        proj_result = project_batter_hits(b, opp, bvp, platoon, park_team, weather, lineup_pos)
    elif p:
        proj_result = project_pitcher_strikeouts(p, bvp=bvp, park=park_team, wx=weather)
    else:
        proj_result = {"projection": line, "mu": line}

    projection = proj_result.get("projection", line)

    # ── Apply learned weights from data/weights/current.json ──
    weights = _load_weights()

    # Prop-type offset (e.g. fantasy score systematically under-projected)
    prop_offsets = weights.get("prop_type_offsets", {})
    offset = prop_offsets.get(stat_internal, 0.0)
    projection += offset

    # Direction bias correction: nudge projection up (more_multiplier) or
    # down (less_multiplier) to counteract systematic MORE/LESS skew.
    # Applied as: if projection > line → trending MORE → apply more_multiplier
    #             if projection < line → trending LESS → apply less_multiplier
    dir_bias = weights.get("direction_bias", {})
    if projection >= line:
        projection *= dir_bias.get("more_multiplier", 1.0)
    else:
        projection *= dir_bias.get("less_multiplier", 1.0)

    # v015: Borderline regression — projections near the line have low signal-to-noise.
    # Backtest shows projection-to-actual correlation is only 0.086-0.091 for batter props.
    # When the projection is very close to the line, the model is essentially guessing.
    # Gently regress borderline projections TOWARD the line to reduce false confidence
    # on coinflip picks. This widens the "no opinion" zone, only surfacing picks where
    # the model has genuine conviction.
    # Batter props: regression strength 0.15 (15% pull toward line within ±15% of line)
    # Pitcher props: regression strength 0.08 (8% — PK has better discrimination at r=0.26)
    if line > 0:
        pct_from_line = abs(projection - line) / line
        is_pitcher_prop = stat_internal in (
            "pitcher_strikeouts", "pitching_outs", "earned_runs",
            "walks_allowed", "hits_allowed",
        )
        regression_strength = 0.08 if is_pitcher_prop else 0.15
        # Only apply within ±15% of line (further away = model has real conviction)
        if pct_from_line < 0.15:
            blend = pct_from_line / 0.15  # 0 at line, 1 at boundary
            # At the line: full regression. At 15% away: no regression.
            pull = regression_strength * (1 - blend)
            projection = projection * (1 - pull) + line * pull

    proj_result["projection"] = round(projection, 3)
    proj_result["mu"] = projection

    prob_result = calculate_over_under_probability(projection, line, stat_internal,
                                                    proj_result=proj_result)

    # Add distribution info for pitcher strikeouts if available
    if stat_internal == "pitcher_strikeouts" and "bb_alpha" in proj_result:
        prob_result["distribution"] = "beta_binomial"
        prob_result["bb_p_over"] = prob_result.get("p_over")
        prob_result["bb_p_under"] = prob_result.get("p_under")

    # Add distribution info for earned runs if available
    if stat_internal == "earned_runs" and "nb_n" in proj_result:
        prob_result["distribution"] = "negative_binomial"
        prob_result["nb_p_over"] = prob_result.get("p_over")
        prob_result["nb_p_under"] = prob_result.get("p_under")

    # ── DATA QUALITY FLAGS ──────────────────────────────────
    # When player-specific data is missing, the projection is just league
    # averages or line-as-projection.  Flag this and penalize confidence.
    result = {
        "player_name": player_name, "stat_type": stat_type,
        "stat_internal": stat_internal, "line": line,
        **proj_result, **prob_result,
    }

    if batter_lineup_context:
        result["lineup_depth_woba"] = batter_lineup_context.get("lineup_depth_woba")
        result["team_avg_woba"] = batter_lineup_context.get("team_avg_woba", result.get("team_avg_woba"))
        result["team_avg_obp"] = batter_lineup_context.get("team_avg_obp")
        result["ahead_obp"] = batter_lineup_context.get("ahead_obp", result.get("ahead_obp"))
        result["ahead_woba"] = batter_lineup_context.get("ahead_woba", result.get("ahead_woba"))
        result["behind_woba"] = batter_lineup_context.get("behind_woba", result.get("behind_woba"))
        result["behind_slg"] = batter_lineup_context.get("behind_slg", result.get("behind_slg"))
        result["behind_k_rate"] = batter_lineup_context.get("behind_k_rate")

    if opp_lineup_context:
        result["opp_lineup_k_rate"] = opp_lineup_context.get("top6_k_rate") or opp_lineup_context.get("avg_k_rate")
        result["opp_lineup_woba"] = opp_lineup_context.get("top5_woba") or opp_lineup_context.get("avg_woba")

    result["has_player_data"] = _has_player_data
    result["has_opp_data"] = bool(opp)
    result["has_lineup_pos"] = lineup_pos is not None
    result["has_park"] = park_team is not None

    if not _has_player_data:
        # No player stats at all — projection is pure league average / line fallback.
        # Slash confidence hard: cap at 0.55 max (D-grade territory).
        raw_conf = result.get("confidence", 0.5)
        result["confidence"] = min(raw_conf, 0.55)
        result["rating"] = _rating_from_confidence(result["confidence"], is_dead_zone=True)
        result["edge"] = round(abs(result["confidence"] - 0.50), 4)
        result["win_prob"] = round(
            _win_prob_from_confidence(result["confidence"], result.get("p_push", 0.0)),
            4,
        )
        result["data_warning"] = "no_player_data"
    else:
        # Player data exists but context may be incomplete.
        # Penalize missing context fields (opponent, lineup, park).
        missing_penalty = 0.0
        if not opp:
            missing_penalty += 0.05   # No opposing pitcher data
        if lineup_pos is None:
            missing_penalty += 0.03   # No confirmed batting order
        if park_team is None:
            missing_penalty += 0.02   # No park factor

        if missing_penalty > 0:
            raw_conf = result.get("confidence", 0.5)
            result["confidence"] = max(raw_conf - missing_penalty, 0.50)
            result["rating"] = _rating_from_confidence(result["confidence"])
            result["edge"] = round(abs(result["confidence"] - 0.50), 4)
            result["win_prob"] = round(
                _win_prob_from_confidence(result["confidence"], result.get("p_push", 0.0)),
                4,
            )

    return result
