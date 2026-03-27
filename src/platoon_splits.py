"""
Player-Specific Platoon Splits

Replaces the generic ±8% platoon adjustment with actual per-player L/R split
data from the MLB Stats API, Bayesian-shrunk toward the league-average split
when sample size is small.

Key insight: platoon splits vary wildly per player.
  - Some LHBs (e.g. Yordan Alvarez) have reverse or neutral splits
  - Some players have 20%+ true splits; others barely move
  - A single ±8% generic factor systematically mis-prices many matchups

Falls back gracefully to the generic ±8% adjustment if:
  - Player MLBAM ID not provided
  - API call fails
  - Fewer than 30 PA against that pitcher hand
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Bayesian prior: PA against a hand needed to reach 50% weight on player-specific data.
# SLG stabilizes ~320 PA, ISO ~160 PA, K% ~60 PA. We use a blend.
SPLIT_PA_PRIOR = 150

# League-average split ratios (2020-2024 FanGraphs data).
# Defined as performance_vs_FAVORABLE_hand / performance_vs_UNFAVORABLE_hand.
# For a LHB:  favorable = vs RHP,  unfavorable = vs LHP
# For a RHB:  favorable = vs LHP,  unfavorable = vs RHP
LG_SPLIT_RATIOS = {
    "ops": 1.11,    # ~30-40 wOBA-point difference maps to ~11% OPS gap
    "avg": 1.10,
    "slg": 1.14,
    "k_rate": 0.91,  # fewer Ks in favorable platoon
    "bb_rate": 1.05,
}

# Don't use player-specific data with fewer than this many PA vs that hand
MIN_PA_ANY_SIGNAL = 30

# Hard cap on any computed adjustment (prevent extreme values from tiny samples)
ADJ_MIN = 0.70
ADJ_MAX = 1.40


# ─────────────────────────────────────────────────────────────────────────────
# MLB Stats API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_season() -> int:
    now = datetime.now()
    return now.year - 1 if now.month < 4 else now.year


def _safe_float(val, default: float = 0.0) -> float:
    """Parse a value that may be a string like '.362', a float, or None."""
    if val is None:
        return default
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=1024)
def _fetch_player_splits(mlbam_id: int, season: int, group: str) -> Optional[dict]:
    """
    Fetch L/R split stats from MLB Stats API for a single player/season.

    Returns {'vs_L': {stat_dict}, 'vs_R': {stat_dict}} or None on failure.
    Results are cached in-memory for the lifetime of the process (per-run cache).

    group: 'hitting' or 'pitching'
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={
                "stats": "statSplits",
                "group": group,
                "season": season,
                "sitCodes": "vl,vr",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        result: dict = {}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                desc = split.get("split", {}).get("description", "")
                s = split.get("stat", {})
                if "Left" in desc:
                    result["vs_L"] = s
                elif "Right" in desc:
                    result["vs_R"] = s

        return result if result else None

    except Exception as exc:
        logger.debug("Platoon split fetch failed (player=%d, season=%d): %s", mlbam_id, season, exc)
        return None


def _get_splits_with_fallback(mlbam_id: int, season: int, group: str) -> Optional[dict]:
    """Try current season, then fall back to prior season if no data."""
    splits = _fetch_player_splits(mlbam_id, season, group)
    if not splits and season == _current_season():
        splits = _fetch_player_splits(mlbam_id, season - 1, group)
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ops(stat: dict) -> float:
    return _safe_float(stat.get("obp")) + _safe_float(stat.get("slg"))


def _k_rate(stat: dict) -> float:
    pa = _safe_float(stat.get("plateAppearances"))
    if pa < 1:
        return 0.0
    return _safe_float(stat.get("strikeOuts")) / pa


def _bb_rate(stat: dict) -> float:
    pa = _safe_float(stat.get("plateAppearances"))
    if pa < 1:
        return 0.0
    return _safe_float(stat.get("baseOnBalls")) / pa


def _iso(stat: dict) -> float:
    return max(_safe_float(stat.get("slg")) - _safe_float(stat.get("avg")), 0.0)


def _bayesian_blend(player_ratio: float, lg_ratio: float, pa: float) -> float:
    """
    Shrink player_ratio toward lg_ratio based on sample size.

    weight = pa / (pa + SPLIT_PA_PRIOR):
      - 0.0 when pa = 0   → use pure league average
      - 0.5 when pa = 150 → 50/50 blend
      - 0.9 when pa = 1350→ mostly player-specific
    """
    weight = pa / (pa + SPLIT_PA_PRIOR)
    blended = player_ratio * weight + lg_ratio * (1.0 - weight)
    return max(ADJ_MIN, min(ADJ_MAX, blended))


# ─────────────────────────────────────────────────────────────────────────────
# Public: batter platoon adjustment
# ─────────────────────────────────────────────────────────────────────────────

def get_batter_platoon_adjustment(
    batter_hand: str,
    pitcher_hand: str,
    mlbam_id: Optional[int] = None,
    season: Optional[int] = None,
) -> dict:
    """
    Compute batter platoon split adjustment.

    When mlbam_id is provided, fetches actual L/R split data from the MLB Stats
    API and computes a player-specific adjustment with Bayesian shrinkage toward
    league-average splits. Falls back to generic ±8% on failure or small sample.

    Returns a dict compatible with the existing `platoon` dict consumed by
    predictor.py:
        {
          "adjustment":     float,  # OPS-proxy multiplier (used for hits, TB, RBI, etc.)
          "k_adjustment":   float,  # K rate multiplier (batter K props)
          "iso_adjustment": float,  # ISO multiplier (HR/TB props)
          "avg_adjustment": float,  # AVG multiplier (hits props)
          "slg_adjustment": float,  # SLG multiplier (TB props)
          "favorable":      bool,
          "batter_hand":    str,
          "pitcher_hand":   str,
          "source":         str,   # "player_specific" | "player_specific_regressed" | "generic"
          "pa_sample":      int,
          "description":    str,
        }
    """
    batter_hand = (batter_hand or "").upper().strip()
    pitcher_hand = (pitcher_hand or "").upper().strip()

    if not batter_hand or not pitcher_hand:
        return _unknown_platoon()

    # Switch hitters always bat from the favorable side — no splits needed
    if batter_hand == "S":
        return {
            "adjustment": 1.02,
            "k_adjustment": 0.98,
            "iso_adjustment": 1.02,
            "avg_adjustment": 1.02,
            "slg_adjustment": 1.02,
            "favorable": True,
            "batter_hand": "S",
            "pitcher_hand": pitcher_hand,
            "source": "generic",
            "pa_sample": 0,
            "description": "Switch hitter — always opposite side",
        }

    is_favorable = (
        (batter_hand == "L" and pitcher_hand == "R")
        or (batter_hand == "R" and pitcher_hand == "L")
    )

    generic = _generic_platoon(batter_hand, pitcher_hand, is_favorable)

    if not mlbam_id:
        return generic

    season = season or _current_season()
    splits = _get_splits_with_fallback(mlbam_id, season, "hitting")
    if not splits:
        return generic

    split_key = f"vs_{pitcher_hand}"
    opp_key = f"vs_{'R' if pitcher_hand == 'L' else 'L'}"
    split_stat = splits.get(split_key)
    opp_stat = splits.get(opp_key)

    if not split_stat:
        return generic

    pa_vs_hand = _safe_float(split_stat.get("plateAppearances"))
    if pa_vs_hand < MIN_PA_ANY_SIGNAL:
        return generic

    # ── Compute per-metric values ──────────────────────────────────────────
    ops_vs = _ops(split_stat)
    avg_vs = _safe_float(split_stat.get("avg"))
    slg_vs = _safe_float(split_stat.get("slg"))
    k_vs = _k_rate(split_stat)
    iso_vs = _iso(split_stat)

    if opp_stat:
        pa_opp = _safe_float(opp_stat.get("plateAppearances"))
        total_pa = pa_vs_hand + pa_opp
        if total_pa > 0:
            w1, w2 = pa_vs_hand / total_pa, pa_opp / total_pa
            ops_base = ops_vs * w1 + _ops(opp_stat) * w2
            avg_base = avg_vs * w1 + _safe_float(opp_stat.get("avg")) * w2
            slg_base = slg_vs * w1 + _safe_float(opp_stat.get("slg")) * w2
            k_base = k_vs * w1 + _k_rate(opp_stat) * w2
            iso_base = iso_vs * w1 + _iso(opp_stat) * w2
        else:
            ops_base = ops_vs
            avg_base = avg_vs
            slg_base = slg_vs
            k_base = k_vs
            iso_base = iso_vs
    else:
        ops_base = ops_vs
        avg_base = avg_vs
        slg_base = slg_vs
        k_base = k_vs
        iso_base = iso_vs

    # ── Raw player-specific ratios ─────────────────────────────────────────
    ops_ratio = ops_vs / ops_base if ops_base > 0.01 else 1.0
    avg_ratio = avg_vs / avg_base if avg_base > 0.05 else 1.0
    slg_ratio = slg_vs / slg_base if slg_base > 0.05 else 1.0
    k_ratio = k_vs / k_base if k_base > 0.01 else 1.0
    iso_ratio = iso_vs / iso_base if iso_base > 0.01 else 1.0

    # ── League-average prior for this platoon direction ────────────────────
    # Favorable → ratios > 1; unfavorable → ratios < 1
    if is_favorable:
        lg_ops = LG_SPLIT_RATIOS["ops"]
        lg_avg = LG_SPLIT_RATIOS["avg"]
        lg_slg = LG_SPLIT_RATIOS["slg"]
        lg_k = LG_SPLIT_RATIOS["k_rate"]
    else:
        lg_ops = 1.0 / LG_SPLIT_RATIOS["ops"]
        lg_avg = 1.0 / LG_SPLIT_RATIOS["avg"]
        lg_slg = 1.0 / LG_SPLIT_RATIOS["slg"]
        lg_k = 1.0 / LG_SPLIT_RATIOS["k_rate"]

    # ── Bayesian shrinkage ─────────────────────────────────────────────────
    adj_ops = _bayesian_blend(ops_ratio, lg_ops, pa_vs_hand)
    adj_avg = _bayesian_blend(avg_ratio, lg_avg, pa_vs_hand)
    adj_slg = _bayesian_blend(slg_ratio, lg_slg, pa_vs_hand)
    adj_k = _bayesian_blend(k_ratio, lg_k, pa_vs_hand)
    adj_iso = _bayesian_blend(iso_ratio, lg_slg, pa_vs_hand) if iso_base > 0.01 else adj_slg

    actual_favorable = adj_ops >= 1.0
    source = "player_specific" if pa_vs_hand >= SPLIT_PA_PRIOR else "player_specific_regressed"

    desc = (
        f"✅ Favorable platoon ({batter_hand}HB vs {pitcher_hand}HP, "
        f"{int(pa_vs_hand)} PA, OPS adj {adj_ops:+.1%})"
        if actual_favorable else
        f"❌ Platoon disadvantage ({batter_hand}HB vs {pitcher_hand}HP, "
        f"{int(pa_vs_hand)} PA, OPS adj {adj_ops:+.1%})"
    )

    return {
        "adjustment": round(adj_ops, 4),
        "k_adjustment": round(adj_k, 4),
        "iso_adjustment": round(adj_iso, 4),
        "avg_adjustment": round(adj_avg, 4),
        "slg_adjustment": round(adj_slg, 4),
        "favorable": actual_favorable,
        "batter_hand": batter_hand,
        "pitcher_hand": pitcher_hand,
        "pa_sample": int(pa_vs_hand),
        "source": source,
        "description": desc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: pitcher platoon adjustment (for K/outs/runs pitcher props)
# ─────────────────────────────────────────────────────────────────────────────

def get_pitcher_platoon_adjustment(
    pitcher_mlbam_id: int,
    opp_lineup_hands: list[str],
    season: Optional[int] = None,
) -> Optional[dict]:
    """
    Compute a pitcher's platoon K-rate adjustment based on the opposing lineup's
    handedness composition and the pitcher's actual L/R split data.

    opp_lineup_hands: list of bat-side strings ('L', 'R', 'S') for the opposing lineup

    Returns a dict with 'k_adjustment' and metadata, or None if data unavailable.
    The returned dict is compatible with the `platoon` key consumed by
    project_pitcher_strikeouts().
    """
    if not pitcher_mlbam_id or not opp_lineup_hands:
        return None

    hands = [h.upper().strip() for h in opp_lineup_hands if h]
    if not hands:
        return None

    season = season or _current_season()
    splits = _get_splits_with_fallback(pitcher_mlbam_id, season, "pitching")
    if not splits:
        return None

    stat_vs_L = splits.get("vs_L")
    stat_vs_R = splits.get("vs_R")
    if not stat_vs_L and not stat_vs_R:
        return None

    # ── Pitcher K rates vs L and vs R ──────────────────────────────────────
    def _pitcher_k_rate(stat: dict) -> float:
        """BF-based K rate for pitchers."""
        bf = _safe_float(stat.get("battersFaced"))
        if bf < 1:
            return 0.0
        return _safe_float(stat.get("strikeOuts")) / bf

    k_vs_L = _pitcher_k_rate(stat_vs_L) if stat_vs_L else None
    k_vs_R = _pitcher_k_rate(stat_vs_R) if stat_vs_R else None

    bf_L = _safe_float(stat_vs_L.get("battersFaced")) if stat_vs_L else 0.0
    bf_R = _safe_float(stat_vs_R.get("battersFaced")) if stat_vs_R else 0.0
    total_bf = bf_L + bf_R

    if total_bf < 1:
        return None

    # Overall K rate baseline
    k_overall = (
        ((k_vs_L or 0.0) * bf_L + (k_vs_R or 0.0) * bf_R) / total_bf
        if total_bf > 0 else 0.0
    )
    if k_overall < 0.001:
        return None

    # ── Lineup handedness composition ──────────────────────────────────────
    # Switch hitters roughly split; treat as 50/50
    n_L = sum(1 for h in hands if h == "L") + sum(0.5 for h in hands if h == "S")
    n_R = sum(1 for h in hands if h == "R") + sum(0.5 for h in hands if h == "S")
    total_n = n_L + n_R
    if total_n < 1:
        return None

    pct_L = n_L / total_n
    pct_R = n_R / total_n

    # ── Blended expected K rate vs this lineup ─────────────────────────────
    k_L_eff = k_vs_L if k_vs_L is not None else k_overall
    k_R_eff = k_vs_R if k_vs_R is not None else k_overall

    # Apply Bayesian shrinkage for each arm split
    k_L_adj = _bayesian_blend(k_L_eff / k_overall, 1.0, bf_L) if k_overall > 0 else 1.0
    k_R_adj = _bayesian_blend(k_R_eff / k_overall, 1.0, bf_R) if k_overall > 0 else 1.0

    blended_k_adj = pct_L * k_L_adj + pct_R * k_R_adj

    # Clamp to reasonable range
    blended_k_adj = max(0.80, min(blended_k_adj, 1.20))

    pct_L_int = int(round(pct_L * 100))
    return {
        "k_adjustment": round(blended_k_adj, 4),
        "pitcher_mlbam_id": pitcher_mlbam_id,
        "pct_lhb": round(pct_L, 2),
        "k_vs_L": round(k_L_eff, 3) if k_vs_L is not None else None,
        "k_vs_R": round(k_R_eff, 3) if k_vs_R is not None else None,
        "k_overall": round(k_overall, 3),
        "source": "player_specific_pitcher",
        "description": (
            f"Pitcher platoon adj {blended_k_adj:+.1%} "
            f"({pct_L_int}% LHB lineup)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generic fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _unknown_platoon() -> dict:
    return {
        "adjustment": 1.0,
        "k_adjustment": 1.0,
        "iso_adjustment": 1.0,
        "avg_adjustment": 1.0,
        "slg_adjustment": 1.0,
        "favorable": None,
        "source": "generic",
        "pa_sample": 0,
        "description": "Unknown handedness",
    }


def _generic_platoon(batter_hand: str, pitcher_hand: str, is_favorable: bool) -> dict:
    """Generic ±8% adjustment — used as fallback when player-specific data unavailable."""
    return {
        "adjustment": 1.08 if is_favorable else 0.92,
        "k_adjustment": 0.92 if is_favorable else 1.08,
        "iso_adjustment": 1.14 if is_favorable else 0.877,
        "avg_adjustment": 1.10 if is_favorable else 0.909,
        "slg_adjustment": 1.14 if is_favorable else 0.877,
        "favorable": is_favorable,
        "batter_hand": batter_hand,
        "pitcher_hand": pitcher_hand,
        "source": "generic",
        "pa_sample": 0,
        "description": (
            f"✅ Favorable platoon ({batter_hand}HB vs {pitcher_hand}HP)"
            if is_favorable else
            f"❌ Same-side platoon ({batter_hand}HB vs {pitcher_hand}HP)"
        ),
    }
