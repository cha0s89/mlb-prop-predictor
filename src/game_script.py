"""
Game-script classification model.

Classifies each game into a "mode" that shifts all prop projections in a
correlated direction.  Call ``classify_game_script()`` once per game, then
multiply each player's raw projection by the relevant adjustment factor.

Scripts (priority order):
  duel        — low total (≤7) + both elite starters
  slugfest    — high total (≥10) or both weak starters
  bullpen     — at least one opener / very short starter
  blowout_risk — heavy ML favourite (≥-200)
  standard    — none of the above (all multipliers = 1.0)
"""

from __future__ import annotations

# ─── Per-script stat multipliers ────────────────────────────────────────────
# Keys match the ``stat_internal`` identifiers used throughout the predictor.

_DUEL_ADJUSTMENTS: dict[str, float] = {
    "pitcher_strikeouts":   1.08,  # elite arms → more Ks
    "pitching_outs":        1.05,  # longer outings
    "hits":                 0.92,  # suppressed offence
    "total_bases":          0.90,
    "hitter_fantasy_score": 0.92,
    "earned_runs":          0.85,
    "hits_allowed":         0.92,
    "rbis":                 0.90,
    "runs":                 0.90,
    "batter_strikeouts":    1.04,  # strong starters punch out batters more
    "hits_runs_rbis":       0.92,
}

_SLUGFEST_ADJUSTMENTS: dict[str, float] = {
    "pitcher_strikeouts":   0.92,  # starters get pulled early
    "pitching_outs":        0.90,
    "hits":                 1.08,
    "total_bases":          1.10,
    "hitter_fantasy_score": 1.12,
    "earned_runs":          1.15,
    "hits_allowed":         1.08,
    "rbis":                 1.10,
    "runs":                 1.10,
    "batter_strikeouts":    0.95,  # weaker arms throw more hittable pitches
    "hits_runs_rbis":       1.08,
}

_BULLPEN_ADJUSTMENTS: dict[str, float] = {
    "pitcher_strikeouts":   0.85,  # short outings = fewer counting Ks
    "pitching_outs":        0.80,
    "batter_strikeouts":    0.95,  # relievers vary; slight reduction
    "hits":                 1.04,  # batters tend to hit relievers harder
    "walks":                1.05,  # openers/bulk-guys issue more walks
    "walks_allowed":        1.05,
}

# Blowout adjustments are ASYMMETRIC: the heavy favourite's pitcher is pulled
# early (outs/Ks DOWN), while the underdog's hitters pad stats in garbage time.
# These two dicts are returned separately as home_adjustments / away_adjustments
# so the caller can apply the right one per team.

_BLOWOUT_FAVOURITE_ADJUSTMENTS: dict[str, float] = {
    "pitcher_strikeouts":   0.92,  # pulled when ahead by a lot
    "pitching_outs":        0.92,
}

_BLOWOUT_UNDERDOG_ADJUSTMENTS: dict[str, float] = {
    "hits":                 1.05,  # garbage-time at-bats inflate hits
    "hitter_fantasy_score": 1.03,
    "hits_runs_rbis":       1.04,
}


# ─── Public API ─────────────────────────────────────────────────────────────

def classify_game_script(
    home_pitcher_profile: dict,
    away_pitcher_profile: dict,
    vegas_total: float,
    home_moneyline: int,
    away_moneyline: int,
    home_bullpen_usage: dict | None = None,  # reserved for future use
    away_bullpen_usage: dict | None = None,
) -> dict:
    """Classify a game into a script mode and return per-team prop multipliers.

    Parameters
    ----------
    home_pitcher_profile:
        Pitcher stat dict for the home starter.  Relevant keys:
        ``era``, ``fip``, ``xfip`` (floats), ``gs`` (int, career game starts),
        ``ip`` (float, season innings pitched).
    away_pitcher_profile:
        Same structure for the away starter.
    vegas_total:
        Sharp-book consensus over/under total.  Pass 0 or ``None`` if unknown.
    home_moneyline:
        American-odds moneyline for the home team (negative = favourite).
        Pass 0 or ``None`` if unknown.
    away_moneyline:
        American-odds moneyline for the away team.
    home_bullpen_usage / away_bullpen_usage:
        Optional bullpen fatigue/usage dicts (reserved; not yet used).

    Returns
    -------
    dict
        ``script``           : str  — one of the five script labels
        ``confidence``       : float 0-1
        ``adjustments``      : dict[str, float] — multipliers for the home team
        ``home_adjustments`` : dict[str, float] — same as ``adjustments``
        ``away_adjustments`` : dict[str, float] — multipliers for the away team
        ``reason``           : str  — human-readable explanation
    """
    hp = home_pitcher_profile or {}
    ap = away_pitcher_profile or {}

    # ── Helper predicates ────────────────────────────────────────────────────

    def _is_elite(profile: dict) -> bool:
        """True if the starter meets at least one elite-arm threshold."""
        era  = profile.get("era")
        fip  = profile.get("fip")
        xfip = profile.get("xfip")
        return (
            (era  is not None and era  < 3.50)
            or (fip  is not None and fip  < 3.50)
            or (xfip is not None and xfip < 3.80)
        )

    def _is_weak(profile: dict) -> bool:
        """True if the starter has a poor ERA (>4.50)."""
        era = profile.get("era")
        return era is not None and era > 4.50

    def _is_opener(profile: dict) -> bool:
        """True if the starter has very few career GS or very few season IP —
        signals an opener or bulk-reliever usage."""
        gs = profile.get("gs")
        ip = profile.get("ip")
        return (gs is not None and gs < 5) or (ip is not None and ip < 20)

    total_ok = vegas_total is not None and vegas_total > 0

    # Moneyline absolute values (0 if unavailable)
    home_ml_abs = abs(home_moneyline) if home_moneyline else 0
    away_ml_abs = abs(away_moneyline) if away_moneyline else 0
    home_is_fav = home_moneyline is not None and home_moneyline < 0 and home_ml_abs >= 200
    away_is_fav = away_moneyline is not None and away_moneyline < 0 and away_ml_abs >= 200

    # ── Classification (priority order) ─────────────────────────────────────

    if total_ok and vegas_total <= 7.0 and _is_elite(hp) and _is_elite(ap):
        script  = "duel"
        # Higher confidence when total is further below 7 and both arms are elite
        conf    = min(0.95, 0.75 + (7.0 - vegas_total) * 0.08)
        home_adj = dict(_DUEL_ADJUSTMENTS)
        away_adj = dict(_DUEL_ADJUSTMENTS)
        reason  = (
            f"Both starters are elite (ERA/FIP/xFIP thresholds met) "
            f"and Vegas total is {vegas_total} (≤7.0)"
        )

    elif total_ok and (vegas_total >= 10.0 or (_is_weak(hp) and _is_weak(ap))):
        script  = "slugfest"
        if total_ok and vegas_total >= 10.0:
            conf = min(0.90, 0.70 + (vegas_total - 10.0) * 0.04)
            reason = f"High Vegas total ({vegas_total} ≥10.0)"
        else:
            conf   = 0.65
            reason = "Both starters have ERA >4.50 (weak pitching matchup)"
        home_adj = dict(_SLUGFEST_ADJUSTMENTS)
        away_adj = dict(_SLUGFEST_ADJUSTMENTS)

    elif _is_opener(hp) or _is_opener(ap):
        script   = "bullpen"
        conf     = 0.70
        home_adj = dict(_BULLPEN_ADJUSTMENTS)
        away_adj = dict(_BULLPEN_ADJUSTMENTS)
        side     = "home" if _is_opener(hp) else "away"
        reason   = f"{'Home' if side == 'home' else 'Away'} starter qualifies as opener/short-start (GS<5 or IP<20)"

    elif home_is_fav or away_is_fav:
        script   = "blowout_risk"
        max_ml   = max(home_ml_abs, away_ml_abs)
        conf     = min(0.80, 0.60 + (max_ml - 200) * 0.001)
        # Asymmetric: favourite pitcher pulled early; underdog hitters inflate stats
        if home_is_fav:
            home_adj = dict(_BLOWOUT_FAVOURITE_ADJUSTMENTS)
            away_adj = dict(_BLOWOUT_UNDERDOG_ADJUSTMENTS)
            reason = (
                f"Home team is a heavy favourite (ML {home_moneyline}). "
                f"Home pitcher may be pulled early; away hitters may pad garbage-time stats."
            )
        else:
            home_adj = dict(_BLOWOUT_UNDERDOG_ADJUSTMENTS)
            away_adj = dict(_BLOWOUT_FAVOURITE_ADJUSTMENTS)
            reason = (
                f"Away team is a heavy favourite (ML {away_moneyline}). "
                f"Away pitcher may be pulled early; home hitters may pad garbage-time stats."
            )

    else:
        script   = "standard"
        conf     = 1.0
        home_adj = {}
        away_adj = {}
        reason   = "No extreme game-script signal detected; standard projections apply."

    return {
        "script":           script,
        "confidence":       round(min(1.0, conf), 3),
        "adjustments":      home_adj,   # default = home-team view
        "home_adjustments": home_adj,
        "away_adjustments": away_adj,
        "reason":           reason,
    }
