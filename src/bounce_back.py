"""Cold-bat bounce-back signal: mean-reversion for elite hitters after outlier bad games."""

from __future__ import annotations

import math


# Batter counting-stat props where bounce-back mean-reversion applies.
# Excludes pitching props and batter_strikeouts (inverse prop).
BOUNCE_BACK_PROPS = {
    "hits", "total_bases", "home_runs", "rbis", "runs",
    "singles", "doubles", "hits_runs_rbis", "hitter_fantasy_score",
}

_BA_FLOOR = 0.280   # minimum season AVG to qualify as an "elite" hitter
_MIN_PA = 50        # minimum PA to trust the season average


def _safe_num(value, fallback: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(n) else n


def detect_bounce_back(player_stats: dict, prop_type: str) -> float:
    """Return a bounce-back multiplier for elite hitters coming off an outlier bad game.

    Applies mean-reversion logic: a high-average hitter who went hitless / produced
    nothing yesterday is statistically more likely to bounce back today.

    Args:
        player_stats: Batter profile dict.  Must contain season ``avg`` and ``pa``.
            Optionally contains a ``recent_game`` sub-dict with yesterday's box-score:
            ``hits``, ``ab``, ``total_bases``, ``h_r_rbi`` (H+R+RBI combined).
        prop_type: Internal stat name (e.g. ``"hits"``, ``"total_bases"``).

    Returns:
        A multiplier >= 1.0.  Returns 1.0 (no adjustment) when the signal does not
        apply — pitcher props, non-qualifying hitters, or no recent-game data.

    Multiplier scale (season BA drives magnitude):
        .280-.299  → 1.05  (mild regression pull)
        .300-.319  → 1.07
        .320+      → 1.10  (strong pull for elite contact hitters)
    """
    if prop_type not in BOUNCE_BACK_PROPS:
        return 1.0

    season_avg = _safe_num(player_stats.get("avg"))
    pa = _safe_num(player_stats.get("pa"))

    if season_avg < _BA_FLOOR or pa < _MIN_PA:
        return 1.0

    recent_game = player_stats.get("recent_game")
    if not recent_game:
        return 1.0

    hits = _safe_num(recent_game.get("hits"), -1.0)
    ab = _safe_num(recent_game.get("ab"))
    total_bases = _safe_num(recent_game.get("total_bases"), -1.0)
    h_r_rbi = _safe_num(recent_game.get("h_r_rbi"), -1.0)

    went_hitless_in_4plus = hits == 0.0 and ab >= 4.0
    zero_tb = total_bases == 0.0 and ab >= 2.0
    zero_production = h_r_rbi == 0.0 and ab >= 2.0

    if not (went_hitless_in_4plus or zero_tb or zero_production):
        return 1.0

    if season_avg >= 0.320:
        return 1.10
    if season_avg >= 0.300:
        return 1.07
    return 1.05
