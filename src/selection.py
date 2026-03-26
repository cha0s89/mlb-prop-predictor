"""
Selection policy helpers for model confidence floors.

These helpers keep the app's displayed/eligible picks aligned with the
confidence-floor strategy stored in the weights file.
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_CONFIDENCE_FLOOR = 0.60


def normalize_stat_key(stat_value: Any) -> str:
    """Normalize a stat label or internal key to snake_case."""
    text = str(stat_value or "").strip().lower()
    text = text.replace("+", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def floor_key(stat_internal: Any, direction: Any) -> str:
    """Build the weights-file key for a prop/direction floor."""
    stat_key = normalize_stat_key(stat_internal)
    pick_key = str(direction or "").strip().lower()
    return f"{stat_key}_{pick_key}" if stat_key and pick_key else ""


def get_confidence_floor(weights: dict | None, stat_internal: Any, direction: Any,
                         default: float = DEFAULT_CONFIDENCE_FLOOR) -> float:
    """Return the configured confidence floor for a prop/direction."""
    weights = weights or {}
    floors = weights.get("per_prop_confidence_floors", {}) or {}
    key = floor_key(stat_internal, direction)
    try:
        return float(floors.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def score_data_certainty(pred: dict) -> dict:
    """Score a pick's data certainty based on what information was available.

    Returns dict with:
        certainty_score: float 0.0-1.0 (1.0 = all data confirmed)
        certainty_label: str — "high" | "medium" | "low"
        certainty_flags: list[str] — what's missing or uncertain
        confidence_cap: float — max confidence allowed given certainty

    The certainty score affects how much we trust the prediction.
    Low-certainty picks should have their confidence capped.
    """
    score = 1.0
    flags = []

    # Player data quality
    if not pred.get("has_player_data", True):
        score -= 0.30
        flags.append("no_player_stats")
    if not pred.get("has_opp_data", True):
        score -= 0.10
        flags.append("no_opponent_data")

    # Lineup confirmation
    if not pred.get("has_lineup_pos"):
        score -= 0.10
        flags.append("lineup_not_confirmed")

    # Park/team identification
    if not pred.get("park_team"):
        score -= 0.05
        flags.append("no_park_factor")

    # Weather data
    if not pred.get("weather_mult") and not pred.get("weather"):
        score -= 0.05
        flags.append("no_weather_data")

    # Umpire assignment
    ump = pred.get("ump")
    if not ump or (isinstance(ump, dict) and not ump.get("known")):
        score -= 0.05
        flags.append("umpire_unknown")

    # BvP matchup (batter props only)
    is_pitcher_prop = pred.get("stat_internal", "") in {
        "pitcher_strikeouts", "pitching_outs", "earned_runs",
        "walks_allowed", "hits_allowed",
    }
    if not is_pitcher_prop and not pred.get("bvp"):
        score -= 0.05
        flags.append("no_bvp_matchup")

    # Platoon data
    if not pred.get("platoon"):
        score -= 0.05
        flags.append("no_platoon_data")

    # Injury status
    if pred.get("injury_status") == "day-to-day":
        score -= 0.15
        flags.append("player_day_to_day")

    score = max(0.0, round(score, 2))

    if score >= 0.85:
        label = "high"
        cap = 1.0  # No cap
    elif score >= 0.65:
        label = "medium"
        cap = 0.80  # Cap at B-grade
    else:
        label = "low"
        cap = 0.65  # Cap at C-grade

    return {
        "certainty_score": score,
        "certainty_label": label,
        "certainty_flags": flags,
        "confidence_cap": cap,
    }


def annotate_prediction_floor(pred: dict, weights: dict | None,
                              default: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    """Annotate a prediction dict with its configured confidence floor."""
    floor = get_confidence_floor(
        weights,
        pred.get("stat_internal") or pred.get("stat_type"),
        pred.get("pick"),
        default=default,
    )
    confidence = float(pred.get("confidence", 0.0) or 0.0)
    pred["confidence_floor"] = floor
    pred["meets_conf_floor"] = confidence >= floor
    return pred
