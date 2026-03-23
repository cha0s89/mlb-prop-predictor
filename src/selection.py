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
