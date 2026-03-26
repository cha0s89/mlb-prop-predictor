"""Prediction canonicalization and deduplication helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


PROP_ALIASES = {
    "hitter_strikeouts": "batter_strikeouts",
}


def canonical_prop_type(prop_type: str | None) -> str:
    value = (prop_type or "").strip()
    return PROP_ALIASES.get(value, value)


def _priority_tuple(pred: Dict) -> Tuple[float, float, int]:
    confidence = float(pred.get("confidence", 0.0) or 0.0)
    edge = abs(float(pred.get("edge", 0.0) or 0.0))
    model_probability = float(pred.get("model_probability", 0.0) or 0.0)
    return (confidence, edge, model_probability)


def dedupe_predictions(predictions: Iterable[Dict]) -> List[Dict]:
    """Collapse conflicting duplicate props to the strongest single side.

    Keyed by game_date/player/prop/line/line_type so the same prop cannot survive
    twice with opposite directions because of duplicate upstream rows.
    """
    deduped: dict[Tuple[str, str, str, float, str], Dict] = {}
    for raw_pred in predictions:
        pred = dict(raw_pred)
        prop_type = canonical_prop_type(pred.get("stat_internal") or pred.get("stat_type"))
        pred["stat_internal"] = prop_type
        if "stat_type" in pred and not pred.get("stat_type"):
            pred["stat_type"] = prop_type
        elif pred.get("stat_type") == "hitter_strikeouts":
            pred["stat_type"] = prop_type

        key = (
            str(pred.get("game_date", "")),
            str(pred.get("player_name", "")),
            prop_type,
            float(pred.get("line", 0.0) or 0.0),
            str(pred.get("line_type", "standard") or "standard"),
        )
        current = deduped.get(key)
        if current is None or _priority_tuple(pred) > _priority_tuple(current):
            deduped[key] = pred

    return list(deduped.values())
