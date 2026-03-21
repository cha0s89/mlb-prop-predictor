"""
Regime change and drift detection for the MLB prop predictor.
Monitors rolling accuracy by prop type and alerts when performance degrades.
Uses CUSUM (cumulative sum) control charts for change detection.
"""

import numpy as np
from collections import defaultdict
from typing import Optional


def cusum_detect(values: list[float], target: float = 0.0,
                 threshold: float = 4.0, drift: float = 0.5) -> dict:
    """Run CUSUM (Cumulative Sum) change detection on a sequence.

    Detects both positive and negative shifts from the target value.

    Args:
        values: Time-ordered sequence of observations (e.g., daily Brier scores)
        target: Expected mean value under no-change hypothesis
        drift: Allowance parameter (k) — how much deviation to tolerate before accumulating
        threshold: Decision boundary (h) — CUSUM triggers when cumulative sum exceeds this

    Returns:
        dict with:
            - detected: bool — whether a change was detected
            - change_point: int or None — index where change was detected
            - direction: "up" or "down" or None
            - cusum_pos: list of positive CUSUM values
            - cusum_neg: list of negative CUSUM values
    """
    n = len(values)
    if n < 5:
        return {"detected": False, "change_point": None, "direction": None,
                "cusum_pos": [], "cusum_neg": []}

    cusum_pos = [0.0]
    cusum_neg = [0.0]
    change_point = None
    direction = None

    for i, x in enumerate(values):
        s_pos = max(0, cusum_pos[-1] + (x - target) - drift)
        s_neg = max(0, cusum_neg[-1] - (x - target) - drift)
        cusum_pos.append(s_pos)
        cusum_neg.append(s_neg)

        if s_pos > threshold and change_point is None:
            change_point = i
            direction = "up"
        elif s_neg > threshold and change_point is None:
            change_point = i
            direction = "down"

    return {
        "detected": change_point is not None,
        "change_point": change_point,
        "direction": direction,
        "cusum_pos": cusum_pos[1:],
        "cusum_neg": cusum_neg[1:],
    }


def rolling_brier(predictions: list[dict], window: int = 50) -> list[float]:
    """Compute rolling Brier score over a window of predictions.

    Each prediction dict should have:
        - confidence: float (our predicted P(correct))
        - result: "W" or "L"

    Returns list of rolling Brier scores.
    """
    if len(predictions) < window:
        return []

    scores = []
    for pred in predictions:
        y = 1.0 if pred.get("result") == "W" else 0.0
        p = pred.get("confidence", 0.5)
        scores.append((p - y) ** 2)

    rolling = []
    for i in range(window, len(scores) + 1):
        window_scores = scores[i - window:i]
        rolling.append(np.mean(window_scores))

    return rolling


def check_model_health(predictions: list[dict], min_sample: int = 50) -> dict:
    """Run full health check on recent predictions.

    Returns:
        dict with:
            - healthy: bool
            - overall_brier: float
            - overall_accuracy: float
            - by_prop: dict of per-prop-type health
            - alerts: list of alert strings
            - regime_change: bool — detected by CUSUM
    """
    if len(predictions) < min_sample:
        return {
            "healthy": True,
            "overall_brier": None,
            "overall_accuracy": None,
            "by_prop": {},
            "alerts": [],
            "regime_change": False,
            "message": f"Need {min_sample} predictions for health check ({len(predictions)} so far)"
        }

    alerts = []
    by_prop = defaultdict(lambda: {"predictions": [], "wins": 0, "total": 0})

    total_brier = 0
    total_wins = 0

    for pred in predictions:
        y = 1.0 if pred.get("result") == "W" else 0.0
        p = pred.get("confidence", 0.5)
        brier = (p - y) ** 2
        total_brier += brier
        total_wins += int(y)

        prop_type = pred.get("stat_type", "unknown")
        by_prop[prop_type]["predictions"].append(pred)
        by_prop[prop_type]["total"] += 1
        by_prop[prop_type]["wins"] += int(y)

    n = len(predictions)
    overall_brier = total_brier / n
    overall_accuracy = total_wins / n

    # Check overall health
    if overall_accuracy < 0.48:
        alerts.append(f"CRITICAL: Overall accuracy {overall_accuracy:.1%} is below kill switch threshold (48%)")
    elif overall_accuracy < 0.52:
        alerts.append(f"WARNING: Overall accuracy {overall_accuracy:.1%} is below breakeven (54.2%)")

    if overall_brier > 0.26:
        alerts.append(f"WARNING: Brier score {overall_brier:.4f} worse than coin flip (0.25)")

    # Check per-prop health
    prop_health = {}
    for prop_type, data in by_prop.items():
        if data["total"] < 10:
            continue
        prop_acc = data["wins"] / data["total"]
        prop_health[prop_type] = {
            "accuracy": round(prop_acc, 4),
            "total": data["total"],
            "wins": data["wins"],
        }
        if prop_acc < 0.45:
            alerts.append(f"ALERT: {prop_type} accuracy {prop_acc:.1%} is critically low ({data['total']} picks)")

    # CUSUM regime change detection on rolling Brier
    brier_values = rolling_brier(predictions, window=min(30, n // 3))
    regime_change = False
    if brier_values:
        # Target Brier = 0.22 (decent model), detect if it drifts above
        cusum_result = cusum_detect(
            brier_values,
            target=0.22,
            threshold=3.0,
            drift=0.02
        )
        regime_change = cusum_result["detected"]
        if regime_change:
            alerts.append(
                f"REGIME CHANGE detected at prediction ~{cusum_result['change_point']} "
                f"(Brier shifting {cusum_result['direction']})"
            )

    healthy = overall_accuracy >= 0.52 and not regime_change and overall_brier < 0.25

    return {
        "healthy": healthy,
        "overall_brier": round(overall_brier, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "by_prop": dict(prop_health),
        "alerts": alerts,
        "regime_change": regime_change,
    }
