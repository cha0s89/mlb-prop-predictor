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


class ADWIN:
    """ADaptive WINdowing drift detector.

    ADWIN automatically adjusts its window size by dropping old data when
    a statistically significant change in the mean is detected. This is
    more robust than fixed-window CUSUM for nonstationary environments
    like MLB props where the market regime can shift gradually.

    Algorithm: Maintains a growing window of observations. At each step,
    checks if splitting the window at any point reveals a significant
    difference in means (using Hoeffding bound). If so, drops the older
    half and signals drift.

    Source: Bifet & Gavalda (2007), "Learning from Time-Changing Data
    with Adaptive Windowing"
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30):
        """
        Args:
            delta: Confidence parameter (lower = fewer false alarms).
                   0.002 is conservative, 0.01 is more sensitive.
            min_window: Minimum observations before checking for drift.
        """
        self.delta = delta
        self.min_window = min_window
        self.window = []
        self.drift_detected = False
        self.drift_count = 0
        self._last_mean_before = None
        self._last_mean_after = None

    def update(self, value: float) -> bool:
        """Add an observation and check for drift.

        Args:
            value: New observation (e.g., 0/1 for correct/incorrect,
                   or a Brier score)

        Returns:
            True if drift was detected on this step
        """
        self.window.append(value)
        self.drift_detected = False

        if len(self.window) < 2 * self.min_window:
            return False

        # Check if splitting the window reveals a significant mean difference
        for i in range(self.min_window, len(self.window) - self.min_window):
            left = self.window[:i]
            right = self.window[i:]

            n_left = len(left)
            n_right = len(right)
            n = n_left + n_right

            mean_left = np.mean(left)
            mean_right = np.mean(right)
            mean_diff = abs(mean_left - mean_right)

            # Hoeffding bound for the difference of means
            # epsilon = sqrt((1/(2*m)) * ln(4*n/delta)) where m = min(n_left, n_right)
            m = min(n_left, n_right)
            epsilon_cut = np.sqrt(
                (1.0 / (2.0 * m)) * np.log(4.0 * n / self.delta)
            )

            if mean_diff > epsilon_cut:
                # Drift detected — drop the older half
                self._last_mean_before = mean_left
                self._last_mean_after = mean_right
                self.window = right  # Keep only recent data
                self.drift_detected = True
                self.drift_count += 1
                return True

        return False

    def get_mean(self) -> float:
        """Current window mean."""
        return float(np.mean(self.window)) if self.window else 0.0

    def get_window_size(self) -> int:
        """Current adaptive window size."""
        return len(self.window)

    def get_status(self) -> dict:
        """Full status for monitoring/display."""
        return {
            "window_size": len(self.window),
            "mean": round(self.get_mean(), 4) if self.window else None,
            "drift_detected": self.drift_detected,
            "total_drifts": self.drift_count,
            "last_mean_before": round(self._last_mean_before, 4) if self._last_mean_before else None,
            "last_mean_after": round(self._last_mean_after, 4) if self._last_mean_after else None,
        }


# Module-level ADWIN instances per prop type (persist across calls within a session)
_ADWIN_MONITORS = {}


def get_adwin_monitor(prop_type: str = "overall", delta: float = 0.002) -> ADWIN:
    """Get or create an ADWIN monitor for a specific prop type."""
    if prop_type not in _ADWIN_MONITORS:
        _ADWIN_MONITORS[prop_type] = ADWIN(delta=delta)
    return _ADWIN_MONITORS[prop_type]


def check_adwin_drift(predictions: list[dict]) -> dict:
    """Run ADWIN drift detection on a batch of predictions.

    Feeds predictions one-by-one into per-prop-type ADWIN monitors.
    Returns drift alerts for any prop type where drift was detected.

    Args:
        predictions: List of graded prediction dicts with:
            - stat_type: prop type
            - confidence: predicted probability
            - result: "W" or "L"

    Returns:
        dict with overall_drift, by_prop drift status, and alerts
    """
    alerts = []
    by_prop = {}

    # Overall monitor
    overall = get_adwin_monitor("overall")

    for pred in predictions:
        y = 1.0 if pred.get("result") == "W" else 0.0
        p = pred.get("confidence", 0.5)
        brier = (p - y) ** 2

        prop_type = pred.get("stat_type", "unknown")

        # Feed to overall monitor
        if overall.update(brier):
            alerts.append(
                f"OVERALL drift: Brier shifted from "
                f"{overall._last_mean_before:.3f} → {overall._last_mean_after:.3f} "
                f"(window={overall.get_window_size()})"
            )

        # Feed to prop-specific monitor
        prop_monitor = get_adwin_monitor(prop_type)
        if prop_monitor.update(brier):
            alerts.append(
                f"{prop_type} drift: Brier shifted from "
                f"{prop_monitor._last_mean_before:.3f} → {prop_monitor._last_mean_after:.3f}"
            )

    # Gather status for all active monitors
    for prop_type, monitor in _ADWIN_MONITORS.items():
        by_prop[prop_type] = monitor.get_status()

    return {
        "overall_drift": overall.drift_detected,
        "by_prop": by_prop,
        "alerts": alerts,
        "active_monitors": len(_ADWIN_MONITORS),
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

    # Log loss (cross-entropy)
    overall_logloss = _compute_log_loss(predictions)

    healthy = overall_accuracy >= 0.52 and not regime_change and overall_brier < 0.25

    return {
        "healthy": healthy,
        "overall_brier": round(overall_brier, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_logloss": round(overall_logloss, 4) if overall_logloss else None,
        "by_prop": dict(prop_health),
        "alerts": alerts,
        "regime_change": regime_change,
    }


def _compute_log_loss(predictions: list[dict]) -> Optional[float]:
    """Compute log loss (cross-entropy) over predictions."""
    if not predictions:
        return None
    eps = 1e-15
    total = 0.0
    for pred in predictions:
        y = 1.0 if pred.get("result") == "W" else 0.0
        p = max(min(pred.get("confidence", 0.5), 1 - eps), eps)
        total += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / len(predictions)


def compute_crps_binary(predicted_prob: float, outcome: int) -> float:
    """Compute CRPS for a binary prediction.

    For binary outcomes, CRPS simplifies to:
      CRPS = (predicted_prob - outcome)^2 + predicted_prob * (1 - predicted_prob) / 3

    This is a strictly proper scoring rule for distributional forecasts.
    Lower is better. CRPS=0.25 for a coin-flip model.

    Args:
        predicted_prob: P(outcome=1), i.e., probability of the positive class
        outcome: 0 or 1 (actual result)

    Returns:
        CRPS value (float)

    Source: Research Reports 3 & 4 — CRPS as proper distributional scoring rule
    """
    return (predicted_prob - outcome) ** 2 + predicted_prob * (1 - predicted_prob) / 3


def compute_crps_batch(predictions: list[dict]) -> Optional[float]:
    """Compute mean CRPS over a batch of binary predictions.

    Each prediction dict should have:
        - confidence: float (P(correct))
        - result: "W" or "L"
    """
    if not predictions:
        return None

    total = 0.0
    for pred in predictions:
        y = 1.0 if pred.get("result") == "W" else 0.0
        p = pred.get("confidence", 0.5)
        total += compute_crps_binary(p, y)

    return total / len(predictions)


def compute_ece(predictions: list[dict], n_bins: int = 10) -> dict:
    """Compute Expected Calibration Error (ECE) with reliability diagram data.

    Bins predictions by confidence, compares predicted vs actual rate.

    Args:
        predictions: list of dicts with confidence and result
        n_bins: number of bins

    Returns:
        dict with ece, max_calibration_error, and bins
    """
    if not predictions:
        return {"ece": None, "max_ce": None, "bins": []}

    bins = [[] for _ in range(n_bins)]

    for pred in predictions:
        p = pred.get("confidence", 0.5)
        y = 1.0 if pred.get("result") == "W" else 0.0
        # Bin index
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    ece = 0.0
    max_ce = 0.0
    bin_data = []

    for i, b in enumerate(bins):
        if not b:
            continue
        avg_conf = np.mean([x[0] for x in b])
        avg_acc = np.mean([x[1] for x in b])
        ce = abs(avg_conf - avg_acc)
        weight = len(b) / len(predictions)
        ece += weight * ce
        max_ce = max(max_ce, ce)
        bin_data.append({
            "bin_center": round((i + 0.5) / n_bins, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "calibration_error": round(ce, 4),
            "count": len(b),
        })

    return {
        "ece": round(ece, 4),
        "max_ce": round(max_ce, 4),
        "bins": bin_data,
    }
