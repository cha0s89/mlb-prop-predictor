"""
Regime change and drift detection for the MLB prop predictor.
Monitors rolling accuracy by prop type and alerts when performance degrades.
Uses CUSUM (cumulative sum) control charts for change detection.
"""

import json
import logging
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# RegimeDetector — persistent CUSUM-based drift detection per prop type
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Persistent CUSUM-based regime change detector per prop type.

    Monitors rolling prediction accuracy and flags when performance degrades
    significantly from the expected baseline, indicating a market regime shift.

    State is stored in SQLite so it survives app restarts.  When drift is
    detected a JSON flag file is written that the offline tuner can pick up.

    Thresholds are expressed in σ units (standard deviations of a single
    Bernoulli prediction under the expected accuracy):
      • warning       — CUSUM statistic ≥ warning_sigma  (default 2σ)
      • drift_detected — CUSUM statistic ≥ alert_sigma   (default 3σ)
    """

    _DRIFT_FLAG_DIR = "."  # directory for flag files; override in tests

    def __init__(
        self,
        db_path: str = "regime_state.db",
        window_size: int = 200,
        expected_accuracy: float = 0.55,
        warning_sigma: float = 2.0,
        alert_sigma: float = 3.0,
        cusum_k: float = 0.5,
    ):
        """
        Args:
            db_path: Path to the SQLite file for persistent state.
            window_size: Rolling window of predictions per prop type (max 200).
            expected_accuracy: Baseline accuracy the model should achieve.
            warning_sigma: CUSUM threshold (in σ) that triggers "warning".
            alert_sigma: CUSUM threshold (in σ) that triggers "drift_detected".
            cusum_k: CUSUM allowance (k) — tolerance before accumulating signal.
                     0.5 σ is the standard choice for detecting 1σ shifts.
        """
        self.db_path = db_path
        self.window_size = window_size
        self.expected_accuracy = expected_accuracy
        self.warning_sigma = warning_sigma
        self.alert_sigma = alert_sigma
        self.cusum_k = cusum_k

        # σ of a single Bernoulli(expected_accuracy) trial
        self._sigma = float(np.sqrt(expected_accuracy * (1.0 - expected_accuracy)))

        self._init_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS regime_predictions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    prop_type     TEXT    NOT NULL,
                    predicted_prob REAL   NOT NULL,
                    actual_outcome INTEGER NOT NULL,
                    ts            TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rp_prop
                    ON regime_predictions(prop_type, id);

                CREATE TABLE IF NOT EXISTS regime_drift_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    prop_type    TEXT    NOT NULL,
                    status       TEXT    NOT NULL,
                    cusum_pos    REAL,
                    cusum_neg    REAL,
                    accuracy     REAL,
                    ts           TEXT    NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0
                );
            """)

    def _get_window(self, prop_type: str) -> list[tuple[float, int]]:
        """Return the last ``window_size`` (predicted_prob, outcome) pairs."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT predicted_prob, actual_outcome
                FROM regime_predictions
                WHERE prop_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (prop_type, self.window_size),
            ).fetchall()
        return list(reversed(rows))  # chronological order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prop_type: str, predicted_prob: float, actual_outcome: int) -> str:
        """Feed one graded prediction and return the current drift status.

        Args:
            prop_type: Prop category, e.g. ``"strikeouts"``, ``"hits"``.
            predicted_prob: Model's predicted probability (0–1).
            actual_outcome: 1 if the prediction was correct, 0 if wrong.

        Returns:
            Current status string: ``"stable"``, ``"warning"``, or
            ``"drift_detected"``.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO regime_predictions (prop_type, predicted_prob, actual_outcome, ts) "
                "VALUES (?, ?, ?, ?)",
                (prop_type, float(predicted_prob), int(actual_outcome), ts),
            )
        return self.check_drift(prop_type)

    def check_drift(self, prop_type: str) -> str:
        """Return drift status for *prop_type* without adding new data.

        Returns:
            ``"stable"``         — no significant drift.
            ``"warning"``        — CUSUM ≥ 2σ (degrading performance).
            ``"drift_detected"`` — CUSUM ≥ 3σ (regime change confirmed).
        """
        window = self._get_window(prop_type)
        stats = self._compute_cusum(window)

        if stats["n"] < 10:
            return "stable"

        max_cusum = max(stats["cusum_neg"], stats["cusum_pos"])

        if max_cusum >= self.alert_sigma:
            self._log_drift(prop_type, "drift_detected", stats)
            return "drift_detected"
        if max_cusum >= self.warning_sigma:
            self._log_drift(prop_type, "warning", stats)
            return "warning"
        return "stable"

    def get_all_statuses(self) -> dict[str, str]:
        """Return ``{prop_type: status}`` for every prop type seen."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT prop_type FROM regime_predictions"
            ).fetchall()
        return {row[0]: self.check_drift(row[0]) for row in rows}

    def reset(self, prop_type: str) -> None:
        """Acknowledge and clear drift state for *prop_type*.

        Marks all drift-log entries as acknowledged, deletes stored
        predictions (so the CUSUM starts fresh), and removes the flag file.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE regime_drift_log SET acknowledged = 1 "
                "WHERE prop_type = ? AND acknowledged = 0",
                (prop_type,),
            )
            conn.execute(
                "DELETE FROM regime_predictions WHERE prop_type = ?",
                (prop_type,),
            )
        flag = os.path.join(self._DRIFT_FLAG_DIR, f"drift_flag_{prop_type}.json")
        try:
            os.remove(flag)
        except FileNotFoundError:
            pass

    def get_stats(self, prop_type: str) -> dict:
        """Detailed diagnostics for *prop_type*."""
        window = self._get_window(prop_type)
        s = self._compute_cusum(window)
        status = self.check_drift(prop_type)
        return {
            "prop_type": prop_type,
            "status": status,
            "n_predictions": s["n"],
            "accuracy": round(s["accuracy"], 4) if s["accuracy"] is not None else None,
            "expected_accuracy": self.expected_accuracy,
            "cusum_neg": round(s["cusum_neg"], 3),
            "cusum_pos": round(s["cusum_pos"], 3),
            "warning_threshold": self.warning_sigma,
            "alert_threshold": self.alert_sigma,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cusum(self, window: list[tuple[float, int]]) -> dict:
        """Compute normalised CUSUM statistics over a prediction window.

        Each residual is normalised by σ so the CUSUM statistic is in
        standard-deviation units.  We track:
          • cusum_neg — accumulates negative deviations (accuracy dropping)
          • cusum_pos — accumulates positive deviations (accuracy rising)
        """
        n = len(window)
        if n < 10:
            return {"cusum_pos": 0.0, "cusum_neg": 0.0, "accuracy": None, "n": n}

        outcomes = [o for _, o in window]
        accuracy = float(np.mean(outcomes))

        s_pos = 0.0
        s_neg = 0.0
        for _, outcome in window:
            # Normalised residual: positive = hit, negative = miss
            residual = (outcome - self.expected_accuracy) / self._sigma
            s_neg = max(0.0, s_neg - residual - self.cusum_k)
            s_pos = max(0.0, s_pos + residual - self.cusum_k)

        return {"cusum_pos": s_pos, "cusum_neg": s_neg, "accuracy": accuracy, "n": n}

    def _log_drift(self, prop_type: str, status: str, stats: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO regime_drift_log "
                "(prop_type, status, cusum_pos, cusum_neg, accuracy, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    prop_type,
                    status,
                    stats.get("cusum_pos"),
                    stats.get("cusum_neg"),
                    stats.get("accuracy"),
                    ts,
                ),
            )

        acc = stats.get("accuracy")
        s_neg = stats.get("cusum_neg", 0.0)
        if status == "drift_detected":
            logger.warning(
                "REGIME DRIFT DETECTED: %s | accuracy=%.3f (expected=%.3f) | "
                "CUSUM_neg=%.2f >= alert threshold=%.1f",
                prop_type,
                acc if acc is not None else 0.0,
                self.expected_accuracy,
                s_neg,
                self.alert_sigma,
            )
            self._write_drift_flag(prop_type, stats)
        else:
            logger.info(
                "REGIME WARNING: %s | accuracy=%.3f | CUSUM_neg=%.2f >= warning=%.1f",
                prop_type,
                acc if acc is not None else 0.0,
                s_neg,
                self.warning_sigma,
            )

    def _write_drift_flag(self, prop_type: str, stats: dict) -> None:
        """Write a JSON flag file that the offline tuner can poll for."""
        flag_path = os.path.join(
            self._DRIFT_FLAG_DIR, f"drift_flag_{prop_type}.json"
        )
        payload = {
            "prop_type": prop_type,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "accuracy": stats.get("accuracy"),
            "cusum_neg": stats.get("cusum_neg"),
            "cusum_pos": stats.get("cusum_pos"),
            "n_predictions": stats.get("n"),
        }
        try:
            with open(flag_path, "w") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.debug("Could not write drift flag for %s: %s", prop_type, exc)
