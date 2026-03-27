"""Tests for the regime detection (drift) module."""

import os
import tempfile
import pytest

from src.drift import (
    RegimeDetector,
    cusum_detect,
    ADWIN,
    rolling_brier,
    check_model_health,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(tmp_path, **kwargs) -> RegimeDetector:
    """Return a RegimeDetector backed by a temp DB with flag files in tmp_path."""
    db = str(tmp_path / "regime.db")
    det = RegimeDetector(db_path=db, **kwargs)
    det._DRIFT_FLAG_DIR = str(tmp_path)
    return det


def _feed(detector: RegimeDetector, prop_type: str, outcomes: list[int],
          prob: float = 0.6) -> list[str]:
    """Feed a list of binary outcomes and return each update's status."""
    return [detector.update(prop_type, prob, o) for o in outcomes]


# ---------------------------------------------------------------------------
# cusum_detect (standalone function)
# ---------------------------------------------------------------------------

class TestCusumDetect:
    def test_returns_no_detect_on_short_series(self):
        result = cusum_detect([0.1, 0.2], target=0.0)
        assert result["detected"] is False

    def test_detects_upward_shift(self):
        # Feed a series that clearly shifts upward
        normal = [0.0] * 20
        shifted = [5.0] * 20
        result = cusum_detect(normal + shifted, target=0.0, threshold=4.0, drift=0.5)
        assert result["detected"] is True
        assert result["direction"] == "up"

    def test_detects_downward_shift(self):
        normal = [0.0] * 20
        shifted = [-5.0] * 20
        result = cusum_detect(normal + shifted, target=0.0, threshold=4.0, drift=0.5)
        assert result["detected"] is True
        assert result["direction"] == "down"

    def test_stable_series_not_detected(self):
        values = [0.01 * i % 0.1 for i in range(50)]  # tiny fluctuations
        result = cusum_detect(values, target=0.05, threshold=10.0, drift=0.5)
        assert result["detected"] is False

    def test_change_point_reasonable(self):
        normal = [0.0] * 30
        shifted = [3.0] * 30
        result = cusum_detect(normal + shifted, target=0.0, threshold=4.0, drift=0.5)
        assert result["detected"] is True
        # Change point should be in the second half
        assert result["change_point"] >= 20


# ---------------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------------

class TestADWIN:
    def test_no_drift_on_constant_stream(self):
        adwin = ADWIN(delta=0.002, min_window=20)
        drifts = [adwin.update(0.5) for _ in range(100)]
        assert not any(drifts)

    def test_detects_mean_shift(self):
        adwin = ADWIN(delta=0.002, min_window=20)
        for _ in range(60):
            adwin.update(0.2)
        drift_found = False
        for _ in range(60):
            if adwin.update(0.8):
                drift_found = True
                break
        assert drift_found

    def test_get_mean_after_drift(self):
        adwin = ADWIN(delta=0.002, min_window=20)
        for _ in range(60):
            adwin.update(0.2)
        for _ in range(60):
            adwin.update(0.8)
        # After drift the window should be tracking the new mean (~0.8)
        assert adwin.get_mean() > 0.5

    def test_status_keys(self):
        adwin = ADWIN()
        adwin.update(0.5)
        status = adwin.get_status()
        for key in ("window_size", "mean", "drift_detected", "total_drifts"):
            assert key in status


# ---------------------------------------------------------------------------
# RegimeDetector — stable behaviour
# ---------------------------------------------------------------------------

class TestRegimeDetectorStable:
    def test_stable_when_accuracy_matches_expected(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55, window_size=50)
        # Feed outcomes with ~55% win rate
        outcomes = [1, 1, 1, 0, 0] * 20  # exactly 60%, close to expected
        statuses = _feed(det, "hits", outcomes)
        # With only 100 observations at reasonable accuracy, should stay stable
        final = det.check_drift("hits")
        assert final in ("stable", "warning")

    def test_not_enough_data_returns_stable(self, tmp_path):
        det = _make_detector(tmp_path)
        _feed(det, "home_runs", [0] * 5)  # only 5 predictions
        assert det.check_drift("home_runs") == "stable"

    def test_get_all_statuses_empty(self, tmp_path):
        det = _make_detector(tmp_path)
        assert det.get_all_statuses() == {}

    def test_get_all_statuses_returns_seen_props(self, tmp_path):
        det = _make_detector(tmp_path)
        _feed(det, "strikeouts", [1] * 20)
        _feed(det, "hits", [1] * 20)
        statuses = det.get_all_statuses()
        assert "strikeouts" in statuses
        assert "hits" in statuses


# ---------------------------------------------------------------------------
# RegimeDetector — drift detection
# ---------------------------------------------------------------------------

class TestRegimeDetectorDrift:
    def _force_drift(self, det: RegimeDetector, prop_type: str):
        """Feed enough consecutive losses to trigger drift_detected."""
        # All zeros (misses) will push the CUSUM_neg well above any threshold
        for _ in range(120):
            det.update(prop_type, 0.65, 0)

    def test_warning_before_drift(self, tmp_path):
        det = _make_detector(
            tmp_path,
            expected_accuracy=0.55,
            warning_sigma=2.0,
            alert_sigma=3.0,
            cusum_k=0.5,
        )
        # Feed 30 stable + 30 bad predictions
        _feed(det, "strikeouts", [1] * 30)
        statuses = _feed(det, "strikeouts", [0] * 50)
        # At some point we should have seen a warning or drift
        assert any(s in ("warning", "drift_detected") for s in statuses)

    def test_drift_detected_on_sustained_losses(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        self._force_drift(det, "rbis")
        assert det.check_drift("rbis") == "drift_detected"

    def test_drift_flag_file_written(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        self._force_drift(det, "strikeouts")
        flag = tmp_path / "drift_flag_strikeouts.json"
        assert flag.exists(), "drift flag file should be written on drift_detected"

    def test_flag_file_contents(self, tmp_path):
        import json
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        self._force_drift(det, "hits")
        with open(tmp_path / "drift_flag_hits.json") as fh:
            payload = json.load(fh)
        assert payload["prop_type"] == "hits"
        assert "detected_at" in payload
        assert payload["n_predictions"] > 0


# ---------------------------------------------------------------------------
# RegimeDetector — reset
# ---------------------------------------------------------------------------

class TestRegimeDetectorReset:
    def test_reset_clears_drift(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        for _ in range(120):
            det.update("rbis", 0.65, 0)
        assert det.check_drift("rbis") == "drift_detected"

        det.reset("rbis")
        assert det.check_drift("rbis") == "stable"

    def test_reset_removes_flag_file(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        for _ in range(120):
            det.update("hits", 0.65, 0)
        flag = tmp_path / "drift_flag_hits.json"
        assert flag.exists()

        det.reset("hits")
        assert not flag.exists()

    def test_reset_nonexistent_prop_is_noop(self, tmp_path):
        det = _make_detector(tmp_path)
        det.reset("nonexistent")  # Should not raise

    def test_reset_restarts_accumulation(self, tmp_path):
        det = _make_detector(tmp_path, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        # Trigger drift
        for _ in range(120):
            det.update("strikeouts", 0.65, 0)
        assert det.check_drift("strikeouts") == "drift_detected"
        det.reset("strikeouts")
        # After reset, feeding a realistic win rate (~58%) should stay stable
        # (alternating wins/losses near baseline — not a sustained streak)
        outcomes = ([1, 1, 1, 0, 0] * 6)  # 60% wins, 30 predictions
        for o in outcomes:
            det.update("strikeouts", 0.65, o)
        assert det.check_drift("strikeouts") == "stable"


# ---------------------------------------------------------------------------
# RegimeDetector — rolling window cap
# ---------------------------------------------------------------------------

class TestRegimeDetectorWindow:
    def test_window_capped_at_window_size(self, tmp_path):
        det = _make_detector(tmp_path, window_size=50, expected_accuracy=0.55,
                             warning_sigma=2.0, alert_sigma=3.0)
        # Feed 200 bad predictions (all losses)
        for _ in range(200):
            det.update("hits", 0.65, 0)
        assert det.check_drift("hits") == "drift_detected"
        # Feed 60 predictions at a realistic ~60% win rate (3W/2L repeating).
        # Each 5-step cycle resets s_pos to 0, so no upward drift either.
        recovery = [1, 1, 1, 0, 0] * 12
        for o in recovery:
            det.update("hits", 0.65, o)
        # The 50-prediction window now contains mostly recovery data,
        # so the downward drift signal should have collapsed.
        stats = det.get_stats("hits")
        assert stats["cusum_neg"] < det.alert_sigma, (
            f"cusum_neg={stats['cusum_neg']} still above alert threshold after recovery"
        )


# ---------------------------------------------------------------------------
# RegimeDetector — persistence
# ---------------------------------------------------------------------------

class TestRegimeDetectorPersistence:
    def test_predictions_survive_restart(self, tmp_path):
        db = str(tmp_path / "regime.db")
        det1 = RegimeDetector(db_path=db, expected_accuracy=0.55,
                              warning_sigma=2.0, alert_sigma=3.0)
        det1._DRIFT_FLAG_DIR = str(tmp_path)
        for _ in range(120):
            det1.update("home_runs", 0.65, 0)

        # Create a new detector pointing at the same DB
        det2 = RegimeDetector(db_path=db, expected_accuracy=0.55,
                              warning_sigma=2.0, alert_sigma=3.0)
        det2._DRIFT_FLAG_DIR = str(tmp_path)
        assert det2.check_drift("home_runs") == "drift_detected"


# ---------------------------------------------------------------------------
# RegimeDetector — get_stats
# ---------------------------------------------------------------------------

class TestRegimeDetectorGetStats:
    def test_get_stats_keys(self, tmp_path):
        det = _make_detector(tmp_path)
        _feed(det, "walks", [1] * 20)
        stats = det.get_stats("walks")
        for key in ("prop_type", "status", "n_predictions", "accuracy",
                    "cusum_neg", "cusum_pos", "warning_threshold", "alert_threshold"):
            assert key in stats, f"missing key: {key}"

    def test_get_stats_accuracy(self, tmp_path):
        det = _make_detector(tmp_path)
        # Feed 10 wins, 10 losses
        _feed(det, "walks", [1] * 10 + [0] * 10)
        stats = det.get_stats("walks")
        assert abs(stats["accuracy"] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# rolling_brier and check_model_health (existing helpers)
# ---------------------------------------------------------------------------

class TestRollingBrier:
    def _preds(self, n: int, win_rate: float = 0.6) -> list[dict]:
        preds = []
        for i in range(n):
            result = "W" if i < int(n * win_rate) else "L"
            preds.append({"confidence": 0.65, "result": result, "stat_type": "hits"})
        return preds

    def test_returns_empty_when_too_few(self):
        assert rolling_brier(self._preds(10), window=50) == []

    def test_length_correct(self):
        preds = self._preds(100)
        rb = rolling_brier(preds, window=20)
        assert len(rb) == 100 - 20 + 1

    def test_values_between_zero_and_one(self):
        rb = rolling_brier(self._preds(100), window=20)
        assert all(0.0 <= v <= 1.0 for v in rb)


class TestCheckModelHealth:
    def _preds(self, n: int, win_rate: float, conf: float = 0.65) -> list[dict]:
        preds = []
        for i in range(n):
            result = "W" if i < int(n * win_rate) else "L"
            preds.append({"confidence": conf, "result": result, "stat_type": "hits"})
        return preds

    def test_needs_min_sample(self):
        result = check_model_health(self._preds(10, win_rate=0.6), min_sample=50)
        assert result["healthy"] is True
        assert "Need" in result["message"]

    def test_healthy_model(self):
        result = check_model_health(self._preds(100, win_rate=0.60), min_sample=50)
        assert "overall_accuracy" in result
        assert result["overall_brier"] is not None

    def test_low_accuracy_triggers_alert(self):
        result = check_model_health(self._preds(100, win_rate=0.40), min_sample=50)
        assert any("accuracy" in a.lower() for a in result["alerts"])
