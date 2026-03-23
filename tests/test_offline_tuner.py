import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.autolearn import get_baseline_weights
from src.offline_tuner import (
    analyze_backtest_tail_signals,
    evaluate_floors,
    evaluate_model_weights,
    load_backtest_dataframe,
    load_calibration_backtest_dataframe,
    load_model_backtest_dataframe,
    load_tail_backtest_dataframe,
    optimize_tail_signal_config,
    evaluate_tail_signal_config,
    optimize_confidence_floors,
    optimize_model_parameters,
)


class OfflineTunerTests(unittest.TestCase):
    def test_optimize_confidence_floors_raises_floor_when_low_confidence_bucket_is_bad(self):
        rows = []
        for i in range(180):
            rows.append({
                "game_date": pd.Timestamp("2025-05-01") + pd.Timedelta(days=i // 10),
                "prop_type": "hits",
                "pick": "LESS",
                "confidence": 0.62 if i < 60 else 0.74,
                "result": "L" if i < 40 else "W",
                "is_win": 0 if i < 40 else 1,
                "floor_key": "hits_less",
            })
        train_df = pd.DataFrame(rows)

        tuned = optimize_confidence_floors(
            train_df,
            {"hits_less": 0.60},
            grid=[0.58, 0.60, 0.70, 0.74],
        )

        self.assertGreaterEqual(tuned["floors"]["hits_less"], 0.70)

    def test_evaluate_floors_reports_accuracy_and_coverage(self):
        df = pd.DataFrame([
            {
                "game_date": pd.Timestamp("2025-07-01"),
                "prop_type": "pitcher_strikeouts",
                "pick": "MORE",
                "confidence": 0.68,
                "result": "W",
                "is_win": 1,
                "floor_key": "pitcher_strikeouts_more",
            },
            {
                "game_date": pd.Timestamp("2025-07-01"),
                "prop_type": "pitcher_strikeouts",
                "pick": "MORE",
                "confidence": 0.62,
                "result": "L",
                "is_win": 0,
                "floor_key": "pitcher_strikeouts_more",
            },
        ])

        report = evaluate_floors(df, {"pitcher_strikeouts_more": 0.65})

        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["available"], 2)
        self.assertAlmostEqual(report["coverage_pct"], 0.5, places=3)
        self.assertAlmostEqual(report["accuracy"], 1.0, places=3)

    def test_model_tuner_pushes_offset_toward_systematic_underprojection(self):
        rows = []
        for i in range(220):
            rows.append({
                "game_date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i // 10),
                "prop_type": "hits",
                "projection": 0.90,
                "line": 1.5,
                "actual": 1.30 + (0.2 if i % 2 else -0.1),
            })
        train_df = pd.DataFrame(rows)
        train_df["actual_over"] = (train_df["actual"] > train_df["line"]).astype(float)

        tuned = optimize_model_parameters(train_df, get_baseline_weights())

        self.assertGreater(tuned["weights"]["prop_type_offsets"]["hits"], 0.0)

    def test_model_weight_evaluation_returns_projection_and_probability_metrics(self):
        df = pd.DataFrame([
            {
                "game_date": pd.Timestamp("2025-06-01"),
                "prop_type": "total_bases",
                "projection": 1.8,
                "line": 1.5,
                "actual": 3.0,
                "actual_over": 1.0,
            },
            {
                "game_date": pd.Timestamp("2025-06-01"),
                "prop_type": "total_bases",
                "projection": 1.1,
                "line": 1.5,
                "actual": 0.0,
                "actual_over": 0.0,
            },
        ])

        metrics = evaluate_model_weights(df, get_baseline_weights())

        self.assertEqual(metrics["rows"], 2)
        self.assertIn("log_loss", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("total_bases", metrics["by_prop"])

    def test_model_tuner_supports_expanded_prop_families(self):
        rows = []
        for i in range(220):
            rows.append({
                "game_date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i // 8),
                "prop_type": "earned_runs",
                "projection": 2.4,
                "line": 1.5,
                "actual": 1.2 + (0.1 if i % 2 else -0.1),
                "actual_over": 0.0,
            })
        train_df = pd.DataFrame(rows)

        tuned = optimize_model_parameters(train_df, get_baseline_weights())

        self.assertIn("earned_runs", tuned["weights"]["prop_type_offsets"])
        self.assertLess(tuned["weights"]["prop_type_offsets"]["earned_runs"], 0.0)

    def test_backtest_loaders_keep_new_prop_families(self):
        sample_rows = [
            {
                "game_date": "2025-06-01",
                "prop_type": "earned_runs",
                "projection": 2.0,
                "line": 1.5,
                "actual": 1.0,
                "pick": "LESS",
                "confidence": 0.64,
                "result": "W",
                "plate_appearances": 0,
                "innings_pitched": 6.0,
            },
            {
                "game_date": "2025-06-01",
                "prop_type": "hits_runs_rbis",
                "projection": 2.1,
                "line": 1.5,
                "actual": 3.0,
                "pick": "MORE",
                "confidence": 0.61,
                "result": "W",
                "plate_appearances": 4,
                "innings_pitched": 0.0,
            },
        ]

        with patch("src.offline_tuner.load_results", return_value=sample_rows):
            floor_df = load_backtest_dataframe("unused.json")
            model_df = load_model_backtest_dataframe("unused.json")

        self.assertEqual(set(floor_df["prop_type"]), {"earned_runs", "hits_runs_rbis"})
        self.assertEqual(set(model_df["prop_type"]), {"earned_runs", "hits_runs_rbis"})

    def test_calibration_loader_keeps_expanded_prop_families(self):
        sample_rows = [
            {
                "game_date": "2025-06-01",
                "prop_type": "runs",
                "projection": 0.72,
                "line": 0.5,
                "actual": 1.0,
                "pick": "MORE",
                "confidence": 0.58,
                "result": "W",
            },
            {
                "game_date": "2025-06-02",
                "prop_type": "earned_runs",
                "projection": 1.6,
                "line": 1.5,
                "actual": 0.0,
                "pick": "LESS",
                "confidence": 0.57,
                "result": "W",
            },
        ]

        with patch("src.offline_tuner.load_results", return_value=sample_rows):
            cal_df = load_calibration_backtest_dataframe("unused.json")

        self.assertEqual(set(cal_df["stat_internal"]), {"runs", "earned_runs"})
        self.assertIn("actual_result", cal_df.columns)

    def test_tail_optimizer_learns_prop_specific_threshold_overrides(self):
        rows = []
        for i in range(220):
            rows.append({
                "game_date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i // 8),
                "prop_type": "hits",
                "actual": 3.0 if i % 5 == 0 else 0.0,
                "breakout_prob": 0.32 if i % 5 == 0 else 0.06,
                "dud_prob": 0.08 if i % 5 == 0 else 0.42,
                "breakout_target": 3.0,
                "dud_target": 0.0,
                "actual_breakout": 1 if i % 5 == 0 else 0,
                "actual_dud": 0 if i % 5 == 0 else 1,
            })
        train_df = pd.DataFrame(rows)
        tuned = optimize_tail_signal_config(train_df, get_baseline_weights())
        cfg = tuned["tail_signal_config"]["label_thresholds_by_prop"]["hits"]
        metrics = evaluate_tail_signal_config(train_df, tuned["tail_signal_config"])

        self.assertIn("breakout_high", cfg)
        self.assertIn("dud_high", cfg)
        self.assertGreater(metrics["by_prop"]["hits"]["breakout_high_precision"], 0.0)
        self.assertGreater(metrics["by_prop"]["hits"]["dud_high_precision"], 0.0)

    def test_tail_analyzer_does_not_promote_identical_config(self):
        rows = []
        for i in range(220):
            rows.append({
                "game_date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i // 4),
                "prop_type": "hits",
                "actual": 3.0 if i % 5 == 0 else 0.0,
                "breakout_prob": 0.32 if i % 5 == 0 else 0.06,
                "dud_prob": 0.08 if i % 5 == 0 else 0.42,
                "breakout_target": 3.0,
                "dud_target": 0.0,
                "actual_breakout": 1 if i % 5 == 0 else 0,
                "actual_dud": 0 if i % 5 == 0 else 1,
            })
        df = pd.DataFrame(rows)
        current_weights = get_baseline_weights()
        same_cfg = current_weights.get("tail_signal_config", {})

        with patch("src.offline_tuner.load_tail_backtest_dataframe", return_value=df), patch(
            "src.offline_tuner.load_current_weights", return_value=current_weights
        ), patch(
            "src.offline_tuner.optimize_tail_signal_config",
            return_value={"tail_signal_config": same_cfg, "recommendations": {}},
        ):
            analysis = analyze_backtest_tail_signals("ignored.json")

        self.assertFalse(analysis["should_apply"])
        self.assertIn("config_changed=False", analysis["reason"])


if __name__ == "__main__":
    unittest.main()
