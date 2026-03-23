import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.autolearn import get_baseline_weights
from src.offline_tuner import (
    evaluate_floors,
    evaluate_model_weights,
    load_backtest_dataframe,
    load_model_backtest_dataframe,
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


if __name__ == "__main__":
    unittest.main()
