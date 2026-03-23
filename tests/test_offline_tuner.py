import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.offline_tuner import evaluate_floors, optimize_confidence_floors


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


if __name__ == "__main__":
    unittest.main()
