import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src import trends
from src.predictor import generate_prediction


class TrendAndSeasonalityTests(unittest.TestCase):
    @patch.object(trends, "PYBASEBALL_OK", True)
    @patch.object(trends, "_lookup_batter_mlbam_id", return_value=660271)
    @patch.object(trends, "statcast_batter")
    def test_get_batter_trend_accepts_player_name(self, mock_statcast_batter, _mock_lookup):
        today = datetime.now()
        rows = []
        for idx in range(20):
            rows.append(
                {
                    "events": "single" if idx < 8 else "field_out",
                    "game_date": today - timedelta(days=idx % 10),
                    "type": "X",
                    "launch_speed": 97.0 if idx < 8 else 88.0,
                }
            )
        mock_statcast_batter.return_value = pd.DataFrame(rows)

        result = trends.get_batter_trend("Shohei Ohtani")

        self.assertTrue(result["has_data"])
        self.assertIn("trend_multiplier", result)

    def test_hits_allowed_projection_gets_early_season_discount(self):
        pitcher = {
            "name": "Test Starter",
            "whip": 1.18,
            "ip": 186.0,
            "gs": 31,
            "bb9": 2.4,
        }

        opening_day = generate_prediction(
            player_name=pitcher["name"],
            stat_type="Hits Allowed",
            stat_internal="hits_allowed",
            line=5.5,
            pitcher_profile=pitcher,
            game_date=date(2026, 3, 27),
        )
        midseason = generate_prediction(
            player_name=pitcher["name"],
            stat_type="Hits Allowed",
            stat_internal="hits_allowed",
            line=5.5,
            pitcher_profile=pitcher,
            game_date=date(2026, 6, 15),
        )

        self.assertLess(opening_day["projection"], midseason["projection"])


if __name__ == "__main__":
    unittest.main()
