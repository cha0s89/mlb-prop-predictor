import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src import trends
from src.predictor import _pitcher_quality_early_season_discount, abs_adjustment_factor, generate_prediction


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

    def test_durable_opening_day_starter_gets_less_harsh_discount(self):
        ace = {"ip": 195.0, "gs": 31, "whip": 0.95, "k_pct": 30.0}
        generic = {"ip": 162.0, "gs": 31, "whip": 1.28, "k_pct": 22.7}

        ace_discount = _pitcher_quality_early_season_discount(ace, date(2026, 3, 27))
        generic_discount = _pitcher_quality_early_season_discount(generic, date(2026, 3, 27))

        self.assertGreater(ace_discount, generic_discount)
        self.assertLessEqual(ace_discount, 1.0)

    def test_abs_adjustment_uses_dynamic_opening_day(self):
        preseason = abs_adjustment_factor("pitcher_strikeouts", date(2027, 3, 20))
        opening_week = abs_adjustment_factor("pitcher_strikeouts", date(2027, 4, 2))

        self.assertEqual(preseason, 1.0)
        self.assertLess(opening_week, 1.0)


if __name__ == "__main__":
    unittest.main()
