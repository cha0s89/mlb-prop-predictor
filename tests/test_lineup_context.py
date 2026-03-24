import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lineup_context import build_player_lineup_context, build_team_lineup_context


class LineupContextTests(unittest.TestCase):
    def setUp(self):
        self.batting_df = pd.DataFrame(
            [
                {"Name": "John Smith", "Team": "NYY", "OBP": 0.380, "wOBA": 0.365, "SLG": 0.520, "ISO": 0.210, "K%": 18.0, "BB%": 11.0, "Spd": 28.0},
                {"Name": "Juan Soto", "Team": "NYY", "OBP": 0.420, "wOBA": 0.405, "SLG": 0.560, "ISO": 0.230, "K%": 16.0, "BB%": 16.0, "Spd": 26.5},
                {"Name": "Aaron Judge", "Team": "NYY", "OBP": 0.410, "wOBA": 0.430, "SLG": 0.620, "ISO": 0.310, "K%": 25.0, "BB%": 15.0, "Spd": 27.2},
                {"Name": "Giancarlo Stanton", "Team": "NYY", "OBP": 0.330, "wOBA": 0.335, "SLG": 0.480, "ISO": 0.210, "K%": 30.0, "BB%": 8.0, "Spd": 24.0},
                {"Name": "Anthony Volpe", "Team": "NYY", "OBP": 0.310, "wOBA": 0.305, "SLG": 0.390, "ISO": 0.140, "K%": 22.0, "BB%": 7.0, "Spd": 29.0},
                {"Name": "John Smith", "Team": "LAD", "OBP": 0.250, "wOBA": 0.255, "SLG": 0.320, "ISO": 0.100, "K%": 31.0, "BB%": 5.0, "Spd": 24.0},
            ]
        )
        self.lineup = [
            {"player_name": "Juan Soto", "batting_order": 1},
            {"player_name": "John Smith", "batting_order": 2},
            {"player_name": "Aaron Judge", "batting_order": 3},
            {"player_name": "Giancarlo Stanton", "batting_order": 4},
            {"player_name": "Anthony Volpe", "batting_order": 5},
        ]

    def test_team_context_uses_team_filtered_rows_for_duplicate_names(self):
        context = build_team_lineup_context(self.lineup, self.batting_df, "NYY")

        self.assertTrue(context["has_data"])
        self.assertEqual(context["matched_count"], 5)
        self.assertGreater(context["avg_woba"], 0.35)
        matched_names = [player["player_name"] for player in context["players"]]
        self.assertIn("John Smith", matched_names)

    def test_player_context_splits_ahead_and_behind_support(self):
        team_context = build_team_lineup_context(self.lineup, self.batting_df, "NYY")
        player_context = build_player_lineup_context("Aaron Judge", team_context)

        self.assertTrue(player_context["has_data"])
        self.assertEqual(player_context["batting_order"], 3)
        self.assertGreater(player_context["ahead_obp"], 0.39)
        self.assertGreater(player_context["behind_slg"], 0.43)
        self.assertGreater(player_context["team_avg_woba"], 0.35)


if __name__ == "__main__":
    unittest.main()
