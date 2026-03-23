import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stats import compute_batter_profile, compute_pitcher_profile


class StatsProfileTests(unittest.TestCase):
    def test_compute_batter_profile_keeps_run_production_fields(self):
        season = pd.Series(
            {
                "Name": "Test Batter",
                "Team": "LAD",
                "PA": 640,
                "G": 155,
                "AVG": 0.301,
                "OBP": 0.392,
                "SLG": 0.521,
                "ISO": 0.220,
                "BABIP": 0.318,
                "K%": 17.5,
                "BB%": 9.2,
                "wOBA": 0.378,
                "wRC+": 146,
                "xwOBA": 0.381,
                "H": 171,
                "BB": 59,
                "R": 104,
                "RBI": 98,
                "2B": 34,
                "3B": 4,
                "HR": 29,
                "SB": 18,
            }
        )
        recent = pd.DataFrame(
            {
                "type": ["X", "X"],
                "launch_speed": [102.1, 97.4],
                "launch_angle": [18.0, 25.0],
                "barrel": [1, 0],
            }
        )

        profile = compute_batter_profile(season, recent)

        self.assertEqual(profile["g"], 155)
        self.assertEqual(profile["r"], 104)
        self.assertEqual(profile["rbi"], 98)
        self.assertEqual(profile["2b"], 34)
        self.assertEqual(profile["3b"], 4)
        self.assertEqual(profile["wrc_plus"], 146.0)
        self.assertAlmostEqual(profile["xwoba"], 0.381)

    def test_compute_pitcher_profile_handles_missing_zone_column(self):
        season = pd.Series(
            {
                "Name": "Test Pitcher",
                "Team": "SEA",
                "IP": 180.0,
                "ERA": 3.21,
                "FIP": 3.45,
                "K/9": 10.2,
                "BB/9": 2.4,
                "K%": 28.1,
                "BB%": 6.3,
                "WHIP": 1.08,
                "HR/9": 0.94,
            }
        )
        recent = pd.DataFrame(
            {
                "description": ["called_strike", "swinging_strike", "ball"],
                "release_speed": [96.1, 95.8, 96.4],
                "pitch_type": ["FF", "SL", "SI"],
            }
        )

        profile = compute_pitcher_profile(season, recent)

        self.assertIn("recent_chase_rate", profile)
        self.assertEqual(profile["recent_chase_rate"], 0.0)
        self.assertGreater(profile["recent_fb_velo"], 0.0)


if __name__ == "__main__":
    unittest.main()
