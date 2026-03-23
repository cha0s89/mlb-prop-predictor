import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.team_context import (
    extract_schedule_dates,
    normalize_team_code,
    pitcher_row_matches_team,
    team_lookup_keys,
)


class TeamContextTests(unittest.TestCase):
    def test_extract_schedule_dates_includes_local_and_utc_days(self):
        dates = extract_schedule_dates(["2026-03-26 20:30:00-0400"])
        self.assertIn("2026-03-26", dates)
        self.assertIn("2026-03-27", dates)

    def test_team_lookup_keys_cover_aliases(self):
        keys = team_lookup_keys("AZ")
        self.assertIn("AZ", keys)
        self.assertIn("ARI", keys)

    def test_pitcher_row_team_validation_catches_impossible_assignment(self):
        self.assertFalse(pitcher_row_matches_team({"Team": "HOU"}, "DET"))
        self.assertTrue(pitcher_row_matches_team({"Team": "DET"}, "DET"))
        self.assertEqual(normalize_team_code("AZ"), "ARI")


if __name__ == "__main__":
    unittest.main()
