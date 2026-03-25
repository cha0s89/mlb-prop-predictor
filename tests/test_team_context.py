import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.team_context import (
    extract_schedule_dates,
    get_team_game_value,
    normalize_team_code,
    pitcher_row_matches_team,
    register_team_game_value,
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

    def test_game_specific_team_lookup_prefers_exact_game(self):
        mapping = {}
        register_team_game_value(mapping, "DET", "Tarik Skubal", game_pk=101, game_time="2026-03-27T01:40:00Z")
        register_team_game_value(mapping, "DET", "Framber Valdez", game_pk=202, game_time="2026-03-26T18:40:00Z")
        self.assertEqual(get_team_game_value(mapping, "DET", game_pk=101), "Tarik Skubal")
        self.assertEqual(get_team_game_value(mapping, "DET", game_pk=202), "Framber Valdez")

    def test_game_specific_team_lookup_matches_equivalent_timestamps(self):
        mapping = {}
        register_team_game_value(mapping, "AZ", "Zac Gallen", game_time="2026-03-27T02:10:00Z")
        self.assertEqual(
            get_team_game_value(mapping, "ARI", game_time="2026-03-26 19:10:00-0700"),
            "Zac Gallen",
        )


if __name__ == "__main__":
    unittest.main()
