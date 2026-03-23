import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtester import filter_nonplays


class BacktesterNonplayTests(unittest.TestCase):
    def test_zero_stat_result_is_kept_when_player_appeared(self):
        plays, stats = filter_nonplays([
            {
                "player_name": "Batter One",
                "prop_type": "hits",
                "actual": 0.0,
                "plate_appearances": 4,
            },
            {
                "player_name": "Batter Two",
                "prop_type": "hits",
                "actual": 0.0,
                "plate_appearances": 0,
            },
        ])

        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0]["player_name"], "Batter One")
        self.assertEqual(stats["nonplays_removed"], 1)

    def test_legacy_rows_without_play_flags_are_left_untouched(self):
        plays, stats = filter_nonplays([
            {"player_name": "Legacy Row", "prop_type": "hits", "actual": 0.0},
        ])

        self.assertEqual(len(plays), 1)
        self.assertEqual(stats["nonplays_removed"], 0)


if __name__ == "__main__":
    unittest.main()
