import unittest
from unittest.mock import patch

from src import backtester


class BacktesterDatasetTests(unittest.TestCase):
    def test_backtest_single_day_emits_context_and_pitcher_props(self):
        fake_schedule = [{"gamePk": 123, "teams": {"home": {"team": {}}, "away": {"team": {}}}}]
        fake_boxscore = {
            "teams": {
                "home": {"team": {"abbreviation": "LAD"}},
                "away": {"team": {"abbreviation": "AZ"}},
            }
        }
        fake_batter = {
            "full_name": "Test Batter",
            "team": "LAD",
            "game_side": "home",
            "pa": 4,
            "batting_order": 2,
            "hits": 2,
            "runs": 1,
            "rbi": 1,
            "bb": 0,
            "k": 1,
            "doubles": 1,
            "triples": 0,
            "hr": 0,
        }
        fake_home_sp = {
            "full_name": "Home Starter",
            "team": "LAD",
            "ip": 6.0,
            "outs": 18,
            "k": 7,
            "bb": 2,
            "hits_allowed": 5,
            "er": 2,
        }
        fake_away_sp = {
            "full_name": "Away Starter",
            "team": "AZ",
            "ip": 5.0,
            "outs": 15,
            "k": 4,
            "bb": 3,
            "hits_allowed": 7,
            "er": 4,
        }

        def fake_extract_starting_pitcher(_boxscore, side):
            return fake_home_sp if side == "home" else fake_away_sp

        def fake_build_walkforward_profile(player_name, game_date, is_pitcher=False):
            if is_pitcher:
                return {"ip": 140.0, "season_current_weight": 0.42, "season_prior_equivalent_ip": 81.0}
            return {"pa": 500, "season_current_weight": 0.55, "season_prior_equivalent_pa": 110.0}

        def fake_generate_prediction(**kwargs):
            stat_internal = kwargs["stat_internal"]
            return {
                "projection": 1.2 if stat_internal == "hits" else 5.4,
                "pick": "MORE" if stat_internal in {"hits", "pitcher_strikeouts", "pitching_outs"} else "LESS",
                "confidence": 0.63,
                "rating": "B",
                "edge": 0.11,
                "p_over": 0.58,
                "p_under": 0.42,
                "p_push": 0.0,
                "win_prob": 0.63,
                "mu": 1.2,
                "regressed_avg": 0.95,
                "expected_pa": 4.3,
                "expected_ab": 3.9,
                "p10": 0.0,
                "p50": 1.0,
                "p90": 3.0,
                "breakout_prob": 0.17,
                "breakout_watch": "Medium",
                "breakout_target": "2+ hits",
                "dud_prob": 0.21,
                "dud_risk": "Low",
                "dud_target": "0 hits",
                "has_lineup_pos": True,
                "has_opp_data": True,
                "has_park": True,
            }

        prop_types = ["hits", "pitcher_strikeouts", "pitching_outs", "earned_runs", "walks_allowed", "hits_allowed"]
        with patch.object(backtester, "fetch_schedule", return_value=fake_schedule), \
             patch.object(backtester, "fetch_boxscore", return_value=fake_boxscore), \
             patch.object(backtester, "extract_all_batters", return_value=[fake_batter]), \
             patch.object(backtester, "extract_starting_pitcher", side_effect=fake_extract_starting_pitcher), \
             patch.object(backtester, "build_walkforward_profile", side_effect=fake_build_walkforward_profile), \
             patch.object(backtester, "generate_prediction", side_effect=fake_generate_prediction):
            rows = backtester.backtest_single_day("2025-05-20", prop_types=prop_types)

        self.assertEqual(len(rows), 1 + 5 + 5)

        batter_row = next(row for row in rows if row["player_name"] == "Test Batter")
        self.assertEqual(batter_row["opponent"], "AZ")
        self.assertEqual(batter_row["park_team"], "LAD")
        self.assertEqual(batter_row["opp_pitcher"], "Away Starter")
        self.assertEqual(batter_row["expected_pa"], 4.3)
        self.assertEqual(batter_row["p90"], 3.0)
        self.assertEqual(batter_row["breakout_watch"], "Medium")
        self.assertTrue(batter_row["has_opp_data"])

        pitcher_rows = [row for row in rows if row["player_name"] in {"Home Starter", "Away Starter"}]
        self.assertEqual({row["prop_type"] for row in pitcher_rows}, {
            "pitcher_strikeouts", "pitching_outs", "earned_runs", "walks_allowed", "hits_allowed"
        })
        home_outs_row = next(row for row in pitcher_rows if row["player_name"] == "Home Starter" and row["prop_type"] == "pitching_outs")
        away_hits_row = next(row for row in pitcher_rows if row["player_name"] == "Away Starter" and row["prop_type"] == "hits_allowed")
        self.assertEqual(home_outs_row["actual"], 18.0)
        self.assertEqual(away_hits_row["actual"], 7.0)
        self.assertEqual(home_outs_row["park_team"], "LAD")
        self.assertEqual(home_outs_row["opponent"], "AZ")


if __name__ == "__main__":
    unittest.main()
