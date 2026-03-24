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
            },
            "officials": [
                {
                    "officialType": "Home Plate",
                    "official": {"fullName": "Manny Gonzalez"},
                }
            ],
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
        fake_teammate = {
            "full_name": "Lead Off",
            "team": "LAD",
            "game_side": "home",
            "pa": 4,
            "batting_order": 1,
            "hits": 1,
            "runs": 1,
            "rbi": 0,
            "bb": 1,
            "k": 0,
            "doubles": 0,
            "triples": 0,
            "hr": 0,
        }
        fake_opp_batter = {
            "full_name": "Opp Slugger",
            "team": "AZ",
            "game_side": "away",
            "pa": 4,
            "batting_order": 3,
            "hits": 1,
            "runs": 0,
            "rbi": 1,
            "bb": 0,
            "k": 2,
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
            if player_name == "Lead Off":
                return {"pa": 520, "obp": 0.380, "woba": 0.360, "slg": 0.450, "iso": 0.170, "k_rate": 18.0, "bb_rate": 10.0, "season_current_weight": 0.61, "season_prior_equivalent_pa": 95.0}
            if player_name == "Opp Slugger":
                return {"pa": 510, "obp": 0.350, "woba": 0.370, "slg": 0.500, "iso": 0.210, "k_rate": 27.0, "bb_rate": 9.0, "season_current_weight": 0.58, "season_prior_equivalent_pa": 102.0}
            return {"pa": 500, "obp": 0.340, "woba": 0.335, "slg": 0.430, "iso": 0.160, "k_rate": 21.0, "bb_rate": 8.0, "season_current_weight": 0.55, "season_prior_equivalent_pa": 110.0}

        captured_kwargs = []
        def fake_generate_prediction(**kwargs):
            captured_kwargs.append(kwargs)
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
             patch.object(backtester, "extract_all_batters", return_value=[fake_teammate, fake_batter, fake_opp_batter]), \
             patch.object(backtester, "extract_starting_pitcher", side_effect=fake_extract_starting_pitcher), \
             patch.object(backtester, "build_walkforward_profile", side_effect=fake_build_walkforward_profile), \
             patch.object(backtester, "generate_prediction", side_effect=fake_generate_prediction):
            rows = backtester.backtest_single_day("2025-05-20", prop_types=prop_types)

        self.assertEqual(len(rows), 3 + 5 + 5)

        batter_row = next(row for row in rows if row["player_name"] == "Test Batter")
        self.assertEqual(batter_row["opponent"], "AZ")
        self.assertEqual(batter_row["park_team"], "LAD")
        self.assertEqual(batter_row["opp_pitcher"], "Away Starter")
        self.assertEqual(batter_row["expected_pa"], 4.3)
        self.assertEqual(batter_row["p90"], 3.0)
        self.assertEqual(batter_row["breakout_watch"], "Medium")
        self.assertTrue(batter_row["has_opp_data"])
        self.assertGreater(batter_row["ahead_obp"], 0.35)
        self.assertGreater(batter_row["team_lineup_woba"], 0.34)

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
        self.assertGreater(home_outs_row["opp_lineup_k_rate"], 20.0)
        self.assertEqual(home_outs_row["umpire"], "Manny Gonzalez")
        self.assertGreater(home_outs_row["ump_k_adjustment"], 0.0)

        hitter_kwargs = next(kwargs for kwargs in captured_kwargs if kwargs["player_name"] == "Test Batter" and kwargs["stat_internal"] == "hits")
        pitcher_kwargs = next(kwargs for kwargs in captured_kwargs if kwargs["player_name"] == "Home Starter" and kwargs["stat_internal"] == "pitcher_strikeouts")
        self.assertTrue(hitter_kwargs["batter_lineup_context"]["has_data"])
        self.assertTrue(pitcher_kwargs["opp_lineup_context"]["has_data"])
        self.assertTrue(pitcher_kwargs["ump"]["known"])


if __name__ == "__main__":
    unittest.main()
