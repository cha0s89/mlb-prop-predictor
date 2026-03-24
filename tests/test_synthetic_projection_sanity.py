import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predictor import (
    generate_prediction,
    project_batter_hits,
    project_batter_home_runs,
    project_hits_runs_rbis,
    project_batter_runs,
    project_batter_rbis,
    project_batter_doubles,
    project_batter_singles,
    project_batter_strikeouts,
    project_batter_stolen_bases,
    project_batter_total_bases,
    project_pitcher_earned_runs,
    project_pitcher_hits_allowed,
    project_pitcher_strikeouts,
    project_pitcher_walks,
)
from src.tail_signals import build_tail_reason_lists


ELITE_BATTER = {
    "name": "Synthetic Elite Batter",
    "avg": 0.300,
    "obp": 0.390,
    "slg": 0.520,
    "iso": 0.220,
    "pa": 650,
    "ab": 575,
    "h": 173,
    "r": 100,
    "rbi": 100,
    "hr": 28,
    "sb": 20,
    "2b": 32,
    "3b": 3,
    "bb": 50,
    "bb_rate": 7.7,
    "k": 95,
    "k_rate": 14.6,
    "woba": 0.375,
    "babip": 0.325,
    "xba": 0.298,
    "xslg": 0.515,
    "barrel_rate": 13.5,
    "recent_barrel_rate": 13.5,
    "hard_hit_pct": 44.0,
    "recent_hard_hit_pct": 44.0,
    "ev90": 107.5,
    "recent_ev90": 107.5,
    "sprint_speed": 29.5,
    "contact_rate": 84.0,
}

TERRIBLE_BATTER = {
    "name": "Synthetic Terrible Batter",
    "avg": 0.185,
    "obp": 0.235,
    "slg": 0.285,
    "iso": 0.100,
    "pa": 380,
    "ab": 340,
    "h": 63,
    "r": 25,
    "rbi": 28,
    "hr": 5,
    "sb": 2,
    "2b": 10,
    "3b": 1,
    "bb": 22,
    "bb_rate": 5.8,
    "k": 122,
    "k_rate": 32.1,
    "woba": 0.235,
    "babip": 0.250,
    "xba": 0.190,
    "xslg": 0.295,
    "barrel_rate": 3.5,
    "recent_barrel_rate": 3.5,
    "hard_hit_pct": 26.0,
    "recent_hard_hit_pct": 26.0,
    "ev90": 100.5,
    "recent_ev90": 100.5,
    "sprint_speed": 24.5,
    "contact_rate": 68.0,
}

ELITE_PITCHER = {
    "name": "Synthetic Elite Pitcher",
    "ip": 190,
    "gs": 32,
    "ip_per_start": 5.94,
    "era": 2.85,
    "fip": 3.05,
    "xfip": 3.10,
    "k9": 10.8,
    "bb9": 2.1,
    "hr9": 0.90,
    "k_pct": 29.2,
    "k_rate": 0.292,
    "bb_pct": 6.0,
    "bb_rate": 0.060,
    "whip": 1.03,
    "h": 148,
    "bb": 44,
    "k": 228,
    "er": 60,
    "bf": 785,
    "recent_csw_pct": 31.5,
    "csw_pct": 31.5,
    "recent_swstr_pct": 13.8,
    "swstr_pct": 13.8,
}

TERRIBLE_PITCHER = {
    "name": "Synthetic Terrible Pitcher",
    "ip": 118,
    "gs": 24,
    "ip_per_start": 4.92,
    "era": 5.95,
    "fip": 5.65,
    "xfip": 5.55,
    "k9": 6.2,
    "bb9": 4.6,
    "hr9": 1.80,
    "k_pct": 17.5,
    "k_rate": 0.175,
    "bb_pct": 12.9,
    "bb_rate": 0.129,
    "whip": 1.58,
    "h": 132,
    "bb": 60,
    "k": 81,
    "er": 78,
    "bf": 515,
    "recent_csw_pct": 24.5,
    "csw_pct": 24.5,
    "recent_swstr_pct": 8.7,
    "swstr_pct": 8.7,
}

STRONG_BATTER_LINEUP_CONTEXT = {
    "has_data": True,
    "ahead_obp": 0.368,
    "ahead_woba": 0.352,
    "ahead_bb_rate": 10.8,
    "behind_woba": 0.364,
    "behind_slg": 0.495,
    "behind_k_rate": 18.5,
    "team_avg_woba": 0.345,
    "team_avg_obp": 0.338,
    "team_avg_k_rate": 20.8,
    "lineup_depth_woba": 0.337,
}

WEAK_BATTER_LINEUP_CONTEXT = {
    "has_data": True,
    "ahead_obp": 0.286,
    "ahead_woba": 0.275,
    "ahead_bb_rate": 6.0,
    "behind_woba": 0.287,
    "behind_slg": 0.351,
    "behind_k_rate": 28.9,
    "team_avg_woba": 0.291,
    "team_avg_obp": 0.296,
    "team_avg_k_rate": 26.7,
    "lineup_depth_woba": 0.287,
}

HIGH_K_OPP_LINEUP_CONTEXT = {
    "has_data": True,
    "avg_k_rate": 26.4,
    "top6_k_rate": 27.2,
    "avg_woba": 0.309,
    "top5_woba": 0.312,
}

LOW_K_OPP_LINEUP_CONTEXT = {
    "has_data": True,
    "avg_k_rate": 18.8,
    "top6_k_rate": 18.1,
    "avg_woba": 0.324,
    "top5_woba": 0.329,
}


class SyntheticProjectionSanityTests(unittest.TestCase):
    def test_elite_batter_beats_terrible_batter_on_production_props(self):
        self.assertGreater(project_batter_hits(ELITE_BATTER)["projection"], project_batter_hits(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_runs(ELITE_BATTER)["projection"], project_batter_runs(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_rbis(ELITE_BATTER)["projection"], project_batter_rbis(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_hits_runs_rbis(ELITE_BATTER)["projection"], project_hits_runs_rbis(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_home_runs(ELITE_BATTER)["projection"], project_batter_home_runs(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_total_bases(ELITE_BATTER)["projection"], project_batter_total_bases(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_singles(ELITE_BATTER)["projection"], project_batter_singles(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_doubles(ELITE_BATTER)["projection"], project_batter_doubles(TERRIBLE_BATTER)["projection"])
        self.assertGreater(project_batter_stolen_bases(ELITE_BATTER)["projection"], project_batter_stolen_bases(TERRIBLE_BATTER)["projection"])

    def test_hrrbi_gap_is_material_for_star_vs_scrub(self):
        elite = project_hits_runs_rbis(ELITE_BATTER, lineup_pos=3)["projection"]
        terrible = project_hits_runs_rbis(TERRIBLE_BATTER, lineup_pos=8)["projection"]
        self.assertGreater(elite - terrible, 1.0)

    def test_lineup_position_moves_runs_and_rbi_opportunity(self):
        leadoff_runs = project_batter_runs(ELITE_BATTER, lineup_pos=1)["projection"]
        cleanup_runs = project_batter_runs(ELITE_BATTER, lineup_pos=4)["projection"]
        leadoff_rbi = project_batter_rbis(ELITE_BATTER, lineup_pos=1)["projection"]
        cleanup_rbi = project_batter_rbis(ELITE_BATTER, lineup_pos=4)["projection"]

        self.assertGreater(leadoff_runs, cleanup_runs)
        self.assertGreater(cleanup_rbi, leadoff_rbi)

    def test_confirmed_lineup_support_moves_runs_and_rbis(self):
        strong_runs = project_batter_runs(
            ELITE_BATTER,
            lineup_pos=3,
            lineup_context=STRONG_BATTER_LINEUP_CONTEXT,
        )["projection"]
        weak_runs = project_batter_runs(
            ELITE_BATTER,
            lineup_pos=3,
            lineup_context=WEAK_BATTER_LINEUP_CONTEXT,
        )["projection"]
        strong_rbi = project_batter_rbis(
            ELITE_BATTER,
            lineup_pos=4,
            lineup_context=STRONG_BATTER_LINEUP_CONTEXT,
        )["projection"]
        weak_rbi = project_batter_rbis(
            ELITE_BATTER,
            lineup_pos=4,
            lineup_context=WEAK_BATTER_LINEUP_CONTEXT,
        )["projection"]

        self.assertGreater(strong_runs, weak_runs)
        self.assertGreater(strong_rbi, weak_rbi)

    def test_elite_batter_strikes_out_less_than_terrible_batter(self):
        elite_k = project_batter_strikeouts(ELITE_BATTER)["projection"]
        terrible_k = project_batter_strikeouts(TERRIBLE_BATTER)["projection"]
        self.assertLess(elite_k, terrible_k)

    def test_elite_pitcher_outprojects_terrible_pitcher(self):
        self.assertGreater(project_pitcher_strikeouts(ELITE_PITCHER)["projection"], project_pitcher_strikeouts(TERRIBLE_PITCHER)["projection"])
        self.assertLess(project_pitcher_earned_runs(ELITE_PITCHER)["projection"], project_pitcher_earned_runs(TERRIBLE_PITCHER)["projection"])
        self.assertLess(project_pitcher_walks(ELITE_PITCHER)["projection"], project_pitcher_walks(TERRIBLE_PITCHER)["projection"])
        self.assertLess(project_pitcher_hits_allowed(ELITE_PITCHER)["projection"], project_pitcher_hits_allowed(TERRIBLE_PITCHER)["projection"])

    def test_confirmed_opposing_lineup_changes_pitcher_k_projection(self):
        high_k = project_pitcher_strikeouts(
            ELITE_PITCHER,
            opp_k_rate=22.7,
            opp_lineup_context=HIGH_K_OPP_LINEUP_CONTEXT,
        )["projection"]
        low_k = project_pitcher_strikeouts(
            ELITE_PITCHER,
            opp_k_rate=22.7,
            opp_lineup_context=LOW_K_OPP_LINEUP_CONTEXT,
        )["projection"]

        self.assertGreater(high_k, low_k)

    def test_generate_prediction_probability_contract_holds_for_synthetic_profiles(self):
        cases = [
            generate_prediction(
                player_name=ELITE_BATTER["name"],
                stat_type="Hits",
                stat_internal="hits",
                line=1.5,
                batter_profile=ELITE_BATTER,
                lineup_pos=2,
            ),
            generate_prediction(
                player_name=TERRIBLE_BATTER["name"],
                stat_type="Runs",
                stat_internal="runs",
                line=0.5,
                batter_profile=TERRIBLE_BATTER,
                lineup_pos=8,
            ),
            generate_prediction(
                player_name=ELITE_PITCHER["name"],
                stat_type="Pitcher Strikeouts",
                stat_internal="pitcher_strikeouts",
                line=7.5,
                pitcher_profile=ELITE_PITCHER,
            ),
            generate_prediction(
                player_name=TERRIBLE_PITCHER["name"],
                stat_type="Hits Allowed",
                stat_internal="hits_allowed",
                line=6.5,
                pitcher_profile=TERRIBLE_PITCHER,
            ),
        ]

        for result in cases:
            self.assertAlmostEqual(
                result["p_over"] + result["p_under"] + result["p_push"],
                1.0,
                places=3,
            )
            expected_pick = "MORE" if result["p_over"] >= result["p_under"] else "LESS"
            self.assertEqual(result["pick"], expected_pick)

    def test_tail_metrics_are_present_and_ordered(self):
        result = generate_prediction(
            player_name=ELITE_BATTER["name"],
            stat_type="Total Bases",
            stat_internal="total_bases",
            line=1.5,
            batter_profile=ELITE_BATTER,
            lineup_pos=2,
        )

        self.assertLessEqual(result["p10"], result["p50"])
        self.assertLessEqual(result["p50"], result["p90"])
        self.assertIn(result["breakout_watch"], {"Low", "Medium", "High"})
        self.assertIn(result["dud_risk"], {"Low", "Medium", "High"})
        self.assertGreaterEqual(result["breakout_prob"], 0.0)
        self.assertLessEqual(result["breakout_prob"], 1.0)
        self.assertGreaterEqual(result["dud_prob"], 0.0)
        self.assertLessEqual(result["dud_prob"], 1.0)

    def test_elite_batter_has_more_ceiling_and_less_dud_risk_than_terrible_batter(self):
        elite = generate_prediction(
            player_name=ELITE_BATTER["name"],
            stat_type="Total Bases",
            stat_internal="total_bases",
            line=1.5,
            batter_profile=ELITE_BATTER,
            lineup_pos=2,
        )
        terrible = generate_prediction(
            player_name=TERRIBLE_BATTER["name"],
            stat_type="Total Bases",
            stat_internal="total_bases",
            line=1.5,
            batter_profile=TERRIBLE_BATTER,
            lineup_pos=8,
        )

        self.assertGreater(elite["breakout_prob"], terrible["breakout_prob"])
        self.assertLess(elite["dud_prob"], terrible["dud_prob"])

    def test_elite_pitcher_has_better_low_er_tail_than_terrible_pitcher(self):
        elite = generate_prediction(
            player_name=ELITE_PITCHER["name"],
            stat_type="Earned Runs",
            stat_internal="earned_runs",
            line=2.5,
            pitcher_profile=ELITE_PITCHER,
        )
        terrible = generate_prediction(
            player_name=TERRIBLE_PITCHER["name"],
            stat_type="Earned Runs",
            stat_internal="earned_runs",
            line=2.5,
            pitcher_profile=TERRIBLE_PITCHER,
        )

        self.assertGreater(elite["breakout_prob"], terrible["breakout_prob"])
        self.assertLess(elite["dud_prob"], terrible["dud_prob"])

    def test_tail_reasons_surface_lineup_support_and_confirmed_whiff_context(self):
        hitter = generate_prediction(
            player_name=ELITE_BATTER["name"],
            stat_type="Runs",
            stat_internal="runs",
            line=0.5,
            batter_profile=ELITE_BATTER,
            lineup_pos=3,
            batter_lineup_context=STRONG_BATTER_LINEUP_CONTEXT,
        )
        pitcher = generate_prediction(
            player_name=ELITE_PITCHER["name"],
            stat_type="Pitcher Strikeouts",
            stat_internal="pitcher_strikeouts",
            line=6.5,
            pitcher_profile=ELITE_PITCHER,
            opp_lineup_context=HIGH_K_OPP_LINEUP_CONTEXT,
        )

        hitter_reasons = build_tail_reason_lists(hitter)
        pitcher_reasons = build_tail_reason_lists(pitcher)

        self.assertTrue(
            any("run-scoring chances" in reason or "lineup depth" in reason for reason in hitter_reasons["breakout"]),
            hitter_reasons,
        )
        self.assertTrue(
            any("whiffs more than average" in reason for reason in pitcher_reasons["breakout"]),
            pitcher_reasons,
        )

    def test_half_line_count_props_can_legitimately_project_above_line_and_still_be_less(self):
        result = generate_prediction(
            player_name=TERRIBLE_BATTER["name"],
            stat_type="Hits",
            stat_internal="hits",
            line=0.5,
            batter_profile=TERRIBLE_BATTER,
            lineup_pos=8,
        )

        self.assertGreater(result["projection"], 0.5)
        self.assertLess(result["p_over"], result["p_under"])
        self.assertEqual(result["pick"], "LESS")


if __name__ == "__main__":
    unittest.main()
