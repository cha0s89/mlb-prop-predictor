import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from scipy.stats import gamma, poisson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predictor import calculate_over_under_probability, get_distribution_config
from src.sharp_odds import _MARKET_TO_DIST_KEY, distribution_reprice, extract_sharp_lines, find_ev_edges


class ProbabilityContractTests(unittest.TestCase):
    def test_integer_line_probability_mass_sums_to_one(self):
        result = calculate_over_under_probability(5.2, 5.0, "pitcher_strikeouts")
        total = result["p_over"] + result["p_under"] + result["p_push"]
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)
        expected_win_prob = round(result["confidence"] * (1 - result["p_push"]), 4)
        self.assertAlmostEqual(result["win_prob"], expected_win_prob, places=4)

    def test_half_line_has_zero_push_mass(self):
        result = calculate_over_under_probability(1.6, 1.5, "total_bases")
        self.assertEqual(result["p_push"], 0.0)
        self.assertAlmostEqual(result["p_over"] + result["p_under"], 1.0, places=3)

    def test_integer_line_uses_strict_more_less_push_semantics(self):
        mu = 5.0
        line = 5.0
        result = calculate_over_under_probability(mu, line, "pitcher_strikeouts")

        expected_over = 1 - poisson.cdf(5, mu)
        expected_under = poisson.cdf(4, mu)
        expected_push = poisson.pmf(5, mu)

        self.assertAlmostEqual(result["p_over"], expected_over, places=3)
        self.assertAlmostEqual(result["p_under"], expected_under, places=3)
        self.assertAlmostEqual(result["p_push"], expected_push, places=3)
        self.assertAlmostEqual(
            result["p_over"] + result["p_under"] + result["p_push"],
            1.0,
            places=3,
        )

    def test_weight_overrides_drive_confidence_shrinkage(self):
        loose = calculate_over_under_probability(
            1.9,
            1.5,
            "total_bases",
            weights_override={
                "distribution_params": {"total_bases": {"type": "negbin", "vr": 2.5}},
                "calibration_blend_weights": {"total_bases": 0.0},
                "confidence_shrinkage": {"default": 1.0},
            },
        )
        tight = calculate_over_under_probability(
            1.9,
            1.5,
            "total_bases",
            weights_override={
                "distribution_params": {"total_bases": {"type": "negbin", "vr": 2.5}},
                "calibration_blend_weights": {"total_bases": 0.0},
                "confidence_shrinkage": {"default": 0.5},
            },
        )

        self.assertGreater(loose["confidence"], tight["confidence"])
        self.assertIn("breakout_prob", loose)
        self.assertIn("dud_prob", loose)

    def test_sharp_edges_include_prediction_schema_fields(self):
        pp_lines = pd.DataFrame([
            {
                "player_name": "Aaron Judge",
                "team": "NYY",
                "stat_type": "Total Bases",
                "stat_internal": "total_bases",
                "line": 1.5,
                "start_time": "2026-03-22T19:05:00Z",
            }
        ])
        sharp_lines = [
            {
                "player": "Aaron Judge",
                "market": "batter_total_bases",
                "line": 1.5,
                "consensus_fair_over": 0.58,
                "consensus_fair_under": 0.42,
                "num_books": 3,
            }
        ]

        edges = find_ev_edges(pp_lines, sharp_lines, min_ev_pct=0.25)
        self.assertEqual(len(edges), 1)
        edge = edges[0]

        for field in (
            "line",
            "projection",
            "confidence",
            "p_over",
            "p_under",
            "p_push",
            "win_prob",
            "edge",
            "stat_internal",
            "model_version",
        ):
            self.assertIn(field, edge)

        self.assertEqual(edge["line"], 1.5)
        self.assertGreater(edge["projection"], 1.5)
        self.assertEqual(edge["stat_internal"], "total_bases")
        self.assertAlmostEqual(edge["confidence"], edge["fair_prob"], places=4)

    def test_distribution_reprice_is_push_aware_for_integer_lines(self):
        result = distribution_reprice(
            market="pitcher_strikeouts",
            sharp_line=5.0,
            pp_line=5.0,
            fair_over=0.55,
            fair_under=0.45,
        )

        self.assertGreater(result["p_push"], 0.0)
        self.assertAlmostEqual(
            result["p_over"] + result["p_under"] + result["p_push"],
            1.0,
            places=3,
        )
        self.assertAlmostEqual(
            result["resolved_over"] + result["resolved_under"],
            1.0,
            places=3,
        )
        self.assertAlmostEqual(result["resolved_over"], 0.55, places=2)

    def test_sharp_market_distribution_mapping_uses_specific_prop_families(self):
        self.assertEqual(_MARKET_TO_DIST_KEY["batter_rbis"], "rbis")
        self.assertEqual(_MARKET_TO_DIST_KEY["batter_runs_scored"], "runs")
        self.assertEqual(_MARKET_TO_DIST_KEY["batter_singles"], "singles")
        self.assertEqual(_MARKET_TO_DIST_KEY["batter_doubles"], "doubles")
        self.assertEqual(_MARKET_TO_DIST_KEY["batter_walks"], "walks")

    def test_hrrbi_uses_right_skewed_distribution_family(self):
        config = get_distribution_config(
            "hits_runs_rbis",
            weights={
                "distribution_params": {
                    "hits_runs_rbis": {"type": "gamma", "vr": 2.8}
                }
            },
        )
        self.assertEqual(config["dist_type"], "gamma")
        self.assertEqual(config["var_ratio"], 2.8)

    def test_gamma_props_do_not_apply_discrete_continuity_correction(self):
        weights = {
            "distribution_params": {
                "hits_runs_rbis": {"type": "gamma", "vr": 2.8},
            },
            "calibration_blend_weights": {"hits_runs_rbis": 0.0},
            "confidence_shrinkage": {"hits_runs_rbis": 1.0},
        }
        mu = 1.54
        line = 1.5
        result = calculate_over_under_probability(
            mu,
            line,
            "hits_runs_rbis",
            weights_override=weights,
        )
        shape = mu / 2.8
        scale = 2.8
        expected_over = gamma.sf(line, shape, scale=scale)
        expected_under = gamma.cdf(line, shape, scale=scale)

        self.assertAlmostEqual(result["p_push"], 0.0, places=6)
        self.assertAlmostEqual(result["p_over"], expected_over, places=4)
        self.assertAlmostEqual(result["p_under"], expected_under, places=4)
        self.assertAlmostEqual(result["p_over"] + result["p_under"], 1.0, places=4)
        self.assertGreater(result["p_over"], 0.30)

    def test_extract_sharp_lines_aggregates_cross_line_books_in_latent_mu_space(self):
        event_data = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {"description": "Gerrit Cole", "name": "Over", "price": -115, "point": 5.5},
                                {"description": "Gerrit Cole", "name": "Under", "price": -105, "point": 5.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "outcomes": [
                                {"description": "Gerrit Cole", "name": "Over", "price": 105, "point": 6.5},
                                {"description": "Gerrit Cole", "name": "Under", "price": -125, "point": 6.5},
                            ],
                        }
                    ],
                },
            ]
        }

        with patch("src.sharp_odds.get_effective_book_weight", side_effect=lambda book, market: 2.0 if book == "fanduel" else 1.0):
            lines = extract_sharp_lines(event_data)

        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertEqual(line["player"], "Gerrit Cole")
        self.assertEqual(line["market"], "pitcher_strikeouts")
        self.assertEqual(line["num_books"], 2)
        self.assertEqual(len(line["book_details"]), 2)
        self.assertGreater(line["consensus_mu"], 5.5)
        self.assertLess(line["consensus_mu"], 6.5)
        self.assertGreater(line["line"], 5.5)
        self.assertLess(line["line"], 6.5)
        self.assertGreater(line["consensus_fair_over"], 0.0)
        self.assertGreater(line["consensus_fair_under"], 0.0)
        self.assertAlmostEqual(
            line["consensus_fair_over"] + line["consensus_fair_under"],
            1.0,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
