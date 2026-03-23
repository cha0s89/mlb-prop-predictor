import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predictor import calculate_over_under_probability
from src.sharp_odds import find_ev_edges


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
            "edge",
            "stat_internal",
            "model_version",
        ):
            self.assertIn(field, edge)

        self.assertEqual(edge["line"], 1.5)
        self.assertEqual(edge["projection"], 1.5)
        self.assertEqual(edge["stat_internal"], "total_bases")
        self.assertAlmostEqual(edge["confidence"], edge["fair_prob"], places=4)


if __name__ == "__main__":
    unittest.main()
