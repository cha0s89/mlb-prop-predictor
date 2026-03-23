import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.selection import annotate_prediction_floor, get_confidence_floor


class SelectionPolicyTests(unittest.TestCase):
    def test_get_confidence_floor_normalizes_display_stat_names(self):
        weights = {
            "per_prop_confidence_floors": {
                "pitcher_strikeouts_more": 0.66,
                "hitter_fantasy_score_less": 0.68,
            }
        }

        self.assertEqual(
            get_confidence_floor(weights, "Pitcher Strikeouts", "MORE"),
            0.66,
        )
        self.assertEqual(
            get_confidence_floor(weights, "Hitter Fantasy Score", "LESS"),
            0.68,
        )

    def test_annotate_prediction_floor_marks_eligibility(self):
        weights = {"per_prop_confidence_floors": {"hits_less": 0.72}}
        pred = {
            "stat_internal": "hits",
            "pick": "LESS",
            "confidence": 0.70,
        }

        annotate_prediction_floor(pred, weights)

        self.assertEqual(pred["confidence_floor"], 0.72)
        self.assertFalse(pred["meets_conf_floor"])


if __name__ == "__main__":
    unittest.main()
