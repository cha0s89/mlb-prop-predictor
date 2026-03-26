import unittest

from src.prediction_cleanup import canonical_prop_type, dedupe_predictions


class PredictionCleanupTests(unittest.TestCase):
    def test_canonical_prop_type_maps_hitter_strikeouts(self):
        self.assertEqual(canonical_prop_type("hitter_strikeouts"), "batter_strikeouts")
        self.assertEqual(canonical_prop_type("hits"), "hits")

    def test_dedupe_predictions_keeps_stronger_conflicting_side(self):
        predictions = [
            {
                "game_date": "2026-03-25",
                "player_name": "Logan Webb",
                "stat_internal": "hits_allowed",
                "line": 4.5,
                "line_type": "standard",
                "pick": "LESS",
                "confidence": 0.5361,
                "edge": 0.0361,
            },
            {
                "game_date": "2026-03-25",
                "player_name": "Logan Webb",
                "stat_internal": "hits_allowed",
                "line": 4.5,
                "line_type": "standard",
                "pick": "MORE",
                "confidence": 0.5085,
                "edge": 0.0085,
            },
        ]

        result = dedupe_predictions(predictions)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pick"], "LESS")


if __name__ == "__main__":
    unittest.main()
