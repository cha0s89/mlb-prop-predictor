import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tail_signals import build_tail_reason_lists, tail_signal_labels, tail_target_text


class TailSignalTests(unittest.TestCase):
    def test_inverse_prop_uses_shutdown_style_labels(self):
        labels = tail_signal_labels("earned_runs")
        self.assertEqual(labels["breakout"], "Shutdown")
        self.assertEqual(labels["dud"], "Blowup Risk")
        self.assertEqual(
            tail_target_text("Earned Runs Allowed", "earned_runs", 1, "breakout"),
            "Earned Runs Allowed <= 1",
        )

    def test_reason_builder_surfaces_projection_and_context(self):
        reasons = build_tail_reason_lists({
            "stat_internal": "earned_runs",
            "stat_type": "Earned Runs Allowed",
            "projection": 1.4,
            "line": 2.5,
            "weather_mult": 0.94,
            "spring_badge": "hot",
            "trend_badge": "hot",
            "team": "SF",
        })
        self.assertTrue(reasons["breakout"])
        self.assertIn("Model projects about", " ".join(reasons["breakout"]))

    def test_pitcher_strikeout_labels_and_reason_text_are_betting_friendly(self):
        labels = tail_signal_labels("pitcher_strikeouts")
        self.assertEqual(labels["breakout"], "K Ceiling")
        self.assertEqual(labels["dud"], "Low-K Risk")

        reasons = build_tail_reason_lists({
            "stat_internal": "pitcher_strikeouts",
            "stat_type": "Pitcher Strikeouts",
            "projection": 4.4,
            "line": 6.0,
            "opp_lineup_k_rate": 20.8,
            "opp_k_rate_source": "team_baseline",
        })
        joined = " ".join(reasons["dud"])
        self.assertIn("Model projects about", joined)
        self.assertIn("Opponent baseline", joined)


if __name__ == "__main__":
    unittest.main()
