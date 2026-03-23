import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kelly import calculate_slip_sizing
from src.parlay_suggest import suggest_slips


class SlipSuggestionTests(unittest.TestCase):
    def _prediction(self, idx: int, pick: str, team: str, stat_internal: str, confidence: float) -> dict:
        return {
            "player_name": f"Player {idx}",
            "team": team,
            "stat_type": stat_internal.replace("_", " ").title(),
            "stat_internal": stat_internal,
            "line": 0.5 if "hits" in stat_internal else 1.5,
            "pick": pick,
            "confidence": confidence,
            "win_prob": confidence,
            "rating": "A" if confidence >= 0.66 else "B",
            "edge": confidence - 0.50,
        }

    def test_suggested_slips_differ_by_at_least_two_legs(self):
        predictions = [
            self._prediction(1, "MORE", "NYY", "hits", 0.72),
            self._prediction(2, "MORE", "BOS", "pitcher_strikeouts", 0.71),
            self._prediction(3, "MORE", "LAD", "total_bases", 0.70),
            self._prediction(4, "MORE", "ATL", "hits_runs_rbis", 0.69),
            self._prediction(5, "MORE", "HOU", "runs", 0.68),
            self._prediction(6, "MORE", "PHI", "rbis", 0.67),
            self._prediction(13, "MORE", "TEX", "walks", 0.66),
            self._prediction(14, "MORE", "MIN", "singles", 0.65),
            self._prediction(7, "LESS", "SEA", "pitcher_strikeouts", 0.71),
            self._prediction(8, "LESS", "SD", "hits", 0.70),
            self._prediction(9, "LESS", "MIL", "earned_runs", 0.69),
            self._prediction(10, "LESS", "CLE", "walks_allowed", 0.68),
            self._prediction(11, "LESS", "BAL", "hits_allowed", 0.67),
            self._prediction(12, "LESS", "TOR", "batter_strikeouts", 0.66),
            self._prediction(15, "LESS", "DET", "walks", 0.65),
            self._prediction(16, "LESS", "STL", "doubles", 0.64),
        ]

        slips = suggest_slips(predictions, num_slips=2, slip_size=5)

        self.assertGreaterEqual(len(slips), 2)

        leg_sets = []
        for slip in slips:
            leg_sets.append({
                (pick["player_name"], pick["stat_internal"], pick["pick"])
                for pick in slip["picks"]
            })

        for i in range(len(leg_sets)):
            for j in range(i + 1, len(leg_sets)):
                overlap = len(leg_sets[i] & leg_sets[j])
                self.assertLessEqual(overlap, 3)

    def test_suggested_slips_do_not_repeat_players_within_a_slip(self):
        predictions = [
            self._prediction(1, "LESS", "BOS", "runs", 0.68),
            self._prediction(1, "LESS", "BOS", "total_bases", 0.67),
            self._prediction(2, "MORE", "NYY", "hits", 0.70),
            self._prediction(3, "MORE", "LAD", "rbis", 0.69),
            self._prediction(4, "LESS", "PHI", "hits", 0.68),
            self._prediction(5, "MORE", "HOU", "total_bases", 0.68),
            self._prediction(6, "LESS", "SEA", "earned_runs", 0.67),
            self._prediction(7, "MORE", "ATL", "runs", 0.67),
            self._prediction(8, "LESS", "CLE", "hits_allowed", 0.66),
            self._prediction(9, "MORE", "MIL", "hits_runs_rbis", 0.66),
        ]

        slips = suggest_slips(predictions, num_slips=1, slip_size=5)
        self.assertEqual(len(slips), 1)
        players = [pick["player_name"] for pick in slips[0]["picks"]]
        self.assertEqual(len(players), len(set(players)))

    def test_flex_kelly_sizing_no_longer_uses_power_play_payouts(self):
        picks = [
            self._prediction(1, "LESS", "BOS", "runs", 0.66),
            self._prediction(2, "MORE", "NYY", "hits", 0.66),
            self._prediction(3, "LESS", "LAD", "total_bases", 0.66),
            self._prediction(4, "MORE", "ATL", "rbis", 0.66),
            self._prediction(5, "LESS", "SEA", "hits_allowed", 0.66),
            self._prediction(6, "MORE", "MIL", "hits_runs_rbis", 0.66),
        ]

        sizing = calculate_slip_sizing(picks, bankroll=100.0, slip_type="6_flex")
        self.assertLess(sizing["payout_mult"], 25.0)
        self.assertLess(sizing["edge_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
