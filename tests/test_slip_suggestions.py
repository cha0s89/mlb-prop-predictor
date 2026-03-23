import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


if __name__ == "__main__":
    unittest.main()
