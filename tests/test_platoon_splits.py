"""
Unit tests for src/platoon_splits.py

Tests cover:
  - Generic fallback behavior (no MLBAM ID)
  - Switch-hitter handling
  - Unknown handedness
  - Bayesian shrinkage math
  - Player-specific adjustment with mocked API data
  - Graceful fallback when API fails
  - Pitcher platoon adjustment
  - Adjustment caps
  - Compatibility with existing predictor platoon dict interface
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is importable
sys.path.insert(0, ".")


class TestGenericFallback(unittest.TestCase):
    """Tests that run without any API calls (no MLBAM ID supplied)."""

    def setUp(self):
        from src.platoon_splits import get_batter_platoon_adjustment
        self.fn = get_batter_platoon_adjustment

    # ── Favorable matchups ────────────────────────────────────────────────

    def test_lhb_vs_rhp_favorable(self):
        result = self.fn("L", "R")
        self.assertTrue(result["favorable"])
        self.assertGreater(result["adjustment"], 1.0)
        self.assertLess(result["k_adjustment"], 1.0)
        self.assertEqual(result["source"], "generic")

    def test_rhb_vs_lhp_favorable(self):
        result = self.fn("R", "L")
        self.assertTrue(result["favorable"])
        self.assertGreater(result["adjustment"], 1.0)

    # ── Unfavorable matchups ──────────────────────────────────────────────

    def test_lhb_vs_lhp_unfavorable(self):
        result = self.fn("L", "L")
        self.assertFalse(result["favorable"])
        self.assertLess(result["adjustment"], 1.0)
        self.assertGreater(result["k_adjustment"], 1.0)
        self.assertEqual(result["source"], "generic")

    def test_rhb_vs_rhp_unfavorable(self):
        result = self.fn("R", "R")
        self.assertFalse(result["favorable"])
        self.assertLess(result["adjustment"], 1.0)

    # ── Switch hitters ────────────────────────────────────────────────────

    def test_switch_hitter_vs_lhp(self):
        result = self.fn("S", "L")
        self.assertTrue(result["favorable"])
        self.assertAlmostEqual(result["adjustment"], 1.02, places=2)

    def test_switch_hitter_vs_rhp(self):
        result = self.fn("S", "R")
        self.assertTrue(result["favorable"])
        self.assertAlmostEqual(result["adjustment"], 1.02, places=2)

    # ── Unknown handedness ────────────────────────────────────────────────

    def test_unknown_batter_hand(self):
        result = self.fn("", "R")
        self.assertIsNone(result["favorable"])
        self.assertAlmostEqual(result["adjustment"], 1.0, places=3)

    def test_unknown_pitcher_hand(self):
        result = self.fn("L", "")
        self.assertIsNone(result["favorable"])
        self.assertAlmostEqual(result["adjustment"], 1.0, places=3)

    # ── Case insensitivity ────────────────────────────────────────────────

    def test_lowercase_hands(self):
        result = self.fn("l", "r")
        self.assertTrue(result["favorable"])
        self.assertGreater(result["adjustment"], 1.0)

    # ── Required keys present ─────────────────────────────────────────────

    def test_required_keys_present(self):
        result = self.fn("L", "R")
        for key in ("adjustment", "k_adjustment", "iso_adjustment", "avg_adjustment",
                    "slg_adjustment", "favorable", "source", "pa_sample", "description"):
            self.assertIn(key, result, f"Missing key: {key}")


class TestBayesianShrinkage(unittest.TestCase):
    """Test the Bayesian blending math in isolation."""

    def test_zero_pa_returns_league_average(self):
        from src.platoon_splits import _bayesian_blend
        lg = 1.10
        result = _bayesian_blend(player_ratio=1.50, lg_ratio=lg, pa=0)
        self.assertAlmostEqual(result, lg, places=3)

    def test_large_sample_approaches_player_ratio(self):
        from src.platoon_splits import _bayesian_blend, SPLIT_PA_PRIOR, ADJ_MAX
        # With 10× the prior, weight ≈ 0.91 — should be close to player_ratio
        result = _bayesian_blend(player_ratio=1.20, lg_ratio=1.10, pa=SPLIT_PA_PRIOR * 10)
        self.assertGreater(result, 1.18)

    def test_half_prior_pa_is_50_50_blend(self):
        from src.platoon_splits import _bayesian_blend, SPLIT_PA_PRIOR
        player, lg = 1.20, 1.00
        result = _bayesian_blend(player, lg, pa=SPLIT_PA_PRIOR)
        expected = (player + lg) / 2  # exact 50/50
        self.assertAlmostEqual(result, expected, places=3)

    def test_cap_applied(self):
        from src.platoon_splits import _bayesian_blend, ADJ_MAX, ADJ_MIN
        # Extreme player ratio should be capped
        result_high = _bayesian_blend(player_ratio=2.0, lg_ratio=1.10, pa=5000)
        self.assertLessEqual(result_high, ADJ_MAX)
        result_low = _bayesian_blend(player_ratio=0.20, lg_ratio=0.90, pa=5000)
        self.assertGreaterEqual(result_low, ADJ_MIN)


class TestPlayerSpecificAdjustment(unittest.TestCase):
    """Test player-specific adjustments using mocked API responses."""

    def _make_split_stat(self, pa, avg, obp, slg, k, bb):
        """Build a minimal MLB Stats API stat dict."""
        return {
            "plateAppearances": pa,
            "avg": str(avg),
            "obp": str(obp),
            "slg": str(slg),
            "strikeOuts": int(pa * k),
            "baseOnBalls": int(pa * bb),
        }

    def test_reverse_split_player(self):
        """A player who hits better vs same-hand should get adj > 1 (overrides generic)."""
        from src.platoon_splits import get_batter_platoon_adjustment

        # LHB who hits MUCH better vs LHP (reverse split) — 200 PA sample
        mock_splits = {
            "vs_L": self._make_split_stat(200, ".320", ".410", ".580", 0.18, 0.12),
            "vs_R": self._make_split_stat(400, ".240", ".310", ".380", 0.26, 0.08),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("L", "L", mlbam_id=99999)

        # LHB vs LHP = normally unfavorable, but this player has reverse splits
        # adj should be > 1.0 because they hit significantly better vs LHP
        self.assertGreater(result["adjustment"], 1.0)
        self.assertEqual(result["source"], "player_specific")
        self.assertTrue(result["favorable"])

    def test_normal_split_player_amplified(self):
        """A player with stronger-than-average favorable split should get > 1.08."""
        from src.platoon_splits import get_batter_platoon_adjustment

        # RHB with large vs LHP advantage
        mock_splits = {
            "vs_L": self._make_split_stat(300, ".330", ".420", ".600", 0.16, 0.13),
            "vs_R": self._make_split_stat(500, ".250", ".320", ".400", 0.24, 0.08),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("R", "L", mlbam_id=99999)

        self.assertGreater(result["adjustment"], 1.08)  # stronger than generic
        self.assertTrue(result["favorable"])

    def test_minimal_split_player_near_neutral(self):
        """A player with near-equal splits should give adjustment close to 1.0."""
        from src.platoon_splits import get_batter_platoon_adjustment

        # Equal performance vs both hands
        mock_splits = {
            "vs_L": self._make_split_stat(300, ".290", ".370", ".490", 0.21, 0.10),
            "vs_R": self._make_split_stat(500, ".291", ".372", ".492", 0.21, 0.10),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("L", "R", mlbam_id=99999)

        # Normally favorable (LHB vs RHP), but very equal splits so adj should be
        # shrunk toward generic 1.08 but also pulled toward player ratio ~1.0
        # Result should be somewhere between 1.0 and 1.08
        self.assertGreater(result["adjustment"], 0.95)

    def test_small_sample_falls_back_to_generic(self):
        """Fewer than MIN_PA_ANY_SIGNAL should return generic adjustment."""
        from src.platoon_splits import get_batter_platoon_adjustment, MIN_PA_ANY_SIGNAL

        mock_splits = {
            "vs_L": self._make_split_stat(MIN_PA_ANY_SIGNAL - 1, ".200", ".280", ".350", 0.30, 0.08),
            "vs_R": self._make_split_stat(400, ".290", ".370", ".490", 0.21, 0.10),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("L", "L", mlbam_id=99999)

        self.assertEqual(result["source"], "generic")

    def test_api_failure_falls_back_gracefully(self):
        """API failure (exception) should return generic fallback silently."""
        from src.platoon_splits import get_batter_platoon_adjustment

        with patch("src.platoon_splits._fetch_player_splits", return_value=None):
            result = get_batter_platoon_adjustment("L", "R", mlbam_id=99999)

        self.assertEqual(result["source"], "generic")
        self.assertGreater(result["adjustment"], 1.0)

    def test_no_mlbam_id_returns_generic(self):
        """Without mlbam_id, should always return generic."""
        from src.platoon_splits import get_batter_platoon_adjustment

        result = get_batter_platoon_adjustment("R", "L", mlbam_id=None)
        self.assertEqual(result["source"], "generic")

    def test_regressed_label_for_small_but_sufficient_sample(self):
        """Between MIN_PA and SPLIT_PA_PRIOR: source should be player_specific_regressed."""
        from src.platoon_splits import (
            get_batter_platoon_adjustment,
            MIN_PA_ANY_SIGNAL,
            SPLIT_PA_PRIOR,
        )

        pa = (MIN_PA_ANY_SIGNAL + SPLIT_PA_PRIOR) // 2  # between thresholds

        mock_splits = {
            "vs_R": self._make_split_stat(pa, ".290", ".370", ".480", 0.22, 0.10),
            "vs_L": self._make_split_stat(300, ".260", ".340", ".420", 0.24, 0.09),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("L", "R", mlbam_id=99999)

        self.assertEqual(result["source"], "player_specific_regressed")
        self.assertIn("pa_sample", result)

    def test_adjustment_dict_compatible_with_predictor(self):
        """Returned dict must have all keys the predictor expects."""
        from src.platoon_splits import get_batter_platoon_adjustment

        mock_splits = {
            "vs_R": self._make_split_stat(300, ".290", ".370", ".480", 0.22, 0.10),
            "vs_L": self._make_split_stat(400, ".270", ".350", ".440", 0.23, 0.09),
        }

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_batter_platoon_adjustment("L", "R", mlbam_id=99999)

        # These are the keys predictor.py reads
        self.assertIn("adjustment", result)
        self.assertIn("k_adjustment", result)
        self.assertIn("iso_adjustment", result)
        self.assertIn("favorable", result)
        self.assertIsInstance(result["adjustment"], float)
        self.assertIsInstance(result["favorable"], bool)


class TestPitcherPlatoonAdjustment(unittest.TestCase):
    """Tests for get_pitcher_platoon_adjustment."""

    def _make_pitcher_split(self, bf, k, bb, avg=".250"):
        return {
            "battersFaced": bf,
            "strikeOuts": k,
            "baseOnBalls": bb,
            "avg": avg,
        }

    def test_lefty_heavy_lineup_hurts_rhp(self):
        """RHP who struggles vs LHB facing mostly LHBs should have adj < 1."""
        from src.platoon_splits import get_pitcher_platoon_adjustment

        # Pitcher: 30% K vs RHB, only 20% K vs LHB
        mock_splits = {
            "vs_L": self._make_pitcher_split(200, k=40, bb=20),   # 20% K vs LHBs
            "vs_R": self._make_pitcher_split(400, k=120, bb=30),  # 30% K vs RHBs
        }
        # Opposing lineup: all left-handed
        opp_hands = ["L"] * 9

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_pitcher_platoon_adjustment(12345, opp_hands)

        self.assertIsNotNone(result)
        self.assertIn("k_adjustment", result)
        self.assertLess(result["k_adjustment"], 1.0)

    def test_righty_heavy_lineup_boosts_rhp(self):
        """RHP who dominates RHB facing mostly RHBs should have adj > 1."""
        from src.platoon_splits import get_pitcher_platoon_adjustment

        mock_splits = {
            "vs_L": self._make_pitcher_split(200, k=40, bb=20),   # 20% K vs LHBs
            "vs_R": self._make_pitcher_split(400, k=140, bb=30),  # 35% K vs RHBs
        }
        opp_hands = ["R"] * 9

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_pitcher_platoon_adjustment(12345, opp_hands)

        self.assertIsNotNone(result)
        self.assertGreater(result["k_adjustment"], 1.0)

    def test_equal_splits_returns_near_one(self):
        """Pitcher with equal splits and mixed lineup → adj ≈ 1.0."""
        from src.platoon_splits import get_pitcher_platoon_adjustment

        mock_splits = {
            "vs_L": self._make_pitcher_split(300, k=75, bb=25),   # 25% K both sides
            "vs_R": self._make_pitcher_split(300, k=75, bb=25),
        }
        opp_hands = ["L", "R", "L", "R", "L", "R", "L", "R", "L"]

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_pitcher_platoon_adjustment(12345, opp_hands)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["k_adjustment"], 1.0, places=2)

    def test_no_lineup_hands_returns_none(self):
        from src.platoon_splits import get_pitcher_platoon_adjustment

        result = get_pitcher_platoon_adjustment(12345, [])
        self.assertIsNone(result)

    def test_api_failure_returns_none(self):
        from src.platoon_splits import get_pitcher_platoon_adjustment

        with patch("src.platoon_splits._fetch_player_splits", return_value=None):
            result = get_pitcher_platoon_adjustment(12345, ["L", "R", "L"])

        self.assertIsNone(result)

    def test_k_adjustment_within_bounds(self):
        """k_adjustment should always be between 0.80 and 1.20."""
        from src.platoon_splits import get_pitcher_platoon_adjustment

        # Extreme split pitcher
        mock_splits = {
            "vs_L": self._make_pitcher_split(500, k=200, bb=40),  # 40% K vs LHB
            "vs_R": self._make_pitcher_split(500, k=50, bb=40),   # 10% K vs RHB
        }
        opp_hands = ["L"] * 9

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            result = get_pitcher_platoon_adjustment(12345, opp_hands)

        self.assertIsNotNone(result)
        self.assertLessEqual(result["k_adjustment"], 1.20)
        self.assertGreaterEqual(result["k_adjustment"], 0.80)

    def test_switch_hitters_counted_as_half_left_half_right(self):
        """Switch hitters should contribute 0.5 to each hand count."""
        from src.platoon_splits import get_pitcher_platoon_adjustment

        mock_splits = {
            "vs_L": self._make_pitcher_split(300, k=90, bb=25),   # 30% K
            "vs_R": self._make_pitcher_split(300, k=60, bb=25),   # 20% K
        }
        opp_hands_pure_left = ["L"] * 9
        opp_hands_switch = ["S"] * 9

        with patch("src.platoon_splits._fetch_player_splits", return_value=mock_splits):
            r_left = get_pitcher_platoon_adjustment(12345, opp_hands_pure_left)
            r_switch = get_pitcher_platoon_adjustment(12345, opp_hands_switch)

        # Switch hitters should yield an adjustment between all-L and all-R
        self.assertIsNotNone(r_left)
        self.assertIsNotNone(r_switch)
        # Since pitcher is better vs L, facing all-L lineup boosts K adj more than all-S
        self.assertGreater(r_left["k_adjustment"], r_switch["k_adjustment"])


class TestMatchupsWrapper(unittest.TestCase):
    """Test that matchups.get_platoon_split_adjustment delegates to platoon_splits."""

    def test_without_mlbam_id_returns_generic(self):
        from src.matchups import get_platoon_split_adjustment
        result = get_platoon_split_adjustment("L", "R")
        self.assertGreater(result["adjustment"], 1.0)
        self.assertIn("favorable", result)

    def test_with_mlbam_id_calls_through(self):
        from src.matchups import get_platoon_split_adjustment

        with patch("src.platoon_splits.get_batter_platoon_adjustment") as mock_fn:
            mock_fn.return_value = {
                "adjustment": 1.15, "k_adjustment": 0.90, "favorable": True,
                "source": "player_specific", "pa_sample": 250,
                "description": "test", "iso_adjustment": 1.14,
                "avg_adjustment": 1.12, "slg_adjustment": 1.16,
            }
            result = get_platoon_split_adjustment("L", "R", mlbam_id=12345)

        mock_fn.assert_called_once_with(
            batter_hand="L", pitcher_hand="R", mlbam_id=12345, season=None
        )
        self.assertAlmostEqual(result["adjustment"], 1.15)


if __name__ == "__main__":
    unittest.main()
