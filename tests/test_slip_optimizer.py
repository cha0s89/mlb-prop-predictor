"""
Tests for src/slip_optimizer.py

Coverage:
 - Correlation matrix construction (shape, symmetry, PSD, known values)
 - Single simulation run (smoke + return-type checks)
 - Known-EV validation for independent legs
 - find_optimal_slips grade filtering and sorting
 - kelly_fraction sizing arithmetic
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.slip_optimizer import (
    FLEX_PAYOUTS,
    SAME_GAME_CORRELATIONS,
    build_correlation_matrix,
    find_optimal_slips,
    kelly_fraction,
    simulate_slip_ev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(prob: float = 0.6, stat: str = "hits", game: str = "G1") -> dict:
    return {"player": "P", "stat_type": stat, "game_id": game, "probability": prob}


def _pred(
    idx: int,
    rating: str = "A",
    prob: float = 0.65,
    stat: str = "hits",
) -> dict:
    """Minimal prediction dict for find_optimal_slips tests."""
    return {
        "player_name": f"Player {idx}",
        "stat_internal": stat,
        "stat_type": stat.replace("_", " ").title(),
        "game_id": f"G{idx}",        # each prediction in its own game → independent
        "probability": prob,
        "win_prob": prob,
        "confidence": prob,
        "rating": rating,
        "opponent": f"OPP{idx}",
    }


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------

class TestBuildCorrelationMatrix(unittest.TestCase):

    def test_diagonal_ones(self):
        legs = [_leg(stat="hits", game="G1"), _leg(stat="total_bases", game="G1")]
        R = build_correlation_matrix(legs)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)

    def test_shape(self):
        legs = [_leg(game=f"G{i}") for i in range(4)]
        R = build_correlation_matrix(legs)
        self.assertEqual(R.shape, (4, 4))

    def test_symmetric(self):
        legs = [
            _leg(stat="hits", game="G1"),
            _leg(stat="total_bases", game="G1"),
            _leg(stat="pitcher_strikeouts", game="G2"),
        ]
        R = build_correlation_matrix(legs)
        np.testing.assert_allclose(R, R.T, atol=1e-12)

    def test_same_game_high_correlation(self):
        """hits + total_bases same game should have correlation ≥ 0.80."""
        legs = [_leg(stat="hits", game="G1"), _leg(stat="total_bases", game="G1")]
        R = build_correlation_matrix(legs)
        self.assertGreaterEqual(R[0, 1], 0.80)

    def test_cross_game_zero_correlation(self):
        """Different game_ids → r = 0."""
        legs = [_leg(stat="hits", game="G1"), _leg(stat="hits", game="G2")]
        R = build_correlation_matrix(legs)
        self.assertAlmostEqual(R[0, 1], 0.0, places=10)

    def test_positive_semi_definite(self):
        """All eigenvalues must be ≥ 0 (within floating-point tolerance)."""
        legs = [
            _leg(stat="hits", game="G1"),
            _leg(stat="total_bases", game="G1"),
            _leg(stat="runs", game="G1"),
            _leg(stat="pitcher_strikeouts", game="G2"),
            _leg(stat="rbis", game="G1"),
        ]
        R = build_correlation_matrix(legs)
        eigvals = np.linalg.eigvalsh(R)
        self.assertTrue(np.all(eigvals >= -1e-6), f"Negative eigenvalue: {eigvals.min()}")

    def test_single_leg(self):
        R = build_correlation_matrix([_leg()])
        self.assertEqual(R.shape, (1, 1))
        self.assertAlmostEqual(float(R[0, 0]), 1.0)

    def test_unknown_stat_same_game_gets_small_positive_r(self):
        """Unrecognised same-game stat pair falls back to 0.10."""
        legs = [
            _leg(stat="unknown_stat_x", game="G1"),
            _leg(stat="unknown_stat_y", game="G1"),
        ]
        R = build_correlation_matrix(legs)
        self.assertGreater(R[0, 1], 0.0)
        self.assertLessEqual(R[0, 1], 0.20)


# ---------------------------------------------------------------------------
# simulate_slip_ev tests
# ---------------------------------------------------------------------------

class TestSimulateSlipEV(unittest.TestCase):

    def test_smoke_two_legs(self):
        """Basic smoke test — must return expected keys."""
        legs = [_leg(0.6, game="G1"), _leg(0.65, game="G2")]
        result = simulate_slip_ev(legs, n_simulations=2_000, seed=42)
        for key in ("ev_payout", "ev_profit", "ev_profit_pct",
                    "hit_rate_all", "win_rates_by_count", "std_dev",
                    "n_simulations", "n_legs"):
            self.assertIn(key, result)

    def test_n_legs_matches_input(self):
        legs = [_leg(game=f"G{i}") for i in range(4)]
        result = simulate_slip_ev(legs, n_simulations=1_000, seed=1)
        self.assertEqual(result["n_legs"], 4)

    def test_win_rates_sum_to_one(self):
        legs = [_leg(0.6, game=f"G{i}") for i in range(3)]
        result = simulate_slip_ev(legs, n_simulations=10_000, seed=2)
        total = sum(result["win_rates_by_count"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_ev_payout_near_certain_legs(self):
        """3 legs each with p≈1 should return ≈ 2.25x (3-flex perfect payout)."""
        legs = [_leg(0.999, game=f"G{i}") for i in range(3)]
        result = simulate_slip_ev(legs, n_simulations=20_000, seed=3)
        self.assertAlmostEqual(result["ev_payout"], 2.25, delta=0.05)

    def test_ev_payout_near_zero_legs(self):
        """3 legs each with p≈0 should give ev_payout ≈ 0."""
        legs = [_leg(0.001, game=f"G{i}") for i in range(3)]
        result = simulate_slip_ev(legs, n_simulations=20_000, seed=4)
        self.assertLess(result["ev_payout"], 0.05)

    def test_empty_legs(self):
        result = simulate_slip_ev([], n_simulations=1_000)
        self.assertEqual(result["n_legs"], 0)
        self.assertEqual(result["ev_profit"], -1.0)

    def test_seeded_reproducibility(self):
        legs = [_leg(0.62, game=f"G{i}") for i in range(4)]
        r1 = simulate_slip_ev(legs, n_simulations=5_000, seed=99)
        r2 = simulate_slip_ev(legs, n_simulations=5_000, seed=99)
        self.assertEqual(r1["ev_payout"], r2["ev_payout"])

    # ------------------------------------------------------------------
    # Known-EV validation
    # ------------------------------------------------------------------

    def test_known_ev_five_independent_60pct_legs(self):
        """Known-EV check: 5 independent 60% legs on 5-flex.

        Analytical EV (closed form, independence assumed):

            EV = Σ_{k=0}^{5}  C(5,k) × p^k × q^(5-k) × payout(k)

        With p = 0.60, q = 0.40 and 5-flex payouts {5:10, 4:2, 3:0.4}:

            k=5: C(5,5)×0.6^5×0.4^0×10   = 0.07776 × 10 = 0.77760
            k=4: C(5,4)×0.6^4×0.4^1× 2   = 0.25920 × 2  = 0.51840
            k=3: C(5,3)×0.6^3×0.4^2× 0.4 = 0.34560 × 0.4= 0.13824
            EV ≈ 1.4342

        The task's "EV ≈ 10 × 0.6^5 = 0.778" refers only to the perfect-hit
        component; the full flex EV including partial payouts is ~1.43.

        Monte Carlo at 100k sims should land within ±0.04 of the analytical
        value at 99% confidence.
        """
        # Different game IDs → zero cross-leg correlation (true independence)
        legs = [_leg(0.6, stat="hits", game=f"GAME_{i}") for i in range(5)]

        p, q = 0.6, 0.4
        payouts_5flex = FLEX_PAYOUTS[5]
        from math import comb
        analytical_ev = sum(
            comb(5, k) * (p ** k) * (q ** (5 - k)) * payouts_5flex.get(k, 0.0)
            for k in range(6)
        )
        # Verify our analytical calculation first
        self.assertAlmostEqual(analytical_ev, 1.4342, delta=0.001)

        result = simulate_slip_ev(legs, n_simulations=100_000, seed=0)
        self.assertAlmostEqual(result["ev_payout"], analytical_ev, delta=0.04)

    def test_known_ev_two_independent_65pct_legs(self):
        """2-flex, p=0.65: EV = 0.65²×3 + 2×0.65×0.35×1.5 = 1.2675 + 0.6825 = 1.95."""
        p, q = 0.65, 0.35
        analytical_ev = p ** 2 * 3.0 + 2 * p * q * 1.5
        self.assertAlmostEqual(analytical_ev, 1.950, delta=0.001)

        legs = [_leg(0.65, game=f"G{i}") for i in range(2)]
        result = simulate_slip_ev(legs, n_simulations=100_000, seed=7)
        self.assertAlmostEqual(result["ev_payout"], analytical_ev, delta=0.04)

    def test_positive_ev_with_strong_edge(self):
        """4 legs each at 70% should yield ev_profit > 0."""
        legs = [_leg(0.70, game=f"G{i}") for i in range(4)]
        result = simulate_slip_ev(legs, n_simulations=50_000, seed=5)
        self.assertGreater(result["ev_payout"], 1.0)

    def test_correlated_legs_different_from_independent(self):
        """Same-game correlated legs should produce measurably different EV
        distribution from independent legs (different std_dev at minimum)."""
        legs_same_game = [
            _leg(0.65, stat="hits", game="SAME"),
            _leg(0.65, stat="total_bases", game="SAME"),
        ]
        legs_independent = [
            _leg(0.65, stat="hits", game="G1"),
            _leg(0.65, stat="hits", game="G2"),
        ]
        r_corr = simulate_slip_ev(legs_same_game, n_simulations=50_000, seed=6)
        r_indep = simulate_slip_ev(legs_independent, n_simulations=50_000, seed=6)

        # High positive correlation (hits + total_bases) → variance increases
        # Both should be numeric floats
        self.assertIsInstance(r_corr["ev_payout"], float)
        self.assertIsInstance(r_indep["ev_payout"], float)
        # Std dev should differ when correlation is non-zero
        self.assertNotAlmostEqual(r_corr["std_dev"], r_indep["std_dev"], places=2)


# ---------------------------------------------------------------------------
# find_optimal_slips tests
# ---------------------------------------------------------------------------

class TestFindOptimalSlips(unittest.TestCase):

    def test_returns_at_most_top_k(self):
        preds = [_pred(i) for i in range(10)]
        slips = find_optimal_slips(preds, top_k=3, slip_sizes=[3], n_simulations=500)
        self.assertLessEqual(len(slips), 3)

    def test_filters_c_and_d_grade(self):
        preds = [_pred(i, rating="C") for i in range(6)]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[3], n_simulations=200)
        self.assertEqual(len(slips), 0)

    def test_a_grade_passes_filter(self):
        preds = [_pred(i, rating="A") for i in range(6)]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[3], n_simulations=200)
        self.assertGreater(len(slips), 0)

    def test_b_grade_passes_filter(self):
        preds = [_pred(i, rating="B") for i in range(6)]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[3], n_simulations=200)
        self.assertGreater(len(slips), 0)

    def test_sorted_by_ev_descending(self):
        preds = [_pred(i) for i in range(8)]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[3], n_simulations=1_000)
        if len(slips) < 2:
            self.skipTest("Too few slips returned to test ordering")
        evs = [s["ev_profit"] for s in slips]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_slip_size_in_output_matches_requested(self):
        preds = [_pred(i) for i in range(6)]
        slips = find_optimal_slips(preds, top_k=10, slip_sizes=[2], n_simulations=500)
        for slip in slips:
            self.assertEqual(slip["slip_size"], 2)

    def test_legs_in_output_have_correct_count(self):
        preds = [_pred(i) for i in range(6)]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[4], n_simulations=500)
        for slip in slips:
            self.assertEqual(len(slip["legs"]), 4)

    def test_empty_predictions_returns_empty(self):
        slips = find_optimal_slips([], top_k=5)
        self.assertEqual(slips, [])

    def test_insufficient_eligible_predictions(self):
        """Only 1 prediction, slip_size=3 → nothing returned."""
        preds = [_pred(0, rating="A")]
        slips = find_optimal_slips(preds, top_k=5, slip_sizes=[3], n_simulations=200)
        self.assertEqual(slips, [])

    def test_multiple_slip_sizes(self):
        preds = [_pred(i) for i in range(8)]
        # top_k=100 > C(8,2)+C(8,3)=84 → all combinations returned
        slips = find_optimal_slips(
            preds, top_k=100, slip_sizes=[2, 3], n_simulations=500
        )
        sizes = {s["slip_size"] for s in slips}
        self.assertIn(2, sizes)
        self.assertIn(3, sizes)


# ---------------------------------------------------------------------------
# kelly_fraction tests
# ---------------------------------------------------------------------------

class TestKellyFraction(unittest.TestCase):

    def test_positive_edge_returns_nonzero_wager(self):
        result = kelly_fraction(ev=1.20, odds=10.0, bankroll=100)
        self.assertGreater(result["kelly_pct"], 0)
        self.assertGreater(result["recommended_wager"], 0)

    def test_no_edge_returns_zero(self):
        result = kelly_fraction(ev=0.95, odds=10.0, bankroll=100)
        self.assertEqual(result["kelly_pct"], 0.0)
        self.assertEqual(result["recommended_wager"], 0.0)

    def test_exactly_breakeven_returns_zero(self):
        result = kelly_fraction(ev=1.0, odds=5.0, bankroll=100)
        self.assertEqual(result["kelly_pct"], 0.0)

    def test_quarter_kelly_is_quarter_of_full_kelly(self):
        result = kelly_fraction(ev=1.5, odds=5.0, bankroll=100)
        # Both values are independently rounded to 2 decimal places,
        # so compare with delta=0.02 to account for rounding error.
        self.assertAlmostEqual(
            result["quarter_kelly_pct"],
            result["kelly_pct"] / 4,
            delta=0.02,
        )

    def test_half_kelly_is_half_of_full_kelly(self):
        result = kelly_fraction(ev=1.5, odds=5.0, bankroll=100)
        self.assertAlmostEqual(
            result["half_kelly_pct"],
            result["kelly_pct"] / 2,
            places=5,
        )

    def test_kelly_capped_at_25_pct(self):
        """Absurdly high edge should be capped at 25% to avoid ruin."""
        result = kelly_fraction(ev=9.0, odds=10.0, bankroll=100)
        self.assertLessEqual(result["kelly_pct"], 25.0)

    def test_bankroll_scales_wager_linearly(self):
        r100 = kelly_fraction(ev=1.30, odds=5.0, bankroll=100)
        r200 = kelly_fraction(ev=1.30, odds=5.0, bankroll=200)
        self.assertAlmostEqual(r200["max_wager"], r100["max_wager"] * 2, places=4)

    def test_recommended_wager_is_quarter_kelly_times_bankroll(self):
        result = kelly_fraction(ev=1.4, odds=10.0, bankroll=100)
        expected = result["quarter_kelly_pct"] / 100 * 100
        self.assertAlmostEqual(result["recommended_wager"], expected, places=4)

    def test_edge_pct_reflects_ev(self):
        result = kelly_fraction(ev=1.15, odds=10.0, bankroll=100)
        self.assertAlmostEqual(result["edge_pct"], 15.0, places=5)

    def test_invalid_odds_returns_zero(self):
        result = kelly_fraction(ev=1.5, odds=0.5, bankroll=100)
        self.assertEqual(result["kelly_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
