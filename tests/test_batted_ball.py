"""
Unit tests for src/batted_ball.py — pitcher-batter batted-ball interaction modeling.

Covers:
  1. Graceful fallback when data is missing or insufficient
  2. Flyball-flyball matchup boosts TB/HR
  3. Groundball pitcher suppresses TB/HR
  4. HR/FB% interaction for flyball batters
  5. Line-drive hitter boosts hits
  6. Whiff-rate × K% interaction for batter strikeouts
  7. Bayesian regression pulls extreme rates toward league average
  8. Multiplier caps (no extreme outliers)
  9. Integration: generate_prediction propagates batted_ball_mult
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.batted_ball import (
    compute_batted_ball_interaction,
    LG_GB_PCT, LG_FB_PCT, LG_LD_PCT, LG_HR_FB, LG_SWSTR, LG_K_RATE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batter(gb=43.0, fb=36.0, ld=21.0, pa=400, k_rate=22.7) -> dict:
    return {"gb_pct": gb, "fb_pct": fb, "ld_pct": ld, "pa": pa, "k_rate": k_rate}


def _pitcher(gb=43.0, fb=36.0, hr_fb=12.0, ip=120.0, swstr=11.3) -> dict:
    return {
        "gb_pct": gb,
        "fb_pct": fb,
        "hr_fb": hr_fb,
        "ip": ip,
        "recent_swstr_pct": swstr,
    }


class TestFallback(unittest.TestCase):
    """Graceful fallback: missing or sparse data returns neutral multipliers."""

    def test_both_none_returns_defaults(self):
        r = compute_batted_ball_interaction(None, None)
        self.assertEqual(r["tb_mult"],   1.0)
        self.assertEqual(r["hr_mult"],   1.0)
        self.assertEqual(r["hits_mult"], 1.0)
        self.assertEqual(r["k_mult"],    1.0)

    def test_empty_dicts_return_defaults(self):
        r = compute_batted_ball_interaction({}, {})
        self.assertEqual(r["tb_mult"],   1.0)
        self.assertEqual(r["hr_mult"],   1.0)

    def test_sparse_batter_no_crash(self):
        """Batter with no batted-ball data should not raise."""
        batter = {"pa": 10, "k_rate": 22.0}   # < 30 BBE threshold
        pitcher = _pitcher()
        r = compute_batted_ball_interaction(batter, pitcher)
        self.assertIsInstance(r["tb_mult"], float)

    def test_sparse_pitcher_no_crash(self):
        """Pitcher with very few IP should not raise."""
        pitcher = {"ip": 5.0, "gb_pct": 50.0}  # < 30 BBE
        r = compute_batted_ball_interaction(_batter(), pitcher)
        self.assertIsInstance(r["tb_mult"], float)

    def test_missing_gb_fb_values_treated_as_zero(self):
        """Profiles without gb_pct/fb_pct keys should fall back gracefully."""
        batter = {"pa": 400, "k_rate": 22.0}          # no gb_pct / fb_pct
        pitcher = {"ip": 120.0, "recent_swstr_pct": 0.0}
        r = compute_batted_ball_interaction(batter, pitcher)
        # source should indicate no usable signal
        self.assertIn(r["source"], ("insufficient_data", "no_signal", "no_data"))

    def test_source_key_always_present(self):
        r = compute_batted_ball_interaction(_batter(), _pitcher())
        self.assertIn("source", r)
        self.assertIsInstance(r["source"], str)


class TestFlybballMatchup(unittest.TestCase):
    """Flyball hitter vs flyball pitcher → TB and HR boost."""

    def _flyball_result(self, b_fb=45.0, p_fb=45.0):
        return compute_batted_ball_interaction(
            _batter(fb=b_fb, gb=100 - b_fb - 21),
            _pitcher(fb=p_fb, gb=100 - p_fb - 12),
        )

    def test_flyball_matchup_boosts_tb(self):
        r = self._flyball_result(b_fb=46.0, p_fb=46.0)
        self.assertGreater(r["tb_mult"], 1.0, "TB should be boosted in flyball matchup")

    def test_flyball_matchup_boosts_hr_more_than_tb(self):
        r = self._flyball_result(b_fb=46.0, p_fb=46.0)
        self.assertGreater(r["hr_mult"], r["tb_mult"], "HR boost should exceed TB boost")

    def test_flyball_boost_bounded(self):
        """Even extreme flyball matchup should not boost more than 15%."""
        r = self._flyball_result(b_fb=55.0, p_fb=55.0)
        self.assertLessEqual(r["tb_mult"], 1.15)
        self.assertLessEqual(r["hr_mult"], 1.20)

    def test_league_avg_fb_no_boost(self):
        """League-average FB% on both sides → no flyball boost."""
        r = self._flyball_result(b_fb=LG_FB_PCT, p_fb=LG_FB_PCT)
        self.assertAlmostEqual(r["tb_mult"], 1.0, places=3)

    def test_flyball_batter_groundball_pitcher_no_boost(self):
        """Flyball batter vs groundball pitcher should NOT get the flyball boost."""
        r = compute_batted_ball_interaction(
            _batter(fb=50.0, gb=30.0),
            _pitcher(fb=28.0, gb=62.0),
        )
        # tb_mult should not be > 1 from flyball signal (GB pitcher suppression may dominate)
        # Just verify the flyball-flyball condition is not triggered
        self.assertNotIn("flyball_matchup", r.get("source", ""))


class TestGroundballPitcher(unittest.TestCase):
    """High GB% pitcher suppresses TB/HR."""

    def test_gb_pitcher_suppresses_tb(self):
        r = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=58.0, fb=22.0),
        )
        self.assertLess(r["tb_mult"], 1.0, "TB should be suppressed by groundball pitcher")

    def test_gb_pitcher_suppresses_hr_more_than_tb(self):
        r = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=58.0, fb=22.0),
        )
        # HR multiplier should be further below 1.0 than TB multiplier
        self.assertLess(r["hr_mult"], r["tb_mult"])

    def test_gb_pitcher_suppression_bounded(self):
        """Even 65% GB pitcher should not suppress more than ~10%."""
        r = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=65.0, fb=18.0),
        )
        self.assertGreaterEqual(r["tb_mult"], 0.90)
        self.assertGreaterEqual(r["hr_mult"], 0.85)

    def test_league_avg_gb_pitcher_no_suppression(self):
        """League-average GB% pitcher → no suppression."""
        r = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=LG_GB_PCT, fb=LG_FB_PCT, swstr=0.0),
        )
        # With swstr=0 the only active signals are batted-ball rates
        self.assertAlmostEqual(r["tb_mult"], 1.0, places=3)


class TestHrFbInteraction(unittest.TestCase):
    """Pitcher HR/FB% × flyball batter = additional HR uplift."""

    def test_high_hr_fb_pitcher_flyball_batter_boosts_hr(self):
        r = compute_batted_ball_interaction(
            _batter(fb=46.0, gb=32.0),
            _pitcher(hr_fb=18.0, fb=40.0, gb=38.0, swstr=0.0),
        )
        self.assertGreater(r["hr_mult"], 1.0)

    def test_low_hr_fb_pitcher_suppresses_hr_for_flyball_batter(self):
        r = compute_batted_ball_interaction(
            _batter(fb=46.0, gb=32.0),
            _pitcher(hr_fb=6.0, fb=40.0, gb=38.0, swstr=0.0),
        )
        self.assertLess(r["hr_mult"], 1.0)

    def test_hr_fb_has_no_effect_on_groundball_batter(self):
        """Groundball batter (low FB%) should see minimal HR/FB interaction."""
        r_gb = compute_batted_ball_interaction(
            _batter(fb=25.0, gb=58.0),
            _pitcher(hr_fb=20.0, fb=40.0, gb=38.0, swstr=0.0),
        )
        r_fb = compute_batted_ball_interaction(
            _batter(fb=50.0, gb=30.0),
            _pitcher(hr_fb=20.0, fb=40.0, gb=38.0, swstr=0.0),
        )
        # Groundball batter should benefit far less from high HR/FB pitcher
        self.assertLess(r_gb["hr_mult"], r_fb["hr_mult"])


class TestLineDriveHitter(unittest.TestCase):
    """LD% above 22% boosts hits via BABIP."""

    def test_high_ld_boosts_hits(self):
        r = compute_batted_ball_interaction(
            _batter(ld=28.0, pa=500),
            _pitcher(swstr=0.0),
        )
        self.assertGreater(r["hits_mult"], 1.0)

    def test_league_avg_ld_no_boost(self):
        r = compute_batted_ball_interaction(
            _batter(ld=LG_LD_PCT, pa=500),
            _pitcher(swstr=0.0),
        )
        self.assertAlmostEqual(r["hits_mult"], 1.0, places=3)

    def test_ld_boost_bounded(self):
        """Even extreme LD% should not boost hits > 4%."""
        r = compute_batted_ball_interaction(
            _batter(ld=35.0, pa=600),
            _pitcher(swstr=0.0),
        )
        self.assertLessEqual(r["hits_mult"], 1.05)


class TestWhiffInteraction(unittest.TestCase):
    """Pitcher swinging-strike rate × batter K% drives strikeout multiplier."""

    def test_high_whiff_pitcher_high_k_batter_amplifies_k(self):
        r = compute_batted_ball_interaction(
            _batter(k_rate=28.0),
            _pitcher(swstr=15.0),
        )
        self.assertGreater(r["k_mult"], 1.0)

    def test_low_whiff_pitcher_low_k_batter_reduces_k(self):
        r = compute_batted_ball_interaction(
            _batter(k_rate=14.0),
            _pitcher(swstr=8.0),
        )
        self.assertLess(r["k_mult"], 1.0)

    def test_league_avg_whiff_league_avg_k_neutral(self):
        r = compute_batted_ball_interaction(
            _batter(k_rate=LG_K_RATE),
            _pitcher(swstr=LG_SWSTR),
        )
        # Should be close to 1.0 — no extreme deviation
        self.assertAlmostEqual(r["k_mult"], 1.0, delta=0.05)

    def test_k_mult_cap(self):
        """K multiplier must stay within ±20%."""
        r = compute_batted_ball_interaction(
            _batter(k_rate=38.0),
            _pitcher(swstr=20.0),
        )
        self.assertLessEqual(r["k_mult"], 1.20)

    def test_k_mult_floor(self):
        r = compute_batted_ball_interaction(
            _batter(k_rate=5.0),
            _pitcher(swstr=5.0),
        )
        self.assertGreaterEqual(r["k_mult"], 0.85)

    def test_no_swstr_data_no_k_interaction(self):
        """Pitcher without swinging-strike data → k_mult stays at 1.0."""
        r = compute_batted_ball_interaction(
            _batter(k_rate=30.0),
            _pitcher(swstr=0.0),
        )
        self.assertEqual(r["k_mult"], 1.0)


class TestBayesianRegression(unittest.TestCase):
    """Extreme raw rates should be pulled toward league average."""

    def test_extreme_gb_pct_regresses(self):
        """A pitcher with 80% GB (absurdly high) over 50 IP should regress substantially."""
        # 50 IP → ~135 BBE → moderate regression
        r_extreme = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=80.0, fb=5.0, ip=50.0, swstr=0.0),
        )
        r_moderate = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=65.0, fb=18.0, ip=50.0, swstr=0.0),
        )
        # The extreme case should suppress TB/HR MORE than moderate, but
        # regression means the extreme case is pulled toward the moderate one
        # (both should suppress; extreme should suppress more)
        self.assertLessEqual(r_extreme["tb_mult"], r_moderate["tb_mult"])

    def test_small_sample_regresses_to_neutral(self):
        """Pitcher with only 10 IP (< 30 BBE threshold) → no signal."""
        r = compute_batted_ball_interaction(
            _batter(),
            _pitcher(gb=70.0, fb=10.0, ip=10.0, swstr=0.0),
        )
        # < 30 BBE threshold means pitcher data is ignored for GB/FB signals
        self.assertAlmostEqual(r["tb_mult"], 1.0, places=3)


class TestGeneratePredictionIntegration(unittest.TestCase):
    """End-to-end: generate_prediction propagates batted_ball_mult."""

    def test_flyball_matchup_propagates_to_total_bases(self):
        """Flyball-flyball matchup should produce a higher TB projection than neutral."""
        from src.predictor import generate_prediction

        neutral_batter = {
            "pa": 500, "slg": 0.450, "iso": 0.180, "avg": 0.270, "obp": 0.340,
            "k_rate": 20.0, "bb_rate": 8.0, "woba": 0.340, "wrc_plus": 120,
            "xslg": 0.450, "xba": 0.270, "xwoba": 0.340,
            "recent_barrel_rate": 10.0, "recent_ev90": 106.0,
            "recent_hard_hit_pct": 40.0, "recent_sweet_spot_pct": 34.0,
            "gb_pct": LG_GB_PCT, "fb_pct": LG_FB_PCT, "ld_pct": LG_LD_PCT,
        }
        flyball_batter = dict(neutral_batter)
        flyball_batter.update({"gb_pct": 28.0, "fb_pct": 50.0, "ld_pct": 22.0})

        flyball_pitcher = {
            "ip": 150.0, "era": 4.00, "fip": 3.90, "k9": 9.0, "bb9": 3.0,
            "k_pct": 24.0, "bb_pct": 8.0, "whip": 1.20, "hr9": 1.10,
            "gb_pct": 28.0, "fb_pct": 50.0, "hr_fb": 14.0,
            "recent_swstr_pct": 0.0,
        }

        r_neutral = generate_prediction(
            "Test Batter", "total_bases", "total_bases", 2.5,
            batter_profile=neutral_batter,
            opp_pitcher_profile={**flyball_pitcher, "gb_pct": LG_GB_PCT, "fb_pct": LG_FB_PCT},
            lineup_pos=4,
        )
        r_flyball = generate_prediction(
            "Test Batter", "total_bases", "total_bases", 2.5,
            batter_profile=flyball_batter,
            opp_pitcher_profile=flyball_pitcher,
            lineup_pos=4,
        )

        self.assertGreater(
            r_flyball["projection"],
            r_neutral["projection"],
            "Flyball-flyball matchup should produce higher TB projection",
        )

    def test_gb_pitcher_suppresses_home_runs(self):
        """Groundball pitcher should suppress HR projection vs league-avg pitcher."""
        from src.predictor import generate_prediction

        batter = {
            "pa": 500, "hr": 25, "slg": 0.500, "iso": 0.230, "avg": 0.270,
            "k_rate": 22.0, "bb_rate": 9.0, "woba": 0.360, "wrc_plus": 140,
            "xslg": 0.500, "xba": 0.270, "xwoba": 0.360,
            "recent_barrel_rate": 14.0, "recent_ev90": 108.0,
            "recent_hard_hit_pct": 48.0, "recent_sweet_spot_pct": 36.0,
            "gb_pct": LG_GB_PCT, "fb_pct": LG_FB_PCT, "ld_pct": LG_LD_PCT,
        }
        avg_pitcher = {
            "ip": 140.0, "era": 4.10, "fip": 4.00, "k9": 8.5, "bb9": 3.2,
            "k_pct": 22.5, "bb_pct": 8.5, "whip": 1.28, "hr9": 1.15,
            "gb_pct": LG_GB_PCT, "fb_pct": LG_FB_PCT, "hr_fb": LG_HR_FB,
            "recent_swstr_pct": 0.0,
        }
        gb_pitcher = dict(avg_pitcher)
        gb_pitcher.update({"gb_pct": 58.0, "fb_pct": 22.0, "hr_fb": 8.0})

        r_avg = generate_prediction(
            "Power Hitter", "home_runs", "home_runs", 0.5,
            batter_profile=batter, opp_pitcher_profile=avg_pitcher, lineup_pos=4,
        )
        r_gb = generate_prediction(
            "Power Hitter", "home_runs", "home_runs", 0.5,
            batter_profile=batter, opp_pitcher_profile=gb_pitcher, lineup_pos=4,
        )

        self.assertLess(
            r_gb["projection"],
            r_avg["projection"],
            "Groundball pitcher should suppress HR projection",
        )

    def test_no_crash_on_missing_batted_ball_data(self):
        """generate_prediction must complete even when profiles lack GB/FB data."""
        from src.predictor import generate_prediction

        batter = {
            "pa": 400, "slg": 0.420, "iso": 0.160, "avg": 0.260, "obp": 0.330,
            "k_rate": 22.0, "bb_rate": 8.0, "woba": 0.320, "wrc_plus": 110,
            # NO gb_pct / fb_pct / ld_pct
        }
        pitcher = {"ip": 130.0, "era": 4.20, "fip": 4.10, "k9": 8.0, "bb9": 3.0,
                   "k_pct": 22.0, "bb_pct": 8.0, "whip": 1.25, "hr9": 1.10}

        r = generate_prediction(
            "Test", "total_bases", "total_bases", 2.5,
            batter_profile=batter, opp_pitcher_profile=pitcher, lineup_pos=4,
        )
        self.assertIn("projection", r)
        self.assertGreater(r["projection"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
