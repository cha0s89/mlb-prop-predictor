"""Tests for cold-bat bounce-back signal in tail_signals.py and predictor.py."""

import pytest
from src.bounce_back import detect_bounce_back


# ─── Helpers ────────────────────────────────────────────────────────────────

def _elite_hitter(avg=0.310, pa=350, recent_game=None) -> dict:
    base = {"avg": avg, "pa": pa}
    if recent_game is not None:
        base["recent_game"] = recent_game
    return base


# ─── No adjustment cases ─────────────────────────────────────────────────────

class TestNoAdjustment:
    def test_pitcher_prop_ignored(self):
        b = _elite_hitter(recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "pitcher_strikeouts") == 1.0

    def test_batter_strikeouts_ignored(self):
        b = _elite_hitter(recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "batter_strikeouts") == 1.0

    def test_walks_allowed_ignored(self):
        b = _elite_hitter(recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "walks_allowed") == 1.0

    def test_low_avg_hitter_no_boost(self):
        b = _elite_hitter(avg=0.240, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == 1.0

    def test_too_few_pa_no_boost(self):
        b = _elite_hitter(avg=0.320, pa=30, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == 1.0

    def test_no_recent_game_data(self):
        b = {"avg": 0.320, "pa": 300}
        assert detect_bounce_back(b, "hits") == 1.0

    def test_good_recent_game_no_boost(self):
        """Player who went 2-for-4 yesterday should get no bounce-back."""
        b = _elite_hitter(recent_game={"hits": 2, "ab": 4, "total_bases": 3, "h_r_rbi": 3})
        assert detect_bounce_back(b, "hits") == 1.0

    def test_only_one_ab_not_enough(self):
        """0-for-1 doesn't qualify as an outlier bad game."""
        b = _elite_hitter(recent_game={"hits": 0, "ab": 1, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == 1.0


# ─── Multiplier scale tests ───────────────────────────────────────────────────

class TestMultiplierScale:
    def test_borderline_280_hitter_5pct_boost(self):
        """.280 hitter hitless in 4 ABs → 1.05 multiplier."""
        b = _elite_hitter(avg=0.280, pa=200, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == pytest.approx(1.05)

    def test_300_hitter_7pct_boost(self):
        """.300 hitter → 1.07 multiplier."""
        b = _elite_hitter(avg=0.300, pa=300, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == pytest.approx(1.07)

    def test_320_hitter_10pct_boost(self):
        """.320+ hitter → maximum 1.10 multiplier."""
        b = _elite_hitter(avg=0.330, pa=400, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == pytest.approx(1.10)

    def test_exactly_320_gets_max(self):
        b = _elite_hitter(avg=0.320, pa=350, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == pytest.approx(1.10)

    def test_exactly_300_gets_mid(self):
        b = _elite_hitter(avg=0.300, pa=300, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits") == pytest.approx(1.07)


# ─── Trigger conditions ───────────────────────────────────────────────────────

class TestTriggerConditions:
    def test_zero_total_bases_triggers(self):
        """0 TB in 2+ ABs qualifies even without explicit hits=0."""
        b = _elite_hitter(avg=0.310, pa=300, recent_game={"hits": 0, "ab": 3, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "total_bases") > 1.0

    def test_zero_h_r_rbi_triggers(self):
        """0 H+R+RBI in 2+ ABs qualifies."""
        b = _elite_hitter(avg=0.310, pa=300, recent_game={"hits": 0, "ab": 3, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hits_runs_rbis") > 1.0

    def test_applies_to_home_runs_prop(self):
        b = _elite_hitter(avg=0.305, pa=280, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "home_runs") > 1.0

    def test_applies_to_fantasy_score(self):
        b = _elite_hitter(avg=0.305, pa=280, recent_game={"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0})
        assert detect_bounce_back(b, "hitter_fantasy_score") > 1.0

    def test_hitless_in_3ab_does_not_trigger_hitless_rule(self):
        """Hitless-in-4+ rule requires 4 ABs. 3 ABs hitless is not enough."""
        b = _elite_hitter(avg=0.310, pa=300, recent_game={"hits": 0, "ab": 3, "total_bases": 1, "h_r_rbi": 1})
        # Has a TB so zero_total_bases = False; has h_r_rbi=1 so zero_production = False
        # only hitless check would trigger but requires ab >= 4
        assert detect_bounce_back(b, "hits") == 1.0


# ─── Integration: predictor wires bounce-back into generate_prediction ────────

class TestPredictorIntegration:
    def test_bounce_back_raises_projection(self):
        """generate_prediction with an elite hitter after a cold game projects higher."""
        from src.predictor import generate_prediction

        cold_game = {"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0}
        batter_cold = {"avg": 0.320, "pa": 350, "recent_game": cold_game,
                       "xba": 0.305, "k_rate": 18.0, "bb_rate": 9.0}
        batter_normal = {"avg": 0.320, "pa": 350,
                         "xba": 0.305, "k_rate": 18.0, "bb_rate": 9.0}

        result_cold = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter_cold,
        )
        result_normal = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter_normal,
        )

        assert result_cold["projection"] > result_normal["projection"], (
            "Bounce-back should raise projection for elite hitter after cold game"
        )

    def test_bounce_back_does_not_affect_pitcher_prop(self):
        """Even if batter_profile is somehow passed for a pitcher prop, no bounce-back."""
        from src.predictor import generate_prediction

        cold_game = {"hits": 0, "ab": 4, "total_bases": 0, "h_r_rbi": 0}
        pitcher = {"avg": 0.320, "pa": 350, "recent_game": cold_game,
                   "era": 3.20, "fip": 3.10, "k9": 9.5, "ip": 150}

        # Should not raise — the bounce-back multiplier is filtered to batter props
        result = generate_prediction(
            "Test Pitcher", "Strikeouts", "pitcher_strikeouts", 6.5,
            pitcher_profile=pitcher,
        )
        assert "bounce_back_mult" not in result.get("proj_result", result)

    def test_no_bounce_back_stored_for_normal_game(self):
        """bounce_back_mult key absent when no cold game was detected."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.310, "pa": 300, "xba": 0.300}
        result = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
        )
        assert result.get("bounce_back_mult") is None
