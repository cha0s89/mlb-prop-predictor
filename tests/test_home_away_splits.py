"""Unit tests for home/away split adjustment module (src/home_away_splits.py)."""

from unittest.mock import patch
import pytest

from src.home_away_splits import (
    _bayesian_blend,
    _batter_mult,
    _pitcher_mult,
    _rate_mult,
    get_home_away_split_multiplier,
    clear_splits_cache,
    _CAP_LOW,
    _CAP_HIGH,
    _PRIOR_PA,
)


# ─── _bayesian_blend ─────────────────────────────────────────────────────────

class TestBayesianBlend:
    def test_zero_sample_returns_neutral(self):
        assert _bayesian_blend(1.20, 0) == 1.0

    def test_huge_sample_approaches_raw(self):
        # 10000 PA: blend should be very close to raw
        blended = _bayesian_blend(1.10, 10000)
        assert blended == pytest.approx(1.10, abs=0.002)

    def test_small_sample_pulls_toward_one(self):
        # raw mult of 1.20 with only 20 PA should be pulled significantly toward 1.0
        blended = _bayesian_blend(1.20, 20)
        assert 1.0 < blended < 1.20

    def test_exact_prior_pa_splits_halfway(self):
        # sample == _PRIOR_PA → blended = (PA*raw + PA*1) / (2*PA) = (raw+1)/2
        raw = 1.20
        expected = (raw + 1.0) / 2
        blended = _bayesian_blend(raw, _PRIOR_PA)
        assert blended == pytest.approx(expected, abs=0.001)

    def test_cap_high_enforced(self):
        # Even with a huge raw multiplier, output can't exceed 1.10
        result = _bayesian_blend(2.0, 500)
        assert result == _CAP_HIGH

    def test_cap_low_enforced(self):
        result = _bayesian_blend(0.5, 500)
        assert result == _CAP_LOW

    def test_neutral_raw_returns_one(self):
        assert _bayesian_blend(1.0, 100) == pytest.approx(1.0)


# ─── _rate_mult ───────────────────────────────────────────────────────────────

class TestRateMult:
    def test_equal_home_away_rates_returns_one(self):
        raw, sample = _rate_mult(10, 10, 100, 100, is_home=True)
        assert raw == pytest.approx(1.0)
        assert sample == 100

    def test_home_better_returns_above_one(self):
        # home: 30 hits / 100 AB = .300, away: 20/100 = .200, overall: .250
        raw, sample = _rate_mult(30, 20, 100, 100, is_home=True)
        assert raw == pytest.approx(1.20)
        assert sample == 100

    def test_away_game_uses_away_rate(self):
        # away: 20 hits / 100 AB = .200, overall = .250
        raw, sample = _rate_mult(30, 20, 100, 100, is_home=False)
        assert raw == pytest.approx(0.80)
        assert sample == 100

    def test_zero_denom_returns_neutral(self):
        raw, sample = _rate_mult(10, 10, 0, 100, is_home=True)
        assert raw == 1.0
        assert sample == 0.0

    def test_zero_total_count_returns_neutral(self):
        raw, sample = _rate_mult(0, 0, 100, 100, is_home=True)
        assert raw == 1.0
        assert sample == 0.0


# ─── _batter_mult ─────────────────────────────────────────────────────────────

def _home_stat(**kwargs):
    base = {
        "plateAppearances": 200, "atBats": 180,
        "hits": 54, "homeRuns": 10, "doubles": 12, "triples": 1,
        "strikeOuts": 40, "baseOnBalls": 20, "runs": 30, "rbi": 28,
        "avg": "0.300", "slg": "0.511",
    }
    base.update(kwargs)
    return base


def _away_stat(**kwargs):
    base = {
        "plateAppearances": 200, "atBats": 180,
        "hits": 45, "homeRuns": 6, "doubles": 9, "triples": 1,
        "strikeOuts": 40, "baseOnBalls": 20, "runs": 22, "rbi": 20,
        "avg": "0.250", "slg": "0.383",
    }
    base.update(kwargs)
    return base


class TestBatterMult:
    def test_hits_home_better(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "hits", is_home=True)
        # home: 54/180=.300, away: 45/180=.250, overall: 99/360=.275 → raw≈1.09
        assert mult > 1.0

    def test_hits_away_worse(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "hits", is_home=False)
        assert mult < 1.0

    def test_total_bases_home_better(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "total_bases", is_home=True)
        assert mult > 1.0

    def test_home_runs_home_better(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "home_runs", is_home=True)
        assert mult > 1.0

    def test_doubles_away_calculation(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "doubles", is_home=False)
        assert mult < 1.0

    def test_singles_derived_correctly(self):
        # singles = hits - hr - dbl - trp
        # home: 54 - 10 - 12 - 1 = 31; away: 45 - 6 - 9 - 1 = 29
        mult = _batter_mult(_home_stat(), _away_stat(), "singles", is_home=True)
        assert mult > 1.0  # more singles at home

    def test_batter_strikeouts_equal_returns_near_one(self):
        # k equal: 40 home, 40 away → raw mult = 1.0
        mult = _batter_mult(_home_stat(), _away_stat(), "batter_strikeouts", is_home=True)
        assert mult == pytest.approx(1.0, abs=0.01)

    def test_walks_home_equal_returns_near_one(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "walks", is_home=True)
        assert mult == pytest.approx(1.0, abs=0.01)

    def test_runs_home_better(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "runs", is_home=True)
        assert mult > 1.0

    def test_too_few_pa_returns_one(self):
        home = _home_stat(plateAppearances=5, atBats=4)
        away = _away_stat(plateAppearances=4, atBats=4)
        mult = _batter_mult(home, away, "hits", is_home=True)
        assert mult == 1.0

    def test_unknown_prop_returns_one(self):
        mult = _batter_mult(_home_stat(), _away_stat(), "pitcher_strikeouts", is_home=True)
        assert mult == 1.0

    def test_cap_high_enforced(self):
        # Extreme split: 0.400 home vs 0.150 away — huge advantage
        home = _home_stat(hits=80, atBats=200, plateAppearances=220)
        away = _away_stat(hits=30, atBats=200, plateAppearances=220)
        mult = _batter_mult(home, away, "hits", is_home=True)
        assert mult <= _CAP_HIGH

    def test_cap_low_enforced(self):
        home = _home_stat(hits=10, atBats=200, plateAppearances=220)
        away = _away_stat(hits=80, atBats=200, plateAppearances=220)
        mult = _batter_mult(home, away, "hits", is_home=True)
        assert mult >= _CAP_LOW


# ─── _pitcher_mult ────────────────────────────────────────────────────────────

def _phome(**kwargs):
    base = {
        "inningsPitched": 60.0,
        "strikeOuts": 70, "earnedRuns": 18, "baseOnBalls": 15, "hits": 50,
        "strikeoutsPer9Inn": "10.50", "era": "2.70",
    }
    base.update(kwargs)
    return base


def _paway(**kwargs):
    base = {
        "inningsPitched": 60.0,
        "strikeOuts": 50, "earnedRuns": 28, "baseOnBalls": 22, "hits": 65,
        "strikeoutsPer9Inn": "7.50", "era": "4.20",
    }
    base.update(kwargs)
    return base


class TestPitcherMult:
    def test_strikeouts_home_better(self):
        # pitcher Ks more at home → projection boost when is_home=True
        mult = _pitcher_mult(_phome(), _paway(), "pitcher_strikeouts", is_home=True)
        assert mult > 1.0

    def test_strikeouts_away_worse(self):
        mult = _pitcher_mult(_phome(), _paway(), "pitcher_strikeouts", is_home=False)
        assert mult < 1.0

    def test_earned_runs_home_better(self):
        # lower ERA at home → lower ER projection at home
        mult = _pitcher_mult(_phome(), _paway(), "earned_runs", is_home=True)
        assert mult < 1.0  # fewer ERs expected at home

    def test_earned_runs_away_worse(self):
        mult = _pitcher_mult(_phome(), _paway(), "earned_runs", is_home=False)
        assert mult > 1.0

    def test_walks_allowed_home(self):
        mult = _pitcher_mult(_phome(), _paway(), "walks_allowed", is_home=True)
        assert mult < 1.0  # fewer walks at home

    def test_hits_allowed_home(self):
        mult = _pitcher_mult(_phome(), _paway(), "hits_allowed", is_home=True)
        assert mult < 1.0

    def test_too_few_ip_returns_one(self):
        home = _phome(inningsPitched=2.0)
        away = _paway(inningsPitched=2.0)
        mult = _pitcher_mult(home, away, "pitcher_strikeouts", is_home=True)
        assert mult == 1.0

    def test_unknown_prop_returns_one(self):
        mult = _pitcher_mult(_phome(), _paway(), "stolen_bases", is_home=True)
        assert mult == 1.0

    def test_cap_enforced(self):
        home = _phome(strikeOuts=200, inningsPitched=60)
        away = _paway(strikeOuts=10, inningsPitched=60)
        mult = _pitcher_mult(home, away, "pitcher_strikeouts", is_home=True)
        assert mult <= _CAP_HIGH


# ─── get_home_away_split_multiplier (public API) ──────────────────────────────

MOCK_HITTING_SPLITS = {
    "home": {
        "plateAppearances": 200, "atBats": 180,
        "hits": 54, "homeRuns": 10, "doubles": 12, "triples": 1,
        "strikeOuts": 40, "baseOnBalls": 20, "runs": 30, "rbi": 28,
        "avg": "0.300", "slg": "0.511",
    },
    "away": {
        "plateAppearances": 200, "atBats": 180,
        "hits": 45, "homeRuns": 6, "doubles": 9, "triples": 1,
        "strikeOuts": 40, "baseOnBalls": 20, "runs": 22, "rbi": 20,
        "avg": "0.250", "slg": "0.383",
    },
}

MOCK_PITCHING_SPLITS = {
    "home": {
        "inningsPitched": 60.0,
        "strikeOuts": 70, "earnedRuns": 18, "baseOnBalls": 15, "hits": 50,
    },
    "away": {
        "inningsPitched": 60.0,
        "strikeOuts": 50, "earnedRuns": 28, "baseOnBalls": 22, "hits": 65,
    },
}


class TestPublicAPI:
    def setup_method(self):
        clear_splits_cache()

    def test_invalid_player_id_returns_one(self):
        assert get_home_away_split_multiplier(0, True, "hits") == 1.0
        assert get_home_away_split_multiplier(-1, True, "hits") == 1.0

    def test_unsupported_batter_prop_returns_one(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_HITTING_SPLITS):
            result = get_home_away_split_multiplier(12345, True, "stolen_bases")
            assert result == 1.0

    def test_unsupported_pitcher_prop_returns_one(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_PITCHING_SPLITS):
            result = get_home_away_split_multiplier(12345, True, "pitching_outs", is_pitcher=True)
            assert result == 1.0

    def test_batter_home_hits_boost(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_HITTING_SPLITS):
            result = get_home_away_split_multiplier(12345, is_home=True, prop_type="hits")
            assert result > 1.0

    def test_batter_away_hits_reduction(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_HITTING_SPLITS):
            result = get_home_away_split_multiplier(12345, is_home=False, prop_type="hits")
            assert result < 1.0

    def test_pitcher_home_strikeouts_boost(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_PITCHING_SPLITS):
            result = get_home_away_split_multiplier(
                12345, is_home=True, prop_type="pitcher_strikeouts", is_pitcher=True
            )
            assert result > 1.0

    def test_pitcher_home_earned_runs_reduction(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_PITCHING_SPLITS):
            result = get_home_away_split_multiplier(
                12345, is_home=True, prop_type="earned_runs", is_pitcher=True
            )
            assert result < 1.0

    def test_api_failure_returns_one(self):
        with patch("src.home_away_splits._fetch_splits", return_value={}):
            result = get_home_away_split_multiplier(12345, True, "hits")
            assert result == 1.0

    def test_result_within_caps(self):
        with patch("src.home_away_splits._fetch_splits", return_value=MOCK_HITTING_SPLITS):
            for prop in ("hits", "total_bases", "home_runs", "runs", "doubles"):
                for is_home in (True, False):
                    result = get_home_away_split_multiplier(12345, is_home, prop)
                    assert _CAP_LOW <= result <= _CAP_HIGH, f"{prop} is_home={is_home}: {result}"

    def test_exception_in_processing_returns_one(self):
        with patch("src.home_away_splits._fetch_splits", side_effect=Exception("network error")):
            result = get_home_away_split_multiplier(12345, True, "hits")
            assert result == 1.0


# ─── predictor integration ────────────────────────────────────────────────────

class TestPredictorIntegration:
    def test_home_advantage_raises_projection(self):
        """generate_prediction with home_away_mult > 1 projects higher than without."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.290, "pa": 300, "xba": 0.280, "k_rate": 20.0, "bb_rate": 8.0}
        result_neutral = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=1.0,
        )
        result_home = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=1.07,
        )
        assert result_home["projection"] > result_neutral["projection"]

    def test_home_away_mult_stored_in_result(self):
        """home_away_mult != 1.0 should be recorded in the result dict."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.290, "pa": 300, "xba": 0.280, "k_rate": 20.0, "bb_rate": 8.0}
        result = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=1.05,
        )
        assert result.get("home_away_mult") == pytest.approx(1.05)

    def test_neutral_mult_not_stored(self):
        """home_away_mult=1.0 should not add home_away_mult key to result."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.290, "pa": 300, "xba": 0.280, "k_rate": 20.0, "bb_rate": 8.0}
        result = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=1.0,
        )
        assert result.get("home_away_mult") is None

    def test_away_disadvantage_lowers_projection(self):
        """home_away_mult < 1 should reduce the projection."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.290, "pa": 300, "xba": 0.280, "k_rate": 20.0, "bb_rate": 8.0}
        result_neutral = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=1.0,
        )
        result_away = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile=batter,
            home_away_mult=0.95,
        )
        assert result_away["projection"] < result_neutral["projection"]
