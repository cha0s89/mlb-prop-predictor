"""Unit tests for day/night split adjustment module (src/day_night_splits.py)."""

from unittest.mock import patch
import pytest

from src.day_night_splits import (
    _bayesian_blend,
    _batter_mult,
    _pitcher_mult,
    _rate_mult,
    get_day_night_split_multiplier,
    get_wrigley_shadow_mult,
    is_day_game,
    clear_splits_cache,
    _CAP_LOW,
    _CAP_HIGH,
    _PRIOR_PA,
    _WRIGLEY_SHADOW_BATTER_K,
    _WRIGLEY_SHADOW_PITCHER_K,
)


# ─── _bayesian_blend ─────────────────────────────────────────────────────────

class TestBayesianBlend:
    def test_zero_sample_returns_neutral(self):
        assert _bayesian_blend(1.20, 0) == 1.0

    def test_huge_sample_approaches_raw(self):
        blended = _bayesian_blend(1.08, 10000)
        assert blended == pytest.approx(1.08, abs=0.002)

    def test_small_sample_pulls_toward_one(self):
        blended = _bayesian_blend(1.20, 20)
        assert 1.0 < blended < 1.20

    def test_exact_prior_pa_splits_halfway(self):
        raw = 1.12
        expected = (raw + 1.0) / 2
        blended = _bayesian_blend(raw, _PRIOR_PA)
        assert blended == pytest.approx(expected, abs=0.001)

    def test_cap_high_enforced(self):
        result = _bayesian_blend(2.0, 500)
        assert result == _CAP_HIGH

    def test_cap_low_enforced(self):
        result = _bayesian_blend(0.5, 500)
        assert result == _CAP_LOW

    def test_neutral_raw_returns_one(self):
        assert _bayesian_blend(1.0, 100) == pytest.approx(1.0)

    def test_caps_tighter_than_home_away(self):
        # day/night caps [0.92, 1.08] are tighter than home/away [0.90, 1.10]
        assert _CAP_LOW == 0.92
        assert _CAP_HIGH == 1.08


# ─── _rate_mult ───────────────────────────────────────────────────────────────

class TestRateMult:
    def test_equal_day_night_returns_one(self):
        raw, sample = _rate_mult(10, 10, 100, 100, is_day=True)
        assert raw == pytest.approx(1.0)
        assert sample == 100

    def test_day_better_returns_above_one(self):
        # day: 30/100=.300, night: 20/100=.200, overall: .250 → raw=1.20
        raw, sample = _rate_mult(30, 20, 100, 100, is_day=True)
        assert raw == pytest.approx(1.20)
        assert sample == 100

    def test_night_game_uses_night_rate(self):
        # night: 20/100=.200, overall: .250 → raw=0.80
        raw, sample = _rate_mult(30, 20, 100, 100, is_day=False)
        assert raw == pytest.approx(0.80)
        assert sample == 100

    def test_zero_denom_returns_neutral(self):
        raw, sample = _rate_mult(10, 10, 0, 100, is_day=True)
        assert raw == 1.0
        assert sample == 0.0

    def test_zero_total_count_returns_neutral(self):
        raw, sample = _rate_mult(0, 0, 100, 100, is_day=True)
        assert raw == 1.0
        assert sample == 0.0


# ─── is_day_game ─────────────────────────────────────────────────────────────

class TestIsDayGame:
    def test_early_afternoon_utc_is_day(self):
        # 1:05 PM ET (17:05 UTC in summer) → 1 PM ET → day
        assert is_day_game("2025-06-15T17:05:00Z") is True

    def test_evening_utc_is_night(self):
        # 7:10 PM ET (23:10 UTC in summer) → night
        assert is_day_game("2025-06-15T23:10:00Z") is False

    def test_exact_5pm_et_is_night(self):
        # 5:00 PM ET = 21:00 UTC (EDT, UTC-4) → hour_et == 17, not < 17 → night
        assert is_day_game("2025-06-15T21:00:00Z") is False

    def test_4_59pm_et_is_day(self):
        # 4:59 PM ET = 20:59 UTC → hour_et == 16 → day
        assert is_day_game("2025-06-15T20:59:00Z") is True

    def test_empty_string_returns_false(self):
        assert is_day_game("") is False

    def test_invalid_string_returns_false(self):
        assert is_day_game("not-a-date") is False

    def test_z_suffix_handled(self):
        # Ensures "Z" is parsed correctly
        assert is_day_game("2025-07-04T17:35:00Z") is True

    def test_winter_utc_offset(self):
        # March game: 1:05 PM ET = 18:05 UTC (EST, UTC-5) → day
        assert is_day_game("2025-03-28T18:05:00Z") is True


# ─── _batter_mult ─────────────────────────────────────────────────────────────

def _day_stat(**kwargs):
    base = {
        "plateAppearances": 120, "atBats": 110,
        "hits": 33, "homeRuns": 6, "doubles": 7, "triples": 1,
        "strikeOuts": 24, "baseOnBalls": 10, "runs": 18,
        "avg": "0.300", "slg": "0.509",
    }
    base.update(kwargs)
    return base


def _night_stat(**kwargs):
    base = {
        "plateAppearances": 280, "atBats": 260,
        "hits": 65, "homeRuns": 10, "doubles": 14, "triples": 2,
        "strikeOuts": 58, "baseOnBalls": 20, "runs": 36,
        "avg": "0.250", "slg": "0.392",
    }
    base.update(kwargs)
    return base


class TestBatterMult:
    def test_hits_day_better(self):
        # day: 33/110=.300, night: 65/260=.250, overall=.270 → raw≈1.11 → blended > 1
        mult = _batter_mult(_day_stat(), _night_stat(), "hits", is_day=True)
        assert mult > 1.0

    def test_hits_night_game_with_day_advantage(self):
        # same data, night game → day-advantage player gets < 1 for night
        mult = _batter_mult(_day_stat(), _night_stat(), "hits", is_day=False)
        assert mult < 1.0

    def test_equal_rates_returns_one(self):
        stat = _day_stat(hits=27, atBats=100, plateAppearances=110)
        mult = _batter_mult(stat, _night_stat(hits=65, atBats=260, plateAppearances=280),
                            "hits", is_day=True)
        # day: 27/100=.270, night: 65/260=.250, overall≈.255 → raw≈1.06, blended slightly > 1
        assert 0.92 <= mult <= 1.08

    def test_total_bases(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "total_bases", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_home_runs(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "home_runs", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_batter_strikeouts(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "batter_strikeouts", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_walks(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "walks", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_singles(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "singles", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_doubles(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "doubles", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_runs(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "runs", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_unknown_prop_returns_one(self):
        mult = _batter_mult(_day_stat(), _night_stat(), "stolen_bases", is_day=True)
        assert mult == 1.0

    def test_tiny_sample_returns_one(self):
        tiny = {"plateAppearances": 3, "atBats": 3, "hits": 1}
        mult = _batter_mult(tiny, tiny, "hits", is_day=True)
        assert mult == 1.0

    def test_cap_enforced(self):
        # Extreme day advantage with large sample
        big_day = _day_stat(hits=40, atBats=100, plateAppearances=110)
        big_night = _night_stat(hits=40, atBats=300, plateAppearances=320)
        mult = _batter_mult(big_day, big_night, "hits", is_day=True)
        assert mult <= _CAP_HIGH


# ─── _pitcher_mult ────────────────────────────────────────────────────────────

def _day_pitcher(**kwargs):
    base = {"inningsPitched": 30.0, "strikeOuts": 35, "earnedRuns": 12,
            "baseOnBalls": 10, "hits": 28}
    base.update(kwargs)
    return base


def _night_pitcher(**kwargs):
    base = {"inningsPitched": 80.0, "strikeOuts": 80, "earnedRuns": 30,
            "baseOnBalls": 25, "hits": 72}
    base.update(kwargs)
    return base


class TestPitcherMult:
    def test_pitcher_strikeouts_day_better(self):
        # day: 35/30 IP = 1.17 K/IP, night: 80/80 = 1.00, overall ≈ 1.05
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "pitcher_strikeouts", is_day=True)
        assert mult > 1.0

    def test_pitcher_strikeouts_night_game(self):
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "pitcher_strikeouts", is_day=False)
        assert mult < 1.0

    def test_earned_runs(self):
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "earned_runs", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_walks_allowed(self):
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "walks_allowed", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_hits_allowed(self):
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "hits_allowed", is_day=True)
        assert 0.92 <= mult <= 1.08

    def test_unknown_prop_returns_one(self):
        mult = _pitcher_mult(_day_pitcher(), _night_pitcher(), "pitching_outs", is_day=True)
        assert mult == 1.0

    def test_insufficient_ip_returns_one(self):
        tiny = {"inningsPitched": 2.0, "strikeOuts": 3}
        mult = _pitcher_mult(tiny, tiny, "pitcher_strikeouts", is_day=True)
        assert mult == 1.0

    def test_cap_enforced(self):
        extreme_day = _day_pitcher(strikeOuts=60, inningsPitched=30.0)
        mult = _pitcher_mult(extreme_day, _night_pitcher(), "pitcher_strikeouts", is_day=True)
        assert mult <= _CAP_HIGH


# ─── get_wrigley_shadow_mult ─────────────────────────────────────────────────

class TestWrigleyShhadow:
    def test_night_game_returns_one(self):
        assert get_wrigley_shadow_mult("pitcher_strikeouts", True, "CHC", is_day=False) == 1.0

    def test_non_wrigley_team_returns_one(self):
        assert get_wrigley_shadow_mult("pitcher_strikeouts", True, "NYM", is_day=True) == 1.0

    def test_no_park_team_returns_one(self):
        assert get_wrigley_shadow_mult("pitcher_strikeouts", True, None, is_day=True) == 1.0

    def test_pitcher_k_suppressed_at_wrigley_day(self):
        mult = get_wrigley_shadow_mult("pitcher_strikeouts", True, "CHC", is_day=True)
        assert mult == _WRIGLEY_SHADOW_PITCHER_K
        assert mult < 1.0

    def test_batter_k_elevated_at_wrigley_day(self):
        mult = get_wrigley_shadow_mult("batter_strikeouts", False, "CHC", is_day=True)
        assert mult == _WRIGLEY_SHADOW_BATTER_K
        assert mult > 1.0

    def test_chn_abbreviation_recognized(self):
        mult = get_wrigley_shadow_mult("pitcher_strikeouts", True, "CHN", is_day=True)
        assert mult == _WRIGLEY_SHADOW_PITCHER_K

    def test_non_k_prop_at_wrigley_day_returns_one(self):
        assert get_wrigley_shadow_mult("hits", False, "CHC", is_day=True) == 1.0
        assert get_wrigley_shadow_mult("earned_runs", True, "CHC", is_day=True) == 1.0


# ─── get_day_night_split_multiplier (public API) ─────────────────────────────

class TestPublicAPI:
    def setup_method(self):
        clear_splits_cache()

    def test_invalid_player_id_returns_one(self):
        assert get_day_night_split_multiplier(0, True, "hits") == 1.0
        assert get_day_night_split_multiplier(-1, True, "hits") == 1.0

    def test_unsupported_batter_prop_returns_one(self):
        assert get_day_night_split_multiplier(123456, True, "stolen_bases") == 1.0

    def test_unsupported_pitcher_prop_returns_one(self):
        assert get_day_night_split_multiplier(123456, True, "pitching_outs", is_pitcher=True) == 1.0

    def test_api_failure_returns_one(self):
        with patch("src.day_night_splits._fetch_splits", return_value={}):
            result = get_day_night_split_multiplier(123456, True, "hits")
        assert result == 1.0

    def test_exception_in_fetch_returns_one(self):
        with patch("src.day_night_splits._fetch_splits", side_effect=RuntimeError("network error")):
            result = get_day_night_split_multiplier(123456, True, "hits")
        assert result == 1.0

    def test_day_game_with_splits(self):
        fake_splits = {
            "day": {
                "plateAppearances": 120, "atBats": 110,
                "hits": 35, "homeRuns": 6, "doubles": 8, "triples": 1,
                "strikeOuts": 22, "baseOnBalls": 10, "runs": 20,
                "avg": "0.318", "slg": "0.527",
            },
            "night": {
                "plateAppearances": 280, "atBats": 260,
                "hits": 60, "homeRuns": 10, "doubles": 14, "triples": 2,
                "strikeOuts": 62, "baseOnBalls": 20, "runs": 34,
                "avg": "0.231", "slg": "0.365",
            },
        }
        with patch("src.day_night_splits._fetch_splits", return_value=fake_splits):
            result = get_day_night_split_multiplier(123456, True, "hits")
        # day player performing better in day games → mult > 1
        assert result > 1.0
        assert result <= _CAP_HIGH

    def test_night_game_with_day_advantage_player(self):
        fake_splits = {
            "day": {
                "plateAppearances": 120, "atBats": 110,
                "hits": 35, "homeRuns": 6, "doubles": 8, "triples": 1,
                "strikeOuts": 22, "baseOnBalls": 10, "runs": 20,
                "avg": "0.318", "slg": "0.527",
            },
            "night": {
                "plateAppearances": 280, "atBats": 260,
                "hits": 60, "homeRuns": 10, "doubles": 14, "triples": 2,
                "strikeOuts": 62, "baseOnBalls": 20, "runs": 34,
                "avg": "0.231", "slg": "0.365",
            },
        }
        with patch("src.day_night_splits._fetch_splits", return_value=fake_splits):
            result = get_day_night_split_multiplier(123456, False, "hits")
        # night game for a day-advantage player → mult < 1
        assert result < 1.0
        assert result >= _CAP_LOW

    def test_pitcher_day_night_split(self):
        fake_splits = {
            "day": {
                "inningsPitched": 30.0,
                "strikeOuts": 38,
                "earnedRuns": 10,
                "baseOnBalls": 8,
                "hits": 25,
            },
            "night": {
                "inningsPitched": 90.0,
                "strikeOuts": 90,
                "earnedRuns": 30,
                "baseOnBalls": 28,
                "hits": 80,
            },
        }
        with patch("src.day_night_splits._fetch_splits", return_value=fake_splits):
            result = get_day_night_split_multiplier(
                654321, True, "pitcher_strikeouts", is_pitcher=True
            )
        assert 0.92 <= result <= 1.08

    def test_result_capped_high(self):
        # Extreme day advantage → should be capped at 1.08
        fake_splits = {
            "day": {
                "plateAppearances": 300, "atBats": 280,
                "hits": 112, "homeRuns": 15, "doubles": 20, "triples": 2,
                "strikeOuts": 40, "baseOnBalls": 20, "runs": 50,
                "avg": "0.400", "slg": "0.650",
            },
            "night": {
                "plateAppearances": 100, "atBats": 90,
                "hits": 18, "homeRuns": 2, "doubles": 4, "triples": 0,
                "strikeOuts": 25, "baseOnBalls": 8, "runs": 10,
                "avg": "0.200", "slg": "0.300",
            },
        }
        with patch("src.day_night_splits._fetch_splits", return_value=fake_splits):
            result = get_day_night_split_multiplier(123456, True, "hits")
        assert result <= _CAP_HIGH

    def test_result_capped_low(self):
        fake_splits = {
            "day": {
                "plateAppearances": 100, "atBats": 90,
                "hits": 18, "homeRuns": 2, "doubles": 4, "triples": 0,
                "strikeOuts": 25, "baseOnBalls": 8, "runs": 10,
                "avg": "0.200", "slg": "0.300",
            },
            "night": {
                "plateAppearances": 300, "atBats": 280,
                "hits": 112, "homeRuns": 15, "doubles": 20, "triples": 2,
                "strikeOuts": 40, "baseOnBalls": 20, "runs": 50,
                "avg": "0.400", "slg": "0.650",
            },
        }
        with patch("src.day_night_splits._fetch_splits", return_value=fake_splits):
            result = get_day_night_split_multiplier(123456, True, "hits")
        assert result >= _CAP_LOW


# ─── predictor integration ───────────────────────────────────────────────────

class TestPredictorIntegration:
    """Verify generate_prediction accepts and applies day_night_mult."""

    def test_day_night_mult_above_one_raises_projection(self):
        from src.predictor import generate_prediction

        base = generate_prediction(
            player_name="Test Player",
            stat_type="Hits",
            stat_internal="hits",
            line=1.5,
            batter_profile={
                "avg": 0.280, "obp": 0.350, "slg": 0.450,
                "k_rate": 0.20, "bb_rate": 0.09,
                "woba": 0.340, "iso": 0.170,
            },
            day_night_mult=1.0,
        )

        boosted = generate_prediction(
            player_name="Test Player",
            stat_type="Hits",
            stat_internal="hits",
            line=1.5,
            batter_profile={
                "avg": 0.280, "obp": 0.350, "slg": 0.450,
                "k_rate": 0.20, "bb_rate": 0.09,
                "woba": 0.340, "iso": 0.170,
            },
            day_night_mult=1.06,
        )

        assert boosted["projection"] > base["projection"]

    def test_day_night_mult_stored_in_result(self):
        from src.predictor import generate_prediction

        result = generate_prediction(
            player_name="Test Player",
            stat_type="Hits",
            stat_internal="hits",
            line=1.5,
            batter_profile={
                "avg": 0.280, "obp": 0.350, "slg": 0.450,
                "k_rate": 0.20, "bb_rate": 0.09,
                "woba": 0.340, "iso": 0.170,
            },
            day_night_mult=1.05,
        )

        assert "day_night_mult" in result
        assert result["day_night_mult"] == pytest.approx(1.05, abs=0.001)

    def test_neutral_mult_not_stored(self):
        from src.predictor import generate_prediction

        result = generate_prediction(
            player_name="Test Player",
            stat_type="Hits",
            stat_internal="hits",
            line=1.5,
            batter_profile={
                "avg": 0.280, "obp": 0.350, "slg": 0.450,
                "k_rate": 0.20, "bb_rate": 0.09,
                "woba": 0.340, "iso": 0.170,
            },
            day_night_mult=1.0,
        )

        assert "day_night_mult" not in result
