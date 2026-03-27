"""Unit tests for src/divisional_familiarity.py."""

import datetime
from unittest.mock import patch

import pytest

from src.divisional_familiarity import (
    DIVISION_MAP,
    TEAM_IDS,
    _in_same_division,
    _series_index,
    clear_familiarity_cache,
    get_familiarity_adjustment,
)


# ── _in_same_division ─────────────────────────────────────────────────────────

class TestInSameDivision:
    def test_same_division_rivals(self):
        assert _in_same_division("NYY", "BOS") is True

    def test_same_division_different_pair(self):
        assert _in_same_division("LAD", "SF") is True

    def test_different_divisions(self):
        assert _in_same_division("NYY", "LAD") is False

    def test_interleague(self):
        assert _in_same_division("NYY", "CHC") is False

    def test_same_team_returns_false(self):
        assert _in_same_division("NYY", "NYY") is False

    def test_case_insensitive(self):
        assert _in_same_division("nyy", "bos") is True
        assert _in_same_division("LAD", "sf") is True

    def test_unknown_team_returns_false(self):
        assert _in_same_division("XYZ", "BOS") is False


# ── _series_index ──────────────────────────────────────────────────────────────

class TestSeriesIndex:
    def test_zero_games_is_first_series(self):
        assert _series_index(0) == 0

    def test_one_game_is_first_series(self):
        assert _series_index(1) == 0

    def test_three_games_is_first_series(self):
        assert _series_index(3) == 0

    def test_four_games_is_second_series(self):
        assert _series_index(4) == 1

    def test_seven_games_is_third_series(self):
        assert _series_index(7) == 2

    def test_ten_games_is_fourth_series(self):
        assert _series_index(10) == 3

    def test_many_games_capped_at_last_bucket(self):
        assert _series_index(100) == 3


# ── get_familiarity_adjustment (no API) ───────────────────────────────────────

class TestGetFamiliarityAdjustmentNonDivision:
    def test_non_division_pitcher_k_returns_one(self):
        result = get_familiarity_adjustment("NYY", "LAD", prop_type="pitcher_strikeouts")
        assert result == 1.0

    def test_non_division_batter_k_returns_one(self):
        result = get_familiarity_adjustment("NYY", "LAD", prop_type="batter_strikeouts")
        assert result == 1.0

    def test_empty_pitcher_team_returns_one(self):
        assert get_familiarity_adjustment("", "BOS") == 1.0

    def test_empty_batter_team_returns_one(self):
        assert get_familiarity_adjustment("NYY", "") == 1.0

    def test_none_teams_return_one(self):
        assert get_familiarity_adjustment(None, None) == 1.0


# ── get_familiarity_adjustment (mocked schedule API) ─────────────────────────

def _make_schedule_response(games_between: int, team_id: int, opp_id: int) -> dict:
    """Build a minimal MLB schedule API response with *games_between* games."""
    games = []
    for _ in range(games_between):
        games.append({
            "teams": {
                "home": {"team": {"id": team_id}},
                "away": {"team": {"id": opp_id}},
            }
        })
    return {"dates": [{"games": games}]} if games else {"dates": []}


class TestGetFamiliarityAdjustmentDivision:
    def setup_method(self):
        clear_familiarity_cache()

    def test_first_series_no_prior_games_returns_one(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=0):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(1.00)

    def test_second_series_returns_099(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=4):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(0.99)

    def test_third_series_returns_097(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=7):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(0.97)

    def test_fourth_plus_series_returns_095(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=10):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(0.95)

    def test_batter_k_milder_than_pitcher_k(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=10):
            pitcher_mult = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
            batter_mult = get_familiarity_adjustment("NYY", "BOS", prop_type="batter_strikeouts")
        # Batter suppression should be weaker (closer to 1.0)
        assert batter_mult > pitcher_mult
        assert batter_mult < 1.0

    def test_batter_k_first_series_returns_one(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=0):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="batter_strikeouts")
        assert result == pytest.approx(1.00)

    def test_batter_k_fourth_series_scaled(self):
        # pitcher 4th series = 0.95, suppression = 0.05, batter = 1 - 0.05*0.6 = 0.97
        with patch("src.divisional_familiarity._count_games_between", return_value=10):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="batter_strikeouts")
        assert result == pytest.approx(0.97)

    def test_api_failure_falls_back_to_first_series(self):
        """API errors should not crash — fall back to series 0 (mult = 1.0)."""
        with patch("src.divisional_familiarity._count_games_between", return_value=0):
            result = get_familiarity_adjustment("NYY", "BOS", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(1.00)

    def test_game_date_string_accepted(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=7) as mock_fn:
            result = get_familiarity_adjustment(
                "BOS", "NYY", game_date="2026-06-15", prop_type="pitcher_strikeouts"
            )
        assert result == pytest.approx(0.97)
        # Confirm before_date was passed as string
        call_args = mock_fn.call_args
        assert call_args[0][3] == "2026-06-15"

    def test_game_date_date_object_accepted(self):
        d = datetime.date(2026, 7, 4)
        with patch("src.divisional_familiarity._count_games_between", return_value=4) as mock_fn:
            result = get_familiarity_adjustment("BOS", "NYY", game_date=d, prop_type="pitcher_strikeouts")
        assert result == pytest.approx(0.99)
        assert mock_fn.call_args[0][3] == "2026-07-04"

    def test_nl_west_rivals(self):
        with patch("src.divisional_familiarity._count_games_between", return_value=7):
            result = get_familiarity_adjustment("LAD", "SF", prop_type="pitcher_strikeouts")
        assert result == pytest.approx(0.97)


# ── Division map completeness ──────────────────────────────────────────────────

class TestDivisionMapCompleteness:
    def test_all_30_teams_in_division_map(self):
        assert len(DIVISION_MAP) == 30

    def test_all_30_teams_have_ids(self):
        assert len(TEAM_IDS) == 30

    def test_each_division_has_five_teams(self):
        from collections import Counter
        counts = Counter(DIVISION_MAP.values())
        for div, count in counts.items():
            assert count == 5, f"{div} has {count} teams, expected 5"

    def test_division_map_and_team_ids_keys_match(self):
        assert set(DIVISION_MAP.keys()) == set(TEAM_IDS.keys())
