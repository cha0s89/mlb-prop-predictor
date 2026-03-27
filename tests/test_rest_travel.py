"""Tests for rest/travel fatigue adjustments (src/rest_travel.py)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from src.rest_travel import (
    _dgdn_mult,
    _travel_mult,
    _short_rest_mult,
    _local_hour,
    _parse_utc_hour,
    _home_team_abbr,
    get_fatigue_adjustment,
    clear_rest_travel_cache,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_game(game_date_utc: str, home_abbr: str, state: str = "Final") -> dict:
    """Construct a minimal schedule-API game dict."""
    return {
        "gameDate": game_date_utc,
        "status": {"detailedState": state},
        "teams": {
            "home": {"team": {"abbreviation": home_abbr}},
            "away": {"team": {"abbreviation": "OPP"}},
        },
    }


# ─── _parse_utc_hour ──────────────────────────────────────────────────────────

class TestParseUtcHour:
    def test_zulu_suffix(self):
        assert _parse_utc_hour("2025-04-10T23:10:00Z") == pytest.approx(23 + 10 / 60)

    def test_offset_aware(self):
        assert _parse_utc_hour("2025-04-10T18:05:00+00:00") == pytest.approx(18 + 5 / 60)

    def test_none_on_empty(self):
        assert _parse_utc_hour("") is None

    def test_none_on_garbage(self):
        assert _parse_utc_hour("not-a-date") is None


# ─── _local_hour ─────────────────────────────────────────────────────────────

class TestLocalHour:
    def test_eastern_offset(self):
        # 23:00 UTC → 18:00 ET (UTC-5)
        assert _local_hour(23.0, -5) == pytest.approx(18.0)

    def test_pacific_offset(self):
        # 23:00 UTC → 15:00 PT (UTC-8)
        assert _local_hour(23.0, -8) == pytest.approx(15.0)

    def test_midnight_wrap(self):
        # 02:00 UTC → 21:00 ET previous day (wraps via % 24)
        assert _local_hour(2.0, -5) == pytest.approx(21.0)


# ─── _home_team_abbr ─────────────────────────────────────────────────────────

class TestHomeTeamAbbr:
    def test_reads_home_abbreviation(self):
        game = _make_game("2025-04-10T20:00:00Z", "NYY")
        assert _home_team_abbr(game, "BOS") == "NYY"

    def test_falls_back_to_querying_team(self):
        game = {"gameDate": "2025-04-10T20:00:00Z", "teams": {}}
        assert _home_team_abbr(game, "BOS") == "BOS"


# ─── _dgdn_mult ───────────────────────────────────────────────────────────────

class TestDgdnMult:
    def _night_game(self, home="NYY") -> dict:
        # 00:10 UTC (next calendar day) = 19:10 ET → night game (≥19:00 local)
        return _make_game("2025-04-10T00:10:00Z", home)

    def _day_game_yest(self, home="NYY") -> dict:
        # 17:10 UTC = 12:10 ET → afternoon/day game
        return _make_game("2025-04-09T17:10:00Z", home)

    # today_utc_hour for a day game in ET: 17:10 UTC = 12:10 ET (<17 local)
    _DAY_GAME_UTC = 17 + 10 / 60   # 12:10 ET
    # today_utc_hour for a night game in ET: 00:10 UTC = 19:10 ET (≥19 = night)
    _NIGHT_GAME_UTC = 0 + 10 / 60

    def test_dgdn_batter_penalty(self):
        """Night game yesterday + day game today → batter penalty."""
        mult = _dgdn_mult(self._night_game(), self._DAY_GAME_UTC, "NYY", is_pitcher=False)
        assert mult == pytest.approx(0.97)

    def test_dgdn_pitcher_penalty(self):
        """Night game yesterday + day game today → reliever penalty."""
        mult = _dgdn_mult(self._night_game(), self._DAY_GAME_UTC, "NYY", is_pitcher=True)
        assert mult == pytest.approx(0.96)

    def test_no_penalty_night_today(self):
        """Yesterday night + today night → not a day game → no penalty."""
        mult = _dgdn_mult(self._night_game(), self._NIGHT_GAME_UTC, "NYY", is_pitcher=False)
        assert mult == pytest.approx(1.0)

    def test_no_penalty_day_yesterday(self):
        """Yesterday day game + today day game → no night game → no penalty."""
        mult = _dgdn_mult(self._day_game_yest(), self._DAY_GAME_UTC, "NYY", is_pitcher=False)
        assert mult == pytest.approx(1.0)

    def test_no_penalty_missing_yesterday_time(self):
        """Game dict with no gameDate → fallback 1.0."""
        game = {"teams": {"home": {"team": {"abbreviation": "NYY"}}}}
        mult = _dgdn_mult(game, self._DAY_GAME_UTC, "NYY", is_pitcher=False)
        assert mult == pytest.approx(1.0)

    def test_timezone_adjustment_west_coast(self):
        """
        LA day game: starts at 20:10 UTC = 12:10 PT (UTC-8) → is a day game.
        Yesterday in LA at 03:10 UTC = 19:10 PT → night game.
        """
        yest = _make_game("2025-04-09T03:10:00Z", "LAD")
        today_utc = 20 + 10 / 60  # 12:10 PT
        mult = _dgdn_mult(yest, today_utc, "LAD", is_pitcher=False)
        assert mult == pytest.approx(0.97)


# ─── _travel_mult ─────────────────────────────────────────────────────────────

class TestTravelMult:
    def _game_at(self, home: str) -> dict:
        return _make_game("2025-04-09T20:00:00Z", home)

    def test_cross_country_3tz_penalty(self):
        """ET (−5) to PT (−8) = 3 timezone difference → 0.96."""
        game = self._game_at("BOS")  # yesterday in Boston (ET)
        mult = _travel_mult(game, "LAD", "LAD")  # today in LA (PT)
        assert mult == pytest.approx(0.96)

    def test_cross_country_reverse(self):
        """PT (−8) to ET (−5) → same 3-tz gap → 0.96."""
        game = self._game_at("LAD")
        mult = _travel_mult(game, "NYY", "NYY")
        assert mult == pytest.approx(0.96)

    def test_one_tz_travel(self):
        """ET (−5) to CT (−6) = 1 timezone → 0.99 minimal penalty."""
        game = self._game_at("BOS")
        mult = _travel_mult(game, "CHC", "CHC")
        assert mult == pytest.approx(0.99)

    def test_same_city_no_penalty(self):
        """Same home team both days → no travel."""
        game = self._game_at("NYY")
        mult = _travel_mult(game, "NYY", "NYY")
        assert mult == pytest.approx(1.0)

    def test_same_tz_different_city(self):
        """BOS (ET) to NYY (ET) = 0 tz diff → no penalty."""
        game = self._game_at("BOS")
        mult = _travel_mult(game, "NYY", "NYY")
        # Same timezone offset (both ET = -5) but different city
        assert mult == pytest.approx(1.0)

    def test_unknown_team_fallback(self):
        """Unknown team abbreviation → 1.0 fallback."""
        game = self._game_at("XYZ")
        mult = _travel_mult(game, "NYY", "NYY")
        assert mult == pytest.approx(1.0)


# ─── _short_rest_mult ─────────────────────────────────────────────────────────

class _FakeLogData:
    """Helper to build fake MLB Stats API game-log responses."""

    @staticmethod
    def with_last_start(last_start_date: str) -> dict:
        return {
            "stats": [{
                "splits": [{
                    "date": last_start_date,
                    "stat": {"gamesStarted": 1, "inningsPitched": "6.0"},
                }]
            }]
        }


class TestShortRestMult:
    def _patch_api(self, last_start: str):
        """Patch both player search and game-log endpoints."""
        search_resp = {"people": [{"id": 999}]}
        log_resp = _FakeLogData.with_last_start(last_start)
        return patch(
            "src.rest_travel._api_get",
            side_effect=lambda ep, params=None: (
                search_resp if "search" in ep else log_resp
            ),
        )

    def test_very_short_rest_under_3_days(self):
        """Last start 2 days ago (1 day rest) → 0.94."""
        today = date(2025, 4, 10)
        last_start = "2025-04-08"  # 1 day rest
        with self._patch_api(last_start):
            mult = _short_rest_mult("John Pitcher", today)
        assert mult == pytest.approx(0.94)

    def test_short_rest_3_days(self):
        """Last start 4 days ago (3 days rest) → 0.97."""
        today = date(2025, 4, 10)
        last_start = "2025-04-06"  # 3 days rest
        with self._patch_api(last_start):
            mult = _short_rest_mult("John Pitcher", today)
        assert mult == pytest.approx(0.97)

    def test_normal_rest_4_plus_days(self):
        """4 or more days rest → 1.0 (normal turn)."""
        today = date(2025, 4, 10)
        last_start = "2025-04-05"  # 4 days rest
        with self._patch_api(last_start):
            mult = _short_rest_mult("John Pitcher", today)
        assert mult == pytest.approx(1.0)

    def test_no_player_name_returns_neutral(self):
        assert _short_rest_mult("", date(2025, 4, 10)) == pytest.approx(1.0)

    def test_api_failure_returns_neutral(self):
        with patch("src.rest_travel._api_get", return_value=None):
            mult = _short_rest_mult("John Pitcher", date(2025, 4, 10))
        assert mult == pytest.approx(1.0)

    def test_no_starts_in_log_returns_neutral(self):
        log = {"stats": [{"splits": [{"date": "2025-04-08", "stat": {"gamesStarted": 0}}]}]}
        search = {"people": [{"id": 999}]}
        with patch("src.rest_travel._api_get",
                   side_effect=lambda ep, params=None: search if "search" in ep else log):
            mult = _short_rest_mult("John Pitcher", date(2025, 4, 10))
        assert mult == pytest.approx(1.0)


# ─── get_fatigue_adjustment (public API) ──────────────────────────────────────

class TestGetFatigueAdjustment:
    """Test the public API with mocked schedule fetches."""

    def setup_method(self):
        clear_rest_travel_cache()

    def _patch_yesterday(self, game: dict | None):
        return patch("src.rest_travel._fetch_yesterday_game", return_value=game)

    def test_no_recent_game_returns_neutral(self):
        with self._patch_yesterday(None):
            mult = get_fatigue_adjustment("NYY", date(2025, 4, 10), "2025-04-10T17:10:00Z")
        assert mult == pytest.approx(1.0)

    def test_dgdn_batter_signal_applied(self):
        """Night game yesterday + day game today → batter penalty < 1.0."""
        # 00:10 UTC = 19:10 ET → proper night game (≥19:00 local)
        night_yest = _make_game("2025-04-10T00:10:00Z", "NYY")
        with self._patch_yesterday(night_yest):
            mult = get_fatigue_adjustment(
                team="NYY",
                game_date=date(2025, 4, 10),
                game_time="2025-04-10T17:10:00Z",  # 12:10 ET → day game
                today_home_team="NYY",
            )
        assert mult < 1.0

    def test_dgdn_pitcher_signal_applied(self):
        """Night game yesterday + day game today → pitcher penalty < 1.0."""
        night_yest = _make_game("2025-04-10T00:10:00Z", "NYY")  # 19:10 ET
        with self._patch_yesterday(night_yest):
            mult = get_fatigue_adjustment(
                team="NYY",
                game_date=date(2025, 4, 10),
                game_time="2025-04-10T17:10:00Z",
                is_pitcher=True,
                today_home_team="NYY",
            )
        assert mult < 1.0

    def test_cross_country_travel_signal(self):
        """Game at ET venue yesterday, today at PT venue → travel penalty."""
        yest_boston = _make_game("2025-04-09T20:00:00Z", "BOS")
        with self._patch_yesterday(yest_boston):
            # Today no-time so only travel applies
            mult = get_fatigue_adjustment(
                team="LAD",
                game_date=date(2025, 4, 10),
                game_time=None,
                today_home_team="LAD",
            )
        assert mult == pytest.approx(0.96)

    def test_compound_signals_multiply(self):
        """DGDN + cross-country travel should compound (mult < either alone).

        Scenario: LAD was away at BOS for a night game, now flies home for a day game.
        Travel: BOS (ET, -5) → LAD (PT, -8) = 3 tz zones → 0.96
        DGDN:   BOS night (7:10 PM ET = 00:10 UTC) + LAD day game → 0.97
        Compound: 0.97 × 0.96 ≈ 0.93
        """
        yest_boston = _make_game("2025-04-10T00:10:00Z", "BOS")  # 19:10 ET night game
        with self._patch_yesterday(yest_boston):
            mult = get_fatigue_adjustment(
                team="LAD",
                game_date=date(2025, 4, 10),
                game_time="2025-04-10T20:10:00Z",  # 12:10 PT → day game
                today_home_team="LAD",
            )
        assert mult < 0.97
        assert mult >= 0.85  # floor applies

    def test_string_game_date_parsed(self):
        """game_date as string should parse without error."""
        night_yest = _make_game("2025-04-10T00:10:00Z", "NYY")  # 19:10 ET
        with self._patch_yesterday(night_yest):
            mult = get_fatigue_adjustment(
                team="NYY",
                game_date="2025-04-10",
                game_time="2025-04-10T17:10:00Z",
                today_home_team="NYY",
            )
        assert 0.85 <= mult <= 1.0

    def test_api_error_graceful_fallback(self):
        """Any exception in the pipeline → 1.0, never crashes."""
        with patch("src.rest_travel._fetch_yesterday_game", side_effect=RuntimeError("oops")):
            mult = get_fatigue_adjustment("NYY", date(2025, 4, 10), "2025-04-10T17:10:00Z")
        assert mult == pytest.approx(1.0)

    def test_empty_team_returns_neutral(self):
        mult = get_fatigue_adjustment("", date(2025, 4, 10), "2025-04-10T17:10:00Z")
        assert mult == pytest.approx(1.0)

    def test_floor_is_respected(self):
        """Even with all penalties stacked, result ≥ 0.85."""
        yest = _make_game("2025-04-10T00:10:00Z", "BOS")  # 19:10 ET night game

        def _mock_short_rest(*a, **kw):
            return 0.80  # artificial extreme

        with self._patch_yesterday(yest), \
             patch("src.rest_travel._short_rest_mult", _mock_short_rest):
            mult = get_fatigue_adjustment(
                team="LAD",
                game_date=date(2025, 4, 10),
                game_time="2025-04-10T20:10:00Z",
                is_pitcher=True,
                today_home_team="LAD",
            )
        assert mult >= 0.85


# ─── predictor integration ────────────────────────────────────────────────────

class TestPredictorIntegration:
    def test_rest_travel_mult_lowers_projection(self):
        """generate_prediction with rest_travel_mult < 1 produces lower projection."""
        from src.predictor import generate_prediction

        batter = {"avg": 0.290, "pa": 300, "xba": 0.280, "k_rate": 20.0, "bb_rate": 8.0}

        result_normal = generate_prediction(
            "Test Batter", "Hits", "hits", 1.5,
            batter_profile=batter,
            rest_travel_mult=1.0,
        )
        result_fatigued = generate_prediction(
            "Test Batter", "Hits", "hits", 1.5,
            batter_profile=batter,
            rest_travel_mult=0.97,
        )

        assert result_fatigued["projection"] < result_normal["projection"]

    def test_rest_travel_mult_stored_in_result(self):
        """rest_travel_mult key appears in result when != 1.0."""
        from src.predictor import generate_prediction

        result = generate_prediction(
            "Test Batter", "Hits", "hits", 1.5,
            rest_travel_mult=0.96,
        )
        assert "rest_travel_mult" in result

    def test_rest_travel_neutral_not_stored(self):
        """rest_travel_mult key absent when multiplier is 1.0."""
        from src.predictor import generate_prediction

        result = generate_prediction(
            "Test Batter", "Hits", "hits", 1.5,
            rest_travel_mult=1.0,
        )
        assert "rest_travel_mult" not in result
