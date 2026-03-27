"""Unit tests for src/recent_form.py — exponential decay L7/L14 form tracking."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.recent_form import (
    _normalize_name,
    _batter_stat,
    _pitcher_stat,
    _weighted_rate,
    _season_rate,
    compute_recent_form_multiplier,
    clear_cache,
    MULT_FLOOR,
    MULT_CAP,
    L7_DECAY,
    L14_DECAY,
    MIN_GAMES_REQUIRED,
    SEASON_MIN_GAMES,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

TODAY = date(2026, 5, 15)


def _batter_game(days_ago: int, hits=1, ab=4, pa=5, tb=1,
                 doubles=0, triples=0, home_runs=0, rbi=0, runs=0,
                 bb=0, hbp=0, sb=0, k=0) -> dict:
    """Build a fake game-log entry for a batter."""
    return {
        "date": TODAY - timedelta(days=days_ago),
        "stat": {
            "atBats": ab,
            "plateAppearances": pa,
            "hits": hits,
            "doubles": doubles,
            "triples": triples,
            "homeRuns": home_runs,
            "rbi": rbi,
            "runs": runs,
            "baseOnBalls": bb,
            "hitByPitch": hbp,
            "stolenBases": sb,
            "strikeOuts": k,
            "totalBases": tb,
        },
    }


def _pitcher_game(days_ago: int, ip="6.0", k=6, er=2, bb=2, hits=5) -> dict:
    """Build a fake game-log entry for a pitcher."""
    return {
        "date": TODAY - timedelta(days=days_ago),
        "stat": {
            "inningsPitched": ip,
            "strikeOuts": k,
            "earnedRuns": er,
            "baseOnBalls": bb,
            "hits": hits,
        },
    }


# ─── _normalize_name ─────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_strips_accents(self):
        assert _normalize_name("Ramón Laureano") == "ramon laureano"

    def test_strips_jr(self):
        # "Jr." suffix is stripped, leaving just the given names
        result = _normalize_name("Ronald Acuña Jr.")
        assert result == "ronald acuna"

    def test_lowercase(self):
        assert _normalize_name("Freddie Freeman") == "freddie freeman"

    def test_hyphen_removed(self):
        assert _normalize_name("Ha-Seong Kim") == "haseong kim"

    def test_empty_string(self):
        assert _normalize_name("") == ""


# ─── _batter_stat ────────────────────────────────────────────────────────────

class TestBatterStat:
    def _stat(self, **kwargs):
        defaults = {"atBats": 4, "plateAppearances": 5, "hits": 2, "doubles": 1,
                    "triples": 0, "homeRuns": 0, "rbi": 1, "runs": 1,
                    "baseOnBalls": 1, "hitByPitch": 0, "stolenBases": 0,
                    "strikeOuts": 1, "totalBases": 3}
        defaults.update(kwargs)
        return defaults

    def test_hits(self):
        val, opp = _batter_stat(self._stat(hits=2, atBats=4), "hits")
        assert val == 2.0 and opp == 4.0

    def test_total_bases(self):
        val, opp = _batter_stat(self._stat(totalBases=5, atBats=4), "total_bases")
        assert val == 5.0 and opp == 4.0

    def test_home_runs_uses_pa(self):
        val, opp = _batter_stat(self._stat(homeRuns=1, plateAppearances=5), "home_runs")
        assert val == 1.0 and opp == 5.0

    def test_rbis_uses_pa(self):
        val, opp = _batter_stat(self._stat(rbi=2, plateAppearances=5), "rbis")
        assert val == 2.0 and opp == 5.0

    def test_walks_uses_pa(self):
        val, opp = _batter_stat(self._stat(baseOnBalls=1, plateAppearances=5), "walks")
        assert val == 1.0 and opp == 5.0

    def test_batter_strikeouts_uses_pa(self):
        val, opp = _batter_stat(self._stat(strikeOuts=2, plateAppearances=5), "batter_strikeouts")
        assert val == 2.0 and opp == 5.0

    def test_singles_computed(self):
        # hits=3, doubles=1, triples=0, hr=1 → singles=1
        val, opp = _batter_stat(
            self._stat(hits=3, doubles=1, triples=0, homeRuns=1, atBats=4), "singles"
        )
        assert val == 1.0 and opp == 4.0

    def test_doubles(self):
        val, opp = _batter_stat(self._stat(doubles=2, atBats=4), "doubles")
        assert val == 2.0 and opp == 4.0

    def test_hits_runs_rbis(self):
        val, opp = _batter_stat(
            self._stat(hits=2, runs=1, rbi=1, plateAppearances=5), "hits_runs_rbis"
        )
        assert val == 4.0 and opp == 5.0

    def test_fantasy_score(self):
        # singles=1 (hits=2, doubles=1, hr=0, triples=0)→singles=1
        # 1*3 + 1*5 + rbi*2=2 + runs*2=2 + bb*2=2 = 14
        val, opp = _batter_stat(
            self._stat(hits=2, doubles=1, triples=0, homeRuns=0,
                       rbi=1, runs=1, baseOnBalls=1, hitByPitch=0,
                       stolenBases=0, plateAppearances=5),
            "hitter_fantasy_score",
        )
        assert val == pytest.approx(3 + 5 + 2 + 2 + 2)  # 14
        assert opp == 5.0

    def test_zero_ab_returns_zero_opp(self):
        val, opp = _batter_stat(
            {"atBats": 0, "plateAppearances": 0}, "hits"
        )
        assert opp == 0.0

    def test_unsupported_prop_returns_zero(self):
        val, opp = _batter_stat(self._stat(), "pitcher_strikeouts")
        assert val == 0.0 and opp == 0.0


# ─── _pitcher_stat ───────────────────────────────────────────────────────────

class TestPitcherStat:
    def test_strikeouts(self):
        val, opp = _pitcher_stat({"inningsPitched": "6.0", "strikeOuts": 7}, "pitcher_strikeouts")
        assert val == 7.0 and opp == 1.0

    def test_pitching_outs_full_innings(self):
        val, opp = _pitcher_stat({"inningsPitched": "6.0"}, "pitching_outs")
        assert val == 18.0 and opp == 1.0

    def test_pitching_outs_partial(self):
        val, opp = _pitcher_stat({"inningsPitched": "6.2"}, "pitching_outs")
        assert val == 20.0 and opp == 1.0

    def test_earned_runs(self):
        val, opp = _pitcher_stat({"inningsPitched": "5.0", "earnedRuns": 3}, "earned_runs")
        assert val == 3.0 and opp == 1.0

    def test_walks_allowed(self):
        val, opp = _pitcher_stat({"inningsPitched": "6.0", "baseOnBalls": 2}, "walks_allowed")
        assert val == 2.0 and opp == 1.0

    def test_hits_allowed(self):
        val, opp = _pitcher_stat({"inningsPitched": "7.0", "hits": 5}, "hits_allowed")
        assert val == 5.0 and opp == 1.0

    def test_zero_ip_returns_zero_opp(self):
        val, opp = _pitcher_stat({"inningsPitched": "0.0", "strikeOuts": 1}, "pitcher_strikeouts")
        assert opp == 0.0

    def test_missing_ip_returns_zero_opp(self):
        val, opp = _pitcher_stat({}, "pitcher_strikeouts")
        assert opp == 0.0


# ─── _weighted_rate ──────────────────────────────────────────────────────────

class TestWeightedRate:
    def _make_games(self, days_and_hits: list[tuple[int, int]], ab_per=4, pa_per=5):
        return [_batter_game(d, hits=h, ab=ab_per, pa=pa_per, tb=h) for d, h in days_and_hits]

    def test_returns_none_when_too_few_games(self):
        games = self._make_games([(1, 2), (2, 1)])  # only 2 games
        result = _weighted_rate(games, "hits", False, L7_DECAY, 7, TODAY)
        assert result is None

    def test_basic_weighted_average(self):
        # yesterday: 2/4, 2 days ago: 0/4, 3 days ago: 4/4
        games = self._make_games([(1, 2), (2, 0), (3, 4)])
        result = _weighted_rate(games, "hits", False, L7_DECAY, 7, TODAY)
        assert result is not None
        # Weight for days_ago: yesterday=1.0, 2_days=0.85, 3_days=0.85^2
        w1 = 1.0
        w2 = L7_DECAY
        w3 = L7_DECAY ** 2
        expected = (w1 * 2 + w2 * 0 + w3 * 4) / (w1 * 4 + w2 * 4 + w3 * 4)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_excludes_games_outside_window(self):
        # L7 window: only last 7 days
        games = self._make_games([(1, 3), (2, 3), (3, 3), (8, 0), (9, 0), (10, 0)])
        # Games at 8,9,10 days ago should be excluded from L7
        result = _weighted_rate(games, "hits", False, L7_DECAY, 7, TODAY)
        assert result is not None
        assert result == pytest.approx(3.0 / 4.0)  # all qualifying games: 3/4

    def test_excludes_today(self):
        # days_ago=0 means today, should be excluded
        games = self._make_games([(0, 4), (1, 1), (2, 1), (3, 1)])
        result = _weighted_rate(games, "hits", False, L7_DECAY, 7, TODAY)
        # Should not include today's game (4/4)
        assert result is not None
        assert result < 1.0  # only the 1/4 games count

    def test_returns_none_when_window_empty(self):
        # All games are older than 7 days
        games = self._make_games([(10, 2), (11, 2), (12, 2), (13, 2)])
        result = _weighted_rate(games, "hits", False, L7_DECAY, 7, TODAY)
        assert result is None

    def test_pitcher_per_game(self):
        games = [_pitcher_game(d, k=k) for d, k in [(1, 8), (2, 6), (3, 7)]]
        result = _weighted_rate(games, "pitcher_strikeouts", True, L7_DECAY, 7, TODAY)
        assert result is not None
        # Each game has opp=1.0, so it's a weighted average of k values
        w1, w2, w3 = 1.0, L7_DECAY, L7_DECAY ** 2
        expected = (w1 * 8 + w2 * 6 + w3 * 7) / (w1 + w2 + w3)
        assert result == pytest.approx(expected, rel=1e-6)


# ─── _season_rate ────────────────────────────────────────────────────────────

class TestSeasonRate:
    def test_returns_none_when_too_few_games(self):
        games = [_batter_game(d, hits=2, ab=4) for d in range(20, 25)]  # 5 games
        result = _season_rate(games, "hits", False, TODAY)
        assert result is None

    def test_basic_average(self):
        # 10 games, each 2/4 → rate = 0.5
        games = [_batter_game(d, hits=2, ab=4) for d in range(15, 25)]
        result = _season_rate(games, "hits", False, TODAY)
        assert result == pytest.approx(0.5)

    def test_excludes_today_and_future(self):
        # Include today (days_ago=0) and yesterday
        games = [_batter_game(d, hits=4, ab=4) for d in range(0, 5)]
        games += [_batter_game(d, hits=1, ab=4) for d in range(5, 25)]
        result = _season_rate(games, "hits", False, TODAY)
        # days_ago=0 excluded; days_ago=1..4 have 4/4=1.0, days_ago=5..24 have 1/4=0.25
        # Only days_ago >= 1 are before TODAY
        assert result is not None
        assert result > 0.25  # hot recent games push it up


# ─── compute_recent_form_multiplier ─────────────────────────────────────────

class TestComputeRecentFormMultiplier:
    """Tests using mocked MLB Stats API calls."""

    def _make_season_games(self, season_rate_hits_per_ab: float,
                           recent_rate_hits_per_ab: float,
                           n_season: int = 30,
                           n_recent: int = 7) -> list[dict]:
        """
        Build a game log where early games reflect season_rate and recent games
        reflect recent_rate (for hits/AB).
        """
        games = []
        # Older games (season baseline)
        for i in range(n_recent + 1, n_recent + 1 + n_season):
            hits = round(season_rate_hits_per_ab * 4)
            games.append(_batter_game(i, hits=hits, ab=4, pa=5, tb=hits))
        # Recent games
        for i in range(1, n_recent + 1):
            hits = round(recent_rate_hits_per_ab * 4)
            games.append(_batter_game(i, hits=hits, ab=4, pa=5, tb=hits))
        return games

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_hot_player_multiplier_above_one(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        # Season rate: 1/4 = 0.25; recent rate: 3/4 = 0.75 → multiplier ~3.0 but capped at 1.15
        mock_logs.return_value = self._make_season_games(0.25, 0.75, n_recent=7)
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == MULT_CAP

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_cold_player_multiplier_below_one(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        # Season rate: 3/4 = 0.75; recent rate: 0/4 = 0 → multiplier ~0 but floored at 0.85
        mock_logs.return_value = self._make_season_games(0.75, 0.0, n_recent=7)
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == MULT_FLOOR

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_neutral_player_near_one(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        # Same rate in recent and season → multiplier ≈ 1.0
        mock_logs.return_value = self._make_season_games(0.50, 0.50, n_recent=7)
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == pytest.approx(1.0, abs=0.05)

    @patch("src.recent_form._get_player_id")
    def test_returns_one_when_player_not_found(self, mock_id):
        mock_id.return_value = None
        result = compute_recent_form_multiplier("Unknown Player", "hits",
                                                 game_date=TODAY)
        assert result == 1.0

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_returns_one_when_empty_game_log(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        mock_logs.return_value = []
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == 1.0

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_returns_one_when_insufficient_season_games(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        # Only 5 season games (< SEASON_MIN_GAMES)
        mock_logs.return_value = [_batter_game(d, hits=2, ab=4) for d in range(5, 10)]
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == 1.0

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_returns_one_when_no_games_in_recent_windows(self, mock_id, mock_logs):
        mock_id.return_value = 12345
        # All games are older than 14 days — nothing in L7 or L14 windows
        season_games = [_batter_game(d, hits=2, ab=4) for d in range(15, 45)]
        mock_logs.return_value = season_games
        result = compute_recent_form_multiplier("Freddie Freeman", "hits",
                                                 game_date=TODAY)
        assert result == 1.0

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_multiplier_clamped_at_floor(self, mock_id, mock_logs):
        mock_id.return_value = 99
        mock_logs.return_value = self._make_season_games(1.0, 0.0, n_recent=7)
        result = compute_recent_form_multiplier("Player X", "hits", game_date=TODAY)
        assert result >= MULT_FLOOR

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_multiplier_clamped_at_cap(self, mock_id, mock_logs):
        mock_id.return_value = 99
        mock_logs.return_value = self._make_season_games(0.0, 1.0, n_recent=7)
        result = compute_recent_form_multiplier("Player X", "hits", game_date=TODAY)
        assert result <= MULT_CAP

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_pitcher_prop_uses_pitching_group(self, mock_id, mock_logs):
        mock_id.return_value = 555
        season = [_pitcher_game(d, k=6) for d in range(8, 38)]
        recent = [_pitcher_game(d, k=10) for d in [1, 2, 3, 4, 5, 6, 7]]
        mock_logs.return_value = season + recent
        result = compute_recent_form_multiplier("Ace Pitcher", "pitcher_strikeouts",
                                                 game_date=TODAY)
        # Recent K/game (10) > season K/game (6) → should be hot → > 1.0
        assert result > 1.0
        assert result <= MULT_CAP
        # Verify pitching group was requested
        mock_logs.assert_called_once_with(555, TODAY.year, "pitching")

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_batter_prop_uses_hitting_group(self, mock_id, mock_logs):
        mock_id.return_value = 444
        mock_logs.return_value = self._make_season_games(0.25, 0.25)
        compute_recent_form_multiplier("Good Batter", "hits", game_date=TODAY)
        mock_logs.assert_called_once_with(444, TODAY.year, "hitting")

    def test_game_date_string_parsing(self):
        with patch("src.recent_form._get_player_id") as mock_id:
            mock_id.return_value = None
            # Should not raise
            result = compute_recent_form_multiplier(
                "X", "hits", game_date="2026-05-15"
            )
            assert result == 1.0
            mock_id.assert_called_once_with("X", 2026)

    def test_game_date_none_uses_today(self):
        with patch("src.recent_form._get_player_id") as mock_id:
            mock_id.return_value = None
            result = compute_recent_form_multiplier("X", "hits", game_date=None)
            assert result == 1.0


# ─── Multiplier bounds verification ─────────────────────────────────────────

class TestMultiplierBounds:
    """Property-style tests: multiplier always in [MULT_FLOOR, MULT_CAP]."""

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_always_within_bounds(self, mock_id, mock_logs):
        mock_id.return_value = 1
        # Try a range of season/recent rate combinations
        test_cases = [
            (0.1, 0.9),   # very hot
            (0.9, 0.1),   # very cold
            (0.5, 0.5),   # neutral
            (0.0, 0.5),   # zero season rate → fallback
            (0.3, 0.3),   # identical
        ]
        for season_r, recent_r in test_cases:
            if season_r == 0.0:
                mock_logs.return_value = [
                    _batter_game(d, hits=round(recent_r * 4), ab=4)
                    for d in range(1, 8)
                ]
            else:
                season_games = [_batter_game(d, hits=round(season_r * 4), ab=4)
                                for d in range(8, 40)]
                recent_games = [_batter_game(d, hits=round(recent_r * 4), ab=4)
                                for d in range(1, 8)]
                mock_logs.return_value = season_games + recent_games
            result = compute_recent_form_multiplier(
                "Tester", "hits", game_date=TODAY
            )
            assert MULT_FLOOR <= result <= MULT_CAP, (
                f"multiplier {result} out of bounds for season={season_r}, recent={recent_r}"
            )


# ─── Predictor integration ───────────────────────────────────────────────────

class TestPredictorIntegration:
    """recent_form_mult appears in generate_prediction output when signal fires."""

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_hot_player_raises_projection(self, mock_id, mock_logs):
        from src.predictor import generate_prediction

        mock_id.return_value = 777

        # Season: 0.25 hit rate; recent: 0.75 hit rate (very hot)
        season_games = [_batter_game(d, hits=1, ab=4, pa=5, tb=1) for d in range(8, 38)]
        recent_games = [_batter_game(d, hits=3, ab=4, pa=5, tb=3) for d in range(1, 8)]
        mock_logs.return_value = season_games + recent_games

        result_hot = generate_prediction(
            "Hot Hitter", "Hits", "hits", 1.5,
            batter_profile={"avg": 0.270, "pa": 200, "xba": 0.260, "k_rate": 20.0},
            game_date=str(TODAY),
        )
        # Disable recent form for baseline
        mock_id.return_value = None
        result_base = generate_prediction(
            "Hot Hitter", "Hits", "hits", 1.5,
            batter_profile={"avg": 0.270, "pa": 200, "xba": 0.260, "k_rate": 20.0},
            game_date=str(TODAY),
        )

        assert result_hot["projection"] > result_base["projection"], (
            "Hot player should have higher projection than baseline"
        )
        assert "recent_form_mult" in result_hot
        assert result_hot["recent_form_mult"] > 1.0

    @patch("src.recent_form._fetch_game_logs")
    @patch("src.recent_form._get_player_id")
    def test_cold_player_lowers_projection(self, mock_id, mock_logs):
        from src.predictor import generate_prediction

        mock_id.return_value = 888

        # Season: 0.75 hit rate; recent: 0.0 hit rate (very cold)
        season_games = [_batter_game(d, hits=3, ab=4, pa=5, tb=3) for d in range(8, 38)]
        recent_games = [_batter_game(d, hits=0, ab=4, pa=5, tb=0) for d in range(1, 8)]
        mock_logs.return_value = season_games + recent_games

        result_cold = generate_prediction(
            "Cold Hitter", "Hits", "hits", 1.5,
            batter_profile={"avg": 0.290, "pa": 200, "xba": 0.280, "k_rate": 18.0},
            game_date=str(TODAY),
        )

        mock_id.return_value = None
        result_base = generate_prediction(
            "Cold Hitter", "Hits", "hits", 1.5,
            batter_profile={"avg": 0.290, "pa": 200, "xba": 0.280, "k_rate": 18.0},
            game_date=str(TODAY),
        )

        assert result_cold["projection"] < result_base["projection"], (
            "Cold player should have lower projection than baseline"
        )
        assert "recent_form_mult" in result_cold
        assert result_cold["recent_form_mult"] < 1.0

    @patch("src.recent_form._get_player_id")
    def test_no_mult_key_when_neutral(self, mock_id):
        """When form multiplier is 1.0, key should not appear in result."""
        from src.predictor import generate_prediction

        mock_id.return_value = None  # triggers 1.0 fallback

        result = generate_prediction(
            "Test Hitter", "Hits", "hits", 1.5,
            batter_profile={"avg": 0.260, "pa": 150},
            game_date=str(TODAY),
        )
        assert result.get("recent_form_mult") is None
