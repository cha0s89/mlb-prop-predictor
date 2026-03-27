"""Tests for src/game_script.py — latent game-script classification model."""

import pytest
from src.game_script import classify_game_script


# ─── Fixture helpers ─────────────────────────────────────────────────────────

def _elite_pitcher(**kwargs) -> dict:
    """Burnes/Cole-calibre starter."""
    base = {"era": 2.80, "fip": 2.90, "xfip": 3.20, "gs": 28, "ip": 180.0}
    base.update(kwargs)
    return base


def _avg_pitcher(**kwargs) -> dict:
    """League-average starter."""
    base = {"era": 4.07, "fip": 4.02, "xfip": 4.00, "gs": 25, "ip": 140.0}
    base.update(kwargs)
    return base


def _weak_pitcher(**kwargs) -> dict:
    """Below-average starter (ERA > 4.50)."""
    base = {"era": 5.20, "fip": 5.10, "xfip": 4.90, "gs": 20, "ip": 110.0}
    base.update(kwargs)
    return base


def _opener(**kwargs) -> dict:
    """Opener / short-starter profile."""
    base = {"era": 3.80, "fip": 3.70, "xfip": 3.90, "gs": 2, "ip": 14.0}
    base.update(kwargs)
    return base


# ─── Duel ────────────────────────────────────────────────────────────────────

class TestDuel:
    def test_burnes_vs_cole_low_total(self):
        """Corbin Burnes vs Gerrit Cole, total 6.5 → Duel."""
        result = classify_game_script(
            home_pitcher_profile=_elite_pitcher(era=2.94, fip=2.81),
            away_pitcher_profile=_elite_pitcher(era=2.63, fip=2.55),
            vegas_total=6.5,
            home_moneyline=-115,
            away_moneyline=-105,
        )
        assert result["script"] == "duel"
        assert result["confidence"] > 0.75

    def test_duel_adjustments_direction(self):
        """Duel: pitcher Ks UP, hits DOWN, earned runs DOWN."""
        result = classify_game_script(
            home_pitcher_profile=_elite_pitcher(),
            away_pitcher_profile=_elite_pitcher(),
            vegas_total=7.0,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        adj = result["home_adjustments"]
        assert adj["pitcher_strikeouts"] > 1.0, "pitcher Ks should be UP in a duel"
        assert adj["pitching_outs"] > 1.0,      "pitching outs should be UP"
        assert adj["hits"] < 1.0,                "hits should be DOWN"
        assert adj["earned_runs"] < 1.0,         "earned runs should be DOWN"
        assert adj["total_bases"] < 1.0,         "TB should be DOWN"

    def test_no_duel_when_total_too_high(self):
        """Elite pitchers but total 8.5 → not a duel."""
        result = classify_game_script(
            home_pitcher_profile=_elite_pitcher(),
            away_pitcher_profile=_elite_pitcher(),
            vegas_total=8.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        assert result["script"] != "duel"

    def test_no_duel_when_one_pitcher_weak(self):
        """One weak arm ruins the duel even with low total."""
        result = classify_game_script(
            home_pitcher_profile=_elite_pitcher(),
            away_pitcher_profile=_weak_pitcher(),
            vegas_total=6.5,
            home_moneyline=-130,
            away_moneyline=110,
        )
        assert result["script"] != "duel"

    def test_duel_via_fip_not_era(self):
        """Pitcher with poor ERA but elite FIP should still trigger duel."""
        good_fip = {"era": 4.20, "fip": 2.90, "xfip": 3.10, "gs": 25, "ip": 150.0}
        result = classify_game_script(
            home_pitcher_profile=good_fip,
            away_pitcher_profile=_elite_pitcher(),
            vegas_total=6.5,
            home_moneyline=-115,
            away_moneyline=-105,
        )
        assert result["script"] == "duel"

    def test_duel_via_xfip(self):
        """Pitcher with xFIP < 3.80 (but ERA/FIP above thresholds) still triggers duel."""
        xfip_only = {"era": 3.70, "fip": 3.60, "xfip": 3.70, "gs": 25, "ip": 150.0}
        result = classify_game_script(
            home_pitcher_profile=xfip_only,
            away_pitcher_profile=_elite_pitcher(),
            vegas_total=6.5,
            home_moneyline=-115,
            away_moneyline=-105,
        )
        assert result["script"] == "duel"


# ─── Slugfest ─────────────────────────────────────────────────────────────────

class TestSlugfest:
    def test_two_weak_pitchers_high_total(self):
        """Two 5+ ERA pitchers, total 10.5 → Slugfest."""
        result = classify_game_script(
            home_pitcher_profile=_weak_pitcher(era=5.40),
            away_pitcher_profile=_weak_pitcher(era=5.10),
            vegas_total=10.5,
            home_moneyline=-115,
            away_moneyline=-105,
        )
        assert result["script"] == "slugfest"

    def test_high_total_alone_triggers_slugfest(self):
        """Total ≥10 alone triggers slugfest regardless of pitcher quality."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=10.0,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        assert result["script"] == "slugfest"

    def test_slugfest_adjustments_direction(self):
        """Slugfest: pitcher Ks DOWN, hits UP, TB UP, earned runs UP."""
        result = classify_game_script(
            home_pitcher_profile=_weak_pitcher(),
            away_pitcher_profile=_weak_pitcher(),
            vegas_total=10.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        adj = result["home_adjustments"]
        assert adj["pitcher_strikeouts"] < 1.0,   "pitcher Ks should be DOWN in slugfest"
        assert adj["pitching_outs"] < 1.0,        "pitching outs DOWN"
        assert adj["hits"] > 1.0,                  "hits UP"
        assert adj["total_bases"] > 1.0,           "TB UP"
        assert adj["earned_runs"] > 1.0,           "earned runs UP"
        assert adj["hitter_fantasy_score"] > 1.0,  "FS UP"


# ─── Bullpen ──────────────────────────────────────────────────────────────────

class TestBullpen:
    def test_opener_gs_below_threshold(self):
        """GS < 5 marks a bullpen game."""
        result = classify_game_script(
            home_pitcher_profile=_opener(gs=2),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        assert result["script"] == "bullpen"

    def test_opener_ip_below_threshold(self):
        """IP < 20 (few innings this season) marks a bullpen game."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(ip=12.0),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        assert result["script"] == "bullpen"

    def test_bullpen_adjustments_direction(self):
        """Bullpen game: pitcher Ks DOWN, pitching outs DOWN."""
        result = classify_game_script(
            home_pitcher_profile=_opener(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        adj = result["home_adjustments"]
        assert adj["pitcher_strikeouts"] < 1.0
        assert adj["pitching_outs"] < 1.0


# ─── Blowout Risk ─────────────────────────────────────────────────────────────

class TestBlowoutRisk:
    def test_heavy_home_favourite(self):
        """Home -250 → blowout risk."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-250,
            away_moneyline=210,
        )
        assert result["script"] == "blowout_risk"

    def test_heavy_away_favourite(self):
        """Away -220 → blowout risk."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=185,
            away_moneyline=-220,
        )
        assert result["script"] == "blowout_risk"

    def test_blowout_asymmetric_adjustments(self):
        """Heavy home favourite: home pitcher gets outs↓, away hitters get hits↑."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-250,
            away_moneyline=210,
        )
        # Favourite (home) pitcher pulled early → outs/Ks DOWN
        home_adj = result["home_adjustments"]
        assert home_adj.get("pitching_outs", 1.0) < 1.0
        assert home_adj.get("pitcher_strikeouts", 1.0) < 1.0
        # Underdog (away) hitters pad garbage-time stats → hits UP
        away_adj = result["away_adjustments"]
        assert away_adj.get("hits", 1.0) > 1.0

    def test_blowout_not_triggered_below_200(self):
        """-180 is strong but below the ≥-200 threshold."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-180,
            away_moneyline=155,
        )
        assert result["script"] == "standard"


# ─── Standard ─────────────────────────────────────────────────────────────────

class TestStandard:
    def test_standard_case(self):
        """Average pitchers, average total, balanced moneyline → standard."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-115,
            away_moneyline=-105,
        )
        assert result["script"] == "standard"
        assert result["adjustments"] == {}

    def test_standard_no_adjustment_multipliers(self):
        """Standard game: all adjustment dicts are empty (no multipliers applied)."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=8.5,
            home_moneyline=-110,
            away_moneyline=-110,
        )
        assert result["home_adjustments"] == {}
        assert result["away_adjustments"] == {}


# ─── Missing-data robustness ──────────────────────────────────────────────────

class TestMissingData:
    def test_empty_profiles(self):
        """Missing pitcher profiles do not crash; falls through to standard."""
        result = classify_game_script(
            home_pitcher_profile={},
            away_pitcher_profile={},
            vegas_total=8.5,
            home_moneyline=0,
            away_moneyline=0,
        )
        assert result["script"] in ("standard", "bullpen")  # no ERA data → may be opener check

    def test_none_profiles(self):
        """None pitcher profiles handled gracefully."""
        result = classify_game_script(
            home_pitcher_profile=None,
            away_pitcher_profile=None,
            vegas_total=8.5,
            home_moneyline=0,
            away_moneyline=0,
        )
        assert "script" in result

    def test_zero_total_skips_total_checks(self):
        """total=0 means no total signal; should still classify based on pitchers."""
        result = classify_game_script(
            home_pitcher_profile=_avg_pitcher(),
            away_pitcher_profile=_avg_pitcher(),
            vegas_total=0,
            home_moneyline=-250,
            away_moneyline=210,
        )
        # With a heavy favourite, should still detect blowout risk
        assert result["script"] == "blowout_risk"

    def test_return_keys_always_present(self):
        """Return dict always has all expected keys."""
        result = classify_game_script({}, {}, 0, 0, 0)
        for key in ("script", "confidence", "adjustments", "home_adjustments", "away_adjustments", "reason"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_in_range(self):
        """Confidence is always between 0 and 1."""
        for total in (6.0, 8.5, 10.5):
            result = classify_game_script(_elite_pitcher(), _elite_pitcher(), total, -120, -100)
            assert 0.0 <= result["confidence"] <= 1.0


# ─── Priority ordering ────────────────────────────────────────────────────────

class TestPriority:
    def test_duel_beats_blowout(self):
        """Duel criteria trump blowout risk when total is very low and pitchers elite."""
        result = classify_game_script(
            home_pitcher_profile=_elite_pitcher(),
            away_pitcher_profile=_elite_pitcher(),
            vegas_total=6.0,
            home_moneyline=-250,
            away_moneyline=210,
        )
        assert result["script"] == "duel"

    def test_slugfest_beats_blowout(self):
        """Slugfest (high total + weak arms) should win over blowout signal."""
        result = classify_game_script(
            home_pitcher_profile=_weak_pitcher(),
            away_pitcher_profile=_weak_pitcher(),
            vegas_total=10.5,
            home_moneyline=-210,
            away_moneyline=175,
        )
        assert result["script"] == "slugfest"
