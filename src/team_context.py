"""Shared helpers for schedule-date extraction and team-context lookups."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

from src.weather import resolve_team


def extract_schedule_dates(start_times: Iterable, fallback_date: str | None = None) -> list[str]:
    """Return all schedule dates implied by PP start times."""
    dates: set[str] = set()
    values = start_times if start_times is not None else []
    for raw_value in values:
        try:
            ts = pd.Timestamp(raw_value)
        except Exception:
            continue
        dates.add(ts.strftime("%Y-%m-%d"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        dates.add(ts.tz_convert("UTC").strftime("%Y-%m-%d"))

    if not dates:
        dates.add(fallback_date or date.today().isoformat())
    return sorted(dates)


def team_lookup_keys(team_abbr: str) -> tuple[str, ...]:
    """Return all lookup keys that should resolve to the same team."""
    raw = str(team_abbr or "").strip().upper()
    if not raw:
        return tuple()
    resolved = resolve_team(raw)
    keys = [raw]
    if resolved and resolved not in keys:
        keys.append(resolved)
    return tuple(keys)


def register_team_value(mapping: dict, team_abbr: str, value) -> None:
    """Store a value under both raw and normalized team abbreviations."""
    for key in team_lookup_keys(team_abbr):
        mapping[key] = value


def _canonical_game_key(game_pk=None, game_time=None) -> str:
    """Return a stable lookup suffix for a specific game."""
    if game_pk not in (None, "", 0):
        return f"pk:{game_pk}"
    if game_time in (None, ""):
        return ""
    try:
        ts = pd.Timestamp(game_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return f"time:{ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')}"
    except Exception:
        return f"time:{str(game_time).strip()}"


def team_game_lookup_keys(team_abbr: str, game_pk=None, game_time=None) -> tuple[str, ...]:
    """Return lookup keys from most specific (game) to least specific (team)."""
    keys: list[str] = []
    game_key = _canonical_game_key(game_pk=game_pk, game_time=game_time)
    for team_key in team_lookup_keys(team_abbr):
        if game_key:
            keys.append(f"{team_key}|{game_key}")
        keys.append(team_key)
    return tuple(keys)


def register_team_game_value(mapping: dict, team_abbr: str, value, game_pk=None, game_time=None) -> None:
    """Store a value under a game-specific key and a team-only fallback."""
    keys = team_game_lookup_keys(team_abbr, game_pk=game_pk, game_time=game_time)
    if not keys:
        return
    for key in keys:
        if "|" in key:
            mapping[key] = value
    for key in keys:
        if "|" not in key:
            mapping.setdefault(key, value)


def get_team_game_value(mapping: dict, team_abbr: str, game_pk=None, game_time=None):
    """Fetch the best available game-specific value for a team."""
    for key in team_game_lookup_keys(team_abbr, game_pk=game_pk, game_time=game_time):
        if key in mapping:
            return mapping[key]
    return None


def normalize_team_code(value) -> str:
    """Normalize a team code from external sources."""
    raw = str(value or "").strip()
    if not raw or raw == "- - -":
        return ""
    return resolve_team(raw)


def pitcher_row_matches_team(stats_row, expected_team: str) -> bool:
    """Return False only when a matched pitcher row clearly belongs to another team."""
    if stats_row is None:
        return True
    expected = normalize_team_code(expected_team)
    if not expected:
        return True
    stats_team = normalize_team_code(stats_row.get("Team", ""))
    if not stats_team:
        return True
    return stats_team == expected
