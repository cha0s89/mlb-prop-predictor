"""Rest Days and Travel Schedule Effects Module

Quantifies three documented fatigue/disruption signals:

1. DAY-GAME-AFTER-NIGHT-GAME (DGDN)
   Yesterday: night game (7 PM+ local start)
   Today: day game (before 5 PM local start)
   → Batters: 0.96-0.98 penalty on all hitting props
   → Relievers: 0.95-0.97 penalty on K props

2. CROSS-COUNTRY TRAVEL
   Yesterday's game city vs. today's game city (home-team timezone proxy).
   3+ timezone difference → 0.96 penalty
   1-2 timezone difference → 0.99 penalty
   Same timezone → no penalty

3. PITCHER SHORT REST
   Starting pitcher last start < 4 days ago → 0.94-0.97 on outs/K props
   (Only applies when is_pitcher=True and player_name provided)

All signals compose multiplicatively.  Every error path falls back to 1.0
so a bad API response never breaks a prediction.

Data source: MLB Stats API (free, no auth required).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, date, timezone
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10

# ── timezone offset table (UTC hours, standard time) ──────────────────────────
# Used only for inter-team timezone-difference arithmetic.
# We use "home team" timezone as a proxy for where the game was played.

_TEAM_TZ_OFFSET: dict[str, int] = {
    # Eastern (UTC-5)
    "ATL": -5, "BAL": -5, "BOS": -5, "CIN": -5, "CLE": -5,
    "DET": -5, "MIA": -5, "NYM": -5, "NYY": -5, "PHI": -5,
    "PIT": -5, "TB":  -5, "TOR": -5, "WSH": -5,
    # Central (UTC-6)
    "CHC": -6, "CWS": -6, "HOU": -6, "KC":  -6, "MIL": -6,
    "MIN": -6, "STL": -6, "TEX": -6,
    # Mountain (UTC-7); Arizona skips DST but we only need the offset difference
    "ARI": -7, "COL": -7,
    # Pacific (UTC-8)
    "LAA": -8, "LAD": -8, "OAK": -8, "SD":  -8, "SEA": -8, "SF":  -8,
}

# Pitcher props that short-rest and DGDN (reliever) affect
_PITCHER_K_OPT_PROPS = {"pitcher_strikeouts", "pitching_outs"}

# Batter hitting props that DGDN affects
_BATTER_HITTING_PROPS = {
    "hits", "total_bases", "home_runs", "rbis", "runs", "singles",
    "doubles", "batter_strikeouts", "walks", "hitter_fantasy_score",
    "hits_runs_rbis",
}


# ─── MLB Stats API helper ─────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """Safe GET to MLB Stats API.  Returns JSON or None on failure."""
    url = f"{MLB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("MLB API error (%s): %s", endpoint, exc)
        return None


def _team_id(abbr: str) -> int:
    """Convert team abbreviation to MLB Stats API team ID."""
    try:
        from src.teams import team_id
        return team_id(abbr)
    except Exception:
        return 0


# ─── schedule fetch ───────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _fetch_team_schedule(team_abbr: str, start_date: str, end_date: str) -> list[dict]:
    """
    Fetch completed (Final) games for *team_abbr* in [start_date, end_date].

    Returns a list of game dicts (from the schedule 'games' array) sorted
    ascending by game date.  Results are lru_cache'd by (team, start, end).
    """
    tid = _team_id(team_abbr)
    if not tid:
        return []

    data = _api_get("/schedule", params={
        "sportId": 1,
        "teamId": tid,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "venue,game(content(summary))",
    })
    if not data or "dates" not in data:
        return []

    games: list[dict] = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append(game)

    return sorted(games, key=lambda g: g.get("gameDate", ""))


def _fetch_yesterday_game(team_abbr: str, today: date) -> Optional[dict]:
    """Return the most recent completed game for *team_abbr* before *today*."""
    # Search up to 7 days back to handle off-days
    window_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    window_end   = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    games = _fetch_team_schedule(team_abbr, window_start, window_end)
    # Find last Final game
    for game in reversed(games):
        state = game.get("status", {}).get("detailedState", "")
        if state == "Final":
            return game
    return None


# ─── parsing helpers ──────────────────────────────────────────────────────────

def _parse_utc_hour(game_date_str: str) -> Optional[float]:
    """Parse 'gameDate' ISO string → UTC fractional hour (0-24)."""
    if not game_date_str:
        return None
    try:
        dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
        return dt.hour + dt.minute / 60.0
    except (ValueError, TypeError):
        return None


def _local_hour(utc_hour: float, tz_offset: int) -> float:
    """Convert UTC fractional hour to local time given tz_offset (e.g. -5)."""
    return (utc_hour + tz_offset) % 24


def _home_team_abbr(game: dict, querying_team: str) -> str:
    """Return the home team abbreviation for a scheduled game dict."""
    home = (
        game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        or game.get("teams", {}).get("home", {}).get("team", {}).get("teamCode", "")
    )
    return home.upper() if home else querying_team.upper()


# ─── signal computations ──────────────────────────────────────────────────────

def _dgdn_mult(
    yesterday_game: dict,
    today_utc_hour: float,
    querying_team: str,
    is_pitcher: bool,
) -> float:
    """
    Return DGDN multiplier.

    Night game: local start ≥ 19:00 (7 PM)
    Day  game: local start < 17:00 (5 PM)
    Uses home team's timezone as location proxy.
    """
    # Today's game: is it a day game?
    home_today = querying_team  # caller should pass today's home team if available
    tz_today = _TEAM_TZ_OFFSET.get(querying_team.upper(), -5)
    today_local = _local_hour(today_utc_hour, tz_today)
    if today_local >= 17.0:
        return 1.0  # Not a day game → no DGDN

    # Yesterday's game: was it a night game?
    yest_utc_hour = _parse_utc_hour(yesterday_game.get("gameDate", ""))
    if yest_utc_hour is None:
        return 1.0

    yest_home = _home_team_abbr(yesterday_game, querying_team)
    tz_yest = _TEAM_TZ_OFFSET.get(yest_home, -5)
    yest_local = _local_hour(yest_utc_hour, tz_yest)

    if yest_local < 19.0:
        return 1.0  # Yesterday wasn't a night game

    # DGDN confirmed
    if is_pitcher:
        return 0.96   # Reliever K penalty (mid-range of 0.95-0.97)
    else:
        return 0.97   # Batter penalty (mid-range of 0.96-0.98)


def _travel_mult(
    yesterday_game: dict,
    today_home_team: str,
    querying_team: str,
) -> float:
    """
    Return cross-country travel multiplier.

    Computes timezone difference between yesterday's game venue and today's.
    3+ tz zones: 0.96 penalty
    1-2 tz zones: 0.99 (minimal)
    Same tz:      1.0
    """
    yest_home = _home_team_abbr(yesterday_game, querying_team)
    tz_yest  = _TEAM_TZ_OFFSET.get(yest_home, None)
    tz_today = _TEAM_TZ_OFFSET.get(today_home_team.upper(), None)

    if tz_yest is None or tz_today is None:
        return 1.0

    # If same city/venue (same home team both days) → no travel
    if yest_home.upper() == today_home_team.upper():
        return 1.0

    tz_diff = abs(tz_yest - tz_today)
    if tz_diff >= 3:
        return 0.96
    if tz_diff >= 1:
        return 0.99
    return 1.0


def _short_rest_mult(player_name: str, today: date) -> float:
    """
    Return pitcher short-rest multiplier.

    Searches last 10 games for a start by *player_name* to find their most
    recent start date.  If < 4 days ago → discount applied.
    Falls back to 1.0 on any error.
    """
    if not player_name:
        return 1.0

    # Look up pitcher MLBAM ID from name
    try:
        search = _api_get("/people/search", params={"names": player_name, "limit": 1})
        if not search:
            return 1.0
        people = search.get("people", [])
        if not people:
            return 1.0
        pitcher_id = people[0].get("id")
        if not pitcher_id:
            return 1.0
    except Exception:
        return 1.0

    # Fetch game log for current season
    season = today.year
    log_data = _api_get(f"/people/{pitcher_id}/stats", params={
        "stats": "gameLog",
        "season": season,
        "group": "pitching",
    })
    if not log_data:
        return 1.0

    # Find most recent start
    last_start_date: Optional[date] = None
    for stat_group in log_data.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            if int(stat.get("gamesStarted", 0)) > 0:
                raw_date = split.get("date", "")
                try:
                    d = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
                    if last_start_date is None or d > last_start_date:
                        last_start_date = d
                except (ValueError, TypeError):
                    continue

    if last_start_date is None:
        return 1.0

    days_rest = (today - last_start_date).days - 1  # days between starts
    if days_rest < 0:
        return 1.0  # same-day or future — data anomaly
    if days_rest < 3:
        return 0.94   # Very short rest (< 3 days)
    if days_rest < 4:
        return 0.97   # Short rest (3 days)
    return 1.0


# ─── public API ───────────────────────────────────────────────────────────────

def get_fatigue_adjustment(
    team: str,
    game_date: date | str | None,
    game_time: str | None,
    player_name: str | None = None,
    is_pitcher: bool = False,
    today_home_team: str | None = None,
) -> float:
    """
    Compute a rest/travel fatigue multiplier for a player.

    Args:
        team:            Team abbreviation (e.g. "NYY").
        game_date:       Date of today's game (date or "YYYY-MM-DD" string).
        game_time:       UTC ISO timestamp of today's game start (e.g. "2025-04-15T17:10:00Z").
        player_name:     Player name — only used for pitcher short-rest lookup.
        is_pitcher:      True if this is a pitcher prop.
        today_home_team: Home team abbreviation for today's game (defaults to *team*).
                         Pass the actual home team so travel detection is accurate
                         when the querying team is visiting.

    Returns:
        float in [0.85, 1.0].  Returns 1.0 on any error or missing data.
    """
    try:
        return _compute_fatigue(team, game_date, game_time, player_name, is_pitcher,
                                today_home_team)
    except Exception as exc:
        logger.debug("rest_travel fallback (1.0) for %s/%s: %s", team, player_name, exc)
        return 1.0


def _compute_fatigue(
    team: str,
    game_date: date | str | None,
    game_time: str | None,
    player_name: str | None,
    is_pitcher: bool,
    today_home_team: str | None,
) -> float:
    if not team:
        return 1.0

    # Resolve today's date
    if game_date is None:
        today = date.today()
    elif isinstance(game_date, str):
        try:
            today = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
        except ValueError:
            return 1.0
    else:
        today = game_date

    home = (today_home_team or team).upper()

    # Parse today's UTC game hour
    today_utc_hour = _parse_utc_hour(game_time) if game_time else None

    # Fetch yesterday's game
    yesterday_game = _fetch_yesterday_game(team.upper(), today)
    if not yesterday_game:
        # No recent game data — apply short rest check only if pitcher
        if is_pitcher and player_name:
            return _short_rest_mult(player_name, today)
        return 1.0

    mult = 1.0

    # Signal 1: DGDN
    if today_utc_hour is not None:
        dgdn = _dgdn_mult(yesterday_game, today_utc_hour, home, is_pitcher)
        mult *= dgdn

    # Signal 2: Travel
    travel = _travel_mult(yesterday_game, home, team.upper())
    mult *= travel

    # Signal 3: Pitcher short rest (starters only)
    if is_pitcher and player_name:
        sr = _short_rest_mult(player_name, today)
        mult *= sr

    return round(max(0.85, min(1.0, mult)), 4)


def clear_rest_travel_cache() -> None:
    """Clear the in-memory schedule cache (useful for testing or daily refresh)."""
    _fetch_team_schedule.cache_clear()
