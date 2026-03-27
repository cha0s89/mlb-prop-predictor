"""Divisional familiarity adjustment for pitcher K props.

Division rivals face each other 19x/season. Pitchers show a measurable 4-6%
K rate decline by the 4th+ series as hitters adapt to familiar sequencing,
pitch selection, and release point.

Graduated multiplier for pitcher_strikeouts:
  1st series  → 1.00 (no adjustment)
  2nd series  → 0.99
  3rd series  → 0.97
  4th+        → 0.95

Batter strikeouts receive ~60% of the pitcher effect (batters also K less
against a well-studied arm, but the signal is weaker than the pitcher side).

Non-division matchups always return 1.0. Graceful fallback to 1.0 on any
API error or missing team data.
"""

from __future__ import annotations

import datetime
import requests
from functools import lru_cache
from typing import Optional

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ── Division membership ───────────────────────────────────────────────────────

DIVISION_MAP: dict[str, str] = {
    # AL East
    "BAL": "AL_EAST", "BOS": "AL_EAST", "NYY": "AL_EAST", "TB": "AL_EAST", "TOR": "AL_EAST",
    # AL Central
    "CWS": "AL_CENTRAL", "CLE": "AL_CENTRAL", "DET": "AL_CENTRAL", "KC": "AL_CENTRAL", "MIN": "AL_CENTRAL",
    # AL West
    "HOU": "AL_WEST", "LAA": "AL_WEST", "OAK": "AL_WEST", "SEA": "AL_WEST", "TEX": "AL_WEST",
    # NL East
    "ATL": "NL_EAST", "MIA": "NL_EAST", "NYM": "NL_EAST", "PHI": "NL_EAST", "WSH": "NL_EAST",
    # NL Central
    "CHC": "NL_CENTRAL", "CIN": "NL_CENTRAL", "MIL": "NL_CENTRAL", "PIT": "NL_CENTRAL", "STL": "NL_CENTRAL",
    # NL West
    "ARI": "NL_WEST", "COL": "NL_WEST", "LAD": "NL_WEST", "SD": "NL_WEST", "SF": "NL_WEST",
}

# MLB Stats API team IDs (for schedule lookup)
TEAM_IDS: dict[str, int] = {
    "BAL": 110, "BOS": 111, "NYY": 147, "TB": 139, "TOR": 141,
    "CWS": 145, "CLE": 114, "DET": 116, "KC": 118, "MIN": 142,
    "HOU": 117, "LAA": 108, "OAK": 133, "SEA": 136, "TEX": 140,
    "ATL": 144, "MIA": 146, "NYM": 121, "PHI": 143, "WSH": 120,
    "CHC": 112, "CIN": 113, "MIL": 158, "PIT": 134, "STL": 138,
    "ARI": 109, "COL": 115, "LAD": 119, "SD": 135, "SF": 137,
}

# Suppression per series index (0 = 1st meeting, 3 = 4th+ meeting)
_PITCHER_MULTS = [1.00, 0.99, 0.97, 0.95]

# Batter K suppression is ~60% as strong as pitcher K suppression
_BATTER_SCALE = 0.6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_same_division(team_a: str, team_b: str) -> bool:
    """Return True if team_a and team_b are different teams in the same division."""
    a = DIVISION_MAP.get(team_a.upper())
    b = DIVISION_MAP.get(team_b.upper())
    return bool(a and b and a == b and team_a.upper() != team_b.upper())


@lru_cache(maxsize=512)
def _count_games_between(team_abbr: str, opp_abbr: str, season: int, before_date: str) -> int:
    """
    Count completed regular-season games between two teams before *before_date*.

    Results are cached by (team, opponent, season, date) so repeated lookups
    within the same process are free.
    """
    team_id = TEAM_IDS.get(team_abbr)
    opp_id = TEAM_IDS.get(opp_abbr)
    if not team_id or not opp_id:
        return 0

    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "sportId": 1,
                "teamId": team_id,
                "season": season,
                "gameType": "R",
                "startDate": f"{season}-03-01",
                "endDate": before_date,
                "hydrate": "teams",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return 0

    count = 0
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home_id = teams.get("home", {}).get("team", {}).get("id")
            away_id = teams.get("away", {}).get("team", {}).get("id")
            if {home_id, away_id} == {team_id, opp_id}:
                count += 1
    return count


def _series_index(games_played: int) -> int:
    """Map games-played to a 0-based series index (assumes ~3-game series)."""
    if games_played <= 0:
        return 0
    return min((games_played - 1) // 3, len(_PITCHER_MULTS) - 1)


def clear_familiarity_cache() -> None:
    """Clear the schedule lookup cache (useful in tests)."""
    _count_games_between.cache_clear()


# ── Public API ────────────────────────────────────────────────────────────────

def get_familiarity_adjustment(
    pitcher_team: str,
    batter_team: str,
    game_date=None,
    season: Optional[int] = None,
    prop_type: str = "pitcher_strikeouts",
) -> float:
    """
    Return a K-rate suppression multiplier based on divisional familiarity.

    Parameters
    ----------
    pitcher_team : str
        Abbreviation of the pitcher's team (e.g. ``"NYY"``).
    batter_team : str
        Abbreviation of the batting team (e.g. ``"BOS"``).
    game_date : str | date | datetime | None
        Date of the game. Used to count prior meetings this season.
        Defaults to today if None.
    season : int | None
        Season year. Inferred from *game_date* or today when None.
    prop_type : str
        ``"pitcher_strikeouts"`` or ``"batter_strikeouts"``.

    Returns
    -------
    float
        Multiplier ≤ 1.0.  Returns ``1.0`` for non-division matchups or on
        any data error.
    """
    if not pitcher_team or not batter_team:
        return 1.0

    pt = pitcher_team.upper()
    bt = batter_team.upper()

    if not _in_same_division(pt, bt):
        return 1.0

    # Resolve date
    if game_date is None:
        resolved_date = datetime.date.today()
    elif isinstance(game_date, datetime.datetime):
        resolved_date = game_date.date()
    elif isinstance(game_date, datetime.date):
        resolved_date = game_date
    else:
        try:
            resolved_date = datetime.date.fromisoformat(str(game_date)[:10])
        except ValueError:
            resolved_date = datetime.date.today()

    if season is None:
        season = resolved_date.year

    before_str = str(resolved_date)

    try:
        games_played = _count_games_between(pt, bt, season, before_str)
    except Exception:
        games_played = 0

    idx = _series_index(games_played)
    pitcher_mult = _PITCHER_MULTS[idx]

    if prop_type == "batter_strikeouts":
        suppression = 1.0 - pitcher_mult          # 0.0 at 1st series, 0.05 at 4th+
        return 1.0 - suppression * _BATTER_SCALE  # milder effect for batters

    return pitcher_mult
