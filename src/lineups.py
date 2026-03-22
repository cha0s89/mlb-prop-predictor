"""
Lineups Module

Fetches MLB Stats API lineup data, batting orders, probable pitchers,
and generates PA adjustment multipliers based on batting position.

This module integrates confirmed lineups into the projection system
by providing context about who's batting when and what pitcher they face.

Data source: MLB Stats API (statsapi.mlb.com) — free, no auth required.
Uses the v1.1 live feed endpoint for boxscore/lineup data.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_API_V11 = "https://statsapi.mlb.com/api/v1.1"

# Request timeout to avoid hanging API calls
REQUEST_TIMEOUT = 10

# In-memory cache for lineup fetches.  Stores results (including None for
# games that failed or have no lineup data) so we never hit the same game_pk
# twice in a single run.  Keyed by game_pk.
_lineup_cache: Dict[int, Optional[dict]] = {}

# Track game_pks that already failed so we log only once per game.
_failed_game_pks: set = set()

# Team ID to abbreviation mapping (key MLB teams)
TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "ARI", 110: "LAD", 111: "COL", 112: "SD",
    113: "SF", 114: "STL", 115: "CHC", 116: "MIL", 117: "PIT",
    118: "CIN", 119: "MIL", 133: "OAK", 134: "SEA", 135: "TEX",
    136: "TOR", 137: "BAL", 138: "BOS", 139: "NYY", 140: "TB",
    141: "NYM", 142: "PHI", 143: "ATL", 144: "WSH", 145: "CWS",
    146: "DET", 147: "MIN", 158: "MIA",
}

# Abbreviation to full name
ABBR_TO_NAME = {
    "LAA": "Los Angeles Angels", "ARI": "Arizona Diamondbacks",
    "LAD": "Los Angeles Dodgers", "COL": "Colorado Rockies",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals", "CHC": "Chicago Cubs",
    "MIL": "Milwaukee Brewers", "PIT": "Pittsburgh Pirates",
    "CIN": "Cincinnati Reds", "OAK": "Oakland Athletics",
    "SEA": "Seattle Mariners", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "NYY": "New York Yankees",
    "TB": "Tampa Bay Rays", "NYM": "New York Mets",
    "PHI": "Philadelphia Phillies", "ATL": "Atlanta Braves",
    "WSH": "Washington Nationals", "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers", "MIN": "Minnesota Twins",
    "MIA": "Miami Marlins",
}

# PA multipliers by batting position (1-9)
# Leadoff get more PAs, 9th hitter fewer
PA_MULTIPLIERS = {
    1: 1.08,  # Leadoff — most PAs
    2: 1.06,
    3: 1.04,
    4: 1.03,
    5: 1.02,
    6: 1.01,
    7: 0.99,
    8: 0.95,
    9: 0.90,   # 9th hitter — fewer PAs
}


# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None,
             base: str = None, silent: bool = False) -> Optional[dict]:
    """
    Safe GET request to MLB Stats API. Returns parsed JSON or None on failure.
    Never raises — all errors are caught and logged once.

    Args:
        endpoint: URL path (e.g. "/schedule")
        params: Query parameters
        base: Override base URL (defaults to MLB_API_BASE v1)
        silent: If True, suppress even the single-line error log
    """
    url = f"{base or MLB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if not silent:
            logger.warning("API error %s: %s", endpoint, e)
        return None
    except ValueError:
        if not silent:
            logger.warning("Invalid JSON from %s", endpoint)
        return None


def _normalize_team_abbr(team_input: str) -> Optional[str]:
    """
    Normalize team input to uppercase abbreviation.
    Handles both abbreviations (NYY) and full names (New York Yankees).
    Returns the standardized abbreviation or None if not found.
    """
    if not team_input:
        return None

    team_input = team_input.upper().strip()

    # Already an abbreviation?
    if team_input in ABBR_TO_NAME:
        return team_input

    # Search in names (case-insensitive)
    for abbr, name in ABBR_TO_NAME.items():
        if name.upper() == team_input:
            return abbr

    return None


# ─────────────────────────────────────────────
# TODAY'S GAMES
# ─────────────────────────────────────────────

def fetch_todays_games(game_dates: List[str] = None) -> List[dict]:
    """
    Fetch MLB game schedule with probable pitchers for given dates.

    Args:
        game_dates: List of date strings (YYYY-MM-DD) to fetch.
                    Defaults to today if not provided.

    Returns a list of game dicts, each containing:
        - game_pk: unique game ID
        - game_time: ISO timestamp of game
        - status: game status string
        - away_team: away team abbreviation
        - home_team: home team abbreviation
        - away_team_name: full away team name
        - home_team_name: full home team name
        - away_pitcher_name: probable pitcher name
        - away_pitcher_id: probable pitcher MLB ID
        - away_pitcher_hand: R or L
        - home_pitcher_name: probable pitcher name
        - home_pitcher_id: probable pitcher MLB ID
        - home_pitcher_hand: R or L

    Returns empty list if API fails.
    """
    if not game_dates:
        game_dates = [datetime.now().strftime("%Y-%m-%d")]

    # Deduplicate dates
    unique_dates = sorted(set(game_dates))

    games = []
    seen_pks = set()
    for gdate in unique_dates:
        data = _api_get(
            "/schedule",
            params={
                "sportId": 1,
                "date": gdate,
                "hydrate": "probablePitcher,linescore,team",
            }
        )

        if not data or "dates" not in data:
            continue

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                try:
                    gpk = game.get("gamePk")
                    if gpk in seen_pks:
                        continue
                    seen_pks.add(gpk)

                    home_team = game.get("teams", {}).get("home", {})
                    away_team = game.get("teams", {}).get("away", {})

                    home_pitcher = home_team.get("probablePitcher", {})
                    away_pitcher = away_team.get("probablePitcher", {})

                    home_abbr = home_team.get("team", {}).get("abbreviation", "")
                    away_abbr = away_team.get("team", {}).get("abbreviation", "")

                    games.append({
                        "game_pk": gpk,
                        "game_time": game.get("gameDate", ""),
                        "status": game.get("status", {}).get("detailedState", ""),
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "home_team_name": home_team.get("team", {}).get("name", ""),
                        "away_team_name": away_team.get("team", {}).get("name", ""),
                        "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                        "home_pitcher_id": home_pitcher.get("id"),
                        "home_pitcher_hand": home_pitcher.get("pitchHand", {}).get("code", ""),
                        "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                        "away_pitcher_id": away_pitcher.get("id"),
                        "away_pitcher_hand": away_pitcher.get("pitchHand", {}).get("code", ""),
                    })
                except Exception as e:
                    print(f"[lineups.py] Error parsing game: {e}")
                    continue

    return games


# ─────────────────────────────────────────────
# CONFIRMED LINEUPS
# ─────────────────────────────────────────────

def fetch_confirmed_lineups(game_pk: int) -> dict:
    """
    Fetch confirmed batting orders for a specific game.

    Uses the v1.1 live feed endpoint which returns boxscore data with
    confirmed lineups.  Results are cached so repeated calls for the
    same game_pk are free.  Failed lookups (spring training games
    without data, 405s, etc.) are cached as empty and logged once.

    Args:
        game_pk: MLB game ID

    Returns:
        Dict with 'away' and 'home' keys, each containing list of:
            {
                'player_id': int,
                'player_name': str,
                'position': str (e.g., 'SS', 'OF'),
                'bat_hand': str (R, L, or S for switch),
                'batting_order': int (1-9),
                'is_starting': bool
            }

        Returns empty dict on failure.
    """
    empty = {"away": [], "home": []}

    # ── Cache check: return immediately if we've seen this game_pk ──
    if game_pk in _lineup_cache:
        return _lineup_cache[game_pk] or empty

    # ── Primary: v1.1 live feed → boxscore ──
    data = _api_get(
        f"/game/{game_pk}/feed/live",
        base=MLB_API_V11,
        silent=(game_pk in _failed_game_pks),
    )

    boxscore = None
    if data and "liveData" in data:
        boxscore = data.get("liveData", {}).get("boxscore")

    # ── Fallback: v1 boxscore endpoint ──
    if not boxscore or "teams" not in boxscore:
        fallback = _api_get(
            f"/game/{game_pk}/boxscore",
            silent=(game_pk in _failed_game_pks),
        )
        if fallback and "teams" in fallback:
            boxscore = fallback

    # ── Both failed — cache the miss, log once ──
    if not boxscore or "teams" not in boxscore:
        if game_pk not in _failed_game_pks:
            logger.info("No lineup data for game %s (spring training or not yet posted)", game_pk)
            _failed_game_pks.add(game_pk)
        _lineup_cache[game_pk] = None
        return empty

    # ── Parse the boxscore ──
    result = {"away": [], "home": []}

    for side in ["home", "away"]:
        try:
            team_data = boxscore.get("teams", {}).get(side, {})
            players = team_data.get("players", {})
            batting_order = team_data.get("battingOrder", [])

            for batting_order_idx, player_id in enumerate(batting_order, 1):
                player_key = f"ID{player_id}"
                player_info = players.get(player_key, {})
                person = player_info.get("person", {})

                result[side].append({
                    "player_id": player_id,
                    "player_name": person.get("fullName", "Unknown"),
                    "position": player_info.get("position", {}).get("abbreviation", ""),
                    "bat_hand": player_info.get("batSide", {}).get("code", ""),
                    "batting_order": batting_order_idx,
                    "is_starting": True,
                })
        except Exception as e:
            logger.warning("Error parsing %s lineup for game %s: %s", side, game_pk, e)
            continue

    _lineup_cache[game_pk] = result
    return result


# ─────────────────────────────────────────────
# BATTING ORDER & PA MULTIPLIERS
# ─────────────────────────────────────────────

def get_batting_order_position(
    player_name: str,
    games: List[dict] = None
) -> Optional[int]:
    """
    Find a player's batting order position (1-9) in today's games.

    Args:
        player_name: Full player name to search for
        games: Pre-fetched list of games from fetch_todays_games().
               If None, will fetch.

    Returns:
        Batting order position (1-9) or None if not found in any lineup.
    """
    if games is None:
        games = fetch_todays_games()

    if not games:
        return None

    # Normalize input name for matching
    search_name = player_name.upper().strip() if player_name else ""
    if not search_name:
        return None

    for game in games:
        # Check both lineups
        lineups = fetch_confirmed_lineups(game.get("game_pk", 0))

        for side in ["home", "away"]:
            for batter in lineups.get(side, []):
                batter_name = batter.get("player_name", "").upper().strip()
                if batter_name == search_name:
                    return batter.get("batting_order")

    return None


def get_pa_multiplier(batting_position: int) -> float:
    """
    Get PA adjustment multiplier for a batting order position.

    Leadoff (1) gets more PAs, 9th hitter gets fewer.

    Args:
        batting_position: Int 1-9

    Returns:
        Float multiplier (e.g., 1.08 for leadoff, 0.90 for 9th)
        Defaults to 1.0 if position invalid.
    """
    if not isinstance(batting_position, int) or batting_position < 1 or batting_position > 9:
        return 1.0

    return PA_MULTIPLIERS.get(batting_position, 1.0)


# ─────────────────────────────────────────────
# PROBABLE PITCHERS
# ─────────────────────────────────────────────

def get_probable_pitcher(team_abbrev: str, games: List[dict] = None) -> Optional[dict]:
    """
    Get the probable pitcher for a team in today's games.

    Args:
        team_abbrev: Team abbreviation (e.g., 'NYY') or full name
        games: Pre-fetched games list. If None, will fetch.

    Returns:
        Dict with pitcher details:
            {
                'pitcher_name': str,
                'pitcher_id': int,
                'hand': str (R or L),
                'team': str (abbrev),
                'opponent': str (abbrev),
                'game_pk': int
            }
        Returns None if pitcher not found.
    """
    if games is None:
        games = fetch_todays_games()

    if not games:
        return None

    # Normalize team input
    team_abbrev = _normalize_team_abbr(team_abbrev)
    if not team_abbrev:
        return None

    for game in games:
        # Home pitcher?
        if game.get("home_team") == team_abbrev:
            pitcher_id = game.get("home_pitcher_id")
            if pitcher_id:
                return {
                    "pitcher_name": game.get("home_pitcher_name"),
                    "pitcher_id": pitcher_id,
                    "hand": game.get("home_pitcher_hand"),
                    "team": game.get("home_team"),
                    "opponent": game.get("away_team"),
                    "game_pk": game.get("game_pk"),
                }

        # Away pitcher?
        if game.get("away_team") == team_abbrev:
            pitcher_id = game.get("away_pitcher_id")
            if pitcher_id:
                return {
                    "pitcher_name": game.get("away_pitcher_name"),
                    "pitcher_id": pitcher_id,
                    "hand": game.get("away_pitcher_hand"),
                    "team": game.get("away_team"),
                    "opponent": game.get("home_team"),
                    "game_pk": game.get("game_pk"),
                }

    return None


# ─────────────────────────────────────────────
# GAME CONTEXT
# ─────────────────────────────────────────────

def get_game_context(team_abbrev: str, games: List[dict] = None) -> Optional[dict]:
    """
    Get full game context for a team: venue, time, probable pitcher, weather if available.

    Args:
        team_abbrev: Team abbreviation or full name
        games: Pre-fetched games list. If None, will fetch.

    Returns:
        Dict with:
            {
                'game_pk': int,
                'game_time': str (ISO timestamp),
                'status': str,
                'team': str (abbrev),
                'opponent': str (abbrev),
                'opponent_name': str,
                'is_home': bool,
                'probable_pitcher': dict (from get_probable_pitcher),
                'venue': str (if available),
                'temp_f': int or None,
                'wind_mph': int or None
            }
        Returns None if team not in any game.
    """
    if games is None:
        games = fetch_todays_games()

    if not games:
        return None

    # Normalize team
    team_abbrev = _normalize_team_abbr(team_abbrev)
    if not team_abbrev:
        return None

    for game in games:
        is_home = False
        opponent = None

        if game.get("home_team") == team_abbrev:
            is_home = True
            opponent = game.get("away_team")
            opponent_name = game.get("away_team_name")
        elif game.get("away_team") == team_abbrev:
            is_home = False
            opponent = game.get("home_team")
            opponent_name = game.get("home_team_name")
        else:
            continue

        # Get probable pitcher for this team
        pitcher = get_probable_pitcher(team_abbrev, games=[game])

        # Try to fetch venue info from live feed (v1.1)
        game_detail = _api_get(
            f"/game/{game.get('game_pk')}/feed/live",
            base=MLB_API_V11,
            silent=True,
        )
        venue = ""
        if game_detail:
            venue = game_detail.get("gameData", {}).get("venue", {}).get("name", "")

        result = {
            "game_pk": game.get("game_pk"),
            "game_time": game.get("game_time"),
            "status": game.get("status"),
            "team": team_abbrev,
            "opponent": opponent,
            "opponent_name": opponent_name,
            "is_home": is_home,
            "probable_pitcher": pitcher,
            "venue": venue,
            "temp_f": None,
            "wind_mph": None,
        }

        return result

    return None


# ─────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────

def clear_lineup_cache():
    """Reset the in-memory lineup cache.  Useful at the start of a new run."""
    _lineup_cache.clear()
    _failed_game_pks.clear()
    fetch_todays_games.cache_clear()


# ─────────────────────────────────────────────
# CONVENIENCE: FETCH ALL LINEUPS FOR TODAY
# ─────────────────────────────────────────────

def fetch_all_lineups() -> dict:
    """
    Fetch confirmed lineups for all games today.

    Results are cached per game_pk — safe to call multiple times.

    Returns:
        Dict mapping game_pk -> {away: [...], home: [...]} lineups
        Empty dict if no games or API failure.
    """
    games = fetch_todays_games()
    if not games:
        return {}

    all_lineups = {}
    for game in games:
        game_pk = game.get("game_pk")
        if game_pk:
            all_lineups[game_pk] = fetch_confirmed_lineups(game_pk)

    return all_lineups


# ─────────────────────────────────────────────
# PLAYER SEARCH IN LINEUPS
# ─────────────────────────────────────────────

def find_player_in_lineup(
    player_name: str,
    games: List[dict] = None
) -> Optional[dict]:
    """
    Find a player in today's lineups with full context.

    Args:
        player_name: Player name to search for
        games: Pre-fetched games list

    Returns:
        Dict with:
            {
                'player_name': str,
                'player_id': int,
                'team': str (abbrev),
                'position': str,
                'bat_hand': str,
                'batting_order': int,
                'pa_multiplier': float,
                'game_pk': int,
                'opponent': str (abbrev)
            }
        Returns None if not found.
    """
    if games is None:
        games = fetch_todays_games()

    if not games:
        return None

    search_name = player_name.upper().strip() if player_name else ""
    if not search_name:
        return None

    for game in games:
        game_pk = game.get("game_pk", 0)
        lineups = fetch_confirmed_lineups(game_pk)

        for side in ["home", "away"]:
            for batter in lineups.get(side, []):
                batter_name = batter.get("player_name", "").upper().strip()
                if batter_name == search_name:
                    batting_order = batter.get("batting_order", 0)
                    return {
                        "player_name": batter.get("player_name"),
                        "player_id": batter.get("player_id"),
                        "team": game.get("home_team") if side == "home" else game.get("away_team"),
                        "position": batter.get("position"),
                        "bat_hand": batter.get("bat_hand"),
                        "batting_order": batting_order,
                        "pa_multiplier": get_pa_multiplier(batting_order),
                        "game_pk": game_pk,
                        "opponent": game.get("away_team") if side == "home" else game.get("home_team"),
                    }

    return None
