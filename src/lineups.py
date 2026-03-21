"""
Lineups Module

Fetches MLB Stats API lineup data, batting orders, probable pitchers,
and generates PA adjustment multipliers based on batting position.

This module integrates confirmed lineups into the projection system
by providing context about who's batting when and what pitcher they face.

Data source: MLB Stats API (statsapi.mlb.com) — free, no auth required.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from functools import lru_cache

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Request timeout to avoid hanging API calls
REQUEST_TIMEOUT = 10

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

def _api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """
    Safe GET request to MLB Stats API. Returns parsed JSON or None on failure.
    Never raises — all errors are caught and logged silently.
    """
    url = f"{MLB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[lineups.py] API error for {endpoint}: {e}")
        return None
    except ValueError:
        print(f"[lineups.py] Invalid JSON from {endpoint}")
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

@lru_cache(maxsize=1)
def fetch_todays_games() -> List[dict]:
    """
    Fetch today's MLB game schedule with probable pitchers.

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
    today = datetime.now().strftime("%Y-%m-%d")

    data = _api_get(
        "/schedule",
        params={
            "sportId": 1,
            "date": today,
            "hydrate": "probablePitcher,linescore,team",
        }
    )

    if not data or "dates" not in data:
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            try:
                home_team = game.get("teams", {}).get("home", {})
                away_team = game.get("teams", {}).get("away", {})

                home_pitcher = home_team.get("probablePitcher", {})
                away_pitcher = away_team.get("probablePitcher", {})

                home_abbr = home_team.get("team", {}).get("abbreviation", "")
                away_abbr = away_team.get("team", {}).get("abbreviation", "")

                games.append({
                    "game_pk": game.get("gamePk"),
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
    # Try the live game feed endpoint first (has confirmed lineups)
    data = _api_get(f"/game/{game_pk}/linescore")

    if not data or "teams" not in data:
        # Fallback to boxscore endpoint
        data = _api_get(f"/game/{game_pk}/boxscore")

        if not data or "teams" not in data:
            return {"away": [], "home": []}

    result = {"away": [], "home": []}

    for side in ["home", "away"]:
        try:
            team_data = data.get("teams", {}).get(side, {})
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
            print(f"[lineups.py] Error parsing {side} lineup: {e}")
            continue

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

        # Try to fetch venue info from game details if available
        game_detail = _api_get(f"/game/{game.get('game_pk')}")
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
# CONVENIENCE: FETCH ALL LINEUPS FOR TODAY
# ─────────────────────────────────────────────

def fetch_all_lineups() -> dict:
    """
    Fetch confirmed lineups for all games today.

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
