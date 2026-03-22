"""
PrizePicks API Client
Fetches current MLB player prop projections from PrizePicks' public API.
"""

import requests
import pandas as pd
from datetime import datetime


PP_PROJECTIONS_URL = "https://partner-api.prizepicks.com/projections"
MLB_LEAGUE_IDS = {"9", "43"}  # 9 = regular season, 43 = Spring Training

# Map PrizePicks stat types to our internal names
STAT_TYPE_MAP = {
    "Pitcher Strikeouts": "pitcher_strikeouts",
    "Strikeouts": "pitcher_strikeouts",
    "Hits Allowed": "hits_allowed",
    "Earned Runs Allowed": "earned_runs",
    "Walks Allowed": "walks_allowed",
    "Pitching Outs": "pitching_outs",
    "Hits": "hits",
    "Total Bases": "total_bases",
    "Home Runs": "home_runs",
    "RBIs": "rbis",
    "Runs": "runs",
    "Stolen Bases": "stolen_bases",
    "Hits+Runs+RBIs": "hits_runs_rbis",
    "Singles": "singles",
    "Doubles": "doubles",
    "Walks": "walks",
    "Batter Strikeouts": "batter_strikeouts",
    "Hitter Fantasy Score": "hitter_fantasy_score",
    "Fantasy Score": "hitter_fantasy_score",
}

PITCHER_PROPS = {"pitcher_strikeouts", "hits_allowed", "earned_runs", "walks_allowed", "pitching_outs"}
BATTER_PROPS = {"hits", "total_bases", "home_runs", "rbis", "runs", "stolen_bases",
                "hits_runs_rbis", "singles", "doubles", "walks", "batter_strikeouts",
                "hitter_fantasy_score"}


def fetch_prizepicks_mlb_lines() -> pd.DataFrame:
    """
    Fetch all current MLB projections from PrizePicks.

    Returns a DataFrame with columns:
        player_name, player_id, team, position, stat_type, line,
        stat_internal, is_pitcher_prop, start_time, opponent, league
    """
    try:
        resp = requests.get(
            PP_PROJECTIONS_URL,
            params={"per_page": 1000, "single_stat": "true"},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch PrizePicks data: {e}")

    # Record that we just pulled fresh data
    try:
        from src.freshness import record_data_pull
        record_data_pull("prizepicks_lines", f"{len(data.get('data', []))} projections")
    except Exception:
        pass

    projections = data.get("data", [])
    included = data.get("included", [])

    # Build lookup for player info from "included" section
    player_lookup = {}
    for item in included:
        if item.get("type") == "new_player":
            attrs = item.get("attributes", {})
            player_lookup[item["id"]] = {
                "name": attrs.get("display_name", attrs.get("name", "Unknown")),
                "team": attrs.get("team", ""),
                "position": attrs.get("position", ""),
            }

    rows = []
    for proj in projections:
        attrs = proj.get("attributes", {})

        # Filter to MLB only — league ID is in relationships, NOT attributes
        league_id = str(
            proj.get("relationships", {})
                .get("league", {})
                .get("data", {})
                .get("id", "")
        )
        game_id = attrs.get("game_id", "")

        is_mlb = league_id in MLB_LEAGUE_IDS or str(game_id).startswith("MLB_")
        if not is_mlb:
            continue

        stat_type_raw = attrs.get("stat_type", "")
        stat_internal = STAT_TYPE_MAP.get(stat_type_raw, stat_type_raw.lower().replace(" ", "_"))
        is_pitcher = stat_internal in PITCHER_PROPS

        # Get player info
        player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
        player_id = player_rel.get("id", "")
        player_info = player_lookup.get(player_id, {})

        line_score = attrs.get("line_score")
        if line_score is None:
            continue

        league_label = "MLB" if league_id == "9" else ("Spring Training" if league_id == "43" else "MLB")

        rows.append({
            "player_name": player_info.get("name", "Unknown"),
            "player_id": player_id,
            "team": player_info.get("team", ""),
            "position": player_info.get("position", ""),
            "stat_type": stat_type_raw,
            "stat_internal": stat_internal,
            "line": float(line_score),
            "is_pitcher_prop": is_pitcher,
            "start_time": attrs.get("start_time", ""),
            "description": attrs.get("description", ""),
            "league": league_label,
        })

    df = pd.DataFrame(rows)

    if not df.empty and "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df = df.sort_values(["start_time", "player_name"]).reset_index(drop=True)

    return df


def get_available_stat_types(df: pd.DataFrame) -> list:
    """Return sorted list of unique stat types in the current lines."""
    if df.empty:
        return []
    return sorted(df["stat_type"].unique().tolist())


def get_available_games(df: pd.DataFrame) -> list:
    """Return list of unique game descriptions."""
    if df.empty:
        return []
    return sorted(df["description"].unique().tolist())
