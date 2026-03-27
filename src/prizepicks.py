"""
PrizePicks API Client
Fetches current MLB player prop projections from PrizePicks' public API.
"""

import logging
import requests
import pandas as pd
from datetime import datetime

_log = logging.getLogger(__name__)


PP_PROJECTIONS_URL = "https://partner-api.prizepicks.com/projections"
# Known MLB league ids seen in the PrizePicks feed.
# 2   = current game slate
# 9   = legacy regular-season feed
# 43  = spring training
# 190 = season-long futures
MLB_LEAGUE_IDS = {"2", "9", "43", "190"}
MLB_TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}

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


def fetch_prizepicks_mlb_lines(include_all: bool = False) -> pd.DataFrame:
    """
    Fetch all current MLB projections from PrizePicks.

    Returns a DataFrame with columns:
        player_name, player_id, team, position, stat_type, line,
        stat_internal, is_pitcher_prop, start_time, opponent, league

    Args:
        include_all: When True, return all MLB-related props from the live
            PrizePicks feed, including futures and non-standard lines. Rows
            include model_eligible and eligibility_reason so the UI can explain
            why a prop is not part of the model board.
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

    # Record that we just pulled fresh data.
    try:
        from src.freshness import record_data_pull
        record_data_pull("prizepicks_lines", f"{len(data.get('data', []))} projections")
    except Exception as e:
        _log.warning("Failed to record freshness for prizepicks: %s", e)

    projections = data.get("data", [])
    included = data.get("included", [])

    # Build lookup for player info from the included section.
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
    season_suffix = str(datetime.now().year)[-2:]
    for proj in projections:
        attrs = proj.get("attributes", {})

        league_id = str(
            proj.get("relationships", {})
                .get("league", {})
                .get("data", {})
                .get("id", "")
        )
        game_id = str(attrs.get("game_id", ""))
        stat_type_raw = attrs.get("stat_type", "")
        stat_internal = STAT_TYPE_MAP.get(stat_type_raw, stat_type_raw.lower().replace(" ", "_"))
        stat_supported = stat_type_raw in STAT_TYPE_MAP

        player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
        player_id = player_rel.get("id", "")
        player_info = player_lookup.get(player_id, {})
        team = player_info.get("team", "")

        is_game_prop = game_id.startswith("MLB_")
        is_futures_prop = (
            not is_game_prop
            and stat_supported
            and team in MLB_TEAM_ABBRS
            and (league_id == "190" or game_id.endswith(season_suffix))
        )
        is_mlb = (
            league_id in MLB_LEAGUE_IDS
            or is_game_prop
            or is_futures_prop
            or (stat_supported and team in MLB_TEAM_ABBRS)
        )
        if not is_mlb:
            continue

        line_score = attrs.get("line_score")
        if line_score is None:
            continue

        league_label = "Spring Training" if league_id == "43" else "MLB"
        is_pitcher = stat_internal in PITCHER_PROPS

        is_promo = attrs.get("is_promo", False)
        discount = attrs.get("discount_percentage")
        odds_type = attrs.get("odds_type", "standard")
        flash_sale = attrs.get("flash_sale_line_score")

        line_type = "standard"
        if is_promo or odds_type not in ("standard", ""):
            line_type = "promo"
        elif discount:
            line_type = "discounted"
        elif flash_sale:
            line_type = "flash_sale"

        model_eligible = True
        eligibility_reason = "eligible"
        if not is_game_prop:
            model_eligible = False
            eligibility_reason = "season_long"
        elif league_id == "43":
            model_eligible = False
            eligibility_reason = "spring_training"
        elif line_type != "standard":
            model_eligible = False
            eligibility_reason = "non_standard_line"

        if not include_all and not model_eligible:
            continue

        rows.append({
            "player_name": player_info.get("name", "Unknown"),
            "player_id": player_id,
            "team": team,
            "position": player_info.get("position", ""),
            "stat_type": stat_type_raw,
            "stat_internal": stat_internal,
            "line": float(line_score),
            "is_pitcher_prop": is_pitcher,
            "start_time": attrs.get("start_time", ""),
            "pp_game_id": game_id,
            "description": attrs.get("description", ""),
            "league": league_label,
            "line_type": line_type,
            "odds_type": odds_type or "standard",
            "model_eligible": model_eligible,
            "eligibility_reason": eligibility_reason,
            "is_game_prop": is_game_prop,
            "is_futures_prop": is_futures_prop,
            "league_id": league_id,
        })

    df = pd.DataFrame(rows)

    if not df.empty and "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df = df.sort_values(["start_time", "player_name", "stat_type"]).reset_index(drop=True)

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
