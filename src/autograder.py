"""
Auto-Grading Module
Automatically grades predictions by pulling actual results from MLB Stats API
box scores after games are final. Matches ungraded predictions to real outcomes
by player name and game date, then calls grade_prediction() to record results.

MLB Stats API endpoints used (free, no auth required):
- Schedule: statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD
- Box score: statsapi.mlb.com/api/v1/game/{game_pk}/boxscore

DraftKings Fantasy Score Weights:
  Single = 3, Double = 5, Triple = 8, Home Run = 10,
  RBI = 2, Run = 2, Walk/HBP = 2, Stolen Base = 5
"""

import re
import unicodedata
import logging
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd

from src.database import get_connection, get_ungraded_predictions, grade_prediction
from src.slips import grade_slip_pick

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15

# DraftKings fantasy scoring weights
DK_WEIGHTS = {
    "single": 3,
    "double": 5,
    "triple": 8,
    "home_run": 10,
    "rbi": 2,
    "run": 2,
    "walk": 2,   # BB + HBP both count
    "hbp": 2,
    "stolen_base": 5,
}

# Maps prediction stat_internal values to the box score field used for grading.
# Each value is the key in the player stats dict returned by extract_player_stats().
STAT_TYPE_MAP = {
    # Batting props
    "hitter_fantasy_score": "fantasy_score",
    "hits": "hits",
    "total_bases": "total_bases",
    "home_runs": "home_runs",
    "rbi": "rbi",
    "runs": "runs",
    "stolen_bases": "stolen_bases",
    "walks": "walks",
    "hits_runs_rbis": "hits_runs_rbis",
    "singles": "singles",
    "doubles": "doubles",
    "triples": "triples",
    "batter_strikeouts": "strikeouts",
    # Pitching props
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitching_outs": "pitching_outs",
    "earned_runs": "earned_runs",
    "walks_allowed": "walks_allowed",
}

# Stats that come from a pitcher's box score entry (not a batter's)
_PITCHING_STAT_INTERNALS = frozenset({
    "pitcher_strikeouts",
    "pitching_outs",
    "earned_runs",
    "walks_allowed",
})


# ---------------------------------------------------------------------------
# Name normalization for matching PrizePicks names to MLB Stats API names
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip accents, suffixes, punctuation and lowercase for fuzzy matching.

    Handles common variations between PrizePicks and MLB Stats API names:
    - Accent characters (e.g. Acuna -> Acuna)
    - Suffixes like Jr., Sr., II, III
    - Periods and hyphens in names
    - Extra whitespace

    Args:
        name: Raw player name string.

    Returns:
        Cleaned lowercase name suitable for comparison.
    """
    if not name:
        return ""
    # Decompose unicode and strip accent marks (combining characters)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    ascii_name = ascii_name.lower().strip()
    # Remove common suffixes
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name)
    # Remove periods, apostrophes, hyphens
    ascii_name = re.sub(r"[.\'\-]", "", ascii_name)
    # Collapse whitespace
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def _names_match(name_a: str, name_b: str) -> bool:
    """Check whether two player names refer to the same person.

    Tries exact normalized match first, then falls back to last-name +
    first-initial matching for cases where one source abbreviates differently.

    Args:
        name_a: First player name.
        name_b: Second player name.

    Returns:
        True if the names are considered a match.
    """
    norm_a = _normalize_name(name_a)
    norm_b = _normalize_name(name_b)

    # Exact normalized match
    if norm_a == norm_b:
        return True

    # Last name + first initial match
    parts_a = norm_a.split()
    parts_b = norm_b.split()
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        # Compare last names and first initials
        if parts_a[-1] == parts_b[-1] and parts_a[0][0] == parts_b[0][0]:
            return True

    return False


# ---------------------------------------------------------------------------
# MLB Stats API interaction
# ---------------------------------------------------------------------------

def fetch_schedule(game_date: str) -> list[dict]:
    """Fetch the MLB schedule for a given date.

    Args:
        game_date: Date string in YYYY-MM-DD format.

    Returns:
        List of dicts, each with keys:
            game_pk (int), status (str), home_team (str), away_team (str).
        Returns empty list on error.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "linescore",
    }
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("Failed to fetch schedule for %s: %s", game_date, exc)
        return []

    games = []
    for game_date_entry in data.get("dates", []):
        for game in game_date_entry.get("games", []):
            status_code = (
                game.get("status", {}).get("codedGameState", "")
            )
            games.append({
                "game_pk": game.get("gamePk"),
                "status": status_code,
                "status_detail": game.get("status", {}).get("detailedState", ""),
                "home_team": game.get("teams", {}).get("home", {}).get("team", {}).get("name", ""),
                "away_team": game.get("teams", {}).get("away", {}).get("team", {}).get("name", ""),
            })
    return games


def fetch_boxscore(game_pk: int) -> Optional[dict]:
    """Fetch the full box score for a single game.

    Args:
        game_pk: MLB game primary key identifier.

    Returns:
        Raw box score JSON dict, or None on error.
    """
    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("Failed to fetch boxscore for game %s: %s", game_pk, exc)
        return None


def _extract_batting_stats(player_data: dict, player_name: str) -> Optional[dict]:
    """Extract batting stats from a single player's box score entry.

    Args:
        player_data: The player dict from the boxscore JSON.
        player_name: Full player name.

    Returns:
        Dict with standardized stat keys, or None if the player did not bat.
    """
    batting = player_data.get("stats", {}).get("batting", {})

    # Skip players who did not appear in the game (no at-bats or walks)
    at_bats = batting.get("atBats", 0)
    walks = batting.get("baseOnBalls", 0)
    hbp = batting.get("hitByPitch", 0)
    if at_bats == 0 and walks == 0 and hbp == 0:
        return None

    hits = batting.get("hits", 0)
    doubles = batting.get("doubles", 0)
    triples = batting.get("triples", 0)
    home_runs = batting.get("homeRuns", 0)
    singles = hits - doubles - triples - home_runs
    rbi = batting.get("rbi", 0)
    runs = batting.get("runs", 0)
    stolen_bases = batting.get("stolenBases", 0)
    total_bases = batting.get("totalBases", 0)

    stats = {
        "player_name": player_name,
        "player_type": "batter",
        "hits": hits,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "rbi": rbi,
        "runs": runs,
        "stolen_bases": stolen_bases,
        "walks": walks,
        "hbp": hbp,
        "strikeouts": batting.get("strikeOuts", 0),
        "at_bats": at_bats,
        "total_bases": total_bases,
        "hits_runs_rbis": hits + runs + rbi,
    }
    stats["fantasy_score"] = calculate_fantasy_score(stats)
    return stats


def _extract_pitching_stats(player_data: dict, player_name: str) -> Optional[dict]:
    """Extract pitching stats from a single player's box score entry.

    Args:
        player_data: The player dict from the boxscore JSON.
        player_name: Full player name.

    Returns:
        Dict with standardized stat keys, or None if the player did not pitch.
    """
    pitching = player_data.get("stats", {}).get("pitching", {})
    innings = pitching.get("inningsPitched", "0")

    # Skip players who did not pitch
    try:
        ip_float = float(innings)
    except (ValueError, TypeError):
        ip_float = 0.0
    if ip_float == 0.0:
        return None

    # Convert MLB innings-pitched notation to total outs.
    # MLB uses fractional notation where ".1" = 1 out, ".2" = 2 outs (not decimals).
    # e.g. "6.2" means 6 complete innings + 2 extra outs = 20 total outs.
    try:
        ip_str = str(innings)
        parts = ip_str.split(".")
        full_innings = int(parts[0])
        extra_outs = int(parts[1]) if len(parts) > 1 else 0
        pitching_outs = full_innings * 3 + extra_outs
    except (ValueError, IndexError):
        pitching_outs = int(ip_float * 3)

    return {
        "player_name": player_name,
        "player_type": "pitcher",
        "pitcher_strikeouts": pitching.get("strikeOuts", 0),
        "innings_pitched": innings,
        "pitching_outs": pitching_outs,
        "hits_allowed": pitching.get("hits", 0),
        "runs_allowed": pitching.get("runs", 0),
        "earned_runs": pitching.get("earnedRuns", 0),
        "walks_allowed": pitching.get("baseOnBalls", 0),
        "home_runs_allowed": pitching.get("homeRuns", 0),
    }


def extract_player_stats(boxscore: dict) -> list[dict]:
    """Extract all player stats from a box score.

    Iterates over both home and away teams, pulling batting stats for
    every position player and pitching stats for every pitcher who appeared.

    Args:
        boxscore: Raw box score JSON dict from fetch_boxscore().

    Returns:
        List of player stat dicts. Each batter has batting fields plus
        fantasy_score; each pitcher has pitching fields.
    """
    all_stats: list[dict] = []

    for side in ("home", "away"):
        team_data = boxscore.get("teams", {}).get(side, {})
        players = team_data.get("players", {})

        for player_key, player_data in players.items():
            full_name = player_data.get("person", {}).get("fullName", "")
            if not full_name:
                continue

            # Batting stats
            bat_stats = _extract_batting_stats(player_data, full_name)
            if bat_stats:
                all_stats.append(bat_stats)

            # Pitching stats
            pitch_stats = _extract_pitching_stats(player_data, full_name)
            if pitch_stats:
                all_stats.append(pitch_stats)

    return all_stats


# ---------------------------------------------------------------------------
# Fantasy score calculation
# ---------------------------------------------------------------------------

def calculate_fantasy_score(stats: dict) -> float:
    """Calculate DraftKings fantasy score from box score batting stats.

    Uses the standard DK scoring weights:
        Single = 3, Double = 5, Triple = 8, HR = 10,
        RBI = 2, Run = 2, Walk/HBP = 2, Stolen Base = 5.

    Args:
        stats: Dict with keys: singles, doubles, triples, home_runs, rbi,
               runs, walks, hbp, stolen_bases.

    Returns:
        Float fantasy score.
    """
    score = 0.0
    score += stats.get("singles", 0) * DK_WEIGHTS["single"]
    score += stats.get("doubles", 0) * DK_WEIGHTS["double"]
    score += stats.get("triples", 0) * DK_WEIGHTS["triple"]
    score += stats.get("home_runs", 0) * DK_WEIGHTS["home_run"]
    score += stats.get("rbi", 0) * DK_WEIGHTS["rbi"]
    score += stats.get("runs", 0) * DK_WEIGHTS["run"]
    score += stats.get("walks", 0) * DK_WEIGHTS["walk"]
    score += stats.get("hbp", 0) * DK_WEIGHTS["hbp"]
    score += stats.get("stolen_bases", 0) * DK_WEIGHTS["stolen_base"]
    return score


# ---------------------------------------------------------------------------
# Prediction grading logic
# ---------------------------------------------------------------------------

def _find_matching_player_stats(
    player_name: str,
    stat_internal: str,
    player_stats_list: list[dict],
) -> Optional[dict]:
    """Find the box score stats entry that matches a prediction's player.

    Tries normalized name matching across all players in the day's box scores.
    For batting props, only matches batter entries; for pitching props, only
    matches pitcher entries.

    Args:
        player_name: Player name from the prediction row.
        stat_internal: Internal stat type string (e.g. "hitter_fantasy_score").
        player_stats_list: All player stats extracted from the day's box scores.

    Returns:
        The matching stats dict, or None if no match found.
    """
    is_pitching_stat = stat_internal in _PITCHING_STAT_INTERNALS
    target_type = "pitcher" if is_pitching_stat else "batter"

    for ps in player_stats_list:
        if ps.get("player_type") != target_type:
            continue
        if _names_match(player_name, ps.get("player_name", "")):
            return ps
    return None


def auto_grade_prediction(
    pred_row: pd.Series,
    player_stats_list: list[dict],
) -> Optional[str]:
    """Grade a single prediction against actual box score results.

    Looks up the player's actual stats, determines the relevant actual value
    based on the prop type, and calls grade_prediction() from the database
    module to record the result.

    Args:
        pred_row: A row from the predictions DataFrame (must have id,
                  player_name, stat_internal, line, pick).
        player_stats_list: All player stats for the game date.

    Returns:
        "W", "L", "push" if graded successfully, or None if the player
        could not be matched or the stat type is unsupported.
    """
    pred_id = int(pred_row["id"])
    player_name = pred_row["player_name"]
    stat_internal = pred_row.get("stat_internal", "")

    # Find the matching player in box score data
    matched = _find_matching_player_stats(
        player_name, stat_internal, player_stats_list
    )
    if matched is None:
        logger.warning(
            "No box score match for '%s' (stat: %s, pred_id: %d)",
            player_name, stat_internal, pred_id,
        )
        return None

    # Determine which actual value to use for grading
    stat_key = STAT_TYPE_MAP.get(stat_internal)
    if stat_key is None:
        logger.warning(
            "Unsupported stat type '%s' for auto-grading (pred_id: %d)",
            stat_internal, pred_id,
        )
        return None

    actual_value = matched.get(stat_key)
    if actual_value is None:
        logger.warning(
            "Stat key '%s' not found in box score for '%s' (pred_id: %d)",
            stat_key, player_name, pred_id,
        )
        return None

    actual_value = float(actual_value)

    # Call the existing database grading function
    result = grade_prediction(pred_id, actual_value)
    return result


# ---------------------------------------------------------------------------
# Slip auto-grading helpers
# ---------------------------------------------------------------------------

def _grade_linked_slip_picks(pred_id: int, actual_value: float) -> int:
    """Grade any slip picks that reference the given prediction ID.

    When a prediction is auto-graded we also want to grade the corresponding
    slip picks so PrizePicks slip P&L is kept up to date without manual work.

    Args:
        pred_id: The predictions.id that was just graded.
        actual_value: The actual numeric result used for grading.

    Returns:
        Number of slip picks graded.
    """
    try:
        conn = get_connection()
        cur = conn.execute(
            "SELECT id FROM slip_picks WHERE prediction_id = ? AND result IS NULL",
            (pred_id,),
        )
        slip_pick_ids = [row[0] for row in cur.fetchall()]
        conn.close()
    except Exception as exc:
        logger.warning("Could not query slip_picks for pred_id %d: %s", pred_id, exc)
        return 0

    graded_count = 0
    for sp_id in slip_pick_ids:
        try:
            grade_slip_pick(sp_id, actual_value)
            graded_count += 1
        except Exception as exc:
            logger.warning("Failed to grade slip_pick %d: %s", sp_id, exc)

    return graded_count


# ---------------------------------------------------------------------------
# Main auto-grade orchestration
# ---------------------------------------------------------------------------

def auto_grade_date(game_date: str) -> dict:
    """Grade all ungraded predictions for a specific date.

    This is the main entry point. It:
    1. Fetches the MLB schedule for the date
    2. Checks which games are final
    3. Pulls box scores for final games
    4. Extracts player stats from all box scores
    5. Matches ungraded predictions to actual results
    6. Grades each matched prediction

    Args:
        game_date: Date string in YYYY-MM-DD format.

    Returns:
        Dict with keys:
            graded (int): Number of predictions successfully graded.
            not_matched (int): Number of predictions that could not be matched.
            skipped_not_final (int): Predictions skipped because their game
                                     is not yet final.
            results (list[dict]): Details of each graded prediction.
            errors (list[str]): Any error messages encountered.
    """
    report = {
        "graded": 0,
        "not_matched": 0,
        "skipped_not_final": 0,
        "slip_picks_graded": 0,
        "results": [],
        "errors": [],
    }

    # Get ungraded predictions for this date
    ungraded = get_ungraded_predictions(game_date=game_date)
    if ungraded.empty:
        report["errors"].append(f"No ungraded predictions found for {game_date}.")
        return report

    # Fetch schedule
    schedule = fetch_schedule(game_date)
    if not schedule:
        report["errors"].append(f"Could not fetch MLB schedule for {game_date}.")
        return report

    # Filter to final games only (codedGameState "F" = Final)
    final_games = [g for g in schedule if g["status"] in ("F", "D")]
    if not final_games:
        non_final_count = len(ungraded)
        report["skipped_not_final"] = non_final_count
        report["errors"].append(
            f"No final games found for {game_date}. "
            f"{non_final_count} prediction(s) remain ungraded."
        )
        return report

    # Pull box scores and extract player stats from all final games
    all_player_stats: list[dict] = []
    for game in final_games:
        boxscore = fetch_boxscore(game["game_pk"])
        if boxscore is None:
            report["errors"].append(
                f"Failed to fetch boxscore for game {game['game_pk']} "
                f"({game['away_team']} @ {game['home_team']})."
            )
            continue
        game_stats = extract_player_stats(boxscore)
        all_player_stats.extend(game_stats)

    if not all_player_stats:
        report["errors"].append(
            f"No player stats extracted from box scores for {game_date}."
        )
        return report

    # Grade each ungraded prediction
    for _, pred_row in ungraded.iterrows():
        try:
            result = auto_grade_prediction(pred_row, all_player_stats)
            if result is not None:
                report["graded"] += 1
                # Determine actual value for the report
                stat_key = STAT_TYPE_MAP.get(pred_row.get("stat_internal", ""), "")
                matched = _find_matching_player_stats(
                    pred_row["player_name"],
                    pred_row.get("stat_internal", ""),
                    all_player_stats,
                )
                actual = matched.get(stat_key, "?") if matched else "?"
                report["results"].append({
                    "pred_id": int(pred_row["id"]),
                    "player": pred_row["player_name"],
                    "stat_type": pred_row.get("stat_type", ""),
                    "line": pred_row.get("line", 0),
                    "pick": pred_row.get("pick", ""),
                    "actual": actual,
                    "result": result,
                })
                # Also grade any slip picks linked to this prediction
                if isinstance(actual, (int, float)):
                    slip_graded = _grade_linked_slip_picks(int(pred_row["id"]), float(actual))
                    report["slip_picks_graded"] += slip_graded
            else:
                report["not_matched"] += 1
        except Exception as exc:
            report["errors"].append(
                f"Error grading pred {pred_row.get('id', '?')} "
                f"({pred_row.get('player_name', '?')}): {exc}"
            )

    return report


def auto_grade_yesterday() -> dict:
    """Convenience function: auto-grade all predictions from yesterday.

    Returns:
        Same dict format as auto_grade_date().
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return auto_grade_date(yesterday)


def auto_grade_all_pending() -> dict:
    """Grade all ungraded predictions across all dates.

    Finds every distinct game_date with ungraded predictions and runs
    auto_grade_date() for each one.

    Returns:
        Dict with keys:
            dates_processed (list[str]): Dates that were processed.
            total_graded (int): Total predictions graded across all dates.
            total_not_matched (int): Total predictions not matched.
            per_date (dict[str, dict]): Individual auto_grade_date() results
                                        keyed by date string.
    """
    all_ungraded = get_ungraded_predictions()
    if all_ungraded.empty:
        return {
            "dates_processed": [],
            "total_graded": 0,
            "total_not_matched": 0,
            "per_date": {},
        }

    dates = sorted(all_ungraded["game_date"].unique())
    combined = {
        "dates_processed": dates,
        "total_graded": 0,
        "total_not_matched": 0,
        "per_date": {},
    }

    for gd in dates:
        result = auto_grade_date(gd)
        combined["per_date"][gd] = result
        combined["total_graded"] += result["graded"]
        combined["total_not_matched"] += result["not_matched"]

    return combined
