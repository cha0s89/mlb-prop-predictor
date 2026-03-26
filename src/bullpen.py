"""
Bullpen Fatigue & Starter Workload Module

Tracks bullpen usage (recent IP/pitches) and starter pitch count patterns
to adjust pitcher and batter projections. Key signals:

1. BULLPEN FATIGUE: A taxed bullpen means starters may go deeper (more K
   opportunity) or the team may use an opener. Also affects opposing batter
   run/RBI projections when facing a gassed bullpen.

2. STARTER PITCH COUNT/INNINGS LIMIT: Detects if a pitcher is likely on a
   pitch count or innings limit based on:
   - Recent IL activation (returning from injury)
   - Season IP pace vs career norms
   - Early-season ramp-up (first 3-4 weeks)
   - Recent start patterns (pitches per outing trending down)

3. STARTER RECENT FORM: Last 3 starts pitches/IP gives a real-time workload
   signal that complements season-long averages.

Data source: MLB Stats API (free, no auth required)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10


def _api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """Safe GET to MLB Stats API. Returns JSON or None on failure."""
    url = f"{MLB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug("MLB API error (%s): %s", endpoint, e)
        return None


# ─────────────────────────────────────────────
# BULLPEN FATIGUE
# ─────────────────────────────────────────────

def fetch_bullpen_usage(team_abbr: str, days_back: int = 3) -> dict:
    """
    Fetch recent bullpen usage for a team over the last N days.

    Uses the MLB Stats API schedule + boxscore endpoints to aggregate
    reliever innings and pitch counts.

    Returns:
        Dict with:
            total_ip: float — total bullpen IP in window
            total_pitches: int — total bullpen pitches
            games_played: int — games in window
            avg_ip_per_game: float
            relievers_used: dict — name -> {ip, pitches, appearances}
            fatigue_level: str — "fresh" | "moderate" | "taxed" | "gassed"
            fatigue_multiplier: float — 0.95-1.05 adjustment
    """
    result = {
        "total_ip": 0.0,
        "total_pitches": 0,
        "games_played": 0,
        "avg_ip_per_game": 0.0,
        "relievers_used": {},
        "fatigue_level": "fresh",
        "fatigue_multiplier": 1.0,
        "has_data": False,
    }

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Fetch recent schedule
    schedule = _api_get("/schedule", params={
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "teamId": _team_id(team_abbr),
    })

    if not schedule or "dates" not in schedule:
        return result

    game_pks = []
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState") == "Final":
                game_pks.append(game["gamePk"])

    if not game_pks:
        return result

    total_bp_ip = 0.0
    total_bp_pitches = 0
    relievers = {}

    for game_pk in game_pks[-5:]:  # Last 5 games max
        boxscore = _api_get(f"/game/{game_pk}/boxscore")
        if not boxscore:
            continue

        # Find this team's side
        for side in ["home", "away"]:
            team_data = boxscore.get("teams", {}).get(side, {})
            team_info = team_data.get("team", {})
            abbr = team_info.get("abbreviation", "")

            if abbr.upper() != team_abbr.upper():
                continue

            pitchers = team_data.get("pitchers", [])
            players = team_data.get("players", {})

            # First pitcher is the starter, rest are bullpen
            for i, pitcher_id in enumerate(pitchers):
                if i == 0:
                    continue  # Skip starter

                key = f"ID{pitcher_id}"
                player_data = players.get(key, {})
                stats = player_data.get("stats", {}).get("pitching", {})
                name = player_data.get("person", {}).get("fullName", f"P{pitcher_id}")

                ip_str = stats.get("inningsPitched", "0")
                try:
                    ip = float(ip_str)
                except (ValueError, TypeError):
                    ip = 0.0

                pitches = int(stats.get("numberOfPitches", 0))

                total_bp_ip += ip
                total_bp_pitches += pitches

                if name not in relievers:
                    relievers[name] = {"ip": 0.0, "pitches": 0, "appearances": 0}
                relievers[name]["ip"] += ip
                relievers[name]["pitches"] += pitches
                relievers[name]["appearances"] += 1

    result["total_ip"] = round(total_bp_ip, 1)
    result["total_pitches"] = total_bp_pitches
    result["games_played"] = len(game_pks)
    result["avg_ip_per_game"] = round(total_bp_ip / max(len(game_pks), 1), 1)
    result["relievers_used"] = relievers
    result["has_data"] = len(game_pks) > 0

    # Classify fatigue level
    # Normal bullpen uses ~3-4 IP per game. 5+ means taxed.
    avg_bp_ip = total_bp_ip / max(len(game_pks), 1)
    if avg_bp_ip >= 5.5 or total_bp_ip >= 15:
        result["fatigue_level"] = "gassed"
        result["fatigue_multiplier"] = 1.04  # Opposing batters benefit
    elif avg_bp_ip >= 4.5 or total_bp_ip >= 12:
        result["fatigue_level"] = "taxed"
        result["fatigue_multiplier"] = 1.02
    elif avg_bp_ip >= 3.5:
        result["fatigue_level"] = "moderate"
        result["fatigue_multiplier"] = 1.0
    else:
        result["fatigue_level"] = "fresh"
        result["fatigue_multiplier"] = 1.0

    return result


# ─────────────────────────────────────────────
# STARTER WORKLOAD / PITCH COUNT DETECTION
# ─────────────────────────────────────────────

def fetch_starter_recent_starts(pitcher_name: str, pitcher_id: int = None,
                                 n_starts: int = 3) -> dict:
    """
    Fetch a starter's recent game logs to detect workload patterns.

    Returns:
        Dict with:
            starts: list of {date, ip, pitches, er, k, decision}
            avg_pitches: float — average pitches per start
            avg_ip: float — average IP per start
            trend: str — "ramping_up" | "steady" | "winding_down"
            likely_pitch_limit: int | None — estimated pitch count cap
            workload_multiplier: float — discount for likely shortened outing
    """
    result = {
        "starts": [],
        "avg_pitches": 0.0,
        "avg_ip": 0.0,
        "trend": "steady",
        "likely_pitch_limit": None,
        "workload_multiplier": 1.0,
        "has_data": False,
    }

    if not pitcher_id:
        return result

    # Fetch game log for current season
    season = datetime.now().year
    log_data = _api_get(f"/people/{pitcher_id}/stats", params={
        "stats": "gameLog",
        "season": season,
        "group": "pitching",
    })

    if not log_data:
        return result

    splits = []
    for stat_group in log_data.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            if stat.get("gamesStarted", 0) > 0:
                splits.append({
                    "date": split.get("date", ""),
                    "ip": float(stat.get("inningsPitched", 0)),
                    "pitches": int(stat.get("numberOfPitches", 0)),
                    "er": int(stat.get("earnedRuns", 0)),
                    "k": int(stat.get("strikeOuts", 0)),
                    "decision": stat.get("decision", ""),
                })

    if not splits:
        return result

    # Take the most recent N starts
    recent = splits[-n_starts:]
    result["starts"] = recent
    result["has_data"] = True

    pitches = [s["pitches"] for s in recent if s["pitches"] > 0]
    ips = [s["ip"] for s in recent if s["ip"] > 0]

    if pitches:
        result["avg_pitches"] = round(sum(pitches) / len(pitches), 0)
    if ips:
        result["avg_ip"] = round(sum(ips) / len(ips), 1)

    # Detect pitch count trend
    if len(pitches) >= 2:
        if pitches[-1] > pitches[0] + 10:
            result["trend"] = "ramping_up"
        elif pitches[-1] < pitches[0] - 10:
            result["trend"] = "winding_down"

    # Estimate pitch limit
    if result["avg_pitches"] > 0:
        if result["avg_pitches"] <= 70:
            result["likely_pitch_limit"] = 75
            result["workload_multiplier"] = 0.82  # Short outing expected
        elif result["avg_pitches"] <= 80:
            result["likely_pitch_limit"] = 85
            result["workload_multiplier"] = 0.90
        elif result["avg_pitches"] <= 90:
            result["likely_pitch_limit"] = 95
            result["workload_multiplier"] = 0.95

    return result


def estimate_pitcher_workload_discount(
    pitcher_profile: dict,
    injury_status: dict = None,
    recent_starts: dict = None,
    game_date=None,
) -> float:
    """
    Combine multiple workload signals into a single IP/K discount multiplier.

    Signals:
    1. IL return: pitcher recently activated from IL → likely pitch count
    2. Recent start pattern: avg pitches trending low
    3. Early season: Opening Day ramp-up (already in predictor, but reinforced here)

    Returns:
        float: Multiplier (0.75-1.0) to apply to pitcher projection
    """
    discount = 1.0

    # IL return detection
    if injury_status and injury_status.get("status") == "active":
        desc = injury_status.get("description", "").lower()
        since = injury_status.get("since", "")
        if "activat" in desc or "reinstat" in desc:
            # Recently activated — likely on pitch count for 2-3 starts
            if since:
                try:
                    activated_date = datetime.strptime(since[:10], "%Y-%m-%d").date()
                    if game_date:
                        if isinstance(game_date, str):
                            game_date = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
                        days_since_activation = (game_date - activated_date).days
                        if days_since_activation <= 14:
                            discount *= 0.85  # First 2 weeks back: ~15% IP discount
                            logger.debug("%s recently activated (%d days ago), applying 0.85 discount",
                                        pitcher_profile.get("name", ""), days_since_activation)
                        elif days_since_activation <= 28:
                            discount *= 0.92  # Weeks 3-4: ~8% discount
                except (ValueError, TypeError):
                    pass

    # Recent start workload pattern
    if recent_starts and recent_starts.get("has_data"):
        wl_mult = recent_starts.get("workload_multiplier", 1.0)
        if wl_mult < 1.0:
            # Don't double-count with IL discount — take the more aggressive one
            discount = min(discount, wl_mult)

    return round(max(discount, 0.75), 3)  # Floor at 25% discount


# ─────────────────────────────────────────────
# TEAM ID LOOKUP
# ─────────────────────────────────────────────

def _team_id(abbr: str) -> int:
    """Convert team abbreviation to MLB Stats API team ID."""
    try:
        from src.teams import team_id
        return team_id(abbr)
    except ImportError:
        return _FALLBACK_IDS.get(abbr.upper(), 0)


_FALLBACK_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SF": 137, "SEA": 136,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
    # Aliases
    "AZ": 109, "CHW": 145, "KCR": 118, "SDP": 135, "SFG": 137,
    "TBR": 139, "WAS": 120, "ATH": 133,
}


def _team_id(abbr: str) -> int:
    """Convert team abbreviation to MLB Stats API team ID."""
    return _TEAM_IDS.get(abbr.upper(), 0)
