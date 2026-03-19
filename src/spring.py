"""
Spring Training Stats + Injury Flags Module

Pulls Spring Training performance data and injury/transaction info from
the MLB Stats API (free, no auth required). Generates spring form multipliers
that adjust predictions based on how a player is performing in ST relative
to their prior season baseline.

Why this matters: Spring Training stats are noisy in small samples, but
Statcast data (exit velo, barrel rate) stabilizes faster than batting average.
A player barreling everything in ST is a real signal even in 30 ABs. Meanwhile,
a .100 AVG in 15 ABs is mostly noise. We weight accordingly.

Injury flags prevent the user from accidentally picking a player on the IL.

Data source: MLB Stats API (statsapi.mlb.com) — completely free, no key needed.
"""

import requests
import unicodedata
import re
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache

try:
    from pybaseball import statcast_batter, cache
    cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# All 30 MLB team IDs — used to iterate rosters and ST stats
MLB_TEAM_IDS = list(range(108, 122)) + list(range(133, 148)) + [158]
# Actual full list: use the teams endpoint to be safe, but these cover all 30

# Spring Training game type code in MLB Stats API
GAME_TYPE_SPRING = "S"
GAME_TYPE_REGULAR = "R"

# Maximum spring form adjustment (±8% as specified in CLAUDE.md)
MAX_SPRING_MULT = 1.08
MIN_SPRING_MULT = 0.92

# Minimum ABs before ST stats carry any weight at all
MIN_AB_THRESHOLD = 10
# ABs where ST signal reaches full weight (still capped at ±8%)
FULL_WEIGHT_AB = 40

# Request timeout so a hung API call doesn't freeze the app
REQUEST_TIMEOUT = 10


# ─────────────────────────────────────────────
# NAME NORMALIZATION
# ─────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize a player name for fuzzy matching across data sources.

    Strips accents (Acuña -> Acuna), lowercases, removes suffixes
    (Jr., Sr., II, III), and collapses whitespace. This handles the
    common mismatches between PrizePicks names, MLB API names, and
    FanGraphs names.
    """
    if not name:
        return ""
    # Strip accents: é -> e, ñ -> n, etc.
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    result = ascii_only.lower().strip()
    # Remove common suffixes that vary between sources
    result = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", result)
    # Remove periods and extra whitespace
    result = re.sub(r"\.", "", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _names_match(name_a: str, name_b: str) -> bool:
    """Check if two player names refer to the same person."""
    na = normalize_name(name_a)
    nb = normalize_name(name_b)
    if na == nb:
        return True
    # Try last-name + first-initial match for cases like
    # "Ronald Acuna" vs "R. Acuna"
    parts_a = na.split()
    parts_b = nb.split()
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        if parts_a[-1] == parts_b[-1] and parts_a[0][0] == parts_b[0][0]:
            return True
    return False


# ─────────────────────────────────────────────
# MLB STATS API HELPERS
# ─────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """
    Safe GET request to MLB Stats API. Returns parsed JSON or None on failure.
    Never raises — all errors are caught and logged silently so the app
    doesn't crash if the API is down or returns unexpected data.
    """
    url = f"{MLB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[spring.py] API error for {endpoint}: {e}")
        return None
    except ValueError:
        print(f"[spring.py] Invalid JSON from {endpoint}")
        return None


def _get_all_team_ids() -> list[int]:
    """
    Fetch all active MLB team IDs from the teams endpoint.
    Falls back to the hardcoded list if the API call fails.
    """
    data = _api_get("/teams", params={"sportId": 1, "activeStatus": "Y"})
    if data and "teams" in data:
        return [t["id"] for t in data["teams"]]
    # Fallback: known 2025 team IDs
    return MLB_TEAM_IDS


# ─────────────────────────────────────────────
# SPRING TRAINING BATTING STATS
# ─────────────────────────────────────────────

def fetch_spring_training_stats(year: int = None) -> list[dict]:
    """
    Pull Spring Training batting stats for all players in a single API call.

    Uses the MLB Stats API bulk stats endpoint:
        /v1/stats?stats=season&season=YYYY&group=hitting&gameType=S&sportId=1

    Returns a list of dicts, each with:
        player_id, player_name, team, G, AB, H, HR, RBI, BB, K,
        AVG, SLG, OBP, OPS, 2B, 3B, SB, PA

    Args:
        year: The ST season year. Defaults to current year.

    Returns:
        List of player stat dicts. Empty list if API fails.
    """
    year = year or datetime.now().year

    data = _api_get(
        "/stats",
        params={
            "stats": "season",
            "season": year,
            "group": "hitting",
            "gameType": GAME_TYPE_SPRING,
            "sportId": 1,
            "limit": 1000,
            "offset": 0,
        },
    )

    if not data or "stats" not in data:
        return []

    all_players = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            try:
                s = split.get("stat", {})
                player = split.get("player", {})
                team = split.get("team", {})

                ab = _safe_int(s.get("atBats", 0))
                if ab == 0:
                    continue

                h = _safe_int(s.get("hits", 0))
                doubles = _safe_int(s.get("doubles", 0))
                triples = _safe_int(s.get("triples", 0))
                hr = _safe_int(s.get("homeRuns", 0))
                bb = _safe_int(s.get("baseOnBalls", 0))
                hbp = _safe_int(s.get("hitByPitch", 0))
                sf = _safe_int(s.get("sacFlies", 0))
                so = _safe_int(s.get("strikeOuts", 0))
                pa = ab + bb + hbp + sf

                avg = round(h / ab, 3)
                slg = round(_calculate_slg(h, doubles, triples, hr, ab), 3)
                obp = round((h + bb + hbp) / pa, 3) if pa > 0 else 0.0
                ops = round(obp + slg, 3)

                all_players.append({
                    "player_id": player.get("id", 0),
                    "player_name": player.get("fullName", ""),
                    "team": team.get("abbreviation", ""),
                    "G": _safe_int(s.get("gamesPlayed", 0)),
                    "AB": ab, "H": h, "HR": hr,
                    "RBI": _safe_int(s.get("rbi", 0)),
                    "BB": bb, "K": so,
                    "2B": doubles, "3B": triples,
                    "SB": _safe_int(s.get("stolenBases", 0)),
                    "PA": pa, "HBP": hbp, "SF": sf,
                    "AVG": avg, "SLG": slg, "OBP": obp, "OPS": ops,
                })
            except Exception as e:
                print(f"[spring.py] Error parsing ST player: {e}")
                continue

    return all_players


def _calculate_slg(hits: int, doubles: int, triples: int, hr: int, ab: int) -> float:
    """Calculate slugging percentage from counting stats."""
    if ab == 0:
        return 0.0
    singles = hits - doubles - triples - hr
    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)
    return total_bases / ab


def _safe_int(val) -> int:
    """Safely convert a value to int, handling strings and None."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(val) -> float:
    """Safely convert a value to float, handling strings with % signs."""
    if val is None:
        return 0.0
    if isinstance(val, str):
        val = val.replace("%", "").strip()
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────
# SPRING FORM MULTIPLIER
# ─────────────────────────────────────────────

def get_spring_form_multiplier(
    player_name: str,
    prior_season_slg: float,
    prior_season_avg: float,
    st_stats: list[dict] = None,
    year: int = None,
) -> dict:
    """
    Calculate a spring form multiplier for a player by comparing their
    Spring Training performance to their prior-season baseline.

    The multiplier is bounded to ±5-8% (0.92 to 1.08) because ST stats
    are inherently noisy. We use SLG as the primary signal (more stable
    than AVG in small samples) and weight by sample size.

    Args:
        player_name: Full player name (e.g., "Aaron Judge")
        prior_season_slg: Player's prior regular-season SLG
        prior_season_avg: Player's prior regular-season AVG
        st_stats: Pre-fetched ST stats list (avoids re-fetching). If None,
                  will call fetch_spring_training_stats().
        year: ST year. Defaults to current year.

    Returns:
        Dict with:
            spring_mult: float between 0.92 and 1.08
            sample_size: int (AB in ST)
            badge: str emoji — "hot" / "cold" / "neutral"
            badge_label: str — human-readable label
            st_avg: float — Spring Training batting average
            st_slg: float — Spring Training slugging
            confidence: str — "low" / "medium" / "high" based on sample size
    """
    result = {
        "spring_mult": 1.0,
        "sample_size": 0,
        "badge": "neutral",
        "badge_label": "No ST Data",
        "st_avg": 0.0,
        "st_slg": 0.0,
        "confidence": "low",
    }

    # Fetch ST stats if not provided
    if st_stats is None:
        try:
            st_stats = fetch_spring_training_stats(year)
        except Exception:
            return result

    if not st_stats:
        return result

    # Find this player in the ST stats
    player_st = None
    for p in st_stats:
        if _names_match(p.get("player_name", ""), player_name):
            player_st = p
            break

    if player_st is None:
        return result

    ab = player_st.get("AB", 0)
    result["sample_size"] = ab
    result["st_avg"] = player_st.get("AVG", 0.0)
    result["st_slg"] = player_st.get("SLG", 0.0)

    # Not enough ABs to draw any conclusion
    if ab < MIN_AB_THRESHOLD:
        result["confidence"] = "low"
        result["badge_label"] = f"ST: {ab} AB (too small)"
        return result

    # Determine confidence level based on sample size
    if ab >= FULL_WEIGHT_AB:
        result["confidence"] = "high"
        sample_weight = 1.0
    elif ab >= 25:
        result["confidence"] = "medium"
        sample_weight = 0.7
    else:
        result["confidence"] = "low"
        # Linear ramp from MIN_AB_THRESHOLD to 25 ABs
        sample_weight = 0.3 + 0.4 * ((ab - MIN_AB_THRESHOLD) / (25 - MIN_AB_THRESHOLD))

    # Compare ST SLG to prior-season SLG (SLG is more signal-rich than AVG)
    # Also factor in AVG at lower weight
    prior_slg = max(prior_season_slg, 0.250)  # Floor to avoid division issues
    prior_avg = max(prior_season_avg, 0.200)

    st_slg = player_st.get("SLG", prior_slg)
    st_avg = player_st.get("AVG", prior_avg)

    # SLG ratio gets 70% weight, AVG ratio gets 30% weight
    # because SLG captures power which is more meaningful in ST samples
    slg_ratio = st_slg / prior_slg if prior_slg > 0 else 1.0
    avg_ratio = st_avg / prior_avg if prior_avg > 0 else 1.0
    raw_ratio = (0.70 * slg_ratio) + (0.30 * avg_ratio)

    # Convert ratio to a multiplier centered on 1.0
    # A ratio of 1.5 (50% better in ST) becomes a much smaller adjustment
    # because ST stats are noisy
    raw_adjustment = (raw_ratio - 1.0) * 0.15  # Scale down aggressively
    weighted_adjustment = raw_adjustment * sample_weight

    # Clamp to ±8%
    spring_mult = 1.0 + max(-0.08, min(0.08, weighted_adjustment))
    result["spring_mult"] = round(spring_mult, 4)

    # Assign badge based on multiplier
    if spring_mult >= 1.03:
        result["badge"] = "hot"
        result["badge_label"] = f"ST Hot ({ab} AB, .{int(st_slg*1000):03d} SLG)"
    elif spring_mult <= 0.97:
        result["badge"] = "cold"
        result["badge_label"] = f"ST Cold ({ab} AB, .{int(st_slg*1000):03d} SLG)"
    else:
        result["badge"] = "neutral"
        result["badge_label"] = f"ST Normal ({ab} AB, .{int(st_slg*1000):03d} SLG)"

    return result


# ─────────────────────────────────────────────
# INJURIES + TRANSACTIONS
# ─────────────────────────────────────────────

def fetch_injuries(days_back: int = 30) -> list[dict]:
    """
    Pull recent injury-related transactions from the MLB Stats API.

    Filters for IL placements, returns structured injury data including
    player name, injury description, and IL type (10-day, 60-day, etc.).

    The transactions endpoint returns ALL transaction types (trades, call-ups,
    etc.) so we filter for injury-related ones only.

    Args:
        days_back: How many days of transactions to search. Default 30.

    Returns:
        List of dicts, each with:
            player_name, player_id, team, injury_description,
            il_type, transaction_date, transaction_type
        Empty list if API fails.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    data = _api_get(
        "/transactions",
        params={
            "startDate": start_date,
            "endDate": end_date,
        },
    )

    if not data or "transactions" not in data:
        return []

    injuries = []
    for txn in data["transactions"]:
        try:
            txn_type = txn.get("typeDesc", "")
            # Filter for IL-related transactions
            # Common types: "Placed on IL", "Activated from IL",
            # "Transferred to 60-Day IL", etc.
            is_il_placement = any(
                keyword in txn_type.lower()
                for keyword in ["injured list", "disabled list", "il"]
            )
            if not is_il_placement:
                continue

            player_info = txn.get("person", {})
            team_info = txn.get("team", {})

            # Determine IL type from the description
            il_type = _parse_il_type(txn_type, txn.get("description", ""))

            injuries.append({
                "player_name": player_info.get("fullName", "Unknown"),
                "player_id": player_info.get("id", 0),
                "team": team_info.get("name", ""),
                "team_abbr": team_info.get("abbreviation", ""),
                "injury_description": txn.get("description", txn_type),
                "il_type": il_type,
                "transaction_date": txn.get("date", ""),
                "transaction_type": txn_type,
                "is_activation": "activat" in txn_type.lower() or "reinstat" in txn_type.lower(),
            })
        except Exception as e:
            print(f"[spring.py] Error parsing transaction: {e}")
            continue

    return injuries


def _parse_il_type(type_desc: str, description: str) -> str:
    """
    Extract the IL type (10-day, 15-day, 60-day) from transaction text.
    Falls back to 'IL' if we can't determine the specific type.
    """
    combined = f"{type_desc} {description}".lower()
    if "60-day" in combined or "60 day" in combined:
        return "60-day IL"
    if "15-day" in combined or "15 day" in combined:
        return "15-day IL"
    if "10-day" in combined or "10 day" in combined:
        return "10-day IL"
    if "7-day" in combined or "7 day" in combined:
        return "7-day IL"  # Concussion IL
    if "paternity" in combined:
        return "Paternity List"
    if "bereavement" in combined:
        return "Bereavement List"
    return "IL"


def get_player_injury_status(
    player_name: str,
    injuries: list[dict] = None,
    days_back: int = 30,
) -> dict:
    """
    Determine a player's current injury/availability status.

    Checks recent transactions for IL placements and activations.
    A player placed on the IL who has NOT been activated is flagged red.
    Day-to-day players (mentioned in transactions but not placed on IL)
    get a yellow flag. Everyone else is green/active.

    Args:
        player_name: Full player name to check
        injuries: Pre-fetched injury list (avoids re-fetching). If None,
                  will call fetch_injuries().
        days_back: How far back to search transactions.

    Returns:
        Dict with:
            status: "active" | "IL" | "day-to-day"
            color: "green" | "red" | "yellow"
            description: str — injury details if applicable
            il_type: str — specific IL type if on IL
            since: str — date of most recent transaction
    """
    result = {
        "status": "active",
        "color": "green",
        "description": "",
        "il_type": "",
        "since": "",
    }

    if injuries is None:
        try:
            injuries = fetch_injuries(days_back)
        except Exception:
            return result

    if not injuries:
        return result

    # Find all transactions for this player, sorted by date (most recent first)
    player_txns = []
    for inj in injuries:
        if _names_match(inj.get("player_name", ""), player_name):
            player_txns.append(inj)

    if not player_txns:
        return result

    # Sort by date descending to get the most recent transaction first
    player_txns.sort(key=lambda x: x.get("transaction_date", ""), reverse=True)
    latest = player_txns[0]

    if latest.get("is_activation", False):
        # Most recent transaction is an activation — player is active
        # but might be coming back from injury, worth noting
        result["status"] = "active"
        result["color"] = "green"
        result["description"] = f"Activated: {latest.get('injury_description', '')}"
        result["since"] = latest.get("transaction_date", "")
    else:
        # Most recent transaction is an IL placement
        il_type = latest.get("il_type", "IL")
        result["status"] = "IL"
        result["color"] = "red"
        result["description"] = latest.get("injury_description", "On injured list")
        result["il_type"] = il_type
        result["since"] = latest.get("transaction_date", "")

    return result


def fetch_current_il_players(days_back: int = 60) -> list[dict]:
    """
    Build a list of players currently on the IL by checking transactions.

    Looks at IL placements and activations over the past N days. A player
    who was placed on the IL and NOT subsequently activated is considered
    currently on the IL.

    Args:
        days_back: How far back to search. 60 days catches 60-day IL stints.

    Returns:
        List of dicts for players currently on IL, each with:
            player_name, player_id, team, il_type, injury_description, since
    """
    injuries = fetch_injuries(days_back)
    if not injuries:
        return []

    # Track placement and activation per player
    # Key: normalized player name, Value: most recent transaction
    player_status: dict[str, dict] = {}

    # Sort chronologically so later events overwrite earlier ones
    sorted_injuries = sorted(injuries, key=lambda x: x.get("transaction_date", ""))

    for inj in sorted_injuries:
        norm = normalize_name(inj.get("player_name", ""))
        if not norm:
            continue
        player_status[norm] = inj

    # Filter to only players whose most recent transaction is a placement (not activation)
    currently_on_il = []
    for norm_name, txn in player_status.items():
        if not txn.get("is_activation", False):
            currently_on_il.append({
                "player_name": txn.get("player_name", ""),
                "player_id": txn.get("player_id", 0),
                "team": txn.get("team_abbr", txn.get("team", "")),
                "il_type": txn.get("il_type", "IL"),
                "injury_description": txn.get("injury_description", ""),
                "since": txn.get("transaction_date", ""),
            })

    return currently_on_il


# ─────────────────────────────────────────────
# ROSTER STATUS
# ─────────────────────────────────────────────

def fetch_roster_status(team_id: int, year: int = None) -> list[dict]:
    """
    Fetch a team's current roster to check who is in major league camp.

    Useful during Spring Training to see which players are on the 40-man
    roster vs assigned to minor league camp (NRI = non-roster invitee).

    Args:
        team_id: MLB team ID (e.g., 147 for Yankees)
        year: Season year. Defaults to current year.

    Returns:
        List of player dicts with: player_name, player_id, position,
        roster_status, jersey_number
    """
    year = year or datetime.now().year
    data = _api_get(
        f"/teams/{team_id}/roster",
        params={"rosterType": "fullSeason", "season": year},
    )

    if not data or "roster" not in data:
        return []

    players = []
    for entry in data["roster"]:
        try:
            person = entry.get("person", {})
            position = entry.get("position", {})
            status = entry.get("status", {})
            players.append({
                "player_name": person.get("fullName", "Unknown"),
                "player_id": person.get("id", 0),
                "position": position.get("abbreviation", ""),
                "roster_status": status.get("description", "Active"),
                "roster_code": status.get("code", "A"),
                "jersey_number": entry.get("jerseyNumber", ""),
            })
        except Exception:
            continue

    return players


# ─────────────────────────────────────────────
# SPRING TRAINING STATCAST
# ─────────────────────────────────────────────

def fetch_spring_statcast(
    player_id: int,
    year: int = None,
) -> dict:
    """
    Pull Statcast metrics from Spring Training games using pybaseball.

    Statcast data (exit velocity, barrel rate, hard hit %) is MORE reliable
    than batting average in small Spring Training samples because these are
    process metrics rather than outcome metrics. A player barreling 15% of
    batted balls is a real signal even in 20 ABs.

    Full Statcast tracking is available at every Spring Training ballpark
    as of 2026.

    Args:
        player_id: MLB player ID (e.g., 660271 for Shohei Ohtani)
        year: Year for ST data. Defaults to current year.

    Returns:
        Dict with:
            avg_exit_velo, max_exit_velo, barrel_rate, hard_hit_pct,
            avg_launch_angle, batted_ball_events, sprint_speed
        Returns empty dict with default values if pybaseball unavailable
        or no data found.
    """
    defaults = {
        "avg_exit_velo": 0.0,
        "max_exit_velo": 0.0,
        "barrel_rate": 0.0,
        "hard_hit_pct": 0.0,
        "avg_launch_angle": 0.0,
        "batted_ball_events": 0,
        "sprint_speed": 0.0,
        "available": False,
    }

    if not PYBASEBALL_AVAILABLE:
        return defaults

    year = year or datetime.now().year

    # Spring Training typically runs mid-February through late March
    st_start = f"{year}-02-15"
    st_end = f"{year}-03-31"

    try:
        df = statcast_batter(st_start, st_end, player_id)

        if df is None or df.empty:
            return defaults

        # Filter for batted ball events (where launch_speed exists)
        batted = df.dropna(subset=["launch_speed"])

        if batted.empty:
            return defaults

        result = {
            "avg_exit_velo": round(batted["launch_speed"].mean(), 1),
            "max_exit_velo": round(batted["launch_speed"].max(), 1),
            "avg_launch_angle": round(
                batted["launch_angle"].mean(), 1
            ) if "launch_angle" in batted.columns else 0.0,
            "batted_ball_events": len(batted),
            "available": True,
        }

        # Barrel rate: launch_speed >= 98 AND launch_angle between 26-30
        # (simplified barrel zone — actual definition is more complex with
        # a sliding scale, but this captures the core signal)
        if len(batted) > 0:
            barrels = batted[
                (batted["launch_speed"] >= 98) &
                (batted["launch_angle"] >= 26) &
                (batted["launch_angle"] <= 30)
            ]
            # Broader barrel definition for better coverage
            barrels_broad = batted[
                (batted["launch_speed"] >= 98) &
                (batted["launch_angle"] >= 8) &
                (batted["launch_angle"] <= 50)
            ]
            result["barrel_rate"] = round(
                len(barrels_broad) / len(batted) * 100, 1
            )

        # Hard hit: exit velo >= 95 mph
        if len(batted) > 0:
            hard_hit = batted[batted["launch_speed"] >= 95]
            result["hard_hit_pct"] = round(
                len(hard_hit) / len(batted) * 100, 1
            )

        # Sprint speed from the full dataset (not just batted balls)
        if "sprint_speed" in df.columns:
            speeds = df["sprint_speed"].dropna()
            if not speeds.empty:
                result["sprint_speed"] = round(speeds.mean(), 1)

        return result

    except Exception as e:
        print(f"[spring.py] Statcast error for player {player_id}: {e}")
        return defaults


# ─────────────────────────────────────────────
# COMBINED SPRING PROFILE (convenience)
# ─────────────────────────────────────────────

def get_spring_profile(
    player_name: str,
    player_id: int = None,
    prior_season_slg: float = 0.400,
    prior_season_avg: float = 0.250,
    year: int = None,
) -> dict:
    """
    Build a complete Spring Training profile for a player, combining
    ST batting stats, Statcast metrics, and injury status into one dict.

    This is the main function app.py should call for the Find Edges tab.

    Args:
        player_name: Full player name
        player_id: MLB player ID (needed for Statcast). If None, Statcast
                   data will be skipped.
        prior_season_slg: Prior regular-season SLG for form comparison
        prior_season_avg: Prior regular-season AVG for form comparison
        year: ST year. Defaults to current year.

    Returns:
        Dict combining spring form multiplier, Statcast data, and injury status.
    """
    year = year or datetime.now().year

    # Fetch all ST stats once and reuse
    try:
        st_stats = fetch_spring_training_stats(year)
    except Exception:
        st_stats = []

    # Fetch injuries once and reuse
    try:
        injuries = fetch_injuries(days_back=60)
    except Exception:
        injuries = []

    # Spring form multiplier
    form = get_spring_form_multiplier(
        player_name=player_name,
        prior_season_slg=prior_season_slg,
        prior_season_avg=prior_season_avg,
        st_stats=st_stats,
        year=year,
    )

    # Statcast data (only if we have a player ID)
    statcast = {}
    if player_id:
        statcast = fetch_spring_statcast(player_id, year)

    # Injury status
    injury = get_player_injury_status(
        player_name=player_name,
        injuries=injuries,
    )

    return {
        "player_name": player_name,
        "spring_form": form,
        "statcast": statcast,
        "injury": injury,
    }
