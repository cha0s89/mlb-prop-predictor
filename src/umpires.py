"""
Umpire Module
Tracks umpire assignments and their impact on strikeout props.
High-K umpires generate 17.5+ total Ks/game vs 14.8 average — a 1-2 K swing per pitcher.
"""

import requests
from datetime import datetime


# Historical umpire K-rate tendencies (games per K above/below average)
# Positive = more Ks than average, Negative = fewer
# Source: UmpScorecards.com averages — update at start of each season
# These are total game K adjustments (split ~evenly between both pitchers)
UMPIRE_K_TENDENCY = {
    # High-K umpires (favor K overs)
    "Manny Gonzalez": +1.8,
    "Quinn Wolcott": +1.6,
    "Tripp Gibson": +1.5,
    "Nick Mahrley": +1.4,
    "Shane Livensparger": +1.3,
    "David Rackley": +1.2,
    "Nate Tomlinson": +1.2,
    "Brennan Miller": +1.1,
    "Ryan Wills": +1.0,
    "Alex Tosi": +1.0,

    # Neutral umpires
    "Pat Hoberg": +0.2,
    "James Hoye": +0.1,
    "Dan Iassogna": 0.0,
    "Bill Welke": 0.0,
    "Chris Guccione": -0.1,

    # Low-K umpires (favor K unders)
    "Angel Hernandez": -1.0,
    "CB Bucknor": -0.9,
    "Laz Diaz": -0.8,
    "Joe West": -0.8,
    "Hunter Wendelstedt": -0.7,
    "Mark Carlson": -0.6,
    "Alfonso Marquez": -0.5,
    "Todd Tichenor": -0.5,
}


# v018 Task 3D: Advanced umpire zone-shape model
# Beyond simple K totals, umpires have measurable zone-shape characteristics:
# - Zone expansion: how many inches beyond the rulebook zone they call strikes
# - Two-strike expansion: additional expansion with 2 strikes (K-seeking behavior)
# - BB boost: umpires with tight zones increase walk rates
# Source: UmpScorecards.com zone analysis, Swish Analytics umpire factors
UMPIRE_ZONE_SHAPE = {
    # name: {expansion: inches beyond zone, two_strike_exp: additional inches at 2K, bb_boost: walk multiplier}
    "Manny Gonzalez":      {"expansion": 1.2, "two_strike_exp": 0.8, "bb_boost": 0.92},
    "Quinn Wolcott":       {"expansion": 1.0, "two_strike_exp": 0.7, "bb_boost": 0.94},
    "Tripp Gibson":        {"expansion": 0.9, "two_strike_exp": 0.6, "bb_boost": 0.95},
    "Nick Mahrley":        {"expansion": 0.8, "two_strike_exp": 0.5, "bb_boost": 0.96},
    "Shane Livensparger":  {"expansion": 0.7, "two_strike_exp": 0.5, "bb_boost": 0.96},
    "Pat Hoberg":          {"expansion": 0.1, "two_strike_exp": 0.1, "bb_boost": 1.00},
    "Dan Iassogna":        {"expansion": 0.0, "two_strike_exp": 0.0, "bb_boost": 1.00},
    "Angel Hernandez":     {"expansion": -0.5, "two_strike_exp": -0.2, "bb_boost": 1.08},
    "CB Bucknor":          {"expansion": -0.4, "two_strike_exp": -0.1, "bb_boost": 1.06},
    "Laz Diaz":            {"expansion": -0.3, "two_strike_exp": 0.0, "bb_boost": 1.05},
    "Hunter Wendelstedt":  {"expansion": -0.3, "two_strike_exp": -0.1, "bb_boost": 1.04},
}


def advanced_umpire_adjustment(umpire_name: str, pitcher_type: str = None,
                                prop_type: str = "pitcher_strikeouts") -> dict:
    """Context-aware umpire adjustment using zone-shape model.

    Umpires with expanded zones amplify K rates, especially for:
    - Edge-worker pitchers (high called strike rate) who paint corners
    - At 2-strike counts where the zone expands further

    Args:
        umpire_name: Umpire name
        pitcher_type: 'edge_worker', 'power', or None
            edge_worker = high called-strike%, relies on framing (e.g., Castillo, Fried)
            power = high whiff rate, less affected by zone (e.g., deGrom, Cole)
        prop_type: Which prop type to adjust for

    Returns:
        dict with k_factor, bb_factor, zone_info
    """
    if not umpire_name:
        return {"k_factor": 1.0, "bb_factor": 1.0, "zone_info": None}

    # Look up zone shape data
    zone = None
    for name, z in UMPIRE_ZONE_SHAPE.items():
        if name.lower() in umpire_name.lower() or umpire_name.lower() in name.lower():
            zone = z
            umpire_name = name
            break

    if zone is None:
        # Fall back to simple K tendency
        basic = get_umpire_k_adjustment(umpire_name)
        return {
            "k_factor": 1.0 + basic["k_adjustment"] * 0.04,  # ~4% per K adjustment unit
            "bb_factor": 1.0 - basic["k_adjustment"] * 0.02,  # Inverse for walks
            "zone_info": None,
        }

    # Base K adjustment from zone expansion
    # Each inch of zone expansion ≈ 3% more called strikes ≈ 1.5% more Ks
    expansion = zone["expansion"]
    k_factor = 1.0 + expansion * 0.015

    # Two-strike expansion multiplier
    # At 2-strike counts (~35% of PAs), additional expansion matters
    two_strike_exp = zone["two_strike_exp"]
    k_factor += two_strike_exp * 0.01 * 0.35  # Weighted by 2-strike PA fraction

    # Edge-worker pitchers benefit more from generous umpires
    # Their stuff lives on the edges — zone expansion directly creates called strikes
    if pitcher_type == "edge_worker":
        k_factor *= 1.05  # 5% amplification for edge workers
    elif pitcher_type == "power":
        # Power pitchers get Ks via whiffs — less affected by zone
        k_factor = 1.0 + (k_factor - 1.0) * 0.6  # Dampen effect by 40%

    # Walk adjustment
    bb_factor = zone["bb_boost"]

    # For walk props, invert: big zone = fewer walks
    if prop_type in ("walks", "walks_allowed", "pitcher_walks"):
        return {
            "k_factor": round(k_factor, 4),
            "bb_factor": round(bb_factor, 4),
            "zone_info": {
                "expansion_inches": expansion,
                "two_strike_exp_inches": two_strike_exp,
                "umpire": umpire_name,
            },
        }

    return {
        "k_factor": round(k_factor, 4),
        "bb_factor": round(bb_factor, 4),
        "zone_info": {
            "expansion_inches": expansion,
            "two_strike_exp_inches": two_strike_exp,
            "umpire": umpire_name,
        },
    }


def get_umpire_k_adjustment(umpire_name: str) -> dict:
    """
    Get the K-rate adjustment for a given umpire.

    Returns:
        dict with umpire name, k_adjustment (per pitcher, so halved),
        impact_label, and whether data exists
    """
    if not umpire_name:
        return {"known": False, "k_adjustment": 0.0, "impact": "Unknown"}

    # Try exact match first, then partial
    adjustment = UMPIRE_K_TENDENCY.get(umpire_name)

    if adjustment is None:
        # Try partial match
        for name, adj in UMPIRE_K_TENDENCY.items():
            if name.lower() in umpire_name.lower() or umpire_name.lower() in name.lower():
                adjustment = adj
                umpire_name = name
                break

    if adjustment is None:
        return {"known": False, "name": umpire_name, "k_adjustment": 0.0, "impact": "No data"}

    # Per-pitcher adjustment (total game K swing / 2 pitchers)
    per_pitcher = adjustment / 2.0

    if per_pitcher > 0.5:
        impact = "🔥 High-K ump"
    elif per_pitcher > 0.2:
        impact = "↗️ Slight K boost"
    elif per_pitcher < -0.5:
        impact = "❄️ Low-K ump"
    elif per_pitcher < -0.2:
        impact = "↘️ Slight K suppress"
    else:
        impact = "➖ Neutral"

    return {
        "known": True,
        "name": umpire_name,
        "k_adjustment": round(per_pitcher, 2),
        "total_game_adjustment": adjustment,
        "impact": impact,
    }


def fetch_todays_umpires(game_date: str = None) -> dict:
    """Fetch today's home-plate umpire assignments from MLB Stats API.

    The schedule endpoint with `hydrate=officials` returns umpire crews
    for each game. We extract the "Home Plate" official for each game
    and map it by both team abbreviation and game_pk.

    Args:
        game_date: Optional YYYY-MM-DD string. Defaults to today.

    Returns:
        dict mapping team abbreviation (e.g. "NYY") → home plate umpire name,
        plus "_by_game_pk" → {game_pk: umpire_name} for direct lookup.
        Empty dict on any failure.
    """
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "sportId": 1,
                "date": game_date,
                "hydrate": "officials",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    umpire_map = {}
    by_game_pk = {}

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            officials = game.get("officials", [])

            hp_umpire = None
            for official in officials:
                if official.get("officialType") == "Home Plate":
                    hp_umpire = official.get("official", {}).get("fullName")
                    break

            if not hp_umpire:
                continue

            # Store by game_pk
            if game_pk:
                by_game_pk[game_pk] = hp_umpire

            # Store by team abbreviation (both teams get the same HP ump)
            teams = game.get("teams", {})
            for side in ("away", "home"):
                team_info = teams.get(side, {}).get("team", {})
                abbrev = team_info.get("abbreviation")
                if abbrev:
                    umpire_map[abbrev] = hp_umpire

    umpire_map["_by_game_pk"] = by_game_pk
    return umpire_map
