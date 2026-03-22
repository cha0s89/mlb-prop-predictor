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


def fetch_todays_umpires() -> dict:
    """
    Attempt to fetch today's umpire assignments.
    Falls back to empty dict if unavailable.

    Note: MLB doesn't have a clean public API for this.
    In production, scrape from covers.com or umpscorecards.com.
    For now, this is a placeholder that can be manually populated.
    """
    # TODO: Implement scraping from UmpScorecards or Covers
    # For now, return empty — user can manually input
    return {}
