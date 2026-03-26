"""
Canonical team metadata — the single source of truth for team abbreviations,
IDs, aliases, full names, and normalization across the entire project.

Every other module should import from here instead of maintaining its own maps.
"""

# MLB Stats API team IDs → canonical abbreviation
TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Reverse: abbreviation → MLB Stats API team ID
ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBR.items()}

# Canonical abbreviation → full team name
ABBR_TO_NAME = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD":  "San Diego Padres",
    "SF":  "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB":  "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

# All known aliases → canonical abbreviation
# Add new aliases here when PrizePicks, FanGraphs, or other sources use different names
TEAM_ALIASES = {
    # Identity (canonical → canonical)
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "CWS": "CWS", "DET": "DET",
    "HOU": "HOU", "KC": "KC", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SD", "SF": "SF", "SEA": "SEA",
    "STL": "STL", "TB": "TB", "TEX": "TEX", "TOR": "TOR", "WSH": "WSH",
    # Common aliases
    "AZ": "ARI",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WAS": "WSH",
    "ATH": "OAK",   # Sacramento Athletics (temporary)
    # Full name fragments (lowercase) used by PrizePicks
    "diamondbacks": "ARI", "braves": "ATL", "orioles": "BAL",
    "red sox": "BOS", "cubs": "CHC", "reds": "CIN",
    "guardians": "CLE", "rockies": "COL", "white sox": "CWS",
    "tigers": "DET", "astros": "HOU", "royals": "KC",
    "angels": "LAA", "dodgers": "LAD", "marlins": "MIA",
    "brewers": "MIL", "twins": "MIN", "mets": "NYM",
    "yankees": "NYY", "athletics": "OAK", "phillies": "PHI",
    "pirates": "PIT", "padres": "SD", "giants": "SF",
    "mariners": "SEA", "cardinals": "STL", "rays": "TB",
    "rangers": "TEX", "blue jays": "TOR", "nationals": "WSH",
}


def normalize_team(raw: str) -> str:
    """Normalize any team string to canonical abbreviation.

    Handles abbreviations, aliases, and partial name matches.
    Returns empty string if no match found.
    """
    if not raw:
        return ""
    raw_clean = raw.strip()

    # Try exact match (case-insensitive on alias keys)
    upper = raw_clean.upper()
    if upper in TEAM_ALIASES:
        return TEAM_ALIASES[upper]

    # Try lowercase (for full name fragments)
    lower = raw_clean.lower()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]

    # Try substring match on full names
    for name_fragment, abbr in TEAM_ALIASES.items():
        if isinstance(name_fragment, str) and len(name_fragment) > 3:
            if name_fragment in lower or lower in name_fragment:
                return abbr

    return ""


def team_id(abbr: str) -> int:
    """Convert canonical abbreviation to MLB Stats API team ID. Returns 0 if unknown."""
    canonical = normalize_team(abbr)
    return ABBR_TO_TEAM_ID.get(canonical, 0)


def team_name(abbr: str) -> str:
    """Convert abbreviation to full team name."""
    canonical = normalize_team(abbr)
    return ABBR_TO_NAME.get(canonical, abbr)
