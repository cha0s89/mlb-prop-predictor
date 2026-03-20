"""
Slip Correlation Warnings

Analyzes a proposed PrizePicks slip for correlated outcomes that
increase variance and reduce expected hit rate.

Correlations to detect:
  - Direction stacking (all MORE or all LESS in 4+ pick slip)
  - Same-team stacking (3+ picks from same team → blowout/rain risk)
  - Prop-type stacking (4+ picks of same prop type)
"""

from collections import Counter


def analyze_slip_correlation(picks: list) -> list:
    """
    Analyze a proposed slip for correlated outcomes.

    Args:
        picks: list of dicts, each with keys: player_name, stat_type, line, pick,
               and optionally team.

    Returns:
        List of warning dicts sorted by severity (high first).
        Each warning: {severity, title, message, suggestion}
    """
    warnings = []

    if not picks or len(picks) < 2:
        return warnings

    # Direction stacking
    directions = [p.get("pick", "") for p in picks]
    dir_counts = Counter(directions)

    if len(picks) >= 4:
        for direction, count in dir_counts.items():
            if count == len(picks):
                warnings.append({
                    "severity": "high",
                    "title": "Direction stacking",
                    "message": (
                        f"All {count} picks are {direction}. If the slate trends "
                        f"{'under' if direction == 'MORE' else 'over'}, you lose everything."
                    ),
                    "suggestion": f"Mix in at least 1-2 {'LESS' if direction == 'MORE' else 'MORE'} picks to diversify.",
                })

    # Same-team stacking
    teams = [p.get("team", "") for p in picks if p.get("team")]
    team_counts = Counter(teams)

    for team, count in team_counts.items():
        if not team:
            continue
        if count >= 3:
            warnings.append({
                "severity": "high",
                "title": "Same-team stacking (3+)",
                "message": (
                    f"{count} picks from {team}. A blowout, rain delay, or "
                    f"early pull wipes all of them."
                ),
                "suggestion": "Limit to 2 picks per team max.",
            })
        elif count == 2:
            # Check if same prop type too
            team_props = [p.get("stat_type", "") for p in picks if p.get("team") == team]
            if len(set(team_props)) == 1:
                warnings.append({
                    "severity": "low",
                    "title": "Same-team pair (same prop)",
                    "message": (
                        f"2 {team} picks on {team_props[0]}. Outcomes are linked "
                        f"to the same game flow."
                    ),
                    "suggestion": "Acceptable, but different prop types would diversify better.",
                })
            else:
                warnings.append({
                    "severity": "low",
                    "title": "Same-team pair",
                    "message": f"2 picks from {team}. Acceptable if different prop types.",
                    "suggestion": "No action needed — different props reduce correlation.",
                })

    # Prop-type stacking
    prop_types = [p.get("stat_type", "") for p in picks]
    prop_counts = Counter(prop_types)

    for prop, count in prop_counts.items():
        if not prop:
            continue
        if count >= 4:
            warnings.append({
                "severity": "high",
                "title": "Prop-type stacking (4+)",
                "message": (
                    f"{count} picks on {prop}. League-wide trends (weather, "
                    f"umpire crew) can move all of them together."
                ),
                "suggestion": "Diversify across 2-3 different prop types.",
            })
        elif count == 3:
            warnings.append({
                "severity": "medium",
                "title": "Prop-type concentration",
                "message": f"3 picks on {prop}. Moderate correlation risk.",
                "suggestion": "Consider swapping one for a different prop type.",
            })

    # Sort by severity: high > medium > low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    warnings.sort(key=lambda w: severity_order.get(w["severity"], 3))

    # If no high warnings, add a positive note
    if not any(w["severity"] == "high" for w in warnings):
        warnings.append({
            "severity": "low",
            "title": "Slip looks reasonable",
            "message": "No major correlation risks detected.",
            "suggestion": "",
        })

    return warnings


def format_warnings_streamlit(warnings: list) -> list:
    """
    Format warnings for Streamlit display.

    Returns:
        List of (emoji, text) tuples.
    """
    emoji_map = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }

    formatted = []
    for w in warnings:
        emoji = emoji_map.get(w["severity"], "⚪")
        text = f"**{w['title']}** — {w['message']}"
        if w.get("suggestion"):
            text += f" *{w['suggestion']}*"
        formatted.append((emoji, text))

    return formatted
