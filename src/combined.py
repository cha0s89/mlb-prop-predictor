"""
Combined Sharp + Projection Scoring

Merges sharp odds edges with projection analysis into a unified score
per pick. Three signal types:

  CONFIRMED:  Sharp + projection agree → highest conviction
  SHARP_ONLY: Sharp edge exists but projection disagrees or is missing
  PROJ_ONLY:  No sharp line, projection-only pick (A/B grade only)

Scoring:
  CONFIRMED = sharp_edge * 0.65 + proj_edge * 0.35
              +0.03 if FanDuel agrees, +0.02 if 3+ books
  SHARP_ONLY = sharp_edge * 0.80 - proj_edge * 0.20 (skip if < 0.02)
               or sharp_edge * 0.85 if no projection
  PROJ_ONLY  = proj_edge * 0.60 (only A/B grade projections)

Combined grades: A+ >= 0.12, A >= 0.08, B >= 0.05, C >= 0.03, D < 0.03
"""

SIGNAL_CONFIRMED = "CONFIRMED"
SIGNAL_SHARP_ONLY = "SHARP_ONLY"
SIGNAL_PROJECTION_ONLY = "PROJECTION_ONLY"

# Dynamic ensemble weights from hedge updater
try:
    from src.ensemble import get_blend_weights as _get_blend_weights
    _HAS_ENSEMBLE = True
except ImportError:
    _HAS_ENSEMBLE = False


def _blend_weights():
    """Get current sharp/proj blend weights (dynamic or fallback)."""
    if _HAS_ENSEMBLE:
        try:
            return _get_blend_weights()
        except Exception:
            pass
    return 0.65, 0.35


def _combined_grade(score: float) -> str:
    """Assign combined grade from score."""
    if score >= 0.12:
        return "A+"
    elif score >= 0.08:
        return "A"
    elif score >= 0.05:
        return "B"
    elif score >= 0.03:
        return "C"
    else:
        return "D"


def score_single_pick(sharp_edge: dict = None, proj_result: dict = None) -> dict | None:
    """
    Score a single pick by combining sharp edge and projection data.

    Args:
        sharp_edge: dict from find_ev_edges() with keys like edge_pct, pick,
                    fair_prob, fanduel_agrees, num_books, player_name, stat_type, etc.
        proj_result: dict from generate_prediction() with keys like pick, edge,
                     confidence, rating, player_name, stat_type, projection, line, etc.

    Returns:
        Scored pick dict or None if pick doesn't qualify.
    """
    if sharp_edge is None and proj_result is None:
        return None

    has_sharp = sharp_edge is not None and sharp_edge.get("edge_pct", 0) > 0
    has_proj = proj_result is not None and proj_result.get("edge", 0) > 0

    # Extract values
    sharp_edge_val = (sharp_edge["edge_pct"] / 100) if has_sharp else 0.0
    proj_edge_val = proj_result.get("edge", 0) if has_proj else 0.0

    sharp_pick = sharp_edge.get("pick", "") if has_sharp else ""
    proj_pick = proj_result.get("pick", "") if has_proj else ""
    proj_rating = proj_result.get("rating", "D") if has_proj else "D"

    # Get dynamic ensemble blend weights (hedge-updated)
    w_sharp, w_proj = _blend_weights()

    # Determine signal type
    if has_sharp and has_proj and sharp_pick == proj_pick:
        signal = SIGNAL_CONFIRMED
        combined = sharp_edge_val * w_sharp + proj_edge_val * w_proj
        if sharp_edge.get("fanduel_agrees"):
            combined += 0.03
        if sharp_edge.get("num_books", 0) >= 3:
            combined += 0.02
        pick = sharp_pick

    elif has_sharp and has_proj and sharp_pick != proj_pick:
        signal = SIGNAL_SHARP_ONLY
        combined = sharp_edge_val * (w_sharp + 0.15) - proj_edge_val * (w_proj - 0.15)
        if combined < 0.02:
            return None
        pick = sharp_pick

    elif has_sharp and not has_proj:
        signal = SIGNAL_SHARP_ONLY
        combined = sharp_edge_val * (w_sharp + 0.20)
        pick = sharp_pick

    elif has_proj and not has_sharp:
        signal = SIGNAL_PROJECTION_ONLY
        if proj_rating not in ("A", "B"):
            return None
        combined = proj_edge_val * (w_proj + 0.25)
        pick = proj_pick

    else:
        return None

    grade = _combined_grade(combined)

    # Build base info from whichever source is available
    player_name = (sharp_edge or proj_result).get("player_name", "")
    team = (sharp_edge or {}).get("team", "") or (proj_result or {}).get("team", "")
    stat_type = (sharp_edge or proj_result).get("stat_type", "")
    stat_internal = (
        (proj_result or {}).get("stat_internal")
        or (sharp_edge or {}).get("stat_internal")
        or stat_type.lower().strip().replace("+", "_").replace(" ", "_")
    )
    line = (sharp_edge or {}).get("pp_line") or (proj_result or {}).get("line", 0)

    return {
        "player_name": player_name,
        "team": team,
        "stat_type": stat_type,
        "stat_internal": stat_internal,
        "line": line,
        "pick": pick,
        "combined_score": round(combined, 4),
        "combined_grade": grade,
        "signal": signal,
        "sharp_edge_pct": round(sharp_edge_val * 100, 2) if has_sharp else None,
        "proj_edge_pct": round(proj_edge_val * 100, 2) if has_proj else None,
        "confidence": proj_result.get("confidence") if has_proj else None,
        "proj_confidence": proj_result.get("confidence") if has_proj else None,
        "proj_rating": proj_rating if has_proj else None,
        "meets_conf_floor": proj_result.get("meets_conf_floor", True) if has_proj else True,
        "fanduel_agrees": sharp_edge.get("fanduel_agrees") if has_sharp else None,
        "num_books": sharp_edge.get("num_books") if has_sharp else None,
    }


def _match_key(player_name: str, stat_type: str) -> str:
    """Build a match key from player name and stat type."""
    return f"{player_name.lower().strip()}|{stat_type.lower().strip()}"


def score_picks(sharp_edges: list, proj_picks: list) -> list:
    """
    Score all picks by combining sharp edges with projections.

    First pass: score all sharp edges (finding matching projections).
    Second pass: score projection-only picks not already covered.

    Args:
        sharp_edges: list of dicts from find_ev_edges()
        proj_picks: list of dicts from generate_prediction()

    Returns:
        Sorted list of scored pick dicts (best first).
    """
    sharp_edges = sharp_edges or []
    proj_picks = proj_picks or []

    # Index projections by (player, stat_type)
    proj_by_key = {}
    for p in proj_picks:
        key = _match_key(p.get("player_name", ""), p.get("stat_type", ""))
        proj_by_key[key] = p

    scored = []
    matched_proj_keys = set()

    # First pass: score sharp edges
    for edge in sharp_edges:
        key = _match_key(edge.get("player_name", ""), edge.get("stat_type", ""))
        proj = proj_by_key.get(key)
        if proj:
            matched_proj_keys.add(key)

        result = score_single_pick(sharp_edge=edge, proj_result=proj)
        if result:
            scored.append(result)

    # Second pass: projection-only picks
    for p in proj_picks:
        key = _match_key(p.get("player_name", ""), p.get("stat_type", ""))
        if key in matched_proj_keys:
            continue
        result = score_single_pick(sharp_edge=None, proj_result=p)
        if result:
            scored.append(result)

    # Sort by combined score descending
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored
