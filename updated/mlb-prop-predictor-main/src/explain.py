"""
Prediction Explanation Module
Shows the full adjustment chain for every pick.

When you look at a pick and wonder "why did the model say MORE?"
this module shows every factor, its direction, its weight, and
how it changed the projection step by step.

Critical for model tuning — if you see the weather adjustment
consistently pulling picks in the wrong direction, you know
to reduce its weight.
"""


def build_explanation(
    player_name: str,
    stat_type: str,
    line: float,
    projection: float,
    pick: str,
    confidence: float,
    rating: str,
    # Individual factor details
    base_stat: float = None,
    base_stat_name: str = "Season rate",
    regressed_stat: float = None,
    xstat_adjustment: dict = None,
    bvp_adjustment: dict = None,
    platoon_adjustment: dict = None,
    opp_quality_adjustment: dict = None,
    park_adjustment: dict = None,
    weather_adjustment: dict = None,
    umpire_adjustment: dict = None,
    lineup_pos_adjustment: dict = None,
    trend_adjustment: dict = None,
    sharp_edge: dict = None,
) -> dict:
    """
    Build a step-by-step explanation of how a projection was calculated.

    Returns a dict with:
    - summary: one-line verdict
    - steps: ordered list of adjustment steps
    - signals_agree: count of factors pointing same direction as pick
    - signals_disagree: count pointing opposite
    - confidence_factors: what's driving confidence up or down
    """
    steps = []
    signals_for = 0
    signals_against = 0

    # Step 1: Base rate
    if base_stat is not None:
        steps.append({
            "order": 1,
            "factor": base_stat_name,
            "detail": f"{base_stat:.3f}" if base_stat < 1 else f"{base_stat:.1f}",
            "impact": "baseline",
            "direction": "neutral",
        })

    # Step 2: Bayesian regression
    if regressed_stat is not None and base_stat is not None:
        delta = regressed_stat - base_stat
        direction = "up" if delta > 0 else ("down" if delta < 0 else "neutral")
        steps.append({
            "order": 2,
            "factor": "Bayesian regression",
            "detail": f"{base_stat:.3f} → {regressed_stat:.3f} (regressed toward league avg)",
            "impact": f"{delta:+.3f}" if abs(delta) < 1 else f"{delta:+.1f}",
            "direction": direction,
        })

    # Step 3: Statcast expected stats (xBA, xSLG, etc.)
    if xstat_adjustment:
        steps.append({
            "order": 3,
            "factor": "Statcast expected stats",
            "detail": xstat_adjustment.get("detail", "xStat blend applied"),
            "impact": xstat_adjustment.get("impact", ""),
            "direction": xstat_adjustment.get("direction", "neutral"),
        })

    # Step 4: BvP matchup
    if bvp_adjustment:
        d = bvp_adjustment
        if d.get("has_data"):
            direction = "supports MORE" if d.get("favorable") else "supports LESS"
            if pick == "MORE" and d.get("favorable"):
                signals_for += 1
            elif pick == "LESS" and not d.get("favorable"):
                signals_for += 1
            else:
                signals_against += 1

            steps.append({
                "order": 4,
                "factor": "BvP matchup history",
                "detail": d.get("detail", f"{d.get('pa', 0)} PA in matchup"),
                "impact": d.get("impact", ""),
                "direction": direction,
                "weight": "15-25% depending on sample",
            })

    # Step 5: Platoon
    if platoon_adjustment:
        d = platoon_adjustment
        favorable = d.get("favorable")
        if favorable is not None:
            if (pick == "MORE" and favorable) or (pick == "LESS" and not favorable):
                signals_for += 1
            else:
                signals_against += 1

        steps.append({
            "order": 5,
            "factor": "Platoon split",
            "detail": d.get("description", ""),
            "impact": f"×{d.get('adjustment', 1.0):.2f}",
            "direction": "favorable" if favorable else "unfavorable",
            "weight": "8-15%",
        })

    # Step 6: Opposing quality
    if opp_quality_adjustment:
        d = opp_quality_adjustment
        steps.append({
            "order": 6,
            "factor": "Opposing quality",
            "detail": d.get("detail", ""),
            "impact": d.get("impact", ""),
            "direction": d.get("direction", "neutral"),
            "weight": "10-15%",
        })

    # Step 7: Park factor
    if park_adjustment:
        d = park_adjustment
        steps.append({
            "order": 7,
            "factor": "Park factor",
            "detail": d.get("detail", f"Park: {d.get('park', 'Unknown')}"),
            "impact": f"×{d.get('multiplier', 1.0):.3f}",
            "direction": d.get("direction", "neutral"),
            "weight": "5-10%",
        })

    # Step 8: Weather
    if weather_adjustment:
        d = weather_adjustment
        steps.append({
            "order": 8,
            "factor": "Weather",
            "detail": d.get("detail", ""),
            "impact": f"×{d.get('multiplier', 1.0):.3f}",
            "direction": d.get("direction", "neutral"),
            "weight": "5%",
        })

    # Step 9: Umpire
    if umpire_adjustment:
        d = umpire_adjustment
        if d.get("known"):
            steps.append({
                "order": 9,
                "factor": "Umpire",
                "detail": d.get("detail", d.get("impact", "")),
                "impact": d.get("k_adjustment", 0),
                "direction": d.get("direction", "neutral"),
                "weight": "10% (K props only)",
            })

    # Step 10: Lineup position
    if lineup_pos_adjustment:
        d = lineup_pos_adjustment
        steps.append({
            "order": 10,
            "factor": "Lineup position",
            "detail": d.get("detail", f"Batting {d.get('position', '?')}th"),
            "impact": f"{d.get('pa_adjustment', 0):+.1f} PA",
            "direction": d.get("direction", "neutral"),
        })

    # Step 11: Recent trend
    if trend_adjustment:
        d = trend_adjustment
        if d.get("has_data"):
            steps.append({
                "order": 11,
                "factor": "Recent trend",
                "detail": d.get("label", ""),
                "impact": f"×{d.get('trend_multiplier', 1.0):.3f}",
                "direction": "hot" if d.get("trend_multiplier", 1.0) > 1.02 else (
                    "cold" if d.get("trend_multiplier", 1.0) < 0.98 else "neutral"
                ),
                "weight": "5-8% (tiebreaker only)",
            })

    # Step 12: Sharp book consensus (the big one)
    if sharp_edge:
        d = sharp_edge
        edge_pct = d.get("edge_pct", 0)
        if edge_pct > 0:
            signals_for += 2  # Double weight — this is the primary signal
            steps.append({
                "order": 0,  # Sorted to top
                "factor": "⭐ Sharp book consensus",
                "detail": f"Fair prob: {d.get('fair_prob', 0)*100:.1f}% | FanDuel {'confirms' if d.get('fanduel_agrees') else 'N/A'} | {d.get('num_books', 0)} books",
                "impact": f"+{edge_pct:.1f}% edge vs PrizePicks",
                "direction": d.get("pick", pick),
                "weight": "40% (PRIMARY signal)",
            })

    # Sort steps by order
    steps.sort(key=lambda x: x.get("order", 99))

    # Build summary
    total_signals = signals_for + signals_against
    agreement_pct = signals_for / total_signals * 100 if total_signals > 0 else 0

    if agreement_pct >= 80:
        agreement = f"✅ Strong agreement ({signals_for}/{total_signals} factors confirm {pick})"
    elif agreement_pct >= 60:
        agreement = f"👍 Moderate agreement ({signals_for}/{total_signals} factors confirm {pick})"
    elif agreement_pct >= 40:
        agreement = f"⚠️ Mixed signals ({signals_for} for, {signals_against} against)"
    else:
        agreement = f"❌ Most factors disagree with {pick} ({signals_against}/{total_signals} oppose)"

    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "line": line,
        "projection": projection,
        "pick": pick,
        "confidence": confidence,
        "rating": rating,
        "steps": steps,
        "signals_for": signals_for,
        "signals_against": signals_against,
        "agreement": agreement,
        "summary": (
            f"{player_name} {stat_type} {line} → {pick} "
            f"(proj: {projection}, conf: {confidence*100:.1f}%, grade: {rating}) "
            f"| {agreement}"
        ),
    }


def format_explanation_text(explanation: dict) -> str:
    """Format an explanation dict into readable text for display."""
    lines = []
    lines.append(f"**{explanation['summary']}**")
    lines.append("")

    for step in explanation.get("steps", []):
        factor = step["factor"]
        detail = step.get("detail", "")
        impact = step.get("impact", "")
        weight = step.get("weight", "")

        line = f"  {factor}: {detail}"
        if impact:
            line += f" → {impact}"
        if weight:
            line += f" *(weight: {weight})*"
        lines.append(line)

    lines.append("")
    lines.append(explanation.get("agreement", ""))

    return "\n".join(lines)
