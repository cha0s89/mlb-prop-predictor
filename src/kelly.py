"""
Kelly Criterion Bankroll Sizing for PrizePicks Props.

Flex slips are not binary wagers, so sizing uses a conservative binary
approximation derived from the simulated positive-return rate and average
positive payout.
"""

from collections import Counter
from typing import Optional

from src.slip_ev import build_correlation_matrix, simulate_slip_ev


# PrizePicks payout multipliers (imported from slips.py logic)
PAYOUT_MULTIPLIERS = {
    "2_power": 3.0,
    "3_power": 6.0,
    "4_power": 10.0,
    "5_power": 20.0,
    "6_power": 25.0,
    "3_flex": 3.0,
    "4_flex": 6.0,
    "5_flex": 10.0,
    "6_flex": 12.5,
}

# Correlation discount for same-game picks (picks on same game are correlated)
SAME_GAME_CORRELATION_DISCOUNT = 0.85


def kelly_fraction(prob_win: float, payout_mult: float) -> float:
    """
    Calculate classic Kelly Criterion fraction.

    The Kelly formula: f = (p * b - q) / b
      p = probability of winning (0-1)
      b = net odds = payout_mult - 1
      q = 1 - p = probability of losing

    Args:
        prob_win: Probability the wager wins, 0-1
        payout_mult: Payout multiplier (e.g., 10.0 for 10x)

    Returns:
        Kelly fraction of bankroll to wager, clamped to [0, 0.25]
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    q = 1 - prob_win
    b = payout_mult - 1

    if b <= 0:
        return 0.0

    kelly = (prob_win * b - q) / b
    # Clamp to [0, 0.25] to never suggest more than 25% of bankroll
    return max(0.0, min(kelly, 0.25))


def half_kelly(prob_win: float, payout_mult: float) -> float:
    """
    Calculate half-Kelly fraction (conservative variant).

    Returns kelly_fraction / 2. Reduces variance while sacrificing some growth rate.

    Args:
        prob_win: Probability the wager wins, 0-1
        payout_mult: Payout multiplier (e.g., 10.0 for 10x)

    Returns:
        Half-Kelly fraction of bankroll to wager, clamped to [0, 0.25]
    """
    return kelly_fraction(prob_win, payout_mult) / 2


def quarter_kelly(prob_win: float, payout_mult: float) -> float:
    """
    Calculate quarter-Kelly fraction (professional standard for props).

    Returns kelly_fraction / 4. Professional standard for player props.
    Full Kelly is mathematically optimal but produces extreme volatility.
    Quarter-Kelly captures ~56% of growth rate while cutting variance by 75%.

    Args:
        prob_win: Probability the wager wins, 0-1
        payout_mult: Payout multiplier (e.g., 10.0 for 10x)

    Returns:
        Quarter-Kelly fraction of bankroll to wager, clamped to [0, 0.25]
    """
    return kelly_fraction(prob_win, payout_mult) / 4


def kelly_with_uncertainty(
    prob_win: float,
    payout_mult: float,
    edge_confidence: float = 1.0
) -> float:
    """
    Calculate Kelly fraction adjusted for edge uncertainty.

    The best weighting is not just which book is sharp, but which book is sharp
    for this market. When edge confidence is lower (uncertain market), reduce the
    Kelly fraction to account for model uncertainty.

    Args:
        prob_win: Probability the wager wins, 0-1
        payout_mult: Payout multiplier (e.g., 10.0 for 10x)
        edge_confidence: Confidence in the edge (0-1), default 1.0 (fully confident)

    Returns:
        Kelly fraction adjusted for uncertainty, clamped to [0, 0.25]
    """
    base_kelly = kelly_fraction(prob_win, payout_mult)
    # Scale down by edge confidence (uncertain edges → smaller fraction)
    adjusted = base_kelly * edge_confidence
    return max(0.0, min(adjusted, 0.25))


def calculate_slip_sizing(
    picks: list[dict],
    bankroll: float,
    slip_type: str,
) -> dict:
    """
    Calculate Kelly-based bankroll sizing for a multi-pick slip.

    Uses quarter-Kelly as the default recommended sizing (professional standard
    for player props). Full Kelly is mathematically optimal but produces extreme
    volatility. Quarter-Kelly captures ~56% of growth rate while cutting variance
    by 75%.

    Args:
        picks: List of pick dicts. Prefers 'win_prob' (outright win chance)
            and falls back to 'confidence' when only resolved-confidence is available.
        bankroll: Total bankroll available, in dollars
        slip_type: Slip type string, e.g. "5_flex", "6_flex"

    Returns:
        Dict with:
          - win_prob: Estimated probability the full slip wins
          - payout_mult: Payout multiplier for this slip type
          - kelly_pct: Full Kelly percentage (0-25)
          - quarter_kelly_pct: Quarter Kelly percentage (0-6.25) [RECOMMENDED DEFAULT]
          - recommended_wager: quarter_kelly_pct * bankroll, rounded to nearest $0.50, min $1
          - max_wager: kelly_pct * bankroll (aggressive ceiling)
          - expected_value: EV in dollars
          - edge_pct: Edge as percentage, (win_prob * payout - 1) * 100
    """
    if not picks or bankroll <= 0:
        return {
            "win_prob": 0.0,
            "payout_mult": 0.0,
            "kelly_pct": 0.0,
            "quarter_kelly_pct": 0.0,
            "recommended_wager": 0.0,
            "max_wager": 0.0,
            "expected_value": 0.0,
            "edge_pct": 0.0,
        }

    # Get payout multiplier for this slip type
    base_payout_mult = PAYOUT_MULTIPLIERS.get(slip_type, 1.0)
    if base_payout_mult <= 1:
        return {
            "win_prob": 0.0,
            "payout_mult": base_payout_mult,
            "kelly_pct": 0.0,
            "quarter_kelly_pct": 0.0,
            "recommended_wager": 0.0,
            "max_wager": 0.0,
            "expected_value": 0.0,
            "edge_pct": 0.0,
        }

    line_type = Counter(str(p.get("line_type", "standard")) for p in picks).most_common(1)[0][0]
    legs = []
    for pick in picks:
        leg_win_prob = pick.get("win_prob")
        if leg_win_prob is None:
            leg_win_prob = pick.get("confidence", 0.5)
        leg_win_prob = max(0.01, min(0.99, float(leg_win_prob)))
        legs.append({
            "team": pick.get("team", ""),
            "pick": pick.get("pick", "MORE"),
            "stat_type": pick.get("stat_internal", pick.get("stat_type", "")),
            "line": pick.get("line", 0.5),
            "win_prob": leg_win_prob,
            "p_push": pick.get("p_push", 0.0),
            "game_id": pick.get("game_pk") or pick.get("game_id") or pick.get("opponent", ""),
        })

    sim = simulate_slip_ev(
        legs,
        entry_type=slip_type,
        n_sims=25_000,
        correlation_matrix=build_correlation_matrix(legs) if len(legs) > 1 else None,
        seed=42,
        line_type=line_type,
    )

    # Use only profitable outcomes for a conservative Kelly approximation.
    win_prob = max(0.0, min(1.0, float(sim.get("win_rate", 0.0))))
    payout_mult = float(sim.get("avg_positive_payout") or base_payout_mult)
    if payout_mult <= 1:
        payout_mult = base_payout_mult

    # Calculate Kelly fractions
    kelly_pct = kelly_fraction(win_prob, payout_mult) * 100
    quarter_kelly_pct = quarter_kelly(win_prob, payout_mult) * 100

    # Calculate recommended wager: quarter-Kelly * bankroll, round to nearest $0.50, min $1
    recommended_raw = (quarter_kelly_pct / 100) * bankroll
    if quarter_kelly_pct <= 0 or sim.get("ev_profit_pct", 0) <= 0:
        recommended_wager = 0.0
    else:
        recommended_wager = max(1.0, round(recommended_raw * 2) / 2)

    # Max wager: full Kelly * bankroll
    max_wager = (kelly_pct / 100) * bankroll

    # Expected value and edge come from the simulation, not the binary approximation.
    ev = recommended_wager * sim.get("ev_profit", 0.0)
    edge_pct = float(sim.get("ev_profit_pct", 0.0))

    return {
        "win_prob": round(win_prob, 4),
        "payout_mult": payout_mult,
        "kelly_pct": round(kelly_pct, 2),
        "quarter_kelly_pct": round(quarter_kelly_pct, 2),
        "recommended_wager": round(recommended_wager, 2),
        "max_wager": round(max_wager, 2),
        "expected_value": round(ev, 2),
        "edge_pct": round(edge_pct, 2),
    }
