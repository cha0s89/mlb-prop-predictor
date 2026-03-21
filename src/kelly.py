"""
Kelly Criterion Bankroll Sizing for PrizePicks Props

Implements conservative Kelly fractional betting strategy to maximize long-term
growth while limiting drawdown risk. Recommended approach: half-Kelly (0.5x) for
safety margin.

Formula: f = (p * b - q) / b
  where p = prob_win, b = net_odds (payout - 1), q = 1 - p
  result is clamped to [0, 0.25] to limit aggressive sizing
"""

from typing import Optional


# PrizePicks payout multipliers (imported from slips.py logic)
PAYOUT_MULTIPLIERS = {
    "2_power": 3.0,
    "3_power": 5.0,
    "4_flex": 5.0,
    "5_flex": 10.0,
    "6_flex": 25.0,
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

    Returns kelly_fraction / 2. Recommended for actual wagering.
    Reduces variance while sacrificing some growth rate.

    Args:
        prob_win: Probability the wager wins, 0-1
        payout_mult: Payout multiplier (e.g., 10.0 for 10x)

    Returns:
        Half-Kelly fraction of bankroll to wager, clamped to [0, 0.25]
    """
    return kelly_fraction(prob_win, payout_mult) / 2


def calculate_slip_sizing(
    picks: list[dict],
    bankroll: float,
    slip_type: str,
) -> dict:
    """
    Calculate Kelly-based bankroll sizing for a multi-pick slip.

    Args:
        picks: List of pick dicts, each with 'confidence' key (0-1 scale)
        bankroll: Total bankroll available, in dollars
        slip_type: Slip type string, e.g. "5_flex", "6_flex"

    Returns:
        Dict with:
          - win_prob: Estimated probability the full slip wins
          - payout_mult: Payout multiplier for this slip type
          - kelly_pct: Full Kelly percentage (0-25)
          - half_kelly_pct: Half Kelly percentage (0-12.5)
          - recommended_wager: half_kelly_pct * bankroll, rounded to nearest $0.50, min $1
          - max_wager: kelly_pct * bankroll (aggressive ceiling)
          - expected_value: EV in dollars
          - edge_pct: Edge as percentage, (win_prob * payout - 1) * 100
    """
    if not picks or bankroll <= 0:
        return {
            "win_prob": 0.0,
            "payout_mult": 0.0,
            "kelly_pct": 0.0,
            "half_kelly_pct": 0.0,
            "recommended_wager": 0.0,
            "max_wager": 0.0,
            "expected_value": 0.0,
            "edge_pct": 0.0,
        }

    # Get payout multiplier for this slip type
    payout_mult = PAYOUT_MULTIPLIERS.get(slip_type, 1.0)
    if payout_mult <= 1:
        return {
            "win_prob": 0.0,
            "payout_mult": payout_mult,
            "kelly_pct": 0.0,
            "half_kelly_pct": 0.0,
            "recommended_wager": 0.0,
            "max_wager": 0.0,
            "expected_value": 0.0,
            "edge_pct": 0.0,
        }

    # Calculate joint win probability
    # Start with product of individual confidences
    joint_prob = 1.0
    for pick in picks:
        conf = pick.get("confidence", 0.5)
        joint_prob *= conf

    # Apply same-game correlation discount
    # (assume all picks are same-game parlay for conservative estimate)
    win_prob = joint_prob * SAME_GAME_CORRELATION_DISCOUNT

    # Calculate Kelly fractions
    kelly_pct = kelly_fraction(win_prob, payout_mult) * 100
    half_kelly_pct = half_kelly(win_prob, payout_mult) * 100

    # Calculate recommended wager: half-Kelly * bankroll, round to nearest $0.50, min $1
    recommended_raw = (half_kelly_pct / 100) * bankroll
    recommended_wager = max(1.0, round(recommended_raw * 2) / 2)

    # Max wager: full Kelly * bankroll
    max_wager = (kelly_pct / 100) * bankroll

    # Expected value: (prob_win * payout) - (prob_loss * wager)
    prob_loss = 1 - win_prob
    payout_return = recommended_wager * payout_mult
    ev = (win_prob * payout_return) - (prob_loss * recommended_wager)

    # Edge percentage: how much better than break-even
    edge_pct = (win_prob * payout_mult - 1) * 100

    return {
        "win_prob": round(win_prob, 4),
        "payout_mult": payout_mult,
        "kelly_pct": round(kelly_pct, 2),
        "half_kelly_pct": round(half_kelly_pct, 2),
        "recommended_wager": round(recommended_wager, 2),
        "max_wager": round(max_wager, 2),
        "expected_value": round(ev, 2),
        "edge_pct": round(edge_pct, 2),
    }
