"""
Devigging methods for removing sportsbook margin from odds.

The Power method is the recommended default for 2-way over/under props.
Clarke, Kovalchik & Ingram (2017) showed it "universally outperforms the
multiplicative method and outperforms or is comparable to the Shin method."
"""
from scipy.optimize import brentq
import math


def multiplicative_devig(odds_over: float, odds_under: float) -> tuple[float, float]:
    """Classic normalization: divide each implied prob by total.
    Simple but doesn't correct for favorite-longshot bias."""
    p_over = implied_prob(odds_over)
    p_under = implied_prob(odds_under)
    total = p_over + p_under
    return p_over / total, p_under / total


def additive_devig(odds_over: float, odds_under: float) -> tuple[float, float]:
    """Subtract vig equally. Can produce negative probs for longshots."""
    p_over = implied_prob(odds_over)
    p_under = implied_prob(odds_under)
    vig = (p_over + p_under - 1) / 2
    return max(0.01, p_over - vig), max(0.01, p_under - vig)


def power_devig(odds_over: float, odds_under: float) -> tuple[float, float]:
    """Power method: raise implied probs to power k such that they sum to 1.
    Best empirical performance for 2-way markets (Clarke et al. 2017).
    Typical k range: 1.2-1.4."""
    p_over = implied_prob(odds_over)
    p_under = implied_prob(odds_under)

    if p_over + p_under <= 1.0:
        # No vig to remove
        return p_over, p_under

    def objective(k):
        return p_over**k + p_under**k - 1.0

    try:
        k = brentq(objective, 0.5, 5.0)
        return p_over**k, p_under**k
    except (ValueError, RuntimeError):
        # Fallback to multiplicative
        return multiplicative_devig(odds_over, odds_under)


def shin_devig(odds_over: float, odds_under: float, max_iter: int = 100) -> tuple[float, float]:
    """Shin method based on insider-trading model (Shin, 1992).
    For 2-way markets, equivalent to additive method.
    Included for completeness and future multi-way markets."""
    # For 2-outcome markets, Shin = Additive
    return additive_devig(odds_over, odds_under)


def implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds >= 100:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def devig(odds_over: float, odds_under: float, method: str = "power") -> tuple[float, float]:
    """Unified devig interface.

    Args:
        odds_over: American odds for OVER
        odds_under: American odds for UNDER
        method: "power" (default, best), "multiplicative", "additive", "shin"
    Returns:
        (fair_prob_over, fair_prob_under) tuple summing to ~1.0
    """
    methods = {
        "power": power_devig,
        "multiplicative": multiplicative_devig,
        "additive": additive_devig,
        "shin": shin_devig,
    }
    fn = methods.get(method, power_devig)
    return fn(odds_over, odds_under)
