"""
Monte Carlo Slip EV Simulator — Rules-Correct PrizePicks Grading

Instead of naive independence assumptions, simulates outcomes accounting for:
  1. Exact PrizePicks payout tables (March 2026)
  2. Tie/DNP/Reboot reversion rules
  3. Intra-slip correlations (same-team, same-game effects)
  4. Push probability per prop type
  5. 3-way outcomes per leg (win/loss/tie)

Source: Deep Research Report 4 (Sierra) — PrizePicks Phase-2 Research
"""

import math
import random
from typing import Optional
from scipy.stats import norm

from src.slips import PAYOUTS, TIE_SPECIAL


# ── Push probability by prop type ──
# Empirical: props with integer lines have ~3-8% push rate;
# half-integer lines have ~0% push
PUSH_RATES = {
    "pitcher_strikeouts": 0.06,
    "hits": 0.07,
    "total_bases": 0.05,
    "home_runs": 0.04,
    "rbis": 0.06,
    "runs": 0.06,
    "stolen_bases": 0.03,
    "hitter_fantasy_score": 0.02,  # Decimal line → rare push
    "earned_runs": 0.06,
    "pitching_outs": 0.04,
    "walks": 0.06,
    "hits_allowed": 0.06,
}


def _push_rate_for_line(stat_type: str, line: float) -> float:
    """Estimate push probability based on stat type and line value."""
    # Half-integer lines (e.g., 4.5, 1.5) cannot push
    if line != int(line):
        return 0.0
    return PUSH_RATES.get(stat_type.lower().replace(" ", "_"), 0.04)


def _apply_prizepicks_rules(
    outcomes: list[str],
    entry_type: str,
) -> float:
    """Apply PrizePicks grading rules to get payout multiplier.

    Args:
        outcomes: list of 'W', 'L', or 'T' per leg
        entry_type: e.g. '5_flex', '3_power'

    Returns:
        Payout multiplier (0 = loss, >0 = some return)
    """
    wins = outcomes.count("W")
    losses = outcomes.count("L")
    ties = outcomes.count("T")

    if wins + losses == 0:
        # All ties or empty → refund
        return 1.0

    # Check special tie cases (e.g., 2-pick Power)
    special = TIE_SPECIAL.get(entry_type, {})
    special_key = (wins, ties)
    if special_key in special:
        return special[special_key]

    # Standard tie handling: revert down one level
    effective_picks = wins + losses
    is_power = "power" in entry_type
    effective_type = f"{effective_picks}_{'power' if is_power else 'flex'}"

    if ties > 0:
        payout_table = PAYOUTS.get(effective_type, PAYOUTS.get(entry_type, {}))
    else:
        payout_table = PAYOUTS.get(entry_type, {})

    return payout_table.get(wins, 0)


def simulate_slip_ev(
    legs: list[dict],
    entry_type: str,
    n_sims: int = 100_000,
    correlation_matrix: Optional[list] = None,
    seed: int = 42,
) -> dict:
    """Monte Carlo simulation of slip expected value.

    Args:
        legs: list of dicts with keys:
            - win_prob: float (probability of winning this leg)
            - stat_type: str (for push rate estimation)
            - line: float (the line value)
            - team: str (optional, for correlation)
        entry_type: e.g. '5_flex', '3_power'
        n_sims: number of simulations
        correlation_matrix: optional NxN correlation matrix for legs
        seed: random seed

    Returns:
        dict with ev_profit, ev_payout, std_dev, win_rate, partial_rate,
        loss_rate, tie_impact, and percentiles
    """
    rng = random.Random(seed)
    n_legs = len(legs)

    # Pre-compute per-leg probabilities
    leg_probs = []
    for leg in legs:
        win_p = leg["win_prob"]
        push_p = _push_rate_for_line(
            leg.get("stat_type", "unknown"),
            leg.get("line", 0.5),
        )
        # Adjust: win_prob is for the outcome direction, push reduces both sides
        loss_p = max(0, 1.0 - win_p - push_p)
        leg_probs.append((win_p, loss_p, push_p))

    # Use Gaussian copula for correlated legs if correlation matrix provided
    use_copula = (
        correlation_matrix is not None
        and len(correlation_matrix) == n_legs
        and n_legs > 1
    )

    payout_sum = 0.0
    payout_sq_sum = 0.0
    win_count = 0
    partial_count = 0
    loss_count = 0
    tie_games = 0  # count of sims where at least one tie

    for _ in range(n_sims):
        if use_copula:
            # Generate correlated uniform random variables via Gaussian copula
            z = _sample_correlated_normals(correlation_matrix, rng)
            u = [norm.cdf(zi) for zi in z]
        else:
            u = [rng.random() for _ in range(n_legs)]

        outcomes = []
        has_tie = False
        for i in range(n_legs):
            win_p, loss_p, push_p = leg_probs[i]
            r = u[i]
            if r < win_p:
                outcomes.append("W")
            elif r < win_p + loss_p:
                outcomes.append("L")
            else:
                outcomes.append("T")
                has_tie = True

        if has_tie:
            tie_games += 1

        mult = _apply_prizepicks_rules(outcomes, entry_type)
        payout_sum += mult
        payout_sq_sum += mult * mult

        if mult > 1:
            win_count += 1
        elif mult > 0 and mult < 1:
            partial_count += 1
        elif mult == 0:
            loss_count += 1

    mean_mult = payout_sum / n_sims
    var_mult = payout_sq_sum / n_sims - mean_mult ** 2
    std_mult = math.sqrt(max(var_mult, 0))

    return {
        "ev_payout": round(mean_mult, 4),
        "ev_profit": round(mean_mult - 1.0, 4),
        "ev_profit_pct": round((mean_mult - 1.0) * 100, 2),
        "std_dev": round(std_mult, 4),
        "win_rate": round(win_count / n_sims, 4),
        "partial_rate": round(partial_count / n_sims, 4),
        "loss_rate": round(loss_count / n_sims, 4),
        "tie_impact": round(tie_games / n_sims, 4),
        "n_sims": n_sims,
        "entry_type": entry_type,
        "sharpe_ratio": round(
            (mean_mult - 1.0) / std_mult if std_mult > 0 else 0, 3
        ),
    }


def _sample_correlated_normals(
    corr_matrix: list[list[float]], rng: random.Random
) -> list[float]:
    """Sample correlated normal random variables using Cholesky decomposition.

    Args:
        corr_matrix: NxN correlation matrix
        rng: random number generator

    Returns:
        List of N correlated standard normal samples
    """
    n = len(corr_matrix)

    # Cholesky decomposition (lower triangular L where R = L @ L^T)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = corr_matrix[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-10))
            else:
                L[i][j] = (corr_matrix[i][j] - s) / L[j][j] if L[j][j] > 0 else 0

    # Generate independent normals
    z_indep = [_standard_normal(rng) for _ in range(n)]

    # Apply Cholesky to correlate
    z_corr = [0.0] * n
    for i in range(n):
        z_corr[i] = sum(L[i][j] * z_indep[j] for j in range(i + 1))

    return z_corr


def _standard_normal(rng: random.Random) -> float:
    """Box-Muller transform for standard normal sample."""
    u1 = max(rng.random(), 1e-10)
    u2 = rng.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def build_correlation_matrix(legs: list[dict]) -> list[list[float]]:
    """Build a correlation matrix for slip legs based on team/game relationships.

    Uses empirical correlation factors from research:
    - Same-team, same-direction: r ≈ 0.25
    - Same-team, opposite-direction: r ≈ 0.10
    - Pitcher K over + opposing team hits under: r ≈ 0.35
    - Cross-game: r ≈ 0.0
    """
    n = len(legs)
    R = [[0.0] * n for _ in range(n)]

    for i in range(n):
        R[i][i] = 1.0
        for j in range(i + 1, n):
            r = _estimate_leg_correlation(legs[i], legs[j])
            R[i][j] = r
            R[j][i] = r

    return R


def _estimate_leg_correlation(leg_a: dict, leg_b: dict) -> float:
    """Estimate correlation between two prop legs."""
    team_a = leg_a.get("team", "").upper()
    team_b = leg_b.get("team", "").upper()
    dir_a = leg_a.get("pick", "MORE").upper()
    dir_b = leg_b.get("pick", "MORE").upper()
    stat_a = leg_a.get("stat_type", "").lower()
    stat_b = leg_b.get("stat_type", "").lower()

    if not team_a or not team_b:
        return 0.0

    # Cross-game: near zero correlation
    game_a = leg_a.get("game_id", leg_a.get("opponent", ""))
    game_b = leg_b.get("game_id", leg_b.get("opponent", ""))
    if team_a != team_b and game_a != game_b:
        return 0.0

    # Same team
    if team_a == team_b:
        if dir_a == dir_b:
            # Same team, same direction → moderate positive correlation
            return 0.25
        else:
            # Same team, opposite direction → slight positive
            return 0.10

    # Same game, different teams (opponent matchup)
    # Pitcher K over + opposing hitter stats → correlated
    pitcher_stats = {"pitcher_strikeouts", "strikeouts", "pitching_outs"}
    hitter_stats = {"hits", "total_bases", "home_runs", "rbis", "runs"}

    if stat_a in pitcher_stats and stat_b in hitter_stats:
        # Pitcher dominance → fewer hits for opposing team
        if dir_a == "MORE" and dir_b == "LESS":
            return 0.35
        elif dir_a == "LESS" and dir_b == "MORE":
            return 0.35
        return -0.15

    if stat_b in pitcher_stats and stat_a in hitter_stats:
        if dir_b == "MORE" and dir_a == "LESS":
            return 0.35
        elif dir_b == "LESS" and dir_a == "MORE":
            return 0.35
        return -0.15

    # Same game, different teams, non-pitcher/hitter cross
    return 0.05


def quick_slip_ev(
    win_probs: list[float],
    entry_type: str = "5_flex",
) -> dict:
    """Fast analytical EV calculation assuming independence and no ties.

    Good for quick comparisons. Use simulate_slip_ev for accuracy.
    """
    from scipy.special import comb

    n = len(win_probs)
    payout_table = PAYOUTS.get(entry_type, {})

    # Compute probability of exactly k wins (product over subsets)
    # For speed with independence assumption, use convolution
    # dp[k] = probability of exactly k wins
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for p in win_probs:
        new_dp = [0.0] * (n + 1)
        for k in range(n + 1):
            if dp[k] > 0:
                new_dp[k] += dp[k] * (1 - p)
                if k + 1 <= n:
                    new_dp[k + 1] += dp[k] * p
        dp = new_dp

    ev_payout = sum(payout_table.get(k, 0) * dp[k] for k in range(n + 1))

    return {
        "ev_payout": round(ev_payout, 4),
        "ev_profit": round(ev_payout - 1.0, 4),
        "ev_profit_pct": round((ev_payout - 1.0) * 100, 2),
        "prob_perfect": round(dp[n], 6),
        "prob_zero": round(dp[0], 6),
    }
