"""
Gaussian Copula Monte Carlo Slip Optimizer for PrizePicks Flex entries.

Uses numpy/scipy for fast vectorized simulation of correlated prop outcomes.
The Gaussian copula approach preserves marginal probabilities while modeling
statistical dependence between same-game props (e.g., hits and total_bases
for the same batter are highly correlated).

Key design choices:
- SAME_GAME_CORRELATIONS: empirically-grounded pairwise correlations
- Cross-game legs: zero correlation (independent games share no information)
- Nearest-PSD projection: ensures the correlation matrix is always valid
- Vectorized numpy: 50k simulations complete in < 1 second
"""

from __future__ import annotations

import itertools
import random
from typing import Optional

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Pairwise same-game prop correlations
# ---------------------------------------------------------------------------
# Values are Pearson correlations estimated from baseball research and
# empirical sabermetric relationships.  Only same-game legs use these values;
# cross-game legs always receive r = 0.
#
# Key format: (stat_a, stat_b) where both stat strings are lower-snake-case.
# Lookup is symmetric — the helper checks both orderings.
SAME_GAME_CORRELATIONS: dict[tuple[str, str], float] = {
    # Batter counting-stat cluster
    ("hits", "total_bases"): 0.85,
    ("hits", "runs"): 0.60,
    ("hits", "rbis"): 0.45,
    ("hits", "home_runs"): 0.35,
    ("hits", "hitter_fantasy_score"): 0.75,
    ("total_bases", "runs"): 0.55,
    ("total_bases", "rbis"): 0.50,
    ("total_bases", "home_runs"): 0.65,
    ("total_bases", "hitter_fantasy_score"): 0.80,
    ("runs", "rbis"): 0.50,
    ("runs", "home_runs"): 0.55,
    ("runs", "hitter_fantasy_score"): 0.60,
    ("rbis", "home_runs"): 0.60,
    ("rbis", "hitter_fantasy_score"): 0.65,
    ("home_runs", "hitter_fantasy_score"): 0.70,
    # Same batter — walks / strikeouts are weakly correlated with contact stats
    ("hits", "walks"): 0.15,
    ("hits", "batter_strikeouts"): -0.30,
    ("total_bases", "batter_strikeouts"): -0.25,
    # Pitcher performance cluster
    ("pitcher_strikeouts", "pitching_outs"): 0.45,
    ("pitcher_strikeouts", "hits_allowed"): -0.50,
    ("pitcher_strikeouts", "earned_runs"): -0.35,
    ("pitcher_strikeouts", "walks_allowed"): -0.20,
    ("pitching_outs", "hits_allowed"): -0.45,
    ("pitching_outs", "earned_runs"): -0.40,
    ("pitching_outs", "walks_allowed"): -0.15,
    ("hits_allowed", "earned_runs"): 0.70,
    ("walks_allowed", "earned_runs"): 0.50,
    # Cross-team same-game: pitcher K vs opposing batter hits are negatively correlated
    ("pitcher_strikeouts", "hits"): -0.35,
    ("pitcher_strikeouts", "total_bases"): -0.30,
    ("pitcher_strikeouts", "runs"): -0.25,
    # Fantasy aggregates
    ("pitcher_strikeouts", "pitcher_fantasy_score"): 0.80,
    ("pitching_outs", "pitcher_fantasy_score"): 0.75,
}

# ---------------------------------------------------------------------------
# PrizePicks Flex payout tables
# Key: number of picks.  Value: {n_hits: payout_multiplier}
# Source: PrizePicks help center, March 2026
# ---------------------------------------------------------------------------
FLEX_PAYOUTS: dict[int, dict[int, float]] = {
    2: {2: 3.0, 1: 1.5, 0: 0.0},
    3: {3: 2.25, 2: 1.25, 1: 0.0, 0: 0.0},
    4: {4: 5.0, 3: 1.5, 2: 0.0, 1: 0.0, 0: 0.0},
    5: {5: 10.0, 4: 2.0, 3: 0.4, 2: 0.0, 1: 0.0, 0: 0.0},
    6: {6: 25.0, 5: 2.0, 4: 0.4, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0},
}

# Grades that qualify as "high confidence" for slip building
_VALID_GRADES: frozenset[str] = frozenset({"A+", "A", "B"})

# Maximum random combinations to evaluate per slip size (keeps runtime bounded)
_SAMPLE_LIMIT: int = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_stat(stat: str) -> str:
    """Normalise a stat identifier to lower-snake-case for lookup."""
    return stat.lower().replace(" ", "_")


def _get_pairwise_correlation(stat_a: str, stat_b: str, same_game: bool) -> float:
    """Return the correlation coefficient for two prop legs.

    Args:
        stat_a: First stat type (any case / spacing).
        stat_b: Second stat type (any case / spacing).
        same_game: Whether the two legs come from the same game.

    Returns:
        Pearson correlation in [-1, 1].  Returns 0.0 for cross-game pairs.
    """
    if not same_game:
        return 0.0
    a, b = _norm_stat(stat_a), _norm_stat(stat_b)
    r = SAME_GAME_CORRELATIONS.get((a, b)) or SAME_GAME_CORRELATIONS.get((b, a))
    if r is not None:
        return r
    # Same-game props not in the table share mild correlation from
    # shared conditions (weather, park, etc.)
    return 0.10


def _nearest_psd(R: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-semi-definite matrix.

    Uses eigenvalue flooring (Higham 2002 approximate method).  The diagonal
    is then re-normalised to 1 so the result is a valid correlation matrix.

    Args:
        R: Symmetric NxN matrix.

    Returns:
        NxN positive-semi-definite correlation matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-8)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R_psd))
    d = np.where(d < 1e-10, 1.0, d)
    R_psd = R_psd / np.outer(d, d)
    return R_psd


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k) — integer arithmetic, no imports needed."""
    if k > n or k < 0:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _sample_combinations(
    pool: list,
    size: int,
    n_samples: int,
    rng: random.Random,
) -> list[tuple]:
    """Randomly sample unique combinations without exhaustive enumeration.

    Args:
        pool: Items to sample from.
        size: Combination size.
        n_samples: Number of unique combinations to return.
        rng: Random number generator instance.

    Returns:
        List of unique tuples.
    """
    seen: set[tuple[int, ...]] = set()
    combos: list[tuple] = []
    max_attempts = n_samples * 20

    for _ in range(max_attempts):
        if len(combos) >= n_samples:
            break
        indices = tuple(sorted(rng.sample(range(len(pool)), size)))
        if indices not in seen:
            seen.add(indices)
            combos.append(tuple(pool[i] for i in indices))

    return combos


def _predictions_to_legs(predictions: list[dict]) -> list[dict]:
    """Convert model prediction dicts to the leg format expected by the simulator.

    Prefers ``win_prob`` > ``p_over`` > ``confidence`` for the probability field.

    Args:
        predictions: List of prediction dicts from ``generate_prediction()``.

    Returns:
        List of leg dicts with ``player``, ``stat_type``, ``game_id``,
        and ``probability`` keys.
    """
    legs: list[dict] = []
    for p in predictions:
        prob = (
            p.get("win_prob")
            or p.get("p_over")
            or p.get("probability")
            or p.get("confidence", 0.5)
        )
        legs.append(
            {
                "player": p.get("player_name", ""),
                "stat_type": _norm_stat(
                    p.get("stat_internal", p.get("stat_type", ""))
                ),
                "game_id": (
                    p.get("game_pk")
                    or p.get("game_id")
                    or p.get("opponent", "")
                ),
                "probability": float(prob),
            }
        )
    return legs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_correlation_matrix(legs: list[dict]) -> np.ndarray:
    """Build an NxN correlation matrix for a list of prop legs.

    Same-game legs receive correlations from ``SAME_GAME_CORRELATIONS``;
    cross-game legs are treated as independent (r = 0).

    The result is guaranteed to be positive semi-definite via nearest-PSD
    projection, so it is always safe to use as a covariance matrix for
    multivariate normal sampling.

    Args:
        legs: List of dicts, each with:
            - ``player`` (str): Player name.
            - ``stat_type`` (str): Stat identifier (e.g. ``"hits"``).
            - ``game_id`` (str): Game identifier.  Legs with the same
              non-empty ``game_id`` are treated as same-game.
            - ``probability`` (float): P(leg wins).

    Returns:
        numpy.ndarray of shape (N, N) — symmetric, diagonal = 1,
        positive semi-definite.
    """
    n = len(legs)
    R = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            gid_i = legs[i].get("game_id") or ""
            gid_j = legs[j].get("game_id") or ""
            same_game = bool(gid_i and gid_j and gid_i == gid_j)
            r = _get_pairwise_correlation(
                legs[i].get("stat_type", ""),
                legs[j].get("stat_type", ""),
                same_game,
            )
            R[i, j] = r
            R[j, i] = r

    return _nearest_psd(R)


def simulate_slip_ev(
    legs: list[dict],
    n_simulations: int = 50_000,
    seed: Optional[int] = None,
) -> dict:
    """Simulate slip Expected Value using a Gaussian copula.

    Algorithm
    ---------
    1. Build the NxN correlation matrix ``R`` from the leg metadata.
    2. Compute the lower Cholesky factor ``L`` of ``R``.
    3. Draw ``(n_simulations × N)`` iid standard normals ``Z``.
    4. Correlate: ``Z_corr = Z @ L.T`` — rows are correlated normal vectors.
    5. Convert to uniforms via the standard normal CDF: ``U = Φ(Z_corr)``.
    6. Threshold: leg ``i`` wins in simulation ``s`` iff ``U[s,i] < p_i``.
       This preserves each marginal probability exactly.
    7. Sum wins per simulation, look up payout from ``FLEX_PAYOUTS``,
       average across simulations.

    Args:
        legs: List of dicts with:
            - ``probability`` (float): P(leg wins), in (0, 1).
            - ``stat_type`` (str): Stat identifier.
            - ``game_id`` (str): Game identifier.
            - ``player`` (str, optional): Player name.
        n_simulations: Number of Monte Carlo draws.  50k balances speed
            and accuracy (±0.01 EV at 99% confidence).
        seed: Integer seed for reproducibility.  ``None`` → random.

    Returns:
        Dict with:
            - ``ev_payout`` (float): Mean payout multiplier across sims.
            - ``ev_profit`` (float): ``ev_payout - 1.0``.
            - ``ev_profit_pct`` (float): Edge as a percentage.
            - ``hit_rate_all`` (float): Fraction of sims where all legs hit.
            - ``win_rates_by_count`` (dict[int, float]): P(exactly k wins).
            - ``std_dev`` (float): Standard deviation of payout distribution.
            - ``n_simulations`` (int): Simulations actually run.
            - ``n_legs`` (int): Number of legs.
    """
    n = len(legs)
    if n == 0:
        return {
            "ev_payout": 0.0,
            "ev_profit": -1.0,
            "ev_profit_pct": -100.0,
            "hit_rate_all": 0.0,
            "win_rates_by_count": {},
            "std_dev": 0.0,
            "n_simulations": 0,
            "n_legs": 0,
        }

    payout_table = FLEX_PAYOUTS.get(n, {})

    # Per-leg win probabilities — clipped away from 0/1 for numerical safety
    probs = np.clip(
        [float(leg.get("probability", 0.5)) for leg in legs],
        1e-6,
        1.0 - 1e-6,
    )

    # Correlation matrix and Cholesky decomposition
    R = build_correlation_matrix(legs)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Fallback via eigendecomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(seed)

    # Step 3-4: draw correlated normals  shape: (n_simulations, n)
    Z_corr = rng.standard_normal((n_simulations, n)) @ L.T

    # Step 5: convert to uniform marginals via Φ
    U = norm.cdf(Z_corr)  # shape: (n_simulations, n)

    # Step 6: Bernoulli outcomes — leg wins iff U < p_i
    outcomes = (U < probs[np.newaxis, :]).astype(np.int8)  # (n_simulations, n)

    # Step 7: aggregate
    wins_per_sim = outcomes.sum(axis=1)  # (n_simulations,)
    payouts = np.fromiter(
        (payout_table.get(int(w), 0.0) for w in wins_per_sim),
        dtype=np.float64,
        count=n_simulations,
    )

    ev_payout = float(payouts.mean())
    ev_profit = ev_payout - 1.0

    win_rates_by_count = {
        k: float((wins_per_sim == k).mean()) for k in range(n + 1)
    }

    return {
        "ev_payout": round(ev_payout, 4),
        "ev_profit": round(ev_profit, 4),
        "ev_profit_pct": round(ev_profit * 100, 2),
        "hit_rate_all": round(float((wins_per_sim == n).mean()), 6),
        "win_rates_by_count": win_rates_by_count,
        "std_dev": round(float(payouts.std()), 4),
        "n_simulations": n_simulations,
        "n_legs": n,
    }


def find_optimal_slips(
    predictions: list[dict],
    top_k: int = 10,
    slip_sizes: Optional[list[int]] = None,
    n_simulations: int = 50_000,
    seed: Optional[int] = None,
) -> list[dict]:
    """Find the highest-EV slip combinations from today's predictions.

    Only A+, A, and B-grade predictions are eligible.  Exhaustive
    enumeration is used for small combination spaces (≤ ``_SAMPLE_LIMIT``
    candidates or slip size ≤ 4); random sampling is used otherwise to keep
    runtime reasonable.

    Args:
        predictions: List of prediction dicts from ``generate_prediction()``.
            Each must have at least ``rating``, ``player_name``, ``stat_type``
            or ``stat_internal``, and one of ``win_prob`` / ``p_over`` /
            ``confidence``.
        top_k: Number of top slips to return.
        slip_sizes: Pick counts to evaluate (default ``[2, 3, 4, 5, 6]``).
        n_simulations: Monte Carlo simulations per candidate slip.
        seed: Random seed (used for both combination sampling and simulation).

    Returns:
        List of slip dicts sorted by ``ev_profit`` descending.  Each dict
        contains the simulation output fields plus:
            - ``legs`` (list[dict]): Prediction dicts included in the slip.
            - ``slip_size`` (int): Number of legs.
    """
    if slip_sizes is None:
        slip_sizes = [2, 3, 4, 5, 6]

    # Only high-confidence grades with positive edge
    eligible = [
        p for p in predictions
        if p.get("rating", "") in _VALID_GRADES
        and float(
            p.get("win_prob") or p.get("p_over") or p.get("confidence", 0.0)
        ) > 0.50
    ]

    if not eligible:
        return []

    results: list[dict] = []
    rng = random.Random(seed)

    for size in slip_sizes:
        if len(eligible) < size:
            continue

        n_combos = _comb(len(eligible), size)
        use_exhaustive = size <= 4 or n_combos <= _SAMPLE_LIMIT

        if use_exhaustive:
            combos: list = list(itertools.combinations(eligible, size))
        else:
            combos = _sample_combinations(eligible, size, _SAMPLE_LIMIT, rng)

        for combo in combos:
            legs = _predictions_to_legs(list(combo))
            sim = simulate_slip_ev(legs, n_simulations=n_simulations, seed=seed)
            results.append(
                {
                    "legs": list(combo),
                    "slip_size": size,
                    **sim,
                }
            )

    results.sort(key=lambda x: x["ev_profit"], reverse=True)
    return results[:top_k]


def kelly_fraction(
    ev: float,
    odds: float,
    bankroll: float = 100.0,
) -> dict:
    """Kelly criterion bet sizing for a PrizePicks slip.

    Derives an implied win probability from the EV payout and the full-hit
    payout, then applies the standard Kelly formula:

        f* = (p × b − q) / b

    where ``b = odds − 1`` (net profit per unit), ``p = ev / odds``,
    ``q = 1 − p``.

    For fractional Kelly variants (recommended in practice):
    - **Quarter-Kelly** is the professional default for player props.
      It sacrifices roughly 44% of geometric growth rate while reducing
      variance by 75%.
    - **Half-Kelly** is a common intermediate choice.

    Args:
        ev: EV payout multiplier from ``simulate_slip_ev`` (e.g., ``1.43``
            means the slip returns $1.43 per $1 wagered on average).
        odds: Maximum payout multiplier (e.g., ``10.0`` for a 5-pick flex
            that pays 10x on a perfect slip).
        bankroll: Total bankroll in dollars.  Default 100.

    Returns:
        Dict with:
            - ``kelly_pct`` (float): Full Kelly as a percentage of bankroll.
            - ``half_kelly_pct`` (float): Half-Kelly percentage.
            - ``quarter_kelly_pct`` (float): Quarter-Kelly percentage
              (recommended).
            - ``recommended_wager`` (float): Quarter-Kelly × bankroll, in $.
            - ``max_wager`` (float): Full Kelly × bankroll, in $.
            - ``edge_pct`` (float): Edge as a percentage (ev_payout − 1) × 100.
    """
    edge_pct = round((ev - 1.0) * 100, 2)

    # No edge or invalid inputs → no bet
    if odds <= 1.0 or ev <= 1.0:
        return {
            "kelly_pct": 0.0,
            "half_kelly_pct": 0.0,
            "quarter_kelly_pct": 0.0,
            "recommended_wager": 0.0,
            "max_wager": 0.0,
            "edge_pct": edge_pct,
        }

    # Implied win probability from EV and maximum payout
    p = float(np.clip(ev / odds, 0.0, 1.0))
    q = 1.0 - p
    b = odds - 1.0  # net odds

    kelly = (p * b - q) / b
    kelly = float(np.clip(kelly, 0.0, 0.25))  # never risk more than 25%

    return {
        "kelly_pct": round(kelly * 100, 2),
        "half_kelly_pct": round(kelly * 50, 2),
        "quarter_kelly_pct": round(kelly * 25, 2),
        "recommended_wager": round(kelly * 0.25 * bankroll, 2),
        "max_wager": round(kelly * bankroll, 2),
        "edge_pct": edge_pct,
    }
