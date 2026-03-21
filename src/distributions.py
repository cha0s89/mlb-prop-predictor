"""
Statistical distributions for MLB prop modeling.
Beta-Binomial for strikeouts, Negative Binomial for runs.
"""
import numpy as np
from scipy.stats import betabinom, nbinom
from scipy.special import comb

# === BETA-BINOMIAL (Pitcher Strikeouts) ===

def betabinom_params(k_rate: float, precision: float = 30.0) -> tuple[float, float]:
    """Convert a pitcher K-rate into Beta-Binomial alpha, beta.

    Args:
        k_rate: Strikeout rate (K%) as decimal (e.g. 0.25 for 25%)
        precision: s = alpha + beta. Higher = less game-to-game variance.
                   Elite stable pitchers: 40-50, volatile pitchers: 10-20, default ~30.
    Returns:
        (alpha, beta) tuple
    """
    alpha = k_rate * precision
    beta = (1 - k_rate) * precision
    return alpha, beta


def prob_over_betabinom(line: float, n_batters: int, alpha: float, beta: float) -> float:
    """P(K >= ceil(line)) using Beta-Binomial CDF."""
    threshold = int(np.ceil(line))
    if threshold <= 0:
        return 1.0
    if threshold > n_batters:
        return 0.0
    return float(1 - betabinom.cdf(threshold - 1, n_batters, alpha, beta))


def prob_under_betabinom(line: float, n_batters: int, alpha: float, beta: float) -> float:
    """P(K <= floor(line)) using Beta-Binomial CDF."""
    threshold = int(np.floor(line))
    if threshold < 0:
        return 0.0
    if threshold >= n_batters:
        return 1.0
    return float(betabinom.cdf(threshold, n_batters, alpha, beta))


def betabinom_mean_var(n_batters: int, alpha: float, beta: float) -> tuple[float, float]:
    """Return (mean, variance) of the Beta-Binomial distribution."""
    s = alpha + beta
    mean = n_batters * alpha / s
    var = n_batters * alpha * beta * (s + n_batters) / (s**2 * (s + 1))
    return mean, var


def estimate_batters_faced(expected_ip: float, k_rate: float = 0.22) -> int:
    """Estimate batters faced from expected innings pitched.

    Rough formula: BF ≈ IP * 3 + hits + walks per inning.
    A simpler approximation: BF ≈ IP * 4.3 (league avg ~4.3 BF/IP).
    Adjust up for high-K pitchers (more pitches = slightly fewer BF actually).
    """
    bf_per_ip = 4.3 - (k_rate - 0.22) * 2.0  # High-K pitchers face slightly fewer per IP
    bf_per_ip = max(3.5, min(5.0, bf_per_ip))
    return max(15, int(round(expected_ip * bf_per_ip)))


def pitcher_k_precision(gs: int, k_rate_stability: float = 0.56) -> float:
    """Estimate precision parameter s based on starts and K-rate stability.

    K% has year-to-year R² of 0.56 — most stable pitching stat.
    More starts = higher confidence = higher precision.
    Range: 10 (very uncertain) to 50 (very stable).
    """
    # Base precision from sample size
    base = min(50.0, max(10.0, gs * 1.5))
    # Adjust by stability factor
    return base * (0.5 + 0.5 * k_rate_stability)


# === NEGATIVE BINOMIAL (Earned Runs) ===

def negbinom_params(mean_runs: float, overdispersion: float = 2.0) -> tuple[float, float]:
    """Convert mean runs and overdispersion ratio into NB parameters.

    MLB runs show overdispersion ratios of ~2.0-2.5.

    Args:
        mean_runs: Expected runs
        overdispersion: variance/mean ratio (>1 means overdispersed)
    Returns:
        (n, p) for scipy.stats.nbinom
    """
    if overdispersion <= 1.0:
        overdispersion = 1.01
    variance = mean_runs * overdispersion
    # NB parameterization: mean = n*p/(1-p), var = n*p/(1-p)^2
    # So: n = mean^2 / (var - mean), p = mean / var
    p = mean_runs / variance  # probability of success
    n = mean_runs * p / (1 - p)  # number of successes
    return max(0.5, n), min(0.999, max(0.001, p))


def prob_over_negbinom(line: float, n: float, p: float) -> float:
    """P(runs >= ceil(line)) using Negative Binomial."""
    threshold = int(np.ceil(line))
    if threshold <= 0:
        return 1.0
    return float(1 - nbinom.cdf(threshold - 1, n, p))


def prob_under_negbinom(line: float, n: float, p: float) -> float:
    """P(runs <= floor(line)) using Negative Binomial."""
    threshold = int(np.floor(line))
    if threshold < 0:
        return 0.0
    return float(nbinom.cdf(threshold, n, p))


# === BAYESIAN STABILIZATION ===

# Stabilization points (PA/BF needed for 50% weight on observed data)
# Source: research compilation from multiple baseball analytics studies
STABILIZATION = {
    # Batter stats (in PA unless noted)
    "batter_k_rate": 60,
    "batter_bb_rate": 120,
    "batter_hr_rate": 170,
    "batter_avg": 910,      # AB
    "batter_babip": 820,    # BIP
    "batter_iso": 160,
    "batter_woba": 200,
    # Pitcher stats (in BF unless noted)
    "pitcher_k_rate": 70,
    "pitcher_bb_rate": 170,
    "pitcher_hr_rate": 1320,
    "pitcher_babip": 2000,   # BIP
    "pitcher_era": 500,      # ~80 IP worth
    "pitcher_fip": 200,
}


def bayesian_stabilize(observed: float, league_mean: float,
                        sample_size: int, stat_key: str) -> float:
    """Regress observed stat toward league mean based on sample size.

    Formula: true_talent = weight * observed + (1-weight) * league_mean
    where weight = sample_size / (sample_size + stabilization_point)

    Args:
        observed: Player's observed rate/stat
        league_mean: League average for this stat
        sample_size: PA, BF, AB, etc depending on stat
        stat_key: Key into STABILIZATION dict
    Returns:
        Regressed estimate of true talent
    """
    stab = STABILIZATION.get(stat_key, 200)  # default 200 if unknown
    weight = sample_size / (sample_size + stab)
    return weight * observed + (1 - weight) * league_mean


# === PUSH PROBABILITY ===

def prob_push(line: float, n_batters: int, alpha: float, beta: float) -> float:
    """P(stat == line) for integer-valued stats. Important for PrizePicks tie handling."""
    if line != int(line):
        return 0.0  # Half-lines can't push
    k = int(line)
    if k < 0 or k > n_batters:
        return 0.0
    return float(betabinom.pmf(k, n_batters, alpha, beta))
