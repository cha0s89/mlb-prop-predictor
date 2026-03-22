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


# === MLE FITTING FROM GAME LOGS ===

def fit_betabinom_mle(strikeouts: np.ndarray, batters_faced: np.ndarray) -> tuple[float, float]:
    """MLE fit of Beta-Binomial alpha, beta from game-by-game K data.

    Args:
        strikeouts: array of K counts per game
        batters_faced: array of BF per game
    Returns:
        (alpha, beta) tuple
    """
    from scipy.optimize import minimize

    def neg_loglik(params):
        a, b = params
        if a <= 0.01 or b <= 0.01:
            return 1e10
        try:
            ll = betabinom.logpmf(strikeouts, batters_faced, a, b).sum()
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    # Initial guess from method of moments
    k_rates = strikeouts / np.maximum(batters_faced, 1)
    p_hat = np.mean(k_rates)
    var_hat = np.var(k_rates)
    if var_hat > 0 and p_hat > 0:
        s_hat = max(2.0, p_hat * (1 - p_hat) / var_hat - 1)
    else:
        s_hat = 20.0

    result = minimize(neg_loglik, [p_hat * s_hat, (1 - p_hat) * s_hat],
                      method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-6})

    alpha, beta = max(0.1, result.x[0]), max(0.1, result.x[1])
    return alpha, beta


def fit_negbinom_mle(counts: np.ndarray) -> tuple[float, float]:
    """MLE fit of Negative Binomial from count data (e.g., earned runs per game).

    Args:
        counts: array of count data per game
    Returns:
        (n, p) tuple for scipy.stats.nbinom
    """
    from scipy.optimize import minimize

    def neg_loglik(params):
        n, p = params
        if n <= 0.01 or p <= 0.001 or p >= 0.999:
            return 1e10
        try:
            ll = nbinom.logpmf(counts.astype(int), n, p).sum()
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    mean = np.mean(counts)
    var = np.var(counts)
    if var > mean and mean > 0:
        p_init = mean / var
        n_init = mean * p_init / (1 - p_init)
    else:
        p_init = 0.5
        n_init = mean if mean > 0 else 1.0

    result = minimize(neg_loglik, [max(0.5, n_init), min(0.95, max(0.05, p_init))],
                      method='Nelder-Mead',
                      options={'maxiter': 1000})

    n, p = max(0.1, result.x[0]), min(0.999, max(0.001, result.x[1]))
    return n, p
