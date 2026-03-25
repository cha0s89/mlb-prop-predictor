"""
Statistical distributions for MLB prop modeling.
Single source of truth for all probability calculations.

Distribution types:
  - Beta-Binomial: pitcher strikeouts (bounded by batters faced)
  - Negative Binomial: all count props (hits, TB, SB, ER, walks, batter Ks, hits allowed, etc.)
  - Gamma: continuous scores (fantasy score)
  - Normal: approximately normal props (pitching outs, H+R+RBI)
  - Poisson: legacy fallback only (all count props now use NegBin)
"""
import numpy as np
from scipy.stats import betabinom, nbinom, poisson, norm, gamma
from scipy.special import comb


def _is_integer_line(line: float) -> bool:
    """Return True when a betting line is effectively an integer."""
    try:
        return float(line).is_integer()
    except (TypeError, ValueError):
        return False


def _strict_over_threshold(line: float) -> int:
    """Return the first discrete outcome that grades MORE as a win."""
    value = float(line)
    return int(value) + 1 if _is_integer_line(value) else int(np.ceil(value))


def _strict_under_threshold(line: float) -> int:
    """Return the last discrete outcome that grades LESS as a win."""
    value = float(line)
    return int(value) - 1 if _is_integer_line(value) else int(np.floor(value))

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
    """P(K > line) using Beta-Binomial CDF with PrizePicks semantics."""
    threshold = _strict_over_threshold(line)
    if threshold <= 0:
        return 1.0
    if threshold > n_batters:
        return 0.0
    return float(1 - betabinom.cdf(threshold - 1, n_batters, alpha, beta))


def prob_under_betabinom(line: float, n_batters: int, alpha: float, beta: float) -> float:
    """P(K < line) using Beta-Binomial CDF with PrizePicks semantics."""
    threshold = _strict_under_threshold(line)
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
    """P(runs > line) using Negative Binomial with PrizePicks semantics."""
    threshold = _strict_over_threshold(line)
    if threshold <= 0:
        return 1.0
    return float(1 - nbinom.cdf(threshold - 1, n, p))


def prob_under_negbinom(line: float, n: float, p: float) -> float:
    """P(runs < line) using Negative Binomial with PrizePicks semantics."""
    threshold = _strict_under_threshold(line)
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


# === POISSON (Walks, Batter Ks, Hits Allowed) ===

def prob_over_poisson(line: float, mu: float) -> float:
    """P(X > line) using Poisson CDF with PrizePicks semantics."""
    if mu <= 0:
        return 0.0
    threshold = _strict_over_threshold(line)
    if threshold <= 0:
        return 1.0
    return float(1 - poisson.cdf(threshold - 1, mu))


def prob_under_poisson(line: float, mu: float) -> float:
    """P(X < line) using Poisson CDF with PrizePicks semantics."""
    if mu <= 0:
        return 1.0
    threshold = _strict_under_threshold(line)
    if threshold < 0:
        return 0.0
    return float(poisson.cdf(threshold, mu))


def prob_push_poisson(line: float, mu: float) -> float:
    """P(X == line) for Poisson. Only non-zero for integer lines."""
    if line != int(line) or mu <= 0:
        return 0.0
    return float(poisson.pmf(int(line), mu))


# === NEGATIVE BINOMIAL (general — from mu + variance ratio) ===

def negbin_from_mu(mu: float, var_ratio: float) -> tuple:
    """Convert mean + variance ratio to scipy NB params (n, p).

    Args:
        mu: expected value
        var_ratio: variance/mean ratio (>1 = overdispersed)
    Returns:
        (n_param, p_param) for scipy.stats.nbinom
        Returns None if not overdispersed (use Poisson instead)
    """
    if var_ratio <= 1.0 or mu <= 0:
        return None  # Not overdispersed — use Poisson
    var = mu * var_ratio
    n_param = (mu ** 2) / (var - mu)
    p_param = mu / var
    return (max(0.5, n_param), min(0.999, max(0.001, p_param)))


def prob_over_negbin_mu(line: float, mu: float, var_ratio: float) -> float:
    """P(X >= ceil(line)) using NegBin from mu + variance ratio.
    Falls back to Poisson if not overdispersed."""
    params = negbin_from_mu(mu, var_ratio)
    if params is None:
        return prob_over_poisson(line, mu)
    return prob_over_negbinom(line, params[0], params[1])


def prob_under_negbin_mu(line: float, mu: float, var_ratio: float) -> float:
    """P(X <= floor(line)) using NegBin from mu + variance ratio.
    Falls back to Poisson if not overdispersed."""
    params = negbin_from_mu(mu, var_ratio)
    if params is None:
        return prob_under_poisson(line, mu)
    return prob_under_negbinom(line, params[0], params[1])


def prob_push_negbin_mu(line: float, mu: float, var_ratio: float) -> float:
    """P(X == line) for NegBin from mu + variance ratio."""
    if line != int(line):
        return 0.0
    params = negbin_from_mu(mu, var_ratio)
    if params is None:
        return prob_push_poisson(line, mu)
    return float(nbinom.pmf(int(line), params[0], params[1]))


# === GAMMA (Fantasy Score — continuous, overdispersed) ===

def prob_over_gamma(line: float, mu: float, var_ratio: float) -> float:
    """P(X > line) for continuous positive props modeled with Gamma."""
    if mu <= 0:
        return 0.0
    shape, scale = gamma_shape_scale(mu, var_ratio)
    return float(gamma.sf(line, shape, scale=scale))


def prob_under_gamma(line: float, mu: float, var_ratio: float) -> float:
    """P(X < line) for continuous positive props modeled with Gamma."""
    if mu <= 0:
        return 1.0
    shape, scale = gamma_shape_scale(mu, var_ratio)
    return float(gamma.cdf(line, shape, scale=scale))


# === NORMAL (Pitching Outs, H+R+RBI — approximately normal) ===

def prob_over_normal(line: float, mu: float, var_ratio: float) -> float:
    """P(X >= line) using Normal CDF with continuity correction."""
    sigma = normal_sigma(mu, var_ratio)
    return float(1 - norm.cdf(line + 0.5, loc=mu, scale=sigma))


def prob_under_normal(line: float, mu: float, var_ratio: float) -> float:
    """P(X <= line) using Normal CDF with continuity correction."""
    sigma = normal_sigma(mu, var_ratio)
    return float(norm.cdf(line - 0.5, loc=mu, scale=sigma))


def gamma_shape_scale(mu: float, var_ratio: float) -> tuple[float, float]:
    """Return shape/scale parameters for a Gamma distribution."""
    var = max(mu * var_ratio, 0.01)
    shape = (mu ** 2) / var
    scale = var / max(mu, 0.001)
    return shape, scale


def normal_sigma(mu: float, var_ratio: float) -> float:
    """Return sigma for approximately normal props."""
    return max(np.sqrt(max(mu, 0.01) * var_ratio), 0.25)


def is_discrete_distribution(dist_type: str) -> bool:
    """Return True when the routed distribution is discrete."""
    return dist_type in {"betabinom", "negbin", "poisson", "binary"}


def _distribution_cdf(value: float, mu: float, dist_type: str,
                      var_ratio: float = 1.5, phi: float = 25,
                      n_batters: int = None,
                      bb_alpha: float = None, bb_beta: float = None) -> float:
    """Internal CDF helper used by percentile/tail calculations."""
    if dist_type == "betabinom":
        if bb_alpha is not None and bb_beta is not None and n_batters:
            return float(betabinom.cdf(value, n_batters, bb_alpha, bb_beta))
        if n_batters and n_batters > 0 and mu > 0:
            k_rate = mu / n_batters
            alpha, beta = betabinom_params(k_rate, phi)
            return float(betabinom.cdf(value, n_batters, alpha, beta))
        return float(poisson.cdf(value, mu))

    if dist_type == "negbin":
        params = negbin_from_mu(mu, var_ratio)
        if params is None:
            return float(poisson.cdf(value, mu))
        return float(nbinom.cdf(value, params[0], params[1]))

    if dist_type == "poisson":
        return float(poisson.cdf(value, mu))

    if dist_type == "gamma":
        shape, scale = gamma_shape_scale(mu, var_ratio)
        return float(gamma.cdf(value, shape, scale=scale))

    if dist_type == "normal":
        sigma = normal_sigma(mu, var_ratio)
        return float(norm.cdf(value, loc=mu, scale=sigma))

    if dist_type == "binary":
        p_one = min(1.0, max(0.0, mu))
        if value < 0:
            return 0.0
        if value < 1:
            return 1.0 - p_one
        return 1.0

    return float(poisson.cdf(value, mu))


def distribution_quantile(prob: float, mu: float, dist_type: str,
                          var_ratio: float = 1.5, phi: float = 25,
                          n_batters: int = None,
                          bb_alpha: float = None, bb_beta: float = None) -> float:
    """Return a percentile/quantile for the configured distribution."""
    q = min(max(float(prob), 0.0), 1.0)

    if dist_type == "betabinom":
        if bb_alpha is not None and bb_beta is not None and n_batters:
            return float(betabinom.ppf(q, n_batters, bb_alpha, bb_beta))
        if n_batters and n_batters > 0 and mu > 0:
            k_rate = mu / n_batters
            alpha, beta = betabinom_params(k_rate, phi)
            return float(betabinom.ppf(q, n_batters, alpha, beta))
        return float(poisson.ppf(q, mu))

    if dist_type == "negbin":
        params = negbin_from_mu(mu, var_ratio)
        if params is None:
            return float(poisson.ppf(q, mu))
        return float(nbinom.ppf(q, params[0], params[1]))

    if dist_type == "poisson":
        return float(poisson.ppf(q, mu))

    if dist_type == "gamma":
        shape, scale = gamma_shape_scale(mu, var_ratio)
        return float(gamma.ppf(q, shape, scale=scale))

    if dist_type == "normal":
        sigma = normal_sigma(mu, var_ratio)
        return float(norm.ppf(q, loc=mu, scale=sigma))

    if dist_type == "binary":
        p_one = min(1.0, max(0.0, mu))
        return 0.0 if q <= (1.0 - p_one) else 1.0

    return float(poisson.ppf(q, mu))


def prob_at_least(threshold: float, mu: float, dist_type: str,
                  var_ratio: float = 1.5, phi: float = 25,
                  n_batters: int = None,
                  bb_alpha: float = None, bb_beta: float = None) -> float:
    """Return P(X >= threshold) with inclusive semantics."""
    if is_discrete_distribution(dist_type):
        cutoff = np.ceil(threshold) - 1
        return float(1.0 - _distribution_cdf(cutoff, mu, dist_type, var_ratio, phi, n_batters, bb_alpha, bb_beta))
    return float(1.0 - _distribution_cdf(threshold, mu, dist_type, var_ratio, phi, n_batters, bb_alpha, bb_beta))


def prob_at_most(threshold: float, mu: float, dist_type: str,
                 var_ratio: float = 1.5, phi: float = 25,
                 n_batters: int = None,
                 bb_alpha: float = None, bb_beta: float = None) -> float:
    """Return P(X <= threshold) with inclusive semantics."""
    if is_discrete_distribution(dist_type):
        cutoff = np.floor(threshold)
        return float(_distribution_cdf(cutoff, mu, dist_type, var_ratio, phi, n_batters, bb_alpha, bb_beta))
    return float(_distribution_cdf(threshold, mu, dist_type, var_ratio, phi, n_batters, bb_alpha, bb_beta))


# === UNIFIED PROBABILITY ROUTER ===

def compute_probabilities(line: float, mu: float, dist_type: str,
                          var_ratio: float = 1.5, phi: float = 25,
                          n_batters: int = None,
                          bb_alpha: float = None, bb_beta: float = None) -> dict:
    """Single entry point for all probability calculations.

    Routes to the correct distribution based on dist_type. Returns
    p_over, p_under, p_push for any prop type.

    Args:
        line: PrizePicks line value
        mu: projected stat value
        dist_type: one of 'betabinom', 'negbin', 'poisson', 'gamma', 'normal', 'binary'
        var_ratio: variance/mean ratio (for negbin, gamma, normal)
        phi: precision parameter (for betabinom)
        n_batters: number of batters faced (for betabinom)
        bb_alpha, bb_beta: pre-computed Beta-Binomial params (override phi)

    Returns:
        dict with p_over, p_under, p_push
    """
    p_push = 0.0

    if dist_type == "betabinom":
        if bb_alpha is not None and bb_beta is not None and n_batters:
            p_over = prob_over_betabinom(line, n_batters, bb_alpha, bb_beta)
            p_under = prob_under_betabinom(line, n_batters, bb_alpha, bb_beta)
            p_push = prob_push(line, n_batters, bb_alpha, bb_beta)
        else:
            # Fallback: derive alpha/beta from mu and phi
            if n_batters and n_batters > 0 and mu > 0:
                k_rate = mu / n_batters
                alpha, beta = betabinom_params(k_rate, phi)
                p_over = prob_over_betabinom(line, n_batters, alpha, beta)
                p_under = prob_under_betabinom(line, n_batters, alpha, beta)
                p_push = prob_push(line, n_batters, alpha, beta)
            else:
                p_over = prob_over_poisson(line, mu)
                p_under = prob_under_poisson(line, mu)
                p_push = prob_push_poisson(line, mu)

    elif dist_type == "negbin":
        p_over = prob_over_negbin_mu(line, mu, var_ratio)
        p_under = prob_under_negbin_mu(line, mu, var_ratio)
        p_push = prob_push_negbin_mu(line, mu, var_ratio)

    elif dist_type == "poisson":
        p_over = prob_over_poisson(line, mu)
        p_under = prob_under_poisson(line, mu)
        p_push = prob_push_poisson(line, mu)

    elif dist_type == "gamma":
        p_over = prob_over_gamma(line, mu, var_ratio)
        p_under = prob_under_gamma(line, mu, var_ratio)
        # Continuous distribution — no push at exact point

    elif dist_type == "normal":
        p_over = prob_over_normal(line, mu, var_ratio)
        p_under = prob_under_normal(line, mu, var_ratio)
        # Continuous distribution — no push at exact point

    elif dist_type == "binary":
        # For home runs: mu is P(1+ HR), must be in [0, 1]
        p_over = min(1.0, max(0.0, mu))
        p_under = 1.0 - p_over

    else:
        # Default fallback: NegBin (Poisson functions used as conservative fallback)
        p_over = prob_over_poisson(line, mu)
        p_under = prob_under_poisson(line, mu)
        p_push = prob_push_poisson(line, mu)

    return {"p_over": p_over, "p_under": p_under, "p_push": p_push}


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
