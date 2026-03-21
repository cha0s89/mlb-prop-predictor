"""
Test different variance ratios to optimize accuracy.

Focus areas:
1. Hitter Fantasy Score variance (currently 4.0)
2. Pitcher Strikeouts variance (currently 1.0 Poisson)
3. Total Bases variance (currently 1.0 Poisson)

Run: python test_variance_ratios.py
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import norm, poisson, nbinom, gamma

BACKTEST_PATH = Path("data/backtest/backtest_2025.json")

V008_OFFSETS = {
    "pitcher_strikeouts": -0.58,
    "batter_strikeouts": 0.0,
    "hits": 0.48,
    "total_bases": 0.78,
    "home_runs": 0.0,
    "rbis": 0.0,
    "runs": 0.0,
    "stolen_bases": 0.0,
    "hitter_fantasy_score": 1.23,
    "hits_runs_rbis": 0.0,
    "pitching_outs": 0.0,
    "earned_runs": 0.0,
    "walks_allowed": 0.0,
    "walks": 0.0,
    "hits_allowed": 0.0,
    "singles": 0.0,
    "doubles": 0.0,
}


def load_backtest():
    """Load actual plays."""
    with open(BACKTEST_PATH) as f:
        data = json.load(f)
    plays = [r for r in data if r.get('actual', 0) > 0]
    print(f"Loaded {len(plays):,} actual plays")
    return plays


def apply_offset(projection, prop_type, offsets):
    offset = offsets.get(prop_type, 0.0)
    return projection + offset


def remove_offset(projection, prop_type, offsets):
    offset = offsets.get(prop_type, 0.0)
    return projection - offset


def calculate_probabilities(projection, line, prop_type, variance_ratios=None):
    """Recalculate CDF probabilities."""
    if variance_ratios is None:
        variance_ratios = {}

    mu = max(projection, 0.01)

    negbin_props = {"stolen_bases": variance_ratios.get("stolen_bases", 2.5)}
    poisson_props = {
        "hits", "total_bases", "rbis", "runs", "walks",
        "pitcher_strikeouts", "earned_runs", "walks_allowed",
        "hits_allowed", "batter_strikeouts", "singles", "doubles",
    }
    gamma_props = {
        "hitter_fantasy_score": variance_ratios.get("hitter_fantasy_score", 1.6),
    }
    normal_props = {"pitching_outs": 1.3, "hits_runs_rbis": 1.5}

    p_over = 0.5
    p_under = 0.5

    if prop_type in negbin_props:
        r = negbin_props[prop_type]
        var = mu * r
        if var > mu and mu > 0:
            n_param = (mu ** 2) / (var - mu)
            p_param = mu / var
            if line == int(line):
                int_line = int(line)
                p_over = 1 - nbinom.cdf(int_line, n_param, p_param)
                p_under = nbinom.cdf(int_line - 1, n_param, p_param)
            else:
                int_line = int(line)
                p_over = 1 - nbinom.cdf(int_line, n_param, p_param)
                p_under = nbinom.cdf(int_line, n_param, p_param)

    elif prop_type in poisson_props:
        if line == int(line):
            int_line = int(line)
            p_over = 1 - poisson.cdf(int_line, mu)
            p_under = poisson.cdf(int_line - 1, mu)
        else:
            int_line = int(line)
            p_over = 1 - poisson.cdf(int_line, mu)
            p_under = poisson.cdf(int_line, mu)

    elif prop_type in gamma_props:
        var_ratio = gamma_props[prop_type]
        var = mu * var_ratio
        shape = (mu ** 2) / var
        scale = var / mu
        if line == int(line):
            p_over = 1 - gamma.cdf(line + 0.5, shape, scale=scale)
            p_under = gamma.cdf(line - 0.5, shape, scale=scale)
        else:
            p_over = 1 - gamma.cdf(line, shape, scale=scale)
            p_under = gamma.cdf(line, shape, scale=scale)

    else:
        var_ratio = normal_props.get(prop_type, 1.5)
        sigma = max(np.sqrt(mu * var_ratio), 0.25)
        if line == int(line):
            p_over = 1 - norm.cdf(line + 0.5, loc=mu, scale=sigma)
            p_under = norm.cdf(line - 0.5, loc=mu, scale=sigma)
        else:
            p_over = 1 - norm.cdf(line, loc=mu, scale=sigma)
            p_under = norm.cdf(line, loc=mu, scale=sigma)

    total = p_over + p_under
    if total > 0:
        p_over /= total
        p_under /= total
    else:
        p_over = 0.5
        p_under = 0.5

    edge = abs(p_over - 0.5)
    pick = "MORE" if p_over > 0.5 else "LESS"
    confidence = max(p_over, p_under)

    if confidence >= 0.70:
        rating = "A"
    elif confidence >= 0.62:
        rating = "B"
    elif confidence >= 0.57:
        rating = "C"
    else:
        rating = "D"

    return {"p_over": p_over, "p_under": p_under, "pick": pick, "confidence": confidence, "rating": rating}


def rescore_with_config(data, variance_ratios):
    """Rescore with given variance ratios."""
    results = defaultdict(lambda: {"W": 0, "L": 0})

    for record in data:
        stored_projection = record.get("projection")
        prop_type = record.get("prop_type")
        line = record.get("line")
        actual = record.get("actual")

        # Remove v008 offsets
        raw_projection = remove_offset(stored_projection, prop_type, V008_OFFSETS)
        # Apply v008 offsets back
        new_projection = apply_offset(raw_projection, prop_type, V008_OFFSETS)

        # Recalculate probabilities with new variance ratios
        probs = calculate_probabilities(new_projection, line, prop_type, variance_ratios)

        # Grade
        pick = probs["pick"]
        if actual > line and pick == "MORE":
            result = "W"
        elif actual <= line and pick == "LESS":
            result = "W"
        else:
            result = "L"

        key = (prop_type, pick)
        results[key][result] += 1

    return results


def main():
    plays = load_backtest()

    print(f"\n{'='*100}")
    print("TESTING FANTASY SCORE VARIANCE RATIOS")
    print(f"{'='*100}")
    print(f"{'FS Variance':<15} {'FS MORE Acc':>15} {'FS LESS Acc':>15} {'Total FS Acc':>15} {'TB MORE':>15} {'PK Ks Acc':>15}")
    print(f"{'-'*100}")

    fs_variances = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    for fs_var in fs_variances:
        variance_ratios = {"hitter_fantasy_score": fs_var}
        results = rescore_with_config(plays, variance_ratios)

        fs_more = results[("hitter_fantasy_score", "MORE")]
        fs_less = results[("hitter_fantasy_score", "LESS")]
        tb_more = results[("total_bases", "MORE")]
        pk_ks_all = results[("pitcher_strikeouts", "MORE")]
        pk_ks_all["L"] += results[("pitcher_strikeouts", "LESS")]["L"]
        pk_ks_all["W"] += results[("pitcher_strikeouts", "LESS")]["W"]

        fs_more_w = fs_more["W"]
        fs_more_total = fs_more["W"] + fs_more["L"]
        fs_more_acc = fs_more_w / fs_more_total if fs_more_total > 0 else 0.0

        fs_less_w = fs_less["W"]
        fs_less_total = fs_less["W"] + fs_less["L"]
        fs_less_acc = fs_less_w / fs_less_total if fs_less_total > 0 else 0.0

        fs_total_acc = (fs_more_w + fs_less_w) / (fs_more_total + fs_less_total)

        tb_more_w = tb_more["W"]
        tb_more_total = tb_more["W"] + tb_more["L"]
        tb_more_acc = tb_more_w / tb_more_total if tb_more_total > 0 else 0.0

        pk_ks_w = pk_ks_all["W"]
        pk_ks_total = pk_ks_all["W"] + pk_ks_all["L"]
        pk_ks_acc = pk_ks_w / pk_ks_total if pk_ks_total > 0 else 0.0

        print(f"{fs_var:<15.1f} {fs_more_acc:>15.1%} {fs_less_acc:>15.1%} {fs_total_acc:>15.1%} {tb_more_acc:>15.1%} {pk_ks_acc:>15.1%}")

    # Test pitcher strikeout variance
    print(f"\n{'='*100}")
    print("TESTING PITCHER STRIKEOUT VARIANCE (negbin)")
    print(f"{'='*100}")
    print(f"{'PK K var':<15} {'PK K MORE':>15} {'PK K LESS':>15} {'PK K Total':>15} {'FS MORE':>15} {'FS LESS':>15}")
    print(f"{'-'*100}")

    pk_variances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

    for pk_var in pk_variances:
        variance_ratios = {"hitter_fantasy_score": 4.0, "stolen_bases": pk_var}
        results = rescore_with_config(plays, variance_ratios)

        pk_more = results[("pitcher_strikeouts", "MORE")]
        pk_less = results[("pitcher_strikeouts", "LESS")]
        fs_more = results[("hitter_fantasy_score", "MORE")]
        fs_less = results[("hitter_fantasy_score", "LESS")]

        pk_more_acc = pk_more["W"] / (pk_more["W"] + pk_more["L"]) if (pk_more["W"] + pk_more["L"]) > 0 else 0.0
        pk_less_acc = pk_less["W"] / (pk_less["W"] + pk_less["L"]) if (pk_less["W"] + pk_less["L"]) > 0 else 0.0
        pk_total_acc = (pk_more["W"] + pk_less["W"]) / (pk_more["W"] + pk_more["L"] + pk_less["W"] + pk_less["L"])

        fs_more_acc = fs_more["W"] / (fs_more["W"] + fs_more["L"]) if (fs_more["W"] + fs_more["L"]) > 0 else 0.0
        fs_less_acc = fs_less["W"] / (fs_less["W"] + fs_less["L"]) if (fs_less["W"] + fs_less["L"]) > 0 else 0.0

        print(f"{pk_var:<15.1f} {pk_more_acc:>15.1%} {pk_less_acc:>15.1%} {pk_total_acc:>15.1%} {fs_more_acc:>15.1%} {fs_less_acc:>15.1%}")


if __name__ == "__main__":
    main()
