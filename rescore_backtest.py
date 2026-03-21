"""
Re-score backtest data with different weight configurations.

The stored projections in backtest_2025.json already include v008 offsets.
To test a different config, we:
  1. Remove v008 offsets (reverse them)
  2. Apply new offsets
  3. Recalculate CDF probabilities using the adjusted projection
  4. Re-pick and grade

This is faster than re-running the full backtest and lets us optimize
variance ratios and other hyperparameters.

Run: python rescore_backtest.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import norm, poisson, nbinom, gamma

# Add src to path so we can import predictor
sys.path.insert(0, str(Path.cwd()))

BACKTEST_PATH = Path("data/backtest/backtest_2025.json")

# v008 offsets that are baked into the stored projections
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
    """Load and filter to actual plays only."""
    if not BACKTEST_PATH.exists():
        print(f"ERROR: {BACKTEST_PATH} not found")
        sys.exit(1)
    with open(BACKTEST_PATH) as f:
        data = json.load(f)

    # Keep only actual plays
    plays = [r for r in data if r.get('actual', 0) > 0]
    print(f"Loaded {len(plays):,} actual plays from backtest")
    return plays


def remove_offset(projection, prop_type, offsets):
    """Reverse an offset to get raw projection."""
    offset = offsets.get(prop_type, 0.0)
    return projection - offset


def apply_offset(projection, prop_type, offsets):
    """Apply an offset to a projection."""
    offset = offsets.get(prop_type, 0.0)
    return projection + offset


def calculate_probabilities(projection, line, prop_type, variance_ratios=None):
    """
    Recalculate CDF probabilities for a given projection and prop type.

    Returns dict with p_over, p_under, pick, confidence, rating.
    """
    if variance_ratios is None:
        variance_ratios = {}

    mu = max(projection, 0.01)

    # Distribution params based on prop type
    negbin_props = {
        "stolen_bases": variance_ratios.get("stolen_bases", 2.5),
    }
    poisson_props = {
        "hits", "total_bases", "rbis", "runs", "walks",
        "pitcher_strikeouts", "earned_runs", "walks_allowed",
        "hits_allowed", "batter_strikeouts", "singles", "doubles",
    }
    gamma_props = {
        "hitter_fantasy_score": variance_ratios.get("hitter_fantasy_score", 1.6),
    }
    normal_props = {
        "pitching_outs": 1.3,
        "hits_runs_rbis": 1.5,
    }

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
        else:
            if line == int(line):
                p_over = 1 - poisson.cdf(int(line), mu)
                p_under = poisson.cdf(int(line) - 1, mu)
            else:
                p_over = 1 - poisson.cdf(int(line), mu)
                p_under = poisson.cdf(int(line), mu)

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

    # Normalize
    total = p_over + p_under
    if total > 0:
        p_over /= total
        p_under /= total
    else:
        p_over = 0.5
        p_under = 0.5

    # Pick and confidence
    edge = abs(p_over - 0.5)
    pick = "MORE" if p_over > 0.5 else "LESS"
    confidence = max(p_over, p_under)

    # Rating (using v008 thresholds)
    if confidence >= 0.70:
        rating = "A"
    elif confidence >= 0.62:
        rating = "B"
    elif confidence >= 0.57:
        rating = "C"
    else:
        rating = "D"

    return {
        "p_over": p_over,
        "p_under": p_under,
        "pick": pick,
        "confidence": confidence,
        "rating": rating,
        "edge": edge,
    }


def rescore_with_config(data, config):
    """
    Rescore backtest with a given config.

    config dict:
      - name: config name
      - offsets: dict of offsets to apply
      - variance_ratios: dict of variance ratios
    """
    results = []

    for record in data:
        # Get stored projection (already has v008 offsets)
        stored_projection = record.get("projection")
        prop_type = record.get("prop_type")
        line = record.get("line")
        actual = record.get("actual")

        # Remove v008 offsets to get raw projection
        raw_projection = remove_offset(stored_projection, prop_type, V008_OFFSETS)

        # Apply new offsets
        new_offsets = config.get("offsets", {})
        new_projection = apply_offset(raw_projection, prop_type, new_offsets)

        # Recalculate probabilities with new variance ratios
        variance_ratios = config.get("variance_ratios", {})
        probs = calculate_probabilities(new_projection, line, prop_type, variance_ratios)

        # Grade
        pick = probs["pick"]
        actual_value = actual
        line_value = line

        if actual_value > line_value and pick == "MORE":
            result = "W"
        elif actual_value <= line_value and pick == "LESS":
            result = "W"
        else:
            result = "L"

        results.append({
            "prop_type": prop_type,
            "pick": pick,
            "confidence": probs["confidence"],
            "rating": probs["rating"],
            "result": result,
            "actual": actual_value,
            "line": line_value,
            "new_projection": new_projection,
        })

    return results


def analyze_results(results, config_name):
    """Analyze rescore results and print report."""
    # Cross-tab: prop type × direction
    buckets = defaultdict(lambda: {"W": 0, "L": 0})
    for r in results:
        key = (r["prop_type"], r["pick"])
        buckets[key][r["result"]] += 1

    # Confidence buckets
    confidence_buckets = defaultdict(lambda: {"W": 0, "L": 0, "count": 0})
    for r in results:
        key = (r["prop_type"], r["pick"])
        conf_level = int(r["confidence"] * 100) // 5 * 5  # Round to nearest 5%
        conf_key = (r["prop_type"], r["pick"], f"{conf_level}-{conf_level+5}%")
        confidence_buckets[conf_key][r["result"]] += 1
        confidence_buckets[conf_key]["count"] += 1

    print(f"\n{'='*80}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*80}")

    print(f"\n{'Prop Type':<25} {'MORE W-L':>12} {'MORE Acc':>10} {'LESS W-L':>12} {'LESS Acc':>10} {'Total Acc':>10}")
    print(f"{'-'*80}")

    prop_types = sorted(set(k[0] for k in buckets.keys()))
    summary = {}

    for prop in prop_types:
        more_bucket = buckets[(prop, "MORE")]
        less_bucket = buckets[(prop, "LESS")]

        more_w, more_l = more_bucket["W"], more_bucket["L"]
        less_w, less_l = less_bucket["W"], less_bucket["L"]

        more_total = more_w + more_l
        less_total = less_w + less_l
        total = more_total + less_total

        more_acc = more_w / more_total if more_total > 0 else 0.0
        less_acc = less_w / less_total if less_total > 0 else 0.0
        total_acc = (more_w + less_w) / total if total > 0 else 0.0

        summary[prop] = {
            "more_acc": more_acc,
            "less_acc": less_acc,
            "total_acc": total_acc,
            "more_count": more_total,
            "less_count": less_total,
        }

        print(f"{prop:<25} {more_w}-{more_l:>9} {more_acc:>10.1%} {less_w}-{less_l:>9} {less_acc:>10.1%} {total_acc:>10.1%}")

    # Confidence calibration
    print(f"\n{'Confidence Calibration (Hits vs Confidence Bin):':}")
    print(f"{'Prop Type':<25} {'Bin':>8} {'Count':>8} {'Win%':>10} {'Expected':>10}")
    print(f"{'-'*60}")

    for (prop, pick, conf_bin) in sorted(confidence_buckets.keys()):
        b = confidence_buckets[(prop, pick, conf_bin)]
        w, l, count = b["W"], b["L"], b["count"]
        if count == 0:
            continue
        acc = w / count if count > 0 else 0.0
        expected = float(conf_bin.split("-")[0]) / 100
        print(f"{prop:<25} {conf_bin:>8} {count:>8} {acc:>10.1%} {expected:>10.1%}")

    return summary


def main():
    plays = load_backtest()

    # Test configs
    configs = [
        {
            "name": "v008_baseline",
            "offsets": V008_OFFSETS,
            "variance_ratios": {"hitter_fantasy_score": 4.0},
        },
        {
            "name": "higher_fs_variance",
            "offsets": V008_OFFSETS,
            "variance_ratios": {"hitter_fantasy_score": 5.0},
        },
        {
            "name": "lower_fs_variance",
            "offsets": V008_OFFSETS,
            "variance_ratios": {"hitter_fantasy_score": 3.0},
        },
        {
            "name": "test_no_offsets",
            "offsets": {},
            "variance_ratios": {"hitter_fantasy_score": 4.0},
        },
        {
            "name": "reduced_offsets_half",
            "offsets": {
                "pitcher_strikeouts": -0.29,
                "hits": 0.24,
                "total_bases": 0.39,
                "hitter_fantasy_score": 0.61,
            },
            "variance_ratios": {"hitter_fantasy_score": 4.0},
        },
    ]

    all_summaries = {}

    for config in configs:
        results = rescore_with_config(plays, config)
        summary = analyze_results(results, config["name"])
        all_summaries[config["name"]] = summary

    # Comparative summary
    print(f"\n{'='*80}")
    print(f"COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Config':<30} {'FS MORE':>12} {'FS LESS':>12} {'TB MORE':>12} {'TB LESS':>12} {'HIT MORE':>12}")
    print(f"{'-'*90}")

    for config_name, summary in all_summaries.items():
        fs_more = summary.get("hitter_fantasy_score", {}).get("more_acc", 0.0)
        fs_less = summary.get("hitter_fantasy_score", {}).get("less_acc", 0.0)
        tb_more = summary.get("total_bases", {}).get("more_acc", 0.0)
        tb_less = summary.get("total_bases", {}).get("less_acc", 0.0)
        hit_more = summary.get("hits", {}).get("more_acc", 0.0)

        print(f"{config_name:<30} {fs_more:>12.1%} {fs_less:>12.1%} {tb_more:>12.1%} {tb_less:>12.1%} {hit_more:>12.1%}")


if __name__ == "__main__":
    main()
