#!/usr/bin/env python3
"""
Detailed analysis of the backtest results to pinpoint projection biases.
"""

import json
from collections import Counter

BACKTEST_PATH = "data/backtest/backtest_2025.json"

def analyze():
    with open(BACKTEST_PATH) as f:
        data = json.load(f)

    print("=" * 80)
    print("BACKTEST BIAS ANALYSIS")
    print("=" * 80)

    for prop_type in ['hitter_fantasy_score', 'pitcher_strikeouts', 'total_bases', 'hits']:
        subset = [r for r in data if r.get('prop_type') == prop_type]
        if not subset:
            continue

        print(f"\n{prop_type.upper()}")
        print("-" * 80)

        # Separate played vs non-played
        played = [r for r in subset if r['actual'] > 0]
        non_played = [r for r in subset if r['actual'] == 0]

        print(f"Records: {len(subset):,} total")
        print(f"  Played (actual > 0): {len(played):,} ({len(played)/len(subset)*100:.1f}%)")
        print(f"  Non-played (actual = 0): {len(non_played):,} ({len(non_played)/len(subset)*100:.1f}%)")

        # Check projection bias on PLAYED data only
        if played:
            played_projs = [r['projection'] for r in played]
            played_actuals = [r['actual'] for r in played]
            played_lines = [r['line'] for r in played]

            mean_proj = sum(played_projs) / len(played_projs)
            mean_actual = sum(played_actuals) / len(played_actuals)
            mean_line = sum(played_lines) / len(played_lines)

            bias = mean_proj - mean_actual

            print(f"\n  ON PLAYED DATA ONLY:")
            print(f"    Mean Projection: {mean_proj:.2f}")
            print(f"    Mean Actual:     {mean_actual:.2f}")
            print(f"    Mean Line:       {mean_line:.2f}")
            print(f"    Bias:            {bias:+.2f} pts")

        # Check direction split
        more = [r for r in subset if r['pick'] == 'MORE']
        less = [r for r in subset if r['pick'] == 'LESS']
        more_acc = sum(1 for r in more if r['result'] == 'W') / len(more) if more else 0
        less_acc = sum(1 for r in less if r['result'] == 'W') / len(less) if less else 0

        print(f"\n  DIRECTION SPLIT:")
        print(f"    MORE: {len(more):,} picks, {more_acc:.1%} accuracy")
        print(f"    LESS: {len(less):,} picks, {less_acc:.1%} accuracy")

        # On PLAYED data only
        if played:
            more_p = [r for r in played if r['pick'] == 'MORE']
            less_p = [r for r in played if r['pick'] == 'LESS']
            more_acc_p = sum(1 for r in more_p if r['result'] == 'W') / len(more_p) if more_p else 0
            less_acc_p = sum(1 for r in less_p if r['result'] == 'W') / len(less_p) if less_p else 0

            print(f"  ON PLAYED DATA ONLY:")
            print(f"    MORE: {len(more_p):,} picks, {more_acc_p:.1%} accuracy")
            print(f"    LESS: {len(less_p):,} picks, {less_acc_p:.1%} accuracy")

        # Rating distribution
        ratings = Counter(r['rating'] for r in subset)
        print(f"\n  RATING DISTRIBUTION:")
        for rating in ['A', 'B', 'C', 'D']:
            count = ratings[rating]
            pct = count / len(subset) * 100
            print(f"    {rating}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    analyze()
