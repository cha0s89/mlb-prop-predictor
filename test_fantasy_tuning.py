#!/usr/bin/env python3
"""
Test fantasy score projections against actual results to identify
the source of the 1.15 pt overprojection bias.
"""

import json
import sys
sys.path.insert(0, '.')

from src.predictor import project_hitter_fantasy_score
from src.backtester import build_walkforward_profile, _actual_value_for_prop, calculate_fantasy_score

# Load actual results
with open('data/backtest/backtest_2025.json') as f:
    records = json.load(f)

fs_records = [r for r in records if r['prop_type'] == 'hitter_fantasy_score']
played = [r for r in fs_records if r['actual'] > 0]

print("=" * 80)
print("FANTASY SCORE BIAS ANALYSIS")
print("=" * 80)
print(f"\nTotal played records: {len(played)}")

# Group by projection bins to see if high-projection cases are the problem
proj_bins = {
    "low (< 6.5)": [r for r in played if r['projection'] < 6.5],
    "medium (6.5-8.0)": [r for r in played if 6.5 <= r['projection'] <= 8.0],
    "high (> 8.0)": [r for r in played if r['projection'] > 8.0],
}

for label, subset in proj_bins.items():
    if not subset:
        continue
    mean_proj = sum(r['projection'] for r in subset) / len(subset)
    mean_actual = sum(r['actual'] for r in subset) / len(subset)
    bias = mean_proj - mean_actual

    # Accuracy
    more = [r for r in subset if r['pick'] == 'MORE']
    less = [r for r in subset if r['pick'] == 'LESS']
    more_acc = sum(1 for r in more if r['result'] == 'W') / len(more) if more else 0
    less_acc = sum(1 for r in less if r['result'] == 'W') / len(less) if less else 0

    print(f"\n{label}: {len(subset):,} records")
    print(f"  Mean projection: {mean_proj:.2f}")
    print(f"  Mean actual: {mean_actual:.2f}")
    print(f"  Bias: {bias:+.2f}")
    print(f"  MORE acc: {more_acc:.1%} ({len(more)} picks)")
    print(f"  LESS acc: {less_acc:.1%} ({len(less)} picks)")

print(f"\n\n=== HYPOTHESIS ===")
print(f"Is the bias uniform across all players, or are some subgroups problematic?")

# Group by actual value (low action, medium action, high action)
action_bins = {
    "low action (0-5 pts)": [r for r in played if r['actual'] < 5],
    "medium action (5-9 pts)": [r for r in played if 5 <= r['actual'] <= 9],
    "high action (9+ pts)": [r for r in played if r['actual'] > 9],
}

for label, subset in action_bins.items():
    if not subset:
        continue
    mean_proj = sum(r['projection'] for r in subset) / len(subset)
    mean_actual = sum(r['actual'] for r in subset) / len(subset)
    bias = mean_proj - mean_actual

    print(f"\n{label}: {len(subset):,} records")
    print(f"  Mean projection: {mean_proj:.2f}")
    print(f"  Mean actual: {mean_actual:.2f}")
    print(f"  Bias: {bias:+.2f}")

print(f"\n\n=== KEY INSIGHT ===")
print(f"The model might be good at estimating high-action players but bad at")
print(f"estimating low-action players (bench, weak hitters, etc)")
print(f"Look at the 'low action' group — is the bias higher there?")
