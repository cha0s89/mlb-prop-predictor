import json
import numpy as np

with open('data/backtest/backtest_2025.json') as f:
    data = json.load(f)

print("=" * 100)
print("FINAL OFFSET CALCULATION - USING PLAYED GAMES ONLY")
print("=" * 100)

# The key insight: non-plays (actual=0) artificially help LESS and hurt MORE
# To correct this, we should:
# 1. Calculate projection bias only on games where actual > 0
# 2. Use that bias as the offset

new_weights = {}

for prop_type in ['pitcher_strikeouts', 'hitter_fantasy_score', 'total_bases', 'hits', 'home_runs']:
    records = [r for r in data if r['prop_type'] == prop_type and r['actual'] > 0]  # PLAYED GAMES ONLY
    
    if not records:
        new_weights[prop_type] = 0.0
        continue
    
    projections = [r['projection'] for r in records]
    actuals = [r['actual'] for r in records]
    
    mean_proj = np.mean(projections)
    mean_actual = np.mean(actuals)
    bias = mean_proj - mean_actual
    
    # Round to 2 decimal places for the offset
    offset = round(bias, 2)
    new_weights[prop_type] = offset
    
    # Also show MORE vs LESS breakdown
    more_recs = [r for r in records if r['pick'] == 'MORE']
    less_recs = [r for r in records if r['pick'] == 'LESS']
    
    print(f"\n{prop_type}:")
    print(f"  Overall (played games only): {len(records)} records")
    print(f"    Mean projection: {mean_proj:.3f}, Mean actual: {mean_actual:.3f}")
    print(f"    Bias (projection - actual): {bias:+.3f}")
    print(f"    → Suggested offset: {offset:+.2f}")
    
    if more_recs:
        more_proj = np.mean([r['projection'] for r in more_recs])
        more_act = np.mean([r['actual'] for r in more_recs])
        more_bias = more_proj - more_act
        print(f"  MORE direction: {len(more_recs)} records, bias {more_bias:+.3f}")
    
    if less_recs:
        less_proj = np.mean([r['projection'] for r in less_recs])
        less_act = np.mean([r['actual'] for r in less_recs])
        less_bias = less_proj - less_act
        print(f"  LESS direction: {len(less_recs)} records, bias {less_bias:+.3f}")

print("\n" + "=" * 100)
print("NEW WEIGHTS (v006) — BASED ON PLAYED GAMES ONLY")
print("=" * 100)
print("\nReplace 'prop_type_offsets' section in data/weights/current.json with:\n")
for key, val in sorted(new_weights.items()):
    print(f'    "{key}": {val:+.2f},')

print("\n\nCOMPARISON TO v005:")
v005 = {
    "pitcher_strikeouts": -0.67,
    "hitter_fantasy_score": -1.07,
    "total_bases": -0.22,
    "hits": -0.12,
    "home_runs": -0.02,
}

print(f"\n{'Prop Type':<25} {'v005':<10} {'v006 (played-only)':<20} {'Difference':<15}")
print("-" * 70)
for key in sorted(new_weights.keys()):
    v005_val = v005.get(key, 0.0)
    v006_val = new_weights[key]
    diff = v006_val - v005_val
    print(f"{key:<25} {v005_val:>+8.2f}  {v006_val:>+18.2f}  {diff:>+13.2f}")
