import json
import numpy as np

with open('data/backtest/backtest_2025.json') as f:
    data = json.load(f)

print("=" * 100)
print("CORRECT OFFSET CALCULATION")
print("=" * 100)
print("\nLogic: offset = actual - projection (add offset to projection to correct bias)")
print()

new_weights = {}

for prop_type in ['pitcher_strikeouts', 'hitter_fantasy_score', 'total_bases', 'hits', 'home_runs']:
    records = [r for r in data if r['prop_type'] == prop_type and r['actual'] > 0]  # Played games only
    
    if not records:
        new_weights[prop_type] = 0.0
        continue
    
    projections = [r['projection'] for r in records]
    actuals = [r['actual'] for r in records]
    
    mean_proj = np.mean(projections)
    mean_actual = np.mean(actuals)
    
    # To correct the projection, we add: offset = actual - projection
    # In aggregate: offset = mean_actual - mean_projection
    offset = round(mean_actual - mean_proj, 2)
    new_weights[prop_type] = offset
    
    print(f"{prop_type}:")
    print(f"  Played games: {len(records)}")
    print(f"  Mean projection: {mean_proj:.3f}")
    print(f"  Mean actual: {mean_actual:.3f}")
    print(f"  Correction needed: actual - projection = {mean_actual - mean_proj:+.3f}")
    print(f"  → Offset: {offset:+.2f}")
    print()

print("=" * 100)
print("V007 WEIGHTS (CORRECTED):\n")
for key in ['pitcher_strikeouts', 'batter_strikeouts', 'hits', 'total_bases', 'home_runs', 'hitter_fantasy_score']:
    print(f'    "{key}": {new_weights.get(key, 0.0):+.2f},')
