import json
import numpy as np

with open('data/backtest/backtest_2025.json') as f:
    data = json.load(f)

print("=" * 100)
print("OFFSET CALCULATION - CORRECTING FOR NON-PLAYS")
print("=" * 100)

results = {}

for prop_type in ['pitcher_strikeouts', 'hitter_fantasy_score', 'total_bases', 'hits', 'home_runs']:
    records = [r for r in data if r['prop_type'] == prop_type]
    if not records:
        continue
    
    # Split by direction
    more_records = [r for r in records if r['pick'] == 'MORE']
    less_records = [r for r in records if r['pick'] == 'LESS']
    
    results[prop_type] = {}
    
    for direction, recs in [('MORE', more_records), ('LESS', less_records)]:
        if not recs:
            continue
        
        plays = [r for r in recs if r['actual'] > 0]
        nonplays = [r for r in recs if r['actual'] == 0]
        
        # Calculate mean bias when actual > 0 (played)
        if plays:
            proj_values = [r['projection'] for r in plays]
            actual_values = [r['actual'] for r in plays]
            mean_proj = np.mean(proj_values)
            mean_actual = np.mean(actual_values)
            bias_played = mean_proj - mean_actual
        else:
            bias_played = 0
        
        # Calculate mean bias on all (including non-plays)
        proj_all = [r['projection'] for r in recs]
        actual_all = [r['actual'] for r in recs]
        bias_all = np.mean(proj_all) - np.mean(actual_all)
        
        # The offset should match bias_played, not bias_all
        results[prop_type][direction] = {
            'sample_size': len(recs),
            'plays': len(plays),
            'nonplays': len(nonplays),
            'bias_all': round(bias_all, 3),
            'bias_played': round(bias_played, 3),
            'suggested_offset': round(bias_played, 2),  # Offset for played games only
        }

# Print results
print(f"\n{'Prop Type':<25} {'Direction':<6} {'Sample':<7} {'Plays':<7} {'NonPlay%':<10} {'Bias(all)':<12} {'Bias(played)':<15} {'Suggested Offset':<18}")
print("-" * 100)

for prop_type in sorted(results.keys()):
    for direction in ['MORE', 'LESS']:
        if direction not in results[prop_type]:
            continue
        r = results[prop_type][direction]
        nonplay_pct = r['nonplays'] / r['sample_size'] * 100 if r['sample_size'] > 0 else 0
        print(f"{prop_type:<25} {direction:<6} {r['sample_size']:<7} {r['plays']:<7} {nonplay_pct:>8.1f}%  {r['bias_all']:>+10.3f}  {r['bias_played']:>+13.3f}  {r['suggested_offset']:>+16.2f}")

print("\n" + "=" * 100)
print("KEY INSIGHT: Offsets should match 'Bias(played)', not 'Bias(all)'")
print("The non-plays are inflating LESS and deflating MORE by ~10-25pp")
print("Solution: Use 'Bias(played)' to set projection offsets")
print("=" * 100)

# Now let's see what v005 weights would be if recalibrated to match played-only bias
print("\n\nRECALIBRATED WEIGHTS (based on played-only bias):\n")
for prop_type in ['pitcher_strikeouts', 'hitter_fantasy_score', 'total_bases', 'hits', 'home_runs']:
    if prop_type not in results:
        continue
    
    # Use the MORE direction for bias calculation (it's more affected by non-plays)
    if 'MORE' in results[prop_type]:
        offset = results[prop_type]['MORE']['suggested_offset']
    else:
        offset = 0
    
    print(f'    "{prop_type}": {offset:.2f},')
