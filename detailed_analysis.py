import json
import numpy as np

# Load backtest data
with open('data/backtest/backtest_2025.json') as f:
    data = json.load(f)

print("=" * 80)
print("DETAILED BACKTEST ANALYSIS (70,730 records)")
print("=" * 80)

# Group by prop type and direction
by_prop_dir = {}
for rec in data:
    key = (rec['prop_type'], rec['pick'])
    if key not in by_prop_dir:
        by_prop_dir[key] = []
    by_prop_dir[key].append(rec)

# For each prop × direction combo, analyze
for (prop_type, direction), recs in sorted(by_prop_dir.items()):
    n = len(recs)
    actual_values = [r['actual'] for r in recs]
    proj_values = [r['projection'] for r in recs]
    
    # Count exact zeros (non-plays)
    zeros = sum(1 for a in actual_values if a == 0.0)
    non_zeros = sum(1 for a in actual_values if a > 0)
    
    # Win/loss
    wins = sum(1 for r in recs if r['result'] == 'W')
    losses = sum(1 for r in recs if r['result'] == 'L')
    
    # Stats for non-zero actuals only
    if non_zeros > 0:
        non_zero_actuals = [a for a in actual_values if a > 0]
        non_zero_projs = [proj_values[i] for i, a in enumerate(actual_values) if a > 0]
        mean_proj_nz = np.mean(non_zero_projs)
        mean_actual_nz = np.mean(non_zero_actuals)
        bias_nz = mean_proj_nz - mean_actual_nz
    else:
        mean_proj_nz = np.mean(proj_values)
        mean_actual_nz = 0
        bias_nz = 0
    
    mean_proj_all = np.mean(proj_values)
    mean_actual_all = np.mean(actual_values)
    bias_all = mean_proj_all - mean_actual_all
    
    win_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    print(f"\n{prop_type:25} × {direction:4} (n={n:5})")
    print(f"  Wins/Losses: {wins:5} - {losses:5} ({win_pct:5.1f}%)")
    print(f"  Non-plays (actual=0):  {zeros:5} ({zeros/n*100:5.1f}%)")
    print(f"  Actual plays (>0):     {non_zeros:5} ({non_zeros/n*100:5.1f}%)")
    print(f"  Mean projection (all): {mean_proj_all:6.3f}  Mean actual: {mean_actual_all:6.3f}  Bias: {bias_all:+6.3f}")
    if non_zeros > 0:
        print(f"  Mean projection (played): {mean_proj_nz:6.3f}  Mean actual: {mean_actual_nz:6.3f}  Bias: {bias_nz:+6.3f}")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Prop Type':<25} {'Direction':<6} {'Sample':<7} {'W%':<7} {'Non-plays':<12} {'Bias (all)':<12} {'Bias (played)':<12}")
print("-" * 80)

for (prop_type, direction), recs in sorted(by_prop_dir.items()):
    n = len(recs)
    actual_values = [r['actual'] for r in recs]
    proj_values = [r['projection'] for r in recs]
    
    zeros = sum(1 for a in actual_values if a == 0.0)
    non_zeros = sum(1 for a in actual_values if a > 0)
    
    wins = sum(1 for r in recs if r['result'] == 'W')
    losses = sum(1 for r in recs if r['result'] == 'L')
    win_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    bias_all = np.mean(proj_values) - np.mean(actual_values)
    
    if non_zeros > 0:
        non_zero_actuals = [a for a in actual_values if a > 0]
        non_zero_projs = [proj_values[i] for i, a in enumerate(actual_values) if a > 0]
        bias_nz = np.mean(non_zero_projs) - np.mean(non_zero_actuals)
    else:
        bias_nz = 0
    
    print(f"{prop_type:<25} {direction:<6} {n:<7} {win_pct:6.1f}% {zeros/n*100:10.1f}%  {bias_all:+10.3f}  {bias_nz:+10.3f}")
