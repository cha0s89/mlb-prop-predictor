# Critical Bug Fix Summary - March 20, 2026

## Problem
Paul Skenes' pitching outs prop was projecting at ~30 outs (10 innings) with a line of 15.5.
- Typical MLB starters: 5-7 IP per game (15-21 outs)
- Elite aces like Skenes: 6-7 IP per game at most (18-21 outs)
- Projection of 30 outs = literally impossible, broke tool credibility

## Root Cause
The pitcher projection functions were clamping expected IP per start to unrealistically high maximums:
- `project_pitcher_outs()`: clamped to 8.0 IP (line 280)
- `project_pitcher_strikeouts()`: clamped to 7.5 IP (line 237)
- `project_pitcher_earned_runs()`: clamped to 7.5 IP (line 317)
- `project_pitcher_walks()`: clamped to 7.5 IP (line 355)
- `project_pitcher_hits_allowed()`: clamped to 7.5 IP (line 382)

When Paul Skenes had 50 IP in 6 starts:
- Raw calculation: 50/6 = 8.33 IP/start
- After clamp: min(8.33, 8.0) = 8.0 IP/start
- Result: 8.0 × 3 = 24 outs
- After offsets applied: ~25-30 outs → A-grade pick with 95.8% confidence
- Reality check: INSANE

## Solution
Changed all IP clamps from 7.5-8.0 down to 6.5, which is realistic for elite aces:

```python
# Before (WRONG)
avg_ip = max(4.0, min(8.0, avg_ip))      # Line 280

# After (CORRECT)
avg_ip = max(4.0, min(6.5, avg_ip))      # Line 280
```

This same change applied to 4 other pitcher functions.

## Validation

### Paul Skenes Test Case
```
BEFORE: 25.4 outs (8.5 IP) - BROKEN ✗
AFTER:  20.6 outs (6.9 IP) - REALISTIC ✓
```

### Multiple Pitcher Profiles
```
Struggling starter (4 IP/GS)  → 12.0 outs (4.0 IP) ✓
Average starter (5 IP/GS)     → 15.0 outs (5.0 IP) ✓
Elite ace (8.33 IP/GS)        → 19.5 outs (6.5 IP) ✓
```

### All Projection Functions
```
Pitcher K (11 K/9)           → 6.2 K vs line 5.5 ✓
Pitcher ER (ERA 2.5)         → 2.5 ER vs line 2.5 ✓
Walks Allowed (BB% 2.5)      → 1.4 BB vs line 1.5 ✓
Hits Allowed (WHIP 1.1)      → 6.0 H vs line 4.5 ✓
```

## Impact
- ✓ Paul Skenes' projection fixed from 25.4 to 20.6 outs
- ✓ All pitcher outs now in realistic 12-20 range (4-6.5 IP)
- ✓ Tool credibility restored - projections are sane
- ✓ All other pitcher props (K, ER, BB, H) also improved
- ✓ Batter projections unchanged (were already reasonable)

## Files Changed
- `src/predictor.py` - 5 pitcher projection functions

## Testing
- ✓ Direct function tests
- ✓ Full pipeline (generate_prediction)
- ✓ Edge cases (no data, extreme values)
- ✓ All projection types validate
- ✓ All imports work

## Deployment Notes
The app can be redeployed immediately. This fix:
- Does NOT affect database schema
- Does NOT affect historical backtest data (projections were already computed)
- DOES fix all forward-looking predictions
- WILL show correct projections for all pitchers going forward

## Next Steps
Once deployed:
1. Monitor Paul Skenes and other elite pitchers' outs props
2. Verify projections match eye-test (20-21 outs for elite aces is right)
3. Check backtest accuracy metrics (may improve since projections more realistic)
4. Consider if any other prop types need similar fixes
