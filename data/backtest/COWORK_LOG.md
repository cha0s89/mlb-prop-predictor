# Cowork Session Log - MLB Prop Predictor Improvement

**Date**: 2026-03-19
**Objective**: Significantly improve prediction accuracy on main bettable props
**Status**: Diagnostic Phase Complete - Root Causes Identified

## Key Findings

### 1. Non-Play Bias (FIXED IN CODE)
- Backtester was including non-playing batters (0 PA)
- 42.6% of total_bases predictions are non-plays
- 26.2% of fantasy_score predictions are non-plays
- **Fix**: Modified `src/backtester.py` to filter `extract_all_batters()` for PA > 0

### 2. Projection Bias for Low-Action Players (ROOT CAUSE)
- Fantasy Score mean projection: 7.53
- Fantasy Score mean actual: 6.41 (overall), 8.68 (on played data)
- "Low action" group (0-5 actual pts): +4.66 pt bias
- Root: Model defaults to league-average (7.5) when player data missing
- **Impact**: Causes massive LESS bias artificially
- **Fix Needed**: Better low-confidence profile detection

### 3. Direction Bias Analysis (COUNTER-INTUITIVE)
- Model predicts LESS 79% for fantasy score (vs 50% optimal)
- LESS accuracy: 70.6%, MORE accuracy: 42.3%
- **Key insight**: Heavy LESS bias is WORKING but suboptimal
- v004 negative offsets are CORRECT (not backwards)
- Missing out on +50% MORE opportunities when they should be taken

## Current Accuracy (38,626 records from April 2025 backtest)

| Prop | Count | MORE | MORE % | LESS | LESS % | Overall |
|------|-------|------|--------|------|--------|---------|
| Pitcher Ks | 834 | 56.5% | 642 | 66.1% | 192 | **59.1%** |
| Fantasy Score | 9,719 | 42.3% | 2,047 | 70.6% | 7,672 | **64.8%** |
| Total Bases | 8,681 | 38.7% | 1,942 | 69.8% | 6,739 | **62.8%** |
| Hits | 8,681 | — | 0 | 80.4% | 8,681 | **80.4%** |

**Problem**: LESS is over-weighted due to non-play bias. Expect 5-15% improvement after non-play filtering.

## Code Changes Made

### 1. src/backtester.py
```python
# Line 220-221: Filter non-playing batters
if stats and full_name and stats.get("pa", 0) > 0:
```

### 2. No weights changes
- v004 offsets are correct (negative offsets are right strategy)
- Don't reverse them

## Next Session Tasks

### CRITICAL (Do First)
1. **Run fresh backtest with PA > 0 filtering**
   - Will show true direction bias without non-play noise
   - Expect: Fantasy Score MORE acc ~50%+, LESS acc ~60%
   
2. **Fix Hits routing**
   - Currently 0 MORE picks (should be 10-20%)
   - Check if hits() is being called or falls back to something else

3. **Improve low-confidence profile detection**
   - Add confidence flag when PA < 50 in early season
   - Use conservative baseline projection (closer to league avg)
   - Don't apply full contextual multipliers

### HIGH PRIORITY
4. **Optimize Expected IP for pitchers**
   - Early season pitchers may have different IP distributions
   - Check if 5.5 IP baseline is correct for April

5. **Tune variance ratios**
   - Current: fantasy_score 2.8, total_bases 1.6
   - These should be learned from accuracy curves

### MEDIUM PRIORITY
6. **Implement direction bias correction**
   - More multiplier / Less multiplier in weights
   - Currently both 1.0 (should be tuned)

7. **Fix hits projection model**
   - Why is it predicting only LESS?
   - Compare to total_bases and home_runs logic

## Files Modified
- `src/backtester.py`: extract_all_batters() now filters PA > 0
- Created diagnostic scripts: test_projections.py, analyze_bias.py, detailed_diagnosis.py

## Session Notes

- Reprocessing old backtest with opposite offsets made things WORSE (proved v4 was right)
- The bias analysis revealed that the backtest data itself was corrupted by non-plays
- Expected accuracy improvements: 5-15% once non-plays are filtered out
- Fresh backtest needed to validate fixes

## Backtest Files
- Current: data/backtest/backtest_2025.json (38,626 records, ~19 days)
- Backup: data/backtest/backtest_2025_old.json
