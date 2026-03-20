# Overnight Optimization Session - Analysis & Fixes

**Date:** 2026-03-19 (continued)
**Status:** Backtest running with PA >= 2 filter (in progress)

## Key Findings from Cross-Tab Analysis

### Current Accuracy (46,955 records, April-Sept 2025)
- **Hits:** 79.9% (0% MORE, 79.9% LESS)
- **Home Runs:** 89.9% (0% MORE, 89.9% LESS)
- **Fantasy Score:** 64.5% (42.1% MORE, 70.4% LESS)
- **Total Bases:** 62.7% (38.9% MORE, 69.5% LESS)
- **Pitcher Ks:** 58.8% (57.3% MORE, 64.0% LESS)

### Projection Bias
- **Hits:** +0.12 pts (minimal bias, model well-calibrated)
- **Fantasy Score:** +1.04 pts (overprojecting despite -0.75 offset)
- **Total Bases:** +0.21 pts (overprojecting despite -0.25 offset)
- **Pitcher Ks:** +0.68 Ks (overprojecting)

## Root Cause: Bench Player Inclusion

**Problem:** Backtest includes players with 1 PA (pinch hitters, defensive replacements)
- 26.4% of fantasy score predictions have actual=0.0
- These create automatic LESS wins and MORE losses
- Example: Elly De La Cruz projected 9.02 fantasy pts, actual 0.0 (pinch hitter who struck out)

**Live vs Backtest:**
- **Backtest:** Includes all roster players (~10-15 per team per game)
- **Live PrizePicks:** Shows props only for probable starters (~9 per team)
- **Gap:** ~25-30% of backtest predictions are for non-starters

**Impact on Accuracy:**
- Non-starters with 0 actual → automatic LESS wins
- Inflates LESS accuracy by 5-10pp
- Suppresses MORE accuracy (because non-starters mostly get 0 pts)

## Fix Applied: PA >= 2 Filter

**Change:** Modified `src/backtester.py` line 233:
```python
# OLD: stats.get("pa", 0) > 0
# NEW: stats.get("pa", 0) >= 2
```

**Rationale:**
- Players with only 1 PA are pinch hitters/defensive replacements
- Don't get props in live PrizePicks
- Excluding them makes backtest reflect actual live conditions
- Should see MORE accuracy improve significantly

**Expected Impact:**
- Fantasy Score MORE: 42% → 48-52% (remove automatic losses)
- Fantasy Score LESS: 70% → 64-68% (lose automatic wins)
- Overall fantasy Score: 64.5% → likely stable or slight improvement
- Pitcher Ks similar dynamics
- Hits/HR unaffected (already all LESS)

## Other Issues Identified

### 1. Hits Projection Too Conservative
- Line 1.5 requires 0.393 AVG (impossible for elite hitters)
- Model correctly projects mostly LESS
- 79.9% accuracy is actually very good
- MORE picks (~5%) would be low-confidence marginal plays
- **Decision:** Model is working as intended. Leave unchanged.

### 2. Fantasy Score Variance Too Low
- Projection std dev: 0.093 (very low variation)
- Actual std dev: 0.896 (high variation)
- Causes overconfident P(over/under) calculations
- **Potential fix:** Increase variance_ratio from 2.8 to 3.5+
- **Status:** Will test after backtest complete

### 3. Pitcher K Overprojection
- Mean projection: 5.55, mean actual: 4.87
- Bias: +0.68 Ks
- Current offset: -0.40 (leaves +0.28 residual bias)
- **Potential fix:** Increase offset to -0.70
- **Status:** Will test after backtest complete

## Next Steps After Backtest

1. Run cross_tab.py to see impact of PA >= 2 filter
2. If MORE accuracy improves 5-10pp: commit fix and document
3. If variance ratio adjustment needed: increase HITTER_FANTASY_SCORE from 2.8 to 3.5
4. Analyze direction bias (why do MORE picks still underperform?)
5. Consider learning weights from live accuracy once we have 50+ picks

## Files Modified
- src/backtester.py: PA >= 2 filter on line 233 (in progress)

