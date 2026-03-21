# Offset & League Average Optimization Report
**Date:** 2026-03-20
**Analysis:** Comprehensive grid search over offset values and variance ratios
**Status:** CRITICAL FINDINGS - IMMEDIATE CHANGES REQUIRED

---

## Executive Summary

The current v009 configuration has a **critical flaw**: the offsets are TOO LARGE, causing the MORE direction accuracy to plummet. The backtest data shows:

| Config | FS MORE | FS LESS | TB MORE | TB LESS |
|--------|---------|---------|---------|---------|
| **Current v009** | 39.5% ❌ | 69.9% | 36.0% ❌ | 70.8% |
| **Zero offsets** | 45.6% ⚠️ | 68.9% | 44.5% ⚠️ | 66.9% |
| **v009 claimed** | 58.4% | 56.8% | 62.5% | — |

**The v009 metadata is INACCURATE.** The claimed 58.4% FS MORE was never actually achieved because the backtest was scored with the OLD variance ratio. The variance ratio adjustment was a PROPOSAL, not a tested result.

---

## Phase 1: League Average Analysis

### Actual Distribution in Backtest (178,402 plays)

```
hitter_fantasy_score (43,583 plays):
  Mean actual:    6.50  (median 5.0)
  Mean projection: 7.54  (with current v009 offsets)
  Bias:          -1.05  (projections are 1.0 pt too high)

total_bases (43,583 plays):
  Mean actual:    1.37  (median 1.0)
  Mean projection: 1.56  (with current offsets)
  Bias:          -0.19

hits (43,583 plays):
  Mean actual:    0.83  (median 1.0)
  Mean projection: 0.96  (with current offsets)
  Bias:          -0.13

pitcher_strikeouts (4,070 plays):
  Mean actual:    4.84  (median 5.0)
  Mean projection: 5.50  (with current offsets)
  Bias:          -0.67
```

### League Average Assessment

The league averages in predictor.py are:
```python
"hits_per_game": 1.44,  # From 0.95 originally
"tb_per_game": 2.35,    # From 1.50 originally
```

These are **per-player-per-game values from the backtest mean actuals**. However:

- **MLB average: ~0.94 hits per batter per game** (8.5 team hits / 9 batters)
- **Backtest shows: 0.83 mean** (biased toward bench players who play less)
- **The 1.44 value is inflated** because it's the mean of all individual plays, including multi-hit games

**Recommendation:** Keep league averages at **0.95 hits_per_game** and **1.50 tb_per_game** for Bayesian regression purposes. These provide proper stabilization toward population mean.

---

## Phase 2: Grid Search Results

Tested 600 offset configurations:
- hitter_fantasy_score: [0.0, 0.3, 0.5, 0.8, 1.0, 1.23]
- total_bases: [0.0, 0.2, 0.4, 0.6, 0.78]
- hits: [0.0, 0.1, 0.2, 0.3, 0.48]
- pitcher_strikeouts: [-0.58, -0.40, -0.20, 0.0]

### Best Configuration (across all 600)

```json
{
  "name": "Grid Search Winner",
  "offsets": {
    "hitter_fantasy_score": 0.0,
    "total_bases": 0.2,
    "hits": 0.0,
    "pitcher_strikeouts": -0.58
  },
  "results": {
    "hitter_fantasy_score_more": 0.4558,
    "hitter_fantasy_score_less": 0.6886,
    "total_bases_more": 0.4501,
    "total_bases_less": 0.6704,
    "hits_more": 0.5000,
    "hits_less": 0.7984,
    "pitcher_strikeouts_more": 0.5605,
    "pitcher_strikeouts_less": 0.6307
  }
}
```

**Problems with this config:**
- FS MORE still at 45.6% (below 54% threshold)
- TB MORE still below 45.5%
- This is a "least bad" result, not a winning config

### Key Finding: The Offset Problem

The large offsets (1.23 for FS, 0.78 for TB, 0.48 for hits) were designed to correct for upward bias in projections. However, they're **overcorrecting the MORE direction** to the point where MORE picks become unbeatable (only 36-45% accuracy).

The MORE direction needs:
1. **Higher starting projections** (not achieved by reducing offsets)
2. **Better projection model** (not achieved by mechanical adjustments)
3. **Smarter line selection** (avoid lines where MORE is structurally weak)

---

## Phase 3: The Real Root Cause

Analysis shows:

```
Mean projection - mean actual (in PrizePicks lines):
- FS: 7.54 - 6.50 = +1.04 pts (projections too high)
- TB: 1.56 - 1.37 = +0.19 bases (projections too high)
- Hits: 0.96 - 0.83 = +0.13 hits (projections too high)
```

**The model systematically OVERESTIMATES player performance.** This is not a calibration problem solvable by offsets. It's a fundamental projection bias.

**Causes:**
1. **Baseline rates are too optimistic** - League averages in the predictor may be from elite player subsets
2. **Factor multipliers are too aggressive** - Park factor, matchup, platoon adjustments may be overstating boosts
3. **Statcast blending is aggressive** - xBA/xSLG may be optimistic relative to actual results in this line universe
4. **Selection bias** - PrizePicks selects props for "line-worthy" events, which may have different distributions than full population

---

## Phase 4: Variance Ratio Question

The v009 proposal claimed that increasing hitter_fantasy_score variance from 4.0 to 5.0 would:
- Improve FS MORE from 50.6% → 58.4% (+7.8pp)
- Maintain FS LESS at 56.8%

**Status: UNTESTED.** The backtest data was scored with the old variance ratio. The variance change was not actually applied and tested on this data.

**Proper variance ratio logic:**
- Higher variance → wider probability distribution → picks pushed toward 50% confidence
- This filters out low-confidence picks, which is good IF they're inaccurate
- BUT it doesn't fix the fundamental over-projection problem

**Variance ratio cannot fix MORE direction if projections are systematically too high.** If a player projects 8.0 for a 7.5 line (biased high), increasing variance just reduces confidence to 50-55% instead of 60-65%. The pick is still MORE, but with lower edge.

---

## Recommendations

### Immediate (For Tonight)

**DO NOT use the variance ratio claim as a fix.** The v009 config as tested shows 39.5% FS MORE — below what you need.

**Option A: Roll back to Zero Offsets**
- Apply zero offsets for FS, TB, and Hits
- Keep PK offset at -0.58
- Result: FS MORE ~45.6%, TB MORE ~44.5%
- This is better than current (39.5%, 36.0%) but still below 54%

**Option B: Find a Balanced Config**
The grid search found that **any configuration that removes/reduces FS offset improves MORE accuracy.** But all tested configs fail to reach 54% on both MORE and LESS simultaneously.

**Why all configs fail:**
The backtest shows FS LESS is naturally strong (67-70%) because that's where the data cluster is (median 5.0 vs line 7.5). MORE will always be weak without fundamentally better projections.

### Root Fix (Requires Model Changes)

To get FS MORE to 54%+, you need to either:

1. **Reduce baseline FS projection by ~0.5-1.0 pts** to match actual performance better
2. **Increase multiplier weights** for factors that predict high-scoring games (better matchup, park factor, hot streak)
3. **Add explicit selection bias correction** for PrizePicks line selection
4. **Filter to only high-confidence MORE picks** (grade A only) and accept lower volume

### Tonight's Decision

**Based on accuracy data, I recommend:**

1. **Update offsets to:**
   ```json
   "hitter_fantasy_score": 0.0,
   "total_bases": 0.2,
   "hits": 0.0,
   "pitcher_strikeouts": -0.58,
   ```

2. **DO NOT apply variance ratio claim** (untested, won't help MORE direction)

3. **Revert league averages to original values:**
   ```python
   "hits_per_game": 0.95,
   "tb_per_game": 1.50,
   ```

4. **Mark TB MORE and FS MORE as "under review"** in the dashboard
   - Trade: FS LESS (68.9%), PK Ks LESS (70.7%), PK Ks MORE (54.0%)
   - Avoid: FS MORE, TB MORE, TB LESS until projections are validated

5. **Schedule urgent projection model review** for this weekend
   - Audit base rates against 2024 actual season
   - Test factor multiplier impacts individually
   - Run sensitivity analysis on park factors

---

## Files to Update

1. **data/weights/current.json**
   - Set offsets as recommended above
   - Remove variance_ratios OR set to 1.0 (don't claim 5.0 without proof)
   - Update description to note "offsets reduced to improve MORE direction"

2. **src/predictor.py** (if reverting league averages)
   - Change hits_per_game back to 0.95
   - Change tb_per_game back to 1.50

3. **app.py**
   - Add warning badges on FS MORE and TB MORE until projections are recalibrated

---

## Data Quality Notes

- **Backtest plays:** 178,402 total
- **FS records:** 43,583 (10,079 MORE, 33,504 LESS)
- **TB records:** 43,583 (19,500 MORE, 24,083 LESS)
- **Hits records:** 43,583 (21,791 MORE, 21,792 LESS)
- **PK records:** 4,070 (1,725 MORE, 2,345 LESS)

Bias is consistent across all major prop types. NOT a statistical fluke.

---

**Analysis complete. Ready for config update and live testing.**
