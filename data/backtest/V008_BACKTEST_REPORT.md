# MLB Prop Edge v008 Backtest Report
**Date:** 2026-03-20
**Version:** v008 (HR binomial model + Fantasy Score variance 4.0 optimization)
**Data Points:** 178,402 predictions across full 2025 MLB season

---

## Executive Summary

**Overall Accuracy: 53.81% (49,018W - 42,072L out of 91,090 decided plays)**

The v008 weights show **strong performance on several prop types** but reveal critical issues requiring immediate attention:

### Key Findings by Direction

| Prop Type | MORE | LESS | Overall | Status |
|-----------|------|------|---------|--------|
| **hits** | 25.0% (4) | **65.1%** ✓ | 65.1% | PROFITABLE on LESS only |
| **hitter_fantasy_score** | 50.6% | **58.2%** ✓ | 56.3% | PROFITABLE on LESS only |
| **home_runs** | N/A | 0.0% ✗ | 0.0% | COMPLETELY BROKEN |
| **pitcher_strikeouts** | **57.9%** ✓ | **61.1%** ✓ | 58.6% | PROFITABLE both directions |
| **total_bases** | **62.5%** ✓ | 44.2% ✗ | 48.9% | PROFITABLE on MORE only |

---

## Detailed Analysis

### 1. PROFITABLE PICKS (Above 54.2% threshold for -110 DK odds)

#### ✓ Hits LESS (65.1% accuracy, 16,372-8,784 record)
- **Volume:** 25,156 picks
- **Edge:** Substantial. Model correctly predicts when batters will get fewer than 1.5 hits
- **Projection Bias:** Mean projection 0.97 vs actual 1.44 (underestimating by 0.47)
- **Status:** Excellent ROI. **ENABLE in live play**

#### ✓ Hitter Fantasy Score LESS (58.2% accuracy, 14,016-10,084 record)
- **Volume:** 24,100 picks
- **Edge:** Solid after variance ratio increase to 4.0
- **Projection Bias:** Mean projection 7.59 vs actual 8.86 (underestimating by 1.27)
- **Status:** Profitable. **Can enable but monitor for convergence**

#### ✓ Pitcher Strikeouts BOTH directions (57.9% MORE, 61.1% LESS)
- **MORE:** 3,099 picks at 57.9% (1,795-1,304)
- **LESS:** 894 picks at 61.1% (546-348)
- **Combined Volume:** 3,993 picks
- **Projection Bias:** Overestimating (mean 5.51 vs actual 4.93)
- **Status:** Strong two-way edge. **ENABLE both directions**

#### ✓ Total Bases MORE (62.5% accuracy, 4,023-2,418 record)
- **Volume:** 6,441 picks
- **Edge:** Strongest single direction in backtest
- **Projection Bias:** Underestimating (mean 1.58 vs actual 2.37)
- **Status:** Excellent. **Prioritize in live play**

### 2. NOT PROFITABLE (Below 54.2% threshold)

#### ✗ Hits MORE (25.0% accuracy, 1-3 record)
- **Volume:** Only 4 picks total
- **Status:** Insufficient data. Not recommended for live play

#### ✗ Hitter Fantasy Score MORE (50.6% accuracy, 3,985-3,891 record)
- **Volume:** 7,876 picks
- **Projection Bias:** Underestimating (mean 7.59 vs actual 8.86)
- **Issue:** Model shows no edge on MORE direction despite large sample
- **Status:** **Disable from live play.** Consider reducing MORE confidence thresholds

#### ✗ Home Runs LESS (0.0% accuracy, 0-4,801 record)
- **Volume:** 4,801 picks (largest single category)
- **Record:** 0-4,801 (perfect loss)
- **Projection Bias:** Severe underestimate (mean 0.14 vs actual 1.07)
- **Status:** **CRITICAL BUG.** Model is fundamentally broken for home runs
- **Root Cause:** New binomial P(1+ HR) model may have incorrect calibration or threshold issue

#### ✗ Total Bases LESS (44.2% accuracy, 8,280-10,439 record)
- **Volume:** 18,719 picks
- **Projection Bias:** Underestimating (mean 1.58 vs actual 2.37)
- **Status:** Systematically wrong. **Disable from live play**

---

## Critical Issues Requiring Immediate Fix

### ISSUE #1: Home Runs Model (0% Accuracy)
The HR LESS category shows **0-4,801 record** — every single prediction lost. This is a catastrophic failure.

**Diagnosis:**
- Mean projection: 0.14 vs actual 1.07
- Model severely underestimates HR likelihood
- New binomial P(1+ HR) model appears to be miscalibrated

**Recommendations:**
1. Review the HR projection model in `src/predictor.py`
   - Check P(1+ HR) calculation vs actual HR rates
   - Verify binomial parameters and confidence interval logic
2. Compare to previous versions to see when this regressed
3. Consider reverting HR model to simpler linear projection or adding calibration step
4. Do NOT use HR props in live play until fixed

### ISSUE #2: Hitter Fantasy Score MORE (50.6% Accuracy)
While not as catastrophic as HR, FS MORE at 50.6% shows the model cannot find edges on the upside.

**Possible causes:**
- Line placement at 7.5 may be optimal (sharp pricing)
- Confidence thresholds too aggressive (cutting off legitimate picks)
- Variance ratio increase to 4.0 too conservative for MORE picks

**Recommendations:**
1. Analyze confidence distribution: are all FS MORE picks getting D-grade?
2. Consider separate variance ratios for MORE vs LESS
3. Or disable FS MORE entirely and only trade LESS

### ISSUE #3: Total Bases LESS (44.2% Accuracy)
Systematically wrong in opposite direction of TB MORE.

**Diagnosis:**
- TB MORE is 62.5% (very profitable)
- TB LESS is 44.2% (losing)
- This suggests a systematic direction bias issue

**Recommendations:**
1. Check if TB LESS is being generated at all
2. If yes, add a direction multiplier (reduce LESS weight)
3. Or completely disable TB LESS and focus on TB MORE

---

## Performance by Sample Size

| Prop Type | Volume | Accuracy | Status |
|-----------|--------|----------|--------|
| Hits | 25,160 | 65.1% | Large sample, highly reliable |
| FS | 31,976 | 56.3% | Large sample, decent but needs work |
| HR | 4,801 | 0.0% | Moderate sample, completely broken |
| Pitcher K | 3,993 | 58.6% | Moderate sample, strong |
| Total Bases | 25,160 | 48.9% | Large sample, unreliable |

**Note:** All props have sufficient sample size (>1,000) for statistical reliability.

---

## Recommendations for Live Play

### ✓ ENABLE These (High Confidence)
1. **Hits LESS** — 65.1% accuracy, 25K+ picks
2. **Pitcher Strikeouts (both directions)** — 57.9%/61.1%, balanced
3. **Total Bases MORE** — 62.5% accuracy, 6.4K picks

### ⚠ CONDITIONAL ENABLE (Needs Monitoring)
1. **Hitter Fantasy Score LESS** — 58.2% accuracy, but monitor for edge degradation

### ✗ DISABLE IMMEDIATELY
1. **Hitter Fantasy Score MORE** — 50.6% accuracy, no edge
2. **Home Runs (ALL)** — 0% accuracy, broken model
3. **Total Bases LESS** — 44.2% accuracy, wrong direction

---

## Prediction Bias Analysis

| Prop Type | Mean Proj | Mean Actual | Line | Bias | Status |
|-----------|-----------|-------------|------|------|--------|
| Hits | 0.97 | 1.44 | 1.5 | -0.47 | Underestimate |
| FS | 7.59 | 8.86 | 7.5 | -1.27 | Underestimate |
| HR | 0.14 | 1.07 | 0.5 | -0.93 | SEVERE Underestimate |
| Pitcher K | 5.51 | 4.93 | 4.5 | +0.58 | Overestimate |
| Total Bases | 1.58 | 2.37 | 1.5 | -0.80 | Underestimate |

**Key Insight:** Most prop types are systematically underestimated (except pitcher K). This explains why LESS props (which go "against" the low line) tend to perform better. Consider adding uniform +0.5 to +1.0 offset across hitter props.

---

## Next Steps

### Immediate (This Session)
1. Fix HR projection model — verify binomial calibration
2. Disable HR, FS MORE, TB LESS from live play configuration
3. Re-run backtest after HR fix to measure improvement

### Short-term (Next Session)
1. Add per-direction variance ratios
2. Implement offset adjustment based on projection bias
3. Backtest new offsets to achieve >55% on all enabled props

### Medium-term
1. Implement live learning system (Task 6) to auto-detect and fix such issues
2. Add confidence calibration curve analysis
3. Consider ensemble model combining multiple approaches

---

## Summary Statistics

- **Total Backtest Records:** 178,402
- **Plays (actual > 0):** 91,090 (51.1%)
- **Non-plays (actual = 0):** 87,312 (48.9%)
- **Overall Win Rate:** 53.81%
- **Profitable Props:** 5 out of 9 (hitting + direction combos)
- **Date Range:** 2025-04-01 to 2025-09-30 (183 days)

---

**Generated:** 2026-03-20 via v008 backtest analysis
