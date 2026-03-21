# v008 Backtest Results - March 20, 2026

This directory contains the complete backtest analysis for v008 weights configuration.

## Quick Start

1. **For Executive Summary:** Read `QUICK_REFERENCE.txt` (2 min read)
2. **For Detailed Analysis:** Read `V008_BACKTEST_REPORT.md` (10 min read)
3. **For Raw Data:** See `backtest_2025.json` (178K records)

## Key Findings

**Overall Accuracy: 53.81%** (49,018W - 42,072L across 91,090 plays)

### Profitable Props (5 of 9)
- ✓ Hits LESS: 65.1% (25,156 picks) — **ENABLE**
- ✓ Pitcher K MORE: 57.9% (3,099 picks) — **ENABLE**
- ✓ Pitcher K LESS: 61.1% (894 picks) — **ENABLE**
- ✓ Total Bases MORE: 62.5% (6,441 picks) — **ENABLE**
- ✓ FS LESS: 58.2% (24,100 picks) — **ENABLE (monitor)**

### Broken/Unprofitable Props (4 of 9)
- ✗ Home Runs LESS: 0.0% (0-4,801) — **CRITICAL BUG - DISABLE**
- ✗ FS MORE: 50.6% (3,985-3,891) — **DISABLE**
- ✗ Total Bases LESS: 44.2% (8,280-10,439) — **DISABLE**
- ✗ Hits MORE: 25.0% (1-3 insufficient) — **DISABLE**

## Files in This Directory

### Reports (Read These)
- `QUICK_REFERENCE.txt` - One-page summary with commands and metrics
- `V008_BACKTEST_REPORT.md` - Comprehensive 10-page analysis
- `BACKTEST_SESSION_SUMMARY.md` - Session notes and next steps
- `README_V008_BACKTEST.md` - This file

### Data Files
- `backtest_2025.json` - Full 178,402 prediction records (64 MB)
- `backtest_2025_report.json` - JSON summary of overall metrics

### Log Files (Reference)
- `backtest_run_v008.log` - Backtest execution log (network issues encountered)

## Critical Issue: Home Runs Model

The new binomial P(1+ HR) model is completely broken:
- **Record:** 0 wins, 4,801 consecutive losses
- **Impact:** Affects ~5% of all picks
- **Status:** DO NOT USE in production
- **Fix:** Review and recalibrate binomial parameters in `src/predictor.py`

## How to Use These Results

### For Production Configuration
1. Read `QUICK_REFERENCE.txt` section on "PRODUCTION CONFIGURATION"
2. Update `app.py` to enable/disable props based on recommendations
3. After fixing HR model, re-run backtest to verify improvement
4. Deploy with monitoring

### For Further Analysis
1. `backtest_2025.json` contains raw data for any custom analysis
2. Run `python cross_tab.py` to regenerate the crosstab
3. Run `python -m src.backtester --report-only` to regenerate JSON report

### For Understanding the Metrics
1. `V008_BACKTEST_REPORT.md` explains each metric
2. `QUICK_REFERENCE.txt` has bias analysis and ROI projections
3. JSON report has structured data for programmatic access

## Backtest Methodology

- **Date Range:** 2025-04-01 to 2025-09-30 (183 days)
- **Walk-Forward:** Stats only from before prediction date (no data leakage)
- **Sample Size:** 91,090 plays (51.1% of 178,402 total predictions)
- **Non-plays:** 87,312 (48.9%) - correctly excluded bench players
- **Props Tested:** Hits, Hitter Fantasy Score, Home Runs, Total Bases, Pitcher K

## Next Steps

1. **URGENT:** Fix Home Runs model (2-3 hours)
   - Would add ~4,800 wins and bring accuracy to 54.1%
   
2. **HIGH:** Update production configuration (30 min)
   - Enable only profitable props
   - Disable broken models
   
3. **MEDIUM:** Re-run backtest (4-6 hours)
   - Verify HR fix works
   - Monitor for new issues
   
4. **ONGOING:** Implement live learning (Task 6)
   - Auto-detect accuracy issues
   - Auto-adjust weights based on live results

## Contact/Questions

See `BACKTEST_SESSION_SUMMARY.md` for detailed recommendations and `V008_BACKTEST_REPORT.md` for full analysis.

---

**Generated:** 2026-03-20
**Version:** v008 (HR binomial model + FS variance ratio 4.0)
**Status:** Analysis complete, HR fix required before production
