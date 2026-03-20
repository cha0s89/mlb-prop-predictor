# Non-Play Bias Fix — Documentation Index

**Session Date:** March 20, 2026
**Status:** ✅ COMPLETE & READY FOR DEPLOYMENT
**Critical Fix:** Yes, enables honest model validation

---

## Quick Start (5 Minutes)

**Problem:** 48.9% of backtest predictions were non-plays (batters with actual=0), creating fake 30pp direction gap

**Solution:** Added filter to remove non-plays from accuracy analysis

**Results:**
- OVERALL: 63.8% → **54.17%** (realistic)
- MORE: 40.4% → **56.55%** (+16pp!)
- LESS: 70.2% → **53.64%** (-16pp corrected)
- GAP: 30pp → **2.9pp** (FIXED!)

**Deploy:** Run `git add src/backtester.py && git commit -m "Fix non-play bias..."`

---

## Documentation Index

### 📋 For Product Owners / Managers
**Start here if:** You need to understand what happened and why it matters

1. **CHANGES_SUMMARY.txt** (← Start here)
   - What changed
   - Why it changed
   - Impact summary
   - Deployment checklist
   - ~5 min read

2. **data/backtest/OVERNIGHT_OPTIMIZATION_SUMMARY.md**
   - Executive summary
   - Key metrics before/after
   - Deployment checklist
   - Risk assessment
   - ~10 min read

### 👨‍💻 For Developers
**Start here if:** You need to understand the code changes and deployment process

1. **NONPLAY_FIX_TECHNICAL.md** (← Start here)
   - Problem statement
   - Solution design
   - Code changes (before/after)
   - Data flow diagrams
   - Testing approach
   - ~15 min read

2. **src/backtester.py** (lines 975 onwards)
   - Actual code implementation
   - `filter_nonplays()` function
   - Modified `generate_backtest_report()`
   - ~10 min code review

### 📊 For Analysts / Data Scientists
**Start here if:** You want to understand the projection biases and model performance

1. **data/backtest/NONPLAY_FIX_ANALYSIS.md** (← Start here)
   - Root cause analysis
   - Projection bias findings (detailed)
   - Direction accuracy patterns
   - Recommendations for improvements
   - ~20 min read

2. **data/backtest/backtest_2025_report.json**
   - Actual backtest results
   - Accuracy by prop type, rating, direction
   - Full statistics
   - Machine-readable format

### ⚙️ For Person Deploying
**Start here if:** You're about to deploy this fix to production

1. **NEXT_STEPS.md** (← Start here)
   - Immediate actions (today)
   - Short/medium term actions
   - Before going live checklist
   - Success criteria
   - ~15 min read

2. **CHANGES_SUMMARY.txt**
   - What to commit
   - What to test
   - Expected output
   - ~5 min read

---

## File Locations Reference

### Code Changes
```
src/backtester.py
  ├─ Line 975: filter_nonplays() — new function
  └─ Line 1006: generate_backtest_report() — modified
```

### Generated Reports
```
data/backtest/
  ├─ backtest_2025_report.json — regenerated with filter
  ├─ NONPLAY_FIX_ANALYSIS.md — technical deep dive
  └─ OVERNIGHT_OPTIMIZATION_SUMMARY.md — session summary
```

### Documentation Files
```
Root directory:
  ├─ NONPLAY_FIX_TECHNICAL.md — for developers
  ├─ NEXT_STEPS.md — for deployer
  ├─ CHANGES_SUMMARY.txt — quick reference
  └─ README_NONPLAY_FIX.md — this file
```

---

## Key Metrics

### Before Fix (Wrong — Includes Non-Plays)
| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | 63.8% | Too high (fake) |
| MORE Accuracy | 40.4% | Too low (fake) |
| LESS Accuracy | 70.2% | Too high (fake) |
| Direction Gap | 30pp | Broken |

### After Fix (Correct — Plays Only)
| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | 54.17% | Honest, realistic |
| MORE Accuracy | 56.55% | Good, room for improvement |
| LESS Accuracy | 53.64% | Normal range |
| Direction Gap | 2.9pp | Healthy |

---

## Root Causes Identified

### Primary: Non-Play Bias (FIXED)
- 48.9% of predictions had actual=0 (player didn't bat)
- These created automatic W/L outcomes
- Inflated LESS by 16pp, deflated MORE by 16pp
- **SOLUTION:** Filter out before analysis

### Secondary: Outdated League Averages (NOT YET FIXED)
- hits_per_game: 0.95 should be 1.44 (34% low)
- tb_per_game: 1.50 should be 2.35 (36% low)
- Causes systematic underprojection
- **SOLUTION:** Update LG dict in predictor.py lines 74-75
- **Impact:** Expected +2-4pp accuracy improvement

### Tertiary: Home Run Model (NEEDS REDESIGN)
- Projects 0.14 HR/game, but line is 0.5
- Always picks LESS, gets 0% accuracy
- v007 offset (+0.93) helps but doesn't solve
- **SOLUTION:** Consider binary logistic model
- **Priority:** Medium (after league averages)

---

## Deployment Flow

```
[User wakes up]
    ↓
[Reads CHANGES_SUMMARY.txt] ← 5 min
    ↓
[Reviews NONPLAY_FIX_TECHNICAL.md] ← 15 min
    ↓
[Commits src/backtester.py]
    git add src/backtester.py
    git commit -m "Fix non-play bias..."
    ↓
[Follows NEXT_STEPS.md]
    ├─ Deploy v007 weights
    ├─ Update league averages
    ├─ Re-run backtest
    └─ Monitor live picks
    ↓
[Verifies results match expectations]
    OVERALL: 54.2% ✓
    MORE: 56.5% ✓
    LESS: 53.6% ✓
    GAP: 2.9pp ✓
    ↓
[Goes live with real money]
```

---

## Testing Checklist

- [ ] Read all relevant documentation
- [ ] Review code changes in src/backtester.py
- [ ] Run: `python3 -m src.backtester` to regenerate report
- [ ] Verify overall accuracy: 54.2% (not 63.8%)
- [ ] Verify MORE accuracy: 56.5% (not 40.4%)
- [ ] Verify LESS accuracy: 53.6% (not 70.2%)
- [ ] Verify non-plays removed: 48.9%
- [ ] Commit changes
- [ ] Deploy to staging
- [ ] Monitor for 1-2 days
- [ ] Deploy to production
- [ ] Monitor live picks

---

## Success Criteria

### Immediate (After Deploying Filter)
- ✅ backtest_2025_report.json shows 54.17% accuracy
- ✅ Direction gap is 2.9pp (not 30pp)
- ✅ nonplay_filter section in report shows statistics

### Short Term (After Deploying v007 + Updates)
- ✅ League averages updated in predictor.py
- ✅ Backtest shows 56-57% accuracy
- ✅ 25+ live picks tracked
- ✅ Live accuracy matches backtest ±3pp

### Long Term (Before Real Money)
- ✅ 250+ live picks tracked
- ✅ Overall accuracy: 54-57%
- ✅ MORE accuracy: 54-58%
- ✅ LESS accuracy: 51-55%
- ✅ No direction bias detected

---

## Common Questions

**Q: Is the model broken?**
A: No. It showed 63.8% accuracy because of non-play bias. True accuracy is 54.17%, which is good.

**Q: What do I need to do right now?**
A: Commit the non-play filter fix. It's low-risk and enables honest validation.

**Q: When should I go live with real money?**
A: After deploying v007 weights + updating league averages, with 25-50 live picks tracked.

**Q: Will the model work if I don't update league averages?**
A: Yes, but accuracy will be 54% instead of 56-57%. Update them after you're confident in the fix.

**Q: What's the risk of deploying this?**
A: Low. The filter is post-processing only. Zero impact on live prediction code.

**Q: How long until I see improvements?**
A: v007 weights: immediate (if deployed). League averages: need to re-run backtest (~2-3 hours).

---

## Support

If you have questions:

1. **For why non-plays matter:** See NONPLAY_FIX_ANALYSIS.md
2. **For how to deploy:** See NEXT_STEPS.md
3. **For technical details:** See NONPLAY_FIX_TECHNICAL.md
4. **For quick reference:** See CHANGES_SUMMARY.txt
5. **For exec summary:** See OVERNIGHT_OPTIMIZATION_SUMMARY.md

All questions should be answerable from these five documents.

---

**Last Updated:** 2026-03-20
**Status:** ✅ Ready for Production
**Recommended Action:** Deploy immediately
**Estimated Effort:** 2-3 hours for full deployment
**Risk Level:** LOW
**Impact:** CRITICAL (enables honest model validation)

---

**Good luck! You've got this. 🚀**
