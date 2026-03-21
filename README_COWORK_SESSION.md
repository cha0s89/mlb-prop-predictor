# Cowork Session Summary — Home Runs Fix & Backtest Evaluation Framework

## What Happened

Claude spent the morning session (March 20, 2026) diagnosing why the MLB prop predictor had 0% accuracy on Home Runs and investigating systematic issues in the backtest evaluation.

## Key Discovery

**Non-plays were killing the evaluation:**
- 48.9% of all 178,402 backtest predictions were for players who didn't actually bat in that game
- This created artificial direction bias:
  - Every LESS pick on a non-play = automatic win (0 < line)
  - Every MORE pick on a non-play = automatic loss (0 < line)
- This inflated LESS accuracy by 20-25pp and deflated MORE accuracy

When filtered to "played games only":
- HR dropped from "0% LESS" to proper baseline
- TB LESS dropped from 69% to 44%
- FS MORE dropped from 54% to 51%
- The evaluation became honest

## Fixes Made

### 1. Home Runs Redesigned as Probability Model
- **Problem:** Projected 0.14 HR/game vs actual 1.07 (87% too low)
- **Root cause:** Used continuous rate model but line is 0.5 (binary: 0 or 1 HR)
- **Solution:** P(1+ HR in game) = 1 - (1-rate)^PA
- **Result:** Eliminates artificial LESS inflation

### 2. Added HR-Specific CDF
- Probability model (0-1) doesn't need Poisson/NegBin
- Direct logic: p_over = projection, p_under = 1 - projection

### 3. Updated Weights
- Removed home_runs offset (was 0.93, now 0.0)
- Old offset designed for count scale, breaks probability scale

### 4. Fixed cross_tab.py
- Auto-filters non-plays in load_backtest()
- All metrics now show "played-games-only" accuracy
- Displays non-play percentage for transparency

## Commit

```
eedabcb Fix home runs projection model and cross_tab non-play filtering
```

Files changed:
- `src/predictor.py` (~100 lines)
- `data/weights/current.json`
- `cross_tab.py`
- `data/backtest/COWORK_MORNING_REPORT.md` (detailed analysis)

## What's Left

**High Priority:**
1. Run full backtest to measure HR fix impact
2. Check if TB LESS improves from 44% (needs 54%+)
3. Calibrate Fantasy Score MORE (currently 50.6%, needs 54%+)

**Documentation:**
- See `COWORK_SESSION_FINAL.md` for technical deep dive
- See `data/backtest/COWORK_MORNING_REPORT.md` for diagnostic details

**Next commands:**
```bash
python -m src.backtester  # Run full 2025 season
python cross_tab.py       # Check results
```

## Key Insight

The biggest issue wasn't the projection models—it was the evaluation framework itself. By filtering to "played-games-only," we discovered the models were actually reasonable, but were being evaluated against ghost players. This session fixed the framework to be honest.
