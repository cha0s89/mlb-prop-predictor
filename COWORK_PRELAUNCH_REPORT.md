# Pre-Opening Day Hardening Report
**Date:** 2026-03-20 | **Opening Day:** 2026-03-27 (7 days)

---

## What Was Done

### Priority 1 — Smart Prop Filtering (DONE)

Added `PP_TRADEABLE`, `PP_NEVER_SHOW`, and `is_tradeable_pick()` near the top of `app.py`.

**Non-tradeable combos blocked by default:**
- `(home_runs, LESS)` — PP almost never offers this
- `(stolen_bases, LESS)` — PP doesn't offer SB LESS
- `(total_bases, LESS)` — offered but 44% accuracy, unprofitable

**Where filtering is applied:**
- Sharp edges section (filters `filt` list before display)
- Projection analysis (filters `filtered` DataFrame before display)
- Best Plays cards (filters `top_plays` list)

**UX:** Shows an info banner listing what was filtered. A checkbox `Show all picks (incl. non-tradeable)` lets the user opt in for transparency. Default is OFF.

---

### Priority 2 — HTML Badge Leak in Expander Labels (VERIFIED CLEAN)

Audited all `st.expander()` calls in `app.py`. Every expander label already uses `grade_label()` (emoji + letter, e.g. `🟢 A`) — not the `badge()` HTML function. No raw HTML spans appear in any expander label. The `badge()` function is defined but unused in labels; it's safe to leave in place for potential future use in `st.markdown()` blocks.

---

### Priority 3 — Opening Day Checklist in Setup Tab (DONE)

Added a checklist section at the top of the Setup tab with 7 items:
1. ✅/❌ Odds API key configured
2. ✅/❌ API credits remaining (warns if < 50)
3. ✅/❌ FanGraphs batting cache (shows player count)
4. ✅/❌ FanGraphs pitching cache (shows pitcher count)
5. ✅/❌ Current weight version (reads `data/weights/current.json`)
6. ✅/❌ Last backtest accuracy (reads `data/backtest/backtest_2025_report.json`)
7. ✅/❌ PrizePicks API reachable (live test fetch)

Each failing check shows a hint explaining how to fix it. Summary line: `X/7 checks passing — Opening Day is March 27`.

---

### Priority 4 — Betting Rules Reminder Card (DONE)

Added a collapsible `📋 Betting Rules & Bankroll Guide` expander in the Find Edges tab, present in **both** the sharp-edges path and the projection-only path.

Contents:
- **BET:** Pitcher Ks, Fantasy Score, TB MORE, Hits, H+R+RBI, Batter Ks
- **AVOID:** TB LESS, HR LESS 0.5, SB LESS 0.5
- **SLIP RULES:** Mix 2–3 MORE + 2–3 LESS, max 2 picks/team, min B grade, 5–6 Pick Flex
- **BANKROLL:** 1–2% per slip, max 5 slips/day, stop if down 10%

Default `expanded=False` — visible but out of the way.

---

### Priority 5 — Slip Builder UX Improvements (DONE)

**MORE/LESS balance counter:** Shows `3 MORE / 2 LESS — ✅ good balance` (or ⚠️ warning if imbalanced by more than 1).

**Payout preview table:** Auto-calculates based on selected slip type and average pick confidence. Shows Correct, Multiplier, Payout ($), and estimated probability for each outcome tier.

**Expected Value display:** Shows `+$X.XX` or `-$X.XX` calculated from binomial probability × payout multipliers. Color-coded green/red.

**Disable Save button logic:**
- Disabled if fewer than 2 picks selected
- Disabled if any high-severity correlation warning exists
- Shows a caption explaining why it's disabled

---

### Priority 6 — Daily Log in Dashboard (DONE)

Added `get_daily_log_summary()` helper that reads the existing `data/daily_logs/<date>.json` files (already created by `save_daily_log()` on each slip save).

Added "Daily Log — Last 14 Days" section at the bottom of the Dashboard tab showing a table with: Date, Props available, A Picks, B Picks, Avg Edge %, Sharp Edges count.

---

## Files Modified

- `app.py` — all changes above (no other files touched)

## Files Not Modified (frozen per instructions)

- `src/predictor.py` — model code frozen during backtest run
- `src/backtester.py` — backtest running separately
- `data/weights/current.json` — weights frozen

## Syntax Check

```
python -m py_compile app.py → SYNTAX OK
```

---

## Pre-Launch Checklist for Opening Day (March 27)

- [ ] Backtest completes on local machine — review accuracy report
- [ ] Push `data/batting_stats_cache.csv` and `data/pitching_stats_cache.csv` to repo (or Streamlit Cloud)
- [ ] Verify Odds API key is set in Streamlit Cloud Secrets
- [ ] Open app morning of March 27, confirm Opening Day Checklist shows 7/7 green
- [ ] Confirm PrizePicks shows real regular-season props (not Spring Training)
- [ ] Test Find Edges tab — confirm HR LESS and SB LESS do NOT appear in Best Plays
- [ ] Build a 5-pick Flex slip with the payout preview visible before locking in
