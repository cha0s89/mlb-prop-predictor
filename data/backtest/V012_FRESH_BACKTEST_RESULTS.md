# V012 Fresh Backtest Results — Full 2025 Season (Played Games Only)

Generated: 2026-03-20

---

## Cross-Tab: Prop Type × Direction Accuracy

| Prop | Direction | W | L | Accuracy | Status |
|------|-----------|---|---|----------|--------|
| hits | LESS 1.5 | 15,805 | 8,788 | **64.3%** | ✅ BET |
| total_bases | MORE 1.5 | 3,995 | 2,359 | **62.9%** | ✅ BET |
| pitcher_strikeouts | MORE 4.5 | 1,230 | 764 | **61.7%** | ✅ BET |
| hitter_fantasy_score | LESS 7.5 | 15,662 | 12,413 | **55.8%** | ✅ BET |
| pitcher_strikeouts | LESS 4.5 | 1,086 | 917 | **54.2%** | ⚠️ BORDERLINE |
| hitter_fantasy_score | MORE 7.5 | — | — | **53.1%** | ❌ AVOID |
| total_bases | LESS 1.5 | — | — | **43.6%** | ❌ AVOID |
| home_runs | LESS | — | — | **0%** | ❌ EVAL BUG (see notes) |

---

## Trading Recommendations

### BET (above 54.2% break-even for 5-Pick Flex)

1. **Hits LESS 1.5** — 64.3% (15,805W / 24,593 plays)
   - Largest sample, highest confidence. Strong systematic edge.
   - Edge source: model correctly identifies low-contact matchups.

2. **TB MORE 1.5** — 62.9% (3,995W / 6,354 plays)
   - Reliable signal. Multi-base hits (2B+HR) are well-modeled by xSLG.

3. **Pitcher Ks MORE 4.5** — 61.7% (1,230W / 1,994 plays)
   - Smaller sample but consistent. Best for high-K arms vs strikeout-prone lineups.

4. **FS LESS 7.5** — 55.8% (15,662W / 28,075 plays)
   - Marginal but profitable at scale. Large sample validates the edge.

### CAUTION

5. **Pitcher Ks LESS 4.5** — 54.2% (1,086W / 2,003 plays)
   - Exactly at break-even for 5-Pick Flex. Small sample. Use sparingly.
   - Best against soft-contact pitchers vs strong contact lineups.

### AVOID

- **FS MORE 7.5** — 53.1% — below break-even, not worth the vig
- **TB LESS 1.5** — 43.6% — systematic model bias toward over-projecting TB
- **HR LESS** — evaluation bug (see below), do not use until fixed
- **SB LESS** — not enough sample / low frequency prop

---

## Projection Bias Analysis

| Prop | Mean Projection | Mean Actual | Bias | Action |
|------|-----------------|-------------|------|--------|
| hitter_fantasy_score | ~8.1 | ~7.4 | +0.7 over | Subtract 0.5–0.7 from FS projections |
| hits | ~1.6 | ~1.5 | +0.1 slight over | Minor — acceptable |
| total_bases | ~2.1 | ~1.9 | +0.2 over | Explains why TB LESS 43.6% |
| pitcher_strikeouts | ~4.8 | ~4.9 | −0.1 under | Good calibration |

**Key finding:** The model systematically over-projects hitter output, particularly total bases. This explains:
- TB LESS hits only 43.6% (model says MORE too often)
- FS MORE only 53.1% (model inflated projections cause too many MORE calls)

**Recommended fix (for autolearn.py):** Apply a −0.5 calibration offset to hitter_fantasy_score projections and a −0.15 offset to total_bases projections before generating picks.

---

## PP_NEVER_SHOW Update

Based on these results, `("hitter_fantasy_score", "MORE")` has been added to `PP_NEVER_SHOW` in app.py.

Full current `PP_NEVER_SHOW`:
```python
PP_NEVER_SHOW = {
    ("home_runs", "LESS"),        # Systematic eval bug
    ("stolen_bases", "LESS"),     # Unprofitable
    ("total_bases", "LESS"),      # 43.6% — systematic over-projection
    ("hitter_fantasy_score", "MORE"),  # 53.1% — below 54.2% threshold
}
```

---

## HR Evaluation Bug

**Problem:** `cross_tab.py` was filtering all `actual=0` records as "non-plays". For home runs, `actual=0` means the batter played but didn't homer — which is a valid **LESS win**, not a non-play.

**Effect:** All HR LESS picks were being dropped from the accuracy calculation, producing 0% (empty denominator). The actual HR LESS accuracy is unknown until this is re-run.

**Fix applied:** `cross_tab.py` now only filters `actual=0` for batter props where zero genuinely means no game played (hits, TB, FS, SB, HRR, runs, RBI, batter Ks). Home runs and pitcher props are exempt from the filter.

**Action required:** Re-run `python cross_tab.py` after this fix to get the true HR LESS accuracy and reassess whether to keep it in `PP_NEVER_SHOW`.

---

## Sample Size Notes

- **Hits LESS** and **FS LESS** have the largest samples (24K+ and 28K+ plays respectively) — highest statistical confidence
- **Pitcher Ks** samples are smaller (~2K each direction) but still statistically meaningful
- **TB MORE** at 6K+ plays is solid
- Minimum 1,000 plays required before drawing conclusions — all tradeable props meet this bar

---

## Next Steps

1. Re-run `python cross_tab.py` to get corrected HR accuracy (bug now fixed)
2. Apply calibration offsets in `src/autolearn.py`:
   - `hitter_fantasy_score`: −0.5 projection offset
   - `total_bases`: −0.15 projection offset
3. Consider adding hits_runs_rbis to the tradeable list once sample is large enough
4. Monitor live results vs these backtest benchmarks — flag if any prop drops >3% below backtest accuracy after 100+ live plays
