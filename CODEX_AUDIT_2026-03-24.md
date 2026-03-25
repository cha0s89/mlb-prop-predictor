# Codex Audit Report — March 24, 2026
## Full Code Audit + LESS Bias Root Cause Analysis

Reviewed: 11 commits (95cc00f → 3c04cc1), all source files, weights, distributions, tests.

---

## PART 1: LESS PICK BIAS — ROOT CAUSES

The model is producing too many LESS picks. Here are the compounding causes, ranked by impact.

### BIAS-1: Confidence Floor Asymmetry [CRITICAL — HIGHEST IMPACT]
**File:** `data/weights/current.json` lines 34-64

The per-prop confidence floors require much HIGHER confidence to surface a MORE pick than a LESS pick:

| Prop | MORE floor | LESS floor | Gap |
|------|-----------|-----------|-----|
| hits | 0.90 | 0.72 | −0.18 |
| total_bases | 0.90 | 0.64 | −0.26 |
| hitter_fantasy_score | 0.56 | 0.68 | +0.12 (LESS harder here) |
| pitcher_strikeouts | 0.66 | 0.66 | 0.00 (balanced) |
| All other props | 0.60 | 0.60 | 0.00 |

**Effect:** For hits, a pick needs 90% confidence to show as MORE but only 72% for LESS. For total_bases, 90% vs 64%. This is the single biggest filter driving LESS dominance on the board.

**Fix:** Either equalize the floors, or at minimum narrow the gap. Consider 0.72/0.72 for hits and 0.70/0.64 for TB as a starting point, then let backtest calibrate.

---

### BIAS-2: Systematic Negative Projection Offsets [CRITICAL]
**File:** `data/weights/current.json` lines 15-33

Nearly every prop type has a negative offset that subtracts from the raw projection before probability is calculated:

| Prop | Offset |
|------|--------|
| hitter_fantasy_score | −2.5 |
| total_bases | −0.6 |
| pitcher_strikeouts | −0.5 |
| walks | −0.5 |
| rbis | −0.45 |
| doubles | −0.4 |
| singles | −0.2 |
| home_runs | −0.15 |
| hits_runs_rbis | −0.1 |
| runs | −0.05 |
| hits | −0.0 |

**Effect:** Subtracting from projections shifts the mean BELOW the PrizePicks line more often, which mechanically increases P(under) and decreases P(over). When you shift the projection down by 0.5 on a line of 5.5, you're handing the model a ~5-8% head start toward LESS. This compounds across all prop types.

**Fix:** Re-evaluate whether these offsets are justified by backtest accuracy. If the model was over-projecting MORE in backtests, offsets are correct. But if the offsets were fitted to a period with LESS-skewed outcomes, they'll perpetuate the bias. Consider zeroing them out and letting the confidence shrinkage handle calibration instead.

---

### BIAS-3: PP_NEVER_SHOW and PP_TRADEABLE Direction Filters [HIGH]
**File:** `app.py` lines 645-664

The tradeable config structurally blocks several MORE directions:

```python
PP_NEVER_SHOW = {
    ("home_runs", "LESS"),      # Blocks HR LESS
    ("stolen_bases", "LESS"),   # Blocks SB LESS
    ("total_bases", "LESS"),    # Blocks TB LESS ← removes TB LESS entirely
    ("hitter_fantasy_score", "MORE"),  # Blocks FS MORE ← removes FS MORE entirely
}

PP_TRADEABLE = {
    "total_bases": {"directions": ["MORE"], ...},  # Only MORE allowed
    ...
}
```

**Effect:** TB can only show as MORE (good), but FS can only show as LESS. Since fantasy score is one of the highest-volume props, blocking FS MORE while allowing FS LESS adds a large batch of LESS-only picks to every board.

**Fix:** If FS MORE has bad accuracy, the floor should handle it (it's at 0.56 already). Consider removing `("hitter_fantasy_score", "MORE")` from PP_NEVER_SHOW and letting the confidence floor gate it naturally. If FS MORE truly can't be profitable, document why.

---

### BIAS-4: Continuity Correction Creates Asymmetry on Skewed Distributions [MEDIUM]
**File:** `src/distributions.py` lines 279-306

For Gamma (fantasy score) and Normal (pitching outs, H+R+RBI):

```python
def prob_over_gamma(line, mu, var_ratio):
    return 1 - gamma.cdf(line + 0.5, shape, scale=scale)   # penalizes MORE

def prob_under_gamma(line, mu, var_ratio):
    return gamma.cdf(line - 0.5, shape, scale=scale)        # penalizes LESS
```

Both sides get a 0.5 penalty, which is correct for discrete push handling. However, because the Gamma distribution is right-skewed, the probability density is higher on the left side of the line than the right. This means the `-0.5` on under removes LESS mass from a denser region than the `+0.5` on over removes from a sparser region.

After normalization (`resolved_over / (resolved_over + resolved_under)`), the net effect is small but consistently favors LESS on Gamma-distributed props (fantasy score).

**Fix:** For continuous distributions (Gamma), consider evaluating at the line directly rather than using the ±0.5 continuity correction. The correction is designed for discrete count props, not continuous scores. Alternatively, use a narrower band (±0.25).

---

### BIAS-5: Fantasy Score Floor Compresses Low Projections [MEDIUM]
**File:** `src/predictor.py` line 1970

```python
mu = max(fantasy_per_pa * exp_pa + fantasy_per_game, 2.0)
```

A hard floor of 2.0 fantasy points means weak projections (1.0-1.9) get bumped up. When these players have lines at 5.0 or higher, the floor makes the projection closer to the line than it should be, reducing the LESS signal. But for players with lines at 3.0-4.0, the floor artificially inflates the projection, reducing the gap and weakening LESS confidence.

The net effect is mixed, but the floor prevents the model from making strong LESS calls on genuinely weak performers.

**Fix:** Replace with a proportional floor: `mu = max(projection, line * 0.15)` to scale with the actual line.

---

### BIAS-6: BABIP Regression Pull-Down [LOW-MEDIUM]
**File:** `src/predictor.py` lines 1167-1171

```python
babip_delta = babip - (xba + 0.050)
if abs(babip_delta) > 0.030:
    reg_avg *= (1 - babip_delta * 0.15)
```

When BABIP > xBA + 0.080 (lucky hitter), the multiplier pulls the projection DOWN. When BABIP < xBA + 0.020 (unlucky hitter), it pulls UP. The formula is mathematically symmetric in direction, but in practice more MLB hitters have elevated BABIP than suppressed BABIP (selection bias — good hitters play more). This creates a net downward pressure on hit projections across the league.

**Fix:** Consider capping the effect at ±3% (`max(-0.03, min(0.03, babip_delta * 0.15))`), or only applying the regression when BABIP is significantly elevated (one-sided correction for luck, not skill).

---

## PART 2: SIGNAL AUDIT FOLLOW-UP — What Got Fixed, What Didn't

### FIXED ✅

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 2 | Wind direction dead code (compass vs keyword mismatch) | **FIXED** | `weather.py` lines 227-274: new `_classify_field_relative_wind()` uses compass bearings + stadium orientation |
| 4 | Pitcher K missing opp_lineup_woba | **FIXED** | `predictor.py` line 2299 passes `opp_lineup_context`; function uses it at lines 850-857 |
| 6 | Walk-forward validation in backtester | **CORRECT** | `backtester.py` lines 612-649: `_get_season_batting()` fetches stats up to (not including) backtest date |

### NOT FIXED ❌

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 1 | `get_batter_trend(0)` always returns neutral | **NOT FIXED** | `app.py` line 1651 and `headless_board.py` line 848 both still call `get_batter_trend(0)`. The `0` is a falsy player_id, so `trends.py` line 38 immediately returns `_neutral_trend()`. The entire hot/cold streak system is still dead. |
| 5 | Stolen bases ignores opposing pitcher | **NOT FIXED** | `predictor.py` line 1682: `project_batter_stolen_bases(b, park=None)` — no `opp_p` parameter. No pitcher pickoff/delivery data used. |
| 8 | `advanced_umpire_adjustment()` never called | **NOT FIXED** | `umpires.py` lines 69-148: defined but unused. Only simple `get_umpire_k_adjustment()` is called. |
| 9 | `build_explanation()` transparency not wired | **NOT FIXED** | `explain.py`: imported but never called in app.py or headless_board.py |

### PARTIALLY FIXED ⚠️

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 3 | Spring multiplier has no seasonal gate | **PARTIALLY FIXED** | `apply_seasonal_spring_blend()` at `spring.py` lines 409-481 now has proper day-based + sample-based decay after Opening Day. It IS called in both `app.py` line 1637 and `headless_board.py` line 834. However, the function defaults to `date.today()` for game_date, which works for live boards but may not correctly handle backtest dates unless explicitly passed. |

---

## PART 3: CODE QUALITY AUDIT — New Commits (95cc00f → 3c04cc1)

### 3a. Commits Reviewed — All Passing ✅

| Commit | Description | Verdict |
|--------|-------------|---------|
| 95cc00f | Preseason training pipeline + promote tuned models | ✅ Solid |
| bb51c81 | Sharp consensus improvements + v022 | ✅ Good devigging updates |
| f147760 | Hitter production props + v023 | ✅ Well-structured |
| a2d7c4b | Repair preseason tracking in nightly grading | ✅ Correct fix |
| f31d0af | Lineup-aware projections + v024 | ✅ Strong addition |
| 50f04a7 | Lineup-aware resumable backtests | ✅ Nice engineering |
| 82d4bdd | Record backtest report | ✅ Data only |
| a41189c | Historical umpire context in backtests | ✅ Good |
| 484ff77 | Game-specific starter context | ✅ Proper team_context wiring |
| 228fa0c | Payout table alignment with PrizePicks | ✅ Correct payouts |
| 3c04cc1 | Sharp availability gate (dynamic, not hardcoded date) | ✅ Smart fix for Opening Day |

### 3b. Issues Found in Current Code

**ISSUE-A: `src/slips.py` is truncated in working tree [CRITICAL]**

The file is corrupted — cut off mid-line at `roi = (net_pr`. Missing the entire return dict for `get_slip_pnl()` and the `init_slips_table()` call. The committed version on HEAD is intact.

**Fix:** Run `git checkout -- src/slips.py` to restore from the last commit.

**ISSUE-B: Early-season IP discount not applied to hits_allowed [MEDIUM]**

`_early_season_ip_discount()` is called for:
- pitcher_strikeouts (line 890) ✅
- pitching_outs (line 978) ✅
- earned_runs (line 1019) ✅
- walks_allowed (line 1072) ✅
- hits_allowed — **NOT CALLED** ❌

`project_pitcher_hits_allowed()` (line 1090) uses `avg_ip = ip / gs` based on last season's full workload. Since early-season starters throw fewer IP, this overpredicts hits allowed in the first few weeks.

**Fix:** Add early-season discount to `project_pitcher_hits_allowed()` the same way it's applied to the other pitcher props.

**ISSUE-C: `_early_season_ip_discount` hardcodes March 27, 2026 [LOW]**

`predictor.py` line 781: `season_start = date(year, 3, 27)` — works for 2026 but will be wrong in 2027. Should use the same `get_opening_day_for_year()` function from `spring.py`.

**ISSUE-D: Unused scipy import [LOW]**

`predictor.py` line 32: `from scipy import stats as sp_stats  # kept for potential future use` — dead import, adds startup overhead.

**ISSUE-E: Line ending noise in working tree**

~15,000 lines of CRLF/LF changes across weight files and app.py. Not actual code changes, just Windows line ending mismatches. Consider adding a `.gitattributes` file:
```
*.py text eol=lf
*.json text eol=lf
*.ps1 text eol=crlf
```

---

## PART 4: PRIORITY FIX ORDER FOR CODEX

Opening Day is March 27. Here's the recommended order:

### Before Opening Day (March 25-26)

1. **Restore slips.py** — `git checkout -- src/slips.py` (30 seconds)
2. **Fix confidence floor asymmetry** — Equalize or narrow hits/TB MORE vs LESS gaps in `current.json` (15 min)
3. **Remove FS MORE from PP_NEVER_SHOW** — Let the 0.56 floor gate it instead (5 min)
4. **Re-evaluate projection offsets** — Zero out or halve the negative offsets, re-run backtest to compare accuracy (1-2 hours)
5. **Fix get_batter_trend(0)** — Pass actual MLB player ID from the game context (30 min)

### Week 1 of Season

6. **Add _early_season_ip_discount to hits_allowed** (10 min)
7. **Add opp_p to stolen_bases projection** — Extract pitcher pickoff/delivery metrics (1 hour)
8. **Fix continuity correction for Gamma** — Evaluate at line directly instead of ±0.5 (30 min)
9. **Wire build_explanation()** for transparency (1 hour)
10. **Add .gitattributes** to stop line ending noise (5 min)

### Later

11. Wire `advanced_umpire_adjustment()`
12. Replace hardcoded Opening Day date with `get_opening_day_for_year()`
13. Remove dead scipy import
14. Proportional fantasy score floor instead of absolute 2.0
