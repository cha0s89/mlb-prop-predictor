# Signal Audit — Disconnected, Broken, and Underused Features

**Date:** 2026-03-23
**Purpose:** Hand this to Codex so it can fix every gap before Opening Day (March 27).

---

## CRITICAL — Broken Signals (Not Working At All)

### 1. `get_batter_trend()` Always Returns Neutral

**File:** `src/trends.py` line 38, called from `app.py` line 1451 and `src/headless_board.py` line 736
**Bug:** The pipeline always calls `get_batter_trend(0)` — passing `0` as the player_id.
Line 38 of trends.py: `if not PYBASEBALL_OK or not player_id: return _neutral_trend()`
Since `0` is falsy, every single batter gets `trend_multiplier: 1.0` (neutral).

**Impact:** The entire trend system (hot/cold streaks over last 7 vs 21 days) is dead code. No batter ever gets a hot or cold badge from real data. The Statcast-based trend analysis (lines 46-121) never executes.

**Fix:** Pass actual MLB player IDs. The pipeline has access to player data from FanGraphs — either map names to IDs or use a different trend data source that works by name.

---

### 2. Wind Direction Adjustment Is Dead Code

**File:** `src/weather.py` lines 266-289
**Bug:** `get_stat_specific_weather_adjustment()` checks for string keywords like `"out"`, `"in"`, `"left"`, `"center"` in the `wind_dir` field. But `fetch_game_weather()` sets `wind_dir` to compass labels like `"N"`, `"NNE"`, `"SW"` (via `_wind_direction_label()` at line 160). These will never match keywords like `"out"` or `"in"`.

**Impact:** All wind-direction-specific adjustments (blowing out boosts HR/TB by up to 5%, blowing in suppresses HR/TB by up to 5%) never fire. Wind speed still works via the basic `_calculate_weather_impact()`, but without direction awareness.

**Fix:** Add stadium orientation data (home plate bearing for each park — publicly available), then compute wind angle relative to field and classify as blowing_out / blowing_in / crosswind using trig.

---

### 3. Spring Multiplier Has No Seasonal Gating

**File:** `src/spring.py` — `get_spring_form_multiplier()`
**Bug:** There is no date check. After Opening Day (March 27), the function continues to fetch Spring Training stats (GAME_TYPE = "S") and apply them to regular-season projections.

**Impact:** A player who hit .500 SLG in his last 10 ST at-bats keeps getting a ~6% boost to all count props throughout April and beyond. The ST stats freeze in the API after the season starts, so stale ST data permanently inflates/deflates projections.

**Fix:** Add seasonal gate:
```python
if date.today() > date(2026, 3, 27):
    return {"spring_mult": 1.0, "badge": "regular_season"}
```

---

## HIGH — Signals Computed But Not Applied

### 4. Stat-Specific Weather Adjustment Is Display-Only

**File:** `src/headless_board.py` lines 710-713, `app.py` lines 1424-1429
**Issue:** `get_stat_specific_weather_adjustment()` computes a per-prop multiplier (e.g., +3% for HR on a hot day) but stores it as `p["weather_mult"]` for display only. It is NOT applied to the actual projection.

The projections use only the generic 3-multiplier system from `_calculate_weather_impact()` (weather_offense_mult, weather_hr_mult, weather_k_mult), which is cruder and doesn't differentiate between prop types beyond offense/HR/K.

**Impact:** The smarter stat-specific adjustments (different effects for hits vs total_bases vs earned_runs vs pitcher_strikeouts) exist in code but don't affect any prediction.

**Fix:** Either (a) replace the generic 3-multiplier system inside predictor.py with stat-specific multipliers, or (b) apply the stat-specific multiplier post-prediction (but be careful not to double-count with the generic multipliers already inside predictor.py).

---

### 5. Humidity and Precipitation Are Fetched But Never Used

**File:** `src/weather.py` — fetched at lines 150, 153, stored in weather dict
**Issue:** Humidity and precipitation data are pulled from Open-Meteo but never referenced in any calculation — not in `_calculate_weather_impact()`, not in `get_stat_specific_weather_adjustment()`, not in `predictor.py`.

**Impact:** Low — humidity's effect on baseball is debatable (the code even has a comment saying "myth debunked"). Precipitation matters only for postponements. But if we're paying the API call, might as well use humidity for HR probability (higher humidity = slightly less air resistance = slightly more carry).

---

### 6. Stolen Bases Ignores Opposing Pitcher Profile

**File:** `src/predictor.py` — `project_batter_stolen_bases()` (lines 1347-1374)
**Issue:** The function accepts `opp_p` (opposing pitcher profile) but never uses it. Every other batter projection function uses the opposing pitcher's stats (WHIP, FIP, HR9, etc.), but stolen bases completely ignores the pitcher.

**Impact:** Pitcher pickoff tendency, slide-step speed, and time to plate are major factors in stolen base success rate. Without opposing pitcher context, SB projections miss a key signal.

**Fix:** At minimum, use the opposing pitcher's BB9 (walk rate) as a proxy — pitchers who walk more batters tend to have more baserunners, creating more SB opportunities.

---

### 7. `opposing_lineup_woba` Not Wired for Pitcher Strikeouts

**File:** `src/predictor.py` line 761
**Issue:** `estimate_pitcher_batters_faced()` accepts `opposing_lineup_woba` to estimate how many batters a pitcher will face (stronger lineups = more BF). The earned_runs projection wires this correctly (line 1030: `opp_woba=opp.get("woba")`), but the pitcher strikeout projection passes `None` (line 877: `opposing_lineup_woba=None`).

**Impact:** Pitcher K projections don't account for how long the pitcher stays in the game when facing strong vs weak lineups. A pitcher facing the Dodgers lineup will face more batters (and get pulled earlier) than one facing a rebuilding team.

**Fix:** Pass `opp_woba` from the game context to `project_pitcher_strikeouts()` the same way earned_runs does.

---

## MEDIUM — Unused or Dead Code

### 8. `advanced_umpire_adjustment()` Never Called

**File:** `src/umpires.py` lines 69-148
**Issue:** A sophisticated zone-shape model exists that considers expansion inches, two-strike expansion, pitcher type (edge_worker vs power), and per-prop adjustments (K vs walks vs other). It's defined but never called — the pipeline only uses the simpler `get_umpire_k_adjustment()`.

**Impact:** The simple version only adjusts K rate ±0.9 per pitcher. The advanced version could differentiate between pitchers who work the edges (benefit more from wide zones) vs power pitchers, and apply walk adjustments more precisely.

---

### 9. `build_explanation()` Imported But Never Called

**File:** `src/explain.py` (272 lines), imported in `app.py` line 49
**Issue:** The function builds step-by-step explanations showing every adjustment factor. It's imported but never actually called in any code path.

**Impact:** Good transparency feature going to waste. Could be wired into the UI's pick detail expanders to show users why a prediction was made.

---

### 10. `barrel_rate` Weight in current.json Not Applied

**File:** `data/weights/current.json` line 107
**Issue:** The weights file defines a `barrel_rate` factor weight, but no corresponding weighting logic in predictor.py uses this weight to scale barrel-rate influence.

**Impact:** Barrel rate IS used as a signal in batter profile construction, but its relative weight compared to other signals isn't configurable via the weights file like other factors are.

---

### 11. `std_pa` and `std_bf` Computed But Discarded

**File:** `src/predictor.py` — `estimate_plate_appearances()` (line 524-529) and `estimate_pitcher_batters_faced()` (line 559-597)
**Issue:** Both functions compute standard deviation of PA/BF estimates but only the mean is ever extracted. The variance information is thrown away.

**Impact:** Could be used for confidence scoring — high-variance PA estimates should produce wider projection distributions and lower confidence.

---

## LOW — Working But Could Be Better

### 12. Calibration Covers Only 4 of 17+ Prop Types

**File:** `src/predictor.py` lines 47-52 (DEFAULT_CALIBRATION_BLEND_WEIGHTS)
**Issue:** Only hits, total_bases, pitcher_strikeouts, and hitter_fantasy_score have configured calibration blends. The other 13+ props (home_runs, rbis, runs, stolen_bases, etc.) default to 0.0 (100% theoretical, 0% empirical calibration).

**Impact:** The 4 calibrated props have demonstrated accuracy improvements (TB went from 56.1% → 65.8% with empirical blending). Extending calibration to more props could yield similar gains.

---

### 13. Calibration File Frozen at v015

**File:** `data/weights/calibration_v015.json`
**Issue:** The empirical calibration tables were generated during v015 but the model is now v018. The `rebuild_calibration_tables()` function in autolearn.py can regenerate these, but it hasn't been run since v015.

**Fix:** Run calibration rebuild after accumulating enough regular-season data.

---

## SUMMARY — Priority Order for Codex

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | `get_batter_trend(0)` always neutral | CRITICAL | Medium — need player ID mapping |
| 2 | Wind direction dead code | CRITICAL | Medium — need stadium orientations |
| 3 | Spring multiplier no seasonal gate | CRITICAL | Easy — add date check |
| 4 | Stat-specific weather display-only | HIGH | Medium — choose integration path |
| 5 | Stolen bases ignores pitcher | HIGH | Easy — add BB9/control signal |
| 6 | Pitcher K missing opp_lineup_woba | HIGH | Easy — copy from earned_runs path |
| 7 | Humidity/precip fetched but unused | MEDIUM | Easy if desired |
| 8 | Advanced umpire model unused | MEDIUM | Medium — wire into pipeline |
| 9 | build_explanation() never called | LOW | Easy — wire into UI |
| 10 | barrel_rate weight disconnected | LOW | Easy — add weighting logic |
| 11 | std_pa/std_bf discarded | LOW | Medium — feed into confidence |
| 12 | Calibration only 4 props | LOW | Medium — needs data accumulation |
| 13 | Calibration frozen at v015 | LOW | Easy — run rebuild |
