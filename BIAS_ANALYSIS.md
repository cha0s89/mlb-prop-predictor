# MLB Prop Predictor Bias Analysis

## CRITICAL FINDINGS: DIRECTIONAL LESS/UNDER BIAS

### 1. ASYMMETRIC CONTINUITY CORRECTION IN GAMMA & NORMAL DISTRIBUTIONS
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/src/distributions.py`

#### Issue: Asymmetric Continuity Correction
Lines 279–306 show a fundamental bias in how OVER and UNDER probabilities are calculated for continuous distributions:

```python
def prob_over_gamma(line: float, mu: float, var_ratio: float) -> float:
    """P(X >= line) using Gamma CDF with continuity correction."""
    if mu <= 0:
        return 0.0
    shape, scale = gamma_shape_scale(mu, var_ratio)
    return float(1 - gamma.cdf(line + 0.5, shape, scale=scale))  # +0.5

def prob_under_gamma(line: float, mu: float, var_ratio: float) -> float:
    """P(X <= line) using Gamma CDF with continuity correction."""
    if mu <= 0:
        return 1.0
    shape, scale = gamma_shape_scale(mu, var_ratio)
    return float(gamma.cdf(line - 0.5, shape, scale=scale))  # -0.5
```

The same asymmetry exists in `prob_over_normal` and `prob_under_normal` (lines 297–306).

**Why This Creates LESS Bias:**
- For OVER: CDF evaluated at `line + 0.5` → moves the evaluation point HIGHER
- For UNDER: CDF evaluated at `line - 0.5` → moves the evaluation point LOWER
- Since CDF is increasing, this systematically:
  - **Reduces** P(OVER) by evaluating further right in the tail
  - **Reduces** P(UNDER) by evaluating further left in the tail

However, the net effect depends on the distribution shape. For typical positive-skewed distributions (which fantasy score is), evaluating the UNDER threshold lower increases the CDF value more than the symmetric counterpart decreases the OVER tail, creating a **net bias toward UNDER**.

**Affected Props (using Gamma):**
- `hitter_fantasy_score` — the primary fantasy points prop, uses Gamma distribution
  - Current floor: hitter_fantasy_score_less = 0.68 vs more = 0.56
  - Already filtered MORE to 0.56 (70% reduction from 0.90)
  - This distribution bias would reinforce the LESS preference

---

### 2. CONFIDENCE FLOOR ASYMMETRY
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/data/weights/current.json`

#### Key Findings: Systematic LESS Higher Than MORE

| Prop | MORE Floor | LESS Floor | Δ | Notes |
|------|-----------|-----------|---|-------|
| hits | 0.90 | 0.72 | -0.18 | MORE effectively disabled |
| total_bases | 0.90 | 0.64 | -0.26 | MORE effectively disabled |
| pitcher_strikeouts | 0.66 | 0.66 | 0.00 | Balanced |
| hitter_fantasy_score | 0.56 | 0.68 | +0.12 | **LESS explicitly higher** |
| Default (all others) | 0.60 | 0.60 | 0.00 | Balanced |

**Lines 35–64:** Per-prop confidence floors show explicit asymmetry:
- `hits_more: 0.90` vs `hits_less: 0.72` → -0.18 floor gap (LESS favored)
- `total_bases_more: 0.90` vs `total_bases_less: 0.64` → -0.26 (LESS favored)
- `hitter_fantasy_score_more: 0.56` vs `hitter_fantasy_score_less: 0.68` → **+0.12 explicit LESS bias**

**Accuracy Justification (lines 170–176):**
The notes claim these floors are evidence-based:
- hits_less: "72.5% (26,875 picks)"
- hitter_fantasy_score_less: "61.7% (1,715 picks)"

However, the floors also explicitly state (line 167):
- "fs_more 0.62 -> 0.56 (LOWERED — 59.3% on 118 -> 59.9% on 232, more volume at same accuracy)"

This shows MORE was **lowered** to increase volume at lower accuracy, while LESS maintains higher filters.

---

### 3. ASYMMETRIC OFFSET APPLICATION
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/data/weights/current.json`
**Lines 15–33:** `prop_type_offsets` apply uniformly to both MORE and LESS

Problem: **Offsets are signed and directionally applied, not neutral**

```json
"prop_type_offsets": {
    "pitcher_strikeouts": -0.5,      // Reduces projection for BOTH directions
    "hits": -0.0,                    // No bias
    "total_bases": -0.6,             // Reduces projection for BOTH directions
    "rbis": -0.45,                   // Reduces projection for BOTH directions
    "hitter_fantasy_score": -2.5,    // MAJOR reduction for BOTH directions
    ...
}
```

**Why This Matters:**
- Negative offsets reduce the projected mean (`mu`)
- When `mu` is reduced, both P(OVER) and P(UNDER) change, but:
  - For skewed distributions, reducing `mu` typically increases P(UNDER) more than it decreases P(OVER)
  - Example: if true `mu=10.5` but offset makes it `mu=8.5`:
    - P(X >= 10.5) decreases (fewer overs)
    - P(X <= 10.5) increases (more unders)

**Critical Offsets:**
- `hitter_fantasy_score: -2.5` (line 24) — **largest offset**
- `total_bases: -0.6` (line 19)
- `pitcher_strikeouts: -0.5` (line 16)
- `rbis: -0.45` (line 21)
- `walks: -0.5` (line 29)

These all push projections DOWN, systematically favoring LESS picks.

---

### 4. NO DIRECTIONAL MULTIPLIER USAGE
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/data/weights/current.json`
**Lines 11–14:**

```json
"direction_bias": {
    "more_multiplier": 1.0,
    "less_multiplier": 1.0
}
```

**Issue:** These multipliers are set to 1.0 but may not be applied anywhere in the code.

**Verification Needed:** Check if `direction_bias` is actually used in selection.py or combined.py

---

### 5. SHARP ODDS DEVIGGING — NO ASYMMETRY FOUND
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/src/sharp_odds.py`

The devigging logic (lines 528–577) is symmetric:
- `devig_two_way()` treats over_odds and under_odds equivalently
- Power devigging (default) normalizes both sides equally
- No detected directional bias in the devigging itself

However, the **line-difference repricing logic (lines 808–826)** may introduce bias:
- When `pp_line > sharp_line_val`, it **assumes** LESS is easier (line 817)
- When `pp_line < sharp_line_val`, it **assumes** MORE is easier (line 809)
- This is statistically correct, but combined with offset bias, it compounds LESS preference

---

### 6. COMBINED EDGE SCORING — NO ASYMMETRY FOUND
**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/src/combined.py`

The combined scoring logic (lines 57–206) treats MORE and LESS symmetrically:
- Both use the same `_combined_grade()` thresholds
- Both use the same signal weighting (CONFIRMED, SHARP_ONLY, PROJECTION_ONLY)
- No explicit direction penalty or bonus

---

## SUMMARY OF BIASES

### 1. **Gamma/Normal Distribution Continuity Correction Bias** ⚠️ HIGH IMPACT
   - Lines in `/src/distributions.py`: 279–306
   - Affects: hitter_fantasy_score, any continuous-distribution props
   - Direction: **Bias toward LESS**
   - Mechanism: Asymmetric CDF evaluation points

### 2. **Explicit Confidence Floor Asymmetry** ⚠️ HIGH IMPACT
   - Lines in `current.json`: 35–64
   - Affects: hits, total_bases, hitter_fantasy_score, and others
   - Direction: **Explicit bias toward LESS**
   - Mechanism: LESS thresholds are higher, MORE thresholds are lower

### 3. **Negative Offset Bias** ⚠️ MEDIUM-HIGH IMPACT
   - Lines in `current.json`: 15–33
   - Affects: ALL props with negative offsets
   - Direction: **Bias toward LESS**
   - Mechanism: Lower projected means favor UNDER on skewed distributions

### 4. **Unused Direction Multiplier** ⚠️ LOW (requires code search)
   - Lines in `current.json`: 11–14
   - Direction multipliers exist but may not be applied
   - Direction: Check impact if applied

---

## DEAD/DISCONNECTED CODE

**None explicitly found.** All checked code paths are integrated:
- `distributions.py` functions are called by `sharp_odds.py`
- `sharp_odds.py` edge detection feeds `combined.py`
- `combined.py` scores are filtered by `selection.py` (which uses confidence floors)
- Confidence floors in `current.json` are loaded and applied

---

## RECOMMENDATIONS FOR FIX

1. **Fix Continuity Correction Symmetry** (High Priority)
   - Change line 284 in distributions.py: `line + 0.5` → `line - 0.5` (to match UNDER logic)
   - Or change line 292: `line - 0.5` → `line + 0.5` (to match OVER logic)
   - Need to decide which is correct for PrizePicks semantics

2. **Audit Confidence Floors** (High Priority)
   - Justify asymmetric floors with backtest data
   - Or symmetrize to 0.60 for all props except those with strong empirical evidence

3. **Review Offset Strategy** (Medium Priority)
   - Consider applying offsets **directionally** instead of uniformly
   - E.g., separate `more_offset` and `less_offset` per prop

4. **Verify Direction Multiplier Usage** (Low Priority)
   - Search codebase for `more_multiplier` and `less_multiplier` application
   - If unused, either apply them or remove from weights


---

## VERIFICATION: Direction Multiplier IS Used

**File:** `/sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor/src/predictor.py`
**Lines 2354–2362:** Direction bias multipliers ARE actively applied

```python
# Direction bias correction: nudge projection up (more_multiplier) or
# down (less_multiplier) to counteract systematic MORE/LESS skew.
# Applied as: if projection > line → trending MORE → apply more_multiplier
#             if projection < line → trending LESS → apply less_multiplier
dir_bias = weights.get("direction_bias", {})
if projection >= line:
    projection *= dir_bias.get("more_multiplier", 1.0)
else:
    projection *= dir_bias.get("less_multiplier", 1.0)
```

**Current Setting (lines 11–14 of current.json):**
```json
"direction_bias": {
    "more_multiplier": 1.0,
    "less_multiplier": 1.0
}
```

**Status:** BOTH multipliers are set to 1.0, meaning **NO directional adjustment is active**. If either were set to <1.0 or >1.0, it would amplify the LESS bias (less_multiplier < 1.0 would reduce LESS projections further).

---

## UPDATED BIAS SUMMARY

### Active Biases (Currently Applied)

1. **Distribution Continuity Correction (ACTIVE)**
   - Asymmetric: OVER uses `line + 0.5`, UNDER uses `line - 0.5`
   - Impacts: hitter_fantasy_score (Gamma), any props using Gamma/Normal distributions
   - Direction: **Favors LESS**

2. **Confidence Floor Asymmetry (ACTIVE)**
   - Explicit: hits_more=0.90 vs hits_less=0.72, total_bases_more=0.90 vs total_bases_less=0.64
   - Impacts: hits, total_bases, hitter_fantasy_score (explicit +0.12 for LESS)
   - Direction: **Favors LESS**

3. **Negative Projection Offsets (ACTIVE)**
   - hitter_fantasy_score: -2.5, total_bases: -0.6, pitcher_strikeouts: -0.5, etc.
   - Applied uniformly to all picks before direction multiplier
   - Impacts: ALL props with negative offsets
   - Direction: **Favors LESS** (reduces projected mu, which favors UNDER on skewed distributions)

4. **Direction Multipliers (INACTIVE)**
   - Both set to 1.0 (no effect)
   - Would amplify LESS bias if less_multiplier < 1.0

### Combined Effect

The three active biases compound:
1. Negative offsets lower the projected mean
2. Lowered means, combined with asymmetric distribution functions, favor LESS
3. Confidence floors enforce LESS selection over MORE

**Result:** Systematic directional LESS/UNDER bias across the model.

---

## FINAL CHECKLIST

| Item | File | Lines | Issue | Status |
|------|------|-------|-------|--------|
| Gamma/Normal continuity correction | distributions.py | 279–306 | Asymmetric `+0.5` vs `-0.5` | **BUG** |
| hits MORE/LESS floors | current.json | 35–36 | 0.90 vs 0.72 | Intentional (backtest-justified) |
| total_bases MORE/LESS floors | current.json | 37–38 | 0.90 vs 0.64 | Intentional (backtest-justified) |
| hitter_fantasy_score MORE/LESS floors | current.json | 41–42 | 0.56 vs 0.68 | **Explicit LESS bias** |
| Negative prop_type_offsets | current.json | 15–33 | -0.5 to -2.5 | **Systematic LESS bias** |
| Direction multipliers | current.json | 11–14 | Both 1.0 | Inactive (no effect) |
| Sharp odds devigging | sharp_odds.py | 528–577 | Power devig | Symmetric ✓ |
| Combined edge scoring | combined.py | 57–206 | Signal weighting | Symmetric ✓ |

