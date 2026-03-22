# Math Verification Test Suite — File Index

## Quick Navigation

### For Quick Understanding (Start Here)
- **README_MATH_VERIFICATION.md** — Overview, structure, and key findings (5 min read)
- **test_output_2026_03_22.txt** — Raw test output (skim for specific props)

### For Deep Analysis
- **TEST_RESULTS_ANALYSIS.md** — Detailed mathematical validation and findings (15 min read)
- **test_projection_math.py** — Source code implementation (reference)

## File Descriptions

### Main Test File
**test_projection_math.py** (26 KB, 640 lines)

The core test suite that validates all projection mathematics. Tests:
- 4 batter profiles (Elite, Average, Terrible, Rookie)
- 2 pitcher profiles (Elite, Terrible)
- 11+ batter properties
- 5 pitcher properties
- 3 contexts per player
- 19 full end-to-end predictions

Run with: `python test_projection_math.py`

Key sections:
- Lines 1-120: Imports and player profiles
- Lines 125-200: Sanity check functions
- Lines 205-350: test_batter_projections()
- Lines 355-420: test_pitcher_projections()
- Lines 425-520: test_generate_prediction()
- Lines 525-580: test_comparative_analysis()
- Lines 585-620: test_park_effects()

### Test Output
**test_output_2026_03_22.txt** (22 KB, 479 lines)

Complete console output from running the test suite. Shows:
- 132 batter projections with context
- 30 pitcher projections with context
- 19 generate_prediction() results
- Comparative analysis results
- Park factor validation

Useful for:
- Spot-checking specific player/prop combinations
- Seeing exact probability outputs
- Verifying park effects
- Understanding confidence calibration

### Analysis Document
**TEST_RESULTS_ANALYSIS.md** (11 KB)

Detailed analysis of test results including:
- Overview of what was tested
- Key findings (6 major validation points)
- Mathematical examples with formulas
- Discussion of edge cases (e.g., why Elite K < Terrible K is correct)
- Probability calibration details
- Regression and stabilization validation
- Summary of findings and recommendations

### Quick Reference
**README_MATH_VERIFICATION.md** (5.3 KB)

Quick start guide covering:
- File overview
- Test structure
- Running instructions
- Key findings summary
- Validation checklist
- How to extend tests
- Notes on implementation

## Test Coverage Summary

### Batter Projections Tested
```
132 projections = 4 players × 11 stats × 3 contexts

Players:
  • Elite Batter (.300/.400/.550)
  • Average Batter (.248/.312/.399)
  • Terrible Batter (.180/.230/.270)
  • Rookie/Unknown (minimal PA)

Stats:
  • Hits, Runs, RBIs, Home Runs, Stolen Bases, Total Bases
  • Strikeouts (batter), Walks, Fantasy Score, Hits+Runs+RBIs

Contexts:
  • Neutral (baseline)
  • vs Elite SP at Coors, batting 3rd
  • vs Terrible SP at SF, batting 7th
```

### Pitcher Projections Tested
```
30 projections = 2 players × 5 stats × 3 contexts

Players:
  • Elite SP (2.80 ERA, 10.5 K/9)
  • Terrible SP (5.80 ERA, 6.0 K/9)

Stats:
  • Strikeouts, Outs, Earned Runs, Walks Allowed, Hits Allowed

Contexts:
  • Neutral
  • At Coors (bad lineup)
  • At SF (good lineup)
```

### Full Pipeline Tests
```
19 generate_prediction() calls covering:
  • All major batter props @ various lines
  • All major pitcher props @ various lines
  • probability calibration check
  • edge direction validation
  • confidence scoring
```

## Key Findings At A Glance

| Aspect | Status | Evidence |
|--------|--------|----------|
| Projection Math | ✓ CORRECT | All outputs within expected ranges |
| Player Ranking | ✓ CORRECT | Elite > Average > Terrible |
| Park Effects | ✓ CORRECT | Coors +3%, SF -2% as expected |
| Probabilities | ✓ CORRECT | P(over) + P(under) = 1.0 |
| Regression | ✓ CORRECT | Rookie (PA=50) regresses properly |
| Confidence | ✓ CORRECT | Scaled 0.5-1.0, appropriate weighting |

## How to Use This Test Suite

### Scenario 1: Verify a specific projection function
→ Search test_projection_math.py for the function name
→ Look at test_output_2026_03_22.txt for example results
→ Read TEST_RESULTS_ANALYSIS.md for mathematical validation

### Scenario 2: Understand probability calibration
→ Start with README_MATH_VERIFICATION.md (Key Findings section)
→ Read TEST_RESULTS_ANALYSIS.md (Probability Calibration section)
→ Run test and check P(over) + P(under) sums

### Scenario 3: Add new tests
→ Read README_MATH_VERIFICATION.md (How to Extend section)
→ Copy an existing test function in test_projection_math.py
→ Add your new profile or context to the appropriate list
→ Run with: python test_projection_math.py

### Scenario 4: Understand why Edge direction "mismatches"
→ Read TEST_RESULTS_ANALYSIS.md (Issue 2 section)
→ Key: Borderline regression and empirical calibration cause this
→ This is correct behavior, not a bug

## Running the Tests

```bash
cd /sessions/serene-exciting-pasteur/mnt/mlb-prop-predictor
python tests/test_projection_math.py
```

Expected output:
- Runtime: 5-10 seconds
- Lines: ~479 lines
- Status: All checks pass (✓)

## File Relationships

```
test_projection_math.py
  ├─ Reads from: src/predictor.py (all projection functions)
  ├─ Uses: LG dict, STAB dict, PARK dicts, PARK_* dicts
  └─ Produces: 479 lines of test output to console

test_output_2026_03_22.txt
  └─ Generated by: test_projection_math.py (run on 2026-03-22)

TEST_RESULTS_ANALYSIS.md
  └─ Analyzes: test_output_2026_03_22.txt results

README_MATH_VERIFICATION.md
  ├─ Explains: test_projection_math.py structure
  └─ References: TEST_RESULTS_ANALYSIS.md for details

This INDEX.md
  └─ Organizes: All test suite files and their relationships
```

## Implementation Notes

The test suite validates these core concepts:

1. **Bayesian Regression** — Testing _regress() function
   - Formula: (PA×Observed + Stab×League) / (PA + Stab)
   - Checked with rookie (PA=50) → heavy regression

2. **Park Factors** — Testing PARK, PARK_HR, PARK_K, PARK_SB dicts
   - Coors (COL): 116 general, 130 HR, 96 K
   - SF: 95 general, 88 HR, 102 K

3. **Log5 Matchup Adjustment** — Testing log5_rate() function
   - Combines pitcher skill, batter skill, league context

4. **Opportunity Estimation** — Testing estimate_plate_appearances() and estimate_batters_faced()
   - PA varies by lineup position (4.0-4.8)
   - BF varies by IP and pitcher quality

5. **TTO Penalty** — Testing tto_k_rate_decay() function
   - K% drops ~3% on third time through order

6. **Probability Distributions** — Testing calculate_over_under_probability()
   - P(over) + P(under) = 1.0
   - Based on theoretical normal/beta-binomial models

## Questions?

Refer to:
- For high-level overview → README_MATH_VERIFICATION.md
- For specific findings → TEST_RESULTS_ANALYSIS.md
- For implementation details → test_projection_math.py
- For raw results → test_output_2026_03_22.txt
