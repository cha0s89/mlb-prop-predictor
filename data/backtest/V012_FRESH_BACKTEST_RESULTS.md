# V012 Fresh Backtest Analysis
*Generated: 2026-03-20 20:53:05*

Source: `backtest_2025.json` — fresh run with current predictor weights


## Overall Summary

- **Total rows in file:** 165,458
- **Non-plays / no-result filtered:** 40,898
- **Decided picks analyzed:** 124,560
- **Overall Accuracy:** 66.56%  (82,901W / 41,659L)

## 1. Cross-Tab: Prop Type × Direction


| Prop Type             | Direction | Wins   | Losses | Total  | Accuracy |
|----------------------|-----------|--------|--------|--------|----------|
| hits                 | MORE      |      1 |      3 |      4 |    25.0% |
| hits                 | LESS      | 15,805 |  8,788 | 24,593 |    64.3% |
|----------------------|-----------|--------|--------|--------|----------|
| hitter_fantasy_score | MORE      |  1,524 |  1,347 |  2,871 |    53.1% |
| hitter_fantasy_score | LESS      | 15,662 | 12,413 | 28,075 |    55.8% |
|----------------------|-----------|--------|--------|--------|----------|
| home_runs            | MORE      |      9 |     20 |     29 |    31.0% |
| home_runs            | LESS      | 35,586 |  4,731 | 40,317 |    88.3% |
|----------------------|-----------|--------|--------|--------|----------|
| pitcher_strikeouts   | MORE      |  1,230 |    786 |  2,016 |    61.0% |
| pitcher_strikeouts   | LESS      |  1,141 |    917 |  2,058 |    55.4% |
|----------------------|-----------|--------|--------|--------|----------|
| total_bases          | MORE      |  3,995 |  2,359 |  6,354 |    62.9% |
| total_bases          | LESS      |  7,948 | 10,295 | 18,243 |    43.6% |
|----------------------|-----------|--------|--------|--------|----------|

**Row totals by prop type:**

| Prop Type             | Total Decided | Accuracy |
|----------------------|---------------|----------|
| hits                 |        24,597 |    64.3% |
| hitter_fantasy_score |        30,946 |    55.5% |
| home_runs            |        40,346 |    88.2% |
| pitcher_strikeouts   |         4,074 |    58.2% |
| total_bases          |        24,597 |    48.6% |

## 2. Non-Play Filter Applied

| Rule | Applies To |
|------|-----------|
| `actual == 0` → excluded | hits, total_bases, hitter_fantasy_score, stolen_bases |
| `actual == 0` → **kept** | home_runs (0 HR is a valid outcome) |
| No filter on zero | pitcher_strikeouts |


## 3. Accuracy by Grade (A/B/C/D)


| Grade | Wins   | Losses | Total  | Accuracy | Notes |
|-------|--------|--------|--------|----------|-------|
| **A** | 51,363 | 13,419 | 64,782 |    79.3% | ← highest confidence |
| **B** | 10,221 |  7,563 | 17,784 |    57.5% | ← should be best? |
| **C** |  8,537 |  7,958 | 16,495 |    51.8% |  |
| **D** | 12,780 | 12,719 | 25,499 |    50.1% |  |

**Key question:** Are higher grades actually more accurate?
Ranking by accuracy: A > B > C > D
A-grade vs D-grade spread: +29.2pp (79.3% vs 50.1%)

## 4. Accuracy by Confidence Bucket


| Confidence | Wins   | Losses | Total  | Accuracy | Calibrated? |
|-----------|--------|--------|--------|----------|-------------|
| 50-55%      |  9,136 |  9,199 | 18,335 |    49.8% | ✓ YES |
| 55-60%      |  8,862 |  8,530 | 17,392 |    51.0% | ✗ NO (overconfident) |
| 60-65%      |  7,843 |  6,577 | 14,420 |    54.4% | ✗ NO (overconfident) |
| 65-70%      |  5,697 |  3,934 |  9,631 |    59.2% | ✗ NO (overconfident) |
| 70%+        | 51,363 | 13,419 | 64,782 |    79.3% | ✓ YES |

## 5. PP_TRADEABLE-Only Breakdown


These are the only prop+direction combos we plan to actually bet:

| Prop Type             | Direction | Wins   | Losses | Total  | Accuracy | vs v011   |
|----------------------|-----------|--------|--------|--------|----------|-----------|
| pitcher_strikeouts   | MORE      |  1,230 |    786 |  2,016 |    61.0% | -1.4pp vs v011 (62.4%) |
| pitcher_strikeouts   | LESS      |  1,141 |    917 |  2,058 |    55.4% | -2.7pp vs v011 (58.1%) |
| hitter_fantasy_score | MORE      |  1,524 |  1,347 |  2,871 |    53.1% | -5.8pp vs v011 (58.9%) |
| hitter_fantasy_score | LESS      | 15,662 | 12,413 | 28,075 |    55.8% | -3.9pp vs v011 (59.7%) |
| total_bases          | MORE      |  3,995 |  2,359 |  6,354 |    62.9% | +0.4pp vs v011 (62.5%) |
| hits                 | MORE      |      1 |      3 |      4 |    25.0% | — |
| hits                 | LESS      | 15,805 |  8,788 | 24,593 |    64.3% | -0.8pp vs v011 (65.1%) |
| home_runs            | LESS      | 35,586 |  4,731 | 40,317 |    88.3% | -0.7pp vs v011 (89.0%) |
| **TRADEABLE TOTAL**  |           | 74,944 | 31,344 | 106,288 |    70.5% | |

**Note on HR LESS:** actual=0 is kept (player genuinely hit 0 HR — valid WIN for LESS).

## 6. Monthly Accuracy Trend


| Month     | Wins   | Losses | Total  | Accuracy | Trend |
|----------|--------|--------|--------|----------|-------|
| April     | 14,535 |  6,945 | 21,480 |    67.7% | baseline |
| May       | 14,921 |  7,416 | 22,337 |    66.8% | ↓ -0.9pp |
| June      | 13,924 |  7,080 | 21,004 |    66.3% | ↓ -0.5pp |
| July      | 13,024 |  6,600 | 19,624 |    66.4% | ↑ +0.1pp |
| August    | 14,113 |  7,434 | 21,547 |    65.5% | ↓ -0.9pp |
| September | 12,384 |  6,184 | 18,568 |    66.7% | ↑ +1.2pp |

## 7. Comparison: v012 Fresh vs v011 Re-Scored


v011 re-scored used the same raw data with patched weights applied retroactively.
v012 fresh is a **new** backtest run with those weights baked into the predictor.
Differences indicate whether re-scoring was faithful or if walk-forward effects changed things.

| Prop × Direction          | v012 Fresh | v011 Target | Δ       | Signal |
|--------------------------|-----------|-------------|---------|--------|
| hitter_fantasy_score MORE  |      53.1% |       58.9% |   -5.8pp | ↓ worse than re-score |
| hitter_fantasy_score LESS  |      55.8% |       59.7% |   -3.9pp | ↓ worse than re-score |
| pitcher_strikeouts   MORE  |      61.0% |       62.4% |   -1.4pp | ↓ worse than re-score |
| pitcher_strikeouts   LESS  |      55.4% |       58.1% |   -2.7pp | ↓ worse than re-score |
| hits                 LESS  |      64.3% |       65.1% |   -0.8pp | ≈ consistent |
| home_runs            LESS  |      88.3% |       89.0% |   -0.7pp | ≈ consistent |
| total_bases          MORE  |      62.9% |       62.5% |   +0.4pp | ≈ consistent |


## 8. Key Findings & Recommendations


### Overall: 66.6% across 124,560 decided picks

**Strongest performers (v012 fresh):**
  🟢 HR LESS: **88.3%**
  🟢 Hits LESS: **64.3%**
  🟢 Total Bases MORE: **62.9%**
  🟢 Pitcher Ks MORE: **61.0%**
  🟡 Fantasy Score LESS: **55.8%**
  🟡 Pitcher Ks LESS: **55.4%**
  🔴 Fantasy Score MORE: **53.1%**
  🔴 Hits MORE: **25.0%**

**Grade calibration:**
  - A: 79.3%  B: 57.5%  C: 51.8%  D: 50.1%
  - ✓ Grades are correctly ordered (A > B > C > D)

**Direction bias:**
  - MORE overall: 60.0%  (11,274 picks)
  - LESS overall: 67.2%  (113,286 picks)
  - ⚠ Significant MORE/LESS imbalance — consider direction bias correction in autolearn.py

---
*Analysis produced by analyze_v012.py — 2026-03-20 20:53*