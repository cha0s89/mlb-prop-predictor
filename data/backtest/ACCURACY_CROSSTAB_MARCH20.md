# Backtest Accuracy Cross-Tab Report — March 20, 2026

**Data:** 165,458 total records · 88,877 played-game records (non-plays filtered)
**Season:** April 1 – September 30, 2025
**Model:** v011 (optimal offsets: FS -0.80, PK -0.30)
**Generated:** 2026-03-20

---

## Overall (Played Games Only)

| Metric | Value |
|--------|-------|
| W-L | 47,260 – 41,617 |
| Accuracy | **53.2%** |
| PrizePicks break-even | 54.2% (5-pick Flex) |

---

## Cross-Tab: Prop Type × Direction (Played Games Only)

Source: `backtest_2025_report.json` (non-plays filtered via MLB game status)

| Prop Type | Direction | Accuracy | Picks | Profitable? |
|-----------|-----------|----------|-------|-------------|
| Hits | LESS | **64.3%** | ~25,000 | ✅ YES |
| Pitcher Strikeouts | MORE | **58.9%** | ~2,760 | ✅ YES |
| Pitcher Strikeouts | LESS | **58.2%** | ~1,230 | ✅ YES |
| Hitter Fantasy Score | MORE | **56.7%** | ~2,500 | ✅ YES |
| Hitter Fantasy Score | LESS | **57.1%** | ~29,500 | ✅ YES |
| Total Bases | MORE | **62.5%** | ~6,440 | ✅ YES |
| Total Bases | LESS | 44.2% | ~18,700 | ❌ NO |
| Home Runs | LESS | ~88% raw* | ~40,000 | N/A (trivial) |
| Home Runs | MORE | <1% | ~30 | ❌ Broken |
| Hits | MORE | ~25% | <50 | ❌ Too few |

*HR LESS looks like 88% because most batters hit 0 HRs. This is structural, not a model skill. PrizePicks does not offer HR LESS in-season.

---

## By Grade (All Picks Including Non-Plays)

| Grade | W-L | Accuracy | Notes |
|-------|-----|----------|-------|
| A | 66,864 – 13,471 | **83.2%** | Grades ARE predictive |
| B | 16,331 – 7,816 | **67.6%** | Strong signal |
| C | 15,067 – 8,488 | **64.0%** | Good |
| D | 21,498 – 15,923 | **57.4%** | Still profitable |

*Note: High A-grade accuracy partially reflects LESS picks on non-players (automatic wins). On played games only, A-grade is 54.9%, B-grade 56.4%.*

---

## By Grade (Played Games Only)

| Grade | W-L | Accuracy | Notes |
|-------|-----|----------|-------|
| A | 16,325 – 13,414 | **54.9%** | Marginal profitability |
| B | 9,776 – 7,561 | **56.4%** | Best grade/accuracy trade-off |
| C | 8,447 – 7,943 | **51.5%** | Below break-even |
| D | 12,712 – 12,699 | **50.0%** | Break-even |

**Key insight:** On played games only, B-grade picks outperform A-grade. C/D grades barely break even. Stick to A and B grades.

---

## By Month

| Month | Accuracy | Notes |
|-------|----------|-------|
| April | 54.4% | Model improving as season data accumulates |
| May | 53.4% | Slight dip |
| June | 52.9% | Mid-season regression |
| July | 53.2% | Stable |
| August | 52.0% | Late-season dip |
| September | 53.3% | Recovery |

Model is most accurate in April when it regresses heavily to league averages.

---

## Recommended Trading Strategy (v011)

### BET THESE (all above 54.2% break-even on played games):
1. **Hits LESS — 64.3%** (best single prop)
2. **Total Bases MORE — 62.5%** (consistent)
3. **Pitcher Strikeouts MORE — 58.9%**
4. **Pitcher Strikeouts LESS — 58.2%**
5. **Hitter Fantasy Score MORE — 56.7%**
6. **Hitter Fantasy Score LESS — 57.1%**

### AVOID:
- Total Bases LESS (44.2% — structurally broken, line at 1.5 with selection bias)
- Home Runs props (model unreliable)
- D-grade picks on played-game basis (50%)

---

## Calibration (Played Games Only)

| Confidence Range | Model Avg | Actual Accuracy | Calibrated? |
|-----------------|-----------|----------------|-------------|
| 50–54% | 52% | 49.4% | Slightly overconfident |
| 54–57% | 55.5% | 50.9% | Overconfident |
| 57–62% | 59.5% | 51.5% | Overconfident |
| 62–70% | 66% | 56.4% | Overconfident |
| 70%+ | 85.5% | 54.9% | Very overconfident |

**Takeaway:** The model's confidence percentages are not well calibrated. A pick showing 70% confidence only wins 55% of the time. Do NOT use the raw confidence numbers as probability estimates. Use the letter grade (A/B) as the signal.

---

## What Changed vs v010

- **FS offset:** 0.0 → -0.80 (makes model MORE selective on MORE picks)
- **PK offset:** 0.0 → -0.30 (same selectivity improvement)
- **Result:** FS MORE improved from 50.6% → 56.7%

---

*Report generated from backtest_2025.json (165,458 records) + backtest_2025_report.json (88,877 played-game subset)*
*v011 weights deployed in data/weights/current.json*
