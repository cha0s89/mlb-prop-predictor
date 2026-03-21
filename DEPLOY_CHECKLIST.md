# MLB Prop Edge — Pre-Opening Day Deploy Checklist
**Opening Day: March 27, 2026**

---

## Environment Variables Required

| Variable | Where to set | Notes |
|---|---|---|
| `ODDS_API_KEY` | Streamlit Cloud Secrets → `ODDS_API_KEY = "..."` | Free at the-odds-api.com, 500 req/month |

No other secrets needed. All other data sources (PrizePicks, MLB Stats API, pybaseball) are free with no auth.

---

## One-Time Setup: Deploy to Streamlit Cloud

1. Push this repo to GitHub (main branch)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo: `mlb-prop-predictor`, branch: `main`, file: `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   ODDS_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — takes ~2 min to build
6. Bookmark the URL on your phone

---

## Pre-Launch Verification (Day Before: March 26)

- [ ] App loads without errors (check Streamlit Cloud logs)
- [ ] Setup tab → Opening Day Checklist shows green for API key
- [ ] Find Edges tab loads PrizePicks lines (Spring Training lines visible)
- [ ] A/B grade picks appear with real projections (not all 0.0%)
- [ ] Elite hitters show projections above 7.5 Fantasy Score line
- [ ] Weak hitters show projections below 7.5 Fantasy Score line
- [ ] MORE and LESS picks both appear (not all one direction)
- [ ] HR LESS and SB LESS are hidden by default (not tradeable on PP)
- [ ] TB LESS is hidden by default (44% accuracy — below break-even)
- [ ] Slip builder: pick 5 players, verify payout calculation appears
- [ ] Grade tab functional
- [ ] Dashboard tab shows (may be empty before first picks)

---

## Opening Day Workflow — March 27

### Morning (~9–10 AM ET, before first pitch)
1. Open app → **Find Edges** tab
2. Sharp odds load automatically (sharp path) — wait for all_edges to populate
3. Filter: **A + B** grades minimum
4. Focus on **Pitcher Strikeouts** first (most reliable prop type)
5. Check **Today's Best Plays** cards at top of page
6. Build a **5-Pick or 6-Pick Flex** slip — mix MORE and LESS
7. Verify: max 2 picks per team, all B+ grade

### Slip Rules (hardcoded from backtest results)
- ✅ **BET:** Pitcher Ks (MORE + LESS), Hitter Fantasy Score, TB MORE, Hits, H+R+RBI
- ❌ **AVOID:** TB LESS (44% acc), HR LESS (PP rarely offers), SB LESS (PP doesn't offer)
- 🎯 **Target:** 5-Pick Flex (break-even 54.2%) or 6-Pick Flex (52.9%)
- 💰 **Bankroll:** 1–2% per slip, max 5 slips/day, stop if down 10%

### Evening (after games ~11 PM ET or next morning)
1. Go to **Grade** tab
2. Click **Auto-Grade** — pulls MLB box scores automatically
3. Review results, check accuracy tracking
4. Dashboard updates with running record

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| SQLite resets on Streamlit Cloud redeploy | Lose prediction history | Export DB before redeploying, or use local install |
| Free Odds API tier: 500 req/month | ~16 requests/day | Run once per morning; don't spam refresh |
| Spring Training stat noise | Less accurate projections early | Model uses prior-year stats + ST adjustment; accuracy improves April+ |
| No live lineup data | May include scratched players | Cross-check PP board manually before submitting |
| pybaseball rate limits | Slow first load | Data is cached for 1 hour; subsequent loads fast |
| No prop coverage for all PP markets | Some PP props not modeled | App only shows props it can project |

---

## Model Status (as of March 2026)

| Prop Type | Backtest Accuracy | Direction | Notes |
|---|---|---|---|
| Hitter Fantasy Score | ~56% | MORE | Best prop type early season |
| Hitter Fantasy Score | ~57% | LESS | |
| Total Bases | ~63% | MORE | |
| Total Bases | ~44% | LESS | **Disabled — below break-even** |
| Pitcher Strikeouts | ~55% | MORE+LESS | Uses NegBin model |

Model weights: see `data/weights/current.json` (version shown in Setup tab)

---

## Troubleshooting

**All picks show 0.0% edge / D-grade:**
- Sharp odds not loaded yet (too early or no API key)
- App falls back to projection-only mode — this is expected; grades will be lower

**PrizePicks shows 0 props:**
- PP API may be slow in the morning or between seasons
- Try refreshing; props typically appear ~8–9 AM ET game days

**Import error on startup:**
```bash
pip install -r requirements.txt
```

**Autograder finds 0 results:**
- MLB Stats API sometimes delays final box scores by 1–2 hours after games end
- Run again next morning — all prior-day games will be graded

---

## Rollback

If the deployed app breaks after a push:
```bash
git revert HEAD
git push origin main
```
Streamlit Cloud auto-redeploys within ~60 seconds.
