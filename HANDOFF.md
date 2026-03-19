# MLB Prop Edge — Chat Handoff (March 18, 2026)

## What exists right now
- Full Streamlit app deployed at: mlb-prop-predictor-ibjxavue4puzdeszptefss.streamlit.app
- GitHub repo: github.com/cha0s89/mlb-prop-predictor (private, user: cha0s89)
- Local path: C:\Users\Unknown\Downloads\mlb-prop-predictor
- 16+ Python modules in src/ — predictor, prizepicks, sharp_odds, spring, autograder, backtester, autolearn, trends, weather, etc.
- SQLite database for prediction logging and grading
- Odds API key configured (500 free credits/month)
- Batting cache: 461 players from 2025 FanGraphs
- Pitching cache: 657 pitchers from 2025 FanGraphs

## What was built/fixed today
1. PrizePicks league filter fixed (relationships.league.data.id, not attributes.league)
2. Hitter Fantasy Score projection model added to predictor.py
3. Player stats wired into app.py (FanGraphs via CSV cache for cloud)
4. Spring Training stats + injury flags from MLB Stats API (src/spring.py)
5. Auto-grading from MLB box scores (src/autograder.py)
6. Backtester with walk-forward fix (src/backtester.py)
7. Self-learning weight system with kill switch (src/autolearn.py)
8. Weight versioning: v002 post-backtest weights active (fs_offset=+0.8, more_mult=1.12, less_mult=0.92)
9. Trends integration with hot/cold badges and cold-elite BUY LOW signal
10. Pitcher stats cached and wired into predictions
11. Progress bar, weather caching, mobile improvements
12. Deployed to Streamlit Cloud with batting/pitching CSV caches

## Currently running
- Backtest v2 (walk-forward fixed) running overnight on local PC
- Claude Code tasks still pushing: bankroll tracker, Statcast expected stats, Today's Best Plays card, daily log feature, factor breakdown expandable rows, UI merge of checkboxes into projection table

## Backtest v1 results (with data leakage — 67.8% inflated)
- 195,596 predictions, 132,580W-63,016L
- MASSIVE LESS bias: MORE 36.3% vs LESS 80.1% — fixed with direction multipliers
- Fantasy Score: 63.4% (best prop type)
- Total Bases: 40.5% (worst — model projects too low)
- Pitcher Ks: 48.0% (coin flip — needs work)
- Grades ARE relatively predictive: A 82.9% > B 66.8% > C 50.4% > D 47.7%
- Waiting for v2 backtest (honest walk-forward) results

## What's next (priority order)
1. Analyze backtest v2 results and tune weights with self-learner
2. Fix any remaining LESS bias after seeing honest numbers
3. Get factor breakdown UI working (explain WHY each pick is recommended)
4. Expand player cache with Statcast expected stats (xBA, xSLG, xwOBA)
5. Dry run before Opening Day (March 27) — paper trade for a few days
6. Consider NBA expansion (way more daily volume than MLB ST)

## Key technical notes
- PrizePicks league IDs: 9=regular season, 43=Spring Training
- attributes.league is ALWAYS null — use relationships.league.data.id
- pybaseball doesn't work on Streamlit Cloud — use CSV caches
- Walk-forward backtest: when predicting April 15, only use stats from before April 15
- Freshness warnings during backtest are expected (it's pulling old data on purpose)
- The self-learner saves weight versions and rolls back if accuracy drops below 45%
- RotoWire comparison CSV uploaded — could import for agreement analysis
