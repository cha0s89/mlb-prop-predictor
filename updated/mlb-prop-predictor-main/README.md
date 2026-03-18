# ⚾ MLB Prop Edge — PrizePicks Market-Based Edge Finder

A free, market-based alternative to Rotowire's $30/month prop predictor. Instead of trying to out-predict the market with models, this tool catches when PrizePicks lines lag behind sharp sportsbook consensus — the approach proven by OddsShopper (18,988-15,316 record, 2.3% ROI on 2024 MLB).

**Cost: $0.** Free Odds API tier + free Streamlit hosting.

---

## How It Works

1. **Pulls PrizePicks lines** via their public API
2. **Pulls sharp sportsbook odds** from FanDuel, Pinnacle, DraftKings, etc. via The Odds API
3. **Devigs the sharp lines** (removes the bookmaker's profit margin) using the power method
4. **Compares fair probabilities** to PrizePicks' implied 50/50 — flags gaps as +EV edges
5. **Grades A through D** based on edge size
6. **Tracks your accuracy** over time with breakdowns by prop type, grade, and direction
7. **Weather, park factors, and umpire data** provide context for why edges exist

## Why This Beats Rotowire

Rotowire uses projection-based analysis — estimating what a player will do, then comparing to the line. That's fine, but it's trying to out-predict the market.

This tool uses **market-based edge finding** — it identifies when PrizePicks' lines disagree with what sharp books already know. FanDuel is the sharpest MLB props book (1.236 weight per Pikkit research), and when FanDuel moves, PrizePicks is slower to follow. That lag = your edge.

## Quick Start

1. Get a free API key at [the-odds-api.com](https://the-odds-api.com) (500 req/month)
2. Clone this repo
3. `pip install -r requirements.txt`
4. `export ODDS_API_KEY=your_key && streamlit run app.py`

**Or deploy to Streamlit Cloud** (access from your phone):
1. Push to GitHub
2. Go to share.streamlit.io → connect repo → add API key in Secrets → Deploy

## Daily Workflow

1. Open app **before 10 AM ET** (highest-edge window)
2. Find Edges tab → focus on **A and B grades**
3. Prioritize **pitcher strikeout props** (most exploitable)
4. Build a **5 or 6 pick Flex** on PrizePicks (never Power plays)
5. Grade results after games → Dashboard tracks accuracy

## Architecture

```
mlb-prop-predictor/
├── app.py                 # Main Streamlit UI
├── src/
│   ├── sharp_odds.py      # The Odds API client + devigging engine (CORE)
│   ├── prizepicks.py      # PrizePicks API client
│   ├── predictor.py       # Bayesian projection engine (secondary)
│   ├── stats.py           # pybaseball / Statcast layer
│   ├── weather.py         # Open-Meteo weather forecasts
│   ├── umpires.py         # Umpire K-rate tendencies
│   └── database.py        # SQLite prediction logging + grading
├── .streamlit/config.toml # Dark theme
├── requirements.txt
└── README.md
```

## Realistic Expectations

| Accuracy | 5-pick Flex ROI | 6-pick Flex ROI |
|:---------|:---------------|:----------------|
| 54.2% (breakeven) | ~0% | ~0% |
| 55% | +5% | +7.5% |
| 57% | +19% | +28.5% |
| 60% | +43% | +66% |

Target: 55-58% per-leg accuracy. Track 250+ entries before evaluating.

---

*Not financial or gambling advice. Bet responsibly.*
