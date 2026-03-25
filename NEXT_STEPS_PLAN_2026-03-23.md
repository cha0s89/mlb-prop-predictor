# MLB Prop Predictor - Remaining Work and Execution Plan

Date: 2026-03-23

## Current Baseline

- Active model: `v024`
- Current weights description: `manual hrrbi distribution tuning`
- Latest completed preseason backtest:
  - rows: `464,176`
  - accuracy: `68.49%`
  - record: `317,913W - 146,263L`
- Current branch state when this file was created: clean

## What Is Already Done

- Strict integer-line `MORE / LESS / PUSH` probability semantics
- Push-aware sharp repricing
- Latent-`mu` sharp consensus across books
- Expanded calibration across major prop families
- Tail/breakout/dud threshold tuning
- Spring-process overlay and season-aware blending
- Lineup-aware hitter and pitcher projections in app/headless flow
- Lineup-aware historical backtests
- Historical umpire context added to backtests
- Mobile responsiveness pass
- Nightly grading mismatch repair
- Headless board pipeline, nightly cycle, and weekly tuning wrapper
- Backtest runner progress/lock improvements

## What Still Needs To Be Done

### 1. Immediate Accuracy Work

These are the highest-value model items still open.

#### 1.1 Rerun preseason bakeoff with the newest backtest realism

Reason:
- Historical umpire context was added after the last completed bakeoff.
- That means the newest backtest/training cycle should be rerun once to see whether the extra realism produces a challenger that beats `v024`.

Success condition:
- Only promote if holdout improves.

#### 1.2 Historical weather/context parity in backtests

Reason:
- Live projections use stronger park/weather context than historical backtests.
- That mismatch means the offline tuner is still training on a less realistic environment than live production.

Target:
- Carry historical weather inputs into the backtest path where feasible.
- Keep it bounded to data we can fetch reliably and reproducibly.

#### 1.3 Stronger breakout/dud model

Current state:
- Tail labels are better than before, but still mostly threshold-driven.

Next step:
- Build a stronger tail layer using backtest-derived labels and richer context.
- Prioritize:
  - pitcher K breakout / dud
  - hitter power breakout / dud
  - hitter run-production breakout / dud

Target outputs:
- better `breakout_prob`
- better `dud_prob`
- better reasons for why the tail is live

### 2. Weak Prop Families To Improve

These are the main prop families still worth focused work.

#### 2.1 `runs`, `rbis`, `hits_runs_rbis`

Current state:
- materially improved, but still the weakest hitter production cluster.

Next features to add:
- stronger bullpen quality/opponent run-prevention context
- stronger lineup certainty handling
- surrounding-hitter quality beyond simple local context
- team-total style support rather than only local player support

#### 2.2 `pitcher_strikeouts`

Current state:
- much better than earlier, but still worth tightening.

Next features to add:
- expected batters faced / leash modeling
- better confirmed-opponent-lineup K context
- historical umpire signal now in backtests
- game-script / bullpen leash sensitivity if practical

#### 2.3 `singles` / `doubles`

Current state:
- no longer proxied through `hits`, which is correct

Next step:
- verify whether the direct models should be distribution/calibration-tuned separately after the next bakeoff

### 3. Live Operational Checks Before Opening Day

These are production-readiness checks, not research.

#### 3.1 PrizePicks live payout verification in the real app

Reason:
- Official help page currently matches the encoded tables, but the real app is the source of truth.

Need to verify:
- `6 Power = 25x`
- `6 Flex = 12.5x`
- `3 Flex = 2.25x / 1.25x`

If the app shows different values:
- update `src/slips.py`
- verify `src/kelly.py`
- verify `src/slip_ev.py`
- verify `src/sharp_odds.py`

#### 3.2 Live regular-season board smoke test

Need to verify when normal regular-season lines are live:
- PrizePicks board ingestion
- sharp odds ingestion
- edge generation
- slip generation
- nightly update button
- no `nan` / empty-state regressions in UI

#### 3.3 Real phone/device smoke test

Need to verify on an actual phone:
- top nav/tabs
- pick cards
- slip builder
- horizontally scrollable tables
- no broken wrapping / zoom issues

### 4. Automation and Reliability

These are mostly in place, but should still be verified end to end.

#### 4.1 Verify scheduled automation is actually registered and running

Need to confirm:
- board capture task
- nightly grading task
- weekly tuning task

#### 4.2 Keep progress/lock behavior clean

Already improved:
- lock file
- progress file
- resumable backtest behavior

Need to monitor:
- no stale lock leftovers
- no accidental double-runs
- no silent interruptions on Windows

## Execution Plan

### Phase A - Next Immediate Pass

1. Rerun preseason backtest+tune with:
   - lineup-aware backtests
   - historical umpire context
2. Compare challenger vs `v024`
3. Promote only if holdout improves

### Phase B - Final Pre-Opening-Day Checks

1. Verify live PrizePicks payout tables in the app
2. Do a live regular-season board/sharp smoke pass
3. Do a real phone/device pass
4. Fix only launch-blocking issues

### Phase C - First Post-Launch Accuracy Pass

1. Add historical weather parity to backtests
2. Improve run-production props further
3. Strengthen breakout/dud modeling
4. Let nightly grading and CLV accumulate current-season signal
5. Continue challenger-vs-champion promotions only when holdout improves

## Promotion Rule

The promotion rule does not change:

- no weight promotion on intuition
- no promotion from training-only improvements
- no promotion unless holdout improves

Metrics to gate on:
- log loss
- Brier
- MAE / RMSE where applicable
- calibration quality
- practical behavior on weak prop families

## Short Version

If we want the best use of time from here:

1. rerun the preseason bakeoff with the newest realism
2. keep `v024` unless a challenger actually beats it
3. verify live payout tables and live board behavior
4. then move to weather parity + stronger breakout/dud modeling
