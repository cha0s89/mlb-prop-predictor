# Audit Changes Needed - 2026-03-22

## Scope

This pass reviewed the current code in `app.py`, `src/`, `tests/`, and the live SQLite data in `data/predictions.db`.

Checks performed:

- `python -m compileall app.py src tests`
- `streamlit.testing.v1.AppTest` initial render of `app.py`
- targeted review of projection math, distribution logic, persistence, and Streamlit output paths
- spot-checks against current PrizePicks data fetched by the app

What did **not** reproduce in this pass:

- no hard Streamlit render exception on initial load
- current PrizePicks board fetch succeeded (`237` MLB props on 2026-03-22)
- controlled same-context batter-vs-pitcher comparisons looked directionally sane when park and lineup spot were held constant

The items below are the changes still needed.

---

## 1) High - Saving sharp edges still writes invalid prediction rows

Files:

- `app.py:921`
- `src/sharp_odds.py:759`
- `src/database.py:123`
- `src/database.py:157`

Problem:

- `app.py` still sends raw sharp-edge dicts directly into `log_batch_predictions(...)`.
- The edge objects built in `src/sharp_odds.py` do not include the prediction schema fields expected by `src/database.py` (`line`, `projection`, `confidence`, `p_over`, `p_under`, `edge`, etc.).
- `src/database.py` silently defaults missing fields to `0` or `""`, so saved rows become unusable for grading and accuracy tracking.
- `grade_prediction(...)` then grades against the stored `line`, which would be `0` for these sharp-edge saves.

Required change:

- Do not write raw sharp-edge dicts into the `predictions` table.
- Either:
  1. create a dedicated `sharp_edges` table, or
  2. add a converter before `log_batch_predictions(...)` that maps sharp-edge fields into the prediction schema consistently.
- If you keep one table, populate at minimum:
  - `line <- pp_line`
  - `pick <- pick`
  - `confidence <- fair_prob`
  - `edge <- edge_pct / 100`
  - `stat_internal <- market` or mapped internal stat name
  - `model_version <- sharp_odds`
- Do not leave `projection` ambiguous. If there is no model projection, store a clearly named sharp-equivalent field in a separate table instead of overloading `projection`.

Why it matters:

- Saved sharp picks may look fine in the UI but will poison all downstream grading and dashboard stats.

---

## 2) High - Probability payloads are mathematically inconsistent for integer lines

Files:

- `src/predictor.py:1626`
- `src/predictor.py:1630`
- `src/predictor.py:1696`

Problem:

- `calculate_over_under_probability(...)` pulls `p_over`, `p_under`, and `p_push` from the distribution layer.
- It then renormalizes only `p_over + p_under` to `1.0` and returns `p_push` unchanged.
- For integer lines, the returned payload can sum to more than `1.0`.

Reproduced examples from the current code:

- `pitcher_strikeouts`, `mu=5.2`, `line=5.0` -> `p_over=0.5055`, `p_under=0.4945`, `p_push=0.1748`, total `1.1748`
- `hits`, `mu=1.1`, `line=1.0` -> `p_over=0.4538`, `p_under=0.5462`, `p_push=0.3005`, total `1.3005`

Why it matters:

- The returned values mix conditional and unconditional probabilities.
- `confidence` is then built from those renormalized side probabilities and is used throughout the app as if it were a true win probability.
- That directly affects pick cards, Kelly sizing, slip EV, calibration, and any saved confidence metrics.

Required change:

- Decide on one contract and keep it consistent:
  - either return unconditional three-way probabilities where `p_over + p_under + p_push = 1`, or
  - return conditional side probabilities and rename them accordingly.
- Then update downstream consumers (`confidence`, Kelly, slip EV, calibration, database saves) to use the same interpretation.
- Add regression tests that explicitly cover integer-line props with non-zero push probability.

---

## 3) High - `projected_stats` is duplicated on every Streamlit rerun

Files:

- `app.py:1350`
- `src/database.py:71`
- `src/database.py:289`
- `src/database.py:323`

Problem:

- `app.py` writes the full board into `projected_stats` every time the Streamlit script reruns.
- `projected_stats` has no uniqueness constraint and `save_projected_stats(...)` always inserts.
- `grade_projected_stats(...)` uses `fetchone()` for `(game_date, player_name, stat_type)`, so only one duplicate row gets updated and the rest stay stale.

Verified current DB state:

- `projected_stats` currently contains `1050` rows.
- There are only `210` unique `(game_date, player_name, stat_type)` groups.
- The duplicate count is `5` for many rows on `2026-03-22`.

Required change:

- Add a uniqueness rule for one board snapshot row. At minimum consider:
  - `(game_date, player_name, stat_type)`
  - or `(game_date, player_name, stat_type, line, pick)` if you intentionally want multiple lines per day
- Change `save_projected_stats(...)` to use upsert semantics.
- Only write a new board snapshot when the board actually changes, not on every widget rerun.
- Change `grade_projected_stats(...)` to update all matching rows or, preferably, rely on the unique row enforced above.

Why it matters:

- Any projection tracking, calibration, or accuracy numbers built off `projected_stats` are currently not trustworthy.

---

## 4) Medium - Sidebar accuracy metric is scaled incorrectly in Streamlit

Files:

- `app.py:764`
- `app.py:768`
- `src/database.py:233`

Problem:

- `get_accuracy_stats()` returns accuracy as a fraction in `[0, 1]`.
- The sidebar displays it as `f"{_sb_acc:.1f}%"`, which would render `0.6%` instead of `60.0%`.
- The delta also subtracts `55` instead of using the same scale as the displayed metric.

Required change:

- Format accuracy as `f"{_sb_acc * 100:.1f}%"`.
- Compare against a break-even value in the same units, for example:
  - fraction math: `_sb_acc - 0.5425`
  - display math: `(_sb_acc - 0.5425) * 100`
- Keep one convention everywhere: fractions internally, percentages only at the display boundary.

Why it matters:

- The current sidebar output is numerically wrong even if the underlying DB data is correct.

---

## 5) Medium - The existing math verification script is misleading and does not fail when it should

Files:

- `tests/test_projection_math.py:492`
- `tests/test_projection_math.py:503`
- `tests/test_projection_math.py:631`
- `tests/test_projection_math.py:638`

Problems:

1. `tests/test_projection_math.py:492` only checks `p_over + p_under`, so it cannot catch the live `p_push` bug.
2. `tests/test_projection_math.py:503` compares `edge` against `"MORE"` / `"LESS"`, but production `edge` is numeric, not a direction label.
3. `tests/test_projection_math.py:631` uses `hasattr(AVERAGE_BATTER, 'k_pct')` on a dict, so that branch always falls through and the park-effect check is not testing what it appears to test.
4. The file prints status lines and exits; it has no real assertion/failure path, so it is not an automated correctness test.

Required change:

- Convert this into real tests with assertions.
- Check direction with `pick`, not `edge`.
- Include `p_push` in probability-mass tests.
- Split "pretty printed exploratory output" from actual regression tests.
- Add controlled-context tests where only one variable changes at a time (park, opponent, lineup spot, etc.).

Why it matters:

- Right now the math test can report false failures, miss real failures, and still exit successfully.

---

## 6) Medium - Slip EV output is not aligned with the model's actual push probabilities

Files:

- `app.py:1710`
- `app.py:1837`
- `src/slip_ev.py:129`
- `src/slip_ev.py:327`

Problem:

- Suggested slips use `quick_slip_ev(...)`, and that function explicitly assumes independence and no ties.
- The UI shows that value as plain `EV` with no warning that ties are excluded.
- The selected-slip Monte Carlo path can consume `p_push`, but `app.py` does not pass model `p_push` into the leg dicts, so it falls back to coarse empirical push rates instead.

Required change:

- Pass per-leg `p_push` from the model into `simulate_slip_ev(...)`.
- Either remove the quick EV from the UI or relabel it clearly as an approximate no-ties estimate.
- Make sure Kelly sizing and slip EV use the same probability semantics as the core prediction model.

Why it matters:

- The slip cards can look precise while still being based on the wrong win/tie assumptions.

---

## 7) Medium - Home run distribution has two conflicting sources of truth

Files:

- `data/weights/current.json:87`
- `src/predictor.py:1592`

Problem:

- The weights file says `home_runs` uses a `binary` distribution.
- The predictor overrides that setting and forces `home_runs` to `negbin`.

Required change:

- Make the weights file and runtime logic agree.
- If `negbin` is the intended model now, update `current.json` and any related comments/docs.
- If `binary` is still intended, remove the override and recalibrate accordingly.

Why it matters:

- Tuning, calibration, and audit output are unreliable when the config file is not the real source of truth.

---

## 8) Low - Preseason messaging and actual control flow do not match

Files:

- `app.py:777`
- `app.py:788`
- `app.py:816`

Problem:

- The UI banner says "Projection-only mode" and that sharp books are unavailable until regular season.
- The code still proceeds into the sharp fetch loop whenever an API key exists.

Required change:

- Either:
  - skip sharp fetches during preseason, or
  - change the message so it matches the real behavior.

Why it matters:

- The app currently tells the user one thing and does another.

---

## 9) Low - Sharp edge filtering UI still does not expose all supported markets

Files:

- `app.py:864`
- `src/sharp_odds.py:30`
- `src/sharp_odds.py:389`

Problem:

- The sharp odds layer now supports a wider market set.
- The Streamlit filter still only offers a few hard-coded prop categories (`Pitcher Ks`, `Batter Hits`, `Total Bases`, `Home Runs`).

Required change:

- Build the filter options from the actual markets present in `all_edges`.
- Keep the human-readable labels in one mapping so UI labels and internal markets stay in sync.

Why it matters:

- Even when the backend supports more markets, the UI still makes the sharp-edge output feel incomplete.

---

## Recommended Fix Order

1. Fix the probability contract (`p_over` / `p_under` / `p_push`) and update downstream consumers.
2. Stop duplicating `projected_stats` and clean the existing table.
3. Separate sharp-edge persistence from projection persistence.
4. Correct the Streamlit percentage displays and slip EV labeling/inputs.
5. Replace the current print-only math script with real regression tests.

---

## Suggested Validation After Fixes

- Add assertion-based tests for integer-line probability mass and push handling.
- Re-run the app once, interact with filters, and verify `projected_stats` row counts do not multiply.
- Save a sharp edge and confirm the stored DB row has the real line/confidence values.
- Spot-check one integer-line prop end-to-end:
  - model output
  - Streamlit card
  - saved DB row
  - slip EV / Kelly display
- Re-run `streamlit.testing.v1.AppTest` after the math and persistence fixes.
