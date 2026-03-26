# Claude Dispatch Handoff - 2026-03-25

## Current live model
- Active tracked weights: `v031`
- File: `C:\Users\Unknown\Downloads\mlb-prop-predictor\data\weights\current.json`
- Description: `promote effective runtime model and retune tail signals`
- Important cleanup: the old unversioned `runtime_override.json` has been cleared locally after its contents were merged into `v031`

## What changed in this pass
- Promoted the effective live model state that had previously been split between:
  - `current.json` (`v030`)
  - `runtime_override.json`
- Rebuilt empirical calibration tables from the full 2025 backtest:
  - `C:\Users\Unknown\Downloads\mlb-prop-predictor\data\weights\calibration_v015.json`
- Re-ran tail-signal optimization and applied the challenger because it cleared holdout:
  - validation tail score: `0.9286 -> 1.0308`
  - holdout tail delta: `+0.0903`
- Saved versioned snapshot:
  - `C:\Users\Unknown\Downloads\mlb-prop-predictor\data\weights\v031_promote_effective_runtime_model_and_retu.json`

## Current validated metrics
These are the current full-model metrics for the live `v031` configuration on the existing 2025 backtest split.

### All graded rows
- Validation:
  - accuracy: `0.6797`
  - log loss: `0.5990`
  - Brier: `0.2062`
  - MAE: `1.1013`
  - RMSE: `2.4944`
- Holdout:
  - accuracy: `0.6857`
  - log loss: `0.5946`
  - Brier: `0.2042`
  - MAE: `1.0915`
  - RMSE: `2.4672`

### Selected picks after confidence floors
- Validation:
  - selected picks: `45262`
  - accuracy: `0.7697`
  - log loss: `0.5236`
- Holdout:
  - selected picks: `42635`
  - accuracy: `0.7749`
  - log loss: `0.5179`

## What was checked
- `C:\Users\Unknown\Downloads\mlb-prop-predictor\src\offline_tuner.py`
  - model challenger did not clear promotion gates
  - floor challenger did not clear promotion gates
  - tail challenger did clear and was applied into `v031`
- Targeted tests passed:
  - `test_autolearn_weights.py`
  - `test_offline_tuner.py`
  - `test_probability_contract.py`

## Important operational note
- The repo on disk is the source of truth.
- `runtime_override.json` should stay empty unless a new live-learning step writes a justified override.
- Do not manually reintroduce old override values after `v031`; they are already baked into the tracked weights.

## Highest-value next work
Do these in order.

### 1. Review live 2026 graded results first
Use actual graded Opening Day and early-season results before making another broad retune.

Focus on:
- `hits`
- `runs`
- `hits_runs_rbis`
- `pitcher_strikeouts`

Why:
- those remain the most sensitive props in live use
- there is no value in reopening the whole 2025 search space again unless fresh 2026 results show a real miss

### 2. Do not run another broad model promotion immediately
Broad backtest retuning has already been exhausted for the current search space.

Only reopen full tuning if one of these is true:
- enough new 2026 graded rows accumulate
- live QA shows persistent bias by prop family
- a concrete feature change lands first

### 3. If a new tweak is needed, keep it narrow
Acceptable narrow targets:
- `hits` opportunity / bias
- `runs` opportunity / lineup environment
- pitcher strikeout opportunity / leash
- tail wording / UX clarity

Avoid:
- blind confidence-floor loosening
- hand-tuning to match another site
- reopening every offset and variance ratio without new data

### 4. Keep live evaluation clean
Before touching weights again:
- run nightly grading
- check projection diagnostics
- compare selected picks vs full board
- verify no duplicate board rows or stale aliases reappeared

## Concrete commands

### Nightly cycle
```powershell
@'
from src.nightly import run_nightly_cycle
import json
print(json.dumps(run_nightly_cycle("2026-03-25"), indent=2))
'@ | python -
```

### Current weights version
```powershell
@'
import json, pathlib
p = pathlib.Path(r"C:\Users\Unknown\Downloads\mlb-prop-predictor\data\weights\current.json")
print(json.loads(p.read_text())["version"])
'@ | python -
```

### Check runtime override stays empty
```powershell
Get-Content C:\Users\Unknown\Downloads\mlb-prop-predictor\data\weights\runtime_override.json
```

### Re-run tail analysis only
```powershell
@'
from src.offline_tuner import analyze_backtest_tail_signals
import json
res = analyze_backtest_tail_signals(r"C:\Users\Unknown\Downloads\mlb-prop-predictor\data\backtest\backtest_2025.json")
print(json.dumps({"should_apply": res.get("should_apply"), "reason": res.get("reason")}, indent=2))
'@ | python -
```

## Promotion rule
- Promote only if holdout improves.
- If holdout does not improve, do not create a new weight version.
- Prefer fewer, well-justified model versions over frequent churn.
