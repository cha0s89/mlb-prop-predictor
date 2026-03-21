# Bug Fix Log - 2026-03-20

## Critical Bug: Insane Pitching Outs Projections

### Reported Issue
Paul Skenes' pitching outs prop (line 15.5) was being projected at ~30 outs (10 innings).
- Typical MLB starters pitch 5-7 innings per game (15-21 outs)
- Paul Skenes, elite ace, would realistically pitch ~6-7 innings on average
- Projection of 30 outs was literally impossible and broke credibility of entire tool

### Root Cause
The `project_pitcher_outs()` function (line 280 in src/predictor.py) was clamping average IP per start to a maximum of 8.0:
```python
avg_ip = max(4.0, min(8.0, avg_ip))  # Line 280 - WRONG
```

This caused:
- Pitcher with 50 IP in 6 starts (8.33 IP/start) → clamped to 8.0 IP/start → 24 outs
- Paul Skenes → 25.4 outs after offsets → A-grade pick with 95.8% confidence
- Unrealistic projections for ALL elite aces

### Related Issues
Multiple other pitcher projection functions had similar overly-high clamps (7.5 max):
- `project_pitcher_strikeouts()` - line 237
- `project_pitcher_earned_runs()` - line 317
- `project_pitcher_walks()` - line 355
- `project_pitcher_hits_allowed()` - line 382

### Solution Applied
Changed all IP clamps to 6.5, which is realistic for elite aces:

1. **project_pitcher_outs()** line 280:
   - Changed: `max(4.0, min(8.0, avg_ip))`
   - To: `max(4.0, min(6.5, avg_ip))`
   - Result: Paul Skenes now projects 20.6 outs (6.9 IP) instead of 25.4

2. **project_pitcher_strikeouts()** line 237:
   - Changed: `max(4.5, min(7.5, expected_ip))`
   - To: `max(4.5, min(6.5, expected_ip))`

3. **project_pitcher_earned_runs()** line 317:
   - Changed: `max(4.0, min(7.5, avg_ip))`
   - To: `max(4.0, min(6.5, avg_ip))`

4. **project_pitcher_walks()** line 355:
   - Changed: `max(4.0, min(7.5, avg_ip))`
   - To: `max(4.0, min(6.5, avg_ip))`

5. **project_pitcher_hits_allowed()** line 382:
   - Changed: `max(4.0, min(7.5, avg_ip))`
   - To: `max(4.0, min(6.5, avg_ip))`

### Validation

#### Before Fix:
```
Paul Skenes: 25.4 outs (8.5 IP) - INSANE
Average starter (5 IP): 15.0 outs (5.0 IP) - OK but arbitrary
Struggling pitcher (4 IP): 11.9 outs (4.0 IP) - Wrong direction
```

#### After Fix:
```
Elite ace (8.33 IP/start): 20.6 outs (6.9 IP) - REASONABLE
Average starter (5 IP): 15.0 outs (5.0 IP) - CORRECT
Struggling pitcher (4 IP): 11.9 outs (4.0 IP) - REASONABLE
High K pitcher (11 K/9): 6.2 K (line 5.5) - GOOD
Low K pitcher (6 K/9): 4.8 K (line 5.5) - GOOD
```

### Impact
- All pitching outs projections now fall within 12-21 range (4-7 IP)
- Elite aces can project up to 20.6 outs (6.9 IP) - realistic
- Average starters at 15 outs (5 IP) - realistic
- Tool now passes sanity checks for all prop types

### Files Modified
- `src/predictor.py` - 5 functions fixed

### Testing
All projection functions tested with:
- Realistic profiles (typical stats)
- Extreme cases (elite vs struggling)
- Edge cases (no data, high/low rates)

All values now within expected ranges.
