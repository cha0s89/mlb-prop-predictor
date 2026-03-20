import json
import numpy as np

with open('data/backtest/backtest_2025.json') as f:
    data = json.load(f)

print("=" * 100)
print("NON-PLAY (actual=0) IMPACT ANALYSIS")
print("=" * 100)

# For each prop type, analyze non-plays
for prop_type in ['hitter_fantasy_score', 'total_bases']:
    print(f"\n{prop_type.upper()}")
    print("-" * 100)
    
    more_recs = [r for r in data if r['prop_type'] == prop_type and r['pick'] == 'MORE']
    less_recs = [r for r in data if r['prop_type'] == prop_type and r['pick'] == 'LESS']
    
    for direction_name, direction_recs in [('MORE', more_recs), ('LESS', less_recs)]:
        plays = [r for r in direction_recs if r['actual'] > 0]
        nonplays = [r for r in direction_recs if r['actual'] == 0]
        
        # Accuracy of non-plays
        # MORE with actual=0 should be a loss (projection>line, but actual<line)
        # LESS with actual=0 should be a win (projection<=line, and actual<line)
        nonplay_wins = sum(1 for r in nonplays if r['result'] == 'W')
        nonplay_losses = sum(1 for r in nonplays if r['result'] == 'L')
        nonplay_acc = nonplay_wins / (nonplay_wins + nonplay_losses) * 100 if (nonplay_wins + nonplay_losses) > 0 else 0
        
        play_wins = sum(1 for r in plays if r['result'] == 'W')
        play_losses = sum(1 for r in plays if r['result'] == 'L')
        play_acc = play_wins / (play_wins + play_losses) * 100 if (play_wins + play_losses) > 0 else 0
        
        # Overall accuracy
        overall_wins = play_wins + nonplay_wins
        overall_losses = play_losses + nonplay_losses
        overall_acc = overall_wins / (overall_wins + overall_losses) * 100 if (overall_wins + overall_losses) > 0 else 0
        
        print(f"\n  {direction_name}:")
        print(f"    Actual plays:        {len(plays):5} records, accuracy {play_acc:5.1f}%")
        print(f"    Non-plays (0):       {len(nonplays):5} records, accuracy {nonplay_acc:5.1f}%")
        print(f"    Overall:             {len(direction_recs):5} records, accuracy {overall_acc:5.1f}%")
        print(f"    Non-play contribution: {len(nonplays) / len(direction_recs) * 100:5.1f}% of picks")
        
        if len(nonplays) > 0 and len(plays) > 0:
            # What would accuracy be WITHOUT non-plays?
            print(f"    IMPACT: Remove non-plays → accuracy {play_acc:.1f}% (was {overall_acc:.1f}%) — difference {play_acc - overall_acc:+.1f}pp")

print("\n" + "=" * 100)
print("KEY INSIGHT")
print("=" * 100)
print("""
Non-plays (actual=0) represent batters who didn't get plate appearances.
PrizePicks only offers props for probable starters, but backtest includes
all players in box scores.

For LESS picks: non-plays are automatic wins (actual < line is always true)
→ Inflates LESS accuracy by ~5-15%

For MORE picks: non-plays are automatic losses (actual < line always true)
→ Depresses MORE accuracy by ~5-15%

To match live PrizePicks conditions, we should EITHER:
A) Filter backtest to only probable starters (PA >= 2 or lineups)
B) Track whether backtest picks would actually be offered live
C) Keep backtest as-is but apply direction multiplier to correct the bias

Current status: PA >= 2 filter is IN CODE (src/backtester.py:236)
but non-plays still exist at 21-44% levels. Likely reason:
→ Some bench players who get selected get 0 actual PAs
→ Filter is checking season stats PA, not actual game PA
""")
