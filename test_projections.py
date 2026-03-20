#!/usr/bin/env python3
"""
Test script to verify projection accuracy for different player types.
Builds realistic player profiles and checks if projections make sense.
"""

import sys
sys.path.insert(0, '.')

from src.predictor import (
    project_hitter_fantasy_score,
    project_pitcher_strikeouts,
    project_batter_total_bases,
    project_batter_hits,
    calculate_over_under_probability,
)

def test_fantasy_score():
    """Test fantasy score for various player types."""
    print("\n" + "="*70)
    print("FANTASY SCORE PROJECTIONS TEST")
    print("="*70)

    # League average hitter (2024 stats)
    league_avg = {
        'pa': 500, 'avg': 0.248, 'obp': 0.312, 'slg': 0.399, 'iso': 0.151,
        'woba': 0.310, 'bb_rate': 8.3, 'k_rate': 22.7,
        'hr': 17, 'sb': 5, 'rbi': 55, 'r': 55, '2b': 24, '3b': 1,
        'sprint_speed': 27.0, 'xba': 0.248, 'xslg': 0.399,
        'barrel_rate': 7.5, 'recent_barrel_rate': 7.5, 'recent_hard_hit_pct': 37.0
    }

    # Elite hitter (Aaron Judge-ish: .350 BA, 50 HR)
    elite = {
        'pa': 550, 'avg': 0.320, 'obp': 0.420, 'slg': 0.630, 'iso': 0.310,
        'woba': 0.410, 'bb_rate': 12.0, 'k_rate': 23.0,
        'hr': 50, 'sb': 3, 'rbi': 140, 'r': 120, '2b': 28, '3b': 1,
        'sprint_speed': 26.5, 'xba': 0.320, 'xslg': 0.620,
        'barrel_rate': 16.0, 'recent_barrel_rate': 16.0, 'recent_hard_hit_pct': 55.0
    }

    # Weak hitter (.220 BA, 10 HR)
    weak = {
        'pa': 450, 'avg': 0.220, 'obp': 0.290, 'slg': 0.350, 'iso': 0.130,
        'woba': 0.280, 'bb_rate': 6.0, 'k_rate': 28.0,
        'hr': 10, 'sb': 2, 'rbi': 40, 'r': 40, '2b': 15, '3b': 1,
        'sprint_speed': 27.0, 'xba': 0.220, 'xslg': 0.340,
        'barrel_rate': 4.0, 'recent_barrel_rate': 4.0, 'recent_hard_hit_pct': 28.0
    }

    line = 7.5

    for name, profile in [("League Avg", league_avg), ("Elite (Judge)", elite), ("Weak", weak)]:
        result = project_hitter_fantasy_score(profile)
        proj = result['projection']
        mu = result['mu']

        # Calculate edge
        prob = calculate_over_under_probability(proj, line, 'hitter_fantasy_score')

        print(f"\n{name:20} Projection: {proj:6.2f}  vs Line {line}")
        print(f"  {prob['pick']:>5} {prob['confidence']:.1%} confidence, rating {prob['rating']}, edge {prob['edge']:.1%}")

        # Reality check
        if name == "League Avg":
            if proj < 7.0:
                print(f"  ⚠️  WARNING: League avg should be ~7.5-8.0, got {proj}")
        elif name == "Elite (Judge)":
            if proj < 10.0:
                print(f"  ⚠️  WARNING: Elite should be ~11-13, got {proj}")
        elif name == "Weak":
            if proj > 6.0:
                print(f"  ⚠️  WARNING: Weak should be ~5-6, got {proj}")


def test_pitcher_ks():
    """Test pitcher K projections."""
    print("\n" + "="*70)
    print("PITCHER STRIKEOUTS PROJECTIONS TEST")
    print("="*70)

    # League average starter (K/9 = 8.58)
    league_avg = {
        'ip': 100, 'gs': 20,  # ~5 IP/start
        'k_pct': 22.7, 'k9': 8.58, 'bb_pct': 8.3, 'bb9': 3.22,
        'recent_csw_pct': 28.5, 'recent_swstr_pct': 11.3
    }

    # Ace (12.0 K/9)
    ace = {
        'ip': 200, 'gs': 33,
        'k_pct': 28.0, 'k9': 12.0, 'bb_pct': 7.0, 'bb9': 2.3,
        'recent_csw_pct': 32.0, 'recent_swstr_pct': 14.0
    }

    # Back-end starter (7.0 K/9)
    back_end = {
        'ip': 80, 'gs': 20,
        'k_pct': 19.0, 'k9': 7.0, 'bb_pct': 9.0, 'bb9': 3.8,
        'recent_csw_pct': 26.0, 'recent_swstr_pct': 9.5
    }

    line = 4.5

    for name, profile in [("League Avg", league_avg), ("Ace", ace), ("Back-end", back_end)]:
        result = project_pitcher_strikeouts(profile, expected_ip=5.5)
        proj = result['projection']

        # Calculate edge
        prob = calculate_over_under_probability(proj, line, 'pitcher_strikeouts')

        print(f"\n{name:20} Projection: {proj:6.2f}  vs Line {line}")
        print(f"  {prob['pick']:>5} {prob['confidence']:.1%} confidence, rating {prob['rating']}, edge {prob['edge']:.1%}")

        # Reality check
        if name == "League Avg":
            if proj < 4.5:
                print(f"  ⚠️  WARNING: Avg should be ~5.0-5.5, got {proj}")
        elif name == "Ace":
            if proj < 7.0:
                print(f"  ⚠️  WARNING: Ace should be ~8-10, got {proj}")
        elif name == "Back-end":
            if proj > 4.0:
                print(f"  ⚠️  WARNING: Back-end should be ~3.5-4.0, got {proj}")


def test_total_bases():
    """Test total bases projections."""
    print("\n" + "="*70)
    print("TOTAL BASES PROJECTIONS TEST")
    print("="*70)

    # League average
    league_avg = {
        'pa': 500, 'avg': 0.248, 'slg': 0.399, 'iso': 0.151,
        'k_rate': 22.7, 'bb_rate': 8.3,
        'recent_barrel_rate': 7.5, 'recent_hard_hit_pct': 37.0,
        'recent_ev90': 88.5, 'xslg': 0.399
    }

    # Power hitter
    power = {
        'pa': 550, 'avg': 0.280, 'slg': 0.550, 'iso': 0.270,
        'k_rate': 25.0, 'bb_rate': 10.0,
        'recent_barrel_rate': 15.0, 'recent_hard_hit_pct': 50.0,
        'recent_ev90': 92.0, 'xslg': 0.550
    }

    # Contact hitter
    contact = {
        'pa': 500, 'avg': 0.310, 'slg': 0.420, 'iso': 0.110,
        'k_rate': 15.0, 'bb_rate': 10.0,
        'recent_barrel_rate': 4.0, 'recent_hard_hit_pct': 32.0,
        'recent_ev90': 85.0, 'xslg': 0.420
    }

    line = 1.5

    for name, profile in [("League Avg", league_avg), ("Power", power), ("Contact", contact)]:
        result = project_batter_total_bases(profile)
        proj = result['projection']

        # Calculate edge
        prob = calculate_over_under_probability(proj, line, 'total_bases')

        print(f"\n{name:20} Projection: {proj:6.2f}  vs Line {line}")
        print(f"  {prob['pick']:>5} {prob['confidence']:.1%} confidence, rating {prob['rating']}, edge {prob['edge']:.1%}")

        # Reality check
        if name == "League Avg":
            if abs(proj - 1.5) > 0.2:
                print(f"  ⚠️  WARNING: Avg should be ~1.5, got {proj}")
        elif name == "Power":
            if proj < 1.9:
                print(f"  ⚠️  WARNING: Power should be ~2.0-2.2, got {proj}")
        elif name == "Contact":
            if proj > 1.5:
                print(f"  ⚠️  WARNING: Contact should be ~1.4-1.5, got {proj}")


if __name__ == "__main__":
    test_fantasy_score()
    test_pitcher_ks()
    test_total_bases()
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
