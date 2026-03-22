"""
Comprehensive Math Verification Test for MLB Prop Predictor

Tests all projection functions with known player profiles across multiple contexts.
Validates:
  1. Mathematical correctness of projection calculations
  2. Sanity checks on output values
  3. Proper regression and weighting
  4. Context effects (park, lineup position, opponent quality)
  5. Probability calibration and distribution properties
  6. Generate_prediction integration
"""

import sys
import os
from datetime import date

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import predictor
from src.predictor import (
    project_pitcher_strikeouts, project_pitcher_outs, project_pitcher_earned_runs,
    project_pitcher_walks, project_pitcher_hits_allowed,
    project_batter_hits, project_batter_runs, project_batter_rbis,
    project_batter_home_runs, project_batter_stolen_bases, project_batter_total_bases,
    project_batter_strikeouts, project_batter_walks, project_hitter_fantasy_score,
    project_hits_runs_rbis, generate_prediction,
    estimate_plate_appearances, estimate_batters_faced,
    LG, STAB, PARK, PARK_HR, PARK_K, PARK_SB
)


# ═══════════════════════════════════════════════════════
# PLAYER PROFILES
# ═══════════════════════════════════════════════════════

ELITE_BATTER = {
    # Season stats
    "name": "Elite Batter",
    "avg": 0.300, "obp": 0.400, "slg": 0.550, "iso": 0.250,
    "pa": 600, "ab": 520, "h": 156, "r": 100, "rbi": 100, "hr": 30, "sb": 20, "2b": 25, "3b": 2,
    "bb": 50, "bb_rate": 8.3, "k": 80, "k_rate": 13.3, "woba": 0.370, "babip": 0.330,
    # Statcast metrics
    "xba": 0.300, "xslg": 0.550, "recent_barrel_rate": 15.0, "barrel_rate": 15.0,
    "recent_hard_hit_pct": 45.0, "hard_hit_pct": 45.0, "recent_ev90": 108.0, "ev90": 108.0,
    "sprint_speed": 30.0, "contact_rate": 86.0,
}

AVERAGE_BATTER = {
    "name": "Average Batter",
    "avg": LG["avg"], "obp": LG["obp"], "slg": LG["slg"], "iso": LG["iso"],
    "pa": 500, "ab": 440, "h": 109, "r": 65, "rbi": 65, "hr": 18, "sb": 8, "2b": 22, "3b": 1,
    "bb": 42, "bb_rate": LG["bb_rate"], "k": 105, "k_rate": LG["k_rate"], "woba": LG["woba"], "babip": LG["babip"],
    "xba": LG["xba"], "xslg": LG["xslg"], "barrel_rate": LG["barrel_rate"],
    "hard_hit_pct": LG["hard_hit_pct"], "ev90": LG["ev90"],
    "sprint_speed": LG["sprint_speed"], "contact_rate": 77.0,
}

TERRIBLE_BATTER = {
    "name": "Terrible Batter",
    "avg": 0.180, "obp": 0.230, "slg": 0.270, "iso": 0.090,
    "pa": 200, "ab": 180, "h": 32, "r": 15, "rbi": 15, "hr": 3, "sb": 1, "2b": 5, "3b": 0,
    "bb": 10, "bb_rate": 5.0, "k": 60, "k_rate": 30.0, "woba": 0.220, "babip": 0.240,
    "xba": 0.185, "xslg": 0.280, "barrel_rate": 3.0,
    "hard_hit_pct": 25.0, "ev90": 100.0,
    "sprint_speed": 24.0, "contact_rate": 70.0,
}

ROOKIE_BATTER = {
    "name": "Rookie/Unknown",
    "avg": 0.250, "obp": 0.310, "slg": 0.400, "iso": 0.150,
    "pa": 50, "ab": 44, "h": 11, "r": 5, "rbi": 5, "hr": 2, "sb": 1, "2b": 2, "3b": 0,
    "bb": 4, "bb_rate": 8.0, "k": 12, "k_rate": 24.0, "woba": 0.300, "babip": 0.270,
    "xba": 0.250, "xslg": 0.400, "barrel_rate": 6.0,
    "hard_hit_pct": 35.0, "ev90": 103.0,
    "sprint_speed": 27.0, "contact_rate": 76.0,
}

ELITE_SP = {
    "name": "Elite SP",
    "ip": 180, "gs": 32, "ip_per_start": 5.625,
    "era": 2.80, "fip": 3.10, "xfip": 3.05, "k9": 10.5, "bb9": 2.2, "hr9": 0.95,
    "k_pct": 28.5, "k_rate": 0.285, "bb_pct": 6.5, "bb_rate": 0.065, "whip": 1.05,
    "h": 150, "bb": 39, "k": 189, "er": 56, "bf": 750,
    "recent_csw_pct": 32.0, "csw_pct": 32.0, "recent_swstr_pct": 14.0, "swstr_pct": 14.0,
}

TERRIBLE_SP = {
    "name": "Terrible SP",
    "ip": 100, "gs": 18, "ip_per_start": 5.556,
    "era": 5.80, "fip": 5.50, "xfip": 5.45, "k9": 6.0, "bb9": 4.5, "hr9": 1.85,
    "k_pct": 17.0, "k_rate": 0.170, "bb_pct": 12.8, "bb_rate": 0.128, "whip": 1.55,
    "h": 125, "bb": 50, "k": 70, "er": 64, "bf": 450,
    "recent_csw_pct": 24.0, "csw_pct": 24.0, "recent_swstr_pct": 8.5, "swstr_pct": 8.5,
}


# ═══════════════════════════════════════════════════════
# SANITY CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════

def flag_issues(projection_dict, prop_type, expected_range=(0, float('inf'))):
    """
    Check for obviously wrong projections.
    Returns list of issues or empty list if OK.
    """
    issues = []
    proj = projection_dict.get("projection", 0)

    # Check reasonable ranges
    if proj < expected_range[0] or proj > expected_range[1]:
        issues.append(f"OUT_OF_RANGE: {proj:.2f} not in [{expected_range[0]}, {expected_range[1]}]")

    # Sanity checks by property type
    if prop_type in ("hits", "runs", "rbis"):
        if proj > 5:
            issues.append(f"SUSPICIOUS_HIGH: {prop_type} projects {proj:.2f} (max ~4-5)")
    elif prop_type == "home_runs":
        if proj > 2:
            issues.append(f"SUSPICIOUS_HIGH: {prop_type} projects {proj:.2f} (max ~1.5 for game)")
    elif prop_type == "stolen_bases":
        if proj > 1:
            issues.append(f"SUSPICIOUS_HIGH: {prop_type} projects {proj:.2f} (max ~1 per game)")
    elif prop_type == "pitcher_strikeouts":
        if proj > 12:
            issues.append(f"SUSPICIOUS_HIGH: {prop_type} projects {proj:.2f} (max capped at 12)")
    elif prop_type == "earned_runs":
        if proj > 8:
            issues.append(f"SUSPICIOUS_HIGH: {prop_type} projects {proj:.2f} (typical 1-5)")
    elif prop_type == "pitcher_outs":
        if proj < 9 or proj > 25:
            issues.append(f"OUT_OF_RANGE: {prop_type} projects {proj:.2f} (typical 9-21)")

    return issues


def test_projection_function(func_name, func, player, contexts):
    """Test a single projection function across multiple contexts."""
    print(f"\n{'='*80}")
    print(f"Testing: {func_name}")
    print(f"Player: {player.get('name', 'Unknown')}")
    print(f"{'='*80}")

    for context_name, kwargs in contexts:
        result = func(player, **kwargs)
        projection = result.get("projection", 0)

        print(f"\n  Context: {context_name}")
        print(f"    Projection: {projection:.3f}")

        # Print additional details if available
        for key in ["regressed_avg", "regressed_slg", "regressed_k_pct", "expected_pa",
                    "expected_ab", "expected_ip", "expected_bf", "regressed_bb_pct"]:
            if key in result:
                print(f"    {key}: {result[key]}")

        issues = flag_issues(result, func_name.replace("project_batter_", "").replace("project_pitcher_", ""))
        if issues:
            for issue in issues:
                print(f"    ⚠️  {issue}")


# ═══════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════

def test_batter_projections():
    """Test all batter projection functions."""
    print("\n" + "="*100)
    print("BATTER PROJECTIONS TEST")
    print("="*100)

    batters = [ELITE_BATTER, AVERAGE_BATTER, TERRIBLE_BATTER, ROOKIE_BATTER]

    # Neutral context
    neutral_contexts = [
        ("Neutral (no park/opp)", {}),
    ]

    # vs Elite SP at Coors (COL), batting 3rd
    elite_sp_coors_contexts = [
        ("vs Elite SP at Coors, batting 3", {
            "opp_p": ELITE_SP,
            "park": "COL",
            "lineup_pos": 3,
        }),
    ]

    # vs Terrible SP at pitcher's park (SF), batting 7th
    terrible_sp_sf_contexts = [
        ("vs Terrible SP at SF, batting 7", {
            "opp_p": TERRIBLE_SP,
            "park": "SF",
            "lineup_pos": 7,
        }),
    ]

    for batter in batters:
        print(f"\n\n{'#'*100}")
        print(f"# {batter['name'].upper()}")
        print(f"# AVG:{batter['avg']:.3f} OBP:{batter['obp']:.3f} SLG:{batter['slg']:.3f} PA:{batter['pa']}")
        print(f"{'#'*100}")

        all_contexts = neutral_contexts + elite_sp_coors_contexts + terrible_sp_sf_contexts

        # Hits
        print("\n\n--- HITS ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_hits(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "hits", (0, 5))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} H")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Runs
        print("\n--- RUNS ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_runs(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "runs", (0, 4))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} R")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # RBIs
        print("\n--- RBIs ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_rbis(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "rbis", (0, 5))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} RBI")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Home Runs
        print("\n--- HOME RUNS ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_home_runs(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "home_runs", (0, 2))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.4f} HR")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Stolen Bases
        print("\n--- STOLEN BASES ---")
        for ctx_name, kwargs in all_contexts:
            # SB only takes park parameter
            sb_kwargs = {"park": kwargs.get("park")}
            result = project_batter_stolen_bases(batter, **sb_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "stolen_bases", (0, 1.5))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} SB")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Total Bases
        print("\n--- TOTAL BASES ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_total_bases(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "total_bases", (0, 8))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} TB")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Strikeouts
        print("\n--- STRIKEOUTS (Batter) ---")
        for ctx_name, kwargs in all_contexts:
            result = project_batter_strikeouts(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "strikeouts", (0, 3))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} K")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Walks
        print("\n--- WALKS (Batter) ---")
        for ctx_name, kwargs in all_contexts:
            # Walks takes opp_p, ump, lineup_pos
            walks_kwargs = {k: v for k, v in kwargs.items() if k in ["opp_p", "ump", "lineup_pos"]}
            result = project_batter_walks(batter, **walks_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "walks", (0, 2))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} BB")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Fantasy Score
        print("\n--- HITTER FANTASY SCORE ---")
        for ctx_name, kwargs in all_contexts:
            result = project_hitter_fantasy_score(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "hitter_fantasy_score", (0, 20))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} PTS")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Hits+Runs+RBIs
        print("\n--- HITS+RUNS+RBIs ---")
        for ctx_name, kwargs in all_contexts:
            result = project_hits_runs_rbis(batter, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "hits_runs_rbis", (0, 12))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} H+R+RBI")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")


def test_pitcher_projections():
    """Test all pitcher projection functions."""
    print("\n\n" + "="*100)
    print("PITCHER PROJECTIONS TEST")
    print("="*100)

    pitchers = [ELITE_SP, TERRIBLE_SP]

    neutral_contexts = [
        ("Neutral", {}),
    ]

    coors_contexts = [
        ("At Coors (bad lineup)", {
            "park": "COL",
            "opp_k_rate": 25.0,
        }),
    ]

    sf_contexts = [
        ("At SF (good lineup)", {
            "park": "SF",
            "opp_k_rate": 20.0,
        }),
    ]

    for pitcher in pitchers:
        print(f"\n\n{'#'*100}")
        print(f"# {pitcher['name'].upper()}")
        print(f"# ERA:{pitcher['era']:.2f} K/9:{pitcher['k9']:.2f} BB/9:{pitcher['bb9']:.2f} IP:{pitcher['ip']}")
        print(f"{'#'*100}")

        all_contexts = neutral_contexts + coors_contexts + sf_contexts

        # Strikeouts
        print("\n--- STRIKEOUTS (Pitcher) ---")
        for ctx_name, kwargs in all_contexts:
            result = project_pitcher_strikeouts(pitcher, **kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "pitcher_strikeouts", (0, 12))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} K")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Outs
        print("\n--- OUTS ---")
        for ctx_name, kwargs in all_contexts:
            outs_kwargs = {k: v for k, v in kwargs.items() if k in ["park", "wx"]}
            result = project_pitcher_outs(pitcher, **outs_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "pitcher_outs", (9, 22))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} Outs")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Earned Runs
        print("\n--- EARNED RUNS ---")
        for ctx_name, kwargs in all_contexts:
            er_kwargs = {k: v for k, v in kwargs.items() if k in ["park", "wx"]}
            # For context with opp_k_rate, set opp_woba accordingly
            if "opp_k_rate" in kwargs:
                er_kwargs["opp_woba"] = 0.290 if kwargs.get("opp_k_rate", 22) > 23 else 0.330
            result = project_pitcher_earned_runs(pitcher, **er_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "earned_runs", (0, 8))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} ER")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Walks
        print("\n--- WALKS ALLOWED ---")
        for ctx_name, kwargs in all_contexts:
            walks_kwargs = {k: v for k, v in kwargs.items() if k in ["park"]}
            result = project_pitcher_walks(pitcher, **walks_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "walks_allowed", (0, 6))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} BB")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")

        # Hits Allowed
        print("\n--- HITS ALLOWED ---")
        for ctx_name, kwargs in all_contexts:
            h_kwargs = {k: v for k, v in kwargs.items() if k in ["park", "wx"]}
            result = project_pitcher_hits_allowed(pitcher, **h_kwargs)
            proj = result["projection"]
            issues = flag_issues(result, "hits_allowed", (0, 12))
            status = "✓" if not issues else "✗"
            print(f"{status} {ctx_name:40} → {proj:.3f} H")
            if issues:
                for issue in issues:
                    print(f"   ⚠️  {issue}")


def test_generate_prediction():
    """Test generate_prediction with various stat types and lines."""
    print("\n\n" + "="*100)
    print("GENERATE_PREDICTION INTEGRATION TEST")
    print("="*100)

    test_cases = [
        # Batter props
        ("Elite Batter", ELITE_BATTER, None, "hits", "hits", 1.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "runs", "runs", 0.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "rbis", "rbis", 1.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "home_runs", "home_runs", 0.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "stolen_bases", "stolen_bases", 0.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "total_bases", "total_bases", 2.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "strikeouts", "batter_strikeouts", 1.5, 3),
        ("Elite Batter", ELITE_BATTER, None, "hitter_fantasy_score", "hitter_fantasy_score", 7.5, 3),

        # Terrible Batter
        ("Terrible Batter", TERRIBLE_BATTER, None, "hits", "hits", 0.5, 3),
        ("Terrible Batter", TERRIBLE_BATTER, None, "runs", "runs", 0.5, 3),
        ("Terrible Batter", TERRIBLE_BATTER, None, "rbis", "rbis", 0.5, 3),

        # Pitcher props
        ("Elite SP", None, ELITE_SP, "strikeouts", "pitcher_strikeouts", 7.5, 3),
        ("Elite SP", None, ELITE_SP, "earned_runs", "earned_runs", 3.5, 3),
        ("Elite SP", None, ELITE_SP, "outs", "pitching_outs", 15.5, 3),
        ("Elite SP", None, ELITE_SP, "walks", "walks_allowed", 2.5, 3),
        ("Elite SP", None, ELITE_SP, "hits", "hits_allowed", 6.5, 3),

        ("Terrible SP", None, TERRIBLE_SP, "strikeouts", "pitcher_strikeouts", 5.5, 3),
        ("Terrible SP", None, TERRIBLE_SP, "earned_runs", "earned_runs", 5.5, 3),
        ("Terrible SP", None, TERRIBLE_SP, "walks", "walks_allowed", 4.5, 3),
    ]

    results_summary = []

    for player_name, batter, pitcher, stat_type, stat_internal, line, lineup_pos in test_cases:
        result = generate_prediction(
            player_name=player_name,
            stat_type=stat_type,
            stat_internal=stat_internal,
            line=line,
            batter_profile=batter,
            pitcher_profile=pitcher,
            lineup_pos=lineup_pos if batter else None,
        )

        projection = result.get("projection", 0)
        p_over = result.get("p_over", 0)
        p_under = result.get("p_under", 0)
        confidence = result.get("confidence", 0)
        edge = result.get("edge", "NONE")
        rating = result.get("rating", "?")

        # Sanity checks
        issues = []

        # Probability calibration
        prob_sum = p_over + p_under
        if abs(prob_sum - 1.0) > 0.01:
            issues.append(f"Prob sum {prob_sum:.3f} ≠ 1.0")

        if confidence < 0.5 or confidence > 1.0:
            issues.append(f"Confidence {confidence:.3f} outside [0.5, 1.0]")

        if p_over < 0.4 or p_over > 0.6:
            pass  # Edge picks expected

        # Check edge direction matches projection
        if projection > line and edge != "MORE":
            if edge != "NONE":
                issues.append(f"Edge mismatch: proj {projection:.2f} > line {line} but edge={edge}")
        elif projection < line and edge != "LESS":
            if edge != "NONE":
                issues.append(f"Edge mismatch: proj {projection:.2f} < line {line} but edge={edge}")

        status = "✓" if not issues else "✗"
        results_summary.append({
            "player": player_name,
            "stat": stat_internal,
            "line": line,
            "projection": projection,
            "p_over": p_over,
            "p_under": p_under,
            "confidence": confidence,
            "edge": edge,
            "rating": rating,
            "issues": issues,
            "status": status,
        })

        print(f"\n{status} {player_name:20} {stat_internal:25} @ {line}")
        print(f"   Projection: {projection:.3f} | Over: {p_over:.3f} Under: {p_under:.3f}")
        print(f"   Confidence: {confidence:.3f} | Edge: {edge:5} | Rating: {rating}")
        if issues:
            for issue in issues:
                print(f"   ⚠️  {issue}")

    # Summary table
    print("\n\n" + "="*100)
    print("GENERATE_PREDICTION SUMMARY")
    print("="*100)
    print(f"{'Player':<20} {'Stat':<25} {'Line':>6} {'Proj':>7} {'P(O)':>6} {'Conf':>6} {'Edge':>6} {'Rating':>6} {'Status':<3}")
    print("-" * 100)
    for r in results_summary:
        issues_str = f" ({len(r['issues'])} issues)" if r['issues'] else ""
        print(f"{r['player']:<20} {r['stat']:<25} {r['line']:>6.1f} {r['projection']:>7.3f} "
              f"{r['p_over']:>6.3f} {r['confidence']:>6.3f} {r['edge']:>6} {r['rating']:>6} {r['status']:<3}{issues_str}")


def test_comparative_analysis():
    """Compare projections across player tiers to ensure proper ranking."""
    print("\n\n" + "="*100)
    print("COMPARATIVE ANALYSIS (Elite vs Average vs Terrible)")
    print("="*100)

    batters_to_compare = [ELITE_BATTER, AVERAGE_BATTER, TERRIBLE_BATTER]
    pitchers_to_compare = [ELITE_SP, TERRIBLE_SP]

    # Batter comparison
    print("\n--- BATTER PROJECTIONS ---")
    print("(Higher stat quality should project higher for all props)")
    print()

    props = [
        ("Hits", project_batter_hits),
        ("Runs", project_batter_runs),
        ("RBIs", project_batter_rbis),
        ("HR", project_batter_home_runs),
        ("TB", project_batter_total_bases),
        ("K", project_batter_strikeouts),
    ]

    for prop_name, func in props:
        print(f"{prop_name}:", end="")
        projections = []
        for batter in batters_to_compare:
            result = func(batter)
            proj = result["projection"]
            projections.append(proj)
            print(f" {batter['name'][:8]:12} {proj:6.3f} |", end="")

        # Check ordering
        is_ordered = projections[0] > projections[1] > projections[2]
        status = "✓" if is_ordered else "✗ MISORDERED"
        print(f" {status}")

    # Pitcher comparison
    print("\n--- PITCHER PROJECTIONS ---")
    print("(Elite SP should project higher K, lower ER/walks than Terrible SP)")
    print()

    pitcher_props = [
        ("K", project_pitcher_strikeouts),
        ("ER", project_pitcher_earned_runs),
        ("BB", project_pitcher_walks),
        ("H", project_pitcher_hits_allowed),
    ]

    for prop_name, func in pitcher_props:
        print(f"{prop_name}:", end="")
        elite_proj = func(ELITE_SP)["projection"]
        terrible_proj = func(TERRIBLE_SP)["projection"]

        # Elite SP should have higher K, lower ER/BB/H
        if prop_name == "K":
            is_correct = elite_proj > terrible_proj
        else:
            is_correct = elite_proj < terrible_proj

        status = "✓" if is_correct else "✗ INCORRECT"
        print(f" Elite: {elite_proj:6.3f} | Terrible: {terrible_proj:6.3f} | {status}")


def test_park_effects():
    """Verify park factors move projections in expected directions."""
    print("\n\n" + "="*100)
    print("PARK FACTOR EFFECTS")
    print("="*100)

    print("\nBatter at neutral vs Coors vs SF:")
    print("(Coors should boost offense, SF should suppress)")
    print()

    neutral = project_batter_hits(AVERAGE_BATTER, park=None)["projection"]
    coors = project_batter_hits(AVERAGE_BATTER, park="COL")["projection"]
    sf = project_batter_hits(AVERAGE_BATTER, park="SF")["projection"]

    print(f"Neutral: {neutral:.3f} | Coors: {coors:.3f} | SF: {sf:.3f}")
    is_correct = neutral < coors and neutral > sf
    status = "✓" if is_correct else "✗"
    print(f"Coors > Neutral > SF: {status}")

    print("\nPitcher strikeouts at neutral vs Coors vs SF:")
    print("(SF should boost K rate, Coors should suppress)")
    print()

    p_neutral = project_pitcher_strikeouts(AVERAGE_BATTER if hasattr(AVERAGE_BATTER, 'k_pct') else ELITE_SP, park=None)["projection"]
    p_coors = project_pitcher_strikeouts(ELITE_SP, park="COL")["projection"]
    p_sf = project_pitcher_strikeouts(ELITE_SP, park="SF")["projection"]

    print(f"Neutral: {p_neutral:.3f} | Coors: {p_coors:.3f} | SF: {p_sf:.3f}")


if __name__ == "__main__":
    print("\n")
    print("█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + "  MLB PROP PREDICTOR — COMPREHENSIVE MATH VERIFICATION TEST".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)

    test_batter_projections()
    test_pitcher_projections()
    test_generate_prediction()
    test_comparative_analysis()
    test_park_effects()

    print("\n\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + "  TEST SUITE COMPLETE".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100 + "\n")
