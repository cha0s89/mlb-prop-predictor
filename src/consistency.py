"""
Cross-Prop Consistency Checks

Ensures projections respect logical constraints:
- Total Bases >= Hits (always true — every hit is at least 1 TB)
- Fantasy Score should be consistent with component projections
- Hits+Runs+RBIs >= Hits (trivially true but catches sign errors)
- Singles <= Hits
- Pitcher Ks <= Pitching Outs (K is a subset of outs)
- Pitching Outs <= 27 (max 9 innings = 27 outs)
- Earned Runs <= Runs Allowed (if both projected)
- Stolen Bases <= Hits + Walks (can't steal if not on base)

When inconsistencies are found, the weaker projection is adjusted
toward the stronger one (the one with more data support).
"""

from typing import Dict, List, Optional


# ── Sanity bounds per stat type ──────────────────────────────────────────────
# (min, max_sane, flag_above)
_SANITY_BOUNDS: Dict[str, tuple] = {
    "pitcher_strikeouts":   (0, 15,  14),
    "pitching_outs":        (0, 27,  24),
    "hits":                 (0, 5,    4),
    "total_bases":          (0, 16,  12),
    "hitter_fantasy_score": (0, 50,  40),
    "earned_runs":          (0, 12,  10),
    "batter_strikeouts":    (0, 5,    4),
    "rbis":                 (0, 8,    6),
    "runs":                 (0, 5,    4),
}


def sanity_check_projection(stat_type: str, projection: float, line: float) -> dict:
    """Check if a projection is within realistic bounds.

    Args:
        stat_type: Internal stat name (e.g. 'pitcher_strikeouts')
        projection: The model's projection
        line: The sportsbook line

    Returns:
        {"sane": bool, "reason": str, "suggested_cap": float | None}
    """
    bounds = _SANITY_BOUNDS.get(stat_type)
    if bounds is None:
        return {"sane": True, "reason": "", "suggested_cap": None}

    min_val, max_val, flag_above = bounds

    if projection < min_val:
        return {
            "sane": False,
            "reason": f"{stat_type} projection {projection:.2f} below minimum {min_val}",
            "suggested_cap": float(min_val),
        }

    if projection > flag_above:
        return {
            "sane": False,
            "reason": f"{stat_type} projection {projection:.2f} exceeds realistic cap {flag_above}",
            "suggested_cap": float(flag_above),
        }

    return {"sane": True, "reason": "", "suggested_cap": None}


def enforce_consistency(predictions: List[Dict]) -> List[Dict]:
    """Apply cross-prop consistency checks to a batch of predictions.

    Groups predictions by player, then enforces logical constraints.
    Adjustments are minimal — just enough to remove logical impossibilities.

    Args:
        predictions: List of prediction dicts with keys:
            player_name, stat_internal, projection, p_over, p_under, confidence

    Returns:
        Same list with adjustments applied and 'consistency_adj' flag added
    """
    # Group by player
    by_player: Dict[str, list] = {}
    for i, pred in enumerate(predictions):
        player = pred.get("player_name", "")
        if player not in by_player:
            by_player[player] = []
        by_player[player].append((i, pred))

    for player, player_preds in by_player.items():
        # Build lookup: stat_internal → (index, pred)
        lookup: Dict[str, tuple] = {}
        for idx, pred in player_preds:
            stat = pred.get("stat_internal", "")
            lookup[stat] = (idx, pred)

        # ── Constraint 1: Total Bases >= Hits ──
        if "total_bases" in lookup and "hits" in lookup:
            tb_idx, tb_pred = lookup["total_bases"]
            h_idx, h_pred = lookup["hits"]
            tb_proj = tb_pred.get("projection", 0)
            h_proj = h_pred.get("projection", 0)

            if tb_proj < h_proj:
                old_tb = tb_proj
                new_tb = h_proj + 0.1  # Slight margin since TB >= H always
                predictions[tb_idx]["projection"] = new_tb
                predictions[tb_idx]["consistency_adj"] = (
                    f"TB {old_tb:.2f} → {new_tb:.2f} (must be ≥ Hits {h_proj:.2f})"
                )

        # ── Constraint 2: Hits+Runs+RBIs >= Hits ──
        if "hits_runs_rbis" in lookup and "hits" in lookup:
            hrr_idx, hrr_pred = lookup["hits_runs_rbis"]
            h_idx, h_pred = lookup["hits"]
            hrr_proj = hrr_pred.get("projection", 0)
            h_proj = h_pred.get("projection", 0)

            if hrr_proj < h_proj:
                old_hrr = hrr_proj
                new_hrr = h_proj + 0.2  # H+R+RBI must exceed H by at least R+RBI
                predictions[hrr_idx]["projection"] = new_hrr
                predictions[hrr_idx]["consistency_adj"] = (
                    f"H+R+RBI {old_hrr:.2f} → {new_hrr:.2f} (must be ≥ Hits {h_proj:.2f})"
                )

        # ── Constraint 3: Singles <= Hits ──
        if "singles" in lookup and "hits" in lookup:
            s_idx, s_pred = lookup["singles"]
            h_idx, h_pred = lookup["hits"]
            s_proj = s_pred.get("projection", 0)
            h_proj = h_pred.get("projection", 0)

            if s_proj > h_proj:
                old_s = s_proj
                new_s = h_proj * 0.65  # Singles typically ~65% of hits
                predictions[s_idx]["projection"] = new_s
                predictions[s_idx]["consistency_adj"] = (
                    f"Singles {old_s:.2f} → {new_s:.2f} (must be ≤ Hits {h_proj:.2f})"
                )

        # ── Constraint 4: Fantasy Score consistency ──
        # DraftKings scoring: 1B=3, 2B=5, 3B=8, HR=10, RBI=2, R=2, BB=2, SB=5, HBP=2
        # If we have component projections, check FS is in the right ballpark
        if "hitter_fantasy_score" in lookup and "hits" in lookup:
            fs_idx, fs_pred = lookup["hitter_fantasy_score"]
            h_idx, h_pred = lookup["hits"]
            fs_proj = fs_pred.get("projection", 0)
            h_proj = h_pred.get("projection", 0)

            # Rough implied FS from hits: avg hit ≈ 3.5 DK pts, plus runs/rbis/walks
            # Conservative estimate: ~5 DK pts per hit + ~3 baseline
            implied_fs_low = h_proj * 3.0

            if fs_proj < implied_fs_low and h_proj > 0.5:
                old_fs = fs_proj
                new_fs = implied_fs_low
                predictions[fs_idx]["projection"] = new_fs
                predictions[fs_idx]["consistency_adj"] = (
                    f"FS {old_fs:.2f} → {new_fs:.2f} (too low for {h_proj:.2f} hits)"
                )

        # ── Constraint 5: Pitcher Ks <= Pitching Outs ──
        # K is a type of out; can't have more Ks than outs recorded
        if "pitcher_strikeouts" in lookup and "pitching_outs" in lookup:
            k_idx, k_pred = lookup["pitcher_strikeouts"]
            o_idx, o_pred = lookup["pitching_outs"]
            k_proj = k_pred.get("projection", 0)
            o_proj = o_pred.get("projection", 0)

            if k_proj > o_proj and o_proj > 0:
                old_k = k_proj
                new_k = o_proj * 0.85  # Ks typically ~25-35% of outs
                predictions[k_idx]["projection"] = new_k
                predictions[k_idx]["consistency_adj"] = (
                    f"Ks {old_k:.2f} → {new_k:.2f} (can't exceed outs {o_proj:.2f})"
                )

        # ── Constraint 6: Pitching Outs <= 27 (cap at 24 for realism) ──
        if "pitching_outs" in lookup:
            o_idx, o_pred = lookup["pitching_outs"]
            o_proj = o_pred.get("projection", 0)

            if o_proj > 24:
                old_o = o_proj
                new_o = 24.0  # ~8 IP practical maximum
                predictions[o_idx]["projection"] = new_o
                predictions[o_idx]["consistency_adj"] = (
                    f"Outs {old_o:.2f} → {new_o:.2f} (capped at 24, ~8 IP)"
                )

        # ── Constraint 7: Earned Runs <= Runs Allowed ──
        # Earned runs are a subset of total runs allowed
        if "earned_runs" in lookup and "runs_allowed" in lookup:
            er_idx, er_pred = lookup["earned_runs"]
            ra_idx, ra_pred = lookup["runs_allowed"]
            er_proj = er_pred.get("projection", 0)
            ra_proj = ra_pred.get("projection", 0)

            if er_proj > ra_proj and ra_proj > 0:
                old_er = er_proj
                new_er = ra_proj
                predictions[er_idx]["projection"] = new_er
                predictions[er_idx]["consistency_adj"] = (
                    f"ER {old_er:.2f} → {new_er:.2f} (can't exceed runs allowed {ra_proj:.2f})"
                )

        # ── Constraint 8: Stolen Bases <= Hits + Walks ──
        # Can't steal if you're not on base
        if "stolen_bases" in lookup:
            sb_idx, sb_pred = lookup["stolen_bases"]
            sb_proj = sb_pred.get("projection", 0)

            on_base = 0.0
            has_on_base = False
            if "hits" in lookup:
                on_base += lookup["hits"][1].get("projection", 0)
                has_on_base = True
            if "walks" in lookup:
                on_base += lookup["walks"][1].get("projection", 0)
                has_on_base = True

            if has_on_base and sb_proj > on_base and on_base > 0:
                old_sb = sb_proj
                new_sb = on_base * 0.5  # SB is a fraction of times on base
                predictions[sb_idx]["projection"] = new_sb
                predictions[sb_idx]["consistency_adj"] = (
                    f"SB {old_sb:.2f} → {new_sb:.2f} (capped at 50% of on-base {on_base:.2f})"
                )

    return predictions


def flag_inconsistencies(predictions: List[Dict]) -> List[str]:
    """Check predictions for inconsistencies WITHOUT modifying them.

    Useful for displaying warnings to the user.

    Returns:
        List of warning strings
    """
    warnings = []
    by_player: Dict[str, dict] = {}
    for pred in predictions:
        player = pred.get("player_name", "")
        if player not in by_player:
            by_player[player] = {}
        stat = pred.get("stat_internal", "")
        by_player[player][stat] = pred.get("projection", 0)

    for player, stats in by_player.items():
        if "total_bases" in stats and "hits" in stats:
            if stats["total_bases"] < stats["hits"]:
                warnings.append(
                    f"⚠️ {player}: TB ({stats['total_bases']:.1f}) < Hits ({stats['hits']:.1f})"
                )

        if "hits_runs_rbis" in stats and "hits" in stats:
            if stats["hits_runs_rbis"] < stats["hits"]:
                warnings.append(
                    f"⚠️ {player}: H+R+RBI ({stats['hits_runs_rbis']:.1f}) < Hits ({stats['hits']:.1f})"
                )

        if "pitcher_strikeouts" in stats and "pitching_outs" in stats:
            if stats["pitcher_strikeouts"] > stats["pitching_outs"]:
                warnings.append(
                    f"⚠️ {player}: Ks ({stats['pitcher_strikeouts']:.1f}) > Outs ({stats['pitching_outs']:.1f})"
                )

        if "pitching_outs" in stats and stats["pitching_outs"] > 24:
            warnings.append(
                f"⚠️ {player}: Outs ({stats['pitching_outs']:.1f}) > 24 (unrealistic)"
            )

        if "stolen_bases" in stats:
            on_base = stats.get("hits", 0) + stats.get("walks", 0)
            if on_base > 0 and stats["stolen_bases"] > on_base:
                warnings.append(
                    f"⚠️ {player}: SB ({stats['stolen_bases']:.1f}) > On-base ({on_base:.1f})"
                )

    return warnings
