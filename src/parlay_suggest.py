"""
Parlay and slip suggestion module for MLB prop predictor.

Generates optimal parlay slip combinations from a list of predictions,
applying rules for diversification, balance, and correlation reduction.
"""

from collections import Counter, defaultdict
from itertools import combinations
import math

try:
    from src.slip_ev import quick_slip_ev, build_correlation_matrix
    HAS_SLIP_EV = True
except ImportError:
    HAS_SLIP_EV = False


# Empirical correlation coefficients between MLB prop types
# Source: TheHammer.bet study of 27,625 games (2010-2021) and BetFirm research
MLB_PROP_CORRELATIONS = {
    # (prop_a_internal, direction_a, prop_b_internal, direction_b): correlation_factor
    # Same-team batter stacking
    ("hits", "MORE", "hits", "MORE"): 0.22,          # teammates
    ("total_bases", "MORE", "total_bases", "MORE"): 0.18,
    ("hits", "MORE", "total_bases", "MORE"): 0.25,
    ("hits", "MORE", "runs", "MORE"): 0.20,
    ("hits", "MORE", "rbis", "MORE"): 0.18,
    # Pitcher K's + game under (moderate positive)
    ("pitcher_strikeouts", "MORE", "earned_runs", "LESS"): 0.30,
    ("pitcher_strikeouts", "MORE", "hits", "LESS"): 0.25,
    # Pitcher K's + outs (contrarian)
    ("pitcher_strikeouts", "MORE", "pitching_outs", "LESS"): 0.15,
    # Anti-correlations (negative)
    ("pitcher_strikeouts", "LESS", "earned_runs", "LESS"): -0.20,
    ("hits", "MORE", "batter_strikeouts", "MORE"): -0.35,
}

MIN_LEG_DIFFERENCE = 2


def _pick_identity(pick: dict) -> tuple[str, str, str]:
    """Stable identity for a pick across slips."""
    return (
        str(pick.get("player_name", "")).strip().lower(),
        str(pick.get("stat_internal") or pick.get("stat_type") or "").strip().lower(),
        str(pick.get("pick", "")).strip().upper(),
    )


def _pick_player_key(pick: dict) -> str:
    """Stable player identity for reuse checks."""
    return str(pick.get("player_name", "")).strip().lower()


def _slip_identities(picks: list[dict]) -> frozenset[tuple[str, str, str]]:
    """Return the set of leg identities in a slip."""
    return frozenset(_pick_identity(p) for p in picks)


def _overlap_count(picks_a: list[dict], picks_b: list[dict]) -> int:
    """Count shared legs between two slips."""
    return len(_slip_identities(picks_a) & _slip_identities(picks_b))


def _max_allowed_overlap(slip_size: int) -> int:
    """Require each additional slip to differ by at least two legs."""
    return max(0, slip_size - MIN_LEG_DIFFERENCE)


def _is_distinct_enough(picks: list[dict], existing_slips: list[dict], slip_size: int) -> bool:
    """Reject slips that overlap too heavily with already selected slips."""
    max_overlap = _max_allowed_overlap(slip_size)
    return all(_overlap_count(picks, slip["picks"]) <= max_overlap for slip in existing_slips)


def _portfolio_penalty(picks: list[dict], existing_slips: list[dict], slip_size: int) -> float:
    """Penalize overlap and leg reuse across the suggested-slip portfolio."""
    if not existing_slips:
        return 0.0

    max_overlap = _max_allowed_overlap(slip_size)
    overlap_penalty = 0.0
    reused_legs = Counter()
    reused_players = Counter()
    for slip in existing_slips:
        overlap = _overlap_count(picks, slip["picks"])
        if overlap > max_overlap:
            overlap_penalty += 40.0 * (overlap - max_overlap)
        reused_legs.update(_slip_identities(slip["picks"]))
        reused_players.update(_pick_player_key(p) for p in slip["picks"])

    reuse_penalty = 6.0 * sum(reused_legs.get(leg, 0) for leg in _slip_identities(picks))
    player_penalty = 4.0 * sum(reused_players.get(player, 0) for player in {_pick_player_key(p) for p in picks})
    return overlap_penalty + reuse_penalty + player_penalty


def _select_portfolio(candidates: list[dict], num_slips: int, slip_size: int) -> list[dict]:
    """Greedily select the best portfolio of slips with low overlap."""
    remaining = list(candidates)
    selected = []

    while remaining and len(selected) < num_slips:
        best_idx = None
        best_score = None

        for idx, slip in enumerate(remaining):
            if selected and not _is_distinct_enough(slip["picks"], selected, slip_size):
                continue

            score = float(slip.get("quality_score", 0))
            if "mc_ev_pct" in slip:
                score += float(slip.get("mc_ev_pct", 0)) * 0.15
            score -= _portfolio_penalty(slip["picks"], selected, slip_size)

            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break

        selected.append(remaining.pop(best_idx))

    return selected


def estimate_slip_correlation(picks: list[dict]) -> float:
    """
    Estimate total pairwise correlation factor for a slip.

    Checks each pair of picks in the slip against the MLB_PROP_CORRELATIONS dict.
    Returns a multiplier >1.0 for positive correlation, <1.0 for negative.
    For same-team picks, applies the correlation factor multiplicatively.

    Args:
        picks: List of prediction dicts, each with 'stat_internal', 'pick', and 'team'

    Returns:
        float: Correlation multiplier (>1.0 for positive, <1.0 for negative)
    """
    if len(picks) < 2:
        return 1.0

    correlation_multiplier = 1.0

    # Check all pairs
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            pick_a = picks[i]
            pick_b = picks[j]

            # Only apply correlation to same-team picks
            if pick_a.get('team') != pick_b.get('team'):
                continue

            prop_a = pick_a.get('stat_internal', '')
            direction_a = pick_a.get('pick', '')
            prop_b = pick_b.get('stat_internal', '')
            direction_b = pick_b.get('pick', '')

            # Check both orderings of the pair
            key_forward = (prop_a, direction_a, prop_b, direction_b)
            key_backward = (prop_b, direction_b, prop_a, direction_a)

            correlation_factor = MLB_PROP_CORRELATIONS.get(
                key_forward,
                MLB_PROP_CORRELATIONS.get(key_backward, 0.0)
            )

            # Apply factor multiplicatively
            if correlation_factor != 0.0:
                correlation_multiplier *= (1.0 + correlation_factor)

    return correlation_multiplier


def correlation_penalty(picks: list[dict]) -> float:
    """Compute a correlation penalty for a set of picks.

    Uses the empirical correlation matrix to measure how much the picks'
    outcomes are linked. Lower is better (more independent legs).

    Returns:
        Float between 0 and 1. 0 = fully independent, 1 = fully correlated.
    """
    if len(picks) < 2:
        return 0.0

    if not HAS_SLIP_EV:
        # Fallback: count same-team pairs
        teams = [p.get('team', '') for p in picks]
        team_counts = Counter(teams)
        pair_count = sum(c * (c - 1) / 2 for c in team_counts.values())
        max_pairs = len(picks) * (len(picks) - 1) / 2
        return pair_count / max_pairs if max_pairs > 0 else 0.0

    # Use the full correlation matrix
    try:
        legs = [{
            "team": p.get("team", ""),
            "pick": p.get("pick", "MORE"),
            "stat_type": p.get("stat_internal", p.get("stat_type", "")),
        } for p in picks]
        R = build_correlation_matrix(legs)
        n = len(R)
        # Average off-diagonal correlation
        total_corr = sum(abs(R[i][j]) for i in range(n) for j in range(i + 1, n))
        num_pairs = n * (n - 1) / 2
        return total_corr / num_pairs if num_pairs > 0 else 0.0
    except Exception:
        return 0.0


def score_slip_quality(picks: list[dict]) -> float:
    """
    Score a slip's quality on a 0-100 scale based on diversification criteria.

    Scoring breakdown (v018 — correlation-aware):
    - Direction balance (MORE/LESS mix): 0-20 points (50/50 is ideal)
    - Team diversity: 0-20 points (fewer duplicate teams is better)
    - Average confidence: 0-20 points (higher is better)
    - Prop type diversity: 0-20 points (fewer duplicate types is better)
    - Low correlation bonus: 0-20 points (less correlated legs = better)

    Args:
        picks: List of prediction dicts

    Returns:
        float: Quality score between 0 and 100
    """
    if not picks:
        return 0.0

    score = 0.0
    total = len(picks)

    # 1. Direction balance (0-20 points)
    directions = [p.get('pick') for p in picks]
    more_count = sum(1 for d in directions if d == 'MORE')

    more_pct = more_count / total if total > 0 else 0
    balance_deviation = abs(more_pct - 0.5)
    direction_score = max(0, 20 * (1 - balance_deviation))
    score += direction_score

    # 2. Team diversity (0-20 points)
    teams = [p.get('team') for p in picks]
    team_counts = Counter(teams)
    duplicate_penalty = sum(max(0, count - 2) for count in team_counts.values())
    team_diversity_score = max(0, 20 * (1 - (duplicate_penalty / total)))
    score += team_diversity_score

    # 3. Average confidence (0-20 points)
    confidences = [p.get('confidence', 0.5) for p in picks]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    confidence_score = avg_confidence * 20
    score += confidence_score

    # 4. Prop type diversity (0-20 points)
    stat_types = [p.get('stat_type') for p in picks]
    type_counts = Counter(stat_types)
    type_penalty = sum(max(0, count - 2) for count in type_counts.values())
    prop_diversity_score = max(0, 20 * (1 - (type_penalty / total)))
    score += prop_diversity_score

    # 5. Correlation penalty (0-20 points) — NEW in v018
    corr = correlation_penalty(picks)
    correlation_score = 20 * (1 - corr)  # Low correlation = high score
    score += correlation_score

    return min(100, score)


def suggest_slips(
    predictions: list[dict],
    num_slips: int = 3,
    slip_size: int = 5
) -> list[dict]:
    """
    Generate optimal parlay slip suggestions from a list of predictions.

    Applies heuristic rules to build diverse, balanced slips:
    - Mix MORE and LESS picks (aim for 2-3 of each per slip)
    - Max 2 picks from the same team per slip
    - Prefer A and B rated picks
    - Prioritize picks from different games (reduce correlation)
    - Diversify prop types (don't stack same type)
    - Each slip should be unique

    Args:
        predictions: List of prediction dicts. Each dict must contain:
            - player_name (str)
            - team (str)
            - stat_type (str)
            - stat_internal (str)
            - line (float)
            - pick (str): 'MORE' or 'LESS'
            - confidence (float): 0-1
            - rating (str): 'A', 'B', 'C', 'D'
            - edge (float): expected value edge
        num_slips: Number of slip suggestions to generate
        slip_size: Number of picks per slip (default 5)

    Returns:
        List of slip dicts, each containing:
        - picks: list of selected prediction dicts
        - avg_confidence: mean confidence of picks
        - direction_balance: string like "3 MORE / 2 LESS"
        - teams_used: list of unique teams
        - risk_level: "low", "medium", or "high"
        - estimated_win_prob: probability of all picks hitting (with correlation discount)
        - label: human-readable label
        - quality_score: 0-100 slip quality metric
    """

    if not predictions or len(predictions) < slip_size:
        return []

    # Sort predictions by rating priority (A > B > C > D), then by confidence
    # Prefer positively correlated picks (research: correlation drops effective breakeven from 54.2% to ~51.5%)
    rating_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    sorted_preds = sorted(
        predictions,
        key=lambda p: (
            rating_order.get(p.get('rating', 'D'), 4),
            -p.get('confidence', 0),
            -p.get('edge', 0)  # Secondary: prefer higher edge picks
        )
    )

    # Separate MORE and LESS picks
    more_picks = [p for p in sorted_preds if p.get('pick') == 'MORE']
    less_picks = [p for p in sorted_preds if p.get('pick') == 'LESS']

    slips = []
    used_combinations = set()

    # Generate candidate slips by trying different direction balances
    # Build target mixes dynamically based on what's available
    target_mixes = []
    if slip_size == 6:
        candidates = [(3, 3), (4, 2), (2, 4), (5, 1), (1, 5)]
    else:
        candidates = [(3, 2), (2, 3), (4, 1), (1, 4)]

    for m, l in candidates:
        if m <= len(more_picks) and l <= len(less_picks) and m + l == slip_size:
            target_mixes.append((m, l))

    for more_count, less_count in target_mixes:
        # Try different combinations (cap search space to avoid blowup)
        _more_pool = min(more_count, len(more_picks))
        _less_pool = min(less_count, len(less_picks))
        _combo_limit = 50  # Don't evaluate more than 50 combos per mix

        _combos_tried = 0
        for more_combo in combinations(range(min(len(more_picks), 15)), _more_pool):
            if len(slips) >= num_slips or _combos_tried > _combo_limit:
                break
            for less_combo in combinations(range(min(len(less_picks), 15)), _less_pool):
                if len(slips) >= num_slips or _combos_tried > _combo_limit:
                    break
                _combos_tried += 1

                selected_more = [more_picks[i] for i in more_combo]
                selected_less = [less_picks[j] for j in less_combo]
                picks = selected_more + selected_less

                if len(picks) != slip_size:
                    continue

                # Validate slip constraints
                if not _is_valid_slip(picks, slip_size):
                    continue

                # Check for uniqueness
                combo_key = frozenset(
                    (p.get('player_name'), p.get('stat_type'), p.get('pick'))
                    for p in picks
                )
                if combo_key in used_combinations:
                    continue

                used_combinations.add(combo_key)

                # Build slip dict
                slip = _build_slip_dict(picks, more_count, less_count)
                slips.append(slip)

    # If we don't have enough slips, use a fallback greedy approach
    if len(slips) < num_slips:
        slips.extend(
            _generate_fallback_slips(
                sorted_preds,
                slip_size,
                num_slips - len(slips),
                existing_signatures={_slip_identities(slip["picks"]) for slip in slips},
            )
        )

    slips = _select_portfolio(slips, num_slips=num_slips, slip_size=slip_size)
    if len(slips) < num_slips:
        slips.extend(
            _generate_fallback_slips(
                sorted_preds,
                slip_size,
                num_slips - len(slips),
                existing_signatures={_slip_identities(slip["picks"]) for slip in slips},
                portfolio_slips=slips,
            )
        )

    # Assign labels based on characteristics
    slips = _assign_slip_labels(slips)

    return slips


def _is_valid_slip(picks: list[dict], slip_size: int) -> bool:
    """
    Validate that a slip meets diversity constraints.

    Rules:
    - Exactly slip_size picks
    - Max 2 picks from the same team

    Args:
        picks: List of prediction dicts
        slip_size: Expected number of picks

    Returns:
        bool: True if valid, False otherwise
    """
    if len(picks) != slip_size:
        return False

    players = [_pick_player_key(p) for p in picks if _pick_player_key(p)]
    if len(players) != len(set(players)):
        return False

    # Max 2 from same team
    teams = [p.get('team') for p in picks]
    team_counts = Counter(teams)
    if any(count > 2 for count in team_counts.values()):
        return False

    return True


def _build_slip_dict(picks: list[dict], more_count: int, less_count: int) -> dict:
    """
    Build a slip dictionary with all required fields.

    Args:
        picks: List of prediction dicts
        more_count: Number of MORE picks
        less_count: Number of LESS picks

    Returns:
        dict: Slip with metadata
    """
    confidences = [p.get('confidence', 0.5) for p in picks]
    leg_win_probs = [p.get('win_prob', p.get('confidence', 0.5)) for p in picks]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    # Estimated perfect-hit probability from outright leg win chances.
    win_prob = 1.0
    for leg_win_prob in leg_win_probs:
        win_prob *= leg_win_prob

    correlation_mult = estimate_slip_correlation(picks)
    corr_pen = correlation_penalty(picks)
    direction_skew = abs(more_count - less_count) / len(picks) if picks else 1.0

    # Determine risk level based on confidence, concentration, and correlation.
    if avg_confidence >= 0.64 and direction_skew <= 0.34 and corr_pen <= 0.12:
        risk_level = "low"
    elif avg_confidence >= 0.58 and direction_skew <= 0.67 and corr_pen <= 0.22:
        risk_level = "medium"
    else:
        risk_level = "high"

    # Team and stat type lists
    teams = list(set(p.get('team') for p in picks))
    stat_types = list(set(p.get('stat_type') for p in picks))

    quality_score = score_slip_quality(picks)

    result = {
        'picks': picks,
        'avg_confidence': round(avg_confidence, 3),
        'direction_balance': f"{more_count} MORE / {less_count} LESS",
        'teams_used': teams,
        'stat_types': stat_types,
        'risk_level': risk_level,
        'estimated_win_prob': round(win_prob, 4),
        'quality_score': round(quality_score, 1),
        'correlation_penalty': round(corr_pen, 3),
        'direction_skew': round(direction_skew, 3),
        'correlation_factor': round(correlation_mult, 3),
        'label': None,  # Assigned later
    }

    # v018: Compute MC EV if available for more accurate ranking
    if HAS_SLIP_EV:
        try:
            n = len(picks)
            entry_type = f"{n}_flex"
            ev_result = quick_slip_ev(
                win_probs=leg_win_probs,
                entry_type=entry_type,
            )
            result['mc_ev_pct'] = ev_result.get('ev_profit_pct', 0)
            result['mc_win_rate'] = ev_result.get('prob_perfect', 0)
        except Exception:
            pass

    return result


def _generate_fallback_slips(
    sorted_preds: list[dict],
    slip_size: int,
    num_needed: int,
    existing_signatures: set[frozenset[tuple[str, str, str]]] | None = None,
    portfolio_slips: list[dict] | None = None,
) -> list[dict]:
    """
    Generate fallback slips using a greedy approach when combinatorial search insufficient.

    Args:
        sorted_preds: Pre-sorted predictions (by rating and confidence)
        slip_size: Number of picks per slip
        num_needed: Number of additional slips to generate

    Returns:
        list: Additional slip dicts
    """
    slips = []
    used_signatures = set(existing_signatures or set())
    portfolio_slips = portfolio_slips or []
    has_more_available = any(p.get("pick") == "MORE" for p in sorted_preds)
    has_less_available = any(p.get("pick") == "LESS" for p in sorted_preds)

    for start_idx in range(0, len(sorted_preds) - slip_size, max(1, slip_size // 2)):
        if len(slips) >= num_needed:
            break

        candidate_picks = sorted_preds[start_idx:start_idx + slip_size * 2]

        # Greedy selection
        selected = []
        used_teams = defaultdict(int)
        used_types = defaultdict(int)

        for pick in candidate_picks:
            if len(selected) >= slip_size:
                break

            team = pick.get('team')
            stat_type = pick.get('stat_type')
            player_key = _pick_player_key(pick)

            # Constraint checks — relaxed to allow up to 3 of same type
            # (early season may have limited prop diversity)
            if used_teams[team] >= 2:
                continue
            if used_types[stat_type] >= 3:
                continue
            if any(_pick_player_key(existing) == player_key for existing in selected):
                continue

            selected.append(pick)
            used_teams[team] += 1
            used_types[stat_type] += 1

        if len(selected) == slip_size:
            signature = _slip_identities(selected)
            if signature in used_signatures:
                continue
            if portfolio_slips and not _is_distinct_enough(selected, portfolio_slips + slips, slip_size):
                continue
            more_count = sum(1 for p in selected if p.get('pick') == 'MORE')
            less_count = slip_size - more_count
            direction_skew = abs(more_count - less_count) / slip_size if slip_size else 1.0
            if has_more_available and has_less_available and direction_skew > 0.67:
                continue
            slip = _build_slip_dict(selected, more_count, less_count)
            slips.append(slip)
            used_signatures.add(signature)

    return slips


def _assign_slip_labels(slips: list[dict]) -> list[dict]:
    """
    Assign human-readable labels to slips based on characteristics.

    Labels:
    - "Best Value Slip" (highest quality_score)
    - "Conservative Slip" (highest avg_confidence, low risk)
    - "Aggressive Upside" (lowest avg_confidence, high potential)
    - "Balanced Slip" (medium confidence, balanced)

    Args:
        slips: List of slip dicts

    Returns:
        list: Slips with assigned labels
    """
    if not slips:
        return slips

    labels_assigned = {}

    # Best Value (highest quality score)
    if slips:
        best_idx = max(range(len(slips)), key=lambda i: slips[i].get('quality_score', 0))
        labels_assigned[best_idx] = "Best Value Slip"

    # Conservative (highest confidence, low risk)
    low_risk_slips = [i for i, s in enumerate(slips) if s.get('risk_level') == 'low']
    if low_risk_slips:
        conservative_idx = max(low_risk_slips, key=lambda i: slips[i].get('avg_confidence', 0))
        if conservative_idx not in labels_assigned:
            labels_assigned[conservative_idx] = "Conservative Slip"

    # Aggressive (highest upside potential)
    high_risk_slips = [i for i, s in enumerate(slips) if s.get('risk_level') == 'high']
    if high_risk_slips:
        aggressive_idx = max(high_risk_slips, key=lambda i: slips[i].get('estimated_win_prob', 0))
        if aggressive_idx not in labels_assigned:
            labels_assigned[aggressive_idx] = "Aggressive Upside"

    # Balanced (medium risk, middle-ground confidence)
    for i, slip in enumerate(slips):
        if i not in labels_assigned:
            if slip.get('risk_level') == 'medium':
                labels_assigned[i] = "Balanced Slip"
                break

    # Fallback labels
    for i, slip in enumerate(slips):
        if i not in labels_assigned:
            labels_assigned[i] = f"Slip #{i + 1}"

    for i, slip in enumerate(slips):
        slip['label'] = labels_assigned.get(i, f"Slip #{i + 1}")

    return slips
