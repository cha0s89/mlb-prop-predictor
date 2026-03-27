"""
Pitcher-Batter Batted Ball Interaction Modeling

Research basis:
  - A flyball hitter facing a flyball pitcher has elevated HR/TB upside
  - A groundball pitcher suppresses extra-base hits for any batter
  - Line drive hitters have elevated BABIP (hits) regardless of pitcher type
  - Pitcher whiff rate × batter swing-and-miss tendency drives K props

Bayesian stabilization (BBE needed for 50% regression):
  - GB%/FB%: ~100 BBE (FanGraphs reliability research)
  - LD%:     ~500 BBE (most regressive batted-ball rate)
  - HR/FB%:  ~300 BBE for pitchers (very volatile)
  - SwStr%:  ~150 BBE

League averages (2025 season):
  - GB%: 43.0%, FB%: 36.0%, LD%: 21.0%
  - Pitcher HR/FB%: 12.0%
  - SwStr%: 11.3%
"""

import logging

_log = logging.getLogger(__name__)

# ── League averages ──────────────────────────────────────────────────────────
LG_GB_PCT  = 43.0   # batter/pitcher ground ball %
LG_FB_PCT  = 36.0   # batter/pitcher fly ball %
LG_LD_PCT  = 21.0   # batter line drive %
LG_HR_FB   = 12.0   # pitcher HR per fly ball %
LG_SWSTR   = 11.3   # pitcher swinging strike %
LG_K_RATE  = 22.7   # batter K%

# ── Stabilization constants (BBE for 50% regression) ────────────────────────
STAB_GB    = 100
STAB_FB    = 100
STAB_LD    = 500
STAB_HR_FB = 300
STAB_SWSTR = 150


def _regress(observed: float, n: int, stab: int, prior: float) -> float:
    """Bayesian regression toward the mean: True Talent = (n·obs + stab·prior) / (n + stab)."""
    if n <= 0:
        return prior
    return (n * observed + stab * prior) / (n + stab)


def _bbe_from_batter(profile: dict) -> int:
    """Estimate batted-ball events from a batter profile (≈ 66% of PA)."""
    pa = profile.get("pa", 0)
    return max(int(pa * 0.66), 0)


def _bbe_from_pitcher(profile: dict) -> int:
    """Estimate batted-ball events from a pitcher profile (≈ 2.7 BBE per IP)."""
    ip = profile.get("ip", 0.0)
    return max(int(ip * 2.7), 0)


def compute_batted_ball_interaction(batter: dict, pitcher: dict) -> dict:
    """
    Compute interaction multipliers for a specific batter vs pitcher matchup.

    Multipliers are applied AFTER platoon splits (inside projection functions)
    and BEFORE confidence/distribution fitting in generate_prediction().

    Returns
    -------
    dict with keys:
        tb_mult   : multiplier for Total Bases projection
        hr_mult   : multiplier for Home Run projection
        hits_mult : multiplier for Hits projection (LD% BABIP effect)
        k_mult    : multiplier for batter Strikeout projection
        source    : human-readable description of active signals
    """
    result = {
        "tb_mult":   1.0,
        "hr_mult":   1.0,
        "hits_mult": 1.0,
        "k_mult":    1.0,
        "source":    "no_data",
    }

    if not batter and not pitcher:
        return result

    batter  = batter  or {}
    pitcher = pitcher or {}

    # ── Batted-ball sample sizes ─────────────────────────────────────────────
    b_bbe = _bbe_from_batter(batter)
    p_bbe = _bbe_from_pitcher(pitcher)

    b_gb_raw   = batter.get("gb_pct",  0.0)
    b_fb_raw   = batter.get("fb_pct",  0.0)
    b_ld_raw   = batter.get("ld_pct",  0.0)

    p_gb_raw   = pitcher.get("gb_pct",  0.0)
    p_fb_raw   = pitcher.get("fb_pct",  0.0)
    p_hr_fb_raw = pitcher.get("hr_fb",  0.0)
    p_swstr_raw = pitcher.get("recent_swstr_pct", 0.0)

    has_batter_bb = b_bbe >= 30 and (b_gb_raw > 0 or b_fb_raw > 0 or b_ld_raw > 0)
    has_pitcher_bb = p_bbe >= 30 and (p_gb_raw > 0 or p_fb_raw > 0)

    if not has_batter_bb and not has_pitcher_bb and p_swstr_raw == 0:
        result["source"] = "insufficient_data"
        return result

    # ── Bayesian regression toward league averages ───────────────────────────
    b_gb = _regress(b_gb_raw, b_bbe, STAB_GB, LG_GB_PCT) if has_batter_bb else LG_GB_PCT
    b_fb = _regress(b_fb_raw, b_bbe, STAB_FB, LG_FB_PCT) if has_batter_bb else LG_FB_PCT
    b_ld = _regress(b_ld_raw, b_bbe, STAB_LD, LG_LD_PCT) if has_batter_bb else LG_LD_PCT

    p_gb = _regress(p_gb_raw, p_bbe, STAB_GB, LG_GB_PCT) if has_pitcher_bb else LG_GB_PCT
    p_fb = _regress(p_fb_raw, p_bbe, STAB_FB, LG_FB_PCT) if has_pitcher_bb else LG_FB_PCT
    p_hr_fb = (
        _regress(p_hr_fb_raw, p_bbe, STAB_HR_FB, LG_HR_FB)
        if has_pitcher_bb and p_hr_fb_raw > 0
        else LG_HR_FB
    )

    sources: list[str] = []

    # ════════════════════════════════════════════════════════════════════════
    # POWER STAT INTERACTIONS (Total Bases / Home Runs)
    # ════════════════════════════════════════════════════════════════════════

    # 1. Flyball-flyball matchup: batter FB% > 40% AND pitcher FB% > 40%
    #    Both players tendency → elevated HR/TB upside (5-10% boost)
    if b_fb > 40.0 and p_fb > 40.0:
        batter_excess  = min((b_fb - 40.0) / 10.0, 1.5)   # 0 at 40%, 1 at 50%
        pitcher_excess = min((p_fb - 40.0) / 10.0, 1.5)
        # Interaction term: product of excesses, capped
        interaction = min(batter_excess * pitcher_excess, 1.0)
        boost = 0.05 + 0.05 * interaction                   # 5-10%
        result["tb_mult"] *= (1.0 + boost)
        result["hr_mult"] *= (1.0 + boost * 1.25)           # HR slightly more sensitive
        sources.append(
            f"flyball_matchup(b_fb={b_fb:.1f}%,p_fb={p_fb:.1f}%,+{boost:.1%})"
        )

    # 2. Groundball pitcher suppression: pitcher GB% > 50%
    #    Fewer fly balls in play → suppressed extra-base hits (5-8% suppress)
    if p_gb > 50.0:
        suppress_factor = min((p_gb - 50.0) / 10.0, 1.0)   # 0 at 50%, 1 at 60%
        suppress = 0.05 + 0.03 * suppress_factor             # 5-8%
        result["tb_mult"] *= (1.0 - suppress)
        result["hr_mult"] *= (1.0 - suppress * 1.35)         # HRs suppressed most
        sources.append(
            f"gb_pitcher(p_gb={p_gb:.1f}%,-{suppress:.1%})"
        )

    # 3. Pitcher HR/FB% interaction with flyball batter
    #    High HR/FB pitcher AND batter hits flyballs → additional HR upside
    if has_pitcher_bb and p_hr_fb > 0 and b_fb > 38.0:
        hr_fb_ratio = p_hr_fb / LG_HR_FB          # > 1.0 means pitcher allows more HRs per FB
        fb_weight   = min((b_fb - 38.0) / 12.0, 1.0)  # scale: 0 at 38%, 1 at 50%
        hr_fb_boost = (hr_fb_ratio - 1.0) * fb_weight * 0.40
        if abs(hr_fb_boost) > 0.005:
            result["hr_mult"] *= (1.0 + hr_fb_boost)
            sources.append(
                f"hr_fb_interaction(p_hr_fb={p_hr_fb:.1f}%,ratio={hr_fb_ratio:.2f})"
            )

    # ════════════════════════════════════════════════════════════════════════
    # HITS INTERACTION (BABIP-driven LD% effect)
    # ════════════════════════════════════════════════════════════════════════

    # 4. Line drive hitter: LD% > 22% boosts BABIP → slight hits uplift
    #    Works regardless of pitcher type (LDs fall in at ~0.720)
    if b_ld > 22.0:
        ld_boost = min((b_ld - 22.0) / 10.0, 1.0) * 0.04  # up to +4% per 10% above avg
        result["hits_mult"] *= (1.0 + ld_boost)
        sources.append(f"ld_hitter(b_ld={b_ld:.1f}%,+{ld_boost:.1%})")

    # ════════════════════════════════════════════════════════════════════════
    # K STRIKEOUT INTERACTION (whiff rate × batter swing-and-miss)
    # ════════════════════════════════════════════════════════════════════════

    # 5. Pitcher swinging-strike rate × batter K% interaction
    #    High-K batter vs high-whiff pitcher amplifies strikeout probability
    if p_swstr_raw > 0:
        p_swstr = _regress(p_swstr_raw, p_bbe if p_bbe > 0 else 50, STAB_SWSTR, LG_SWSTR)
        b_k_pct = batter.get("k_rate", LG_K_RATE)   # already a % (e.g., 22.7)

        # Normalize both to league average
        p_factor = p_swstr / LG_SWSTR
        b_factor = b_k_pct  / LG_K_RATE

        # Interaction: both above-average → amplified K probability
        raw_interaction = p_factor * b_factor
        k_delta = (raw_interaction - 1.0) * 0.30     # scale interaction to 30% weight
        k_mult  = max(0.85, min(1.0 + k_delta, 1.20))  # cap ±15-20%

        if abs(k_mult - 1.0) > 0.005:
            result["k_mult"] *= k_mult
            sources.append(
                f"whiff_interaction(p_swstr={p_swstr:.1f}%,b_k%={b_k_pct:.1f}%,x{k_mult:.3f})"
            )

    result["source"] = "; ".join(sources) if sources else "no_signal"
    _log.debug("batted_ball_interaction: %s", result)
    return result
