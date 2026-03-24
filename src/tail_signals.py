"""Helpers for breakout/dud labeling and human-readable drivers."""

from __future__ import annotations

import math

from src.predictor import PARK_FACTORS, PARK_FACTORS_HR, PARK_FACTORS_K
from src.weather import resolve_team


INVERSE_GOOD_PROPS = {"earned_runs", "walks_allowed", "hits_allowed", "batter_strikeouts"}
RBI_CONTEXT_PROPS = {"rbis", "hits_runs_rbis", "hitter_fantasy_score"}
RUN_CONTEXT_PROPS = {"runs", "hits_runs_rbis", "hitter_fantasy_score"}

TAIL_LABELS = {
    "earned_runs": {"breakout": "Shutdown", "dud": "Blowup Risk"},
    "hits_allowed": {"breakout": "Shutdown", "dud": "Traffic Risk"},
    "walks_allowed": {"breakout": "Command", "dud": "Wildness Risk"},
    "batter_strikeouts": {"breakout": "Contact", "dud": "Whiff Risk"},
}


def _safe_num(value, fallback=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(number) else number


def _is_high_good_prop(stat_internal: str) -> bool:
    return stat_internal not in INVERSE_GOOD_PROPS


def tail_signal_labels(stat_internal: str) -> dict[str, str]:
    labels = TAIL_LABELS.get(stat_internal, {})
    return {
        "breakout": labels.get("breakout", "Ceiling"),
        "dud": labels.get("dud", "Dud Risk"),
    }


def tail_target_text(stat_type: str, stat_internal: str, target, kind: str) -> str:
    if target is None:
        return ""
    stat_label = stat_type or stat_internal.replace("_", " ").title()
    high_good = _is_high_good_prop(stat_internal)
    operator = ">=" if high_good else "<="
    if kind != "breakout":
        operator = "<=" if high_good else ">="
    return f"{stat_label} {operator} {target}"


def _park_factor_for_prop(team: str, stat_internal: str) -> float | None:
    resolved_team = resolve_team(team) if team else None
    if not resolved_team:
        return None
    if stat_internal == "home_runs":
        return float(PARK_FACTORS_HR.get(resolved_team, 100))
    if stat_internal == "pitcher_strikeouts":
        return float(PARK_FACTORS_K.get(resolved_team, 100))
    return float(PARK_FACTORS.get(resolved_team, 100))


def build_tail_reason_lists(prediction: dict, max_items: int = 3) -> dict[str, list[str]]:
    """Return top breakout/dud drivers from already-computed prediction metadata."""
    stat_internal = str(prediction.get("stat_internal", "")).strip()
    stat_label = prediction.get("stat_type") or stat_internal.replace("_", " ").title()
    high_good = _is_high_good_prop(stat_internal)
    breakout_label = tail_signal_labels(stat_internal)["breakout"].lower()
    dud_label = tail_signal_labels(stat_internal)["dud"].lower()

    breakout_reasons: list[tuple[float, str]] = []
    dud_reasons: list[tuple[float, str]] = []

    def add_reason(bucket: list[tuple[float, str]], score: float, text: str) -> None:
        if score > 0 and text:
            bucket.append((score, text))

    projection = _safe_num(prediction.get("projection"), 0.0)
    line = _safe_num(prediction.get("line"), 0.0)
    diff = projection - line
    if line > 0 and abs(diff) > 0:
        gap_score = abs(diff) / max(line, 0.5)
        if (high_good and diff > 0) or ((not high_good) and diff < 0):
            add_reason(
                breakout_reasons,
                gap_score * 2.0,
                f"Projection sits {abs(diff):.2f} {'above' if diff > 0 else 'below'} the line.",
            )
        else:
            add_reason(
                dud_reasons,
                gap_score * 2.0,
                f"Projection drifts {abs(diff):.2f} to the wrong side of the line.",
            )

    weather_mult = _safe_num(prediction.get("weather_mult"), 1.0)
    if abs(weather_mult - 1.0) >= 0.03:
        favorable = (high_good and weather_mult > 1.0) or ((not high_good) and weather_mult < 1.0)
        favorable_text = (
            f"Weather is boosting the {stat_label.lower()} environment."
            if high_good
            else f"Weather is suppressing the {stat_label.lower()} environment."
        )
        unfavorable_text = (
            f"Weather is suppressing the {stat_label.lower()} environment."
            if high_good
            else f"Weather is boosting the {stat_label.lower()} environment."
        )
        add_reason(
            breakout_reasons if favorable else dud_reasons,
            abs(weather_mult - 1.0),
            favorable_text if favorable else unfavorable_text,
        )

    park_factor = _park_factor_for_prop(prediction.get("team", ""), stat_internal)
    if park_factor is not None and abs(park_factor - 100) >= 3:
        favorable = (high_good and park_factor > 100) or ((not high_good) and park_factor < 100)
        add_reason(
            breakout_reasons if favorable else dud_reasons,
            abs(park_factor - 100) / 100.0,
            (
                f"Park context leans {breakout_label} ({park_factor:.0f} factor)."
                if favorable
                else f"Park context raises {dud_label} ({park_factor:.0f} factor)."
            ),
        )

    pa_mult = _safe_num(prediction.get("pa_multiplier"), 1.0)
    batting_order = prediction.get("batting_order")
    if abs(pa_mult - 1.0) >= 0.03 and batting_order:
        favorable = (high_good and pa_mult > 1.0) or ((not high_good) and pa_mult < 1.0)
        add_reason(
            breakout_reasons if favorable else dud_reasons,
            abs(pa_mult - 1.0),
            (
                f"Lineup spot adds opportunity from the #{batting_order} slot."
                if favorable
                else f"Lineup spot trims opportunity from the #{batting_order} slot."
            ),
        )

    platoon_favorable = prediction.get("platoon_favorable")
    if platoon_favorable is not None:
        favorable = bool(platoon_favorable)
        add_reason(
            breakout_reasons if favorable else dud_reasons,
            0.08,
            prediction.get("platoon") or ("Favorable platoon setup." if favorable else "Unfavorable platoon setup."),
        )

    spring_badge = prediction.get("spring_badge")
    if spring_badge == "hot":
        add_reason(breakout_reasons, 0.06, "Spring form is running hot.")
    elif spring_badge == "cold":
        add_reason(dud_reasons, 0.06, "Spring form is running cold.")

    trend_badge = prediction.get("trend_badge")
    if trend_badge == "hot":
        add_reason(breakout_reasons, 0.05, "Recent trend is pointing up.")
    elif trend_badge == "cold":
        add_reason(dud_reasons, 0.05, "Recent trend is pointing down.")

    opp_lineup_k_rate = _safe_num(prediction.get("opp_lineup_k_rate"), 0.0)
    if stat_internal == "pitcher_strikeouts" and opp_lineup_k_rate > 0:
        k_gap = abs(opp_lineup_k_rate - 22.7)
        if k_gap >= 1.2:
            favorable = opp_lineup_k_rate > 22.7
            add_reason(
                breakout_reasons if favorable else dud_reasons,
                k_gap / 10.0,
                (
                    f"Confirmed lineup whiffs more than average ({opp_lineup_k_rate:.1f}% K rate)."
                    if favorable
                    else f"Confirmed lineup makes more contact than average ({opp_lineup_k_rate:.1f}% K rate)."
                ),
            )

    ahead_obp = _safe_num(prediction.get("ahead_obp"), 0.0)
    ahead_woba = _safe_num(prediction.get("ahead_woba"), 0.0)
    if stat_internal in RBI_CONTEXT_PROPS and (ahead_obp > 0 or ahead_woba > 0):
        support_score = 0.0
        if ahead_obp > 0:
            support_score += abs(ahead_obp - 0.320) * 6.5
        if ahead_woba > 0:
            support_score += abs(ahead_woba - 0.315) * 5.0
        if support_score >= 0.08:
            favorable = (ahead_obp and ahead_obp >= 0.320) or (ahead_woba and ahead_woba >= 0.315)
            add_reason(
                breakout_reasons if favorable else dud_reasons,
                support_score,
                (
                    "Hitters ahead of him are creating RBI chances."
                    if favorable
                    else "Top-of-order support is weak for RBI chances."
                ),
            )

    behind_woba = _safe_num(prediction.get("behind_woba"), 0.0)
    behind_slg = _safe_num(prediction.get("behind_slg"), 0.0)
    if stat_internal in RUN_CONTEXT_PROPS and (behind_woba > 0 or behind_slg > 0):
        run_support_score = 0.0
        if behind_woba > 0:
            run_support_score += abs(behind_woba - 0.315) * 4.5
        if behind_slg > 0:
            run_support_score += abs(behind_slg - 0.400) * 3.5
        if run_support_score >= 0.08:
            favorable = (behind_woba and behind_woba >= 0.315) or (behind_slg and behind_slg >= 0.400)
            add_reason(
                breakout_reasons if favorable else dud_reasons,
                run_support_score,
                (
                    "Hitters behind him can cash in run-scoring chances."
                    if favorable
                    else "Lineup support behind him is thin."
                ),
            )

    team_avg_woba = _safe_num(prediction.get("team_avg_woba"), 0.0)
    lineup_depth_woba = _safe_num(prediction.get("lineup_depth_woba"), 0.0)
    if stat_internal in RBI_CONTEXT_PROPS | RUN_CONTEXT_PROPS and (team_avg_woba > 0 or lineup_depth_woba > 0):
        quality = max(team_avg_woba, lineup_depth_woba)
        if abs(quality - 0.315) >= 0.010:
            favorable = quality > 0.315
            add_reason(
                breakout_reasons if favorable else dud_reasons,
                abs(quality - 0.315) * 3.5,
                (
                    "Confirmed lineup depth supports a higher run environment."
                    if favorable
                    else "Confirmed lineup depth drags on the run environment."
                ),
            )

    breakout_reasons.sort(key=lambda item: item[0], reverse=True)
    dud_reasons.sort(key=lambda item: item[0], reverse=True)
    return {
        "breakout": [text for _, text in breakout_reasons[:max_items]],
        "dud": [text for _, text in dud_reasons[:max_items]],
    }
