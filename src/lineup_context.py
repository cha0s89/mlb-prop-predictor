"""Lineup-derived context features for hitter and pitcher projections."""

from __future__ import annotations

import unicodedata
from typing import Any

import pandas as pd

from src.team_context import normalize_team_code


LEAGUE_DEFAULTS = {
    "k_rate": 22.7,
    "obp": 0.320,
    "woba": 0.315,
    "slg": 0.400,
    "iso": 0.160,
    "bb_rate": 8.5,
    "sprint_speed": 27.0,
}


def _normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"):
        text = text.replace(suffix, "")
    return " ".join(text.split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _match_batter_row(player_name: str, batting_df: pd.DataFrame):
    if batting_df.empty or "Name" not in batting_df.columns:
        return None

    target = _normalize_name(player_name)
    if not target:
        return None

    for _, row in batting_df.iterrows():
        if _normalize_name(row.get("Name", "")) == target:
            return row

    parts = target.split()
    if len(parts) >= 2:
        last = parts[-1]
        first_init = parts[0][0]
        for _, row in batting_df.iterrows():
            row_name = _normalize_name(row.get("Name", ""))
            row_parts = row_name.split()
            if len(row_parts) >= 2 and row_parts[-1] == last and row_parts[0][:1] == first_init:
                return row

    if parts:
        last = parts[-1]
        matches = []
        for _, row in batting_df.iterrows():
            row_name = _normalize_name(row.get("Name", ""))
            row_parts = row_name.split()
            if row_parts and row_parts[-1] == last:
                matches.append(row)
        if len(matches) == 1:
            return matches[0]

    return None


def _extract_metrics(row) -> dict[str, float]:
    return {
        "obp": _safe_float(row.get("OBP"), LEAGUE_DEFAULTS["obp"]),
        "woba": _safe_float(row.get("wOBA"), LEAGUE_DEFAULTS["woba"]),
        "slg": _safe_float(row.get("SLG"), LEAGUE_DEFAULTS["slg"]),
        "iso": _safe_float(row.get("ISO"), LEAGUE_DEFAULTS["iso"]),
        "k_rate": _safe_float(row.get("K%"), LEAGUE_DEFAULTS["k_rate"]),
        "bb_rate": _safe_float(row.get("BB%"), LEAGUE_DEFAULTS["bb_rate"]),
        "sprint_speed": _safe_float(row.get("Spd"), LEAGUE_DEFAULTS["sprint_speed"]),
    }


def _mean_or_default(values: list[float], default_key: str) -> float:
    if not values:
        return LEAGUE_DEFAULTS[default_key]
    return float(sum(values) / len(values))


def build_team_lineup_context(
    lineup: list[dict],
    batting_df: pd.DataFrame,
    team_abbr: str | None = None,
) -> dict:
    """Return aggregate lineup context from a confirmed batting order."""
    if not lineup:
        return {"has_data": False, "lineup_size": 0, "matched_count": 0}

    filtered_batting_df = batting_df
    if team_abbr and not batting_df.empty and "Team" in batting_df.columns:
        team_code = normalize_team_code(team_abbr)
        if team_code:
            team_mask = batting_df["Team"].apply(normalize_team_code) == team_code
            if bool(team_mask.any()):
                filtered_batting_df = batting_df[team_mask]

    players: list[dict[str, Any]] = []
    for slot in sorted(lineup, key=lambda item: item.get("batting_order", 99)):
        row = _match_batter_row(slot.get("player_name", ""), filtered_batting_df)
        if row is None:
            continue
        metrics = _extract_metrics(row)
        players.append(
            {
                "player_name": slot.get("player_name", ""),
                "batting_order": int(slot.get("batting_order", len(players) + 1) or len(players) + 1),
                **metrics,
            }
        )

    if not players:
        return {
            "has_data": False,
            "lineup_size": len(lineup),
            "matched_count": 0,
            "avg_k_rate": LEAGUE_DEFAULTS["k_rate"],
            "avg_obp": LEAGUE_DEFAULTS["obp"],
            "avg_woba": LEAGUE_DEFAULTS["woba"],
            "avg_slg": LEAGUE_DEFAULTS["slg"],
            "avg_iso": LEAGUE_DEFAULTS["iso"],
            "top6_k_rate": LEAGUE_DEFAULTS["k_rate"],
            "top5_woba": LEAGUE_DEFAULTS["woba"],
            "bottom3_k_rate": LEAGUE_DEFAULTS["k_rate"],
            "lineup_depth_woba": LEAGUE_DEFAULTS["woba"],
            "players": [],
        }

    ordered = sorted(players, key=lambda item: item["batting_order"])

    avg_k_rate = _mean_or_default([p["k_rate"] for p in ordered], "k_rate")
    avg_obp = _mean_or_default([p["obp"] for p in ordered], "obp")
    avg_woba = _mean_or_default([p["woba"] for p in ordered], "woba")
    avg_slg = _mean_or_default([p["slg"] for p in ordered], "slg")
    avg_iso = _mean_or_default([p["iso"] for p in ordered], "iso")

    top = ordered[: min(6, len(ordered))]
    bottom = ordered[-min(3, len(ordered)) :]
    middle = ordered[2:7] if len(ordered) >= 5 else ordered

    return {
        "has_data": True,
        "lineup_size": len(lineup),
        "matched_count": len(ordered),
        "avg_k_rate": round(avg_k_rate, 2),
        "avg_obp": round(avg_obp, 3),
        "avg_woba": round(avg_woba, 3),
        "avg_slg": round(avg_slg, 3),
        "avg_iso": round(avg_iso, 3),
        "top6_k_rate": round(_mean_or_default([p["k_rate"] for p in top], "k_rate"), 2),
        "top5_woba": round(_mean_or_default([p["woba"] for p in top[:5]], "woba"), 3),
        "bottom3_k_rate": round(_mean_or_default([p["k_rate"] for p in bottom], "k_rate"), 2),
        "lineup_depth_woba": round(_mean_or_default([p["woba"] for p in middle], "woba"), 3),
        "players": ordered,
    }


def build_player_lineup_context(player_name: str, team_context: dict) -> dict:
    """Return batter-specific lineup support metrics from team lineup context."""
    players = list(team_context.get("players") or [])
    if not players:
        return {
            "has_data": False,
            "team_avg_woba": team_context.get("avg_woba", LEAGUE_DEFAULTS["woba"]),
            "team_avg_obp": team_context.get("avg_obp", LEAGUE_DEFAULTS["obp"]),
            "team_avg_k_rate": team_context.get("avg_k_rate", LEAGUE_DEFAULTS["k_rate"]),
        }

    target = _normalize_name(player_name)
    player_idx = None
    for idx, item in enumerate(players):
        if _normalize_name(item.get("player_name", "")) == target:
            player_idx = idx
            break
    if player_idx is None:
        return {
            "has_data": False,
            "team_avg_woba": team_context.get("avg_woba", LEAGUE_DEFAULTS["woba"]),
            "team_avg_obp": team_context.get("avg_obp", LEAGUE_DEFAULTS["obp"]),
            "team_avg_k_rate": team_context.get("avg_k_rate", LEAGUE_DEFAULTS["k_rate"]),
        }

    ahead = players[:player_idx]
    behind = players[player_idx + 1 :]
    immediate_ahead = ahead[-3:]
    immediate_behind = behind[:3]

    return {
        "has_data": True,
        "batting_order": players[player_idx]["batting_order"],
        "team_avg_woba": team_context.get("avg_woba", LEAGUE_DEFAULTS["woba"]),
        "team_avg_obp": team_context.get("avg_obp", LEAGUE_DEFAULTS["obp"]),
        "team_avg_k_rate": team_context.get("avg_k_rate", LEAGUE_DEFAULTS["k_rate"]),
        "ahead_obp": round(_mean_or_default([p["obp"] for p in immediate_ahead], "obp"), 3),
        "ahead_woba": round(_mean_or_default([p["woba"] for p in immediate_ahead], "woba"), 3),
        "ahead_bb_rate": round(_mean_or_default([p["bb_rate"] for p in immediate_ahead], "bb_rate"), 2),
        "behind_slg": round(_mean_or_default([p["slg"] for p in immediate_behind], "slg"), 3),
        "behind_woba": round(_mean_or_default([p["woba"] for p in immediate_behind], "woba"), 3),
        "behind_iso": round(_mean_or_default([p["iso"] for p in immediate_behind], "iso"), 3),
        "behind_k_rate": round(_mean_or_default([p["k_rate"] for p in immediate_behind], "k_rate"), 2),
        "lineup_depth_woba": team_context.get("lineup_depth_woba", LEAGUE_DEFAULTS["woba"]),
    }
