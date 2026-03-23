"""
Headless Board Pipeline — extracts the full board-build logic from app.py
so both the Streamlit UI and scheduled automation produce identical predictions.

Usage:
    from src.headless_board import build_board
    result = build_board()
    # result["predictions"], result["edges"], result["shadow_samples"], ...

This module is 100% free of Streamlit imports.  It replaces st.spinner/st.progress
with Python logging, and st.cache_data with plain function calls (caching is
unnecessary for one-shot CLI runs).
"""

from __future__ import annotations

import json
import logging
import math
import os
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────────
from src.prizepicks import fetch_prizepicks_mlb_lines
from src.sharp_odds import (
    fetch_mlb_events,
    fetch_event_props,
    extract_sharp_lines,
    find_ev_edges,
    get_api_key,
    has_cached_odds_today,
)
from src.weather import (
    fetch_game_weather,
    resolve_team,
    STADIUMS,
    get_stat_specific_weather_adjustment,
)
from src.umpires import get_umpire_k_adjustment, fetch_todays_umpires
from src.predictor import (
    generate_prediction,
    calculate_over_under_probability,
    PARK_FACTORS,
    PARK_FACTORS_HR,
    PARK_FACTORS_K,
)
from src.database import (
    init_db,
    log_batch_predictions,
    save_projected_stats,
    init_projected_stats_table,
)
from src.stats import fetch_batting_leaders, fetch_pitching_leaders
from src.spring import (
    get_player_injury_status,
    get_spring_form_multiplier,
    fetch_spring_training_stats,
    fetch_injuries,
)
from src.trends import get_batter_trend
from src.lineups import (
    fetch_todays_games,
    get_batting_order_position,
    get_pa_multiplier,
    get_game_context,
    get_probable_pitcher,
    fetch_confirmed_lineups,
)
from src.matchups import get_platoon_split_adjustment
from src.board_logger import log_board_snapshot, ensure_shadow_sample
from src.line_snapshots import snapshot_pp_lines
from src.consistency import enforce_consistency
from src.selection import annotate_prediction_floor, get_confidence_floor
from src.autolearn import load_current_weights
from src.combined import score_picks
from src.team_context import (
    extract_schedule_dates,
    normalize_team_code,
    pitcher_row_matches_team,
    register_team_value,
)

logger = logging.getLogger(__name__)

# ── Timezone helpers ─────────────────────────────────────────────────────────
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = ZoneInfo("UTC")

# ── Constants (mirrored from app.py so predictions are identical) ────────────
PITCHER_STAT_INTERNALS = {
    "pitcher_strikeouts",
    "hits_allowed",
    "earned_runs",
    "walks_allowed",
    "pitching_outs",
}

MIN_REALISTIC_LINE: dict = {
    "hits_runs_rbis": 1.0,
    "hitter_fantasy_score": 3.0,
    "total_bases": 0.5,
    "pitcher_strikeouts": 2.5,
    "pitching_outs": 10.5,
    "earned_runs": 0.5,
    "walks_allowed": 0.5,
    "hits_allowed": 1.5,
}

MAX_EDGE_PCT = 0.35

COUNT_PROPS = {
    "hits", "total_bases", "rbis", "runs", "stolen_bases",
    "hits_runs_rbis", "batter_strikeouts", "walks", "singles", "doubles",
    "pitcher_strikeouts", "pitching_outs", "earned_runs",
    "walks_allowed", "hits_allowed", "hitter_fantasy_score", "home_runs",
}

OPENING_DAY = date(2026, 3, 27)

# ── Helpers (pure-Python, no Streamlit) ──────────────────────────────────────


def _safe_num(val, fallback=0.0):
    if val is None:
        return fallback
    try:
        f = float(val)
        return fallback if math.isnan(f) else f
    except (ValueError, TypeError):
        return fallback


def _confidence_rating(confidence: float) -> str:
    if confidence >= 0.66:
        return "A"
    if confidence >= 0.60:
        return "B"
    if confidence >= 0.55:
        return "C"
    return "D"


def _sync_pick_metrics(pred: dict) -> None:
    """Keep confidence-derived fields aligned after app-side adjustments."""
    conf = max(0.0, min(_safe_num(pred.get("confidence"), 0.5), 1.0))
    push_p = max(0.0, min(_safe_num(pred.get("p_push"), 0.0), 1.0))
    pred["confidence"] = conf
    pred["edge"] = round(abs(conf - 0.50), 4)
    pred["win_prob"] = round(conf * max(0.0, 1.0 - push_p), 4)
    pred["rating"] = _confidence_rating(conf)


def _game_date_from_iso(iso_str: str) -> str:
    """Resolve a game date from an ISO timestamp using Pacific calendar days."""
    if not iso_str:
        return date.today().isoformat()
    try:
        dt = pd.Timestamp(iso_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize(UTC_TZ)
        return dt.tz_convert(PACIFIC_TZ).date().isoformat()
    except Exception:
        return date.today().isoformat()


def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]:
        name = name.replace(suffix, "")
    return name.strip()


# ── Stats loading (no Streamlit cache) ───────────────────────────────────────


def load_batting_stats() -> pd.DataFrame:
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "batting_stats_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) >= 50:
            return df
    year = datetime.now().year
    df = fetch_batting_leaders(year, min_pa=100)
    if df.empty or len(df) < 50:
        df = fetch_batting_leaders(year - 1, min_pa=100)
    if df.empty or len(df) < 50:
        df = fetch_batting_leaders(year - 2, min_pa=100)
    return df


def load_pitching_stats() -> pd.DataFrame:
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "pitching_stats_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) >= 20:
            return df
    year = datetime.now().year
    df = fetch_pitching_leaders(year, min_ip=10)
    if df.empty or len(df) < 20:
        df = fetch_pitching_leaders(year - 1, min_ip=10)
    if not df.empty:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception:
            pass
    return df


def match_player_stats(player_name: str, batting_df: pd.DataFrame):
    if batting_df.empty or "Name" not in batting_df.columns:
        return None
    norm_target = _normalize_name(player_name)
    for _, row in batting_df.iterrows():
        if _normalize_name(str(row.get("Name", ""))) == norm_target:
            return row
    parts = norm_target.split()
    if len(parts) >= 2:
        last = parts[-1]
        first_init = parts[0][0] if parts[0] else ""
        for _, row in batting_df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            rparts = rn.split()
            if len(rparts) >= 2 and rparts[-1] == last and rparts[0] and rparts[0][0] == first_init:
                return row
    if parts:
        last = parts[-1]
        matches = []
        for _, row in batting_df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            if rn.split()[-1] == last if rn.split() else False:
                matches.append(row)
        if len(matches) == 1:
            return matches[0]
    return None


def match_pitcher_stats(player_name: str, pitching_df: pd.DataFrame):
    if pitching_df.empty or "Name" not in pitching_df.columns:
        return None
    norm_target = _normalize_name(player_name)
    for _, row in pitching_df.iterrows():
        if _normalize_name(str(row.get("Name", ""))) == norm_target:
            return row
    parts = norm_target.split()
    if len(parts) >= 2:
        last = parts[-1]
        first_init = parts[0][0] if parts[0] else ""
        for _, row in pitching_df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            rparts = rn.split()
            if len(rparts) >= 2 and rparts[-1] == last and rparts[0] and rparts[0][0] == first_init:
                return row
    return None


def build_batter_profile(stats_row: pd.Series) -> dict:
    def safe_pct(val):
        if isinstance(val, str):
            return float(val.replace("%", "").strip())
        return float(val) if val else 0.0
    return {
        "pa": int(stats_row.get("PA", 0)),
        "avg": float(stats_row.get("AVG", 0)),
        "obp": float(stats_row.get("OBP", 0)),
        "slg": float(stats_row.get("SLG", 0)),
        "iso": float(stats_row.get("ISO", 0)),
        "woba": float(stats_row.get("wOBA", 0)),
        "bb_rate": safe_pct(stats_row.get("BB%", 0)),
        "k_rate": safe_pct(stats_row.get("K%", 0)),
        "hr": int(stats_row.get("HR", 0)),
        "sb": int(stats_row.get("SB", 0)),
        "rbi": int(stats_row.get("RBI", 0)),
        "r": int(stats_row.get("R", 0)),
        "2b": int(stats_row.get("2B", 0)),
        "3b": int(stats_row.get("3B", 0)),
        "babip": float(stats_row.get("BABIP", 0)),
        "sprint_speed": 27.0,
        "xba": float(stats_row.get("xBA", 0) or 0),
        "xslg": float(stats_row.get("xSLG", 0) or 0),
        "xwoba": float(stats_row.get("xwOBA", 0) or 0),
        "barrel_rate": float(stats_row.get("Barrel%", 0) or 0),
        "recent_barrel_rate": float(stats_row.get("Barrel%", 0) or 0),
        "recent_hard_hit_pct": float(stats_row.get("HardHit%", 0) or 0),
    }


def build_pitcher_profile(stats_row) -> dict:
    def safe_pct(val):
        if isinstance(val, str):
            return float(val.replace("%", "").strip())
        return float(val) if val else 0.0
    return {
        "ip": float(stats_row.get("IP", 0)),
        "era": float(stats_row.get("ERA", 0)),
        "fip": float(stats_row.get("FIP", 0)),
        "k9": float(stats_row.get("K/9", 0)),
        "bb9": float(stats_row.get("BB/9", 0)),
        "whip": float(stats_row.get("WHIP", 0)),
        "k_pct": safe_pct(stats_row.get("K%", 0)),
        "bb_pct": safe_pct(stats_row.get("BB%", 0)),
        "hr9": float(stats_row.get("HR/9", 0)),
        "gs": int(float(stats_row.get("GS", stats_row.get("G", 1)))),
        "xfip": float(stats_row.get("xFIP", stats_row.get("FIP", 0))),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_board(
    *,
    skip_sharp: bool = False,
    dry_run: bool = False,
    api_key: str | None = None,
) -> dict:
    """Run the full board-build pipeline and return predictions + metadata.

    This mirrors the exact sequence of Steps 1-13 from app.py's projection
    analysis path, but without any Streamlit dependencies.

    Args:
        skip_sharp: If True, skip sharp-odds fetch entirely (saves API credits).
        dry_run: If True, generate predictions but don't persist anything.
        api_key: Odds API key override.  Falls back to get_api_key().

    Returns:
        dict with keys:
            predictions  — list of prediction dicts (same shape as app.py)
            edges        — list of sharp edge dicts (may be empty)
            shadow_samples — list of shadow sample result dicts
            stats        — summary counters
            errors       — list of non-fatal error strings
    """
    errors: List[str] = []
    result = {
        "predictions": [],
        "edges": [],
        "shadow_samples": [],
        "stats": {},
        "errors": errors,
    }

    # Ensure DB tables exist
    try:
        init_db()
        init_projected_stats_table()
    except Exception as exc:
        errors.append(f"DB init: {exc}")

    # ── STEP 1: Fetch PrizePicks lines ───────────────────────────────────────
    logger.info("Step 1/13: Fetching PrizePicks MLB lines")
    try:
        all_pp_lines = fetch_prizepicks_mlb_lines(include_all=True)
    except Exception as exc:
        errors.append(f"PP fetch: {exc}")
        all_pp_lines = pd.DataFrame()

    if not all_pp_lines.empty and "model_eligible" in all_pp_lines.columns:
        pp_lines = all_pp_lines[all_pp_lines["model_eligible"] == True].copy()
    else:
        pp_lines = all_pp_lines.copy()

    if pp_lines.empty:
        logger.warning("No MLB lines on PrizePicks — nothing to do.")
        result["stats"]["pp_lines"] = 0
        return result

    result["stats"]["pp_lines"] = len(pp_lines)
    logger.info("  %d model-eligible PP lines", len(pp_lines))

    # Snapshot PP lines for CLV tracking
    if not dry_run:
        try:
            snapshot_pp_lines(pp_lines)
        except Exception as exc:
            errors.append(f"Line snapshot: {exc}")

    # ── STEP 2: Load player stats ────────────────────────────────────────────
    logger.info("Step 2/13: Loading player stats")
    batting_df = pd.DataFrame()
    pitching_df = pd.DataFrame()
    try:
        batting_df = load_batting_stats()
    except Exception as exc:
        errors.append(f"Batting stats: {exc}")
    try:
        pitching_df = load_pitching_stats()
    except Exception as exc:
        errors.append(f"Pitching stats: {exc}")
    logger.info("  %d batters, %d pitchers loaded", len(batting_df), len(pitching_df))

    # ── STEP 3: Extract game dates + fetch schedule ──────────────────────────
    logger.info("Step 3/13: Resolving game dates and fetching schedule")
    pp_game_dates = set(extract_schedule_dates(
        pp_lines["start_time"].dropna() if "start_time" in pp_lines.columns else [],
        fallback_date=datetime.now().strftime("%Y-%m-%d"),
    ))

    todays_games: list = []
    try:
        todays_games = fetch_todays_games(game_dates=list(sorted(pp_game_dates)))
    except Exception as exc:
        errors.append(f"Schedule fetch: {exc}")
    logger.info("  %d games for dates %s", len(todays_games), sorted(pp_game_dates))

    # Detect preseason
    _days_to_opening = (OPENING_DAY - date.today()).days
    _is_preseason = _days_to_opening > 0

    # ── STEP 4: Weather cache ────────────────────────────────────────────────
    logger.info("Step 4/13: Fetching weather")
    teams_in_slate: set = set()
    for _, row in pp_lines.iterrows():
        t = row.get("team", "")
        if t:
            r = resolve_team(t)
            if r and r in STADIUMS:
                teams_in_slate.add(r)

    weather_cache: dict = {}
    for team_abbr in sorted(teams_in_slate):
        try:
            weather_cache[team_abbr] = fetch_game_weather(team_abbr)
        except Exception:
            weather_cache[team_abbr] = None
    logger.info("  Weather for %d stadiums", len(weather_cache))

    # ── STEP 5: Spring training stats + injuries ─────────────────────────────
    logger.info("Step 5/13: Loading spring training data & injuries")
    st_stats: list = []
    injury_list: list = []
    try:
        st_stats = fetch_spring_training_stats()
    except Exception:
        pass
    try:
        injury_list = fetch_injuries(days_back=60)
    except Exception:
        pass

    # ── STEP 6: Umpires ──────────────────────────────────────────────────────
    logger.info("Step 6/13: Fetching umpire assignments")
    umpire_map: dict = {}
    try:
        umpire_map = fetch_todays_umpires()
    except Exception:
        pass

    # ── STEP 7: Build opposing pitcher + team lookups ────────────────────────
    logger.info("Step 7/13: Building opposing pitcher lookups")
    opp_pitcher_lookup: dict = {}
    team_pitcher_lookup: dict = {}
    opp_team_k_lookup: dict = {}
    team_k_rate_map: dict = {}
    if not batting_df.empty and "Team" in batting_df.columns and "K%" in batting_df.columns:
        try:
            team_rates = batting_df[["Team", "K%"]].copy()
            team_rates["_team_code"] = team_rates["Team"].apply(normalize_team_code)
            team_rates = team_rates[team_rates["_team_code"] != ""]
            for team_code, subset in team_rates.groupby("_team_code"):
                k_vals = subset["K%"].apply(
                    lambda x: float(str(x).replace("%", "").strip()) if pd.notna(x) else 22.7
                )
                team_k_rate_map[team_code] = round(k_vals.mean(), 1)
        except Exception:
            team_k_rate_map = {}
    try:
        for game in todays_games:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            home_pitcher_name = game.get("home_pitcher_name", "")
            away_pitcher_name = game.get("away_pitcher_name", "")
            home_matched = None
            away_matched = None
            if home_team and home_pitcher_name and home_pitcher_name != "TBD" and not pitching_df.empty:
                home_matched = match_pitcher_stats(home_pitcher_name, pitching_df)
            if away_team and away_pitcher_name and away_pitcher_name != "TBD" and not pitching_df.empty:
                away_matched = match_pitcher_stats(away_pitcher_name, pitching_df)
            home_pitcher_valid = bool(home_pitcher_name and home_pitcher_name != "TBD")
            away_pitcher_valid = bool(away_pitcher_name and away_pitcher_name != "TBD")
            if home_pitcher_valid and not pitcher_row_matches_team(home_matched, home_team):
                home_pitcher_valid = False
            if away_pitcher_valid and not pitcher_row_matches_team(away_matched, away_team):
                away_pitcher_valid = False
            if home_team:
                register_team_value(
                    team_pitcher_lookup,
                    home_team,
                    home_pitcher_name if home_pitcher_valid else "",
                )
            if away_team:
                register_team_value(
                    team_pitcher_lookup,
                    away_team,
                    away_pitcher_name if away_pitcher_valid else "",
                )
            # Home batters face AWAY pitcher
            if away_team:
                opp_info = {
                    "name": away_pitcher_name if away_pitcher_valid else "",
                    "hand": game.get("away_pitcher_hand", ""),
                    "id": game.get("away_pitcher_id"),
                }
                if away_pitcher_valid and away_matched is not None:
                    opp_info["profile"] = build_pitcher_profile(away_matched)
                register_team_value(opp_pitcher_lookup, home_team, opp_info)
            # Away batters face HOME pitcher
            if home_team:
                opp_info = {
                    "name": home_pitcher_name if home_pitcher_valid else "",
                    "hand": game.get("home_pitcher_hand", ""),
                    "id": game.get("home_pitcher_id"),
                }
                if home_pitcher_valid and home_matched is not None:
                    opp_info["profile"] = build_pitcher_profile(home_matched)
                register_team_value(opp_pitcher_lookup, away_team, opp_info)
            # Opposing team K rate for pitcher K projections
            for pitcher_team, opp_batting_team in [
                (home_team, away_team),
                (away_team, home_team),
            ]:
                norm_opp_team = normalize_team_code(opp_batting_team)
                if pitcher_team and norm_opp_team in team_k_rate_map:
                    register_team_value(opp_team_k_lookup, pitcher_team, team_k_rate_map[norm_opp_team])
    except Exception as exc:
        errors.append(f"Pitcher lookup build: {exc}")

    # ── STEP 8: Build lineup / game-context caches ───────────────────────────
    logger.info("Step 8/13: Building lineup and game-context caches")
    batter_hand_cache: dict = {}
    batting_order_cache: dict = {}
    game_context_cache: dict = {}
    try:
        for game in todays_games:
            game_pk = game.get("game_pk")
            if not game_pk:
                continue
            lineups = fetch_confirmed_lineups(game_pk)
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            for side, team_abbr, opp_abbr in [
                ("home", home_team, away_team),
                ("away", away_team, home_team),
            ]:
                for batter in lineups.get(side, []):
                    name = batter.get("player_name", "").upper().strip()
                    hand = batter.get("bat_hand", "")
                    order = batter.get("batting_order")
                    if name and hand:
                        batter_hand_cache[name] = hand
                    if name and order:
                        batting_order_cache[name] = order
            for team_abbr, opp_abbr, is_home in [
                (home_team, away_team, True),
                (away_team, home_team, False),
            ]:
                if team_abbr:
                    register_team_value(game_context_cache, team_abbr, {
                        "opponent": opp_abbr,
                        "game_time": game.get("game_time", ""),
                        "game_pk": game_pk,
                        "is_home": is_home,
                    })
    except Exception as exc:
        errors.append(f"Lineup cache build: {exc}")

    # ── STEP 9 (optional): Sharp edges ───────────────────────────────────────
    all_edges: list = []
    if not skip_sharp and not _is_preseason:
        logger.info("Step 9/13: Fetching sharp lines & devigging")
        if api_key is None:
            api_key = get_api_key()
        if api_key:
            try:
                events = fetch_mlb_events(api_key)
                pp_teams = set()
                if not pp_lines.empty and "team" in pp_lines.columns:
                    pp_teams = set(pp_lines["team"].dropna().str.lower().unique())
                for event in (events or [])[:15]:
                    eid = event.get("id", "")
                    if not eid:
                        continue
                    if pp_teams:
                        _home = event.get("home_team", "").lower()
                        _away = event.get("away_team", "").lower()
                        if not any(
                            t in _home or t in _away or _home in t or _away in t
                            for t in pp_teams
                        ):
                            continue
                    resp = fetch_event_props(eid, api_key=api_key)
                    if resp and "data" in resp:
                        sharp = extract_sharp_lines(resp["data"])
                        if sharp:
                            all_edges.extend(
                                find_ev_edges(pp_lines, sharp, min_ev_pct=0.25)
                            )
                if all_edges:
                    all_edges.sort(key=lambda x: x["edge_pct"], reverse=True)
                logger.info("  %d sharp edges found", len(all_edges))
            except Exception as exc:
                errors.append(f"Sharp odds: {exc}")
    else:
        logger.info("Step 9/13: Skipping sharp lines (preseason or skip_sharp)")

    result["edges"] = all_edges

    # ── STEP 10: Main prediction loop ────────────────────────────────────────
    logger.info("Step 10/13: Running predictions for %d props", len(pp_lines))
    preds: list = []
    pred_errors = 0
    active_weights = load_current_weights()

    for i, (_, row) in enumerate(pp_lines.iterrows()):
        try:
            team = row.get("team", "")
            stat_int = row.get("stat_internal", "")
            wx = None
            if team:
                r = resolve_team(team)
                if r in weather_cache:
                    wx = weather_cache[r]
            else:
                r = None

            r_team = resolve_team(team) if team else None

            batter_profile = None
            pitcher_profile = None
            matched = None
            is_pitcher_prop = stat_int in PITCHER_STAT_INTERNALS
            if is_pitcher_prop and not pitching_df.empty:
                matched = match_pitcher_stats(row["player_name"], pitching_df)
                if matched is not None:
                    pitcher_profile = build_pitcher_profile(matched)
            elif not is_pitcher_prop and not batting_df.empty:
                matched = match_player_stats(row["player_name"], batting_df)
                if matched is not None:
                    batter_profile = build_batter_profile(matched)

            # Umpire
            ump_name = None
            if r_team and umpire_map:
                ump_name = umpire_map.get(r_team)
            ump_adj = get_umpire_k_adjustment(ump_name) if ump_name else None

            # Batting order
            batting_pos = None
            if not is_pitcher_prop:
                batting_pos = batting_order_cache.get(
                    row["player_name"].upper().strip()
                )

            # Opposing pitcher profile
            opp_pitcher_profile = None
            platoon_adj = None
            if not is_pitcher_prop and r_team and r_team in opp_pitcher_lookup:
                opp_info = opp_pitcher_lookup[r_team]
                opp_pitcher_profile = opp_info.get("profile")
                opp_hand = opp_info.get("hand", "")
                if opp_hand:
                    bat_hand = batter_hand_cache.get(
                        row["player_name"].upper().strip(), ""
                    )
                    if bat_hand:
                        platoon_adj = get_platoon_split_adjustment(bat_hand, opp_hand)

            # Opposing team K rate
            opp_k_rate = None
            if is_pitcher_prop and r_team:
                opp_k_rate = opp_team_k_lookup.get(r_team)

            # Line sanity check
            _min_line = MIN_REALISTIC_LINE.get(stat_int, 0)
            if _min_line and float(row.get("line", 0)) < _min_line:
                continue

            p = generate_prediction(
                player_name=row["player_name"],
                stat_type=row["stat_type"],
                stat_internal=stat_int,
                line=row["line"],
                batter_profile=batter_profile,
                pitcher_profile=pitcher_profile,
                opp_pitcher_profile=opp_pitcher_profile,
                opp_team_k_rate=opp_k_rate,
                platoon=platoon_adj,
                park_team=r_team,
                weather=wx,
                ump=ump_adj,
                lineup_pos=batting_pos,
            )
            p["game_date"] = _game_date_from_iso(row.get("start_time", ""))
            p["game_time_utc"] = row.get("start_time", "")

            _is_count_prop = stat_int in COUNT_PROPS

            # Stat-specific weather display (NOT applied — already inside predictor)
            if wx:
                wx_mult = get_stat_specific_weather_adjustment(wx, stat_int)
                p["weather_mult"] = wx_mult

            # Spring form multiplier
            prior_slg = 0.400
            prior_avg = 0.250
            if matched is not None:
                if "SLG" in matched.index:
                    prior_slg = float(matched["SLG"])
                if "AVG" in matched.index:
                    prior_avg = float(matched["AVG"])
            spring = get_spring_form_multiplier(
                player_name=row["player_name"],
                prior_season_slg=prior_slg,
                prior_season_avg=prior_avg,
                st_stats=st_stats,
            )
            spring_mult = spring["spring_mult"]
            if _is_count_prop:
                p["projection"] = round(p["projection"] * spring_mult, 2)
            p["spring_mult"] = spring_mult
            p["spring_badge"] = spring["badge"]

            # Trend multiplier
            trend = get_batter_trend(0)
            trend_mult = trend.get("trend_multiplier", 1.0)
            trend_mult = max(0.92, min(1.08, trend_mult))
            if _is_count_prop:
                p["projection"] = round(p["projection"] * trend_mult, 2)
            p["trend_mult"] = trend_mult
            if trend_mult >= 1.03:
                p["trend_badge"] = "hot"
            elif trend_mult <= 0.97:
                p["trend_badge"] = "cold"
            else:
                p["trend_badge"] = "neutral"

            # Buy-low
            p["buy_low"] = False
            if not is_pitcher_prop and matched is not None:
                season_woba = float(matched.get("wOBA", 0)) if "wOBA" in matched.index else 0.0
                _obp = float(matched.get("OBP", 0)) if "OBP" in matched.index else 0.0
                _slg = float(matched.get("SLG", 0)) if "SLG" in matched.index else 0.0
                season_ops = float(matched.get("OPS", _obp + _slg)) if "OPS" in matched.index else (_obp + _slg)
                is_elite = season_woba > 0.350 or season_ops > 0.850
                is_cold = trend_mult < 0.97 or spring_mult < 0.97
                if is_elite and is_cold:
                    p["buy_low"] = True
                    if _is_count_prop:
                        p["projection"] = round(p["projection"] * 1.04, 2)

            # Refresh probabilities after spring/trend/buy-low
            _refreshed_prob = calculate_over_under_probability(
                p["projection"],
                p["line"],
                stat_int,
                proj_result=p,
            )
            p.update(_refreshed_prob)

            # Injury check
            injury = get_player_injury_status(
                player_name=row["player_name"],
                injuries=injury_list,
            )
            p["injury_status"] = injury["status"]
            p["injury_color"] = injury["color"]
            if injury["status"] == "IL":
                continue
            if injury["status"] == "day-to-day":
                p["confidence"] = max(p.get("confidence", 0.5) * 0.85, 0.50)
                _sync_pick_metrics(p)

            # Data quality gate
            if not p.get("has_player_data", True):
                continue
            if is_pitcher_prop:
                if r_team is None:
                    continue
            else:
                _no_opp = not p.get("has_opp_data") and not (
                    r_team and r_team in opp_pitcher_lookup
                )
                _no_lineup = batting_pos is None
                _no_park = r_team is None
                if _no_opp and _no_lineup and _no_park:
                    continue

            # Batting order metadata
            if batting_pos:
                p["batting_order"] = batting_pos
                p["pa_multiplier"] = round(get_pa_multiplier(batting_pos), 3)

            # Game context
            if r_team and r_team in game_context_cache:
                gctx = game_context_cache[r_team]
                raw_game_time = gctx.get("game_time", "")
                p["opponent"] = gctx.get("opponent", "")
                p["game_time_utc"] = raw_game_time
                p["game_date"] = _game_date_from_iso(raw_game_time)
                p["game_pk"] = gctx.get("game_pk")
                p["is_home"] = gctx.get("is_home", False)
                p["team_pitcher"] = team_pitcher_lookup.get(r_team, "")
                if r_team in opp_pitcher_lookup:
                    p["opp_pitcher"] = opp_pitcher_lookup[r_team].get("name", "")

            if platoon_adj and platoon_adj.get("favorable") is not None:
                p["platoon"] = platoon_adj["description"]
                p["platoon_favorable"] = platoon_adj["favorable"]
            if ump_name:
                p["umpire"] = ump_name

            p["team"] = team
            p["stat_internal"] = stat_int
            p["line_type"] = row.get("line_type", "standard")
            p["odds_type"] = row.get("odds_type", "standard")

            # Edge cap
            _raw_edge = abs(_safe_num(p.get("edge"), 0))
            if _raw_edge > MAX_EDGE_PCT:
                p["confidence"] = min(p.get("confidence", 0.5), 0.65)
                _sync_pick_metrics(p)
                p["edge_capped"] = True

            annotate_prediction_floor(p, active_weights)

            preds.append(p)

        except Exception as exc:
            pred_errors += 1
            if pred_errors <= 5:
                logger.warning(
                    "Prediction error for %s/%s: %s",
                    row.get("player_name", "?"),
                    row.get("stat_type", "?"),
                    exc,
                )

    if pred_errors:
        logger.warning("  %d prop(s) skipped due to errors", pred_errors)
    logger.info("  %d predictions generated", len(preds))

    # ── STEP 11: Consistency enforcement ─────────────────────────────────────
    if preds:
        logger.info("Step 11/13: Enforcing cross-prop consistency")
        try:
            preds = enforce_consistency(preds)
        except Exception as exc:
            errors.append(f"Consistency: {exc}")

    # ── STEP 12: Persist (unless dry_run) ────────────────────────────────────
    if preds and not dry_run:
        logger.info("Step 12/13: Saving projected stats + board snapshot")
        try:
            stats_to_save = [
                {
                    "game_date": p.get("game_date", date.today().isoformat()),
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "stat_type": p.get("stat_internal", ""),
                    "projected_value": p["projection"],
                    "line": p.get("line", 0),
                    "pick": p.get("pick", ""),
                    "confidence": p.get("confidence", 0),
                    "rating": p.get("rating", ""),
                }
                for p in preds
            ]
            save_projected_stats(stats_to_save)
        except Exception as exc:
            errors.append(f"Save projected stats: {exc}")
        try:
            log_board_snapshot(preds, edges=all_edges)
        except Exception as exc:
            errors.append(f"Board snapshot: {exc}")
    elif dry_run:
        logger.info("Step 12/13: Dry run — skipping persistence")
    else:
        logger.info("Step 12/13: No predictions to persist")

    # ── STEP 13: Shadow sample ───────────────────────────────────────────────
    shadow_results: list = []
    if preds and not dry_run:
        logger.info("Step 13/13: Creating shadow samples")
        try:
            game_dates = sorted(
                {
                    p.get("game_date", date.today().isoformat())
                    for p in preds
                    if p.get("game_date")
                }
            )
            for gd in game_dates:
                shadow_results.append(ensure_shadow_sample(gd, sample_size=50))
        except Exception as exc:
            errors.append(f"Shadow sample: {exc}")
    else:
        logger.info("Step 13/13: Skipping shadow sample (dry_run or no preds)")

    result["predictions"] = preds
    result["shadow_samples"] = shadow_results
    result["stats"].update(
        {
            "predictions": len(preds),
            "edges": len(all_edges),
            "pred_errors": pred_errors,
            "games": len(todays_games),
            "stadiums_weather": len(weather_cache),
        }
    )

    logger.info(
        "Board build complete: %d predictions, %d edges, %d errors",
        len(preds),
        len(all_edges),
        len(errors),
    )
    return result
