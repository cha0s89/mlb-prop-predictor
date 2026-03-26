"""
⚾ MLB Prop Edge v2 — Market-Based Edge Finder
Core: Compare PrizePicks lines to devigged sharp sportsbook odds.
When sharp books disagree with PrizePicks → that's the edge.
Statcast data provides the "why" confirmation layer.
"""

import logging
import math
import streamlit as st
import pandas as pd

_log = logging.getLogger(__name__)
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.prizepicks import fetch_prizepicks_mlb_lines
from src.sharp_odds import (
    fetch_mlb_events, fetch_event_props, extract_sharp_lines,
    find_ev_edges, get_api_usage, get_api_key, PP_TO_ODDS_API,
    get_credits_remaining, clear_odds_cache, ODDS_CACHE_TTL_SECONDS,
    has_cached_odds_today, get_cache_age_minutes,
)
from src.weather import fetch_game_weather, resolve_team, STADIUMS, get_stat_specific_weather_adjustment
from src.umpires import get_umpire_k_adjustment, fetch_todays_umpires
from src.predictor import (
    generate_prediction, calculate_over_under_probability,
    PARK_FACTORS, PARK_FACTORS_HR, PARK_FACTORS_K,
)
from src.database import (
    init_db, log_prediction, log_batch_predictions,
    get_accuracy_stats, get_graded_predictions, get_ungraded_predictions,
    grade_prediction,
)
from src.stats import fetch_batting_leaders, fetch_pitching_leaders
from src.slips import (
    init_slips_table, create_slip, get_slips, get_slip_picks,
    get_slip_pnl, grade_slip_pick, finalize_slip, PAYOUTS, BREAKEVEN,
)
from src.autograder import auto_grade_date, auto_grade_yesterday
from src.autolearn import run_adjustment_cycle, load_current_weights
from src.nightly import run_nightly_cycle
from src.spring import (
    apply_seasonal_spring_blend,
    get_player_injury_status, get_spring_form_multiplier,
    fetch_spring_training_stats, fetch_injuries, fetch_recent_transactions,
    fetch_mlb_news,
)
from src.trends import get_batter_trend
from src.explain import build_explanation
from src.combined import score_picks, SIGNAL_CONFIRMED, SIGNAL_SHARP_ONLY, SIGNAL_PROJECTION_ONLY
from src.slip_warnings import analyze_slip_correlation
from src.lineups import (
    fetch_todays_games, get_batting_order_position, get_pa_multiplier,
    get_game_context, get_probable_pitcher, fetch_confirmed_lineups,
)
from src.matchups import get_platoon_split_adjustment, get_bvp_matchup, lookup_player_id
from src.database import (
    save_projected_stats, get_projection_accuracy,
    get_projection_history, init_projected_stats_table,
    get_calibration_data, get_projection_diagnostics,
)
from src.kelly import calculate_slip_sizing
from src.parlay_suggest import suggest_slips
from src.prediction_cleanup import dedupe_predictions
# drift module available for nightly cycle (not used in UI)
from src.slip_ev import simulate_slip_ev, quick_slip_ev, build_correlation_matrix
from src.board_logger import (
    log_board_snapshot,
    mark_as_bet,
    ensure_shadow_sample,
    get_board_stats,
    get_shadow_sample_stats,
)
from src.line_snapshots import snapshot_pp_lines
from src.consistency import enforce_consistency
from src.selection import annotate_prediction_floor, get_confidence_floor
from src.tail_signals import build_tail_reason_lists, tail_signal_labels, tail_target_text
from src.team_context import (
    extract_schedule_dates,
    normalize_team_code,
    pitcher_row_matches_team,
    get_team_game_value,
    register_team_game_value,
    register_team_value,
)
from src.lineup_context import build_player_lineup_context, build_team_lineup_context


# ─────────────────────────────────────────────
# NaN SANITIZATION — prevent raw NaN/None from leaking into UI
# ─────────────────────────────────────────────

def _safe(val, fallback="—"):
    """Return fallback if val is None, NaN, empty string, or the literal string 'nan'."""
    if val is None:
        return fallback
    if isinstance(val, float) and math.isnan(val):
        return fallback
    if isinstance(val, str) and val.strip().lower() in ("nan", "none", ""):
        return fallback
    return val


def _safe_num(val, fallback=0.0):
    """Return fallback if val is None or NaN, else float(val)."""
    if val is None:
        return fallback
    try:
        f = float(val)
        return fallback if math.isnan(f) else f
    except (ValueError, TypeError):
        return fallback


def _has_real_number(val) -> bool:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return False
    return not math.isnan(num)


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = ZoneInfo("UTC")
FRESHNESS_LOG_PATH = Path("data/freshness.json")
PLAYER_SEARCH_ALL = "All current players"
PLAYER_STATUS_LABELS = {
    "eligible": "Model eligible",
    "non_standard_line": "Excluded - promo/goblin/demon",
    "spring_training": "Excluded - spring training",
    "season_long": "Excluded - season-long/futures",
}


def _format_clock_pacific(dt: datetime) -> str:
    local_dt = dt.astimezone(PACIFIC_TZ)
    return f"{local_dt.strftime('%I:%M %p').lstrip('0')} {local_dt.tzname() or 'PT'}"


def _format_datetime_pacific(dt: datetime) -> str:
    local_dt = dt.astimezone(PACIFIC_TZ)
    date_part = local_dt.strftime("%b %d").replace(" 0", " ")
    time_part = local_dt.strftime("%I:%M %p").lstrip("0")
    return f"{date_part} {time_part} {local_dt.tzname() or 'PT'}"


def _current_pacific_time() -> datetime:
    return datetime.now(PACIFIC_TZ)


def _player_choice_label(player_name: str, team: str) -> str:
    team = (team or "").strip()
    return f"{player_name} ({team})" if team else player_name


def _utc_to_pst(iso_str: str) -> str:
    """Convert an ISO-8601 UTC timestamp to a friendly PST/PDT string.

    Examples:
        '2026-03-27T22:05:00Z' → 'Mar 27 3:05 PM PDT'
        '' → ''
    """
    if not iso_str:
        return ""
    try:
        dt = pd.Timestamp(iso_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize(UTC_TZ)
        return _format_datetime_pacific(dt.to_pydatetime())
    except Exception:
        return iso_str


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


def _header_refresh_label() -> str:
    try:
        if FRESHNESS_LOG_PATH.exists():
            log = json.loads(FRESHNESS_LOG_PATH.read_text(encoding="utf-8"))
            latest_unix = max(
                float(entry.get("unix"))
                for entry in log.values()
                if isinstance(entry, dict) and entry.get("unix") is not None
            )
            return _format_clock_pacific(datetime.fromtimestamp(latest_unix, tz=UTC_TZ))
    except Exception:
        pass
    return _format_clock_pacific(_current_pacific_time())


def _ensure_ui_session_state() -> None:
    """Initialize Streamlit session flags used by the manual data mode."""
    defaults = {
        "manual_pp_fetch": False,
        "manual_sharp_fetch": False,
        "manual_news_fetch": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
# CACHED WRAPPERS — eliminate redundant API calls across Streamlit reruns
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_pp_lines(include_all: bool = False):
    """PrizePicks lines - cached 5 min."""
    return fetch_prizepicks_mlb_lines(include_all=include_all)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_todays_games(game_dates: tuple = None):
    """MLB schedule + probable pitchers — cached 1 hr.

    Args:
        game_dates: Tuple of date strings (YYYY-MM-DD) to fetch.
                    Uses tuple (not list) for Streamlit cache hashability.
    """
    return fetch_todays_games(game_dates=list(game_dates) if game_dates else None)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_umpires():
    """Home-plate umpire map — cached 1 hr."""
    return fetch_todays_umpires()

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_weather(team_abbr: str):
    """Weather per stadium — cached 30 min."""
    return fetch_game_weather(team_abbr)

@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_sharp_events(api_key: str):
    """Sharp book events list — cached 2 hrs (+ disk cache)."""
    return fetch_mlb_events(api_key)

@st.cache_data(ttl=ODDS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_event_props(event_id: str, api_key: str):
    """Props for one event — cached 2 hrs (+ disk cache, saves API credits)."""
    return fetch_event_props(event_id, api_key=api_key)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_spring_stats():
    """Spring training stats — cached 1 hr."""
    return fetch_spring_training_stats()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_injuries():
    """Injury list — cached 1 hr."""
    return fetch_injuries(days_back=60)

@st.cache_data(ttl=600, show_spinner=False)
def _cached_lineups(game_pk: int):
    """Confirmed lineups per game — cached 10 min."""
    return fetch_confirmed_lineups(game_pk)

st.set_page_config(page_title="MLB Prop Edge", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
_ensure_ui_session_state()

# ── Auto-grade pending predictions on app startup ────────────────────────
# Grades yesterday's (and any older ungraded) picks automatically so you
# never have to remember to press a button.
if "startup_autograde_done" not in st.session_state:
    st.session_state["startup_autograde_done"] = True
    try:
        _ag_yesterday = (date.today() - timedelta(days=1)).isoformat()
        _ag_result = auto_grade_date(_ag_yesterday)
        if _ag_result.get("graded", 0) > 0:
            _ag_wins = sum(1 for r in _ag_result.get("results", []) if r.get("result") == "W")
            _ag_losses = sum(1 for r in _ag_result.get("results", []) if r.get("result") == "L")
            st.toast(f"Auto-graded {_ag_result['graded']} picks from yesterday: {_ag_wins}W-{_ag_losses}L", icon="✅")
    except Exception:
        pass  # Silently skip if grading fails (games not final, no data, etc.)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* === BASE === */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: #070d1a;
        background-image: radial-gradient(circle at 1px 1px, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

    /* === HERO === */
    .hero-wrapper {
        background: linear-gradient(135deg, #070d1a 0%, #0c1828 45%, #071a10 100%);
        border: 1px solid rgba(0,200,83,0.1);
        border-radius: 16px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::after {
        content: ''; position: absolute; top: 0; right: 0;
        width: 320px; height: 100%;
        background: radial-gradient(ellipse at top right, rgba(0,200,83,0.05) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-logo { font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 1.75rem; color: #E8ECF1; letter-spacing: -0.5px; line-height: 1; margin-bottom: 0.2rem; }
    .hero-logo .accent { color: #00C853; }
    .hero-sub { font-weight: 300; font-size: 0.8rem; color: rgba(232,236,241,0.38); letter-spacing: 0.5px; }
    .hero-meta { display: flex; gap: 1.8rem; margin-top: 0.9rem; flex-wrap: wrap; }
    .hero-meta-pill {
        display: flex; align-items: center; gap: 0.45rem;
        font-size: 0.72rem; color: rgba(232,236,241,0.4);
    }
    .hero-meta-pill .pip {
        width: 6px; height: 6px; border-radius: 50%; background: #00C853;
        box-shadow: 0 0 7px rgba(0,200,83,0.55); flex-shrink: 0;
    }
    .hero-meta-pill .pip.amber { background: #FFB300; box-shadow: 0 0 7px rgba(255,179,0,0.45); }
    .hero-meta-pill .pip.blue { background: #4da6ff; box-shadow: 0 0 7px rgba(77,166,255,0.45); }
    .hero-meta-pill strong { color: rgba(232,236,241,0.75); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

    /* === CARDS === */
    .card {
        background: linear-gradient(145deg, #0d1526, #0a1020);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .card:hover { border-color: rgba(255,255,255,0.12); }
    .card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: transparent; }
    .card.card-g::after { background: linear-gradient(90deg, transparent, #00C853, transparent); }
    .card.card-r::after { background: linear-gradient(90deg, transparent, #FF4444, transparent); }
    .card.card-y::after { background: linear-gradient(90deg, transparent, #FFB300, transparent); }
    .card.card-b::after { background: linear-gradient(90deg, transparent, #4da6ff, transparent); }
    .card .lbl { font-size: 0.62rem; color: rgba(232,236,241,0.32); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.45rem; }
    .card .val { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.65rem; color: #E8ECF1; line-height: 1; }
    .card .val.g { color: #00C853; } .card .val.y { color: #FFB300; } .card .val.r { color: #FF4444; } .card .val.b { color: #4da6ff; }
    .card .sub { font-size: 0.68rem; color: rgba(232,236,241,0.28); margin-top: 0.3rem; }

    /* === BADGES === */
    .badge { display: inline-block; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.6rem; padding: 0.18rem 0.55rem; border-radius: 5px; letter-spacing: 1px; vertical-align: middle; }
    .badge-a { background: rgba(0,200,83,0.14); color: #00C853; border: 1px solid rgba(0,200,83,0.22); }
    .badge-b { background: rgba(77,166,255,0.12); color: #4da6ff; border: 1px solid rgba(77,166,255,0.2); }
    .badge-c { background: rgba(255,179,0,0.12); color: #FFB300; border: 1px solid rgba(255,179,0,0.2); }
    .badge-d { background: rgba(255,68,68,0.1); color: #FF4444; border: 1px solid rgba(255,68,68,0.15); }

    /* === DIRECTION === */
    .more { color: #00C853; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .less { color: #FF4444; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .dir-chip { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.75rem; padding: 0.2rem 0.65rem; border-radius: 5px; display: inline-block; }
    .dir-chip.more { background: rgba(0,200,83,0.12); color: #00C853; }
    .dir-chip.less { background: rgba(255,68,68,0.12); color: #FF4444; }

    /* === SECTION HEADERS === */
    .section-hdr { font-weight: 700; font-size: 0.72rem; color: rgba(232,236,241,0.4); text-transform: uppercase; letter-spacing: 2.5px; margin: 1.6rem 0 0.9rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); }

    /* === INFO / WARNING STRIPS === */
    .info-strip { background: rgba(77,166,255,0.05); border: 1px solid rgba(77,166,255,0.1); border-left: 3px solid #4da6ff; border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.8rem; color: rgba(232,236,241,0.55); margin-bottom: 0.9rem; }
    .info-strip .hl { color: #E8ECF1; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .warn-strip { background: rgba(255,179,0,0.05); border: 1px solid rgba(255,179,0,0.12); border-left: 3px solid #FFB300; border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.8rem; color: rgba(232,236,241,0.6); margin-bottom: 0.9rem; }
    .warn-strip strong { color: #FFB300; }
    .alert-strip { background: rgba(255,68,68,0.05); border: 1px solid rgba(255,68,68,0.12); border-left: 3px solid #FF4444; border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.8rem; color: rgba(232,236,241,0.6); margin-bottom: 0.9rem; }
    .alert-strip strong { color: #FF4444; }

    /* === CONTROL / STATUS PANELS === */
    .control-shell {
        background: linear-gradient(145deg, rgba(13,21,38,0.9), rgba(8,14,26,0.96));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .control-shell .title {
        font-size: 0.72rem;
        color: rgba(232,236,241,0.4);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.35rem;
    }
    .control-shell .body {
        font-size: 0.82rem;
        color: rgba(232,236,241,0.62);
        line-height: 1.45;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .status-card {
        background: linear-gradient(145deg, rgba(15,24,41,0.95), rgba(10,16,32,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    .status-card .eyebrow {
        font-size: 0.63rem;
        color: rgba(232,236,241,0.34);
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-bottom: 0.35rem;
    }
    .status-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.92rem;
        color: #E8ECF1;
    }
    .status-card .value.good { color: #00C853; }
    .status-card .value.warn { color: #FFB300; }
    .status-card .value.bad { color: #FF4444; }
    .status-card .sub {
        font-size: 0.74rem;
        color: rgba(232,236,241,0.36);
        margin-top: 0.25rem;
        line-height: 1.35;
    }
    .panel-shell {
        background: linear-gradient(145deg, rgba(13,21,38,0.82), rgba(9,16,31,0.92));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .panel-shell .panel-title {
        font-size: 0.72rem;
        color: rgba(232,236,241,0.38);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.55rem;
    }

    /* === PICK CARDS === */
    .pick-card {
        background: linear-gradient(145deg, #0d1526, #0a1020);
        border: 1px solid rgba(255,255,255,0.07);
        border-left: 4px solid;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .pick-card:hover { border-color: rgba(255,255,255,0.12); background: linear-gradient(145deg, #0f1729, #0c1123); }
    .pick-card.more-pick { border-left-color: #00C853; }
    .pick-card.less-pick { border-left-color: #FF4444; }

    .pick-card-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
    .pick-card-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; }
    .pick-card-player { font-weight: 700; font-size: 0.95rem; color: #E8ECF1; flex: 1; }
    .pick-card-team { font-size: 0.7rem; color: rgba(232,236,241,0.4); font-family: 'JetBrains Mono', monospace; font-weight: 600; }
    .pick-card-stat { font-size: 0.75rem; color: rgba(232,236,241,0.45); }
    .pick-card-line { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; color: #E8ECF1; }
    .pick-card-proj { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; }
    .pick-card-delta { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; }
    .pick-card-edge { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.8rem; }

    .pick-card-conf { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.6rem; }
    .pick-card-conf-label { font-size: 0.7rem; color: rgba(232,236,241,0.4); font-family: 'JetBrains Mono', monospace; }

    /* === CONFIDENCE BAR === */
    .conf-track { background: rgba(255,255,255,0.06); border-radius: 3px; height: 6px; margin: 0; overflow: hidden; flex: 1; }
    .conf-fill { height: 100%; border-radius: 3px; }
    .conf-fill.high { background: linear-gradient(90deg, #00963e, #00C853); }
    .conf-fill.med { background: linear-gradient(90deg, #b37d00, #FFB300); }
    .conf-fill.low { background: linear-gradient(90deg, #aa2200, #FF4444); }

    /* === BEST PLAYS === */
    .best-play {
        background: linear-gradient(145deg, #0d1828, #091020);
        border: 1px solid rgba(0,200,83,0.1);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,200,83,0.08);
        transition: all 0.2s ease;
    }
    .best-play:hover { border-color: rgba(0,200,83,0.2); box-shadow: 0 0 30px rgba(0,200,83,0.12); }
    .best-play::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(0,200,83,0.4), transparent); }
    .bp-name { font-weight: 700; font-size: 0.95rem; color: #E8ECF1; margin-bottom: 0.15rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bp-prop { font-size: 0.7rem; color: rgba(232,236,241,0.38); margin-bottom: 0.45rem; }
    .bp-pick { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.15rem; }
    .bp-pick.more { color: #00C853; } .bp-pick.less { color: #FF4444; }
    .bp-conf { font-size: 0.68rem; color: rgba(232,236,241,0.35); font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem; }

    /* === PICK ROW EXPANDERS === */
    [data-testid="stExpander"] { background: rgba(13,21,38,0.6) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; margin-bottom: 0.35rem !important; }
    [data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.1) !important; background: rgba(13,21,38,0.8) !important; }

    /* === SLIP CARDS === */
    .slip-win { border-left: 3px solid #00C853 !important; }
    .slip-loss { border-left: 3px solid #FF4444 !important; }
    .slip-pending { border-left: 3px solid rgba(255,255,255,0.15) !important; }

    /* === FACTOR BARS === */
    .factor-bar { display: flex; align-items: center; gap: 0.5rem; margin: 0.22rem 0; font-size: 0.8rem; }
    .factor-bar .f-name { color: rgba(232,236,241,0.5); min-width: 130px; }
    .factor-bar .f-impact { font-family: 'JetBrains Mono', monospace; font-weight: 600; min-width: 55px; }
    .factor-bar .f-impact.pos { color: #00C853; } .factor-bar .f-impact.neg { color: #FF4444; } .factor-bar .f-impact.neu { color: rgba(232,236,241,0.3); }

    /* === ACCURACY DISPLAY === */
    .acc-big { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 3rem; line-height: 1; }
    .acc-big.good { color: #00C853; } .acc-big.ok { color: #FFB300; } .acc-big.bad { color: #FF4444; }
    .acc-label { font-size: 0.65rem; color: rgba(232,236,241,0.3); text-transform: uppercase; letter-spacing: 2px; margin-top: 0.2rem; }

    /* === GRADE FEED === */
    .grade-feed-row { display: flex; align-items: center; gap: 1rem; padding: 0.7rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.85rem; }
    .grade-feed-row:last-child { border-bottom: none; }
    .gfr-player { flex: 1; font-weight: 600; color: #E8ECF1; }
    .gfr-prop { color: rgba(232,236,241,0.45); font-size: 0.78rem; }
    .gfr-result { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.9rem; }
    .gfr-result.w { color: #00C853; } .gfr-result.l { color: #FF4444; } .gfr-result.p { color: rgba(232,236,241,0.4); }

    /* === STREAMLIT OVERRIDES === */
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1.2px; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00963e, #00C853) !important;
        border: none !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0,200,83,0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(0,200,83,0.3) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.82rem; }

    /* === MOBILE === */
    @media (max-width: 768px) {
        .stApp [data-testid="block-container"] { padding-left: 0.8rem; padding-right: 0.8rem; }
        .stApp [data-testid="stDataFrame"] { overflow-x: auto !important; }
        .stApp [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.6rem !important;
        }
        .stApp [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: none;
            gap: 0.3rem;
        }
        .stApp [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stApp [data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 0 0 auto;
            min-height: 44px;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .hero-wrapper {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 12px;
        }
        .hero-wrapper::after { width: 45%; }
        .hero-logo { font-size: 1.35rem; }
        .hero-sub { font-size: 0.74rem; }
        .hero-meta { gap: 0.55rem; margin-top: 0.8rem; }
        .hero-meta-pill { font-size: 0.68rem; }
        .section-hdr { font-size: 0.64rem; letter-spacing: 1.6px; margin-top: 1.2rem; }
        .card .val { font-size: 1.25rem; }
        .card { padding: 0.85rem 0.95rem; }
        .info-strip, .warn-strip, .alert-strip { padding: 0.7rem 0.8rem; font-size: 0.76rem; }
        .pick-card { padding: 0.8rem 0.9rem; }
        .pick-card-header,
        .pick-card-row,
        .pick-card-conf,
        .factor-bar,
        .grade-feed-row {
            flex-wrap: wrap;
            align-items: flex-start;
            gap: 0.35rem 0.5rem;
        }
        .pick-card-player,
        .gfr-player {
            width: 100%;
        }
        .pick-card-team { width: 100%; }
        .factor-bar .f-name,
        .factor-bar .f-impact {
            min-width: 0;
        }
        .bp-name { white-space: normal; }
        .status-grid { grid-template-columns: 1fr; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] { font-size: 16px !important; }
    }
</style>
""", unsafe_allow_html=True)

def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v)
def badge(r): return f'<span class="badge badge-{r.lower()}">{r}</span>'
def pick_span(p): return f'<span class="{"more" if p=="MORE" else "less"}">{p}</span>'
def grade_label(r):
    icons = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
    return f"{icons.get(r, '⚪')} {r}"


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


def _display_pitcher_name(name: str) -> str:
    cleaned = _safe(name, "").strip()
    return cleaned if cleaned else "TBD"


def _meets_confidence_floor(pred: dict) -> bool:
    """Return whether a pick clears the configured per-prop confidence floor."""
    if "meets_conf_floor" in pred:
        return bool(pred.get("meets_conf_floor"))

    floor = get_confidence_floor(
        load_current_weights(),
        pred.get("stat_internal") or pred.get("stat_type"),
        pred.get("pick"),
    )
    confidence = _safe_num(pred.get("confidence", pred.get("proj_confidence")), 0.0)
    return confidence >= floor


def _derive_top_plays(scored_all: list[dict], pdf: pd.DataFrame) -> list[dict]:
    trivial_less_props = {"stolen_bases", "home_runs"}

    def _is_trivial(pick: dict) -> bool:
        return (
            pick.get("pick") == "LESS"
            and float(pick.get("line", 99)) <= 0.5
            and pick.get("stat_type", "").lower().replace(" ", "_") in trivial_less_props
        )

    if scored_all:
        top_plays = [
            s for s in scored_all
            if s["combined_grade"] in ("A+", "A")
            and is_tradeable_pick(s.get("stat_internal", s.get("stat_type", "")), s.get("pick", ""))
            and _meets_confidence_floor(s)
            and not _is_trivial(s)
        ][:5]
    else:
        top_plays = []

    if not top_plays and not pdf.empty:
        tradeable_best = pdf[
            pdf.apply(
                lambda r: (
                    is_tradeable_pick(r.get("stat_internal", ""), r.get("pick", ""))
                    and _meets_confidence_floor(r)
                ),
                axis=1,
            )
        ].sort_values(["confidence", "edge"], ascending=False)
        for _, tp in tradeable_best.head(5).iterrows():
            top_plays.append({
                "player_name": tp["player_name"],
                "stat_type": tp["stat_type"],
                "line": tp["line"],
                "pick": tp["pick"],
                "combined_score": tp.get("edge", 0),
                "combined_grade": tp.get("rating", "B"),
                "signal": SIGNAL_PROJECTION_ONLY,
                "proj_confidence": tp.get("confidence", 0.5),
            })
            if len(top_plays) >= 5:
                break

    return top_plays


def _render_top_plays(top_plays: list[dict]) -> None:
    if not top_plays:
        return

    st.markdown('<div class="section-hdr">Today\'s Best Plays</div>', unsafe_allow_html=True)
    grade_icons = {"A+": "A+", "A": "A", "B": "B", "C": "C", "D": "D"}
    signal_labels = {
        SIGNAL_CONFIRMED: "CONFIRMED",
        SIGNAL_SHARP_ONLY: "SHARP",
        SIGNAL_PROJECTION_ONLY: "PROJECTION",
    }
    bp_cols = st.columns(min(len(top_plays), 5))
    for idx, tp in enumerate(top_plays[:5]):
        pick_cls = "more" if tp["pick"] == "MORE" else "less"
        grade_icon = grade_icons.get(tp.get("combined_grade", ""), "")
        conf_val = _safe_num(tp.get("proj_confidence"), 0)
        conf_pct = int(conf_val * 100)
        conf_cls = "high" if conf_val > 0.6 else ("med" if conf_val > 0.52 else "low")
        sig_label = signal_labels.get(tp.get("signal", ""), "PROJECTION")
        line_val = _safe_num(tp.get("line"), 0.0)
        line_text = f"{line_val:g}" if isinstance(line_val, (int, float)) else str(tp.get("line", ""))
        with bp_cols[idx]:
            st.markdown(f'''<div class="best-play">
                <div class="bp-name">{tp["player_name"]}</div>
                <div class="bp-prop">{tp["stat_type"]} - Line {line_text}</div>
                <div class="bp-pick {pick_cls}">{tp["pick"]}</div>
                <div class="conf-track"><div class="conf-fill {conf_cls}" style="width:{min(conf_pct,100)}%"></div></div>
                <div class="bp-conf">{grade_icon} {tp.get("combined_grade","?")} - {conf_pct}% conf - {sig_label}</div>
            </div>''', unsafe_allow_html=True)


# ── PrizePicks tradeable prop configurations ──────────────────────────────────
PP_TRADEABLE: dict = {
    "pitcher_strikeouts":  {"directions": ["MORE", "LESS"], "lines": [3.5, 4.5, 5.5, 6.5, 7.5]},
    "hitter_fantasy_score": {"directions": ["MORE", "LESS"], "lines": [7.5]},
    "total_bases":         {"directions": ["MORE"], "lines": [1.5, 2.5]},
    "hits_runs_rbis":      {"directions": ["MORE", "LESS"], "lines": [1.5, 2.5]},
    "hits":                {"directions": ["MORE", "LESS"], "lines": [0.5, 1.5]},
    "batter_strikeouts":   {"directions": ["MORE", "LESS"], "lines": [0.5]},
    "walks_allowed":       {"directions": ["MORE", "LESS"], "lines": [1.5, 2.5]},
    "earned_runs":         {"directions": ["MORE", "LESS"], "lines": [1.5, 2.5]},
    "pitching_outs":       {"directions": ["MORE", "LESS"], "lines": [15.5, 16.5, 17.5]},
    "runs":                {"directions": ["MORE"], "lines": [0.5]},
    "rbis":                {"directions": ["MORE"], "lines": [0.5]},
}

PP_NEVER_SHOW: set = {
    ("home_runs", "LESS"),
    ("stolen_bases", "LESS"),
    ("total_bases", "LESS"),
    ("hitter_fantasy_score", "MORE"),
}

# ── Minimum realistic lines per prop type ────────────────────────────────
# Lines below these thresholds are spring training / promo artifacts.
# A regular-season H+R+RBI line is always >= 1.5; 0.5 is a garbage line.
MIN_REALISTIC_LINE: dict = {
    "hits_runs_rbis":      1.0,
    "hitter_fantasy_score": 3.0,
    "total_bases":         0.5,   # 0.5 is legitimate for TB
    "pitcher_strikeouts":  2.5,
    "pitching_outs":       10.5,
    "earned_runs":         0.5,
    "walks_allowed":       0.5,
    "hits_allowed":        1.5,
}

# Maximum edge cap — any edge above this is almost certainly a data artifact
MAX_EDGE_PCT = 0.35  # 35% — real edges are 5-15%

_PP_FILTERED_LABELS = {
    ("home_runs", "LESS"): "HR LESS",
    ("stolen_bases", "LESS"): "SB LESS",
    ("total_bases", "LESS"): "TB LESS",
}


def is_tradeable_pick(stat_internal: str, direction: str) -> bool:
    """Return False if this (stat_internal, direction) is not tradeable on PrizePicks."""
    if not stat_internal or not direction:
        return False
    if (stat_internal, direction) in PP_NEVER_SHOW:
        return False
    cfg = PP_TRADEABLE.get(stat_internal)
    if cfg is None:
        return False
    return direction in cfg["directions"]


@st.cache_data(ttl=3600)
def load_batting_stats():
    """Load batting leaders from cached CSV first, fall back to pybaseball."""
    import os
    cache_path = os.path.join(os.path.dirname(__file__), "data", "batting_stats_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) >= 50:
            return df

    from datetime import datetime
    year = datetime.now().year
    df = fetch_batting_leaders(year, min_pa=100)
    if df.empty or len(df) < 50:
        df = fetch_batting_leaders(year - 1, min_pa=100)
    if df.empty or len(df) < 50:
        df = fetch_batting_leaders(year - 2, min_pa=100)
    return df


def _normalize_name(name: str) -> str:
    """Normalize player name for matching: lowercase, strip accents, remove Jr/Sr/III."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]:
        name = name.replace(suffix, "")
    return name.strip()


def _build_name_index(df: pd.DataFrame) -> dict:
    """Build normalized-name lookup dicts for O(1) player matching."""
    exact = {}          # norm_name -> row
    initial_last = {}   # (first_initial, last_name) -> row
    last_only = {}      # last_name -> [rows]
    for idx, row in df.iterrows():
        raw = str(row.get("Name", ""))
        norm = _normalize_name(raw)
        if norm not in exact:
            exact[norm] = row
        parts = norm.split()
        if len(parts) >= 2:
            key = (parts[0][0], parts[-1])
            if key not in initial_last:
                initial_last[key] = row
            last_only.setdefault(parts[-1], []).append(row)
    return {"exact": exact, "initial_last": initial_last, "last_only": last_only}


_batting_index_cache: dict | None = None
_pitching_index_cache: dict | None = None


def _get_batting_index(batting_df: pd.DataFrame) -> dict:
    global _batting_index_cache
    if _batting_index_cache is None:
        _batting_index_cache = _build_name_index(batting_df)
    return _batting_index_cache


def _get_pitching_index(pitching_df: pd.DataFrame) -> dict:
    global _pitching_index_cache
    if _pitching_index_cache is None:
        _pitching_index_cache = _build_name_index(pitching_df)
    return _pitching_index_cache


def match_player_stats(player_name: str, batting_df: pd.DataFrame) -> pd.Series:
    """Match PrizePicks player name to FanGraphs batting row."""
    if batting_df.empty or "Name" not in batting_df.columns:
        return None
    idx = _get_batting_index(batting_df)
    norm_target = _normalize_name(player_name)
    if norm_target in idx["exact"]:
        return idx["exact"][norm_target]
    parts = norm_target.split()
    if len(parts) >= 2:
        key = (parts[0][0], parts[-1])
        if key in idx["initial_last"]:
            return idx["initial_last"][key]
    if parts:
        candidates = idx["last_only"].get(parts[-1], [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def build_batter_profile(stats_row: pd.Series) -> dict:
    """Convert a FanGraphs DataFrame row into the batter profile dict the predictor expects."""
    def safe_pct(val):
        if isinstance(val, str):
            val = float(val.replace("%", "").strip())
        else:
            val = float(val) if val else 0.0
        return val * 100.0 if 0 < val < 1 else val

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


PITCHER_STAT_INTERNALS = {"pitcher_strikeouts", "hits_allowed", "earned_runs", "walks_allowed", "pitching_outs"}


@st.cache_data(ttl=3600)
def load_pitching_stats():
    """Load pitching leaders from cached CSV first, fall back to pybaseball."""
    import os
    cache_path = os.path.join(os.path.dirname(__file__), "data", "pitching_stats_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) >= 20:
            return df
    from datetime import datetime as _dt
    year = _dt.now().year
    df = fetch_pitching_leaders(year, min_ip=10)
    if df.empty or len(df) < 20:
        df = fetch_pitching_leaders(year - 1, min_ip=10)
    if not df.empty:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception as e:
            _log.warning("Failed to cache pitching stats: %s", e)
            pass
    return df


def match_pitcher_stats(player_name: str, pitching_df: pd.DataFrame):
    """Match PrizePicks player name to FanGraphs pitching row."""
    if pitching_df.empty or "Name" not in pitching_df.columns:
        return None
    idx = _get_pitching_index(pitching_df)
    norm_target = _normalize_name(player_name)
    if norm_target in idx["exact"]:
        return idx["exact"][norm_target]
    parts = norm_target.split()
    if len(parts) >= 2:
        key = (parts[0][0], parts[-1])
        if key in idx["initial_last"]:
            return idx["initial_last"][key]
    return None


def build_pitcher_profile(stats_row) -> dict:
    """Convert a FanGraphs pitching row into the pitcher profile dict the predictor expects."""
    def safe_pct(val):
        if isinstance(val, str):
            val = float(val.replace("%", "").strip())
        else:
            val = float(val) if val else 0.0
        return val * 100.0 if 0 < val < 1 else val
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
        # Statcast pitch quality metrics for K-rate adjustments
        "recent_csw_pct": float(stats_row.get("CSW%", 0) or 0),
        "recent_swstr_pct": float(stats_row.get("SwStr%", 0) or 0),
        "ip_per_start": float(stats_row.get("IP", 0)) / max(float(stats_row.get("GS", stats_row.get("G", 1))), 1),
    }


def get_daily_log_summary(days: int = 14) -> list[dict]:
    """Read daily log JSON files and return summary rows for the last N days."""
    import json as _j, os as _o
    log_dir = _o.path.join(_o.path.dirname(__file__), "data", "daily_logs")
    if not _o.path.exists(log_dir):
        return []
    rows = []
    for i in range(days - 1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        path = _o.path.join(log_dir, f"{d}.json")
        if not _o.path.exists(path):
            continue
        try:
            with open(path) as _f:
                data = _j.load(_f)
            preds = data.get("predictions", [])
            rows.append({
                "Date": d,
                "Props": len(preds),
                "A Picks": sum(1 for p in preds if p.get("rating") == "A"),
                "B Picks": sum(1 for p in preds if p.get("rating") == "B"),
                "Avg Edge": (
                    f"{sum(p.get('edge', 0) or 0 for p in preds) / len(preds) * 100:.1f}%"
                    if preds else "—"
                ),
                "Sharp Edges": sum(1 for p in preds if p.get("edge", 0) and p["edge"] > 0.03),
            })
        except Exception:
            pass
    return rows


def save_daily_log(preds_list: list, log_date: str = None):
    """Save a snapshot of today's projections to data/daily_logs/<date>.json."""
    import json, os
    log_date = log_date or date.today().isoformat()
    log_dir = os.path.join(os.path.dirname(__file__), "data", "daily_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_date}.json")
    snapshot = []
    for p in preds_list:
        snapshot.append({
            "player_name": p.get("player_name"),
            "stat_type": p.get("stat_type"),
            "stat_internal": p.get("stat_internal"),
            "line": p.get("line"),
            "projection": p.get("projection"),
            "pick": p.get("pick"),
            "confidence": p.get("confidence"),
            "edge": p.get("edge"),
            "rating": p.get("rating"),
            "spring_mult": p.get("spring_mult"),
            "trend_mult": p.get("trend_mult"),
            "buy_low": p.get("buy_low"),
            "team": p.get("team"),
        })
    with open(log_path, "w") as f:
        json.dump({"date": log_date, "model_version": "v2", "predictions": snapshot}, f, indent=2)
    return log_path


def get_bankroll_history() -> list[dict]:
    """Build a running bankroll balance from slip history."""
    conn_fn = __import__("src.database", fromlist=["get_connection"]).get_connection
    conn = conn_fn()
    slips = pd.read_sql_query(
        "SELECT * FROM slips WHERE status != 'pending' ORDER BY game_date ASC, id ASC", conn
    )
    conn.close()
    if slips.empty:
        return []
    history = []
    balance = st.session_state.get("starting_bankroll", 100.0)
    history.append({"date": slips.iloc[0]["game_date"], "balance": balance, "label": "Start"})
    for _, s in slips.iterrows():
        balance += s["net_profit"]
        history.append({
            "date": s["game_date"],
            "balance": round(balance, 2),
            "label": f"Slip #{s['id']}: {'+' if s['net_profit']>=0 else ''}{s['net_profit']:.2f}",
        })
    return history


init_db()
init_slips_table()
init_projected_stats_table()

try:
    _wts = load_current_weights()
    _model_ver = _wts.get("version", "v1") if isinstance(_wts, dict) else "v1"
except Exception:
    _model_ver = "v1"
_freshness_str = _header_refresh_label()

st.markdown(f"""<div class="hero-wrapper">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div class="hero-logo">⚾ MLB Prop <span class="accent">Edge</span></div>
      <div class="hero-sub">Sharp-Based Prop Predictions &nbsp;·&nbsp; Statcast Confirmed &nbsp;·&nbsp; Self-Learning Model</div>
      <div class="hero-meta">
        <div class="hero-meta-pill"><span class="pip"></span>Model <strong>{_model_ver}</strong> active</div>
        <div class="hero-meta-pill"><span class="pip amber"></span>Refreshed <strong>{_freshness_str}</strong></div>
        <div class="hero-meta-pill"><span class="pip blue"></span>FanDuel · Pinnacle · DraftKings</div>
        <div class="hero-meta-pill"><span class="pip amber"></span>Manual odds mode <strong>enabled</strong></div>
      </div>
    </div>
    <div style="text-align:right;padding-top:0.2rem;">
      <div style="font-size:0.6rem;color:rgba(232,236,241,0.2);letter-spacing:2px;text-transform:uppercase;">Season 2026</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:rgba(232,236,241,0.5);font-weight:600;margin-top:0.2rem;">+EV FINDER</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Sidebar: Config & Status ──
with st.sidebar:
    st.markdown("### Settings")
    _sb_key = get_api_key()
    if _sb_key:
        st.success("Odds API key configured")
        _cached_creds = get_credits_remaining()
        _cache_age = get_cache_age_minutes()
        _has_cache = has_cached_odds_today()
        if _cached_creds >= 0:
            _cred_color = "🟢" if _cached_creds > 200 else ("🟡" if _cached_creds > 50 else "🔴")
            st.caption(f"{_cred_color} {_cached_creds} credits remaining")
        if _has_cache and _cache_age >= 0:
            _hrs = _cache_age // 60
            _mins = _cache_age % 60
            _age_str = f"{_hrs}h {_mins}m" if _hrs else f"{_mins}m"
            st.caption(f"📦 Odds cached ({_age_str} ago)")
        elif not _has_cache:
            st.caption("📦 No cached odds — manual pull required")
        _cred_col1, _cred_col2 = st.columns(2)
        with _cred_col1:
            if st.button("Check Credits", key="sb_credits"):
                _u = get_api_usage(_sb_key)
                _remaining = _u.get('remaining', 0)
                _used = _u.get('used', 0)
                st.info(f"Remaining: **{_remaining}** · Used: **{_used}**")
                if isinstance(_remaining, (int, float)) and _remaining < 100:
                    st.error(f"⚠️ Only {_remaining} credits left!")
                elif isinstance(_remaining, (int, float)) and _remaining < 250:
                    st.warning(f"Credits getting low ({_remaining})")
        with _cred_col2:
            if st.button("🔄 Refresh Odds", key="sb_refresh_odds"):
                clear_odds_cache()
                _cached_sharp_events.clear()
                _cached_event_props.clear()
                st.success("Odds cache cleared — refreshing...")
                st.rerun()
    else:
        st.warning("No Odds API key — add `ODDS_API_KEY` to Streamlit Secrets or `.env`")
    st.number_input("Starting Bankroll ($)", min_value=10.0, value=st.session_state.get("starting_bankroll", 100.0), step=10.0, key="sb_bankroll")
    st.session_state["starting_bankroll"] = st.session_state.get("sb_bankroll", 100.0)

    # Quick accuracy snapshot
    _sb_stats = get_accuracy_stats()
    if _sb_stats["total"] > 0:
        st.markdown("### Record")
        _sb_acc = _sb_stats["accuracy"]
        _five_flex_be = BREAKEVEN.get("5_flex", 0.54253)
        st.metric(
            "Accuracy",
            f"{_sb_acc * 100:.1f}%",
            delta=f"{(_sb_acc - _five_flex_be) * 100:+.1f} pts vs 5-flex BE",
        )
        st.caption(f"{_sb_stats['wins']}W – {_sb_stats['losses']}L ({_sb_stats['total']} picks)")

tab_edge, tab_news, tab_slips, tab_grade, tab_qa = st.tabs(
    ["🎯 FIND EDGES", "📰 NEWS", "🎫 MY SLIPS", "✅ GRADE", "📈 MODEL QA"]
)

with tab_edge:
    api_key = get_api_key()
    has_sharp = bool(api_key)
    pp_requested = bool(st.session_state.get("manual_pp_fetch", False))
    sharp_requested = bool(st.session_state.get("manual_sharp_fetch", False))
    sharp_events = []
    sharp_events_available = False
    all_pp_lines = pd.DataFrame()
    pp_lines = pd.DataFrame()
    selected_player_label = PLAYER_SEARCH_ALL
    selected_player_name = None
    selected_player_team = None

    st.markdown(
        """
        <div class="control-shell">
          <div class="title">Manual Data Mode</div>
          <div class="body">
            External pulls are disabled until you explicitly request them. This keeps Odds API credit usage under control while you test the app.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctl1, ctl2, ctl3, ctl4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with ctl1:
        if st.button("Load PrizePicks Board", key="edge_load_pp", type="primary"):
            st.session_state["manual_pp_fetch"] = True
            pp_requested = True
    with ctl2:
        if st.button("Load Sharp Odds", key="edge_load_sharp", disabled=not has_sharp):
            st.session_state["manual_pp_fetch"] = True
            st.session_state["manual_sharp_fetch"] = True
            pp_requested = True
            sharp_requested = True
    with ctl3:
        if st.button("Refresh PrizePicks", key="edge_refresh_pp", disabled=not pp_requested):
            _cached_pp_lines.clear()
            st.session_state["manual_pp_fetch"] = True
            pp_requested = True
    with ctl4:
        if st.button("Refresh Sharp Odds", key="edge_refresh_sharp", disabled=not has_sharp):
            _cached_sharp_events.clear()
            _cached_event_props.clear()
            clear_odds_cache()
            st.session_state["manual_sharp_fetch"] = True
            st.session_state["manual_pp_fetch"] = True
            pp_requested = True
            sharp_requested = True

    pp_status = "Loaded" if pp_requested else "Idle"
    pp_status_class = "good" if pp_requested else "warn"
    sharp_status = "Loaded" if sharp_requested and has_sharp else ("No API key" if not has_sharp else "Idle")
    sharp_status_class = "good" if sharp_requested and has_sharp else ("bad" if not has_sharp else "warn")
    mode_status = "Manual mode"
    mode_sub = "Only button-driven pulls are allowed in this session."
    st.markdown(
        f"""
        <div class="status-grid">
          <div class="status-card">
            <div class="eyebrow">PrizePicks board</div>
            <div class="value {pp_status_class}">{pp_status}</div>
            <div class="sub">Board data stays local until you click a load or refresh button.</div>
          </div>
          <div class="status-card">
            <div class="eyebrow">Sharp odds</div>
            <div class="value {sharp_status_class}">{sharp_status}</div>
            <div class="sub">The Odds API is only touched after a manual request.</div>
          </div>
          <div class="status-card">
            <div class="eyebrow">App mode</div>
            <div class="value warn">{mode_status}</div>
            <div class="sub">{mode_sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_sharp and sharp_requested:
        try:
            sharp_events = _cached_sharp_events(api_key) or []
            sharp_events_available = bool(sharp_events)
        except Exception:
            sharp_events = []
            sharp_events_available = False

    if not api_key:
        st.markdown(
            '<div class="alert-strip"><strong>No Odds API key found.</strong> '
            'Sharp book comparison is disabled. Create a <code>.env</code> file in the project folder with: '
            '<code>ODDS_API_KEY=your_key_here</code><br>'
            'Free key at <a href="https://the-odds-api.com" target="_blank">the-odds-api.com</a> - 500 req/month, no credit card.</div>',
            unsafe_allow_html=True
        )
    elif not sharp_requested:
        st.markdown(
            '<div class="info-strip"><strong>Sharp odds are paused.</strong> Click <span class="hl">Load Sharp Odds</span> only when you want a live odds pull.</div>',
            unsafe_allow_html=True
        )
    elif not sharp_events_available:
        st.markdown(
            '<div class="warn-strip"><strong>No sharp-book events posted yet.</strong> Projection-only mode for now; sharp comparison will appear once sportsbooks publish MLB player-prop events.</div>',
            unsafe_allow_html=True
        )
    else:
        _creds_display = get_credits_remaining()
        _creds_str = str(_creds_display) if _creds_display >= 0 else "?"
        _age_display = get_cache_age_minutes()
        _age_msg = f"Cached {_age_display}m ago" if _age_display >= 0 else "Fresh pull this session"
        st.markdown(f'<div class="info-strip">Odds API active | <span class="hl">{_creds_str}</span> credits remaining | {_age_msg} | {len(sharp_events)} MLB events | Sharp books: FanDuel, Pinnacle, DraftKings</div>', unsafe_allow_html=True)

    if pp_requested:
        with st.spinner("Pulling PrizePicks MLB lines..."):
            try:
                all_pp_lines = _cached_pp_lines(include_all=True)
            except Exception:
                all_pp_lines = pd.DataFrame()
        if not all_pp_lines.empty and "model_eligible" in all_pp_lines.columns:
            pp_lines = all_pp_lines[all_pp_lines["model_eligible"] == True].copy()
        else:
            pp_lines = all_pp_lines.copy()
    else:
        st.markdown(
            '<div class="panel-shell"><div class="panel-title">Board data is paused</div>'
            '<div style="font-size:0.84rem;color:rgba(232,236,241,0.62);line-height:1.45;">'
            'Click <strong>Load PrizePicks Board</strong> to pull current lines. Until then, the app will not hit PrizePicks or the Odds API from this tab.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Snapshot PP lines for CLV tracking and stale-line detection
    if not pp_lines.empty:
        try:
            snapshot_pp_lines(pp_lines)
        except Exception:
            pass

    if pp_requested and pp_lines.empty:
        st.info("No MLB lines on PrizePicks right now. Lines usually post by 10 AM ET.")
    elif not pp_lines.empty:
        st.markdown(f"**{len(pp_lines)} MLB props** on PrizePicks today")
        if not all_pp_lines.empty:
            player_rows = (
                all_pp_lines[["player_name", "team"]]
                .dropna(subset=["player_name"])
                .drop_duplicates()
                .sort_values(["player_name", "team"])
            )
            player_rows = player_rows[player_rows["player_name"].astype(str).str.strip() != ""]
            player_lookup = {
                _player_choice_label(row["player_name"], row.get("team", "")): (
                    row["player_name"],
                    row.get("team", ""),
                )
                for _, row in player_rows.iterrows()
            }
            selected_player_label = st.selectbox(
                "Find player props",
                [PLAYER_SEARCH_ALL] + list(player_lookup.keys()),
                index=0,
                help="Type a player name to see every current PrizePicks prop for that player, including lines excluded from the model board.",
                key="player_prop_search",
            )
            if selected_player_label != PLAYER_SEARCH_ALL:
                selected_player_name, selected_player_team = player_lookup[selected_player_label]
        all_edges = []
        _game_totals_by_team: dict[str, float] = {}  # team_name_lower -> game total O/U
        if has_sharp and sharp_events_available:
            total_sharp_lines = 0
            events_with_props = 0
            _skipped_events = 0
            with st.spinner("Fetching sharp lines & devigging..."):
                events = sharp_events
                # Smart filter: only fetch props for games with PP lines
                _pp_teams = set()
                if not pp_lines.empty and "team" in pp_lines.columns:
                    _pp_teams = set(pp_lines["team"].dropna().str.lower().unique())
                for event in (events or [])[:15]:
                    eid = event.get("id","")
                    if not eid: continue
                    # Skip events where neither team has PP lines
                    if _pp_teams:
                        _home = event.get("home_team","").lower()
                        _away = event.get("away_team","").lower()
                        if not any(t in _home or t in _away or _home in t or _away in t for t in _pp_teams):
                            _skipped_events += 1
                            continue
                    result = _cached_event_props(eid, api_key=api_key)
                    if result and "data" in result:
                        sharp = extract_sharp_lines(result["data"])
                        if sharp:
                            events_with_props += 1
                            total_sharp_lines += len(sharp)
                        all_edges.extend(find_ev_edges(pp_lines, sharp, min_ev_pct=0.25))
                        # Extract game total for run-environment nudge
                        from src.sharp_odds import extract_game_total
                        gt = extract_game_total(result["data"])
                        if gt:
                            _ev_home = event.get("home_team", "").lower()
                            _ev_away = event.get("away_team", "").lower()
                            if _ev_home:
                                _game_totals_by_team[_ev_home] = gt
                            if _ev_away:
                                _game_totals_by_team[_ev_away] = gt
            _skip_msg = f" · {_skipped_events} skipped (no PP lines)" if _skipped_events else ""
            st.caption(f"Scanned {len(events or [])} events · {events_with_props} had props · {total_sharp_lines} sharp lines · {len(all_edges)} edges{_skip_msg}")

        if all_edges:
            all_edges.sort(key=lambda x: x["edge_pct"], reverse=True)
            a_n = sum(1 for e in all_edges if e["rating"]=="A")
            b_n = sum(1 for e in all_edges if e["rating"]=="B")
            avg_e = np.mean([e["edge_pct"] for e in all_edges])
            best_e = max(e["edge_pct"] for e in all_edges)
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(f'<div class="card card-g"><div class="lbl">A-Grade Edges</div><div class="val g">{a_n}</div><div class="sub">Highest confidence</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="card card-b"><div class="lbl">B-Grade Edges</div><div class="val b">{b_n}</div><div class="sub">Strong signal</div></div>', unsafe_allow_html=True)
            with c3:
                cls = "g" if avg_e>5 else ("y" if avg_e>3 else "r")
                ccard = "card-g" if avg_e>5 else ("card-y" if avg_e>3 else "card-r")
                st.markdown(f'<div class="card {ccard}"><div class="lbl">Avg Edge</div><div class="val {cls}">{avg_e:.1f}%</div><div class="sub">vs PrizePicks line</div></div>', unsafe_allow_html=True)
            with c4:
                bcls = "g" if best_e>8 else ("y" if best_e>5 else "")
                st.markdown(f'<div class="card"><div class="lbl">Best Edge</div><div class="val {bcls}">+{best_e:.1f}%</div><div class="sub">{len(all_edges)} total edges</div></div>', unsafe_allow_html=True)

            _sharp_prop_options = ["All"] + sorted({e.get("stat_type", "") for e in all_edges if e.get("stat_type")})
            f1,f2,f3 = st.columns([2,2,2])
            with f1: min_grade = st.selectbox("Min grade", ["A only","A + B","A + B + C","All"], index=1)
            with f2: prop_f = st.selectbox("Prop type", _sharp_prop_options)
            with f3: show_all_sharp = st.checkbox("Show all picks (incl. non-tradeable)", value=False, key="show_all_sharp")
            gm = {"A only":["A"],"A + B":["A","B"],"A + B + C":["A","B","C"],"All":["A","B","C","D"]}
            filt = [e for e in all_edges if e["rating"] in gm[min_grade]]
            if prop_f != "All":
                filt = [e for e in filt if e.get("stat_type") == prop_f]
            if prop_f == "Total Bases":
                less_tb = [e for e in filt if e.get("pick") == "LESS"]
                if less_tb:
                    st.markdown('<div class="warn-strip"><strong>TB LESS Warning:</strong> Total Bases LESS picks historically underperform — trade with extra caution</div>', unsafe_allow_html=True)
            if not show_all_sharp:
                _hidden_sharp = [e for e in filt if not is_tradeable_pick(e.get("stat_internal", e.get("market", "")), e.get("pick", ""))]
                filt = [e for e in filt if is_tradeable_pick(e.get("stat_internal", e.get("market", "")), e.get("pick", ""))]
                _hidden_labels = list({_PP_FILTERED_LABELS.get((e.get("stat_internal",""), e.get("pick","")), f"{e.get('stat_type','')} {e.get('pick','')}") for e in _hidden_sharp})
                if _hidden_labels:
                    st.info(f"Showing tradeable picks only. Filtered out: {', '.join(_hidden_labels)}")

            with st.expander("Betting Rules & Bankroll Guide", expanded=False):
                _brc1, _brc2 = st.columns(2)
                with _brc1:
                    st.markdown("**BET THESE PROPS** *(v017 — 71.7% combined)*")
                    st.markdown(
                        "- Hits LESS 1.5 — **72.5%**\n"
                        "- Pitcher Ks MORE 4.5 — **69.6%**\n"
                        "- Pitcher Ks LESS 4.5 — **69.2%**\n"
                        "- TB LESS 1.5 — **65.8%**\n"
                        "- FS LESS 7.5 — **61.7%**\n"
                        "- FS MORE 7.5 — **59.9%**"
                    )
                    st.markdown("**AVOID:** Hits MORE, TB MORE (disabled), HR LESS, SB LESS")
                with _brc2:
                    st.markdown("**SLIP RULES:** Mix MORE+LESS, max 2 picks/team, min B grade, 5–6 Pick Flex")
                    st.markdown("**BANKROLL:** 1–2% per slip, max 5 slips/day, stop if down 10%")

            if filt:
                st.markdown(f'<div class="section-hdr">Sharp Edges — {len(filt)} found · ranked by edge %</div>', unsafe_allow_html=True)
                edf = pd.DataFrame(filt)
                disp = edf[["player_name","team","stat_type","pp_line","pick","edge_pct","fair_prob","rating","num_books"]].copy()
                disp.columns = ["Player","Team","Prop","Line","Pick","Edge %","Fair Prob","Grade","Books"]
                disp["Fair Prob"] = disp["Fair Prob"].apply(lambda x: f"{x*100:.1f}%")
                disp["Edge %"] = disp["Edge %"].apply(lambda x: f"+{x:.1f}%")
                st.dataframe(disp, hide_index=True, width="stretch", height=min(len(disp)*38+40,600))

                st.markdown('<div class="section-hdr">Pick Details</div>', unsafe_allow_html=True)
                for edge in filt[:8]:
                    with st.expander(f"{grade_label(edge['rating'])} **{edge['player_name']}** — {edge['stat_type']} | Line: {edge['pp_line']} | Edge: +{edge['edge_pct']:.1f}%"):
                        c1,c2,c3,c4 = st.columns(4)
                        with c1: st.metric("Pick", edge["pick"])
                        with c2: st.metric("Fair Prob", f"{edge['fair_prob']*100:.1f}%")
                        with c3: st.metric("Edge", f"+{edge['edge_pct']:.1f}%")
                        with c4: st.metric("Books", edge["num_books"])
                        if edge.get("fanduel_agrees"): st.success("FanDuel confirms this side")

                st.markdown("---")
                if st.button("Save Edges", type="primary"):
                    rows_to_save = [
                        {**edge, "game_date": _game_date_from_iso(edge.get("start_time", ""))}
                        for edge in filt
                    ]
                    log_batch_predictions(rows_to_save)
                    st.success(f"Saved {len(filt)} predictions!")
            else: st.info("No edges match filters.")
        elif has_sharp and sharp_events_available:
            st.info("No sharp book player prop edges found right now — odds may not be posted yet. Showing projection-based analysis below.")

        if not all_edges:
            st.markdown('<div class="section-hdr">Projection Analysis</div>', unsafe_allow_html=True)
            stats_failed = False
            pitching_df = pd.DataFrame()
            with st.spinner("Loading player stats..."):
                try:
                    batting_df = load_batting_stats()
                except Exception:
                    batting_df = pd.DataFrame()
                    stats_failed = True
                try:
                    pitching_df = load_pitching_stats()
                except Exception:
                    pitching_df = pd.DataFrame()
            if stats_failed or batting_df.empty:
                st.warning("Could not load player stats — using league averages for projections.")
            else:
                p_cap = f"Loaded {len(batting_df)} batters"
                if not pitching_df.empty:
                    p_cap += f" + {len(pitching_df)} pitchers"
                p_cap += " from FanGraphs"
                st.caption(p_cap)
            preds = []
            teams_in_slate = set()
            for _, row in pp_lines.iterrows():
                t = row.get("team", "")
                if t:
                    r = resolve_team(t)
                    if r and r in STADIUMS:
                        teams_in_slate.add(r)
            # Extract actual game dates from PrizePicks data (may be future dates
            # like Opening Day), NOT today's date which could be spring training.
            _pp_game_dates = extract_schedule_dates(
                pp_lines["start_time"].dropna() if "start_time" in pp_lines.columns else [],
            )
            # Fetch games for the actual dates PP has props for
            try:
                todays_games = _cached_todays_games(game_dates=tuple(sorted(_pp_game_dates)))
            except Exception:
                todays_games = []

            weather_cache = {}
            if teams_in_slate:
                wx_prog = st.progress(0, text="Fetching weather for stadiums...")
                for j, team_abbr in enumerate(sorted(teams_in_slate)):
                    wx_prog.progress((j + 1) / len(teams_in_slate), text=f"Weather: {STADIUMS[team_abbr]['name']} ({j+1}/{len(teams_in_slate)})")
                    try:
                        weather_cache[team_abbr] = _cached_weather(team_abbr)
                    except Exception:
                        weather_cache[team_abbr] = None
                wx_prog.empty()

            st_stats = []
            injury_list = []
            with st.spinner("Loading Spring Training data & injuries..."):
                try:
                    st_stats = _cached_spring_stats()
                except Exception:
                    st_stats = []
                try:
                    injury_list = _cached_injuries()
                except Exception:
                    injury_list = []

            # Fetch today's home-plate umpire assignments
            umpire_map = {}
            try:
                umpire_map = _cached_umpires()
            except Exception:
                pass

            # Pre-build opposing pitcher lookup: team_abbr → {pitcher_name, hand, profile}
            # For batter props, "opposing pitcher" is the pitcher the batter faces
            # For pitcher props, "opposing lineup" stats come from the other team's batting
            opp_pitcher_lookup = {}  # team_abbr -> dict with pitcher info + FanGraphs profile
            team_pitcher_lookup = {}  # team_abbr -> own probable starting pitcher
            opp_team_k_lookup = {}   # team_abbr -> team K% for pitcher K projections
            team_k_rate_map = {}
            if not batting_df.empty and "Team" in batting_df.columns and "K%" in batting_df.columns:
                try:
                    _team_rates = batting_df[["Team", "K%"]].copy()
                    _team_rates["_team_code"] = _team_rates["Team"].apply(normalize_team_code)
                    _team_rates = _team_rates[_team_rates["_team_code"] != ""]
                    for _team_code, _subset in _team_rates.groupby("_team_code"):
                        k_vals = _subset["K%"].apply(
                            lambda x: (
                                (lambda v: v * 100.0 if 0 < v < 1 else v)(
                                    float(str(x).replace("%", "").strip())
                                )
                                if pd.notna(x)
                                else 22.7
                            )
                        )
                        team_k_rate_map[_team_code] = round(k_vals.mean(), 1)
                except Exception:
                    team_k_rate_map = {}
            try:
                for game in todays_games:
                    game_pk = game.get("game_pk")
                    game_time = game.get("game_time", "")
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
                    home_pitcher_display = _display_pitcher_name(home_pitcher_name)
                    away_pitcher_display = _display_pitcher_name(away_pitcher_name)
                    home_pitcher_valid = (
                        home_pitcher_display != "TBD"
                        and home_matched is not None
                        and pitcher_row_matches_team(home_matched, home_team)
                    )
                    away_pitcher_valid = (
                        away_pitcher_display != "TBD"
                        and away_matched is not None
                        and pitcher_row_matches_team(away_matched, away_team)
                    )
                    if home_team:
                        register_team_game_value(
                            team_pitcher_lookup,
                            home_team,
                            home_pitcher_display,
                            game_pk=game_pk,
                            game_time=game_time,
                        )
                    if away_team:
                        register_team_game_value(
                            team_pitcher_lookup,
                            away_team,
                            away_pitcher_display,
                            game_pk=game_pk,
                            game_time=game_time,
                        )
                    # Home batters face the AWAY pitcher
                    if away_team:
                        opp_info = {
                            "name": away_pitcher_display,
                            "hand": game.get("away_pitcher_hand", ""),
                            "id": game.get("away_pitcher_id"),
                        }
                        if away_pitcher_valid and away_matched is not None:
                            opp_info["profile"] = build_pitcher_profile(away_matched)
                        register_team_game_value(
                            opp_pitcher_lookup,
                            home_team,
                            opp_info,
                            game_pk=game_pk,
                            game_time=game_time,
                        )
                    # Away batters face the HOME pitcher
                    if home_team:
                        opp_info = {
                            "name": home_pitcher_display,
                            "hand": game.get("home_pitcher_hand", ""),
                            "id": game.get("home_pitcher_id"),
                        }
                        if home_pitcher_valid and home_matched is not None:
                            opp_info["profile"] = build_pitcher_profile(home_matched)
                        register_team_game_value(
                            opp_pitcher_lookup,
                            away_team,
                            opp_info,
                            game_pk=game_pk,
                            game_time=game_time,
                        )

                    # Build opposing team K rate for pitcher K projections
                    # Pitcher on home team faces away lineup, and vice versa
                    for pitcher_team, opp_batting_team in [(home_team, away_team), (away_team, home_team)]:
                        norm_opp_team = normalize_team_code(opp_batting_team)
                        if pitcher_team and norm_opp_team in team_k_rate_map:
                            register_team_game_value(
                                opp_team_k_lookup,
                                pitcher_team,
                                team_k_rate_map[norm_opp_team],
                                game_pk=game_pk,
                                game_time=game_time,
                            )
            except Exception:
                pass

            # Pre-build batter hand + batting order caches from confirmed lineups
            # Batch all game lineups upfront (cached per game_pk) to avoid N+1
            batter_hand_cache = {}        # player_name_upper → bat_hand (R/L/S)
            batting_order_cache = {}      # player_name_upper → batting_order (1-9)
            game_context_cache = {}       # team/game → {opponent, game_time, venue, ...}
            team_lineup_context_cache = {}  # team_abbr → confirmed lineup quality context
            try:
                for game in todays_games:
                    game_pk = game.get("game_pk")
                    if not game_pk:
                        continue
                    lineups = _cached_lineups(game_pk)
                    home_team = game.get("home_team", "")
                    away_team = game.get("away_team", "")

                    for side, team_abbr, opp_abbr in [
                        ("home", home_team, away_team),
                        ("away", away_team, home_team),
                    ]:
                        side_lineup = lineups.get(side, [])
                        for batter in side_lineup:
                            name = batter.get("player_name", "").upper().strip()
                            hand = batter.get("bat_hand", "")
                            order = batter.get("batting_order")
                            if name and hand:
                                batter_hand_cache[name] = hand
                            if name and order:
                                batting_order_cache[name] = order
                        if team_abbr:
                            register_team_value(
                                team_lineup_context_cache,
                                team_abbr,
                                build_team_lineup_context(side_lineup, batting_df, team_abbr),
                            )

                    # Build game context for both teams (no extra API call)
                    for team_abbr, opp_abbr, is_home in [
                        (home_team, away_team, True),
                        (away_team, home_team, False),
                    ]:
                        if team_abbr:
                            register_team_game_value(
                                game_context_cache,
                                team_abbr,
                                {
                                    "opponent": opp_abbr,
                                    "game_time": game.get("game_time", ""),
                                    "game_pk": game_pk,
                                    "is_home": is_home,
                                },
                                game_pk=game_pk,
                                game_time=game.get("game_time", ""),
                            )

                    for pitcher_team, opp_batting_team in [(home_team, away_team), (away_team, home_team)]:
                        if not pitcher_team or not opp_batting_team:
                            continue
                        opp_lineup_context = team_lineup_context_cache.get(resolve_team(opp_batting_team) or opp_batting_team)
                        if opp_lineup_context and opp_lineup_context.get("has_data"):
                            register_team_game_value(
                                opp_team_k_lookup,
                                pitcher_team,
                                opp_lineup_context.get("top6_k_rate") or opp_lineup_context.get("avg_k_rate"),
                                game_pk=game_pk,
                                game_time=game.get("game_time", ""),
                            )
            except Exception:
                pass

            prog = st.progress(0, text="Running projections...")
            total = len(pp_lines)
            _pred_errors = 0
            active_weights = load_current_weights()
            for i, (_, row) in enumerate(pp_lines.iterrows()):
              try:
                prog.progress((i + 1) / total, text=f"Projecting {i + 1}/{total}...")
                team = row.get("team","")
                stat_int = row.get("stat_internal", "")
                wx = None
                if team:
                    r = resolve_team(team)
                    if r in weather_cache:
                        wx = weather_cache[r]

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

                # Look up home-plate umpire for this player's game
                ump_name = None
                r_team = resolve_team(team) if team else None
                if r_team and umpire_map:
                    ump_name = umpire_map.get(r_team)
                ump_adj = get_umpire_k_adjustment(ump_name) if ump_name else None

                # Look up batting order position from pre-built cache (no API call)
                batting_pos = None
                batter_lineup_context = None
                opp_lineup_context = None
                if not is_pitcher_prop:
                    batting_pos = batting_order_cache.get(
                        row["player_name"].upper().strip()
                    )
                    team_lineup_context = team_lineup_context_cache.get(r_team) if r_team else None
                    if team_lineup_context:
                        batter_lineup_context = build_player_lineup_context(row["player_name"], team_lineup_context)
                        if batting_pos is None and batter_lineup_context.get("batting_order"):
                            batting_pos = batter_lineup_context["batting_order"]

                # Look up opposing pitcher profile (for batter props)
                opp_pitcher_profile = None
                platoon_adj = None
                lookup_game_pk = row.get("game_pk")
                lookup_game_time = row.get("start_time") or row.get("game_time_utc")
                if not is_pitcher_prop and r_team:
                    opp_info = get_team_game_value(
                        opp_pitcher_lookup,
                        r_team,
                        game_pk=lookup_game_pk,
                        game_time=lookup_game_time,
                    )
                else:
                    opp_info = None
                if opp_info:
                    opp_pitcher_profile = opp_info.get("profile")
                    # Platoon splits: need batter hand + pitcher hand
                    opp_hand = opp_info.get("hand", "")
                    if opp_hand:
                        bat_hand = batter_hand_cache.get(
                            row["player_name"].upper().strip(), ""
                        )
                        if bat_hand:
                            platoon_adj = get_platoon_split_adjustment(bat_hand, opp_hand)

                # Opposing team K rate (for pitcher K projections)
                opp_k_rate = None
                if is_pitcher_prop and r_team:
                    opp_k_rate = get_team_game_value(
                        opp_team_k_lookup,
                        r_team,
                        game_pk=lookup_game_pk,
                        game_time=lookup_game_time,
                    )
                    gctx = get_team_game_value(
                        game_context_cache,
                        r_team,
                        game_pk=lookup_game_pk,
                        game_time=lookup_game_time,
                    ) or {}
                    opp_team = gctx.get("opponent")
                    if opp_team:
                        opp_lineup_context = team_lineup_context_cache.get(resolve_team(opp_team) or opp_team)
                        if opp_lineup_context and opp_lineup_context.get("has_data"):
                            opp_k_rate = opp_lineup_context.get("top6_k_rate") or opp_lineup_context.get("avg_k_rate") or opp_k_rate

                # BvP matchup (batter props only)
                bvp_data = None
                if not is_pitcher_prop and opp_info and opp_info.get("id"):
                    try:
                        batter_lookup = lookup_player_id(row["player_name"])
                        if batter_lookup.get("found") and batter_lookup.get("mlbam_id"):
                            bvp_data = get_bvp_matchup(
                                batter_lookup["mlbam_id"],
                                int(opp_info["id"]),
                            )
                            if not bvp_data.get("has_data"):
                                bvp_data = None
                    except Exception:
                        bvp_data = None

                # ── LINE SANITY CHECK ────────────────────────────
                # Skip props with unrealistically low lines (spring training / promo artifacts).
                # These create fake massive edges when real projections are compared to garbage lines.
                _min_line = MIN_REALISTIC_LINE.get(stat_int, 0)
                if _min_line and float(row.get("line", 0)) < _min_line:
                    continue

                # Resolve Vegas game total for this player's game
                _vgt = None
                if _game_totals_by_team and team:
                    _team_lower = team.lower()
                    _vgt = _game_totals_by_team.get(_team_lower)
                    if _vgt is None:
                        # Fuzzy match: PP team name may be abbreviation or partial
                        for _gt_team, _gt_val in _game_totals_by_team.items():
                            if _team_lower in _gt_team or _gt_team in _team_lower:
                                _vgt = _gt_val
                                break

                p = generate_prediction(
                    player_name=row["player_name"],
                    stat_type=row["stat_type"],
                    stat_internal=stat_int,
                    line=row["line"],
                    batter_profile=batter_profile,
                    pitcher_profile=pitcher_profile,
                    opp_pitcher_profile=opp_pitcher_profile,
                    opp_team_k_rate=opp_k_rate,
                    bvp=bvp_data,
                    platoon=platoon_adj,
                    park_team=r_team,
                    weather=wx,
                    ump=ump_adj,
                    lineup_pos=batting_pos,
                    batter_lineup_context=batter_lineup_context,
                    opp_lineup_context=opp_lineup_context,
                    game_date=_game_date_from_iso(row.get("start_time", "")),
                    vegas_game_total=_vgt,
                )
                p["game_date"] = _game_date_from_iso(row.get("start_time", ""))
                p["game_time_utc"] = row.get("start_time", "")
                if _vgt:
                    p["vegas_game_total"] = _vgt

                # Props that are count-based (safe to apply multipliers to)
                # home_runs now returns expected count — include in COUNT_PROPS for spring/trend multipliers
                _COUNT_PROPS = {"hits", "total_bases", "rbis", "runs", "stolen_bases",
                                "hits_runs_rbis", "batter_strikeouts", "walks", "singles", "doubles",
                                "pitcher_strikeouts", "pitching_outs", "earned_runs",
                                "walks_allowed", "hits_allowed", "hitter_fantasy_score", "home_runs"}
                _is_count_prop = stat_int in _COUNT_PROPS

                # v018: Stat-specific weather adjustment (research-backed)
                # SKIP: weather is already applied inside projection functions.
                # Applying it again here would double-count. Only record it for display.
                if wx:
                    wx_mult = get_stat_specific_weather_adjustment(wx, stat_int)
                    p["weather_mult"] = wx_mult  # For display only, NOT applied to projection

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
                season_sample = 0.0
                sample_key = "IP" if is_pitcher_prop else "PA"
                if matched is not None and sample_key in matched.index:
                    try:
                        season_sample = float(matched.get(sample_key, 0) or 0)
                    except (TypeError, ValueError):
                        season_sample = 0.0
                spring_mult = apply_seasonal_spring_blend(
                    spring["spring_mult"],
                    game_date=p["game_date"],
                    current_sample=season_sample,
                    is_pitcher=is_pitcher_prop,
                    prop_type=stat_int,
                    config=active_weights.get("seasonal_spring_blend", {}),
                )
                # Only apply spring multiplier to count-based props (not probabilities)
                if _is_count_prop:
                    p["projection"] = round(p["projection"] * spring_mult, 2)
                p["spring_mult"] = spring_mult
                p["spring_badge"] = spring["badge"]

                trend = get_batter_trend(row["player_name"]) if not is_pitcher_prop else {"trend_multiplier": 1.0}
                trend_mult = trend.get("trend_multiplier", 1.0)
                trend_mult = max(0.92, min(1.08, trend_mult))
                # Only apply trend multiplier to count-based props
                if _is_count_prop:
                    p["projection"] = round(p["projection"] * trend_mult, 2)
                p["trend_mult"] = trend_mult
                if trend_mult >= 1.03:
                    p["trend_badge"] = "hot"
                elif trend_mult <= 0.97:
                    p["trend_badge"] = "cold"
                else:
                    p["trend_badge"] = "neutral"

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
                        # Only apply buy_low to count props
                        if _is_count_prop:
                            p["projection"] = round(p["projection"] * 1.04, 2)

                # Spring/trend/buy-low modifiers change the projection itself, so the
                # displayed probabilities and pick direction must be refreshed too.
                _refreshed_prob = calculate_over_under_probability(
                    p["projection"],
                    p["line"],
                    stat_int,
                    proj_result=p,
                )
                p.update(_refreshed_prob)

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

                # ── DATA QUALITY GATE ────────────────────────
                # Skip players with no player-specific data entirely.
                # These are pure league-average projections with zero edge.
                if not p.get("has_player_data", True):
                    continue  # Don't show picks where we have no data

                # Also skip if ALL context is missing — zero informational value.
                # NOTE: Pitcher props don't have batting_pos, so we only check
                # opponent and park for pitchers. Batters need 2+ of 3 context fields.
                if is_pitcher_prop:
                    # Pitchers need at least a resolved team (for park factor)
                    if r_team is None:
                        continue
                else:
                    _opp_lookup = get_team_game_value(
                        opp_pitcher_lookup,
                        r_team,
                        game_pk=lookup_game_pk,
                        game_time=lookup_game_time,
                    ) if r_team else None
                    _no_opp = not p.get("has_opp_data") and not _opp_lookup
                    _no_lineup = batting_pos is None
                    _no_park = r_team is None
                    if _no_opp and _no_lineup and _no_park:
                        continue

                # v018: Lineup position display info
                # NOTE: PA adjustment now handled INSIDE generate_prediction() via
                # lineup_pos → estimate_plate_appearances() (Task 3A). No post-hoc
                # multiplier needed — that would double-count the effect.
                if batting_pos:
                    p["batting_order"] = batting_pos
                    p["pa_multiplier"] = round(get_pa_multiplier(batting_pos), 3)

                # v018: Game context from pre-built cache (no API call)
                gctx = get_team_game_value(
                    game_context_cache,
                    r_team,
                    game_pk=lookup_game_pk,
                    game_time=lookup_game_time,
                ) if r_team else None
                if gctx:
                    raw_game_time = gctx.get("game_time", "")
                    p["opponent"] = gctx.get("opponent", "")
                    p["game_time"] = _utc_to_pst(raw_game_time)
                    p["game_time_utc"] = raw_game_time
                    p["game_date"] = _game_date_from_iso(raw_game_time)
                    p["game_pk"] = gctx.get("game_pk")
                    p["is_home"] = gctx.get("is_home", False)
                    p["team_pitcher"] = _display_pitcher_name(
                        get_team_game_value(
                            team_pitcher_lookup,
                            r_team,
                            game_pk=gctx.get("game_pk"),
                            game_time=raw_game_time,
                        )
                    )
                    opp_info = get_team_game_value(
                        opp_pitcher_lookup,
                        r_team,
                        game_pk=gctx.get("game_pk"),
                        game_time=raw_game_time,
                    )
                    if opp_info:
                        p["opp_pitcher"] = _display_pitcher_name(opp_info.get("name", ""))

                # Add platoon and matchup context to result for display
                if platoon_adj and platoon_adj.get("favorable") is not None:
                    p["platoon"] = platoon_adj["description"]
                    p["platoon_favorable"] = platoon_adj["favorable"]
                if ump_name:
                    p["umpire"] = ump_name

                p["team"] = team
                p["stat_internal"] = stat_int
                # v018: Pass through PrizePicks line type (standard/promo/goblin/demon)
                p["line_type"] = row.get("line_type", "standard")
                p["odds_type"] = row.get("odds_type", "standard")

                # ── EDGE CAP ─────────────────────────────────────
                # Any edge above MAX_EDGE_PCT is almost certainly a data artifact
                # (garbage line, spring training, promo line, etc.)
                # Cap the edge and downgrade confidence accordingly.
                _raw_edge = abs(_safe_num(p.get("edge"), 0))
                if _raw_edge > MAX_EDGE_PCT:
                    # Clamp confidence to reflect that we don't trust this edge
                    p["confidence"] = min(p.get("confidence", 0.5), 0.65)
                    _sync_pick_metrics(p)
                    p["edge_capped"] = True

                annotate_prediction_floor(p, active_weights)

                preds.append(p)
              except Exception as _pred_exc:
                _pred_errors += 1
                if _pred_errors <= 3:
                    import logging; logging.getLogger("app").warning(
                        "Prediction error for %s/%s: %s",
                        row.get("player_name","?"), row.get("stat_type","?"), _pred_exc
                    )
            prog.empty()
            if _pred_errors:
                st.caption(f"⚠️ {_pred_errors} prop(s) skipped due to data issues")

            # v018: Cross-prop consistency checks (TB >= Hits, etc.)
                if preds:
                    try:
                        preds = enforce_consistency(preds)
                    except Exception as e:
                        _log.warning("Consistency check failed: %s", e)
                    preds = dedupe_predictions(preds)

            if preds:
                # v018: Save projected stats for tracking accuracy over time
                try:
                    stats_to_save = [{
                        "game_date": p.get("game_date", date.today().isoformat()),
                        "player_name": p["player_name"],
                        "team": p.get("team", ""),
                        "stat_type": p.get("stat_internal", ""),
                        "projected_value": p["projection"],
                        "line": p.get("line", 0),
                        "pick": p.get("pick", ""),
                        "confidence": p.get("confidence", 0),
                        "rating": p.get("rating", ""),
                    } for p in preds]
                    save_projected_stats(stats_to_save)
                except Exception as e:
                    _log.warning("Failed to save projected stats: %s", e)

                # v018: Log full board snapshot (all props, not just selected)
                try:
                    log_board_snapshot(preds, edges=all_edges)
                except Exception as e:
                    _log.warning("Failed to log board snapshot: %s", e)
                shadow_sample_results = []
                try:
                    for _game_date in sorted({
                        p.get("game_date", date.today().isoformat())
                        for p in preds
                        if p.get("game_date")
                    }):
                        shadow_sample_results.append(
                            ensure_shadow_sample(_game_date, sample_size=50)
                        )
                except Exception:
                    shadow_sample_results = []

                pdf = pd.DataFrame(preds).sort_values("confidence", ascending=False)

                scored_all = score_picks(all_edges, preds)
                ab_combined = [s for s in scored_all if s["combined_grade"] in ("A+", "A", "B")]
                confirmed_count = sum(1 for s in ab_combined if s["signal"] == SIGNAL_CONFIRMED)
                a_proj = sum(1 for p in preds if p.get("rating") == "A")
                ab_proj = sum(1 for p in preds if p.get("rating") in ("A","B"))
                qs1, qs2, qs3, qs4 = st.columns(4)
                with qs1: st.markdown(f'<div class="card card-b"><div class="lbl">Props Available</div><div class="val b">{len(pp_lines)}</div><div class="sub">On PrizePicks</div></div>', unsafe_allow_html=True)
                with qs2: st.markdown(f'<div class="card card-g"><div class="lbl">A-Grade Picks</div><div class="val g">{a_proj}</div><div class="sub">Highest confidence</div></div>', unsafe_allow_html=True)
                with qs3: st.markdown(f'<div class="card"><div class="lbl">A+B Picks</div><div class="val">{ab_proj}</div><div class="sub">Strong signal</div></div>', unsafe_allow_html=True)
                with qs4:
                    confirmed_str = str(confirmed_count) if confirmed_count else "—"
                    st.markdown(f'<div class="card {"card-g" if confirmed_count else ""}"><div class="lbl">Sharp Confirmed</div><div class="val {"g" if confirmed_count else ""}">{confirmed_str}</div><div class="sub">Both sources agree</div></div>', unsafe_allow_html=True)
                if all_edges and confirmed_count:
                    st.markdown(f'<div class="info-strip"><span class="hl">{confirmed_count}</span> picks confirmed by both sharp books and projection model</div>', unsafe_allow_html=True)
                elif not all_edges:
                    st.markdown('<div class="warn-strip"><strong>No sharp lines</strong> — showing projection-only analysis. Add Odds API key for full edge detection.</div>', unsafe_allow_html=True)

                if shadow_sample_results:
                    _shadow_locked = sum(r.get("selected_now", 0) for r in shadow_sample_results)
                    _shadow_total = sum(r.get("shadow_sample_size", 0) for r in shadow_sample_results)
                    st.caption(
                        f"Shadow QA sample locked: {_shadow_total} props across {len(shadow_sample_results)} game date(s)"
                        + (f" ({_shadow_locked} newly selected this run)." if _shadow_locked else ".")
                    )

                top_plays = _derive_top_plays(scored_all, pdf)
                _render_top_plays(top_plays)

                if selected_player_name:
                    raw_mask = all_pp_lines["player_name"].eq(selected_player_name)
                    if selected_player_team:
                        raw_mask &= all_pp_lines["team"].fillna("").eq(selected_player_team)
                    player_raw = all_pp_lines.loc[raw_mask].copy()

                    model_mask = pdf["player_name"].eq(selected_player_name)
                    if selected_player_team:
                        model_mask &= pdf["team"].fillna("").eq(selected_player_team)
                    player_model = pdf.loc[model_mask].copy()

                    st.markdown(f'<div class="section-hdr">Player Search - {selected_player_label}</div>', unsafe_allow_html=True)
                    if not player_raw.empty:
                        raw_disp = player_raw[[
                            "stat_type", "line", "odds_type", "league",
                            "start_time", "eligibility_reason",
                        ]].copy()
                        raw_disp["start_time"] = raw_disp["start_time"].apply(
                            lambda x: _utc_to_pst(x) if pd.notna(x) and str(x) else "-"
                        )
                        raw_disp["odds_type"] = raw_disp["odds_type"].fillna("standard").astype(str).str.replace("_", " ").str.title()
                        raw_disp["eligibility_reason"] = raw_disp["eligibility_reason"].map(PLAYER_STATUS_LABELS).fillna("Excluded")
                        raw_disp.columns = ["Prop", "Line", "Odds Type", "League", "Start", "Model Status"]
                        st.dataframe(raw_disp, hide_index=True, width="stretch")
                    if not player_model.empty:
                        model_disp = player_model[[
                            "stat_type", "line", "projection", "pick", "confidence", "rating"
                        ]].copy()
                        model_disp["projection"] = model_disp["projection"].apply(lambda x: round(_safe_num(x, 0.0), 2))
                        model_disp["confidence"] = model_disp["confidence"].apply(lambda x: f"{_safe_num(x, 0.0) * 100:.1f}%")
                        model_disp.columns = ["Projected Prop", "Line", "Projection", "Pick", "Confidence", "Grade"]
                        st.caption("Model-eligible game props for this player")
                        st.dataframe(model_disp, hide_index=True, width="stretch")
                    else:
                        st.info(
                            f"{selected_player_label} has no model-eligible standard game props on the current board. "
                            "If you still see rows above, they are season-long or non-standard PrizePicks lines."
                        )

                # ── GAME SCORE DIAGNOSTIC ────────────────────────
                # Aggregate per-team projected runs and hits for sanity check.
                # Sources: standalone runs/hits props + HRR combo breakdowns.
                with st.expander("Game Score Projections", expanded=False):
                    def _estimate_team_runs(team_bucket):
                        if not team_bucket.get("runs"):
                            return None
                        avg_runs = sum(team_bucket["runs"]) / len(team_bucket["runs"])
                        return round(avg_runs * 9, 1)

                    def _estimate_team_hits(team_bucket):
                        if not team_bucket.get("hits"):
                            return None
                        avg_hits = sum(team_bucket["hits"]) / len(team_bucket["hits"])
                        return round(avg_hits * 9, 1)

                    _game_scores = {}
                    for p in preds:
                        _team = p.get("team", "")
                        _stat = p.get("stat_internal", "")
                        _proj = _safe_num(p.get("projection"), 0)
                        if not _team:
                            continue
                        _team_game_key = p.get("game_pk") or f"{_team}|{_safe(p.get('game_time'), '')}|{_safe(p.get('opponent'), '')}"
                        if _team_game_key not in _game_scores:
                            _r_team = resolve_team(_team) if _team else None
                            _pf = PARK_FACTORS.get(_r_team, 100) if _r_team else 100
                            _game_scores[_team_game_key] = {
                                "team": _team,
                                "runs": [],
                                "hits": [],
                                "hr_proj": [],
                                "players": set(),
                                "opponent": _safe(p.get("opponent"), ""),
                                "opp_pitcher": _display_pitcher_name(p.get("opp_pitcher", "")),
                                "team_pitcher": _display_pitcher_name(p.get("team_pitcher", "")),
                                "game_time": _safe(p.get("game_time"), ""),
                                "park_factor": _pf,
                                "game_pk": p.get("game_pk"),
                                "is_home": bool(p.get("is_home", False)),
                            }
                        _bucket = _game_scores[_team_game_key]
                        _bucket["players"].add(p.get("player_name", ""))
                        if not _bucket.get("opponent"):
                            _bucket["opponent"] = _safe(p.get("opponent"), "")
                        if not _bucket.get("opp_pitcher"):
                            _bucket["opp_pitcher"] = _display_pitcher_name(p.get("opp_pitcher", ""))
                        if not _bucket.get("team_pitcher"):
                            _bucket["team_pitcher"] = _display_pitcher_name(p.get("team_pitcher", ""))
                        if not _bucket.get("game_time"):
                            _bucket["game_time"] = _safe(p.get("game_time"), "")
                        if not _bucket.get("game_pk") and p.get("game_pk"):
                            _bucket["game_pk"] = p.get("game_pk")
                        if p.get("is_home"):
                            _bucket["is_home"] = True

                        if _stat == "runs":
                            _bucket["runs"].append(_proj)
                        elif _stat == "hits":
                            _bucket["hits"].append(_proj)
                        elif _stat == "home_runs":
                            _bucket["hr_proj"].append(_proj)
                        elif _stat == "hits_runs_rbis":
                            _h = _safe_num(p.get("hits_proj"), 0)
                            _r = _safe_num(p.get("runs_proj"), 0)
                            if _h > 0:
                                _bucket["hits"].append(_h)
                            if _r > 0:
                                _bucket["runs"].append(_r)

                    if _game_scores:
                        _team_rows = []
                        _matchup_buckets = {}
                        for _gs in sorted(_game_scores.values(), key=lambda item: (item.get("game_time", ""), item.get("team", ""))):
                            _tm = _gs.get("team", "")
                            _est_team_runs = _estimate_team_runs(_gs)
                            _est_team_hits = _estimate_team_hits(_gs)
                            _avg_hr = sum(_gs["hr_proj"]) / len(_gs["hr_proj"]) if _gs["hr_proj"] else None
                            _n_players = len(_gs["players"] - {""})
                            _pf_val = _gs.get("park_factor", 100)
                            _pf_str = f"{_pf_val}" if _pf_val != 100 else "100"
                            _team_rows.append({
                                "Team": _tm,
                                "Facing": _gs.get("opponent", ""),
                                "Opp SP Faced": _display_pitcher_name(_gs.get("opp_pitcher", "")),
                                "Own SP": _display_pitcher_name(_gs.get("team_pitcher", "")),
                                "Game": _gs.get("game_time", "") or "-",
                                "Park": _pf_str,
                                "Est Runs": _est_team_runs,
                                "Est Hits": _est_team_hits,
                                "Avg HR Proj": _avg_hr,
                                "# Players": _n_players if _n_players else None,
                            })

                            _opp = _gs.get("opponent", "")
                            _game_key = _gs.get("game_pk") or f"{'/'.join(sorted([_tm, _opp]))}|{_gs.get('game_time', '')}"
                            _matchup_buckets.setdefault(_game_key, []).append({
                                "team": _tm,
                                "opponent": _opp,
                                "team_pitcher": _display_pitcher_name(_gs.get("team_pitcher", "")),
                                "game_time": _gs.get("game_time", "") or "-",
                                "is_home": bool(_gs.get("is_home", False)),
                                "est_runs": _est_team_runs,
                            })

                        _matchup_rows = []
                        for _entries in _matchup_buckets.values():
                            if len(_entries) < 2:
                                continue
                            _away = next((item for item in _entries if not item.get("is_home")), None)
                            _home = next((item for item in _entries if item.get("is_home")), None)
                            if _away is None or _home is None:
                                _sorted_entries = sorted(_entries, key=lambda item: item["team"])
                                _away = _away or _sorted_entries[0]
                                _home = _home or _sorted_entries[-1]
                                if _away["team"] == _home["team"] and len(_sorted_entries) > 1:
                                    _home = _sorted_entries[1]
                            if _away["team"] == _home["team"]:
                                continue

                            _away_runs = _away.get("est_runs")
                            _home_runs = _home.get("est_runs")
                            _score_text = "-"
                            _winner = "-"
                            _margin = "-"
                            if _has_real_number(_away_runs) and _has_real_number(_home_runs):
                                _away_score = int(round(_away_runs))
                                _home_score = int(round(_home_runs))
                                _score_text = f"{_away['team']} {_away_score} - {_home_score} {_home['team']}"
                                if _away_score > _home_score:
                                    _winner = _away["team"]
                                elif _home_score > _away_score:
                                    _winner = _home["team"]
                                else:
                                    _winner = "Toss-up"
                                _margin = f"{abs(_away_runs - _home_runs):.1f}"

                            _matchup_rows.append({
                                "Game": f"{_away['team']} at {_home['team']}",
                                "Start": _home.get("game_time") or _away.get("game_time") or "-",
                                "Away SP": _display_pitcher_name(_away.get("team_pitcher", "")),
                                "Home SP": _display_pitcher_name(_home.get("team_pitcher", "")),
                                "Projected Score": _score_text,
                                "Projected Winner": _winner,
                                "Margin": _margin,
                            })

                        if _matchup_rows:
                            _matchup_df = pd.DataFrame(_matchup_rows).sort_values(["Start", "Game"])
                            st.caption("Projected scores are rounded to whole runs. If the rounded score is tied, the winner stays Toss-up.")
                            st.dataframe(_matchup_df, hide_index=True, width="stretch")
                        else:
                            st.caption("Projected score rows need both teams in a game to have enough source props on the current board.")

                        st.caption("Team offense view: 'Opp SP Faced' is the opposing starter that team's hitters are facing. The earlier Paul Skenes example for NYM was labeled ambiguously, not reversed.")
                        _team_df = pd.DataFrame(_team_rows)
                        _team_display = _team_df.copy()
                        for _col in ("Est Runs", "Est Hits"):
                            _team_display[_col] = _team_display[_col].apply(
                                lambda x: f"{x:.1f}" if _has_real_number(x) else "-"
                            )
                        _team_display["Avg HR Proj"] = _team_display["Avg HR Proj"].apply(
                            lambda x: f"{x:.2f}" if _has_real_number(x) else "-"
                        )
                        _team_display["# Players"] = _team_display["# Players"].apply(
                            lambda x: str(int(x)) if _has_real_number(x) else "-"
                        )
                        st.dataframe(_team_display, hide_index=True, width="stretch")
                        st.caption("- means the current slate has no matching run/hit source props for that team, not a literal zero projection.")

                        _run_vals = [
                            row["Est Runs"] for row in _team_rows
                            if _has_real_number(row["Est Runs"]) and row["Est Runs"] > 0
                        ]
                        if len(_run_vals) >= 4:
                            _run_mean = sum(_run_vals) / len(_run_vals)
                            _run_std = (sum((x - _run_mean) ** 2 for x in _run_vals) / len(_run_vals)) ** 0.5
                            if _run_std < 0.5:
                                st.warning(f"Low variance: all teams project {_run_mean:.1f} +/- {_run_std:.1f} runs. Model may not be differentiating matchups well.")
                            else:
                                st.caption(
                                    f"Run spread: {min(_run_vals):.1f} - {max(_run_vals):.1f} (sd={_run_std:.1f}). "
                                    f"{'Good differentiation.' if _run_std > 1.0 else 'Moderate differentiation.'}"
                                )
                    else:
                        st.caption("No per-team projections available yet.")

                st.markdown('<div class="section-hdr">Filter Picks</div>', unsafe_allow_html=True)
                prop_types_available = sorted(pdf["stat_type"].unique().tolist())
                f1, f2, f3 = st.columns([3, 2, 2])
                with f1:
                    proj_prop_filter = st.radio("Prop Type", ["All"] + prop_types_available, horizontal=True, key="proj_prop_f")
                with f2:
                    proj_grade_filter = st.radio("Min Grade", ["A only", "A+B", "A+B+C", "All"], index=1, key="proj_grade_f")
                _grade_map_radio = {"A only": ["A"], "A+B": ["A","B"], "A+B+C": ["A","B","C"], "All": ["A","B","C","D"]}
                with f3:
                    show_all_proj = st.checkbox("Show all picks (incl. non-tradeable)", value=False, key="show_all_proj")

                filtered = pdf[pdf["rating"].isin(_grade_map_radio[proj_grade_filter])].copy()
                if proj_prop_filter != "All":
                    filtered = filtered[filtered["stat_type"] == proj_prop_filter]
                if not show_all_proj:
                    _before_filter = len(filtered)
                    filtered = filtered[filtered.apply(
                        lambda r: is_tradeable_pick(r.get("stat_internal", ""), r.get("pick", "")), axis=1
                    )].copy()
                    _removed = _before_filter - len(filtered)
                    if _removed > 0:
                        st.info(f"Showing tradeable picks only. Filtered out {_removed} pick(s)")

                    _before_floor = len(filtered)
                    filtered = filtered[filtered["meets_conf_floor"].fillna(False)].copy()
                    _removed_floor = _before_floor - len(filtered)
                    if _removed_floor > 0:
                        st.info(f"Applied model confidence floors. Filtered out {_removed_floor} pick(s) below tuned thresholds.")

                slip_candidates = filtered.head(40).reset_index(drop=True)
                selected_picks = []

                if not slip_candidates.empty:
                    slip_candidates = slip_candidates.sort_values(
                        ["stat_type", "confidence"], ascending=[True, False]
                    ).reset_index(drop=True)

                _prev_stat_type = None
                for pick_idx, (_, pick_row) in enumerate(slip_candidates.iterrows()):
                    if pick_row["stat_type"] != _prev_stat_type:
                        if _prev_stat_type is not None:
                            st.markdown('<hr style="margin:0.3rem 0;border:none;border-top:1px solid rgba(255,255,255,0.08);">', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:0.7rem;color:rgba(232,236,241,0.4);letter-spacing:1px;text-transform:uppercase;padding:0.3rem 0 0.1rem 0;margin-top:0.8rem;">{pick_row["stat_type"]}</div>', unsafe_allow_html=True)
                        _prev_stat_type = pick_row["stat_type"]
                    if pick_idx >= 40:
                        break
                    health_icon = {"IL": "🔴", "day-to-day": "🟡", "active": "🟢"}.get(pick_row.get("injury_status", "active"), "🟢")
                    form_icon = {"hot": "🔥", "cold": "❄️", "neutral": "—"}.get(pick_row.get("spring_badge", "neutral"), "—")
                    trend_icon = {"hot": "🔥", "cold": "❄️", "neutral": "—"}.get(pick_row.get("trend_badge", "neutral"), "—")
                    buy_tag = " 🎯" if pick_row.get("buy_low") else ""
                    _lt = pick_row.get("line_type", "standard")
                    promo_tag = " 👺" if _lt == "promo" else (" 💰" if _lt in ("discounted", "flash_sale") else "")
                    pick_cls = "more-pick" if pick_row["pick"] == "MORE" else "less-pick"

                    chk_col, info_col = st.columns([0.06, 0.94])
                    with chk_col:
                        checked = st.checkbox("Select pick", key=f"proj_pick_{pick_idx}", label_visibility="collapsed")
                    with info_col:
                        conf_val = _safe_num(pick_row.get('confidence'), 0.5)
                        conf_pct = int(conf_val * 100)
                        win_prob_edge_val = _safe_num(pick_row.get('edge'), 0) * 100
                        win_prob_edge_pct = f"{win_prob_edge_val:.1f}%"
                        proj = _safe_num(pick_row.get('projection'), 0)
                        line_val = _safe_num(pick_row.get('line'), 0)
                        proj_delta = proj - line_val
                        proj_delta_str = f"{proj_delta:+.2f}"
                        proj_delta_color = "#00C853" if (proj_delta > 0 and pick_row["pick"] == "MORE") or (proj_delta < 0 and pick_row["pick"] == "LESS") else "#FFB300"
                        breakout_watch = _safe(pick_row.get("breakout_watch"), "Low")
                        dud_risk = _safe(pick_row.get("dud_risk"), "Low")
                        _tail_labels = tail_signal_labels(_safe(pick_row.get("stat_internal"), ""))
                        tail_badges = []
                        if breakout_watch in ("Medium", "High"):
                            tail_color = "#00C853" if breakout_watch == "High" else "#29B6F6"
                            tail_badges.append(
                                f'<span style="font-size:0.64rem;padding:0.12rem 0.35rem;border-radius:999px;'
                                f'background:rgba(0,200,83,0.10);color:{tail_color};border:1px solid rgba(0,200,83,0.20);">'
                                f'{_tail_labels["breakout"]} {breakout_watch}</span>'
                            )
                        if dud_risk in ("Medium", "High"):
                            dud_color = "#FFB300" if dud_risk == "Medium" else "#FF5252"
                            tail_badges.append(
                                f'<span style="font-size:0.64rem;padding:0.12rem 0.35rem;border-radius:999px;'
                                f'background:rgba(255,82,82,0.10);color:{dud_color};border:1px solid rgba(255,82,82,0.18);">'
                                f'{_tail_labels["dud"]} {dud_risk}</span>'
                            )
                        tail_badges_html = (
                            f'<div style="display:flex;gap:0.35rem;flex-wrap:wrap;margin-top:0.45rem;">{"".join(tail_badges)}</div>'
                            if tail_badges else ""
                        )

                        _conf_fill_cls = "high" if conf_val > 0.6 else ("med" if conf_val > 0.52 else "low")
                        pick_card_html = (
                            f'<div class="pick-card {pick_cls}">'
                            f'<div class="pick-card-header">'
                            f'<span class="badge badge-{_safe(pick_row.get("rating"), "D").lower()}">{_safe(pick_row.get("rating"), "D")}</span>'
                            f'<span class="pick-card-player">{_safe(pick_row.get("player_name"), "Unknown")}{promo_tag}</span>'
                            f'<span class="pick-card-team">{_safe(pick_row.get("team"), "")}</span>'
                            f'<span class="dir-chip {"more" if pick_row["pick"] == "MORE" else "less"}">{pick_row["pick"]}</span>'
                            f'</div>'
                            f'<div class="pick-card-row">'
                            f'<span class="pick-card-stat">{_safe(pick_row.get("stat_type"), "Unknown")}</span>'
                            f'<span class="pick-card-line">Line {line_val}</span>'
                            f'<span class="pick-card-proj">Proj {proj:.2f}</span>'
                            f'<span class="pick-card-delta" style="color:{proj_delta_color};">Δ {proj_delta_str}</span>'
                            f'<span class="pick-card-edge" style="color:#29B6F6;">WP +{win_prob_edge_pct}</span>'
                            f'</div>'
                            f'<div class="pick-card-conf">'
                            f'<div class="conf-track" style="flex:1"><div class="conf-fill {_conf_fill_cls}" style="width:{min(conf_pct,100)}%"></div></div>'
                            f'<span class="pick-card-conf-label">{conf_pct}%</span>'
                            f'</div>'
                            f'{tail_badges_html}'
                            f'</div>'
                        )
                        st.markdown(pick_card_html, unsafe_allow_html=True)

                        # Expandable detail: projected statline + game context
                        with st.expander("Details", expanded=False):
                            det1, det2 = st.columns(2)
                            with det1:
                                st.markdown("**Game Context**")
                                _opp = _safe(pick_row.get("opponent"))
                                _venue = _safe(pick_row.get("venue"))
                                _opp_p = _safe(pick_row.get("opp_pitcher"))
                                _bat_ord = pick_row.get("batting_order")
                                if _bat_ord is not None:
                                    try:
                                        _bat_ord = int(_bat_ord)
                                    except (ValueError, TypeError):
                                        _bat_ord = None
                                _gt = _safe(pick_row.get("game_time"), "")
                                ctx_lines = []
                                if _opp != "—":
                                    _opp_str = f"**vs** {_opp}"
                                    if _gt and _gt != "—":
                                        _opp_str += f"  ·  {_gt}"
                                    ctx_lines.append(_opp_str)
                                if _opp_p != "—":
                                    ctx_lines.append(f"**Opp Pitcher:** {_opp_p}")
                                if _bat_ord:
                                    ctx_lines.append(f"**Batting:** {_bat_ord}{'st' if _bat_ord==1 else 'nd' if _bat_ord==2 else 'rd' if _bat_ord==3 else 'th'} in order")
                                if _venue != "—":
                                    ctx_lines.append(f"**Park:** {_venue}")
                                # Park factor
                                _pk_team_raw = _safe(pick_row.get("team"), "")
                                _pk_team = resolve_team(_pk_team_raw) if _pk_team_raw else None
                                if _pk_team and _pk_team in PARK_FACTORS:
                                    _pf = PARK_FACTORS[_pk_team] - 100
                                    if _pf != 0:
                                        _pf_label = "hitter-friendly" if _pf > 0 else "pitcher-friendly"
                                        ctx_lines.append(f"**Park Factor:** {_pf:+d}% ({_pf_label})")
                                st.markdown("  \n".join(ctx_lines) if ctx_lines else "No game context available")

                            with det2:
                                st.markdown("**Projected Statline**")
                                _si = pick_row.get("stat_internal", "")
                                _is_pitcher = pick_row.get("is_pitcher_prop", False) or _si in ("pitcher_strikeouts", "earned_runs", "hits_allowed", "walks_allowed", "pitching_outs")
                                proj_val = _safe_num(pick_row.get("projection"), 0)
                                line_val = _safe_num(pick_row.get("line"), 0)
                                diff = proj_val - line_val
                                diff_color = "#00C853" if (diff > 0 and pick_row["pick"] == "MORE") or (diff < 0 and pick_row["pick"] == "LESS") else "#FF4444"
                                st.markdown(f'**{pick_row["stat_type"]}:** <span style="font-family:JetBrains Mono;font-weight:700;color:{diff_color}">{proj_val:.2f}</span> vs line {line_val}  ({"+" if diff>=0 else ""}{diff:.2f})', unsafe_allow_html=True)
                                st.markdown(f"**Confidence:** {conf_pct}% | **Win Prob Edge:** +{win_prob_edge_pct} | **Projection Delta:** {proj_delta_str}")
                                p10_val = _safe_num(pick_row.get("p10"), proj_val)
                                p50_val = _safe_num(pick_row.get("p50"), proj_val)
                                p90_val = _safe_num(pick_row.get("p90"), proj_val)
                                st.markdown(f"**Distribution:** P10 {p10_val:.2f} · Median {p50_val:.2f} · P90 {p90_val:.2f}")
                                breakout_prob = _safe_num(pick_row.get("breakout_prob"), 0.0) * 100
                                dud_prob = _safe_num(pick_row.get("dud_prob"), 0.0) * 100
                                breakout_target = pick_row.get("breakout_target")
                                dud_target = pick_row.get("dud_target")
                                tail_reasons = build_tail_reason_lists(dict(pick_row))
                                tail_lines = []
                                if breakout_target is not None:
                                    tail_lines.append(
                                        f"**{_tail_labels['breakout']}:** {breakout_watch} ({breakout_prob:.1f}% for "
                                        + tail_target_text(pick_row["stat_type"], _si, breakout_target, "breakout")
                                        + ")"
                                    )
                                if dud_target is not None:
                                    tail_lines.append(
                                        f"**{_tail_labels['dud']}:** {dud_risk} ({dud_prob:.1f}% for "
                                        + tail_target_text(pick_row["stat_type"], _si, dud_target, "dud")
                                        + ")"
                                    )
                                if tail_lines:
                                    st.markdown("  \n".join(tail_lines))
                                if breakout_watch in ("Medium", "High") and tail_reasons["breakout"]:
                                    st.markdown(
                                        f"**Why {_tail_labels['breakout'].lower()}:**  \n"
                                        + "\n".join(f"- {reason}" for reason in tail_reasons["breakout"])
                                    )
                                if dud_risk in ("Medium", "High") and tail_reasons["dud"]:
                                    st.markdown(
                                        f"**Why {_tail_labels['dud'].lower()}:**  \n"
                                        + "\n".join(f"- {reason}" for reason in tail_reasons["dud"])
                                    )
                                if _bat_ord:
                                    _pa_mult = _safe_num(pick_row.get('pa_multiplier'), 1.0)
                                    st.markdown(f"**PA Multiplier:** {_pa_mult:.2f}x (lineup spot #{_bat_ord})")
                                # Health/form
                                _health = pick_row.get("injury_status", "active")
                                _spring = pick_row.get("spring_badge", "neutral")
                                _trend = pick_row.get("trend_badge", "neutral")
                                status_parts = []
                                status_parts.append(f"Health: {health_icon}")
                                if _spring != "neutral":
                                    status_parts.append(f"Form: {form_icon}")
                                if _trend != "neutral":
                                    status_parts.append(f"Trend: {trend_icon}")
                                st.markdown(" · ".join(status_parts))

                    if checked:
                        selected_picks.append(pick_row)

                # ── Smart Parlay Suggestions ──
                _suggest_preds = [dict(row) for _, row in slip_candidates.iterrows()] if not slip_candidates.empty else []
                if len(_suggest_preds) >= 5:
                    st.markdown('<div class="section-hdr">Smart Slip Suggestions</div>', unsafe_allow_html=True)
                    st.caption("Auto-generated optimal slips based on confidence, diversity, and correlation rules")
                    _slip_size = st.radio("Slip size", [6, 5], horizontal=True, key="suggest_size")
                    try:
                        _suggested = suggest_slips(_suggest_preds, num_slips=3, slip_size=_slip_size)
                        if _suggested:
                            _sg_cols = st.columns(min(len(_suggested), 3))
                            for _si, _sg in enumerate(_suggested[:3]):
                                with _sg_cols[_si]:
                                    _sg_quality = _sg.get("quality_score", 0)
                                    _sg_border = "#00C853" if _sg_quality >= 80 else ("#FFB300" if _sg_quality >= 60 else "rgba(255,255,255,0.1)")
                                    _sg_wp = _sg.get("estimated_win_prob", 0)

                                    # Kelly sizing
                                    _sg_kelly = calculate_slip_sizing(
                                        _sg["picks"],
                                        bankroll=st.session_state.get("starting_bankroll", 100.0),
                                        slip_type=f"{_slip_size}_flex"
                                    )

                                    st.markdown(f'''<div style="background:linear-gradient(145deg,#0d1828,#091020);border:1px solid {_sg_border};border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                                            <span style="font-weight:700;color:#E8ECF1;font-size:0.85rem;">{_sg["label"]}</span>
                                            <span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{_sg_border};font-weight:600;">Q{int(_sg_quality)}</span>
                                        </div>
                                        <div style="font-size:0.72rem;color:rgba(232,236,241,0.4);margin-bottom:0.5rem;">{_sg["direction_balance"]} · {_sg.get("risk_level","").title()} Risk</div>
                                    </div>''', unsafe_allow_html=True)

                                    for _sp in _sg["picks"]:
                                        _sp_cls = "more" if _sp["pick"] == "MORE" else "less"
                                        _sp_conf = int(_sp.get("confidence", 0) * 100)
                                        st.markdown(
                                            f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0.5rem;font-size:0.78rem;border-bottom:1px solid rgba(255,255,255,0.04);">'
                                            f'<span class="dir-chip {_sp_cls}" style="font-size:0.65rem;padding:0.12rem 0.4rem;">{_sp["pick"]}</span>'
                                            f'<span style="color:#E8ECF1;font-weight:500;flex:1;">{_sp["player_name"]}</span>'
                                            f'<span style="color:rgba(232,236,241,0.35);font-size:0.72rem;">{_sp.get("stat_type","")}</span>'
                                            f'<span style="font-family:JetBrains Mono;font-size:0.72rem;color:rgba(232,236,241,0.5);">{_sp_conf}%</span>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                                    # v018: Monte Carlo EV for suggested slip
                                    try:
                                        _sg_legs = [{
                                            "team": sp.get("team", ""),
                                            "pick": sp.get("pick", "MORE"),
                                            "stat_type": sp.get("stat_internal", sp.get("stat_type", "")),
                                            "line": sp.get("line", 0.5),
                                            "win_prob": sp.get("win_prob", sp.get("confidence", 0.55)),
                                            "p_push": sp.get("p_push", 0.0),
                                            "game_id": sp.get("game_pk") or sp.get("game_id") or sp.get("opponent", ""),
                                        } for sp in _sg["picks"]]
                                        _line_type_counts = {}
                                        for _sp in _sg["picks"]:
                                            _lt = _sp.get("line_type", "standard")
                                            _line_type_counts[_lt] = _line_type_counts.get(_lt, 0) + 1
                                        _sg_line_type = max(_line_type_counts, key=_line_type_counts.get)
                                        _sg_ev = simulate_slip_ev(
                                            _sg_legs,
                                            entry_type=f"{_slip_size}_flex",
                                            n_sims=25_000,
                                            correlation_matrix=build_correlation_matrix(_sg_legs),
                                            seed=42,
                                            line_type=_sg_line_type,
                                        )
                                        _sg_ev_pct = _sg_ev["ev_profit_pct"]
                                        _sg_ev_color = "#00C853" if _sg_ev_pct > 0 else ("#FFB300" if _sg_ev_pct > -10 else "#FF4444")
                                        _sg_ev_label = f"{_sg_ev_pct:+.1f}%"
                                        st.markdown(
                                            f'<div style="margin-top:0.4rem;padding:0.4rem 0.6rem;background:rgba(0,100,200,0.04);border:1px solid rgba(0,100,200,0.08);border-radius:8px;font-size:0.72rem;">'
                                            f'<span style="color:rgba(232,236,241,0.4);">Modeled EV:</span> '
                                            f'<span style="font-family:JetBrains Mono;font-weight:700;color:{_sg_ev_color};">{_sg_ev_label}</span>'
                                            f' · <span style="color:rgba(232,236,241,0.4);">Positive Return:</span> '
                                            f'<span style="font-family:JetBrains Mono;font-size:0.7rem;color:rgba(232,236,241,0.5);">{_sg_ev["win_rate"]*100:.1f}%</span>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    except Exception:
                                        pass

                                    # Kelly sizing display
                                    _rec_wager = _sg_kelly.get("recommended_wager", 1.0)
                                    _edge = _sg_kelly.get("edge_pct", 0)
                                    _edge_color = "#00C853" if _edge > 0 else "#FF4444"
                                    st.markdown(
                                        f'<div style="margin-top:0.5rem;padding:0.5rem 0.6rem;background:rgba(0,200,83,0.04);border:1px solid rgba(0,200,83,0.08);border-radius:8px;font-size:0.75rem;">'
                                        f'<span style="color:rgba(232,236,241,0.4);">Kelly Approx:</span> '
                                        f'<span style="font-family:JetBrains Mono;font-weight:700;color:#E8ECF1;">${_rec_wager:.2f}</span>'
                                        f' · <span style="color:rgba(232,236,241,0.4);">Edge:</span> '
                                        f'<span style="font-family:JetBrains Mono;font-weight:600;color:{_edge_color};">{_edge:+.1f}%</span>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.caption("Not enough qualifying picks to build suggested slips.")
                    except Exception as _e:
                        st.caption(f"Slip suggestion unavailable: {_e}")

                if selected_picks:
                    st.markdown('<div class="section-hdr">Create Slip from Selected</div>', unsafe_allow_html=True)
                    slip_df = pd.DataFrame(selected_picks)
                    st.dataframe(slip_df[["player_name","stat_type","line","pick","rating","confidence"]], hide_index=True, width="stretch")

                    # v018: Correlation warnings for same-game picks
                    try:
                        _pick_dicts = [{"player_name": p["player_name"], "team": p.get("team", ""), "stat_type": p.get("stat_type", ""), "pick": p.get("pick", "")} for _, p in slip_df.iterrows()]
                        _corr = analyze_slip_correlation(_pick_dicts)
                        if _corr.get("warnings"):
                            _warn_level = _corr.get("severity", "low")
                            _warn_color = "#FF4444" if _warn_level == "high" else "#FFB300"
                            _warn_icon = "🔴" if _warn_level == "high" else "⚠️"
                            _warn_msgs = _corr["warnings"]
                            st.markdown(
                                f'<div class="warn-strip">{_warn_icon} <strong>Correlation Warning:</strong> '
                                + " | ".join(_warn_msgs[:3])
                                + '</div>', unsafe_allow_html=True
                            )
                            # Show same-game picks specifically
                            _same_game = _corr.get("same_game_groups", {})
                            if _same_game:
                                for _team, _players in _same_game.items():
                                    if len(_players) > 1:
                                        st.markdown(
                                            f'<div style="font-size:0.78rem;color:{_warn_color};padding:0.3rem 0.8rem;">'
                                            f'Same game ({_team}): {", ".join(_players)} — outcomes are correlated, reduces parlay edge</div>',
                                            unsafe_allow_html=True
                                        )
                    except Exception:
                        pass

                    slip_type = st.selectbox(
                        "Slip type",
                        ["6_flex", "5_flex", "4_flex", "3_power", "2_power"],
                        index=0,
                        key="slip_type_select",
                    )
                    slip_amt = st.number_input("Wager ($)", min_value=1.0, value=5.0, step=1.0, key="slip_amt_select")
                    _num_needed = int(slip_type[0])

                    # Show expected payout + Kelly sizing + house edge warning
                    if slip_type in PAYOUTS:
                        _payout_table = PAYOUTS[slip_type]
                        _payout_mult = _payout_table.get(_num_needed, 0)  # Perfect payout
                        _expected_payout = slip_amt * _payout_mult
                        _be = BREAKEVEN.get(slip_type, 0.5)
                        # Show partial payouts for flex
                        _partial_str = ""
                        if "flex" in slip_type:
                            _partials = [(k, v) for k, v in sorted(_payout_table.items(), reverse=True) if v > 0 and k < _num_needed]
                            if _partials:
                                _partial_str = " · Partial: " + ", ".join(f"{k}/{_num_needed}={v}x" for k, v in _partials)
                        st.markdown(
                            f'<div class="info-strip">Payout: <span class="hl">{_payout_mult}x</span> · '
                            f'Expected: <span class="hl">${_expected_payout:.2f}</span> · '
                            f'Break-even: <span class="hl">{_be*100:.1f}%</span> per leg{_partial_str}</div>',
                            unsafe_allow_html=True
                        )
                        # Warn on high house edge entries
                        if slip_type == "3_power":
                            st.markdown(
                                '<div class="warn-strip">⚠️ <strong>3-Pick Power has the highest house edge</strong> — '
                                'consider 5 or 6-Pick Flex for better long-term EV.</div>',
                                unsafe_allow_html=True
                            )
                        # Kelly Criterion sizing
                        try:
                            _sel_picks_for_kelly = [{
                                "confidence": p.get("confidence", 0.55),
                                "win_prob": p.get("win_prob", p.get("confidence", 0.55)),
                                "p_push": p.get("p_push", 0.0),
                            } for _, p in slip_df.iterrows()]
                            _kelly_result = calculate_slip_sizing(
                                _sel_picks_for_kelly,
                                bankroll=st.session_state.get("starting_bankroll", 100.0),
                                slip_type=slip_type
                            )
                            _k_rec = _kelly_result.get("recommended_wager", 1.0)
                            _k_edge = _kelly_result.get("edge_pct", 0)
                            _k_wp = _kelly_result.get("win_prob", 0) * 100
                            _k_cls = "#00C853" if _k_edge > 0 else "#FF4444"
                            st.markdown(
                                f'<div class="info-strip">'
                                f'Kelly Approx Wager: <span class="hl">${_k_rec:.2f}</span> · '
                                f'Win Prob: <span class="hl">{_k_wp:.1f}%</span> · '
                                f'Edge: <span style="color:{_k_cls};font-weight:600">{_k_edge:+.1f}%</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        except Exception:
                            pass

                    # v018: Monte Carlo EV simulation for selected slip
                    if len(selected_picks) >= 2:
                        try:
                            _mc_legs = []
                            _has_promo = False
                            for _, _sp in slip_df.iterrows():
                                _mc_legs.append({
                                    "win_prob": _sp.get("win_prob", _sp.get("confidence", 0.55)),
                                    "p_push": _sp.get("p_push", 0.0),
                                    "stat_type": _sp.get("stat_internal", _sp.get("stat_type", "")),
                                    "line": _sp.get("line", 0.5),
                                    "team": _sp.get("team", ""),
                                    "pick": _sp.get("pick", "MORE"),
                                })
                                if _sp.get("line_type", "standard") != "standard":
                                    _has_promo = True

                            # Build correlation matrix and run full MC sim
                            _mc_corr = build_correlation_matrix(_mc_legs)
                            _mc_result = simulate_slip_ev(
                                _mc_legs, entry_type=slip_type,
                                n_sims=50_000, correlation_matrix=_mc_corr,
                            )
                            # Also get quick analytical EV for comparison
                            _qa_result = quick_slip_ev(
                                [l["win_prob"] for l in _mc_legs],
                                entry_type=slip_type,
                            )

                            _mc_ev = _mc_result["ev_profit_pct"]
                            _mc_color = "#00C853" if _mc_ev > 0 else ("#FFB300" if _mc_ev > -15 else "#FF4444")
                            _mc_wr = _mc_result["win_rate"] * 100
                            _mc_sharpe = _mc_result["sharpe_ratio"]

                            st.markdown(
                                f'<div style="margin-top:0.6rem;padding:0.7rem 1rem;background:linear-gradient(145deg,rgba(0,100,200,0.06),rgba(0,200,83,0.03));border:1px solid rgba(0,150,255,0.12);border-radius:10px;">'
                                f'<div style="font-size:0.7rem;color:rgba(232,236,241,0.4);letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">Monte Carlo EV Simulation (50K runs)</div>'
                                f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;">'
                                f'<div><span style="font-size:0.72rem;color:rgba(232,236,241,0.4);">EV:</span> <span style="font-family:JetBrains Mono;font-weight:700;font-size:0.9rem;color:{_mc_color};">{_mc_ev:+.1f}%</span></div>'
                                f'<div><span style="font-size:0.72rem;color:rgba(232,236,241,0.4);">Win Rate:</span> <span style="font-family:JetBrains Mono;font-weight:600;color:#E8ECF1;">{_mc_wr:.1f}%</span></div>'
                                f'<div><span style="font-size:0.72rem;color:rgba(232,236,241,0.4);">Sharpe:</span> <span style="font-family:JetBrains Mono;font-weight:600;color:#E8ECF1;">{_mc_sharpe:.2f}</span></div>'
                                f'<div><span style="font-size:0.72rem;color:rgba(232,236,241,0.4);">Tie Impact:</span> <span style="font-family:JetBrains Mono;font-size:0.8rem;color:rgba(232,236,241,0.5);">{_mc_result["tie_impact"]*100:.1f}%</span></div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            # Show comparison: MC (correlated) vs Analytical (independent)
                            _qa_ev = _qa_result["ev_profit_pct"]
                            _ev_gap = _mc_ev - _qa_ev
                            if abs(_ev_gap) > 0.5:
                                _gap_label = "Correlation drag" if _ev_gap < 0 else "Correlation benefit"
                                st.markdown(
                                    f'<div style="padding:0 1rem 0.5rem;font-size:0.7rem;color:rgba(232,236,241,0.35);">'
                                    f'Independent EV: {_qa_ev:+.1f}% · {_gap_label}: {_ev_gap:+.1f}pp'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                            st.markdown('</div>', unsafe_allow_html=True)

                            # Warn on promo/goblin lines
                            if _has_promo:
                                st.markdown(
                                    '<div class="warn-strip">👺 <strong>Promo/Goblin line detected</strong> — '
                                    'actual payouts may be lower than standard. Check PrizePicks in-app for real payout rates.</div>',
                                    unsafe_allow_html=True
                                )

                        except Exception as _mc_err:
                            st.caption(f"EV simulation unavailable: {_mc_err}")

                    if st.button("Build Slip", type="primary"):
                        if len(selected_picks) == _num_needed:
                            picks_for_slip = []
                            slip_game_dates = []
                            for _, pick_row in slip_df.iterrows():
                                pick_date = pick_row.get("game_date", date.today().isoformat())
                                slip_game_dates.append(pick_date)
                                prediction_payload = pick_row.to_dict()
                                prediction_id = log_prediction(prediction_payload, game_date=pick_date)
                                picks_for_slip.append({
                                    "prediction_id": prediction_id,
                                    "player_name": pick_row["player_name"],
                                    "stat_type": pick_row["stat_type"],
                                    "line": pick_row["line"],
                                    "pick": pick_row["pick"],
                                    "game_date": pick_date,
                                })
                                mark_as_bet(
                                    pick_row["player_name"],
                                    pick_row.get("stat_internal", pick_row["stat_type"]),
                                    pick_date,
                                )
                            slip_game_date = max(slip_game_dates) if slip_game_dates else date.today().isoformat()
                            slip_id = create_slip(slip_game_date, slip_type, slip_amt, picks_for_slip)
                            st.success(f"Slip #{slip_id} created!")
                            st.rerun()
                        else:
                            st.warning(f"Select exactly {_num_needed} picks for a {slip_type}.")

# ─── 📰 NEWS TAB ─────────────────────────────────────────────────────────
with tab_news:
    st.markdown('<div class="section-hdr">MLB News &amp; Transactions</div>', unsafe_allow_html=True)
    if st.button("Load News & Transactions", key="news_load_manual"):
        st.session_state["manual_news_fetch"] = True

    if not st.session_state.get("manual_news_fetch", False):
        st.markdown(
            '<div class="panel-shell"><div class="panel-title">News feeds are paused</div>'
            '<div style="font-size:0.84rem;color:rgba(232,236,241,0.62);line-height:1.45;">'
            'This tab is also in manual mode. Click <strong>Load News &amp; Transactions</strong> only when you actually need external news or injury pulls.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        news_col1, news_col2 = st.columns([3, 2])

        # ── MLB Headlines ──
        with news_col1:
            st.markdown("#### Headlines")
            try:
                _news_items = fetch_mlb_news(max_items=12)
            except Exception:
                _news_items = []

            if _news_items:
                for _art in _news_items:
                    _link = _art.get("url", "")
                    _title = _art.get("title", "Untitled")
                    _snippet = _art.get("snippet", "")
                    _date = _art.get("date", "")
                    _title_html = f'<a href="{_link}" target="_blank" style="color:#4FC3F7;text-decoration:none;font-weight:600;">{_title}</a>' if _link else f'<span style="font-weight:600;">{_title}</span>'
                    st.markdown(
                        f'<div style="padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.08);">'
                        f'{_title_html}'
                        f'<div style="font-size:0.82rem;color:#aaa;margin-top:3px;">{_snippet}</div>'
                        f'<div style="font-size:0.72rem;color:#666;margin-top:2px;">{_date}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No MLB news available right now. Check back closer to game time.")

        # ── Transactions & Injuries ──
        with news_col2:
            st.markdown("#### Recent Transactions")
            _txn_days = st.selectbox("Lookback", [3, 7, 14], index=0, key="news_txn_days", label_visibility="collapsed")
            try:
                _txns = fetch_recent_transactions(days_back=_txn_days)
            except Exception:
                _txns = []

            if _txns:
                # Group by category
                from collections import defaultdict
                _cat_groups = defaultdict(list)
                for _t in _txns:
                    _cat_groups[_t.get("category", "other")].append(_t)

                _cat_labels = {
                    "injury": "🏥 Injury List",
                    "activation": "✅ Activations",
                    "roster": "📋 Roster Moves",
                    "trade": "🔄 Trades",
                    "signing": "✍️ Signings",
                    "release": "❌ Releases",
                    "other": "📰 Other",
                }
                for _cat_key in ["injury", "activation", "roster", "trade", "signing", "release", "other"]:
                    _items = _cat_groups.get(_cat_key, [])
                    if not _items:
                        continue
                    st.markdown(f"**{_cat_labels.get(_cat_key, _cat_key)}** ({len(_items)})")
                    for _t in _items[:8]:
                        _desc = _t.get("description", "")
                        if len(_desc) > 120:
                            _desc = _desc[:117] + "..."
                        st.markdown(
                            f'<div style="font-size:0.82rem;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
                            f'{_t.get("icon","")} <b>{_t.get("player_name","")}</b> '
                            f'<span style="color:#888;">({_t.get("team","")})</span><br/>'
                            f'<span style="color:#aaa;">{_desc}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("")
            else:
                st.info("No recent transactions found.")

            # ── Injury Report ──
            st.markdown("#### Injury Report")
            try:
                _injuries = fetch_injuries(days_back=14)
            except Exception:
                _injuries = []

            if _injuries:
                _active_il = [i for i in _injuries if not i.get("is_activation")]
                _returned = [i for i in _injuries if i.get("is_activation")]

                if _active_il:
                    st.markdown(f"**Currently on IL** ({len(_active_il)})")
                    for _inj in _active_il[:15]:
                        st.markdown(
                            f'<div style="font-size:0.82rem;padding:3px 0;">'
                            f'🚑 <b>{_inj["player_name"]}</b> '
                            f'<span style="color:#888;">({_inj.get("team_abbr", _inj.get("team",""))})</span> — '
                            f'<span style="color:#FF8A80;">{_inj.get("il_type","IL")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if _returned:
                    st.markdown(f"**Recently Activated** ({len(_returned)})")
                    for _inj in _returned[:10]:
                        st.markdown(
                            f'<div style="font-size:0.82rem;padding:3px 0;">'
                            f'✅ <b>{_inj["player_name"]}</b> '
                            f'<span style="color:#888;">({_inj.get("team_abbr", _inj.get("team",""))})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No injury data available.")

with tab_slips:
    st.markdown('<div class="section-hdr">Slip Tracker &amp; P&amp;L</div>', unsafe_allow_html=True)
    pnl = get_slip_pnl(30)
    if pnl["slips_total"] > 0:
        c1,c2,c3,c4 = st.columns(4)
        pnl_pos = pnl["net_profit"] >= 0
        cls = "g" if pnl_pos else "r"
        ccard = "card-g" if pnl_pos else "card-r"
        with c1: st.markdown(f'<div class="card {ccard}"><div class="lbl">Net P&amp;L</div><div class="val {cls}">${pnl["net_profit"]:+.2f}</div><div class="sub">Last 30 days</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card {ccard}"><div class="lbl">ROI</div><div class="val {cls}">{pnl["roi"]:+.1f}%</div><div class="sub">Return on entry</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><div class="lbl">Wagered</div><div class="val">${pnl["total_wagered"]:.0f}</div><div class="sub">{pnl["slips_total"]} slips</div></div>', unsafe_allow_html=True)
        with c4:
            win_rate = pnl["slips_won"] / pnl["slips_total"] if pnl["slips_total"] else 0
            wr_cls = "g" if win_rate >= 0.5 else "r"
            st.markdown(f'<div class="card"><div class="lbl">Record</div><div class="val {wr_cls}">{pnl["slips_won"]}W-{pnl["slips_lost"]}L</div><div class="sub">{win_rate*100:.0f}% win rate</div></div>', unsafe_allow_html=True)

        br_history = get_bankroll_history()
        if len(br_history) >= 2:
            st.markdown('<div class="section-hdr">Bankroll Tracker</div>', unsafe_allow_html=True)
            balances = [h["balance"] for h in br_history]
            high_water = max(balances)
            max_dd = min(b - max(balances[:i+1]) for i, b in enumerate(balances))
            brc1, brc2, brc3 = st.columns(3)
            starting = st.session_state.get("starting_bankroll", 100.0)
            with brc1: st.metric("Current Bankroll", f"${balances[-1]:.2f}", f"{balances[-1]-starting:+.2f}")
            with brc2: st.metric("High Water Mark", f"${high_water:.2f}")
            with brc3: st.metric("Max Drawdown", f"${max_dd:.2f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["date"] for h in br_history],
                y=balances,
                mode="lines+markers",
                line=dict(color="#00C853" if balances[-1] >= starting else "#FF5252", width=2),
                marker=dict(size=5),
                hovertext=[h["label"] for h in br_history],
                hoverinfo="text+y",
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0), height=250,
                yaxis_title="Balance ($)", xaxis_title="",
            )
            st.plotly_chart(fig, width="stretch")
    with st.expander("Create New Slip"):
        sl_date = st.date_input("Game date", value=date.today(), key="slip_date")
        sl_type = st.selectbox("Entry type", ["6_flex", "5_flex", "4_flex", "3_power", "2_power"], index=0)
        sl_amount = st.number_input("Entry amount ($)", min_value=1.0, value=5.0, step=1.0)
        num = int(sl_type[0])
        st.caption(f"Break-even: {BREAKEVEN.get(sl_type, 0.55)*100:.1f}%")
        slip_picks_input = []
        for i in range(num):
            cols = st.columns([3, 2, 1, 1])
            with cols[0]: pn = st.text_input(f"Player {i+1}", key=f"sp_{i}")
            with cols[1]: stype = st.text_input("Prop", key=f"st_{i}")
            with cols[2]: ln = st.number_input("Line", min_value=0.0, step=0.5, key=f"sl_{i}")
            with cols[3]: pk = st.selectbox("Pick", ["MORE", "LESS"], key=f"spk_{i}")
            if pn and stype:
                slip_picks_input.append({"player_name": pn, "stat_type": stype, "line": ln, "pick": pk})
        if st.button("Save Slip", type="primary") and len(slip_picks_input) == num:
            sid = create_slip(sl_date.isoformat(), sl_type, sl_amount, slip_picks_input)
            st.success(f"Slip #{sid} saved!")
            st.rerun()
    slips_df = get_slips(20)
    if not slips_df.empty:
        st.markdown('<div class="section-hdr">Recent Slips</div>', unsafe_allow_html=True)
        for _, slip in slips_df.iterrows():
            _slip_status = slip["status"]
            status_icon = {"win": "✅", "loss": "❌", "partial": "🟡", "push": "➖", "pending": "⏳"}.get(_slip_status, "⏳")
            _slip_pnl_str = ""
            if _slip_status == "win":
                _slip_pnl_str = f" · +${slip.get('net_profit', 0):.2f}"
            elif _slip_status == "loss":
                _slip_pnl_str = f" · -${abs(slip.get('net_profit', slip.get('entry_amount', 0))):.2f}"
            with st.expander(f"{status_icon} **Slip #{slip['id']}** — {slip['entry_type']} · ${slip['entry_amount']:.0f} entry · {slip['game_date']}{_slip_pnl_str}"):
                sp = get_slip_picks(slip["id"])
                if not sp.empty:
                    for _, p in sp.iterrows():
                        res_icon = {"W": "✅", "L": "❌", "push": "➖"}.get(p.get("result"), "⏳")
                        st.markdown(f"{res_icon} **{p['player_name']}** — {p['stat_type']} · Line: `{p['line']}` · **{p['pick']}**" + (f" · Actual: `{p['actual_result']}`" if p.get("actual_result") else ""))
                if slip["status"] != "pending":
                    st.markdown(f"**Payout:** {slip['payout_mult']}x = **${slip['payout_amount']:.2f}** · Net: **${slip['net_profit']:+.2f}**")
    else:
        st.info("No slips yet. Create one above or from the Find Edges tab.")

with tab_grade:
    st.markdown('<div class="section-hdr">Grade Results</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-strip">Auto-Grade pulls actual results from the MLB Stats API and grades all pending picks automatically.</div>', unsafe_allow_html=True)

    ag_col1, ag_col2 = st.columns([1, 3])
    with ag_col1:
        if st.button("Auto-Grade Yesterday", type="primary"):
            with st.spinner("Pulling box scores and grading..."):
                try:
                    result = auto_grade_yesterday()
                    if result["graded"] > 0:
                        wins = sum(1 for r in result["results"] if r.get("result") == "W")
                        losses = sum(1 for r in result["results"] if r.get("result") == "L")
                        st.success(f"Auto-graded {result['graded']} picks: {wins}W-{losses}L")
                        try:
                            learn_result = run_adjustment_cycle(min_sample=25)
                            if learn_result.get("adjusted"):
                                st.info(f"Model auto-tuned: {learn_result.get('reason', 'weights updated')}")
                            elif learn_result.get("reason"):
                                st.caption(f"Auto-tune: {learn_result.get('reason')}")
                        except Exception as e:
                            st.caption(f"Auto-tune skipped: {e}")
                    else:
                        st.info("No picks to auto-grade (games may not be final yet)")
                    if result.get("errors"):
                        st.warning(f"{len(result['errors'])} errors: {result['errors'][:3]}")
                except Exception as e:
                    st.error(f"Auto-grade failed: {e}")
    with ag_col2:
        ag_date = st.date_input("Or auto-grade specific date", value=date.today()-timedelta(days=1), key="ag_date")
        if st.button("Auto-Grade This Date"):
            with st.spinner("Grading..."):
                try:
                    result = auto_grade_date(ag_date.isoformat())
                    st.success(f"Graded {result['graded']} picks")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.markdown('<div class="section-hdr">Manual Grading</div>', unsafe_allow_html=True)
    gd = st.date_input("Select date to grade", value=date.today()-timedelta(days=1))
    ug = get_ungraded_predictions(gd.isoformat())
    if ug.empty:
        st.info(f"No pending picks for {gd}. All caught up!")
    else:
        st.markdown(f'<div style="font-size:0.8rem;color:rgba(232,236,241,0.4);margin-bottom:0.8rem;">{len(ug)} picks waiting for results</div>', unsafe_allow_html=True)
        for _,row in ug.iterrows():
            pick_dir_color = "#00C853" if row['pick'] == "MORE" else "#FF4444"
            ci,cinp,cb=st.columns([3,1,1])
            with ci:
                st.markdown(
                    f'<div style="padding:0.3rem 0">'
                    f'<span style="font-weight:600;color:#E8ECF1">{row["player_name"]}</span> '
                    f'<span style="color:rgba(232,236,241,0.4);font-size:0.8rem">· {row["stat_type"]} · Line {row["line"]} · </span>'
                    f'<span style="color:{pick_dir_color};font-weight:700;font-family:JetBrains Mono,monospace">{row["pick"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with cinp:
                a=st.number_input("Actual",min_value=0.0,step=0.5,key=f"a_{row['id']}",label_visibility="collapsed")
            with cb:
                if st.button("Grade",key=f"g_{row['id']}"):
                    r=grade_prediction(row["id"],a)
                    st.success("✅ W" if r=="W" else ("❌ L" if r=="L" else "➖ Push"))
                    st.rerun()

    rg=get_graded_predictions(30)
    if not rg.empty:
        st.markdown('<div class="section-hdr">Recent Results</div>', unsafe_allow_html=True)
        feed_rows = []
        for _, gr in rg.iterrows():
            res = gr.get("result", "")
            res_icon = "✅" if res == "W" else ("❌" if res == "L" else "➖")
            res_cls = "w" if res == "W" else ("l" if res == "L" else "p")
            pick_color = "#00C853" if gr["pick"] == "MORE" else "#FF4444"
            feed_rows.append(
                f'<div class="grade-feed-row">'
                f'<span style="font-size:1rem">{res_icon}</span>'
                f'<span class="gfr-player">{gr["player_name"]}</span>'
                f'<span class="gfr-prop">{gr["stat_type"]} · Line {gr["line"]}</span>'
                f'<span style="color:{pick_color};font-family:JetBrains Mono,monospace;font-size:0.8rem;font-weight:700">{gr["pick"]}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:rgba(232,236,241,0.4)">→ {gr.get("actual_result","?")}</span>'
                f'<span class="gfr-result {res_cls}">{res}</span>'
                f'<span style="font-size:0.7rem;color:rgba(232,236,241,0.25)">{gr["game_date"]}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.5rem 0.8rem;overflow:hidden">{"".join(feed_rows[:20])}</div>',
            unsafe_allow_html=True
        )

with tab_qa:
    st.markdown('<div class="section-hdr">Model QA Dashboard</div>', unsafe_allow_html=True)
    qa_days = st.selectbox("Evaluation window", [7, 14, 30, 60, 90], index=2, key="qa_window")
    qa_run_col, qa_date_col = st.columns([1, 2])
    with qa_date_col:
        qa_run_date = st.date_input(
            "Nightly update date",
            value=date.today() - timedelta(days=1),
            key="qa_nightly_date",
        )
    with qa_run_col:
        st.markdown('<div style="height:1.7rem"></div>', unsafe_allow_html=True)
        if st.button("Run Nightly Update", key="qa_run_nightly"):
            with st.spinner("Running nightly cycle..."):
                try:
                    nightly_result = run_nightly_cycle(qa_run_date.isoformat())
                    grading = nightly_result.get("phase_results", {}).get("grading", {})
                    metrics = nightly_result.get("phase_results", {}).get("metrics", {})
                    st.success(
                        f"Nightly cycle complete: {grading.get('total_graded', 0)} picks graded, "
                        f"{grading.get('projected_stats_graded', 0)} tracked rows updated."
                    )
                    if metrics.get("reason"):
                        st.caption(metrics["reason"])
                except Exception as exc:
                    st.error(f"Nightly cycle failed: {exc}")

    qa_projection = get_projection_diagnostics(days_back=qa_days)
    qa_accuracy = get_projection_accuracy(days_back=qa_days)
    qa_calibration = get_calibration_data(days_back=qa_days)
    qa_board = get_board_stats(days=qa_days)
    qa_shadow = get_shadow_sample_stats(days=qa_days)
    qa_weights = load_current_weights()

    model_version = qa_weights.get("version", "unknown")
    st.caption(
        f"Window: last {qa_days} days | Active weights: {model_version} | "
        "Bias is actual minus projection, so positive means the model ran low."
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        total_proj = qa_projection.get("total", 0)
        st.markdown(
            f'<div class="card"><div class="lbl">Graded Projections</div><div class="val">{total_proj}</div><div class="sub">Full board tracking</div></div>',
            unsafe_allow_html=True,
        )
    with q2:
        mae_val = qa_projection.get("mae")
        mae_str = f"{mae_val:.3f}" if isinstance(mae_val, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">MAE</div><div class="val">{mae_str}</div><div class="sub">Average absolute error</div></div>',
            unsafe_allow_html=True,
        )
    with q3:
        rmse_val = qa_projection.get("rmse")
        rmse_str = f"{rmse_val:.3f}" if isinstance(rmse_val, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">RMSE</div><div class="val">{rmse_str}</div><div class="sub">Punishes larger misses</div></div>',
            unsafe_allow_html=True,
        )
    with q4:
        shadow_acc = qa_shadow.get("accuracy")
        shadow_str = f"{shadow_acc * 100:.1f}%" if isinstance(shadow_acc, (int, float)) else "-"
        st.markdown(
            f'<div class="card card-b"><div class="lbl">Shadow Accuracy</div><div class="val b">{shadow_str}</div><div class="sub">{qa_shadow.get("graded", 0)} graded shadow picks</div></div>',
            unsafe_allow_html=True,
        )

    q5, q6, q7, q8 = st.columns(4)
    with q5:
        acc_val = qa_accuracy.get("accuracy")
        acc_str = f"{acc_val * 100:.1f}%" if isinstance(acc_val, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">Pick Accuracy</div><div class="val">{acc_str}</div><div class="sub">Projected picks only</div></div>',
            unsafe_allow_html=True,
        )
    with q6:
        brier_val = qa_accuracy.get("brier_score")
        brier_str = f"{brier_val:.3f}" if isinstance(brier_val, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">Brier</div><div class="val">{brier_str}</div><div class="sub">Lower is better</div></div>',
            unsafe_allow_html=True,
        )
    with q7:
        bias_val = qa_projection.get("bias")
        bias_cls = "g" if isinstance(bias_val, (int, float)) and abs(bias_val) < 0.1 else ("r" if isinstance(bias_val, (int, float)) else "")
        bias_str = f"{bias_val:+.3f}" if isinstance(bias_val, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">Bias</div><div class="val {bias_cls}">{bias_str}</div><div class="sub">Actual - projection</div></div>',
            unsafe_allow_html=True,
        )
    with q8:
        sel_bias = qa_board.get("selection_bias")
        sel_str = f"{sel_bias * 100:+.1f}pp" if isinstance(sel_bias, (int, float)) else "-"
        st.markdown(
            f'<div class="card"><div class="lbl">Bet Selection Bias</div><div class="val">{sel_str}</div><div class="sub">Bet accuracy - non-bet accuracy</div></div>',
            unsafe_allow_html=True,
        )

    if qa_projection.get("total", 0) == 0:
        st.info("No graded projection history yet. Once nightly grading runs on tracked dates, this tab will fill in.")
    else:
        st.markdown('<div class="section-hdr">Calibration</div>', unsafe_allow_html=True)
        cal_df = pd.DataFrame(qa_calibration.get("buckets", []))
        if not cal_df.empty:
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(
                x=[0.50, 0.75],
                y=[0.50, 0.75],
                mode="lines",
                name="Perfect",
                line=dict(color="rgba(255,255,255,0.25)", dash="dash"),
            ))
            cal_fig.add_trace(go.Scatter(
                x=cal_df["predicted_mean"],
                y=cal_df["actual_rate"],
                mode="lines+markers+text",
                text=cal_df["bucket"],
                textposition="top center",
                name="Observed",
                line=dict(color="#4da6ff", width=3),
                marker=dict(size=9, color="#00C853"),
            ))
            cal_fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=280,
                xaxis_title="Predicted confidence",
                yaxis_title="Actual win rate",
                xaxis=dict(range=[0.49, 0.76]),
                yaxis=dict(range=[0.40, 0.80]),
            )
            st.plotly_chart(cal_fig, width="stretch")
            cal_display = cal_df.copy()
            cal_display["predicted_mean"] = cal_display["predicted_mean"].map(lambda x: f"{x * 100:.1f}%")
            cal_display["actual_rate"] = cal_display["actual_rate"].map(lambda x: f"{x * 100:.1f}%")
            cal_display.columns = ["Bucket", "Predicted", "Actual", "Count"]
            st.dataframe(cal_display, hide_index=True, width="stretch")
        else:
            st.caption("Not enough graded picks for a calibration curve in this window.")

        st.markdown('<div class="section-hdr">Residuals By Prop</div>', unsafe_allow_html=True)
        by_prop_rows = []
        for prop_type, stats in sorted(qa_projection.get("by_stat_type", {}).items()):
            board_prop = qa_board.get("by_prop", {}).get(prop_type, {})
            shadow_prop = qa_shadow.get("by_prop", {}).get(prop_type, {})
            by_prop_rows.append({
                "Prop": prop_type,
                "Count": stats.get("count", 0),
                "MAE": stats.get("mae"),
                "RMSE": stats.get("rmse"),
                "Bias": stats.get("bias"),
                "Accuracy": stats.get("accuracy"),
                "Board Acc": board_prop.get("accuracy"),
                "Shadow Acc": shadow_prop.get("accuracy"),
            })
        if by_prop_rows:
            by_prop_df = pd.DataFrame(by_prop_rows).sort_values(["MAE", "Count"], ascending=[False, False])
            for _col in ["MAE", "RMSE", "Bias"]:
                by_prop_df[_col] = by_prop_df[_col].map(lambda x: f"{x:+.3f}" if _col == "Bias" and isinstance(x, (int, float)) else (f"{x:.3f}" if isinstance(x, (int, float)) else "-"))
            for _col in ["Accuracy", "Board Acc", "Shadow Acc"]:
                by_prop_df[_col] = by_prop_df[_col].map(lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "-")
            st.dataframe(by_prop_df, hide_index=True, width="stretch")

        st.markdown('<div class="section-hdr">Board Coverage</div>', unsafe_allow_html=True)
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            board_all = qa_board.get("accuracy_all")
            st.metric("Board Accuracy", f"{board_all * 100:.1f}%" if isinstance(board_all, (int, float)) else "-")
        with bc2:
            bet_acc = qa_board.get("accuracy_bet")
            st.metric("Bet Accuracy", f"{bet_acc * 100:.1f}%" if isinstance(bet_acc, (int, float)) else "-")
        with bc3:
            non_bet_acc = qa_board.get("accuracy_nobet")
            st.metric("Non-Bet Accuracy", f"{non_bet_acc * 100:.1f}%" if isinstance(non_bet_acc, (int, float)) else "-")
        with bc4:
            st.metric("Shadow Pending", str(qa_shadow.get("pending", 0)))

        st.markdown('<div class="section-hdr">Worst Misses</div>', unsafe_allow_html=True)
        worst_df = pd.DataFrame(qa_projection.get("worst_misses", []))
        if not worst_df.empty:
            worst_df = worst_df.rename(columns={
                "game_date": "Date",
                "player_name": "Player",
                "team": "Team",
                "stat_type": "Prop",
                "projected_value": "Projection",
                "actual_value": "Actual",
                "error": "Bias",
                "line": "Line",
                "pick": "Pick",
                "confidence": "Confidence",
            })
            for _col in ["Projection", "Actual", "Bias", "Line"]:
                worst_df[_col] = worst_df[_col].map(lambda x: f"{x:+.3f}" if _col == "Bias" and isinstance(x, (int, float)) else (f"{x:.3f}" if isinstance(x, (int, float)) else "-"))
            worst_df["Confidence"] = worst_df["Confidence"].map(lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "-")
            st.dataframe(worst_df, hide_index=True, width="stretch")

        st.markdown('<div class="section-hdr">Shadow Sample Review</div>', unsafe_allow_html=True)
        shadow_rows_df = pd.DataFrame(qa_shadow.get("recent_rows", []))
        if not shadow_rows_df.empty:
            shadow_rows_df = shadow_rows_df.rename(columns={
                "date": "Date",
                "player_name": "Player",
                "team": "Team",
                "prop_type": "Prop",
                "line": "Line",
                "direction": "Pick",
                "confidence": "Confidence",
                "actual_stat": "Actual",
                "outcome": "Outcome",
                "was_bet": "Was Bet",
            })
            shadow_rows_df["Confidence"] = shadow_rows_df["Confidence"].map(lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "-")
            shadow_rows_df["Outcome"] = shadow_rows_df["Outcome"].map(lambda x: "W" if x == 1 else ("L" if x == 0 else "Pending"))
            shadow_rows_df["Was Bet"] = shadow_rows_df["Was Bet"].map(lambda x: "Yes" if x == 1 else "No")
            st.dataframe(shadow_rows_df, hide_index=True, width="stretch")
        else:
            st.caption("Shadow sample rows will appear here after the board has been logged and graded.")
