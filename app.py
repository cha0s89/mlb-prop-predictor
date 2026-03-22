"""
⚾ MLB Prop Edge v2 — Market-Based Edge Finder
Core: Compare PrizePicks lines to devigged sharp sportsbook odds.
When sharp books disagree with PrizePicks → that's the edge.
Statcast data provides the "why" confirmation layer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, date, timedelta

from src.prizepicks import fetch_prizepicks_mlb_lines, get_available_stat_types
from src.sharp_odds import (
    fetch_mlb_events, fetch_event_props, extract_sharp_lines,
    find_ev_edges, get_api_usage, get_api_key, PP_TO_ODDS_API,
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
    GOBLIN_PAYOUTS, DEMON_PAYOUTS, get_payout_table,
)
from src.autograder import auto_grade_date, auto_grade_yesterday
from src.autolearn import run_adjustment_cycle, load_current_weights, get_weight_history
from src.spring import (
    get_player_injury_status, get_spring_form_multiplier,
    fetch_spring_training_stats, fetch_injuries, fetch_recent_transactions,
)
from src.trends import get_batter_trend
from src.explain import build_explanation, format_explanation_text
from src.combined import score_picks, SIGNAL_CONFIRMED, SIGNAL_SHARP_ONLY, SIGNAL_PROJECTION_ONLY
from src.slip_warnings import analyze_slip_correlation, format_warnings_streamlit
from src.lineups import (
    fetch_todays_games, get_batting_order_position, get_pa_multiplier,
    get_game_context, get_probable_pitcher, fetch_confirmed_lineups,
)
from src.matchups import get_platoon_split_adjustment
from src.database import (
    save_projected_stats, get_projection_accuracy,
    get_projection_history, init_projected_stats_table,
    get_calibration_data,
)
from src.kelly import half_kelly, calculate_slip_sizing
from src.parlay_suggest import suggest_slips, score_slip_quality
from src.drift import check_model_health, compute_crps_batch, compute_ece
from src.slip_ev import simulate_slip_ev, quick_slip_ev, build_correlation_matrix
from src.board_logger import log_board_snapshot, mark_as_bet
from src.line_snapshots import snapshot_pp_lines, detect_stale_lines, get_line_movement_summary
from src.consistency import enforce_consistency, flag_inconsistencies


# ─────────────────────────────────────────────
# CACHED WRAPPERS — eliminate redundant API calls across Streamlit reruns
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_pp_lines():
    """PrizePicks lines — cached 5 min."""
    return fetch_prizepicks_mlb_lines()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_todays_games():
    """MLB schedule + probable pitchers — cached 1 hr."""
    return fetch_todays_games()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_umpires():
    """Home-plate umpire map — cached 1 hr."""
    return fetch_todays_umpires()

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_weather(team_abbr: str):
    """Weather per stadium — cached 30 min."""
    return fetch_game_weather(team_abbr)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_sharp_events(api_key: str):
    """Sharp book events list — cached 5 min."""
    return fetch_mlb_events(api_key)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_event_props(event_id: str, api_key: str):
    """Props for one event — cached 5 min (saves 1 API credit per hit)."""
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
        .stApp [data-testid="stDataFrame"] { overflow-x: auto !important; }
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .hero-logo { font-size: 1.35rem; }
        .card .val { font-size: 1.25rem; }
        .stApp button[kind="secondary"], .stApp [data-testid="stTab"] { min-height: 44px; }
        .pick-card { padding: 0.8rem 0.9rem; }
    }
</style>
""", unsafe_allow_html=True)

def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v)
def badge(r): return f'<span class="badge badge-{r.lower()}">{r}</span>'
def pick_span(p): return f'<span class="{"more" if p=="MORE" else "less"}">{p}</span>'
def grade_label(r):
    icons = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
    return f"{icons.get(r, '⚪')} {r}"


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


def match_player_stats(player_name: str, batting_df: pd.DataFrame) -> pd.Series:
    """Match PrizePicks player name to FanGraphs batting row."""
    if batting_df.empty or "Name" not in batting_df.columns:
        return None
    norm_target = _normalize_name(player_name)
    for idx, row in batting_df.iterrows():
        if _normalize_name(str(row.get("Name", ""))) == norm_target:
            return row
    parts = norm_target.split()
    if len(parts) >= 2:
        last = parts[-1]
        first_init = parts[0][0] if parts[0] else ""
        for idx, row in batting_df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            rparts = rn.split()
            if len(rparts) >= 2:
                if rparts[-1] == last and rparts[0] and rparts[0][0] == first_init:
                    return row
    if parts:
        last = parts[-1]
        matches = []
        for idx, row in batting_df.iterrows():
            rn = _normalize_name(str(row.get("Name", "")))
            if rn.split()[-1] == last if rn.split() else False:
                matches.append(row)
        if len(matches) == 1:
            return matches[0]
    return None


def build_batter_profile(stats_row: pd.Series) -> dict:
    """Convert a FanGraphs DataFrame row into the batter profile dict the predictor expects."""
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
        except Exception:
            pass
    return df


def match_pitcher_stats(player_name: str, pitching_df: pd.DataFrame):
    """Match PrizePicks player name to FanGraphs pitching row."""
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


def build_pitcher_profile(stats_row) -> dict:
    """Convert a FanGraphs pitching row into the pitcher profile dict the predictor expects."""
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
_freshness_str = datetime.now().strftime("%I:%M %p").lstrip("0")

st.markdown(f"""<div class="hero-wrapper">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div class="hero-logo">⚾ MLB Prop <span class="accent">Edge</span></div>
      <div class="hero-sub">Sharp-Based Prop Predictions &nbsp;·&nbsp; Statcast Confirmed &nbsp;·&nbsp; Self-Learning Model</div>
      <div class="hero-meta">
        <div class="hero-meta-pill"><span class="pip"></span>Model <strong>{_model_ver}</strong> active</div>
        <div class="hero-meta-pill"><span class="pip amber"></span>Refreshed <strong>{_freshness_str}</strong></div>
        <div class="hero-meta-pill"><span class="pip blue"></span>FanDuel · Pinnacle · DraftKings</div>
      </div>
    </div>
    <div style="text-align:right;padding-top:0.2rem;">
      <div style="font-size:0.6rem;color:rgba(232,236,241,0.2);letter-spacing:2px;text-transform:uppercase;">Season 2026</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:rgba(232,236,241,0.5);font-weight:600;margin-top:0.2rem;">+EV FINDER</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

tab_edge, tab_slips, tab_picks, tab_dash, tab_grade, tab_setup = st.tabs(["🎯 FIND EDGES", "🎫 MY SLIPS", "📋 ALL LINES", "📊 DASHBOARD", "✅ GRADE", "⚙️ SETUP"])

with tab_edge:
    api_key = get_api_key()
    has_sharp = bool(api_key)
    # Check if we're in preseason
    _days_to_opening = (date(2026, 3, 27) - date.today()).days
    _is_preseason = _days_to_opening > 0

    if not api_key:
        st.markdown(
            '<div class="alert-strip"><strong>No Odds API key found.</strong> '
            'Sharp book comparison is disabled. Create a <code>.env</code> file in the project folder with: '
            '<code>ODDS_API_KEY=your_key_here</code><br>'
            'Free key at <a href="https://the-odds-api.com" target="_blank">the-odds-api.com</a> — 500 req/month, no credit card.</div>',
            unsafe_allow_html=True
        )
    if _is_preseason:
        st.markdown(
            f'<div class="warn-strip"><strong>Preseason:</strong> Opening Day is {_days_to_opening} day{"s" if _days_to_opening != 1 else ""} away (March 27). '
            f'Sharp books won\'t have MLB player props until close to Opening Day. '
            f'Projection-based analysis is still available from PrizePicks lines.</div>',
            unsafe_allow_html=True
        )
    else:
        usage = get_api_usage(api_key)
        st.markdown(f'<div class="info-strip">Odds API active &nbsp;·&nbsp; <span class="hl">{usage.get("remaining","?")}</span> credits remaining &nbsp;·&nbsp; Sharp books: FanDuel · Pinnacle · DraftKings</div>', unsafe_allow_html=True)

    with st.spinner("Pulling PrizePicks MLB lines..."):
        try: pp_lines = _cached_pp_lines()
        except: pp_lines = pd.DataFrame()

    # Snapshot PP lines for CLV tracking and stale-line detection
    if not pp_lines.empty:
        try:
            snapshot_pp_lines(pp_lines)
        except Exception:
            pass

    if pp_lines.empty:
        st.info("No MLB lines on PrizePicks right now. Lines usually post by 10 AM ET.")
    else:
        st.markdown(f"**{len(pp_lines)} MLB props** on PrizePicks today")
        all_edges = []
        if has_sharp:
            total_sharp_lines = 0
            events_with_props = 0
            with st.spinner("Fetching sharp lines & devigging..."):
                events = _cached_sharp_events(api_key)
                for event in (events or [])[:15]:
                    eid = event.get("id","")
                    if not eid: continue
                    result = _cached_event_props(eid, api_key=api_key)
                    if result and "data" in result:
                        sharp = extract_sharp_lines(result["data"])
                        if sharp:
                            events_with_props += 1
                            total_sharp_lines += len(sharp)
                        all_edges.extend(find_ev_edges(pp_lines, sharp, min_ev_pct=0.25))
            st.caption(f"Scanned {len(events or [])} events · {events_with_props} had props · {total_sharp_lines} sharp lines · {len(all_edges)} edges")

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

            f1,f2,f3 = st.columns([2,2,2])
            with f1: min_grade = st.selectbox("Min grade", ["A only","A + B","A + B + C","All"], index=1)
            with f2: prop_f = st.selectbox("Prop type", ["All","Pitcher Ks","Batter Hits","Total Bases","Home Runs"])
            with f3: show_all_sharp = st.checkbox("Show all picks (incl. non-tradeable)", value=False, key="show_all_sharp")
            gm = {"A only":["A"],"A + B":["A","B"],"A + B + C":["A","B","C"],"All":["A","B","C","D"]}
            filt = [e for e in all_edges if e["rating"] in gm[min_grade]]
            if prop_f=="Pitcher Ks": filt=[e for e in filt if "strikeout" in e.get("market","").lower() and "batter" not in e.get("market","").lower()]
            elif prop_f=="Batter Hits": filt=[e for e in filt if "hits" in e.get("market","").lower()]
            elif prop_f=="Total Bases":
                filt=[e for e in filt if "total_bases" in e.get("market","").lower()]
                less_tb = [e for e in filt if e.get("pick") == "LESS"]
                if less_tb:
                    st.markdown('<div class="warn-strip"><strong>TB LESS Warning:</strong> Total Bases LESS picks historically underperform — trade with extra caution</div>', unsafe_allow_html=True)
            elif prop_f=="Home Runs": filt=[e for e in filt if "home_run" in e.get("market","").lower()]
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
                st.dataframe(disp, hide_index=True, use_container_width=True, height=min(len(disp)*38+40,600))

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
                    log_batch_predictions(filt, date.today().isoformat())
                    st.success(f"Saved {len(filt)} predictions!")
            else: st.info("No edges match filters.")
        elif has_sharp:
            if _is_preseason:
                st.info("Sharp book player props aren't available until Opening Day. Projection-based analysis is below.")
            else:
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
            # Fetch today's games for lineup/batting order data
            try:
                todays_games = _cached_todays_games()
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
            opp_pitcher_lookup = {}  # team_abbr → dict with pitcher info + FanGraphs profile
            opp_team_k_lookup = {}   # team_abbr → team K% for pitcher K projections
            try:
                for game in todays_games:
                    home_team = game.get("home_team", "")
                    away_team = game.get("away_team", "")
                    # Home batters face the AWAY pitcher
                    if away_team and game.get("away_pitcher_name") != "TBD":
                        opp_info = {
                            "name": game.get("away_pitcher_name", ""),
                            "hand": game.get("away_pitcher_hand", ""),
                            "id": game.get("away_pitcher_id"),
                        }
                        # Try to find opposing pitcher in FanGraphs stats
                        if not pitching_df.empty:
                            opp_matched = match_pitcher_stats(opp_info["name"], pitching_df)
                            if opp_matched is not None:
                                opp_info["profile"] = build_pitcher_profile(opp_matched)
                        opp_pitcher_lookup[home_team] = opp_info
                    # Away batters face the HOME pitcher
                    if home_team and game.get("home_pitcher_name") != "TBD":
                        opp_info = {
                            "name": game.get("home_pitcher_name", ""),
                            "hand": game.get("home_pitcher_hand", ""),
                            "id": game.get("home_pitcher_id"),
                        }
                        if not pitching_df.empty:
                            opp_matched = match_pitcher_stats(opp_info["name"], pitching_df)
                            if opp_matched is not None:
                                opp_info["profile"] = build_pitcher_profile(opp_matched)
                        opp_pitcher_lookup[away_team] = opp_info

                    # Build opposing team K rate for pitcher K projections
                    # Pitcher on home team faces away lineup, and vice versa
                    if not batting_df.empty:
                        for pitcher_team, opp_batting_team in [(home_team, away_team), (away_team, home_team)]:
                            if pitcher_team:
                                # Average K% of opposing team's batters in our FanGraphs data
                                team_batters = batting_df[
                                    batting_df["Team"].str.contains(opp_batting_team, case=False, na=False)
                                ] if "Team" in batting_df.columns else pd.DataFrame()
                                if not team_batters.empty and "K%" in team_batters.columns:
                                    try:
                                        k_vals = team_batters["K%"].apply(
                                            lambda x: float(str(x).replace("%", "").strip()) if pd.notna(x) else 22.7
                                        )
                                        opp_team_k_lookup[pitcher_team] = round(k_vals.mean(), 1)
                                    except Exception:
                                        pass
            except Exception:
                pass

            # Pre-build batter hand + batting order caches from confirmed lineups
            # Batch all game lineups upfront (cached per game_pk) to avoid N+1
            batter_hand_cache = {}   # player_name_upper → bat_hand (R/L/S)
            batting_order_cache = {} # player_name_upper → batting_order (1-9)
            game_context_cache = {}  # team_abbr → {opponent, game_time, venue, ...}
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
                        for batter in lineups.get(side, []):
                            name = batter.get("player_name", "").upper().strip()
                            hand = batter.get("bat_hand", "")
                            order = batter.get("batting_order")
                            if name and hand:
                                batter_hand_cache[name] = hand
                            if name and order:
                                batting_order_cache[name] = order

                    # Build game context for both teams (no extra API call)
                    for team_abbr, opp_abbr, is_home in [
                        (home_team, away_team, True),
                        (away_team, home_team, False),
                    ]:
                        if team_abbr:
                            game_context_cache[team_abbr] = {
                                "opponent": opp_abbr,
                                "game_time": game.get("game_time", ""),
                                "game_pk": game_pk,
                                "is_home": is_home,
                            }
            except Exception:
                pass

            prog = st.progress(0, text="Running projections...")
            total = len(pp_lines)
            for i, (_, row) in enumerate(pp_lines.iterrows()):
                prog.progress((i + 1) / total, text=f"Projecting {i + 1}/{total}...")
                team = row.get("team","")
                stat_int = row["stat_internal"]
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
                if not is_pitcher_prop:
                    batting_pos = batting_order_cache.get(
                        row["player_name"].upper().strip()
                    )

                # Look up opposing pitcher profile (for batter props)
                opp_pitcher_profile = None
                platoon_adj = None
                if not is_pitcher_prop and r_team and r_team in opp_pitcher_lookup:
                    opp_info = opp_pitcher_lookup[r_team]
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
                    opp_k_rate = opp_team_k_lookup.get(r_team)

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

                # Props that are count-based (safe to apply multipliers to)
                # home_runs returns a PROBABILITY (0-1), not a count — never multiply it
                _COUNT_PROPS = {"hits", "total_bases", "rbis", "runs", "stolen_bases",
                                "hits_runs_rbis", "batter_strikeouts", "walks", "singles", "doubles",
                                "pitcher_strikeouts", "pitching_outs", "earned_runs",
                                "walks_allowed", "hits_allowed", "hitter_fantasy_score"}
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
                spring_mult = spring["spring_mult"]
                # Only apply spring multiplier to count-based props (not probabilities)
                if _is_count_prop:
                    p["projection"] = round(p["projection"] * spring_mult, 2)
                p["spring_mult"] = spring_mult
                p["spring_badge"] = spring["badge"]

                trend = get_batter_trend(0)
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

                injury = get_player_injury_status(
                    player_name=row["player_name"],
                    injuries=injury_list,
                )
                p["injury_status"] = injury["status"]
                p["injury_color"] = injury["color"]

                if injury["status"] == "IL":
                    continue
                if injury["status"] == "day-to-day":
                    p["confidence"] = max(p.get("confidence", 0.5) * 0.85, 0)
                    if p.get("rating") in ("A", "A+"):
                        p["rating"] = "B"

                # v018: Lineup position display info
                # NOTE: PA adjustment now handled INSIDE generate_prediction() via
                # lineup_pos → estimate_plate_appearances() (Task 3A). No post-hoc
                # multiplier needed — that would double-count the effect.
                if batting_pos:
                    p["batting_order"] = batting_pos
                    p["pa_multiplier"] = round(get_pa_multiplier(batting_pos), 3)

                # v018: Game context from pre-built cache (no API call)
                if r_team and r_team in game_context_cache:
                    gctx = game_context_cache[r_team]
                    p["opponent"] = gctx.get("opponent", "")
                    p["game_time"] = gctx.get("game_time", "")
                    # Opposing pitcher name from the pre-built lookup
                    if r_team in opp_pitcher_lookup:
                        p["opp_pitcher"] = opp_pitcher_lookup[r_team].get("name", "")

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
                preds.append(p)
            prog.empty()

            # v018: Cross-prop consistency checks (TB >= Hits, etc.)
            if preds:
                try:
                    preds = enforce_consistency(preds)
                except Exception:
                    pass

            if preds:
                # v018: Save projected stats for tracking accuracy over time
                try:
                    stats_to_save = [{
                        "game_date": date.today().isoformat(),
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
                except Exception:
                    pass

                # v018: Log full board snapshot (all props, not just selected)
                try:
                    log_board_snapshot(preds, edges=all_edges)
                except Exception:
                    pass

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

                with st.expander("Betting Rules & Bankroll Guide", expanded=False):
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("**BET THESE PROPS**")
                        st.markdown(
                            "- Pitcher Strikeouts (MORE and LESS)\n"
                            "- Hitter Fantasy Score (MORE and LESS)\n"
                            "- Total Bases MORE\n"
                            "- Hits (MORE and LESS)\n"
                            "- Hits+Runs+RBIs (H+R+RBI)\n"
                            "- Batter Strikeouts"
                        )
                        st.markdown("**AVOID THESE**")
                        st.markdown(
                            "- Total Bases LESS (44% — not worth it)\n"
                            "- Home Runs LESS (rarely offered)\n"
                            "- SB LESS 0.5 (not offered)"
                        )
                    with rc2:
                        st.markdown("**SLIP RULES**")
                        st.markdown(
                            "- Mix 2–3 MORE + 2–3 LESS per slip\n"
                            "- Max 2 picks per team per slip\n"
                            "- Min B grade — never take D-grade\n"
                            "- Prefer 5-Pick or 6-Pick Flex\n"
                            "- Avoid 2-Pick or 3-Pick Power Play"
                        )
                        st.markdown("**BANKROLL**")
                        st.markdown(
                            "- 1–2% of bankroll per slip\n"
                            "- Max 5 slips per day\n"
                            "- Break-even: 5-pick = 54.2%, 6-pick = 52.9%\n"
                            "- Stop if down 10% in a day"
                        )

                _TRIVIAL_LESS_PROPS = {"stolen_bases", "home_runs"}
                def _is_trivial(pick: dict) -> bool:
                    return (
                        pick.get("pick") == "LESS"
                        and float(pick.get("line", 99)) <= 0.5
                        and pick.get("stat_type", "").lower().replace(" ", "_") in _TRIVIAL_LESS_PROPS
                    )

                if scored_all:
                    top_plays = [s for s in scored_all
                                 if s["combined_grade"] in ("A+", "A")
                                 and is_tradeable_pick(s.get("stat_internal", s.get("stat_type", "")), s.get("pick", ""))
                                 and not _is_trivial(s)][:5]
                else:
                    top_plays = []
                if not top_plays:
                    tradeable_a = pdf[(pdf["rating"] == "A") & pdf.apply(
                        lambda r: is_tradeable_pick(r.get("stat_internal", ""), r.get("pick", "")), axis=1
                    )]
                    for _, tp in tradeable_a.head(5).iterrows():
                        top_plays.append({
                            "player_name": tp["player_name"],
                            "stat_type": tp["stat_type"],
                            "line": tp["line"],
                            "pick": tp["pick"],
                            "combined_score": tp.get("edge", 0),
                            "combined_grade": "A",
                            "signal": SIGNAL_PROJECTION_ONLY,
                            "proj_confidence": tp.get("confidence", 0.5),
                        })
                        if len(top_plays) >= 5:
                            break

                if top_plays:
                    st.markdown('<div class="section-hdr">Today\'s Best Plays</div>', unsafe_allow_html=True)
                    _bp_grade_emoji = {"A+": "💎", "A": "⬆", "B": "▲", "C": "◆", "D": "▽"}
                    _bp_signal_labels = {SIGNAL_CONFIRMED: ("confirmed", "CONFIRMED"), SIGNAL_SHARP_ONLY: ("sharp", "SHARP"), SIGNAL_PROJECTION_ONLY: ("proj", "PROJECTION")}
                    bp_cols = st.columns(min(len(top_plays), 5))
                    for idx, tp in enumerate(top_plays[:5]):
                        pick_cls = "more" if tp["pick"] == "MORE" else "less"
                        sig_key, sig_label = _bp_signal_labels.get(tp.get("signal", ""), ("proj", ""))
                        grade_icon = _bp_grade_emoji.get(tp.get("combined_grade", ""), "")
                        conf_val = tp.get("proj_confidence", 0) or 0
                        conf_pct = int(conf_val * 100)
                        conf_cls = "high" if conf_val > 0.6 else ("med" if conf_val > 0.52 else "low")
                        with bp_cols[idx]:
                            st.markdown(f'''<div class="best-play">
                                <div class="bp-name">{tp["player_name"]}</div>
                                <div class="bp-prop">{tp["stat_type"]} · Line {tp["line"]}</div>
                                <div class="bp-pick {pick_cls}">{tp["pick"]}</div>
                                <div class="conf-track"><div class="conf-fill {conf_cls}" style="width:{min(conf_pct,100)}%"></div></div>
                                <div class="bp-conf">{grade_icon} {tp.get("combined_grade","?")} · {conf_pct}% conf · {sig_label}</div>
                            </div>''', unsafe_allow_html=True)

                # ── News & Transactions Feed ──
                with st.expander("📰 Recent Transactions & News", expanded=False):
                    try:
                        _recent_txns = fetch_recent_transactions(days_back=3)
                        if _recent_txns:
                            _news_html = []
                            for _txn in _recent_txns[:20]:
                                _txn_date = _txn.get("date", "")[:10]
                                _news_html.append(
                                    f'<div style="display:flex;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">'
                                    f'<span style="min-width:24px">{_txn["icon"]}</span>'
                                    f'<span style="color:rgba(232,236,241,0.4);min-width:55px;font-family:JetBrains Mono;font-size:0.72rem;">{_txn.get("team","")}</span>'
                                    f'<span style="color:#E8ECF1;flex:1">{_txn["description"][:120]}</span>'
                                    f'<span style="color:rgba(232,236,241,0.25);font-size:0.7rem;min-width:70px">{_txn_date}</span>'
                                    f'</div>'
                                )
                            st.markdown("".join(_news_html), unsafe_allow_html=True)
                        else:
                            st.caption("No recent transactions found.")
                    except Exception:
                        st.caption("Transaction feed unavailable.")

                st.markdown('<div class="section-hdr">Filter Picks</div>', unsafe_allow_html=True)
                prop_types_available = sorted(pdf["stat_type"].unique().tolist())
                f1, f2, f3 = st.columns([3, 2, 2])
                with f1:
                    proj_prop_filter = st.radio("Prop Type", ["All"] + prop_types_available[:6], horizontal=True, key="proj_prop_f")
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
                    spring_icon = {"hot": "🔥", "cold": "❄️", "neutral": "—"}.get(pick_row.get("spring_badge", "neutral"), "—")
                    trend_icon = {"hot": "🔥", "cold": "❄️", "neutral": "—"}.get(pick_row.get("trend_badge", "neutral"), "—")
                    buy_tag = " 🎯" if pick_row.get("buy_low") else ""
                    _lt = pick_row.get("line_type", "standard")
                    promo_tag = " 👺" if _lt == "promo" else (" 💰" if _lt in ("discounted", "flash_sale") else "")
                    pick_cls = "more-pick" if pick_row["pick"] == "MORE" else "less-pick"

                    chk_col, info_col = st.columns([0.06, 0.94])
                    with chk_col:
                        checked = st.checkbox("", key=f"proj_pick_{pick_idx}", label_visibility="collapsed")
                    with info_col:
                        conf_val = pick_row['confidence']
                        conf_pct = int(conf_val * 100)
                        edge_val = pick_row['edge']*100
                        edge_pct = f"{edge_val:.1f}%"
                        proj = pick_row.get('projection', 0)

                        pick_card_html = f'''<div class="pick-card {pick_cls}">
                            <div class="pick-card-header">
                                <span class="badge badge-{pick_row['rating'].lower()}">{pick_row['rating']}</span>
                                <span class="pick-card-player">{pick_row['player_name']}{promo_tag}</span>
                                <span class="pick-card-team">{pick_row.get('team', '')}</span>
                                <span class="dir-chip {"more" if pick_row["pick"]=="MORE" else "less"}">{pick_row["pick"]}</span>
                            </div>
                            <div class="pick-card-row">
                                <span class="pick-card-stat">{pick_row['stat_type']}</span>
                                <span class="pick-card-line">Line {pick_row['line']}</span>
                                <span class="pick-card-proj">Proj {proj:.2f}</span>
                                <span class="pick-card-edge" style="color:#00C853;">+{edge_pct}</span>
                            </div>
                            <div class="pick-card-conf">
                                <div class="conf-track" style="flex:1"><div class="conf-fill {"high" if conf_val>0.6 else ("med" if conf_val>0.52 else "low")}" style="width:{min(conf_pct,100)}%"></div></div>
                                <span class="pick-card-conf-label">{conf_pct}%</span>
                            </div>
                        </div>'''
                        st.markdown(pick_card_html, unsafe_allow_html=True)

                        # Expandable detail: projected statline + game context
                        with st.expander("Details", expanded=False):
                            det1, det2 = st.columns(2)
                            with det1:
                                st.markdown("**Game Context**")
                                _opp = pick_row.get("opponent", "—")
                                _venue = pick_row.get("venue", "—")
                                _opp_p = pick_row.get("opp_pitcher", "—")
                                _bat_ord = pick_row.get("batting_order")
                                _gt = pick_row.get("game_time", "")
                                ctx_lines = []
                                if _opp and _opp != "—":
                                    ctx_lines.append(f"**vs** {_opp}")
                                if _opp_p and _opp_p != "—":
                                    ctx_lines.append(f"**Opp Pitcher:** {_opp_p}")
                                if _bat_ord:
                                    ctx_lines.append(f"**Batting:** {_bat_ord}{'st' if _bat_ord==1 else 'nd' if _bat_ord==2 else 'rd' if _bat_ord==3 else 'th'} in order")
                                if _venue and _venue != "—":
                                    ctx_lines.append(f"**Park:** {_venue}")
                                # Park factor
                                _pk_team = resolve_team(pick_row.get("team", "")) if pick_row.get("team") else None
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
                                proj_val = pick_row.get("projection", 0)
                                line_val = pick_row.get("line", 0)
                                diff = proj_val - line_val
                                diff_color = "#00C853" if (diff > 0 and pick_row["pick"] == "MORE") or (diff < 0 and pick_row["pick"] == "LESS") else "#FF4444"
                                st.markdown(f'**{pick_row["stat_type"]}:** <span style="font-family:JetBrains Mono;font-weight:700;color:{diff_color}">{proj_val:.2f}</span> vs line {line_val}  ({"+" if diff>=0 else ""}{diff:.2f})', unsafe_allow_html=True)
                                st.markdown(f"**Confidence:** {conf_pct}% · **Edge:** +{edge_pct}")
                                if pick_row.get("batting_order"):
                                    st.markdown(f"**PA Multiplier:** {pick_row.get('pa_multiplier', 1.0):.2f}x (lineup spot #{pick_row['batting_order']})")
                                # Health/form
                                _health = pick_row.get("injury_status", "active")
                                _spring = pick_row.get("spring_badge", "neutral")
                                _trend = pick_row.get("trend_badge", "neutral")
                                status_parts = []
                                status_parts.append(f"Health: {health_icon}")
                                if _spring != "neutral":
                                    status_parts.append(f"Spring: {spring_icon}")
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
                                        _sg_probs = [sp.get("confidence", 0.55) for sp in _sg["picks"]]
                                        _sg_ev = quick_slip_ev(_sg_probs, entry_type=f"{_slip_size}_flex")
                                        _sg_ev_pct = _sg_ev["ev_profit_pct"]
                                        _sg_ev_color = "#00C853" if _sg_ev_pct > 0 else ("#FFB300" if _sg_ev_pct > -10 else "#FF4444")
                                        _sg_ev_label = f"{_sg_ev_pct:+.1f}%"
                                        st.markdown(
                                            f'<div style="margin-top:0.4rem;padding:0.4rem 0.6rem;background:rgba(0,100,200,0.04);border:1px solid rgba(0,100,200,0.08);border-radius:8px;font-size:0.72rem;">'
                                            f'<span style="color:rgba(232,236,241,0.4);">EV:</span> '
                                            f'<span style="font-family:JetBrains Mono;font-weight:700;color:{_sg_ev_color};">{_sg_ev_label}</span>'
                                            f' · <span style="color:rgba(232,236,241,0.4);">Perfect:</span> '
                                            f'<span style="font-family:JetBrains Mono;font-size:0.7rem;color:rgba(232,236,241,0.5);">{_sg_ev["prob_perfect"]*100:.1f}%</span>'
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
                                        f'<span style="color:rgba(232,236,241,0.4);">Kelly Wager:</span> '
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
                    st.dataframe(slip_df[["player_name","stat_type","line","pick","rating","confidence"]], hide_index=True, use_container_width=True)

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

                    slip_type = st.selectbox("Slip type", ["6_flex", "5_flex", "4_flex", "3_power", "2_power"], key="slip_type_select")
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
                            _sel_picks_for_kelly = [{"confidence": p.get("confidence", 0.55)} for _, p in slip_df.iterrows()]
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
                                f'Kelly Suggested Wager: <span class="hl">${_k_rec:.2f}</span> · '
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
                                    "win_prob": _sp.get("confidence", 0.55),
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
                            picks_for_slip = [{"player_name":p['player_name'],"stat_type":p['stat_type'],"line":p['line'],"pick":p['pick']} for _,p in slip_df.iterrows()]
                            slip_id = create_slip(date.today().isoformat(), slip_type, slip_amt, picks_for_slip)
                            st.success(f"Slip #{slip_id} created!")
                            st.rerun()
                        else:
                            st.warning(f"Select exactly {_num_needed} picks for a {slip_type}.")

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
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("Create New Slip"):
        sl_date = st.date_input("Game date", value=date.today(), key="slip_date")
        sl_type = st.selectbox("Entry type", ["5_flex", "6_flex", "4_flex", "3_power", "2_power"])
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

with tab_picks:
    st.markdown('<div class="section-hdr">All PrizePicks MLB Lines</div>', unsafe_allow_html=True)
    if pp_lines.empty:
        st.info("No lines available right now. Lines usually post by 10 AM ET.")
    else:
        st.markdown(f'<div class="info-strip"><span class="hl">{len(pp_lines)}</span> props on the board today</div>', unsafe_allow_html=True)
        s = st.text_input("Search player", "", key="s2", placeholder="Type player name...")
        sf = st.multiselect("Filter prop types", get_available_stat_types(pp_lines), default=get_available_stat_types(pp_lines), key="sf2")
        fa = pp_lines.copy()
        if s: fa = fa[fa["player_name"].str.contains(s, case=False, na=False)]
        if sf: fa = fa[fa["stat_type"].isin(sf)]
        da = fa[["player_name","team","stat_type","line","start_time"]].copy()
        da.columns = ["Player","Team","Prop","Line","Time"]
        if "Time" in da.columns: da["Time"] = pd.to_datetime(da["Time"], errors="coerce").dt.strftime("%-I:%M %p")
        st.dataframe(da, hide_index=True, use_container_width=True)

with tab_dash:
    st.markdown('<div class="section-hdr">Performance Dashboard</div>', unsafe_allow_html=True)
    stats = get_accuracy_stats()
    if stats["total"]==0:
        st.info("No graded picks yet. Use the Grade tab to log results after games.")
    else:
        acc=stats["accuracy"]
        acc_cls = "good" if acc>=0.55 else ("ok" if acc>=0.50 else "bad")
        target_hit = acc >= 0.542
        da1, da2 = st.columns([1, 2])
        with da1:
            st.markdown(f'''<div style="text-align:center;padding:1.5rem 1rem;background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:14px;">
                <div class="acc-big {acc_cls}">{pct(acc)}</div>
                <div class="acc-label">Overall Accuracy</div>
                <div style="margin-top:0.8rem;font-size:0.75rem;color:rgba(232,236,241,0.4);">{stats["wins"]}W · {stats["losses"]}L · {stats["total"]} graded</div>
                <div style="margin-top:0.5rem;">
                    <span style="font-size:0.72rem;padding:0.2rem 0.6rem;border-radius:20px;{"background:rgba(0,200,83,0.12);color:#00C853" if target_hit else "background:rgba(255,68,68,0.1);color:#FF4444"}">
                        {"✓ Above 54.2% target" if target_hit else "✗ Below 54.2% target"}
                    </span>
                </div>
            </div>''', unsafe_allow_html=True)
        with da2:
            st.markdown("**Accuracy by Grade**")
            grade_rows = []
            for r in "ABCD":
                gr = stats.get('by_rating', {}).get(r, {})
                gr_acc = gr.get('accuracy', 0)
                gr_w = gr.get('wins', 0)
                gr_t = gr.get('total', 0)
                gr_l = gr_t - gr_w
                bar_w = int(gr_acc * 100)
                bar_cls = "high" if gr_acc > 0.55 else ("med" if gr_acc > 0.50 else "low")
                grade_rows.append(f'''<div style="display:flex;align-items:center;gap:0.8rem;padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span class="badge badge-{r.lower()}" style="width:20px;text-align:center">{r}</span>
                    <div style="flex:1"><div class="conf-track"><div class="conf-fill {bar_cls}" style="width:{bar_w}%"></div></div></div>
                    <span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#E8ECF1;min-width:45px">{pct(gr_acc)}</span>
                    <span style="font-size:0.72rem;color:rgba(232,236,241,0.35);min-width:55px">{gr_w}W-{gr_l}L</span>
                </div>''')
            st.markdown("".join(grade_rows), unsafe_allow_html=True)

            st.markdown("**More vs Less**")
            dir_cols = st.columns(2)
            for i, d in enumerate(["MORE", "LESS"]):
                dv = stats.get('by_direction', {}).get(d, {})
                d_acc = dv.get('accuracy', 0)
                d_w = dv.get('wins', 0)
                d_t = dv.get('total', 0)
                d_color = "#00C853" if d == "MORE" else "#FF4444"
                with dir_cols[i]:
                    st.markdown(f'<div class="card"><div class="lbl">{d}</div><div class="val" style="color:{d_color}">{pct(d_acc)}</div><div class="sub">{d_w}W-{d_t-d_w}L</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Model Tuning &amp; Weight History</div>', unsafe_allow_html=True)
    tune_col1, tune_col2 = st.columns(2)
    with tune_col1:
        if st.button("Run Model Tuning", type="secondary"):
            with st.spinner("Analyzing graded picks and adjusting weights..."):
                try:
                    result = run_adjustment_cycle(min_sample=25)
                    if result.get("adjusted"):
                        st.success(f"Weights adjusted! New version: {result.get('version_new', '?')}")
                        if result.get("changes"):
                            for ch in result["changes"][:5]:
                                st.markdown(f"- {ch.get('description', ch)}")
                    else:
                        st.info(result.get("reason", "No adjustments needed"))
                except Exception as e:
                    st.error(f"Tuning failed: {e}")
    with tune_col2:
        try:
            history = get_weight_history()
            if history:
                st.markdown(f"**{len(history)} adjustments** in history")
                for h in history[-3:]:
                    st.caption(f"{h.get('timestamp', '?')}: {h.get('description', 'adjustment')}")
        except Exception:
            st.caption("No adjustment history yet")

    # ── NIGHTLY SELF-IMPROVEMENT CYCLE ────────────────────────
    st.markdown('<div class="section-hdr">Self-Improvement Cycle</div>', unsafe_allow_html=True)

    ni_col1, ni_col2 = st.columns([1, 3])
    with ni_col1:
        if st.button("Run Nightly Update", type="primary", help="Grade today's picks, update weights, check drift"):
            with st.spinner("Running self-improvement cycle..."):
                try:
                    from src.nightly import run_nightly_cycle
                    nightly_res = run_nightly_cycle()
                    st.session_state["nightly_results"] = nightly_res
                except Exception as e:
                    st.error(f"Nightly cycle failed: {e}")

    if "nightly_results" in st.session_state:
        nr = st.session_state["nightly_results"]

        # Grading summary
        grading = nr.get("phase_results", {}).get("grading", {})
        if grading.get("total_graded", 0) > 0:
            g_w = grading.get("wins", 0)
            g_l = grading.get("losses", 0)
            g_t = grading.get("total_graded", 0)
            g_acc = g_w / g_t if g_t else 0
            gm1, gm2, gm3, gm4 = st.columns(4)
            gm1.metric("Graded", g_t)
            gm2.metric("Record", f"{g_w}W-{g_l}L")
            gm3.metric("Hit Rate", f"{g_acc:.1%}")
            gm4.metric("Pending", grading.get("pending", 0))

        # Metrics from autolearn
        metrics = nr.get("phase_results", {}).get("metrics", {})
        if metrics.get("brier_score") is not None:
            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("Brier Score", f"{metrics['brier_score']:.4f}",
                        help="Lower is better. 0.25 = coin flip.")
            mm2.metric("Log Loss", f"{metrics.get('log_loss', 0):.4f}")
            mm3.metric("Accuracy", f"{metrics.get('accuracy_before', 0):.1%}" if metrics.get("accuracy_before") else "—")

        # Weight update
        w_info = nr.get("phase_results", {}).get("weights", {})
        if w_info.get("updated"):
            st.success(f"Ensemble weights updated — sharp: {w_info['new_weights'].get('sharp_odds', 0):.0%} / "
                       f"projection: {w_info['new_weights'].get('projection', 0):.0%} / "
                       f"form: {w_info['new_weights'].get('recent_form', 0):.0%}")
        elif w_info.get("reason"):
            st.info(f"Weights not updated: {w_info['reason']}")

        # Drift alerts
        for alert in nr.get("drift_alerts", []):
            if isinstance(alert, dict):
                st.warning(f"Drift: {alert.get('prop_type', '?')} degraded {alert.get('degradation_pct', '?')}%")
            else:
                st.warning(f"Drift: {alert}")

        # Calibration warnings
        for warn in nr.get("calibration_warnings", []):
            st.warning(f"Miscalibrated {warn['bin']}: predicted {warn['predicted']:.1%} vs actual {warn['actual']:.1%} (n={warn['n']})")

        # Errors
        for err in nr.get("errors", []):
            st.error(err)

    # ── MODEL HEALTH (historical) ──────────────────────────
    st.markdown('<div class="section-hdr">Model Health (14-day trend)</div>', unsafe_allow_html=True)
    try:
        from src.nightly import get_nightly_history
        health_rows = get_nightly_history(days=14)
        if health_rows:
            health_df = pd.DataFrame(health_rows)
            if "accuracy" in health_df.columns and health_df["accuracy"].notna().any():
                chart_df = health_df[["date", "accuracy", "brier"]].dropna(subset=["accuracy"]).set_index("date")
                st.line_chart(chart_df)

            # Current ensemble weights
            try:
                w_path = Path("data/weights/current.json")
                if w_path.exists():
                    with open(w_path) as f:
                        cw = json.load(f)
                    ew = cw.get("ensemble_weights", {})
                    if ew:
                        ew_str = " · ".join(f"{k}: {v:.0%}" for k, v in ew.items())
                        st.caption(f"Current ensemble weights: {ew_str}")
            except Exception:
                pass
        else:
            st.caption("No nightly logs yet. Run the nightly update after games finish (~11 PM ET).")
    except Exception:
        st.caption("No nightly logs yet.")

    st.markdown('<div class="section-hdr">Backtest Accuracy — v017 (Played Games Only)</div>', unsafe_allow_html=True)
    _bt_rows = [
        {"Prop": "Hits LESS 1.5",        "W": 19494, "Total": 26875, "Accuracy": 0.725},
        {"Prop": "Pitcher Ks MORE 4.5",   "W":   353, "Total":   507, "Accuracy": 0.696},
        {"Prop": "Pitcher Ks LESS 4.5",   "W":   297, "Total":   429, "Accuracy": 0.692},
        {"Prop": "TB LESS 1.5",           "W":   185, "Total":   281, "Accuracy": 0.658},
        {"Prop": "FS LESS 7.5",           "W":  1058, "Total":  1715, "Accuracy": 0.617},
        {"Prop": "FS MORE 7.5",           "W":   139, "Total":   232, "Accuracy": 0.599},
    ]
    _bt_html = []
    for _r in _bt_rows:
        _acc = _r["Accuracy"]
        _color = "#00C853" if _acc >= 0.60 else ("#F9A825" if _acc >= 0.57 else "#FF4444")
        _bar_w = int(_acc * 100)
        _bt_html.append(
            f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="min-width:220px;font-size:0.82rem;color:#E8ECF1;">{_r["Prop"]}</span>'
            f'<div style="flex:1"><div class="conf-track"><div class="conf-fill" style="width:{_bar_w}%;background:{_color};border-radius:4px;height:8px;"></div></div></div>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.85rem;min-width:52px;color:{_color};font-weight:600">{_acc:.1%}</span>'
            f'<span style="font-size:0.72rem;color:rgba(232,236,241,0.35);min-width:100px">{_r["W"]:,}W / {_r["Total"]:,}</span>'
            f'</div>'
        )
    st.markdown("".join(_bt_html), unsafe_allow_html=True)
    st.caption("Green ≥60% · Yellow 57–60% · Red <57% · Source: 2025 full-season backtest (v017, 128K+ predictions)")

    # ── Accuracy Transparency Panel ──
    st.markdown('<div class="section-hdr">Accuracy Transparency — How We Prove It</div>', unsafe_allow_html=True)
    _transp_html = f'''<div style="background:linear-gradient(145deg,#0d1828,#091020);border:1px solid rgba(0,200,83,0.08);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
                <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">Backtest Verified</div>
                <div style="font-family:JetBrains Mono,monospace;font-weight:700;font-size:2rem;color:#00C853;line-height:1;">71.7%</div>
                <div style="font-size:0.75rem;color:rgba(232,236,241,0.4);margin-top:0.3rem;">30,039 picks · Full 2025 season</div>
                <div style="font-size:0.7rem;color:rgba(232,236,241,0.3);margin-top:0.2rem;">Walk-forward: no future data leakage</div>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">Model Architecture</div>
                <div style="font-size:0.8rem;color:#E8ECF1;line-height:1.7;">
                    <span style="color:#4da6ff;">Layer 1:</span> Bayesian projections<br>
                    <span style="color:#00C853;">Layer 2:</span> Empirical calibration<br>
                    <span style="color:#FFB300;">Layer 3:</span> Confidence floor filtering
                </div>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">Self-Learning</div>
                <div style="font-size:0.8rem;color:#E8ECF1;line-height:1.7;">
                    <span style="color:rgba(232,236,241,0.5);">Calibration rebuild:</span> every 100+ picks<br>
                    <span style="color:rgba(232,236,241,0.5);">Floor re-optimization:</span> every 200+ picks<br>
                    <span style="color:rgba(232,236,241,0.5);">Kill switch:</span> auto-rollback at &lt;48%
                </div>
            </div>
        </div>
        <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(255,255,255,0.05);display:flex;gap:1.5rem;flex-wrap:wrap;">
            <div style="font-size:0.72rem;color:rgba(232,236,241,0.35);">
                <span style="color:rgba(0,200,83,0.6);font-weight:600;">128K+</span> calibration predictions
            </div>
            <div style="font-size:0.72rem;color:rgba(232,236,241,0.35);">
                <span style="color:rgba(77,166,255,0.6);font-weight:600;">10+</span> contextual factors per pick
            </div>
            <div style="font-size:0.72rem;color:rgba(232,236,241,0.35);">
                <span style="color:rgba(255,179,0,0.6);font-weight:600;">Every</span> pick tracked & graded
            </div>
        </div>
    </div>'''
    st.markdown(_transp_html, unsafe_allow_html=True)

    # ── Projection Accuracy (v018: track projected values vs actuals) ──
    st.markdown('<div class="section-hdr">Projection Accuracy — Live Results</div>', unsafe_allow_html=True)
    try:
        _proj_acc = get_projection_accuracy(days_back=30)
        if _proj_acc and _proj_acc.get("total_graded", 0) > 0:
            _pa1, _pa2, _pa3 = st.columns(3)
            _overall = _proj_acc.get("overall_accuracy", 0)
            _cls = "good" if _overall >= 65 else ("ok" if _overall >= 57 else "bad")
            with _pa1:
                st.markdown(f'<div class="acc-big {_cls}">{_overall:.1f}%</div><div class="acc-label">Live Pick Accuracy (30d)</div>', unsafe_allow_html=True)
            with _pa2:
                st.metric("Total Graded", f"{_proj_acc.get('total_graded', 0):,}")
            with _pa3:
                _avg_err = _proj_acc.get("avg_projection_error", 0)
                st.metric("Avg Projection Error", f"{_avg_err:.2f}" if _avg_err else "—")

            # Per-prop breakdown
            _by_prop = _proj_acc.get("by_prop", {})
            if _by_prop:
                _prop_html = []
                for _pname, _pdata in sorted(_by_prop.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
                    _pacc = _pdata.get("accuracy", 0)
                    _pclr = "#00C853" if _pacc >= 60 else ("#FFB300" if _pacc >= 57 else "#FF4444")
                    _pw = int(_pacc)
                    _prop_html.append(
                        f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="min-width:200px;font-size:0.8rem;color:#E8ECF1;">{_pname.replace("_"," ").title()}</span>'
                        f'<div style="flex:1"><div class="conf-track"><div class="conf-fill" style="width:{_pw}%;background:{_pclr};border-radius:4px;height:6px;"></div></div></div>'
                        f'<span style="font-family:JetBrains Mono;font-size:0.82rem;min-width:48px;color:{_pclr};font-weight:600">{_pacc:.1f}%</span>'
                        f'<span style="font-size:0.7rem;color:rgba(232,236,241,0.35);">{_pdata.get("total", 0)} picks</span>'
                        f'</div>'
                    )
                st.markdown("".join(_prop_html), unsafe_allow_html=True)
        else:
            st.caption("No graded projections yet. Grade results in the Grade tab to see live accuracy tracking.")
    except Exception:
        st.caption("Projection accuracy tracking will appear here once games are graded.")

    # ── Calibration & Proper Scoring Rules ──
    st.markdown('<div class="section-hdr">Model Calibration — Brier Score &amp; Reliability</div>', unsafe_allow_html=True)
    try:
        _cal = get_calibration_data(days_back=30)
        if _cal and _cal.get("total", 0) > 10:
            _cal_c1, _cal_c2 = st.columns(2)
            with _cal_c1:
                _brier = _cal.get("brier_score", 0)
                _ll = _cal.get("log_loss", 0)
                _brier_cls = "#00C853" if _brier < 0.23 else ("#FFB300" if _brier < 0.25 else "#FF4444")
                st.markdown(f'''<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;">
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">Proper Scoring Rules</div>
                    <div style="display:flex;gap:2rem;">
                        <div>
                            <div style="font-size:0.7rem;color:rgba(232,236,241,0.4);">Brier Score</div>
                            <div style="font-family:JetBrains Mono;font-weight:700;font-size:1.4rem;color:{_brier_cls};">{_brier:.4f}</div>
                            <div style="font-size:0.65rem;color:rgba(232,236,241,0.25);">Lower is better · 0.25 = coin flip</div>
                        </div>
                        <div>
                            <div style="font-size:0.7rem;color:rgba(232,236,241,0.4);">Log Loss</div>
                            <div style="font-family:JetBrains Mono;font-weight:700;font-size:1.4rem;color:#E8ECF1;">{_ll:.4f}</div>
                            <div style="font-size:0.65rem;color:rgba(232,236,241,0.25);">Lower is better · 0.693 = coin flip</div>
                        </div>
                    </div>
                </div>''', unsafe_allow_html=True)
            with _cal_c2:
                _buckets = _cal.get("buckets", [])
                if _buckets:
                    st.markdown('''<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;">
                        <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">Reliability Diagram</div>''', unsafe_allow_html=True)
                    _rel_html = []
                    for _b in _buckets:
                        _pred = _b.get("predicted_mean", 0) * 100
                        _actual = _b.get("actual_rate", 0) * 100
                        _cnt = _b.get("count", 0)
                        _diff = _actual - _pred
                        _diff_clr = "#00C853" if abs(_diff) < 3 else ("#FFB300" if abs(_diff) < 5 else "#FF4444")
                        _rel_html.append(
                            f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.78rem;">'
                            f'<span style="min-width:60px;color:rgba(232,236,241,0.5);">Pred {_pred:.0f}%</span>'
                            f'<span style="min-width:65px;font-family:JetBrains Mono;color:#E8ECF1;font-weight:600;">Act {_actual:.1f}%</span>'
                            f'<span style="min-width:50px;font-family:JetBrains Mono;color:{_diff_clr};font-size:0.72rem;">{_diff:+.1f}%</span>'
                            f'<span style="font-size:0.68rem;color:rgba(232,236,241,0.25);">n={_cnt}</span>'
                            f'</div>'
                        )
                    st.markdown("".join(_rel_html) + '</div>', unsafe_allow_html=True)
                else:
                    st.caption("Not enough data for calibration buckets yet.")
        else:
            st.caption("Calibration data requires 10+ graded picks. Keep grading!")
    except Exception:
        st.caption("Calibration data will appear here once enough games are graded.")

    # ── Model Health — Regime Change Detection (CUSUM) ──
    st.markdown('<div class="section-hdr">Model Health — Regime Change Detection</div>', unsafe_allow_html=True)
    try:
        _graded_df = get_graded_predictions(90)
        if not _graded_df.empty and len(_graded_df) >= 20:
            _health_preds = []
            for _, _hr in _graded_df.iterrows():
                _health_preds.append({
                    "result": _hr.get("result", "L"),
                    "confidence": float(_hr.get("confidence", 0.5)) if _hr.get("confidence") else 0.5,
                    "stat_type": _hr.get("stat_type", "unknown"),
                })
            _health = check_model_health(_health_preds, min_sample=20)

            _health_icon = "🟢" if _health.get("healthy") else "🔴"
            _health_label = "Healthy" if _health.get("healthy") else "Degraded"
            _health_color = "#00C853" if _health.get("healthy") else "#FF4444"

            _h1, _h2, _h3 = st.columns(3)
            with _h1:
                st.markdown(f'''<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">System Status</div>
                    <div style="font-size:1.8rem;">{_health_icon}</div>
                    <div style="font-family:JetBrains Mono;font-weight:700;font-size:1.1rem;color:{_health_color};">{_health_label}</div>
                </div>''', unsafe_allow_html=True)
            with _h2:
                _ob = _health.get("overall_brier")
                _ob_str = f"{_ob:.4f}" if _ob else "—"
                _ob_clr = "#00C853" if _ob and _ob < 0.23 else ("#FFB300" if _ob and _ob < 0.25 else "#FF4444")
                st.markdown(f'''<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">Live Brier Score</div>
                    <div style="font-family:JetBrains Mono;font-weight:700;font-size:1.4rem;color:{_ob_clr};">{_ob_str}</div>
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.25);">Target &lt; 0.22</div>
                </div>''', unsafe_allow_html=True)
            with _h3:
                _oa = _health.get("overall_accuracy")
                _oa_str = f"{_oa:.1%}" if _oa else "—"
                _oa_clr = "#00C853" if _oa and _oa >= 0.55 else ("#FFB300" if _oa and _oa >= 0.50 else "#FF4444")
                st.markdown(f'''<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">Live Accuracy (90d)</div>
                    <div style="font-family:JetBrains Mono;font-weight:700;font-size:1.4rem;color:{_oa_clr};">{_oa_str}</div>
                    <div style="font-size:0.65rem;color:rgba(232,236,241,0.25);">Target ≥ 54.2%</div>
                </div>''', unsafe_allow_html=True)

            # Show alerts
            _alerts = _health.get("alerts", [])
            if _alerts:
                for _alert in _alerts:
                    if "CRITICAL" in _alert:
                        st.error(_alert)
                    elif "REGIME" in _alert:
                        st.warning(f"⚠️ {_alert}")
                    else:
                        st.warning(_alert)
            else:
                st.caption("✅ No alerts — all metrics within normal range.")

            # Per-prop breakdown
            _bp = _health.get("by_prop", {})
            if _bp:
                _bp_rows = []
                for _ptype, _pinfo in sorted(_bp.items(), key=lambda x: x[1].get("accuracy", 0)):
                    _pacc = _pinfo.get("accuracy", 0)
                    _pclr = "#00C853" if _pacc >= 0.55 else ("#FFB300" if _pacc >= 0.50 else "#FF4444")
                    _pw = int(_pacc * 100)
                    _bp_rows.append(
                        f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.78rem;">'
                        f'<span style="min-width:180px;color:#E8ECF1;">{_ptype.replace("_"," ").title()}</span>'
                        f'<div style="flex:1"><div class="conf-track"><div class="conf-fill" style="width:{_pw}%;background:{_pclr};border-radius:4px;height:6px;"></div></div></div>'
                        f'<span style="font-family:JetBrains Mono;font-size:0.82rem;min-width:48px;color:{_pclr};font-weight:600">{_pacc:.1%}</span>'
                        f'<span style="font-size:0.68rem;color:rgba(232,236,241,0.25);">{_pinfo.get("total",0)} picks</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div style="background:linear-gradient(145deg,#0d1526,#0a1020);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.8rem 1rem;margin-top:0.5rem;">'
                    f'<div style="font-size:0.65rem;color:rgba(232,236,241,0.3);text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">Per-Prop Health</div>'
                    f'{"".join(_bp_rows)}</div>',
                    unsafe_allow_html=True
                )

            if _health.get("regime_change"):
                st.markdown(f'''<div style="background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.25);border-radius:10px;padding:0.8rem 1rem;margin-top:0.6rem;">
                    <div style="font-size:0.75rem;color:#FF4444;font-weight:600;">⚠️ Regime Change Detected</div>
                    <div style="font-size:0.72rem;color:rgba(232,236,241,0.5);margin-top:0.3rem;">CUSUM analysis detected a statistically significant shift in model performance. Consider pausing bets and running model tuning.</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.caption("Need 20+ graded picks for health monitoring. Keep grading!")
    except Exception as _he:
        st.caption(f"Health monitoring will appear here once games are graded. ({_he})")

    st.markdown('<div class="section-hdr">Daily Log — Last 14 Days</div>', unsafe_allow_html=True)
    _log_rows = get_daily_log_summary(14)
    if _log_rows:
        _log_df = pd.DataFrame(_log_rows)
        st.dataframe(_log_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No daily logs yet — logs are created automatically when you lock in picks.")

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

with tab_setup:
    st.markdown('<div class="section-hdr">Opening Day Checklist</div>', unsafe_allow_html=True)

    import os as _os, json as _json
    _checks = []

    _api_key = get_api_key()
    _checks.append(("Odds API key configured", bool(_api_key), "Add ODDS_API_KEY to Streamlit Secrets or .env"))

    _credits_remaining = None
    if _api_key:
        try:
            _usage = get_api_usage(_api_key)
            _credits_remaining = _usage.get("remaining", "?")
            _credits_ok = isinstance(_credits_remaining, int) and _credits_remaining > 50
            _checks.append((f"API credits remaining: {_credits_remaining}", _credits_ok, "Running low — consider upgrading"))
        except Exception:
            _checks.append(("API credits remaining", False, "Could not fetch usage"))
    else:
        _checks.append(("API credits remaining", False, "No API key set"))

    _bat_cache = _os.path.join(_os.path.dirname(__file__), "data", "batting_stats_cache.csv")
    if _os.path.exists(_bat_cache):
        try:
            _bat_df = pd.read_csv(_bat_cache)
            _bat_count = len(_bat_df)
            _checks.append((f"Batting cache loaded: {_bat_count} players", _bat_count >= 50, "Run cache refresh"))
        except Exception:
            _checks.append(("Batting cache loaded", False, "Cache file exists but couldn't be read"))
    else:
        _checks.append(("FanGraphs batting cache", False, "No cache — run the app once with internet"))

    _pit_cache = _os.path.join(_os.path.dirname(__file__), "data", "pitching_stats_cache.csv")
    if _os.path.exists(_pit_cache):
        try:
            _pit_df = pd.read_csv(_pit_cache)
            _pit_count = len(_pit_df)
            _checks.append((f"Pitching cache loaded: {_pit_count} pitchers", _pit_count >= 20, "Run cache refresh"))
        except Exception:
            _checks.append(("Pitching cache loaded", False, "Cache file exists but couldn't be read"))
    else:
        _checks.append(("FanGraphs pitching cache", False, "No cache — run the app once with internet"))

    _weights_path = _os.path.join(_os.path.dirname(__file__), "data", "weights", "current.json")
    if _os.path.exists(_weights_path):
        try:
            with open(_weights_path) as _wf:
                _wdata = _json.load(_wf)
            _wver = _wdata.get("version", "unknown")
            _checks.append((f"Model weights: {_wver}", True, ""))
        except Exception:
            _checks.append(("Model weights: current.json", False, "File exists but couldn't be parsed"))
    else:
        _checks.append(("Model weights (current.json)", False, "Using predictor defaults"))

    _bt_path = _os.path.join(_os.path.dirname(__file__), "data", "backtest", "backtest_2025_report.json")
    if _os.path.exists(_bt_path):
        try:
            with open(_bt_path) as _bf:
                _bt = _json.load(_bf)
            _bt_acc = _bt.get("overall_accuracy", _bt.get("accuracy", None))
            _bt_label = f"Backtest accuracy: {_bt_acc*100:.1f}%" if _bt_acc else "Backtest complete (accuracy unknown)"
            _checks.append((_bt_label, True, ""))
        except Exception:
            _checks.append(("Backtest report", False, "File exists but couldn't be parsed"))
    else:
        _checks.append(("Backtest 2025 report", False, "Not yet run"))

    try:
        _pp_test = _cached_pp_lines()
        _pp_ok = not _pp_test.empty
        _checks.append((f"PrizePicks API reachable ({len(_pp_test)} props today)" if _pp_ok else "PrizePicks API reachable (0 props)", _pp_ok, "Check PrizePicks API or network"))
    except Exception as _e:
        _checks.append(("PrizePicks API reachable", False, f"Error: {_e}"))

    _ready = sum(1 for _, ok, _ in _checks if ok)
    _total_checks = len(_checks)
    st.caption(f"**{_ready}/{_total_checks} checks passing** — Opening Day is March 27")
    for _label, _ok, _hint in _checks:
        _icon = "✅" if _ok else "❌"
        if _hint and not _ok:
            st.markdown(f"{_icon} {_label} — *{_hint}*")
        else:
            st.markdown(f"{_icon} {_label}")
    st.markdown("---")
    st.markdown('<div class="section-hdr">Configuration</div>', unsafe_allow_html=True)

    setup_c1, setup_c2 = st.columns(2)
    with setup_c1:
        st.markdown("""
**Step 1 — Get API Key**

Free at [the-odds-api.com](https://the-odds-api.com) · 500 req/month · No credit card required

**Step 2 — Add to Streamlit Secrets**
```
ODDS_API_KEY = "your_key_here"
```

**Step 3 — Daily Workflow**
1. Open app **before 10 AM ET** — best edge window
2. Go to **Find Edges** — lines auto-load
3. Focus on **A + B grades**, pitcher K props first
4. Build a **5 or 6 pick Flex** on PrizePicks
5. After games → **Grade** tab to log results
6. **Dashboard** tracks your edge over time
""")

    with setup_c2:
        st.markdown("**Entry Type Cheat Sheet**")
        st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
<thead><tr>
  <th style="text-align:left;padding:0.5rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:rgba(232,236,241,0.4);font-weight:600;text-transform:uppercase;font-size:0.65rem;letter-spacing:1.5px">Entry Type</th>
  <th style="text-align:center;padding:0.5rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:rgba(232,236,241,0.4);font-weight:600;text-transform:uppercase;font-size:0.65rem;letter-spacing:1.5px">Break-even</th>
  <th style="text-align:center;padding:0.5rem 0.8rem;border-bottom:1px solid rgba(255,255,255,0.08);color:rgba(232,236,241,0.4);font-weight:600;text-transform:uppercase;font-size:0.65rem;letter-spacing:1.5px">Verdict</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
  <td style="padding:0.55rem 0.8rem;color:#E8ECF1;font-weight:500">6-Pick Flex</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;font-family:JetBrains Mono,monospace;color:#E8ECF1">52.9%</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;color:#00C853;font-weight:700">✓ Best Value</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
  <td style="padding:0.55rem 0.8rem;color:#E8ECF1;font-weight:500">5-Pick Flex</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;font-family:JetBrains Mono,monospace;color:#E8ECF1">54.2%</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;color:#00C853;font-weight:700">✓ Use This</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
  <td style="padding:0.55rem 0.8rem;color:rgba(232,236,241,0.5)">2-Pick Power</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;font-family:JetBrains Mono,monospace;color:rgba(232,236,241,0.5)">57.7%</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;color:#FF4444">✗ Avoid</td>
</tr>
<tr>
  <td style="padding:0.55rem 0.8rem;color:rgba(232,236,241,0.5)">3-Pick Power</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;font-family:JetBrains Mono,monospace;color:rgba(232,236,241,0.5)">59.8%</td>
  <td style="padding:0.55rem 0.8rem;text-align:center;color:#FF4444">✗ Never</td>
</tr>
</tbody></table>

<div style="margin-top:1rem;padding:0.8rem 1rem;background:rgba(0,200,83,0.05);border:1px solid rgba(0,200,83,0.1);border-radius:8px;font-size:0.8rem;color:rgba(232,236,241,0.6);">
  <strong style="color:#00C853">Why this beats Rotowire:</strong><br>
  Rotowire tries to out-predict the market. This tool catches when PrizePicks lags behind sharp books. You're not smarter than Vegas — you're <em>faster</em>.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Bankroll</div>', unsafe_allow_html=True)
    br_val = st.number_input("Starting bankroll ($)", min_value=1.0, value=st.session_state.get("starting_bankroll", 100.0), step=10.0, key="br_input")
    if br_val != st.session_state.get("starting_bankroll", 100.0):
        st.session_state["starting_bankroll"] = br_val
        st.success(f"Starting bankroll set to ${br_val:.2f}")

    st.markdown("---")
    if st.button("Check API Credits"):
        k=get_api_key()
        if k:
            u=get_api_usage(k)
            st.info(f"Remaining: **{u.get('remaining','?')}** · Used: **{u.get('used','?')}**")
        else: st.warning("No API key set.")
