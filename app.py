"""
⚾ MLB Prop Edge v2 — Market-Based Edge Finder
Core: Compare PrizePicks lines to devigged sharp sportsbook odds.
When sharp books disagree with PrizePicks → that's the edge.
Statcast data provides the "why" confirmation layer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

from src.prizepicks import fetch_prizepicks_mlb_lines, get_available_stat_types
from src.sharp_odds import (
    fetch_mlb_events, fetch_event_props, extract_sharp_lines,
    find_ev_edges, get_api_usage, get_api_key, PP_TO_ODDS_API,
)
from src.weather import fetch_game_weather, resolve_team, STADIUMS
from src.umpires import get_umpire_k_adjustment
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
from src.autolearn import run_adjustment_cycle, load_current_weights, get_weight_history
from src.spring import (
    get_player_injury_status, get_spring_form_multiplier,
    fetch_spring_training_stats, fetch_injuries,
)
from src.trends import get_batter_trend
from src.explain import build_explanation, format_explanation_text
from src.combined import score_picks, SIGNAL_CONFIRMED, SIGNAL_SHARP_ONLY, SIGNAL_PROJECTION_ONLY
from src.slip_warnings import analyze_slip_correlation, format_warnings_streamlit

st.set_page_config(page_title="MLB Prop Edge", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* === BASE === */
    .stApp { font-family: 'Outfit', sans-serif; background: #070d1a; }
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
    }
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

    /* === BEST PLAYS === */
    .best-play {
        background: linear-gradient(145deg, #0d1828, #091020);
        border: 1px solid rgba(0,200,83,0.1);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
        position: relative;
        overflow: hidden;
    }
    .best-play::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(0,200,83,0.4), transparent); }
    .bp-name { font-weight: 700; font-size: 0.95rem; color: #E8ECF1; margin-bottom: 0.15rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bp-prop { font-size: 0.7rem; color: rgba(232,236,241,0.38); margin-bottom: 0.45rem; }
    .bp-pick { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.15rem; }
    .bp-pick.more { color: #00C853; } .bp-pick.less { color: #FF4444; }
    .bp-conf { font-size: 0.68rem; color: rgba(232,236,241,0.35); font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem; }

    /* === CONFIDENCE BAR === */
    .conf-track { background: rgba(255,255,255,0.06); border-radius: 3px; height: 4px; margin-top: 0.5rem; overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 3px; }
    .conf-fill.high { background: linear-gradient(90deg, #00963e, #00C853); }
    .conf-fill.med { background: linear-gradient(90deg, #b37d00, #FFB300); }
    .conf-fill.low { background: linear-gradient(90deg, #aa2200, #FF4444); }

    /* === PICK ROW EXPANDERS === */
    [data-testid="stExpander"] { background: rgba(13,21,38,0.6) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; margin-bottom: 0.35rem !important; }
    [data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.1) !important; }

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
    }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.82rem; }

    /* === MOBILE === */
    @media (max-width: 768px) {
        .stApp [data-testid="stDataFrame"] { overflow-x: auto !important; }
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .hero-logo { font-size: 1.35rem; }
        .card .val { font-size: 1.25rem; }
        .stApp button[kind="secondary"], .stApp [data-testid="stTab"] { min-height: 44px; }
    }
</style>
""", unsafe_allow_html=True)

def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v)
def badge(r): return f'<span class="badge badge-{r.lower()}">{r}</span>'
def pick_span(p): return f'<span class="{"more" if p=="MORE" else "less"}">{p}</span>'
def grade_label(r):
    icons = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
    return f"{icons.get(r, '⚪')} {r}"


@st.cache_data(ttl=3600)
def load_batting_stats():
    """Load batting leaders from cached CSV first, fall back to pybaseball."""
    import os
    cache_path = os.path.join(os.path.dirname(__file__), "data", "batting_stats_cache.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) >= 50:
            return df

    # Fallback: try pybaseball (won't work on Streamlit Cloud)
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
    # Try exact match (normalized)
    for idx, row in batting_df.iterrows():
        if _normalize_name(str(row.get("Name", ""))) == norm_target:
            return row
    # Try last name + first initial
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
    # Try unique last name match
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
        """Handle K% and BB% — FanGraphs sometimes returns as strings with % signs."""
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
        "sprint_speed": 27.0,  # Default; Statcast-sourced if available
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
    # Cache to CSV for next time
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
    # Get starting bankroll from session state or default
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

# Load model version for header
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
    if not api_key:
        st.warning("⚠️ Add your Odds API key in the Setup tab to enable sharp line comparison.")
        st.info("Free key at [the-odds-api.com](https://the-odds-api.com) — 500 req/month, no credit card.")
    else:
        usage = get_api_usage(api_key)
        st.markdown(f'<div class="info-strip">🔑 Odds API active &nbsp;·&nbsp; <span class="hl">{usage.get("remaining","?")}</span> credits remaining &nbsp;·&nbsp; Sharp books: FanDuel · Pinnacle · DraftKings</div>', unsafe_allow_html=True)

    with st.spinner("Pulling PrizePicks MLB lines..."):
        try: pp_lines = fetch_prizepicks_mlb_lines()
        except: pp_lines = pd.DataFrame()

    if pp_lines.empty:
        st.info("No MLB lines on PrizePicks right now. Lines usually post by 10 AM ET.")
    else:
        st.markdown(f"**{len(pp_lines)} MLB props** on PrizePicks today")
        all_edges = []
        if has_sharp:
            total_sharp_lines = 0
            events_with_props = 0
            with st.spinner("Fetching sharp lines & devigging..."):
                events = fetch_mlb_events(api_key)
                for event in (events or [])[:15]:
                    eid = event.get("id","")
                    if not eid: continue
                    result = fetch_event_props(eid, api_key=api_key)
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

            f1,f2 = st.columns(2)
            with f1: min_grade = st.selectbox("Min grade", ["A only","A + B","A + B + C","All"], index=1)
            with f2: prop_f = st.selectbox("Prop type", ["All","Pitcher Ks","Batter Hits","Total Bases","Home Runs"])
            gm = {"A only":["A"],"A + B":["A","B"],"A + B + C":["A","B","C"],"All":["A","B","C","D"]}
            filt = [e for e in all_edges if e["rating"] in gm[min_grade]]
            if prop_f=="Pitcher Ks": filt=[e for e in filt if "strikeout" in e.get("market","").lower() and "batter" not in e.get("market","").lower()]
            elif prop_f=="Batter Hits": filt=[e for e in filt if "hits" in e.get("market","").lower()]
            elif prop_f=="Total Bases":
                filt=[e for e in filt if "total_bases" in e.get("market","").lower()]
                # Warning: TB LESS historically underperforms
                less_tb = [e for e in filt if e.get("pick") == "LESS"]
                if less_tb:
                    st.markdown('<div class="warn-strip">⚠️ <strong>TB LESS Warning:</strong> Total Bases LESS picks historically underperform — trade with extra caution</div>', unsafe_allow_html=True)
            elif prop_f=="Home Runs": filt=[e for e in filt if "home_run" in e.get("market","").lower()]

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
                        if edge.get("fanduel_agrees"): st.success("✅ FanDuel confirms this side")

                st.markdown("---")
                if st.button("💾 Log Edges", type="primary"):
                    log_batch_predictions(filt, date.today().isoformat())
                    st.success(f"Saved {len(filt)} predictions!")
            else: st.info("No edges match filters.")
        elif has_sharp: st.info("Sharp book props not available for Spring Training — showing projection-based analysis below.")

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
                st.warning("⚠️ Could not load player stats — using league averages for projections.")
            else:
                p_cap = f"Loaded {len(batting_df)} batters"
                if not pitching_df.empty:
                    p_cap += f" + {len(pitching_df)} pitchers"
                p_cap += " from FanGraphs"
                st.caption(p_cap)
            preds = []
            # Pre-fetch weather for all unique teams at once (one API call per stadium, not per player)
            teams_in_slate = set()
            for _, row in pp_lines.iterrows():
                t = row.get("team", "")
                if t:
                    r = resolve_team(t)
                    if r and r in STADIUMS:
                        teams_in_slate.add(r)
            weather_cache = {}
            if teams_in_slate:
                wx_prog = st.progress(0, text="Fetching weather for stadiums...")
                for j, team_abbr in enumerate(sorted(teams_in_slate)):
                    wx_prog.progress((j + 1) / len(teams_in_slate), text=f"Weather: {STADIUMS[team_abbr]['name']} ({j+1}/{len(teams_in_slate)})")
                    try:
                        weather_cache[team_abbr] = fetch_game_weather(team_abbr)
                    except Exception:
                        weather_cache[team_abbr] = None
                wx_prog.empty()

            # Pre-fetch Spring Training stats + injuries (one API call each)
            st_stats = []
            injury_list = []
            with st.spinner("Loading Spring Training data & injuries..."):
                try:
                    st_stats = fetch_spring_training_stats()
                except Exception:
                    st_stats = []
                try:
                    injury_list = fetch_injuries(days_back=60)
                except Exception:
                    injury_list = []

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

                # Build batter or pitcher profile from FanGraphs stats
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

                p = generate_prediction(
                    player_name=row["player_name"],
                    stat_type=row["stat_type"],
                    stat_internal=stat_int,
                    line=row["line"],
                    batter_profile=batter_profile,
                    pitcher_profile=pitcher_profile,
                    park_team=resolve_team(team) if team else None,
                    weather=wx,
                )

                # Spring form multiplier — compare ST performance to prior season
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
                p["projection"] = round(p["projection"] * spring_mult, 2)
                p["spring_mult"] = spring_mult
                p["spring_badge"] = spring["badge"]

                # Trend multiplier — recent form tiebreaker (±5-8% max)
                trend = get_batter_trend(0)  # No player_id available, returns neutral
                trend_mult = trend.get("trend_multiplier", 1.0)
                # Clamp trend to ±8% max
                trend_mult = max(0.92, min(1.08, trend_mult))
                p["projection"] = round(p["projection"] * trend_mult, 2)
                p["trend_mult"] = trend_mult
                if trend_mult >= 1.03:
                    p["trend_badge"] = "hot"
                elif trend_mult <= 0.97:
                    p["trend_badge"] = "cold"
                else:
                    p["trend_badge"] = "neutral"

                # Cold elite buy-low signal: recent cold streak + elite season talent
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
                        # +4% boost — cold elite regression signal
                        p["projection"] = round(p["projection"] * 1.04, 2)

                # Injury status
                injury = get_player_injury_status(
                    player_name=row["player_name"],
                    injuries=injury_list,
                )
                p["injury_status"] = injury["status"]
                p["injury_color"] = injury["color"]

                p["team"] = team
                p["stat_internal"] = stat_int
                preds.append(p)
            prog.empty()

            if preds:
                pdf = pd.DataFrame(preds).sort_values("confidence", ascending=False)

                # ── Quick Summary ──
                scored_all = score_picks(all_edges, preds)
                ab_combined = [s for s in scored_all if s["combined_grade"] in ("A+", "A", "B")]
                confirmed_count = sum(1 for s in ab_combined if s["signal"] == SIGNAL_CONFIRMED)
                # Summary stat cards
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
                    st.markdown(f'<div class="info-strip">✅ <span class="hl">{confirmed_count}</span> picks confirmed by both sharp books and projection model</div>', unsafe_allow_html=True)
                elif not all_edges:
                    st.markdown('<div class="warn-strip">⚠️ <strong>No sharp lines</strong> — showing projection-only analysis. Add Odds API key for full edge detection.</div>', unsafe_allow_html=True)

                # ── Today's Best Plays ──
                # Use combined scoring when sharp edges exist, fall back to projection ranking
                # Trivial low-line LESS picks (SB 0.5, HR 0.5) always look like
                # huge edges but provide no real signal — exclude from best plays.
                _TRIVIAL_LESS_PROPS = {"stolen_bases", "home_runs"}
                def _is_trivial(pick: dict) -> bool:
                    return (
                        pick.get("pick") == "LESS"
                        and float(pick.get("line", 99)) <= 0.5
                        and pick.get("stat_type", "").lower().replace(" ", "_") in _TRIVIAL_LESS_PROPS
                    )

                if scored_all:
                    top_plays = [s for s in scored_all if s["combined_grade"] in ("A+", "A") and not _is_trivial(s)][:5]
                else:
                    top_plays = []
                if not top_plays:
                    # Fall back to top A-grade projections, skipping trivial LESS picks
                    for _, tp in pdf[pdf["rating"] == "A"].head(20).iterrows():
                        if _is_trivial(tp):
                            continue
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
                    _bp_signal_labels = {SIGNAL_CONFIRMED: ("confirmed", "✓ CONFIRMED"), SIGNAL_SHARP_ONLY: ("sharp", "📊 SHARP"), SIGNAL_PROJECTION_ONLY: ("proj", "🔮 PROJ")}
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
                                <div class="bp-prop">{tp["stat_type"]} &nbsp;·&nbsp; Line {tp["line"]}</div>
                                <div class="bp-pick {pick_cls}">{tp["pick"]}</div>
                                <div class="conf-track"><div class="conf-fill {conf_cls}" style="width:{min(conf_pct,100)}%"></div></div>
                                <div class="bp-conf">{grade_icon} {tp.get("combined_grade","?")} &nbsp;·&nbsp; {conf_pct}% conf &nbsp;·&nbsp; {sig_label}</div>
                            </div>''', unsafe_allow_html=True)

                # ── Filters ──
                st.markdown('<div class="section-hdr">Filter Picks</div>', unsafe_allow_html=True)
                prop_types_available = sorted(pdf["stat_type"].unique().tolist())
                f1, f2 = st.columns([3, 2])
                with f1:
                    proj_prop_filter = st.radio("Prop Type", ["All"] + prop_types_available[:6], horizontal=True, key="proj_prop_f")
                with f2:
                    proj_grade_filter = st.radio("Min Grade", ["A only", "A+B", "A+B+C", "All"], index=1, key="proj_grade_f")
                _grade_map_radio = {"A only": ["A"], "A+B": ["A","B"], "A+B+C": ["A","B","C"], "All": ["A","B","C","D"]}

                filtered = pdf[pdf["rating"].isin(_grade_map_radio[proj_grade_filter])].copy()
                if proj_prop_filter != "All":
                    filtered = filtered[filtered["stat_type"] == proj_prop_filter]

                # ── Unified projection table with checkboxes ──
                slip_candidates = filtered.head(40).reset_index(drop=True)
                selected_picks = []

                for pick_idx, (_, pick_row) in enumerate(slip_candidates.iterrows()):
                    if pick_idx >= 40:
                        break
                    health_icon = {"IL": "🔴 IL", "day-to-day": "🟡 DTD", "active": "🟢"}.get(pick_row.get("injury_status", "active"), "🟢")
                    spring_icon = {"hot": "🔥", "cold": "❄️", "neutral": "➖"}.get(pick_row.get("spring_badge", "neutral"), "➖")
                    trend_icon = {"hot": "🔥", "cold": "❄️", "neutral": "➖"}.get(pick_row.get("trend_badge", "neutral"), "➖")
                    buy_tag = " 🎯" if pick_row.get("buy_low") else ""
                    pick_cls = "more" if pick_row["pick"] == "MORE" else "less"

                    chk_col, info_col = st.columns([0.05, 0.95])
                    with chk_col:
                        checked = st.checkbox("", key=f"proj_pick_{pick_idx}", label_visibility="collapsed")
                    with info_col:
                        conf_val = pick_row['confidence']
                        conf_pct = f"{conf_val*100:.1f}%"
                        edge_val = pick_row['edge']*100
                        edge_pct = f"{edge_val:.1f}%"
                        _dir_arrow = "▲" if pick_row['pick'] == "MORE" else "▼"
                        _dir_color = "#00C853" if pick_row['pick'] == "MORE" else "#FF4444"
                        with st.expander(
                            f"{grade_label(pick_row['rating'])}  **{pick_row['player_name']}**  {pick_row.get('team','')} · "
                            f"{pick_row['stat_type']} {pick_row['line']} → "
                            f"**{pick_row['pick']}** {pick_row['projection']} · "
                            f"Conf {conf_pct} · +{edge_pct} · "
                            f"{trend_icon} {health_icon} {spring_icon}{buy_tag}",
                        ):
                            # Factor breakdown
                            fc1, fc2 = st.columns(2)
                            with fc1:
                                st.markdown("**Projection Breakdown**")
                                factors = []
                                # Park factor
                                park_team = resolve_team(pick_row.get("team", "")) if pick_row.get("team") else None
                                if park_team and park_team in PARK_FACTORS:
                                    pf = PARK_FACTORS[park_team]
                                    pf_pct = (pf - 1.0) * 100
                                    factors.append(("Park factor", pf_pct))
                                # Weather
                                if pick_row.get("team"):
                                    rt = resolve_team(pick_row["team"])
                                    wxd = weather_cache.get(rt)
                                    if wxd and wxd.get("weather_offense_mult"):
                                        wx_pct = (wxd["weather_offense_mult"] - 1.0) * 100
                                        factors.append(("Weather", wx_pct))
                                # Spring form
                                sm = pick_row.get("spring_mult", 1.0)
                                if sm != 1.0:
                                    factors.append(("Spring form", (sm - 1.0) * 100))
                                # Trend
                                tm = pick_row.get("trend_mult", 1.0)
                                if tm != 1.0:
                                    factors.append(("Recent trend", (tm - 1.0) * 100))
                                # Buy-low
                                if pick_row.get("buy_low"):
                                    factors.append(("Elite buy-low boost", 4.0))

                                if factors:
                                    for fname, fpct in factors:
                                        cls = "pos" if fpct > 0.1 else ("neg" if fpct < -0.1 else "neu")
                                        st.markdown(f'<div class="factor-bar"><span class="f-name">{fname}</span><span class="f-impact {cls}">{fpct:+.1f}%</span></div>', unsafe_allow_html=True)
                                else:
                                    st.caption("Base projection (no major adjustments)")
                            with fc2:
                                st.markdown("**Key Stats**")
                                proj_v = pick_row['projection']
                                line_v = pick_row['line']
                                diff = proj_v - line_v
                                diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
                                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:{"#00C853" if diff>=0 else "#FF4444"}">{proj_v} <span style="font-size:0.7rem;color:rgba(232,236,241,0.4)">proj ({diff_str} vs {line_v} line)</span></div>', unsafe_allow_html=True)
                                conf_cls2 = "high" if conf_val > 0.6 else ("med" if conf_val > 0.52 else "low")
                                st.markdown(f'<div class="conf-track" style="margin:0.5rem 0"><div class="conf-fill {conf_cls2}" style="width:{min(int(conf_val*100),100)}%"></div></div>', unsafe_allow_html=True)
                                st.caption(f"Confidence: {conf_pct} · Edge: +{edge_pct}")
                                st.caption(f"Health: {health_icon} · Spring: {spring_icon} · Trend: {trend_icon}")
                                if pick_row.get("buy_low"):
                                    st.markdown('<div class="warn-strip" style="margin-top:0.5rem">🎯 <strong>BUY LOW</strong> — Elite player in cold streak, regression expected</div>', unsafe_allow_html=True)

                    if checked:
                        selected_picks.append({
                            "player_name": pick_row["player_name"],
                            "stat_type": pick_row["stat_type"],
                            "line": pick_row["line"],
                            "pick": pick_row["pick"],
                        })

                # ── Combined Analysis ──
                _combined_grade_emoji = {"A+": "💎", "A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
                _signal_badge = {
                    SIGNAL_CONFIRMED: "✅ CONFIRMED",
                    SIGNAL_SHARP_ONLY: "📊 SHARP",
                    SIGNAL_PROJECTION_ONLY: "🔮 PROJECTION",
                }
                scored_combined = score_picks(all_edges, preds)
                if scored_combined:
                    st.markdown('<div class="section-hdr">Combined Analysis</div>', unsafe_allow_html=True)
                    cdf = pd.DataFrame(scored_combined)
                    cdisp = cdf[["combined_grade", "player_name", "team", "stat_type", "line", "pick", "combined_score", "signal"]].head(30).copy()
                    cdisp.columns = ["Grade", "Player", "Team", "Prop", "Line", "Pick", "Score", "Signal"]
                    cdisp["Grade"] = cdisp["Grade"].apply(lambda g: f"{_combined_grade_emoji.get(g, '⚪')} {g}")
                    cdisp["Signal"] = cdisp["Signal"].apply(lambda s: _signal_badge.get(s, s))
                    cdisp["Score"] = cdisp["Score"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(cdisp, hide_index=True, use_container_width=True)

                # ── Slip builder ──
                if selected_picks:
                    st.markdown('<div class="section-hdr">Build Slip</div>', unsafe_allow_html=True)
                    # Show selected picks as chips
                    def _pick_chip(p):
                        c = "#00C853" if p["pick"] == "MORE" else "#FF4444"
                        return (
                            f'<span style="display:inline-block;background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.2);'
                            f'border-radius:20px;padding:0.2rem 0.75rem;font-size:0.78rem;color:#E8ECF1;margin:0.2rem;">'
                            f'<span style="color:{c};font-weight:700;">{p["pick"]}</span>'
                            f' {p["player_name"]} &middot; {p["stat_type"]} {p["line"]}</span>'
                        )
                    chips_html = " ".join(_pick_chip(p) for p in selected_picks)
                    st.markdown(f'<div style="margin-bottom:0.8rem;line-height:2.2">{chips_html}</div>', unsafe_allow_html=True)
                    # Slip correlation warnings
                    slip_warns = analyze_slip_correlation(selected_picks)
                    if any(w["severity"] == "high" for w in slip_warns):
                        st.warning("⚠️ This slip has correlation risks — review warnings below")
                    for emoji, text in format_warnings_streamlit(slip_warns):
                        st.markdown(f"{emoji} {text}")

                    # Estimated slip strength from combined scores
                    if scored_all:
                        _scored_lookup = {
                            f"{s['player_name'].lower()}|{s['stat_type'].lower()}": s
                            for s in scored_all
                        }
                        slip_scores = []
                        for sp in selected_picks:
                            key = f"{sp['player_name'].lower()}|{sp['stat_type'].lower()}"
                            matched = _scored_lookup.get(key)
                            if matched:
                                slip_scores.append(matched["combined_score"])
                        if slip_scores:
                            avg_score = sum(slip_scores) / len(slip_scores)
                            strength_pct = min(50 + avg_score * 300, 85)  # Scale to readable %
                            st.caption(f"Estimated slip strength: **{strength_pct:.0f}%** (avg combined score: {avg_score:.3f})")

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        slip_type_opts = [f"{n}-Pick {'Power Play' if n <= 3 else 'Flex'}" for n in range(2, 7)]
                        proj_slip_type = st.selectbox("Slip type", slip_type_opts, key="proj_slip_type")
                    with sc2:
                        proj_slip_amount = st.number_input("Entry ($)", min_value=1.0, value=5.0, step=1.0, key="proj_slip_amt")
                    if st.button("Lock In Picks", type="primary", key="proj_save_slip"):
                        num_needed = int(proj_slip_type[0])
                        if len(selected_picks) == num_needed:
                            sid = create_slip(date.today().isoformat(), proj_slip_type, proj_slip_amount, selected_picks)
                            # Save daily log snapshot
                            try:
                                save_daily_log(preds)
                            except Exception:
                                pass
                            st.success(f"Slip #{sid} saved! Daily projection log saved.")
                            st.rerun()
                        else:
                            st.warning(f"Select exactly {num_needed} picks for a {proj_slip_type}.")

with tab_slips:
    st.markdown('<div class="section-hdr">Slip Tracker &amp; P&amp;L</div>', unsafe_allow_html=True)
    # P&L Summary
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

        # Bankroll chart
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
    # Create new slip
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
    # Show recent slips
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
        st.markdown(f'<div class="info-strip">📋 <span class="hl">{len(pp_lines)}</span> props on the board today</div>', unsafe_allow_html=True)
        s = st.text_input("🔍 Search player", "", key="s2", placeholder="Type player name...")
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
        # Large accuracy display
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
            # Grade breakdown with inline bars
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

            st.markdown("**More vs Less**", help="Direction bias analysis")
            dir_cols = st.columns(2)
            for i, d in enumerate(["MORE", "LESS"]):
                dv = stats.get('by_direction', {}).get(d, {})
                d_acc = dv.get('accuracy', 0)
                d_w = dv.get('wins', 0)
                d_t = dv.get('total', 0)
                d_color = "#00C853" if d == "MORE" else "#FF4444"
                with dir_cols[i]:
                    st.markdown(f'<div class="card"><div class="lbl">{d}</div><div class="val" style="color:{d_color}">{pct(d_acc)}</div><div class="sub">{d_w}W-{d_t-d_w}L</div></div>', unsafe_allow_html=True)

    # Model Tuning Section
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
                        st.info(result.get("reason", "No adjustments needed (insufficient data or already optimal)"))
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

with tab_grade:
    st.markdown('<div class="section-hdr">Grade Results</div>', unsafe_allow_html=True)

    # Auto-grade banner
    st.markdown('<div class="info-strip">⚡ Auto-Grade pulls actual results from the MLB Stats API and grades all pending picks automatically.</div>', unsafe_allow_html=True)

    # Auto-grade button
    ag_col1, ag_col2 = st.columns([1, 3])
    with ag_col1:
        if st.button("⚡ Auto-Grade Yesterday", type="primary"):
            with st.spinner("Pulling box scores and grading..."):
                try:
                    result = auto_grade_yesterday()
                    if result["graded"] > 0:
                        wins = sum(1 for r in result["results"] if r.get("result") == "W")
                        losses = sum(1 for r in result["results"] if r.get("result") == "L")
                        st.success(f"Auto-graded {result['graded']} picks: {wins}W-{losses}L")
                        # Auto-trigger model learning after grading
                        try:
                            learn_result = run_adjustment_cycle(min_sample=25)
                            if learn_result.get("adjusted"):
                                st.info(f"🧠 Model auto-tuned: {learn_result.get('reason', 'weights updated')}")
                            elif learn_result.get("reason"):
                                st.caption(f"🧠 Auto-tune: {learn_result.get('reason')}")
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
        # Build styled feed
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
