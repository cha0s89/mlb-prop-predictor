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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Outfit', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .hero { background: linear-gradient(135deg, #0f1923 0%, #1a2940 50%, #0d2818 100%); border: 1px solid rgba(0,200,83,0.15); border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.2rem; }
    .hero h1 { font-family: 'Outfit'; font-weight: 800; font-size: 2rem; color: #E8ECF1; margin: 0 0 0.2rem 0; }
    .hero .sub { font-weight: 300; font-size: 0.9rem; color: rgba(232,236,241,0.45); margin: 0; }
    .hero .accent { color: #00C853; }
    .card { background: linear-gradient(145deg, #121929, #0f1520); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
    .card .lbl { font-size: 0.7rem; color: rgba(232,236,241,0.4); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.3rem; }
    .card .val { font-family: 'JetBrains Mono'; font-weight: 700; font-size: 1.6rem; color: #E8ECF1; line-height: 1; }
    .card .val.g { color: #00C853; } .card .val.y { color: #FFB300; } .card .val.r { color: #FF5252; }
    .badge { display: inline-block; font-family: 'JetBrains Mono'; font-weight: 700; font-size: 0.65rem; padding: 0.2rem 0.5rem; border-radius: 5px; letter-spacing: 1px; }
    .badge-a { background: rgba(0,200,83,0.15); color: #00C853; } .badge-b { background: rgba(0,230,118,0.12); color: #00E676; }
    .badge-c { background: rgba(255,179,0,0.12); color: #FFB300; } .badge-d { background: rgba(255,82,82,0.10); color: #FF5252; }
    .more { color: #00C853; font-family: 'JetBrains Mono'; font-weight: 700; }
    .less { color: #FF5252; font-family: 'JetBrains Mono'; font-weight: 700; }
    .info-strip { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.82rem; color: rgba(232,236,241,0.6); margin-bottom: 0.8rem; }
    .info-strip .hl { color: #E8ECF1; font-weight: 600; font-family: 'JetBrains Mono'; }
    .section-hdr { font-weight: 600; font-size: 1.05rem; color: rgba(232,236,241,0.8); margin: 1.2rem 0 0.6rem 0; padding-bottom: 0.3rem; border-bottom: 1px solid rgba(255,255,255,0.06); }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono'; font-weight: 700; }
    .best-play { background: linear-gradient(145deg, #121929, #0f1520); border: 1px solid rgba(0,200,83,0.18); border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
    .best-play .bp-name { font-weight: 700; font-size: 1rem; color: #E8ECF1; }
    .best-play .bp-prop { font-size: 0.8rem; color: rgba(232,236,241,0.5); }
    .best-play .bp-pick { font-family: 'JetBrains Mono'; font-weight: 700; font-size: 1.1rem; }
    .best-play .bp-pick.more { color: #00C853; } .best-play .bp-pick.less { color: #FF5252; }
    .best-play .bp-conf { font-family: 'JetBrains Mono'; font-size: 0.8rem; color: rgba(232,236,241,0.6); }
    .factor-bar { display: flex; align-items: center; gap: 0.4rem; margin: 0.25rem 0; font-size: 0.82rem; }
    .factor-bar .f-name { color: rgba(232,236,241,0.6); min-width: 120px; }
    .factor-bar .f-impact { font-family: 'JetBrains Mono'; font-weight: 600; }
    .factor-bar .f-impact.pos { color: #00C853; } .factor-bar .f-impact.neg { color: #FF5252; } .factor-bar .f-impact.neu { color: rgba(232,236,241,0.4); }
    @media (max-width: 768px) {
        .stApp [data-testid="stDataFrame"] { overflow-x: auto !important; -webkit-overflow-scrolling: touch; }
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .hero h1 { font-size: 1.4rem; } .hero .sub { font-size: 0.75rem; }
        .card .val { font-size: 1.2rem; } .card .lbl { font-size: 0.6rem; }
        .stApp button[kind="secondary"], .stApp [data-testid="stTab"] { min-height: 44px; font-size: 0.9rem; padding: 0.5rem 0.8rem; }
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
# Defines which (stat_internal, direction) combos are actually available to bet
# on PrizePicks. HR LESS and SB LESS appear mathematically good but can't be bet.
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
}

_PP_FILTERED_LABELS = {
    ("home_runs", "LESS"): "HR LESS",
    ("stolen_bases", "LESS"): "SB LESS",
    ("total_bases", "LESS"): "TB LESS",
}


def is_tradeable_pick(stat_internal: str, direction: str) -> bool:
    """Return False if this (stat_internal, direction) is not tradeable on PrizePicks."""
    if (stat_internal, direction) in PP_NEVER_SHOW:
        return False
    cfg = PP_TRADEABLE.get(stat_internal)
    if cfg is None:
        return True  # Unknown prop type: pass through rather than hide
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

st.markdown("""<div class="hero"><h1>⚾ MLB Prop <span class="accent">Edge</span></h1><p class="sub">Sharp book devigging × PrizePicks line comparison × Statcast confirmation</p></div>""", unsafe_allow_html=True)

tab_edge, tab_slips, tab_picks, tab_dash, tab_grade, tab_setup = st.tabs(["🎯 FIND EDGES", "🎫 MY SLIPS", "📋 ALL LINES", "📊 DASHBOARD", "✅ GRADE", "⚙️ SETUP"])

with tab_edge:
    api_key = get_api_key()
    has_sharp = bool(api_key)
    if not api_key:
        st.warning("⚠️ Add your Odds API key in the Setup tab to enable sharp line comparison.")
        st.info("Free key at [the-odds-api.com](https://the-odds-api.com) — 500 req/month, no credit card.")
    else:
        usage = get_api_usage(api_key)
        st.markdown(f'<div class="info-strip">🔑 API active · <span class="hl">{usage.get("remaining","?")}</span> credits left · Sharp books: FanDuel (1.24x), Pinnacle, DraftKings</div>', unsafe_allow_html=True)

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
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(f'<div class="card"><div class="lbl">A-GRADE</div><div class="val g">{a_n}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="card"><div class="lbl">B-GRADE</div><div class="val">{b_n}</div></div>', unsafe_allow_html=True)
            with c3:
                cls = "g" if avg_e>5 else ("y" if avg_e>3 else "r")
                st.markdown(f'<div class="card"><div class="lbl">AVG EDGE</div><div class="val {cls}">{avg_e:.1f}%</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="card"><div class="lbl">TOTAL EDGES</div><div class="val">{len(all_edges)}</div></div>', unsafe_allow_html=True)

            f1,f2,f3 = st.columns([2,2,2])
            with f1: min_grade = st.selectbox("Min grade", ["A only","A + B","A + B + C","All"], index=1)
            with f2: prop_f = st.selectbox("Prop type", ["All","Pitcher Ks","Batter Hits","Total Bases","Home Runs"])
            with f3: show_all_sharp = st.checkbox("Show all picks (incl. non-tradeable)", value=False, key="show_all_sharp")
            gm = {"A only":["A"],"A + B":["A","B"],"A + B + C":["A","B","C"],"All":["A","B","C","D"]}
            filt = [e for e in all_edges if e["rating"] in gm[min_grade]]
            if prop_f=="Pitcher Ks": filt=[e for e in filt if "strikeout" in e.get("market","").lower() and "batter" not in e.get("market","").lower()]
            elif prop_f=="Batter Hits": filt=[e for e in filt if "hits" in e.get("market","").lower()]
            elif prop_f=="Total Bases": filt=[e for e in filt if "total_bases" in e.get("market","").lower()]
            elif prop_f=="Home Runs": filt=[e for e in filt if "home_run" in e.get("market","").lower()]
            # Filter non-tradeable picks unless user opts in
            if not show_all_sharp:
                _hidden_sharp = [e for e in filt if not is_tradeable_pick(e.get("stat_internal", e.get("market", "")), e.get("pick", ""))]
                filt = [e for e in filt if is_tradeable_pick(e.get("stat_internal", e.get("market", "")), e.get("pick", ""))]
                _hidden_labels = list({_PP_FILTERED_LABELS.get((e.get("stat_internal",""), e.get("pick","")), f"{e.get('stat_type','')} {e.get('pick','')}") for e in _hidden_sharp})
                if _hidden_labels:
                    st.info(f"ℹ️ Showing tradeable picks only. Filtered out: {', '.join(_hidden_labels)} (not available or unprofitable on PrizePicks)")

            with st.expander("📋 Betting Rules & Bankroll Guide", expanded=False):
                _brc1, _brc2 = st.columns(2)
                with _brc1:
                    st.markdown("**✅ BET THESE PROPS**")
                    st.markdown(
                        "- Pitcher Strikeouts (MORE and LESS)\n"
                        "- Hitter Fantasy Score (MORE and LESS)\n"
                        "- Total Bases MORE · Hits · H+R+RBI"
                    )
                    st.markdown("**❌ AVOID:** TB LESS, HR LESS 0.5, SB LESS 0.5")
                with _brc2:
                    st.markdown("**🎫 SLIP RULES:** Mix MORE+LESS, max 2 picks/team, min B grade, 5–6 Pick Flex")
                    st.markdown("**💰 BANKROLL:** 1–2% per slip, max 5 slips/day, stop if down 10%")

            if filt:
                st.markdown(f'<div class="section-hdr">🎯 {len(filt)} Edges — Best First</div>', unsafe_allow_html=True)
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
                with st.expander("📋 Quick Summary"):
                    st.markdown(f"**{len(pp_lines)}** props available · **{len(all_edges)}** sharp edges found · **{len(ab_combined)}** A/B combined picks")
                    if all_edges and confirmed_count:
                        st.markdown(f"**{len(ab_combined)} strong picks today — {confirmed_count} confirmed by sharp books**")
                    elif all_edges:
                        st.markdown(f"**{len(ab_combined)} strong picks today** — sharp lines available but no confirmed overlaps")
                    else:
                        st.markdown("**No sharp lines yet** — showing projection-only analysis")

                # ── Betting Rules Card ──
                with st.expander("📋 Betting Rules & Bankroll Guide", expanded=False):
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("**✅ BET THESE PROPS**")
                        st.markdown(
                            "- Pitcher Strikeouts (MORE and LESS)\n"
                            "- Hitter Fantasy Score (MORE and LESS)\n"
                            "- Total Bases MORE\n"
                            "- Hits (MORE and LESS)\n"
                            "- Hits+Runs+RBIs (H+R+RBI)\n"
                            "- Batter Strikeouts"
                        )
                        st.markdown("**❌ AVOID THESE**")
                        st.markdown(
                            "- Total Bases LESS (44% accuracy — not worth it)\n"
                            "- Home Runs LESS (PP rarely offers it)\n"
                            "- SB LESS 0.5 (PP doesn't offer SB LESS)"
                        )
                    with rc2:
                        st.markdown("**🎫 SLIP RULES**")
                        st.markdown(
                            "- Mix 2–3 MORE + 2–3 LESS per slip\n"
                            "- Max 2 picks per team per slip\n"
                            "- Min B grade — never take D-grade picks\n"
                            "- Prefer 5-Pick or 6-Pick Flex\n"
                            "- Avoid 2-Pick or 3-Pick Power Play"
                        )
                        st.markdown("**💰 BANKROLL**")
                        st.markdown(
                            "- 1–2% of bankroll per slip\n"
                            "- Max 5 slips per day\n"
                            "- Break-even: 5-pick Flex = 54.2%, 6-pick = 52.9%\n"
                            "- Stop if down 10% in a day"
                        )

                # ── Today's Best Plays ──
                # Use combined scoring when sharp edges exist, fall back to projection ranking
                if scored_all:
                    top_plays = [s for s in scored_all
                                 if s["combined_grade"] in ("A+", "A")
                                 and is_tradeable_pick(s.get("stat_internal", s.get("stat_type", "")), s.get("pick", ""))][:5]
                else:
                    top_plays = []
                if not top_plays:
                    # Fall back to top A-grade projections (tradeable only)
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

                if top_plays:
                    st.markdown('<div class="section-hdr">Today\'s Best Plays</div>', unsafe_allow_html=True)
                    _bp_grade_emoji = {"A+": "💎", "A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
                    _bp_signal_short = {SIGNAL_CONFIRMED: "✅", SIGNAL_SHARP_ONLY: "📊", SIGNAL_PROJECTION_ONLY: "🔮"}
                    bp_cols = st.columns(min(len(top_plays), 5))
                    for idx, tp in enumerate(top_plays[:5]):
                        pick_cls = "more" if tp["pick"] == "MORE" else "less"
                        sig_icon = _bp_signal_short.get(tp.get("signal", ""), "")
                        grade_icon = _bp_grade_emoji.get(tp.get("combined_grade", ""), "")
                        conf_val = tp.get("proj_confidence", 0) or 0
                        with bp_cols[idx]:
                            st.markdown(f'''<div class="best-play">
                                <div class="bp-name">{grade_icon} {tp["player_name"]} {sig_icon}</div>
                                <div class="bp-prop">{tp["stat_type"]} · Line {tp["line"]}</div>
                                <div class="bp-pick {pick_cls}">{tp["pick"]}</div>
                                <div class="bp-conf">Score: {tp["combined_score"]:.3f} · {tp["combined_grade"]}</div>
                            </div>''', unsafe_allow_html=True)

                # ── Filters ──
                f1, f2, f3 = st.columns([2, 2, 2])
                prop_types_available = sorted(pdf["stat_type"].unique().tolist())
                with f1:
                    proj_prop_filter = st.selectbox("Prop Type", ["All"] + prop_types_available, key="proj_prop_f")
                with f2:
                    proj_grade_filter = st.selectbox("Min Grade", ["A only", "A + B", "A + B + C", "All"], index=1, key="proj_grade_f")
                with f3:
                    show_all_proj = st.checkbox("Show all picks (incl. non-tradeable)", value=False, key="show_all_proj")

                grade_map = {"A only": ["A"], "A + B": ["A", "B"], "A + B + C": ["A", "B", "C"], "All": ["A", "B", "C", "D"]}
                filtered = pdf[pdf["rating"].isin(grade_map[proj_grade_filter])].copy()
                if proj_prop_filter != "All":
                    filtered = filtered[filtered["stat_type"] == proj_prop_filter]
                # Filter non-tradeable picks unless user opts in
                if not show_all_proj:
                    _before_filter = len(filtered)
                    filtered = filtered[filtered.apply(
                        lambda r: is_tradeable_pick(r.get("stat_internal", ""), r.get("pick", "")), axis=1
                    )].copy()
                    _removed = _before_filter - len(filtered)
                    if _removed > 0:
                        st.info(f"ℹ️ Showing tradeable picks only. Filtered out {_removed} pick(s) — HR LESS, SB LESS, TB LESS (not available or unprofitable on PrizePicks)")

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
                        conf_pct = f"{pick_row['confidence']*100:.1f}%"
                        edge_pct = f"{pick_row['edge']*100:.1f}%"
                        with st.expander(
                            f"{grade_label(pick_row['rating'])} **{pick_row['player_name']}** · {pick_row.get('team','')} · "
                            f"{pick_row['stat_type']} {pick_row['line']} → "
                            f"**{pick_row['pick']}** {pick_row['projection']} · "
                            f"{trend_icon} · Conf {conf_pct} · Edge {edge_pct} · "
                            f"{health_icon} {spring_icon}{buy_tag}",
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
                                st.caption(f"Projection: {pick_row['projection']} vs Line: {pick_row['line']}")
                                st.caption(f"Confidence: {conf_pct} · Edge: {edge_pct}")
                                st.caption(f"Health: {health_icon} · Spring: {spring_icon} · Trend: {trend_icon}")
                                if pick_row.get("buy_low"):
                                    st.success("🎯 BUY LOW — Elite player in cold streak, regression expected")

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

                    # MORE/LESS balance indicator
                    _more_count = sum(1 for p in selected_picks if p["pick"] == "MORE")
                    _less_count = sum(1 for p in selected_picks if p["pick"] == "LESS")
                    _balance_note = "✅ good balance" if abs(_more_count - _less_count) <= 1 else "⚠️ consider balancing MORE/LESS"
                    st.success(f"{len(selected_picks)} pick(s): **{_more_count} MORE / {_less_count} LESS** — {_balance_note}")

                    # Slip correlation warnings
                    slip_warns = analyze_slip_correlation(selected_picks)
                    _has_high_warn = any(w["severity"] == "high" for w in slip_warns)
                    if _has_high_warn:
                        st.warning("⚠️ This slip has high-severity correlation risks — review warnings below")
                    for emoji, text in format_warnings_streamlit(slip_warns):
                        st.markdown(f"{emoji} {text}")

                    # Estimated slip strength from combined scores
                    _avg_confidence = 0.5
                    if scored_all:
                        _scored_lookup = {
                            f"{s['player_name'].lower()}|{s['stat_type'].lower()}": s
                            for s in scored_all
                        }
                        slip_scores = []
                        for sp in selected_picks:
                            key = f"{sp['player_name'].lower()}|{sp['stat_type'].lower()}"
                            _matched_score = _scored_lookup.get(key)
                            if _matched_score:
                                slip_scores.append(_matched_score["combined_score"])
                        if slip_scores:
                            avg_score = sum(slip_scores) / len(slip_scores)
                            strength_pct = min(50 + avg_score * 300, 85)
                            st.caption(f"Estimated slip strength: **{strength_pct:.0f}%** (avg combined score: {avg_score:.3f})")

                    # Per-pick confidence for EV calculation
                    _conf_vals = []
                    for sp in selected_picks:
                        _pk_row = pdf[(pdf["player_name"] == sp["player_name"]) & (pdf["stat_type"] == sp["stat_type"])]
                        if not _pk_row.empty:
                            _conf_vals.append(float(_pk_row.iloc[0].get("confidence", 0.5)))
                    _avg_confidence = (sum(_conf_vals) / len(_conf_vals)) if _conf_vals else 0.5

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        slip_type_opts = [f"{n}-Pick {'Power Play' if n <= 3 else 'Flex'}" for n in range(2, 7)]
                        proj_slip_type = st.selectbox("Slip type", slip_type_opts, key="proj_slip_type")
                    with sc2:
                        proj_slip_amount = st.number_input("Entry ($)", min_value=1.0, value=5.0, step=1.0, key="proj_slip_amt")

                    # Auto-calculate Flex payout preview
                    _slip_key_map = {
                        "2-Pick Power Play": "2_power", "3-Pick Power Play": "3_power",
                        "4-Pick Flex": "4_flex", "5-Pick Flex": "5_flex", "6-Pick Flex": "6_flex",
                    }
                    _slip_key = _slip_key_map.get(proj_slip_type, "5_flex")
                    _payouts_for_type = PAYOUTS.get(_slip_key, {})
                    if _payouts_for_type:
                        _num_picks = int(proj_slip_type[0])
                        _payout_rows = []
                        for _wins, _mult in sorted(_payouts_for_type.items(), reverse=True):
                            if _mult > 0:
                                # EV of this outcome: C(_num_picks, _wins) * p^wins * (1-p)^(n-wins) * mult
                                from math import comb
                                _prob = comb(_num_picks, _wins) * (_avg_confidence ** _wins) * ((1 - _avg_confidence) ** (_num_picks - _wins))
                                _payout_rows.append({
                                    "Correct": f"{_wins}/{_num_picks}",
                                    "Multiplier": f"{_mult}x",
                                    "Payout": f"${_mult * proj_slip_amount:.2f}",
                                    "Est. Prob": f"{_prob*100:.1f}%",
                                })
                        if _payout_rows:
                            st.caption("**Payout preview** (based on avg pick confidence)")
                            st.dataframe(pd.DataFrame(_payout_rows), hide_index=True, use_container_width=True)

                        # Expected value
                        _ev = sum(
                            comb(_num_picks, _w) * (_avg_confidence ** _w) * ((1 - _avg_confidence) ** (_num_picks - _w)) * _m * proj_slip_amount
                            for _w, _m in _payouts_for_type.items()
                        ) - proj_slip_amount
                        _ev_color = "g" if _ev >= 0 else "r"
                        st.markdown(f'<div class="card" style="margin-top:0.4rem"><div class="lbl">EXPECTED VALUE</div><div class="val {_ev_color}">${_ev:+.2f}</div></div>', unsafe_allow_html=True)

                    _num_needed = int(proj_slip_type[0])
                    _save_disabled = len(selected_picks) < 2 or _has_high_warn
                    if _save_disabled and len(selected_picks) < 2:
                        st.caption("Select at least 2 picks to save a slip.")
                    elif _save_disabled and _has_high_warn:
                        st.caption("Resolve high-severity correlation warnings to save.")

                    if st.button("Lock In Picks", type="primary", key="proj_save_slip", disabled=_save_disabled):
                        if len(selected_picks) == _num_needed:
                            sid = create_slip(date.today().isoformat(), proj_slip_type, proj_slip_amount, selected_picks)
                            try:
                                save_daily_log(preds)
                            except Exception:
                                pass
                            st.success(f"Slip #{sid} saved! Daily projection log saved.")
                            st.rerun()
                        else:
                            st.warning(f"Select exactly {_num_needed} picks for a {proj_slip_type}.")

with tab_slips:
    st.markdown('<div class="section-hdr">PrizePicks Slip Tracker</div>', unsafe_allow_html=True)
    # P&L Summary
    pnl = get_slip_pnl(30)
    if pnl["slips_total"] > 0:
        c1,c2,c3,c4 = st.columns(4)
        cls = "g" if pnl["net_profit"] >= 0 else "r"
        with c1: st.markdown(f'<div class="card"><div class="lbl">NET P&L</div><div class="val {cls}">${pnl["net_profit"]:+.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><div class="lbl">ROI</div><div class="val {cls}">{pnl["roi"]:+.1f}%</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><div class="lbl">WAGERED</div><div class="val">${pnl["total_wagered"]:.0f}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="card"><div class="lbl">RECORD</div><div class="val">{pnl["slips_won"]}W-{pnl["slips_lost"]}L</div></div>', unsafe_allow_html=True)

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
            status_icon = {"win": "🟢", "loss": "🔴", "partial": "🟡", "push": "⚪", "pending": "⏳"}.get(slip["status"], "⏳")
            with st.expander(f"{status_icon} Slip #{slip['id']} — {slip['entry_type']} · ${slip['entry_amount']:.0f} · {slip['game_date']} · {slip['status'].upper()}"):
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
    if pp_lines.empty: st.info("No lines available.")
    else:
        s = st.text_input("Search player", "", key="s2")
        sf = st.multiselect("Props", get_available_stat_types(pp_lines), default=get_available_stat_types(pp_lines), key="sf2")
        fa = pp_lines.copy()
        if s: fa = fa[fa["player_name"].str.contains(s, case=False, na=False)]
        if sf: fa = fa[fa["stat_type"].isin(sf)]
        da = fa[["player_name","team","stat_type","line","start_time"]].copy()
        da.columns = ["Player","Team","Prop","Line","Time"]
        if "Time" in da.columns: da["Time"] = pd.to_datetime(da["Time"], errors="coerce").dt.strftime("%-I:%M %p")
        st.dataframe(da, hide_index=True, use_container_width=True)

with tab_dash:
    st.markdown('<div class="section-hdr">Accuracy Dashboard</div>', unsafe_allow_html=True)
    stats = get_accuracy_stats()
    if stats["total"]==0: st.info("No graded picks yet. Grade results in the Grade tab.")
    else:
        acc=stats["accuracy"]
        c1,c2,c3,c4=st.columns(4)
        cls="g" if acc>=0.55 else ("y" if acc>=0.50 else "r")
        with c1: st.markdown(f'<div class="card"><div class="lbl">ACCURACY</div><div class="val {cls}">{pct(acc)}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><div class="lbl">RECORD</div><div class="val">{stats["wins"]}W-{stats["losses"]}L</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><div class="lbl">GRADED</div><div class="val">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with c4:
            t = "✅ ABOVE" if acc>=0.542 else "❌ BELOW"
            st.markdown(f'<div class="card"><div class="lbl">VS TARGET</div><div class="val {"g" if acc>=0.542 else "r"}">{t}</div></div>', unsafe_allow_html=True)
        r1,r2=st.columns(2)
        with r1:
            st.markdown("**By Grade**")
            rows=[{"Grade":r,"W-L":f"{stats.get('by_rating',{}).get(r,{}).get('wins',0)}-{stats.get('by_rating',{}).get(r,{}).get('total',0)-stats.get('by_rating',{}).get(r,{}).get('wins',0)}","Acc":pct(stats.get('by_rating',{}).get(r,{}).get('accuracy',0))} for r in "ABCD"]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with r2:
            st.markdown("**By Direction**")
            rows=[{"Pick":d,"W-L":f"{stats.get('by_direction',{}).get(d,{}).get('wins',0)}-{stats.get('by_direction',{}).get(d,{}).get('total',0)-stats.get('by_direction',{}).get(d,{}).get('wins',0)}","Acc":pct(stats.get('by_direction',{}).get(d,{}).get('accuracy',0))} for d in ["MORE","LESS"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Model Tuning Section
    st.markdown('<div class="section-hdr">Model Tuning</div>', unsafe_allow_html=True)
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

    # ── Daily Log (last 14 days) ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Daily Log — Last 14 Days</div>', unsafe_allow_html=True)
    _log_rows = get_daily_log_summary(14)
    if _log_rows:
        _log_df = pd.DataFrame(_log_rows)
        st.dataframe(_log_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No daily logs yet — logs are created automatically when you lock in picks.")

with tab_grade:
    st.markdown('<div class="section-hdr">Grade Past Picks</div>', unsafe_allow_html=True)

    # Auto-grade button
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

    st.markdown("---")
    st.markdown("**Manual Grading**")
    gd = st.date_input("Date", value=date.today()-timedelta(days=1))
    ug = get_ungraded_predictions(gd.isoformat())
    if ug.empty: st.info(f"Nothing to grade for {gd}.")
    else:
        st.markdown(f"**{len(ug)} ungraded**")
        for _,row in ug.iterrows():
            ci,cinp,cb=st.columns([3,1,1])
            with ci: st.markdown(f"**{row['player_name']}** — {row['stat_type']} · Line: `{row['line']}` · Pick: **{row['pick']}**")
            with cinp: a=st.number_input("Actual",min_value=0.0,step=0.5,key=f"a_{row['id']}",label_visibility="collapsed")
            with cb:
                if st.button("Grade",key=f"g_{row['id']}"):
                    r=grade_prediction(row["id"],a)
                    st.success("✅ W" if r=="W" else ("❌ L" if r=="L" else "➖ Push"))
                    st.rerun()
    rg=get_graded_predictions(30)
    if not rg.empty:
        st.markdown('<div class="section-hdr">Recent</div>', unsafe_allow_html=True)
        d=rg[["game_date","player_name","stat_type","line","pick","actual_result","result"]].copy()
        d.columns=["Date","Player","Prop","Line","Pick","Actual","Result"]
        st.dataframe(d,hide_index=True,use_container_width=True)

with tab_setup:
    st.markdown('<div class="section-hdr">Setup</div>', unsafe_allow_html=True)

    # ── Opening Day Readiness Checklist ──────────────────────────────────────
    st.markdown('<div class="section-hdr">⚾ Opening Day Checklist</div>', unsafe_allow_html=True)
    import os as _os, json as _json
    _checks = []

    # 1. Odds API key
    _api_key = get_api_key()
    _checks.append(("Odds API key configured", bool(_api_key), "Add ODDS_API_KEY to Streamlit Secrets or .env"))

    # 2. API credits remaining
    _credits_remaining = None
    if _api_key:
        try:
            _usage = get_api_usage(_api_key)
            _credits_remaining = _usage.get("remaining", "?")
            _credits_ok = isinstance(_credits_remaining, int) and _credits_remaining > 50
            _checks.append((f"API credits remaining: {_credits_remaining}", _credits_ok, "Running low — consider upgrading or rationing calls"))
        except Exception:
            _checks.append(("API credits remaining", False, "Could not fetch usage"))
    else:
        _checks.append(("API credits remaining", False, "No API key set"))

    # 3. FanGraphs batting cache
    _bat_cache = _os.path.join(_os.path.dirname(__file__), "data", "batting_stats_cache.csv")
    if _os.path.exists(_bat_cache):
        try:
            _bat_df = pd.read_csv(_bat_cache)
            _bat_count = len(_bat_df)
            _checks.append((f"Batting cache loaded: {_bat_count} players", _bat_count >= 50, "Run cache refresh or check data/ directory"))
        except Exception:
            _checks.append(("Batting cache loaded", False, "Cache file exists but couldn't be read"))
    else:
        _checks.append(("FanGraphs batting cache", False, "No cache — run the app once with internet to build it"))

    # 4. FanGraphs pitching cache
    _pit_cache = _os.path.join(_os.path.dirname(__file__), "data", "pitching_stats_cache.csv")
    if _os.path.exists(_pit_cache):
        try:
            _pit_df = pd.read_csv(_pit_cache)
            _pit_count = len(_pit_df)
            _checks.append((f"Pitching cache loaded: {_pit_count} pitchers", _pit_count >= 20, "Run cache refresh or check data/ directory"))
        except Exception:
            _checks.append(("Pitching cache loaded", False, "Cache file exists but couldn't be read"))
    else:
        _checks.append(("FanGraphs pitching cache", False, "No cache — run the app once with internet to build it"))

    # 5. Weight version
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
        _checks.append(("Model weights (current.json)", False, "No weights file — using predictor defaults"))

    # 6. Last backtest accuracy
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
        _checks.append(("Backtest 2025 report", False, "Not yet run — backtest in progress"))

    # 7. PrizePicks API reachable
    try:
        _pp_test = fetch_prizepicks_mlb_lines()
        _pp_ok = not _pp_test.empty
        _checks.append((f"PrizePicks API reachable ({len(_pp_test)} props today)" if _pp_ok else "PrizePicks API reachable (0 props — may be early)", _pp_ok, "Check PrizePicks API or network"))
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

    st.markdown("""
**Step 1:** Get free API key at [the-odds-api.com](https://the-odds-api.com) (500 req/month, no card)

**Step 2:** For Streamlit Cloud, add in Settings → Secrets:
```
ODDS_API_KEY = "your_key_here"
```

**Step 3: Daily Workflow**
1. Open app **before 10 AM ET** (best edge window)
2. **Find Edges** tab auto-pulls PrizePicks + sharp lines
3. Focus on **A + B grades**, especially pitcher K props
4. Build **5 or 6 pick Flex** entry on PrizePicks
5. After games → **Grade** tab to log results
6. **Dashboard** tracks accuracy over time

---

**Why this beats Rotowire:**

Rotowire = projection-based (tries to out-predict the market)

This tool = market-based (catches when PrizePicks lags behind sharp books)

The projection engine is still here for confirmation, but the core edge comes from FanDuel devigged lines disagreeing with PrizePicks. You're not trying to be smarter than Vegas — you're being faster.

---

**Entry Type Cheat Sheet:**

| Type | Break-even | Verdict |
|:-----|:----------|:--------|
| 5-pick Flex | ~54.2% | ✅ Use this |
| 6-pick Flex | ~52.9% | ✅ Best value |
| 3-pick Power | ~59.8% | ❌ Never |
| 2-pick Power | ~57.7% | ❌ Avoid |
""")
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
