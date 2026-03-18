"""
⚾ MLB Prop Edge v2.1 — Market-Based Edge Finder + Slip Builder
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

from src.prizepicks import fetch_prizepicks_mlb_lines, get_available_stat_types
from src.sharp_odds import (
    fetch_mlb_events, fetch_event_props, extract_sharp_lines,
    find_ev_edges, get_api_usage, get_api_key,
)
from src.weather import fetch_game_weather, resolve_team, STADIUMS
from src.umpires import get_umpire_k_adjustment
from src.predictor import generate_prediction, calculate_over_under_probability
from src.database import (
    init_db, log_prediction, log_batch_predictions,
    get_accuracy_stats, get_graded_predictions, get_ungraded_predictions,
    grade_prediction,
)
from src.slips import (
    init_slips_table, save_slip, grade_slip, get_all_slips,
    get_slip_stats, get_ungraded_slips, FLEX_PAYOUTS,
)

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
    .slip-pick { background: rgba(0,200,83,0.08); border: 1px solid rgba(0,200,83,0.2); border-radius: 8px; padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v)
def badge(r): return f'<span class="badge badge-{r.lower()}">{r}</span>'

init_db()
init_slips_table()

if "selected_picks" not in st.session_state:
    st.session_state.selected_picks = []

st.markdown("""<div class="hero"><h1>⚾ MLB Prop <span class="accent">Edge</span></h1><p class="sub">Sharp book devigging × PrizePicks line comparison × Slip builder</p></div>""", unsafe_allow_html=True)

tab_edge, tab_slips, tab_picks, tab_dash, tab_grade, tab_setup = st.tabs([
    "🎯 FIND EDGES", "🎫 MY SLIPS", "📋 ALL LINES", "📊 DASHBOARD", "✅ GRADE", "⚙️ SETUP"
])

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
        try:
            pp_lines = fetch_prizepicks_mlb_lines()
        except Exception:
            pp_lines = pd.DataFrame()

    if pp_lines.empty:
        st.info("No MLB lines on PrizePicks right now. Lines usually post by 10 AM ET.")
    else:
        st.markdown(f"**{len(pp_lines)} MLB props** on PrizePicks today")
        all_edges = []
        if has_sharp:
            with st.spinner("Fetching sharp lines & devigging..."):
                events = fetch_mlb_events(api_key)
                for event in (events or [])[:15]:
                    eid = event.get("id", "")
                    if not eid:
                        continue
                    result = fetch_event_props(eid, api_key=api_key)
                    if result and "data" in result:
                        sharp = extract_sharp_lines(result["data"])
                        all_edges.extend(find_ev_edges(pp_lines, sharp, min_ev_pct=0.25))

        if not all_edges:
            for _, row in pp_lines.iterrows():
                team = row.get("team", "")
                wx = None
                if team:
                    r = resolve_team(team)
                    if r in STADIUMS:
                        try:
                            wx = fetch_game_weather(r)
                        except Exception:
                            pass
                pred = generate_prediction(
                    player_name=row["player_name"], stat_type=row["stat_type"],
                    stat_internal=row["stat_internal"], line=row["line"],
                    park_team=resolve_team(team) if team else None, weather=wx,
                )
                pred["team"] = team
                pred["pp_line"] = row["line"]
                pred["edge_pct"] = pred.get("edge", 0) * 100
                pred["fair_prob"] = pred.get("confidence", 0.5)
                pred["num_books"] = 0
                pred["fanduel_agrees"] = False
                pred["market"] = row.get("stat_internal", "")
                all_edges.append(pred)

        if all_edges:
            all_edges.sort(key=lambda x: x.get("edge_pct", 0), reverse=True)
            a_n = sum(1 for e in all_edges if e.get("rating") == "A")
            b_n = sum(1 for e in all_edges if e.get("rating") == "B")
            avg_e = np.mean([e.get("edge_pct", 0) for e in all_edges]) if all_edges else 0
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="card"><div class="lbl">A-GRADE</div><div class="val g">{a_n}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="card"><div class="lbl">B-GRADE</div><div class="val">{b_n}</div></div>', unsafe_allow_html=True)
            with c3:
                cls = "g" if avg_e > 5 else ("y" if avg_e > 3 else "r")
                st.markdown(f'<div class="card"><div class="lbl">AVG EDGE</div><div class="val {cls}">{avg_e:.1f}%</div></div>', unsafe_allow_html=True)
            with c4:
                sel_count = len(st.session_state.selected_picks)
                st.markdown(f'<div class="card"><div class="lbl">SELECTED</div><div class="val {"g" if 5 <= sel_count <= 6 else "y" if sel_count > 0 else ""}">{sel_count}</div></div>', unsafe_allow_html=True)

            f1, f2 = st.columns(2)
            with f1:
                min_grade = st.selectbox("Min grade", ["A only", "A + B", "A + B + C", "All"], index=1)
            with f2:
                prop_f = st.selectbox("Prop type", ["All", "Pitcher Ks", "Batter Hits", "Total Bases", "Home Runs"])
            gm = {"A only": ["A"], "A + B": ["A", "B"], "A + B + C": ["A", "B", "C"], "All": ["A", "B", "C", "D"]}
            filt = [e for e in all_edges if e.get("rating", "D") in gm[min_grade]]
            if prop_f == "Pitcher Ks":
                filt = [e for e in filt if "strikeout" in e.get("market", "").lower() and "batter" not in e.get("market", "").lower()]
            elif prop_f == "Batter Hits":
                filt = [e for e in filt if "hits" in e.get("market", "").lower()]
            elif prop_f == "Total Bases":
                filt = [e for e in filt if "total_bases" in e.get("market", "").lower()]
            elif prop_f == "Home Runs":
                filt = [e for e in filt if "home_run" in e.get("market", "").lower()]

            if filt:
                st.markdown(f'<div class="section-hdr">🎯 {len(filt)} Edges — Check picks to add to your slip</div>', unsafe_allow_html=True)

                for i, edge in enumerate(filt):
                    pick_key = f"{edge['player_name']}|{edge.get('stat_type','')}|{edge.get('pp_line',0)}"
                    is_selected = pick_key in [p.get("key") for p in st.session_state.selected_picks]

                    col_check, col_info, col_pick, col_edge, col_grade = st.columns([0.5, 4, 1, 1, 0.8])
                    with col_check:
                        checked = st.checkbox("", value=is_selected, key=f"chk_{i}", label_visibility="collapsed")
                        if checked and not is_selected:
                            st.session_state.selected_picks.append({"key": pick_key, "data": edge})
                            st.rerun()
                        elif not checked and is_selected:
                            st.session_state.selected_picks = [p for p in st.session_state.selected_picks if p["key"] != pick_key]
                            st.rerun()
                    with col_info:
                        st.markdown(f"**{edge['player_name']}** · {edge.get('team','')} · {edge.get('stat_type','')}")
                    with col_pick:
                        pick_dir = edge.get("pick", "")
                        cls = "more" if pick_dir == "MORE" else "less"
                        st.markdown(f'<span class="{cls}">{pick_dir}</span> `{edge.get("pp_line", 0)}`', unsafe_allow_html=True)
                    with col_edge:
                        st.markdown(f"`+{edge.get('edge_pct', 0):.1f}%`")
                    with col_grade:
                        st.markdown(badge(edge.get("rating", "D")), unsafe_allow_html=True)

                selected = st.session_state.selected_picks
                if selected:
                    st.markdown("---")
                    st.markdown(f'<div class="section-hdr">🎫 Slip Builder — {len(selected)} picks selected</div>', unsafe_allow_html=True)
                    for p in selected:
                        d = p["data"]
                        pick_cls = "more" if d.get("pick") == "MORE" else "less"
                        st.markdown(
                            f'<div class="slip-pick">{badge(d.get("rating","D"))} '
                            f'<strong>{d["player_name"]}</strong> · {d.get("stat_type","")} · '
                            f'<span class="{pick_cls}">{d.get("pick","")}</span> {d.get("pp_line",0)} · '
                            f'+{d.get("edge_pct",0):.1f}% edge</div>',
                            unsafe_allow_html=True,
                        )

                    n = len(selected)
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        slip_type = f"{n}-pick"
                        mode = st.selectbox("Entry type", ["Flex", "Power"], key="slip_mode")
                        st.caption(f"**{slip_type} {mode}**")
                    with sc2:
                        amount = st.number_input("Wager ($)", min_value=1.0, value=10.0, step=5.0, key="slip_amt")
                    with sc3:
                        payouts = FLEX_PAYOUTS.get(n, {})
                        if mode == "Flex" and payouts:
                            best = max(payouts.values())
                            partial = payouts.get(n - 1, 0)
                            st.metric("Max payout", f"${amount * best:.0f}")
                            st.caption(f"{n-1}/{n} correct = ${amount * partial:.0f}")

                    if n < 2:
                        st.warning("Need at least 2 picks for a PrizePicks entry.")
                    elif n > 6:
                        st.warning("PrizePicks max is 6 picks. Remove some.")
                    else:
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button(f"💾 Save {slip_type} {mode} Slip", type="primary", use_container_width=True):
                                pick_ids = []
                                for p in selected:
                                    d = p["data"]
                                    log_prediction(d, date.today().isoformat())
                                    from src.database import get_connection
                                    conn = get_connection()
                                    cur = conn.execute("SELECT MAX(id) FROM predictions")
                                    pid = cur.fetchone()[0]
                                    conn.close()
                                    pick_ids.append(pid)
                                slip_id = save_slip(
                                    pick_ids=pick_ids,
                                    picks_data=[p["data"] for p in selected],
                                    slip_type=slip_type,
                                    entry_mode=mode.lower(),
                                    entry_amount=amount,
                                )
                                st.success(f"✅ Saved {slip_type} {mode} slip (ID: {slip_id}) — ${amount:.0f} entry")
                                st.session_state.selected_picks = []
                                st.rerun()
                        with b2:
                            if st.button("🗑️ Clear selections", use_container_width=True):
                                st.session_state.selected_picks = []
                                st.rerun()
            else:
                st.info("No edges match your filters.")

with tab_slips:
    st.markdown('<div class="section-hdr">🎫 My Pick Slips</div>', unsafe_allow_html=True)
    slip_stats = get_slip_stats()
    if slip_stats.get("total_slips", 0) == 0:
        st.info("No slips saved yet. Go to Find Edges, check the picks you like, and save them as a slip.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        profit = slip_stats.get("total_profit", 0)
        roi = slip_stats.get("roi", 0)
        with c1:
            cls = "g" if profit > 0 else "r"
            st.markdown(f'<div class="card"><div class="lbl">TOTAL P&L</div><div class="val {cls}">${profit:+.2f}</div></div>', unsafe_allow_html=True)
        with c2:
            cls = "g" if roi > 0 else "r"
            st.markdown(f'<div class="card"><div class="lbl">ROI</div><div class="val {cls}">{roi:+.1f}%</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card"><div class="lbl">RECORD</div><div class="val">{slip_stats.get("wins",0)}W-{slip_stats.get("losses",0)}L</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="card"><div class="lbl">WAGERED</div><div class="val">${slip_stats.get("total_wagered",0):.0f}</div></div>', unsafe_allow_html=True)

        by_type = slip_stats.get("by_type", {})
        if by_type:
            st.markdown("**By Entry Type**")
            type_rows = []
            for st_name, data in sorted(by_type.items()):
                type_rows.append({"Type": st_name, "W": data.get("wins", 0), "L": data.get("losses", 0),
                    "Partial": data.get("partial", 0), "Profit": f"${data.get('profit', 0):+.2f}",
                    "ROI": f"{data.get('roi', 0):+.1f}%"})
            st.dataframe(pd.DataFrame(type_rows), hide_index=True, use_container_width=True)

    all_slips_df = get_all_slips(50)
    if not all_slips_df.empty:
        st.markdown('<div class="section-hdr">Slip History</div>', unsafe_allow_html=True)
        disp = all_slips_df[["game_date", "slip_type", "entry_mode", "entry_amount",
            "total_picks", "correct_picks", "payout_multiplier", "profit", "result", "picks_summary"]].copy()
        disp.columns = ["Date", "Type", "Mode", "Wager", "Picks", "Correct", "Mult", "Profit", "Result", "Summary"]
        disp["Wager"] = disp["Wager"].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "")
        disp["Profit"] = disp["Profit"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
        disp["Mult"] = disp["Mult"].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "—")
        disp["Result"] = disp["Result"].fillna("⏳ Pending")
        st.dataframe(disp, hide_index=True, use_container_width=True)

    ungraded_slips = get_ungraded_slips()
    if not ungraded_slips.empty:
        st.markdown('<div class="section-hdr">Grade Pending Slips</div>', unsafe_allow_html=True)
        st.caption("Grade individual picks first in the Grade tab, then click here to calculate slip results.")
        for _, slip_row in ungraded_slips.iterrows():
            col_i, col_b = st.columns([4, 1])
            with col_i:
                st.markdown(f"**{slip_row['slip_type']} {slip_row['entry_mode']}** · {slip_row['game_date']} · ${slip_row['entry_amount']:.0f}")
            with col_b:
                if st.button("Grade", key=f"gs_{slip_row['id']}"):
                    result = grade_slip(slip_row["id"])
                    if result.get("status") == "graded":
                        r = result["result"]
                        emoji = "✅" if r == "win" else ("🟡" if "partial" in r else "❌")
                        st.success(f"{emoji} {result['correct']}/{result['total']} · {result['multiplier']}x · ${result.get('profit', 0):+.2f}")
                    elif result.get("status") == "incomplete":
                        st.warning(f"⏳ {result.get('ungraded', 0)} picks still need grading")
                    st.rerun()

with tab_picks:
    st.markdown('<div class="section-hdr">All PrizePicks MLB Lines</div>', unsafe_allow_html=True)
    if pp_lines.empty:
        st.info("No lines available.")
    else:
        s = st.text_input("Search player", "", key="s2")
        sf = st.multiselect("Props", get_available_stat_types(pp_lines), default=get_available_stat_types(pp_lines), key="sf2")
        fa = pp_lines.copy()
        if s:
            fa = fa[fa["player_name"].str.contains(s, case=False, na=False)]
        if sf:
            fa = fa[fa["stat_type"].isin(sf)]
        da = fa[["player_name", "team", "stat_type", "line", "start_time"]].copy()
        da.columns = ["Player", "Team", "Prop", "Line", "Time"]
        if "Time" in da.columns:
            da["Time"] = pd.to_datetime(da["Time"], errors="coerce").dt.strftime("%-I:%M %p")
        st.dataframe(da, hide_index=True, use_container_width=True)

with tab_dash:
    st.markdown('<div class="section-hdr">Accuracy Dashboard</div>', unsafe_allow_html=True)
    stats = get_accuracy_stats()
    if stats["total"] == 0:
        st.info("No graded picks yet. Grade results in the Grade tab.")
    else:
        acc = stats["accuracy"]
        c1, c2, c3, c4 = st.columns(4)
        cls = "g" if acc >= 0.55 else ("y" if acc >= 0.50 else "r")
        with c1:
            st.markdown(f'<div class="card"><div class="lbl">ACCURACY</div><div class="val {cls}">{pct(acc)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card"><div class="lbl">RECORD</div><div class="val">{stats["wins"]}W-{stats["losses"]}L</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card"><div class="lbl">GRADED</div><div class="val">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with c4:
            t = "✅ ABOVE" if acc >= 0.542 else "❌ BELOW"
            st.markdown(f'<div class="card"><div class="lbl">VS TARGET</div><div class="val {"g" if acc >= 0.542 else "r"}">{t}</div></div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**By Grade**")
            rows = []
            for r in "ABCD":
                d = stats.get('by_rating', {}).get(r, {})
                rows.append({"Grade": r, "W-L": f"{d.get('wins',0)}-{d.get('total',0)-d.get('wins',0)}", "Acc": pct(d.get('accuracy', 0))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with r2:
            st.markdown("**By Direction**")
            rows = []
            for d in ["MORE", "LESS"]:
                data = stats.get('by_direction', {}).get(d, {})
                rows.append({"Pick": d, "W-L": f"{data.get('wins',0)}-{data.get('total',0)-data.get('wins',0)}", "Acc": pct(data.get('accuracy', 0))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_grade:
    st.markdown('<div class="section-hdr">Grade Past Picks</div>', unsafe_allow_html=True)
    gd = st.date_input("Date", value=date.today() - timedelta(days=1))
    ug = get_ungraded_predictions(gd.isoformat())
    if ug.empty:
        st.info(f"Nothing to grade for {gd}.")
    else:
        st.markdown(f"**{len(ug)} ungraded**")
        for _, row in ug.iterrows():
            ci, cinp, cb = st.columns([3, 1, 1])
            with ci:
                st.markdown(f"**{row['player_name']}** — {row['stat_type']} · Line: `{row['line']}` · Pick: **{row['pick']}**")
            with cinp:
                a = st.number_input("Actual", min_value=0.0, step=0.5, key=f"a_{row['id']}", label_visibility="collapsed")
            with cb:
                if st.button("Grade", key=f"g_{row['id']}"):
                    r = grade_prediction(row["id"], a)
                    st.success("✅ W" if r == "W" else ("❌ L" if r == "L" else "➖ Push"))
                    st.rerun()
    rg = get_graded_predictions(30)
    if not rg.empty:
        st.markdown('<div class="section-hdr">Recent</div>', unsafe_allow_html=True)
        d = rg[["game_date", "player_name", "stat_type", "line", "pick", "actual_result", "result"]].copy()
        d.columns = ["Date", "Player", "Prop", "Line", "Pick", "Actual", "Result"]
        st.dataframe(d, hide_index=True, use_container_width=True)

with tab_setup:
    st.markdown('<div class="section-hdr">Setup</div>', unsafe_allow_html=True)
    st.markdown("""
**Step 1:** Get free API key at [the-odds-api.com](https://the-odds-api.com) (500 req/month, no card)

**Step 2:** In Streamlit Cloud → Settings → Secrets:
```
ODDS_API_KEY = "your_key_here"
```

**Step 3: Daily Workflow**
1. Open app **before 10 AM ET**
2. **Find Edges** → check the picks you want
3. **Save as slip** → 5 or 6 pick Flex
4. Place on PrizePicks
5. After games → **Grade** tab → grade individual picks
6. Then **My Slips** → grade the slip → see P&L

---

**Entry Type Cheat Sheet:**

| Type | Break-even | Payouts | Verdict |
|:-----|:----------|:--------|:--------|
| 2-pick Flex | ~57.7% | 3x / 1.5x | Hard |
| 3-pick Flex | ~59.8% | 5x / 1.5x | ❌ Avoid |
| 4-pick Flex | ~54.8% | 10x / 2x / 0.4x | Okay |
| 5-pick Flex | ~54.2% | 10x / 2x / 0.4x | ✅ Sweet spot |
| 6-pick Flex | ~52.9% | 25x / 2x / 0.4x | ✅ Best value |
""")
    if st.button("Check API Credits"):
        k = get_api_key()
        if k:
            u = get_api_usage(k)
            st.info(f"Remaining: **{u.get('remaining', '?')}** · Used: **{u.get('used', '?')}**")
        else:
            st.warning("No API key set.")
