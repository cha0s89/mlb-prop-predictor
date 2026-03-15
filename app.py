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
</style>
""", unsafe_allow_html=True)

def pct(v): return f"{v*100:.1f}%" if isinstance(v, (int, float)) else str(v)
def badge(r): return f'<span class="badge badge-{r.lower()}">{r}</span>'
def pick_span(p): return f'<span class="{"more" if p=="MORE" else "less"}">{p}</span>'

init_db()

st.markdown("""<div class="hero"><h1>⚾ MLB Prop <span class="accent">Edge</span></h1><p class="sub">Sharp book devigging × PrizePicks line comparison × Statcast confirmation</p></div>""", unsafe_allow_html=True)

tab_edge, tab_picks, tab_dash, tab_grade, tab_setup = st.tabs(["🎯 FIND EDGES", "📋 ALL LINES", "📊 DASHBOARD", "✅ GRADE", "⚙️ SETUP"])

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
            with st.spinner("Fetching sharp lines & devigging..."):
                events = fetch_mlb_events(api_key)
                for event in (events or [])[:15]:
                    eid = event.get("id","")
                    if not eid: continue
                    result = fetch_event_props(eid, api_key=api_key)
                    if result and "data" in result:
                        sharp = extract_sharp_lines(result["data"])
                        all_edges.extend(find_ev_edges(pp_lines, sharp, min_ev_pct=0.25))

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

            f1,f2 = st.columns(2)
            with f1: min_grade = st.selectbox("Min grade", ["A only","A + B","A + B + C","All"], index=1)
            with f2: prop_f = st.selectbox("Prop type", ["All","Pitcher Ks","Batter Hits","Total Bases","Home Runs"])
            gm = {"A only":["A"],"A + B":["A","B"],"A + B + C":["A","B","C"],"All":["A","B","C","D"]}
            filt = [e for e in all_edges if e["rating"] in gm[min_grade]]
            if prop_f=="Pitcher Ks": filt=[e for e in filt if "strikeout" in e.get("market","").lower() and "batter" not in e.get("market","").lower()]
            elif prop_f=="Batter Hits": filt=[e for e in filt if "hits" in e.get("market","").lower()]
            elif prop_f=="Total Bases": filt=[e for e in filt if "total_bases" in e.get("market","").lower()]
            elif prop_f=="Home Runs": filt=[e for e in filt if "home_run" in e.get("market","").lower()]

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
                    with st.expander(f"{badge(edge['rating'])} **{edge['player_name']}** — {edge['stat_type']} | Line: {edge['pp_line']} | Edge: +{edge['edge_pct']:.1f}%"):
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
        elif has_sharp: st.info("No edges found. Lines may be properly priced or props not posted yet.")

        if not has_sharp:
            st.markdown('<div class="section-hdr">Projection Analysis (No Sharp Odds)</div>', unsafe_allow_html=True)
            preds = []
            for _, row in pp_lines.iterrows():
                team = row.get("team","")
                wx = None
                if team:
                    r = resolve_team(team)
                    if r in STADIUMS:
                        try: wx = fetch_game_weather(r)
                        except: pass
                p = generate_prediction(player_name=row["player_name"], stat_type=row["stat_type"], stat_internal=row["stat_internal"], line=row["line"], park_team=resolve_team(team) if team else None, weather=wx)
                p["team"] = team
                preds.append(p)
            pdf = pd.DataFrame(preds).sort_values("confidence", ascending=False)
            d = pdf[["player_name","team","stat_type","line","projection","pick","confidence","edge","rating"]].head(30).copy()
            d.columns = ["Player","Team","Prop","Line","Proj","Pick","Conf","Edge","Grade"]
            d["Conf"] = d["Conf"].apply(lambda x: f"{x*100:.1f}%")
            d["Edge"] = d["Edge"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(d, hide_index=True, use_container_width=True)

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

with tab_grade:
    st.markdown('<div class="section-hdr">Grade Past Picks</div>', unsafe_allow_html=True)
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
    if st.button("Check API Credits"):
        k=get_api_key()
        if k:
            u=get_api_usage(k)
            st.info(f"Remaining: **{u.get('remaining','?')}** · Used: **{u.get('used','?')}**")
        else: st.warning("No API key set.")
