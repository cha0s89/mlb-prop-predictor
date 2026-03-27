#!/usr/bin/env python3
"""
v039 rescore: honest tradeable accuracy with corrected PP_NEVER_SHOW.

Disabled (floor 0.97):
  - rbis_less (structural edge collapse)
  - hits_runs_rbis_more (46.3% even with 0.65 floor on variable lines)

PP_NEVER_SHOW blocks (not tradeable on PrizePicks):
  - home_runs LESS, stolen_bases LESS, total_bases LESS
  - hitter_fantasy_score MORE
  - rbis LESS, doubles LESS, triples LESS
  - singles both, walks both, hits_allowed both
"""
import json
from collections import defaultdict

BACKTEST_PATH = "C:/Users/Unknown/Downloads/mlb-prop-predictor/data/backtest/backtest_2025_v035_rescored.json"

# ── PP_NEVER_SHOW: hard blocks regardless of confidence ───────────────────────
NEVER_SHOW = {
    ("home_runs", "LESS"),
    ("stolen_bases", "LESS"),
    ("total_bases", "LESS"),
    ("hitter_fantasy_score", "MORE"),
    ("rbis", "LESS"),
    ("doubles", "LESS"),
    ("triples", "LESS"),
    ("singles", "MORE"),
    ("singles", "LESS"),
    ("walks", "MORE"),
    ("walks", "LESS"),
    ("hits_allowed", "MORE"),
    ("hits_allowed", "LESS"),
}

# ── PP_TRADEABLE: what PrizePicks actually offers ─────────────────────────────
PP_TRADEABLE = {
    "pitcher_strikeouts":   {"MORE", "LESS"},
    "hitter_fantasy_score": {"MORE", "LESS"},
    "total_bases":          {"MORE"},           # LESS is NEVER_SHOW
    "hits_runs_rbis":       {"MORE", "LESS"},
    "hits":                 {"MORE", "LESS"},
    "batter_strikeouts":    {"MORE", "LESS"},
    "walks_allowed":        {"MORE", "LESS"},
    "earned_runs":          {"MORE", "LESS"},
    "pitching_outs":        {"MORE", "LESS"},
    "runs":                 {"MORE"},
    "rbis":                 {"MORE"},           # LESS is NEVER_SHOW
}

def is_tradeable(prop_type: str, pick: str) -> bool:
    if (prop_type, pick) in NEVER_SHOW:
        return False
    cfg = PP_TRADEABLE.get(prop_type)
    if cfg is None:
        return False
    return pick in cfg

# ── v038 baseline floors (current.json before v039) ──────────────────────────
FLOORS_V038 = {
    "hits_more": 0.72,        "hits_less": 0.72,
    "total_bases_more": 0.64, "total_bases_less": 0.72,
    "pitcher_strikeouts_more": 0.66, "pitcher_strikeouts_less": 0.66,
    "hitter_fantasy_score_more": 0.65, "hitter_fantasy_score_less": 0.68,
    "pitching_outs_more": 0.60, "pitching_outs_less": 0.60,
    "earned_runs_more": 0.60, "earned_runs_less": 0.60,
    "walks_allowed_more": 0.95, "walks_allowed_less": 0.95,
    "hits_allowed_more": 0.60, "hits_allowed_less": 0.60,
    "batter_strikeouts_more": 0.66, "batter_strikeouts_less": 0.66,
    "walks_more": 0.60, "walks_less": 0.70,
    "rbis_more": 0.60, "rbis_less": 0.63,
    "runs_more": 0.68, "runs_less": 0.68,
    "singles_more": 0.95, "singles_less": 0.95,
    "doubles_more": 0.60, "doubles_less": 0.60,
    "hits_runs_rbis_more": 0.95, "hits_runs_rbis_less": 0.95,
}

# ── v039 floors: rbis_less and hits_runs_rbis_more disabled ──────────────────
FLOORS_V039 = {
    **FLOORS_V038,
    "rbis_less": 0.97,
    "hits_runs_rbis_more": 0.97,
}

# ── Analysis helpers ──────────────────────────────────────────────────────────

def apply_floors(data, floors, label):
    selected = []
    by_prop = defaultdict(lambda: {"sel": 0, "wins": 0, "all": 0, "all_wins": 0})

    for row in data:
        pt = row["prop_type"]
        pick = row.get("pick", "")
        if not is_tradeable(pt, pick):
            continue
        conf = float(row.get("confidence", 0))
        result = row.get("result", "")
        is_win = 1 if result == "W" else 0
        key = f"{pt}_{pick.lower()}"
        fl = floors.get(key, 0.60)

        by_prop[key]["all"] += 1
        by_prop[key]["all_wins"] += is_win

        if conf >= fl:
            selected.append({"prop_type": pt, "pick": pick, "conf": conf, "win": is_win, "key": key})
            by_prop[key]["sel"] += 1
            by_prop[key]["wins"] += is_win

    n = len(selected)
    wins = sum(r["win"] for r in selected)
    acc = wins / n if n else 0

    prop_rows = []
    for key in sorted(by_prop):
        d = by_prop[key]
        fl = floors.get(key, 0.60)
        sel_acc = d["wins"] / d["sel"] if d["sel"] else None
        all_acc = d["all_wins"] / d["all"] if d["all"] else None
        if d["all"] > 0:
            prop_rows.append({
                "key": key, "floor": fl,
                "all_n": d["all"], "all_acc": all_acc,
                "sel_n": d["sel"], "sel_acc": sel_acc,
            })

    return {"label": label, "selected": n, "wins": wins, "accuracy": acc, "props": prop_rows}


def print_summary(r):
    print(f"\n{'='*60}")
    print(f"  {r['label']} — HONEST TRADEABLE RESCORE")
    print(f"{'='*60}")
    print(f"  Total picks : {r['selected']:,}")
    print(f"  Accuracy    : {r['accuracy']*100:.2f}%")
    print(f"  Wins/Losses : {r['wins']:,} W / {r['selected'] - r['wins']:,} L")
    print(f"  Daily avg   : ~{r['selected'] / 183:.0f} picks/day (183 game-days)")


def print_prop_table(r):
    print(f"\n{'─'*72}")
    print(f"  PER-PROP BREAKDOWN — {r['label']}")
    print(f"{'─'*72}")
    print(f"  {'Prop+Dir':<35} {'Floor':>6} {'Sel N':>8} {'Sel Acc':>9} {'All Acc':>9}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*9} {'-'*9}")
    for row in sorted(r["props"], key=lambda x: -(x["sel_n"])):
        sa = f"{row['sel_acc']*100:.1f}%" if row["sel_acc"] is not None else "n/a"
        aa = f"{row['all_acc']*100:.1f}%" if row["all_acc"] is not None else "n/a"
        print(f"  {row['key']:<35} {row['floor']:>6.2f} {row['sel_n']:>8,} {sa:>9} {aa:>9}")
    print(f"  {'TOTAL':<35} {'':>6} {r['selected']:>8,} {r['accuracy']*100:>8.2f}%")


def print_comparison(r_base, r_new):
    print(f"\n{'='*60}")
    print("  COMPARISON: v038 (baseline) vs v039")
    print(f"{'='*60}")
    print(f"  {'Version':<10} {'Picks':>10} {'Accuracy':>10} {'Delta':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    base_acc = r_base["accuracy"]
    for r in [r_base, r_new]:
        delta = f"{(r['accuracy'] - base_acc)*100:+.2f}pp" if r["label"] != r_base["label"] else "baseline"
        print(f"  {r['label']:<10} {r['selected']:>10,} {r['accuracy']*100:>9.2f}%  {delta:>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading backtest data...")
with open(BACKTEST_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data):,} picks")

print("\nScoring v038 (baseline — pre-v039 floors, same NEVER_SHOW filter)...")
r038 = apply_floors(data, FLOORS_V038, "v038")

print("Scoring v039...")
r039 = apply_floors(data, FLOORS_V039, "v039")

print_comparison(r038, r039)
print_summary(r039)
print_prop_table(r039)

print(f"\n{'='*60}")
best = max([r038, r039], key=lambda r: r["accuracy"])
print(f"BEST: {best['label']}  {best['accuracy']*100:.2f}%  ({best['selected']:,} picks)")
delta_picks = r039["selected"] - r038["selected"]
delta_acc   = (r039["accuracy"] - r038["accuracy"]) * 100
print(f"v039 vs v038: {delta_picks:+,} picks  {delta_acc:+.2f}pp accuracy")
print(f"{'='*60}")
