#!/usr/bin/env python3
"""
Optimization Cycle 4: v037 vs v038 rescore.

Uses backtest_2025_v035_rescored.json (464K picks with confidence data).
v037 = current/v034 floors (baseline at 73.21% / 89K picks).
v038 = unlock weak props with empirically-tested floors.
"""
import json
from collections import defaultdict

BACKTEST_PATH = "C:/Users/Unknown/Downloads/mlb-prop-predictor/data/backtest/backtest_2025_v035_rescored.json"

# ── Tradeable prop definitions ────────────────────────────────────────────────
TRADEABLE_PROPS = {
    "pitcher_strikeouts", "pitching_outs", "hits", "total_bases",
    "hitter_fantasy_score", "earned_runs", "runs", "rbis",
    "batter_strikeouts", "walks_allowed", "hits_runs_rbis",
    "home_runs", "doubles", "triples",
}
TRADEABLE_MORE_ONLY = {"home_runs", "doubles", "triples"}
BLOCKED_PROPS = {"singles", "walks", "hits_allowed"}

def is_tradeable(prop_type, pick):
    if prop_type in BLOCKED_PROPS:
        return False
    if prop_type in TRADEABLE_MORE_ONLY:
        return pick == "MORE"
    return prop_type in TRADEABLE_PROPS

# ── Floor configs ─────────────────────────────────────────────────────────────

# v037 = current.json as-was before this update (v034 floors)
FLOORS_V037 = {
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

# v038: unlock weak props, disable anti-predictive ones
FLOORS_V038 = {
    "hits_more": 0.72,        "hits_less": 0.72,
    "total_bases_more": 0.97, "total_bases_less": 0.58,   # tb_more disabled; tb_less relaxed
    "pitcher_strikeouts_more": 0.66, "pitcher_strikeouts_less": 0.66,
    "hitter_fantasy_score_more": 0.65, "hitter_fantasy_score_less": 0.68,
    "pitching_outs_more": 0.60, "pitching_outs_less": 0.60,
    "earned_runs_more": 0.55, "earned_runs_less": 0.97,   # er_more relaxed; er_less disabled
    "walks_allowed_more": 0.97, "walks_allowed_less": 0.55, # wa_more disabled; wa_less unlocked
    "hits_allowed_more": 0.60, "hits_allowed_less": 0.60,
    "batter_strikeouts_more": 0.66, "batter_strikeouts_less": 0.66,
    "walks_more": 0.60, "walks_less": 0.70,
    "rbis_more": 0.97, "rbis_less": 0.55,   # rbis_more disabled; rbis_less relaxed
    "runs_more": 0.68, "runs_less": 0.68,
    "singles_more": 0.95, "singles_less": 0.95,
    "doubles_more": 0.60, "doubles_less": 0.60,
    "hits_runs_rbis_more": 0.65, "hits_runs_rbis_less": 0.62,  # unlocked
}

# ── Analysis helpers ──────────────────────────────────────────────────────────

def floor_key(prop_type, pick):
    return f"{prop_type}_{pick.lower()}"

def apply_floors(data, floors, label):
    """Filter tradeable picks by floors, return stats."""
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
        key = floor_key(pt, pick)
        fl = floors.get(key, 0.60)

        pd_key = f"{pt}_{pick.lower()}"
        by_prop[pd_key]["all"] += 1
        by_prop[pd_key]["all_wins"] += is_win

        if conf >= fl:
            selected.append({"prop_type": pt, "pick": pick, "conf": conf, "win": is_win, "key": pd_key})
            by_prop[pd_key]["sel"] += 1
            by_prop[pd_key]["wins"] += is_win

    n = len(selected)
    wins = sum(r["win"] for r in selected)
    acc = wins / n if n else 0

    prop_rows = []
    for pd_key in sorted(by_prop):
        d = by_prop[pd_key]
        pt_dir = pd_key  # e.g. "hits_runs_rbis_more"
        # Split on last underscore to get direction
        last_under = pd_key.rfind("_")
        direction = pd_key[last_under+1:]
        pt = pd_key[:last_under]
        key = f"{pt}_{direction}"
        fl = floors.get(key, 0.60)
        sel_acc = d["wins"] / d["sel"] if d["sel"] else None
        all_acc = d["all_wins"] / d["all"] if d["all"] else None
        if d["all"] > 0:
            prop_rows.append({
                "key": pd_key,
                "floor": fl,
                "all_n": d["all"],
                "all_acc": all_acc,
                "sel_n": d["sel"],
                "sel_acc": sel_acc,
            })

    return {
        "label": label,
        "selected": n,
        "wins": wins,
        "accuracy": acc,
        "props": prop_rows,
    }

def grade_breakdown(data, floors, label):
    """Break down by confidence grade bucket: A>=0.70, B 0.62-0.70, C 0.57-0.62, D <0.57."""
    grades = {
        "A": {"floor": 0.70, "sel": 0, "wins": 0},
        "B": {"floor": 0.62, "sel": 0, "wins": 0},
        "C": {"floor": 0.57, "sel": 0, "wins": 0},
        "D": {"floor": 0.0,  "sel": 0, "wins": 0},
    }

    for row in data:
        pt = row["prop_type"]
        pick = row.get("pick", "")
        if not is_tradeable(pt, pick):
            continue
        conf = float(row.get("confidence", 0))
        result = row.get("result", "")
        is_win = 1 if result == "W" else 0
        key = floor_key(pt, pick)
        fl = floors.get(key, 0.60)

        if conf < fl:
            continue

        if conf >= 0.70:
            g = "A"
        elif conf >= 0.62:
            g = "B"
        elif conf >= 0.57:
            g = "C"
        else:
            g = "D"
        grades[g]["sel"] += 1
        grades[g]["wins"] += is_win

    return grades

def direction_summary(results):
    more_sel = more_wins = less_sel = less_wins = 0
    for row in results["props"]:
        if row["sel_n"] == 0:
            continue
        if row["key"].endswith("_more"):
            more_sel += row["sel_n"]
            more_wins += round(row["sel_n"] * row["sel_acc"])
        else:
            less_sel += row["sel_n"]
            less_wins += round(row["sel_n"] * row["sel_acc"])
    return {
        "MORE": {"sel": more_sel, "acc": more_wins / more_sel if more_sel else 0},
        "LESS": {"sel": less_sel, "acc": less_wins / less_sel if less_sel else 0},
    }

# ── Print helpers ─────────────────────────────────────────────────────────────

def print_summary_table(results_list):
    print("\n" + "="*70)
    print("COMPARISON: v037 (baseline) vs v038  (TRADEABLE ONLY)")
    print("="*70)
    print(f"{'Version':<10} {'Picks':>10} {'Accuracy':>10} {'Delta vs v037':>15}")
    print("-"*50)
    base_acc = None
    for r in results_list:
        acc = r["accuracy"]
        delta = f"{(acc - base_acc)*100:+.2f}pp" if base_acc else "baseline"
        if base_acc is None:
            base_acc = acc
        print(f"  {r['label']:<8} {r['selected']:>10,} {acc*100:>9.2f}%  {delta:>15}")

def print_prop_breakdown(results):
    print(f"\n{'─'*75}")
    print(f"PER-PROP ACCURACY — {results['label']}")
    print(f"{'─'*75}")
    print(f"  {'Prop+Dir':<38} {'Floor':>6} {'Sel N':>8} {'Sel Acc':>9} {'All Acc':>9}")
    print(f"  {'-'*38} {'-'*6} {'-'*8} {'-'*9} {'-'*9}")
    for row in sorted(results["props"], key=lambda x: -(x["sel_n"])):
        sa = f"{row['sel_acc']*100:.1f}%" if row["sel_acc"] is not None else "n/a"
        aa = f"{row['all_acc']*100:.1f}%" if row["all_acc"] is not None else "n/a"
        print(f"  {row['key']:<38} {row['floor']:>6.2f} {row['sel_n']:>8,} {sa:>9} {aa:>9}")
    total = results["selected"]
    acc = results["accuracy"]
    print(f"  {'TOTAL':<38} {'':>6} {total:>8,} {acc*100:>8.2f}%")

def print_grade_breakdown(grades, label):
    print(f"\n{'─'*55}")
    print(f"GRADE BREAKDOWN — {label}")
    print(f"{'─'*55}")
    print(f"  {'Grade':<8} {'Conf Bucket':<20} {'Picks':>8} {'Accuracy':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*10}")
    grade_labels = {
        "A": ">=0.70",
        "B": "0.62-0.70",
        "C": "0.57-0.62",
        "D": "<0.57",
    }
    total_sel = total_wins = 0
    for g in ["A", "B", "C", "D"]:
        d = grades[g]
        acc = d["wins"] / d["sel"] if d["sel"] else 0
        total_sel += d["sel"]
        total_wins += d["wins"]
        print(f"  {g:<8} {grade_labels[g]:<20} {d['sel']:>8,} {acc*100:>9.2f}%")
    total_acc = total_wins / total_sel if total_sel else 0
    print(f"  {'TOTAL':<8} {'':>20} {total_sel:>8,} {total_acc*100:>9.2f}%")

def print_direction_table(results_list):
    print(f"\n{'─'*55}")
    print("DIRECTION SPLIT (tradeable selected picks)")
    print(f"{'─'*55}")
    print(f"  {'Version':<10} {'MORE N':>8} {'MORE%':>8} {'LESS N':>8} {'LESS%':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_list:
        d = direction_summary(r)
        print(f"  {r['label']:<10} {d['MORE']['sel']:>8,} {d['MORE']['acc']*100:>7.1f}% {d['LESS']['sel']:>8,} {d['LESS']['acc']*100:>7.1f}%")

def print_delta_props(r037, r038):
    """Show props that changed materially between versions."""
    print(f"\n{'─'*75}")
    print("PROPS THAT CHANGED IN v038 (vs v037)")
    print(f"{'─'*75}")
    map037 = {r["key"]: r for r in r037["props"]}
    map038 = {r["key"]: r for r in r038["props"]}
    changed_keys = set()
    for k in map037:
        if map037[k]["floor"] != map038.get(k, {}).get("floor", map037[k]["floor"]):
            changed_keys.add(k)
    for k in map038:
        if k not in map037:
            changed_keys.add(k)

    print(f"  {'Prop+Dir':<38} {'v037 floor':>10} {'v038 floor':>10} {'v037 sel':>9} {'v038 sel':>9} {'v037 acc':>9} {'v038 acc':>9}")
    print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for k in sorted(changed_keys):
        r7 = map037.get(k, {})
        r8 = map038.get(k, {})
        f7 = r7.get("floor", "n/a")
        f8 = r8.get("floor", "n/a")
        s7 = r7.get("sel_n", 0)
        s8 = r8.get("sel_n", 0)
        a7 = f"{r7['sel_acc']*100:.1f}%" if r7.get("sel_acc") else "n/a"
        a8 = f"{r8['sel_acc']*100:.1f}%" if r8.get("sel_acc") else "n/a"
        print(f"  {k:<38} {f7!r:>10} {f8!r:>10} {s7:>9,} {s8:>9,} {a7:>9} {a8:>9}")

# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading backtest data...")
with open(BACKTEST_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data):,} picks")

print("\nScoring v037 (baseline)...")
r037 = apply_floors(data, FLOORS_V037, "v037")

print("Scoring v038...")
r038 = apply_floors(data, FLOORS_V038, "v038")

# Summary comparison
print_summary_table([r037, r038])

# Direction split
print_direction_table([r037, r038])

# Grade breakdowns
grades_v037 = grade_breakdown(data, FLOORS_V037, "v037")
grades_v038 = grade_breakdown(data, FLOORS_V038, "v038")
print_grade_breakdown(grades_v037, "v037 (baseline)")
print_grade_breakdown(grades_v038, "v038")

# Per-prop breakdown for v038
print_prop_breakdown(r038)

# Delta view — what changed?
print_delta_props(r037, r038)

# Best
print(f"\n{'='*55}")
best = max([r037, r038], key=lambda r: r["accuracy"])
print(f"BEST: {best['label']}  {best['accuracy']*100:.2f}%  ({best['selected']:,} picks)")
delta_picks = r038["selected"] - r037["selected"]
delta_acc   = (r038["accuracy"] - r037["accuracy"]) * 100
print(f"v038 vs v037: {delta_picks:+,} picks  {delta_acc:+.2f}pp accuracy")
print(f"{'='*55}")
