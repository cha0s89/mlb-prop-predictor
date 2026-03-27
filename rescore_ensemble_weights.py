#!/usr/bin/env python3
"""
Optimization Cycle 4: Ensemble Weight Sensitivity Testing
==========================================================

Methodology
-----------
1. Load backtest_2025_v032_old.json (464K predictions).
2. Re-score each entry via calculate_over_under_probability() using current
   weights (current.json) — same as rescore_v033.py.  This gives the
   projection-only confidence score (call it proj_conf).

3. Model the combined.py PROJECTION_ONLY path for each ensemble config:
       w_proj_norm = proj_weight / (proj_weight + sharp_weight)
       eff_conf    = 0.5 + (w_proj_norm + 0.25) * (proj_conf - 0.5)
   Sharp-odds and recent-form are absent from the backtest, so the
   projection edge is the only signal — matching the projection-only
   code path in combined.py (line 113).

4. Apply v036 confidence floors to eff_conf and measure accuracy.

The v036 reference baseline (119,494 picks / 72.73%) was measured by
applying floors to proj_conf directly (equivalent to scale_factor = 1.0,
i.e. cfg3 60/20/20).

Usage:
    python rescore_ensemble_weights.py [--dry-run N]
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PROJECT = "C:/Users/Unknown/Downloads/mlb-prop-predictor"
SRC_DIR = os.path.join(MAIN_PROJECT, "src")
sys.path.insert(0, SRC_DIR)
os.chdir(MAIN_PROJECT)  # ensures data/ paths resolve correctly

from predictor import calculate_over_under_probability, _load_weights  # noqa: E402

BACKTEST_PATH = os.path.join(MAIN_PROJECT, "data", "backtest", "backtest_2025_v032_old.json")

# ── Tradeable prop definitions ─────────────────────────────────────────────────
TRADEABLE_PROPS = {
    "pitcher_strikeouts", "pitching_outs", "hits", "total_bases",
    "hitter_fantasy_score", "earned_runs", "runs", "rbis",
    "batter_strikeouts", "walks_allowed", "hits_runs_rbis",
    "home_runs", "doubles", "triples",
}
TRADEABLE_MORE_ONLY = {"home_runs", "doubles", "triples"}
BLOCKED_PROPS = {"singles", "walks", "hits_allowed"}


def is_tradeable(pt, pick):
    if pt in BLOCKED_PROPS:
        return False
    if pt in TRADEABLE_MORE_ONLY:
        return pick == "MORE"
    return pt in TRADEABLE_PROPS


# ── v036 confidence floors ─────────────────────────────────────────────────────
FLOORS_V036 = {
    "hits_more": 0.72,        "hits_less": 0.72,
    "total_bases_more": 0.95, "total_bases_less": 0.72,
    "pitcher_strikeouts_more": 0.66, "pitcher_strikeouts_less": 0.66,
    "hitter_fantasy_score_more": 0.65, "hitter_fantasy_score_less": 0.68,
    "pitching_outs_more": 0.60, "pitching_outs_less": 0.60,
    "earned_runs_more": 0.60, "earned_runs_less": 0.95,
    "walks_allowed_more": 0.95, "walks_allowed_less": 0.57,
    "hits_allowed_more": 0.60, "hits_allowed_less": 0.60,
    "batter_strikeouts_more": 0.66, "batter_strikeouts_less": 0.62,
    "walks_more": 0.60, "walks_less": 0.70,
    "rbis_more": 0.95, "rbis_less": 0.63,
    "runs_more": 0.68, "runs_less": 0.68,
    "singles_more": 0.95, "singles_less": 0.95,
    "doubles_more": 0.60, "doubles_less": 0.60,
    "hits_runs_rbis_more": 0.68, "hits_runs_rbis_less": 0.74,
    "home_runs_more": 0.60, "home_runs_less": 0.60,
}

# Grade thresholds applied to eff_confidence
CONF_GRADE_THRESHOLDS = {"A": 0.70, "B": 0.62, "C": 0.57, "D": 0.0}


def assign_grade(conf):
    if conf >= 0.70: return "A"
    if conf >= 0.62: return "B"
    if conf >= 0.57: return "C"
    return "D"


# ── Ensemble configs ───────────────────────────────────────────────────────────
# (label, w_proj, w_sharp, w_recent)
# scale_factor = w_proj/(w_proj+w_sharp) + 0.25  (combined.py projection-only path)
CONFIGS = [
    ("v036_ref_scale1.0",  0.60, 0.00, 0.40),  # w_proj/(w_proj+w_sharp)=1.0 → scale=1.25? No:
    # v036 reference: floors applied directly to proj_conf (scale=1.0)
    # This is modeled as w_proj=0.75, w_sharp=0.00 → norm=1.0 → scale=1.25 ... no.
    # The reference is simply conf >= floor, which means eff_conf = proj_conf (scale factor implicitly 1.0).
    # We model it with w_proj+0.25=1.0 → w_proj=0.75 norm'd
    # Actually: to get scale=1.0, we need w_proj_norm = 0.75 → w_proj/(w_proj+w_sharp)=0.75
    # e.g. w_proj=0.60, w_sharp=0.20 → norm=0.75 → scale=1.0 ✓  (matches cfg3!)
    # See note below about how to get the reference.
]

# The reference v036 (119,494 picks / 72.73%) uses floors directly on proj_conf.
# In combined.py terms, this is scale=1.0.
# scale = w_proj_norm + 0.25 = 1.0  =>  w_proj_norm = 0.75
# This happens when w_proj/(w_proj+w_sharp) = 0.75  e.g. proj=0.60, sharp=0.20

CONFIGS_FINAL = [
    # Reference: same as applying floors to raw proj_conf (scale=1.0)
    ("v036_reference",    0.60, 0.20, 0.20),   # scale=1.0  [matches user's v036 baseline]
    # User-stated current weights
    ("baseline_60/25/15", 0.60, 0.25, 0.15),   # scale=0.956
    # Test candidates (proj/sharp/recent)
    ("cfg1_65/20/15",    0.65, 0.20, 0.15),    # scale=1.015
    ("cfg2_55/30/15",    0.55, 0.30, 0.15),    # scale=0.897
    ("cfg3_60/20/20",    0.60, 0.20, 0.20),    # scale=1.000  [same as reference]
    ("cfg4_70/15/15",    0.70, 0.15, 0.15),    # scale=1.074
]


def get_scale(w_proj, w_sharp):
    """Combined.py PROJECTION_ONLY path: scale = w_proj_norm + 0.25."""
    total = w_proj + w_sharp
    if total <= 0:
        return 1.0
    return round(w_proj / total + 0.25, 4)


# ── Rescore one entry ──────────────────────────────────────────────────────────
def rescore_entry(row, weights):
    proj     = row.get("projection", 0.0)
    line     = row.get("line", 0.5)
    pt       = row.get("prop_type", "")
    try:
        calc     = calculate_over_under_probability(proj, line, pt,
                                                    weights_override=weights)
        conf     = calc.get("confidence", 0.5)
        pick     = calc.get("pick", row.get("pick", "LESS"))
    except Exception:
        conf     = float(row.get("confidence", 0.5))
        pick     = row.get("pick", "LESS")
    return conf, pick


# ── Apply one ensemble config ──────────────────────────────────────────────────
def apply_config(proj_data, label, w_proj, w_sharp, w_recent):
    """proj_data: list of (proj_conf, pick, result) tuples."""
    sf = get_scale(w_proj, w_sharp)
    by_grade = defaultdict(lambda: {"sel": 0, "wins": 0})
    by_prop  = defaultdict(lambda: {"sel": 0, "wins": 0, "all": 0, "all_wins": 0})
    selected = []

    for pt, pick, proj_conf, is_win in proj_data:
        if not is_tradeable(pt, pick):
            continue
        raw_edge = proj_conf - 0.5
        eff_conf = 0.5 + sf * raw_edge
        fk       = f"{pt}_{pick.lower()}"
        floor    = FLOORS_V036.get(fk, 0.60)

        by_prop[fk]["all"]      += 1
        by_prop[fk]["all_wins"] += is_win

        if eff_conf >= floor:
            grade = assign_grade(eff_conf)
            by_grade[grade]["sel"]  += 1
            by_grade[grade]["wins"] += is_win
            by_prop[fk]["sel"]  += 1
            by_prop[fk]["wins"] += is_win
            selected.append(is_win)

    n    = len(selected)
    wins = sum(selected)
    acc  = wins / n if n else 0.0

    grade_rows = {}
    for g in ["A", "B", "C", "D"]:
        d = by_grade[g]
        g_acc = d["wins"] / d["sel"] if d["sel"] else None
        grade_rows[g] = {"picks": d["sel"], "wins": d["wins"], "acc": g_acc}

    prop_rows = sorted(
        [{"key": k, "sel": v["sel"],
          "sel_acc": v["wins"] / v["sel"] if v["sel"] else 0,
          "all": v["all"],
          "all_acc": v["all_wins"] / v["all"] if v["all"] else 0}
         for k, v in by_prop.items() if v["sel"] > 0],
        key=lambda x: -x["sel"]
    )

    return {
        "label": label, "sf": sf, "w_proj": w_proj, "w_sharp": w_sharp,
        "selected": n, "wins": wins, "accuracy": acc,
        "grades": grade_rows, "props": prop_rows,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=0, metavar="N")
    args = parser.parse_args()

    print(f"[{datetime.now():%H:%M:%S}] Loading weights...")
    weights = _load_weights()
    print(f"[{datetime.now():%H:%M:%S}] Weights version: {weights.get('version','?')}")

    print(f"[{datetime.now():%H:%M:%S}] Loading backtest...")
    with open(BACKTEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if args.dry_run:
        data = data[:args.dry_run]
        print(f"[{datetime.now():%H:%M:%S}] DRY RUN: {args.dry_run:,} records")
    else:
        print(f"[{datetime.now():%H:%M:%S}] Loaded {len(data):,} records")

    # ── Phase 1: Re-score all records with calculate_over_under_probability ──
    print(f"\n[{datetime.now():%H:%M:%S}] Re-scoring via calculate_over_under_probability...")
    print("  (This is the same as rescore_v033.py — no API calls, uses stored projections)")

    proj_data = []   # list of (pt, pick, proj_conf, is_win)
    errors    = 0
    t0        = time.time()
    REPORT_EVERY = 50_000

    for i, row in enumerate(data):
        if i > 0 and i % REPORT_EVERY == 0:
            elapsed  = time.time() - t0
            rate     = i / elapsed
            rem      = (len(data) - i) / rate
            print(f"  [{datetime.now():%H:%M:%S}] {i:,}/{len(data):,} | "
                  f"{rate:,.0f}/s | ~{rem:.0f}s left | errors: {errors}")

        pt     = row.get("prop_type", "")
        result = row.get("result", "")
        is_win = 1 if result == "W" else 0

        try:
            conf, pick = rescore_entry(row, weights)
        except Exception as e:
            conf = float(row.get("confidence", 0.5))
            pick = row.get("pick", "LESS")
            errors += 1

        proj_data.append((pt, pick, conf, is_win))

    elapsed = time.time() - t0
    print(f"[{datetime.now():%H:%M:%S}] Re-score done in {elapsed:.1f}s | errors: {errors}")

    # ── Phase 2: Apply each ensemble config ──────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Applying ensemble weight configs...")
    results = []
    for label, wp, ws, wr in CONFIGS_FINAL:
        r = apply_config(proj_data, label, wp, ws, wr)
        results.append(r)
        print(f"  {label:<22}  scale={r['sf']:.3f}  "
              f"picks={r['selected']:>10,}  acc={r['accuracy']*100:.2f}%")

    # ── Report ────────────────────────────────────────────────────────────────
    V036_BASELINE_ACC   = 0.7273
    V036_BASELINE_PICKS = 119_494

    print("\n" + "=" * 84)
    print("ENSEMBLE WEIGHT SENSITIVITY — tradeable props only, v036 floors")
    print("Model: eff_conf = 0.5 + (w_proj_norm+0.25) * proj_edge  (projection-only path)")
    print("=" * 84)
    print(f"  {'Config':<22}  {'Scale':>6}  {'Picks':>10}  {'Accuracy':>9}  "
          f"{'vs v036':>9}  {'dPicks':>8}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

    ref_acc   = None
    ref_picks = None
    best      = None

    for r in results:
        if ref_acc is None and r["sf"] == 1.0:
            ref_acc   = r["accuracy"]
            ref_picks = r["selected"]

    ref_acc   = ref_acc   or V036_BASELINE_ACC
    ref_picks = ref_picks or V036_BASELINE_PICKS

    for r in results:
        da  = (r["accuracy"] - V036_BASELINE_ACC) * 100
        dn  = r["selected"] - V036_BASELINE_PICKS
        tag = ""
        is_ref = (r["sf"] == 1.0)
        if is_ref:
            tag = "  [ref ~ v036]"
        elif best is None or r["accuracy"] > best["accuracy"]:
            best = r
            tag = "  <-- BEST"
        print(f"  {r['label']:<22}  {r['sf']:>6.3f}  {r['selected']:>10,}  "
              f"{r['accuracy']*100:>8.2f}%  {da:>+8.2f}pp  {dn:>+8,}{tag}")

    print(f"\n  Reported v036 baseline:  {V036_BASELINE_PICKS:>10,}    72.73%  (reference)")

    # Per-grade table
    print("\n" + "=" * 84)
    print("PER-GRADE BREAKDOWN  (A: >=70% | B: >=62% | C: >=57% | D: below)")
    print("=" * 84)
    print(f"  {'Config':<22}  {'Grade A':>16}  {'Grade B':>16}  {'Grade C':>16}  {'Grade D':>16}")
    print(f"  {'-'*22}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*16}")
    for r in results:
        row = f"  {r['label']:<22}"
        for g in ["A", "B", "C", "D"]:
            d = r["grades"][g]
            if d["picks"] > 0 and d["acc"] is not None:
                row += f"  {d['picks']:>6,} ({d['acc']*100:.1f}%)"
            else:
                row += f"  {'---':>16}"
        print(row)

    # Per-prop for best
    if best:
        print(f"\n{'='*84}")
        print(f"TOP PROPS — {best['label']} (scale={best['sf']:.3f}, "
              f"{best['selected']:,} picks, {best['accuracy']*100:.2f}%)")
        print(f"{'='*84}")
        print(f"  {'Prop+Dir':<40}  {'Sel N':>8}  {'Sel Acc':>9}  {'All Acc':>9}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*9}  {'-'*9}")
        for row in best["props"][:25]:
            sa = f"{row['sel_acc']*100:.1f}%"
            aa = f"{row['all_acc']*100:.1f}%"
            print(f"  {row['key']:<40}  {row['sel']:>8,}  {sa:>9}  {aa:>9}")

    # Recommendation
    print(f"\n{'='*84}")
    print("RECOMMENDATION")
    print(f"{'='*84}")
    if best and best["accuracy"] > V036_BASELINE_ACC:
        print(f"  WINNER: {best['label']}  ({best['accuracy']*100:.2f}%  "
              f"{(best['accuracy']-V036_BASELINE_ACC)*100:+.2f}pp vs v036)")
        print(f"  Picks:  {best['selected']:,}  ({best['selected']-V036_BASELINE_PICKS:+,} vs v036)")
        print(f"  --> Recommend updating ensemble_weights in current.json")
    else:
        print("  No config beats v036 72.73% — ensemble weight sensitivity is LOW")
        print("  The projection-only signal dominates backtest outcomes.")
        print("  Ensemble weights matter most when sharp odds data is available (live system).")
    print()


if __name__ == "__main__":
    main()
