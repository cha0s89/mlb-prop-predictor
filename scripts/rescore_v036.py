"""
Rescore script for v036 cycle 3 optimization.
Compares tradeable-only accuracy BEFORE (v032 floors) and AFTER (v036 floors)
using backtest_2025_v032_old.json as the data source.

Usage:
    py scripts/rescore_v036.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

BACKTEST_FILE = "C:/Users/Unknown/Downloads/mlb-prop-predictor/data/backtest/backtest_2025_v032_old.json"

# Tradeable prop types (matching the v034 report)
TRADEABLE_PROPS = {
    "batter_strikeouts",
    "earned_runs",
    "hits",
    "hits_runs_rbis",
    "hitter_fantasy_score",
    "pitcher_strikeouts",
    "pitching_outs",
    "rbis",
    "runs",
    "total_bases",
    "walks_allowed",
}

# v032/v034 baseline floors (before this cycle)
FLOORS_BEFORE: dict[str, float] = {
    "batter_strikeouts_less": 0.66,
    "batter_strikeouts_more": 0.66,
    "earned_runs_less": 0.60,
    "earned_runs_more": 0.55,
    "hits_less": 0.72,
    "hits_more": 0.72,
    "hits_runs_rbis_less": 0.95,
    "hits_runs_rbis_more": 0.55,
    "hitter_fantasy_score_less": 0.68,
    "hitter_fantasy_score_more": 0.65,
    "pitcher_strikeouts_less": 0.66,
    "pitcher_strikeouts_more": 0.66,
    "pitching_outs_less": 0.58,
    "pitching_outs_more": 0.60,
    "rbis_less": 0.63,
    "rbis_more": 0.60,
    "runs_less": 0.68,
    "runs_more": 0.68,
    "total_bases_less": 0.72,
    "total_bases_more": 0.64,
    "walks_allowed_less": 0.95,
    "walks_allowed_more": 0.95,
}

# v036 floors (after cycle 3)
FLOORS_AFTER: dict[str, float] = {
    "batter_strikeouts_less": 0.65,
    "batter_strikeouts_more": 0.66,
    "earned_runs_less": 0.97,   # DISABLED
    "earned_runs_more": 0.55,
    "hits_less": 0.55,          # OPENED
    "hits_more": 0.72,
    "hits_runs_rbis_less": 0.65,  # OPENED from 0.95
    "hits_runs_rbis_more": 0.97,  # DISABLED
    "hitter_fantasy_score_less": 0.58,  # OPENED
    "hitter_fantasy_score_more": 0.65,
    "pitcher_strikeouts_less": 0.62,    # TIGHTENED
    "pitcher_strikeouts_more": 0.62,    # TIGHTENED
    "pitching_outs_less": 0.58,
    "pitching_outs_more": 0.60,
    "rbis_less": 0.58,          # OPENED
    "rbis_more": 0.97,          # DISABLED
    "runs_less": 0.68,
    "runs_more": 0.68,
    "total_bases_less": 0.58,   # OPENED
    "total_bases_more": 0.97,   # DISABLED
    "walks_allowed_less": 0.62,  # OPENED from 0.95
    "walks_allowed_more": 0.97,  # DISABLED
}


def score_dataset(rows: list[dict], floors: dict[str, float]) -> dict:
    """Apply floors, count selected picks and accuracy."""
    combo_stats: dict[str, dict] = {}

    for row in rows:
        prop = row.get("prop_type", "")
        if prop not in TRADEABLE_PROPS:
            continue
        direction = row.get("pick", "").upper()
        conf = row.get("confidence", 0.0) or 0.0
        result = row.get("result", "")

        key = f"{prop}_{direction}"
        floor = floors.get(key.lower(), 0.0)

        if key.lower() not in combo_stats:
            combo_stats[key.lower()] = {
                "all_n": 0, "all_wins": 0,
                "sel_n": 0, "sel_wins": 0,
                "floor": floor,
            }

        s = combo_stats[key.lower()]
        s["all_n"] += 1
        if result == "W":
            s["all_wins"] += 1

        if conf >= floor:
            s["sel_n"] += 1
            if result == "W":
                s["sel_wins"] += 1

    return combo_stats


def pct(wins: int, n: int) -> str:
    return f"{100 * wins / n:.1f}%" if n else "n/a"


def summarize(label: str, floors: dict[str, float], stats: dict) -> tuple[int, int]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    header = f"{'Combo':<35} {'Floor':>6} {'All N':>8} {'All Acc':>8} {'Sel N':>8} {'Sel Acc':>8}"
    print(header)
    print("-" * 75)

    total_all_n = total_all_wins = 0
    total_sel_n = total_sel_wins = 0

    for key in sorted(stats):
        s = stats[key]
        floor = floors.get(key, 0.0)
        all_acc = pct(s["all_wins"], s["all_n"])
        sel_acc = pct(s["sel_wins"], s["sel_n"])
        print(
            f"{key:<35} {floor:>6.2f} {s['all_n']:>8,} {all_acc:>8} {s['sel_n']:>8,} {sel_acc:>8}"
        )
        total_all_n += s["all_n"]
        total_all_wins += s["all_wins"]
        total_sel_n += s["sel_n"]
        total_sel_wins += s["sel_wins"]

    print("-" * 75)
    print(
        f"{'TOTAL (tradeable)':<35} {'':>6} {total_all_n:>8,} {pct(total_all_wins, total_all_n):>8} "
        f"{total_sel_n:>8,} {pct(total_sel_wins, total_sel_n):>8}"
    )
    return total_sel_n, total_sel_wins


def main() -> None:
    print(f"Loading {BACKTEST_FILE} ...")
    with open(BACKTEST_FILE, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Loaded {len(rows):,} rows.")

    stats_before = score_dataset(rows, FLOORS_BEFORE)
    stats_after = score_dataset(rows, FLOORS_AFTER)

    sel_n_before, sel_wins_before = summarize("BEFORE — v034 baseline floors", FLOORS_BEFORE, stats_before)
    sel_n_after, sel_wins_after = summarize("AFTER  — v036 cycle 3 floors", FLOORS_AFTER, stats_after)

    before_acc = 100 * sel_wins_before / sel_n_before if sel_n_before else 0
    after_acc = 100 * sel_wins_after / sel_n_after if sel_n_after else 0
    delta_acc = after_acc - before_acc
    delta_vol = sel_n_after - sel_n_before

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  BEFORE: {sel_n_before:,} picks @ {before_acc:.2f}%")
    print(f"  AFTER:  {sel_n_after:,} picks @ {after_acc:.2f}%")
    print(f"  Delta accuracy : {delta_acc:+.2f}pp")
    print(f"  Delta volume   : {delta_vol:+,} picks")
    print("=" * 70)


if __name__ == "__main__":
    main()
