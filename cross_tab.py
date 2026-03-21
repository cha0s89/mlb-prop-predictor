"""
Backtest Cross-Tab Analysis

Reads data/backtest/backtest_2025.json and prints:
  1. Prop type × direction cross-tab (accuracy for each combo)
  2. Mean projection vs mean actual vs mean line per prop type
  3. Line distribution per prop type (most common lines)
  4. Top 30 prop × line × direction combos by volume with accuracy

Run: python cross_tab.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

BACKTEST_PATH = Path("data/backtest/backtest_2025.json")


def load_backtest():
    if not BACKTEST_PATH.exists():
        print(f"ERROR: {BACKTEST_PATH} not found")
        sys.exit(1)
    with open(BACKTEST_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} backtest records")

    # Filter non-plays (actual=0) which artificially inflate LESS and deflate MORE.
    # Exception: home_runs — actual=0 means the batter played but didn't homer, which is
    # a valid LESS result, not a non-play. Pitcher props also use 0 as a real value.
    BATTER_PROPS_SKIP_ZERO = {"hits", "total_bases", "hitter_fantasy_score", "stolen_bases",
                               "hits_runs_rbis", "runs", "rbis", "batter_strikeouts"}

    def is_nonplay(r):
        prop = r.get("prop_type", "")
        return r.get("actual", 0) == 0 and prop in BATTER_PROPS_SKIP_ZERO

    nonplays = [r for r in data if is_nonplay(r)]
    plays = [r for r in data if not is_nonplay(r)]
    print(f"  Non-plays (actual=0, batter props only): {len(nonplays):,} ({100*len(nonplays)/len(data):.1f}%)")
    print(f"  Actual plays: {len(plays):,} ({100*len(plays)/len(data):.1f}%)\n")

    return plays


def cross_tab(data):
    """Prop type × direction accuracy cross-tab."""
    print("=" * 70)
    print("1. PROP TYPE × DIRECTION CROSS-TAB")
    print("=" * 70)

    buckets = {}
    for row in data:
        prop = row.get("prop_type", "unknown")
        pick = row.get("pick", "")
        result = row.get("result", "")
        if result not in ("W", "L"):
            continue
        key = (prop, pick)
        if key not in buckets:
            buckets[key] = {"W": 0, "L": 0}
        buckets[key][result] += 1

    props = sorted(set(k[0] for k in buckets))
    directions = ["MORE", "LESS"]

    print(f"\n{'Prop Type':<25} {'MORE W-L':>12} {'MORE Acc':>10} {'LESS W-L':>12} {'LESS Acc':>10} {'Total Acc':>10}")
    print("-" * 82)

    for prop in props:
        parts = []
        total_w, total_l = 0, 0
        for d in directions:
            b = buckets.get((prop, d), {"W": 0, "L": 0})
            w, l = b["W"], b["L"]
            total_w += w
            total_l += l
            total = w + l
            acc = f"{w / total:.1%}" if total > 0 else "N/A"
            parts.append((f"{w}-{l}", acc))
        total = total_w + total_l
        total_acc = f"{total_w / total:.1%}" if total > 0 else "N/A"
        print(f"{prop:<25} {parts[0][0]:>12} {parts[0][1]:>10} {parts[1][0]:>12} {parts[1][1]:>10} {total_acc:>10}")

    print()


def bias_table(data):
    """Mean projection vs mean actual vs mean line per prop type."""
    print("=" * 70)
    print("2. PROJECTION BIAS BY PROP TYPE")
    print("=" * 70)

    buckets = {}
    for row in data:
        prop = row.get("prop_type", "unknown")
        proj = row.get("projection")
        actual = row.get("actual")
        line = row.get("line")
        if proj is None or actual is None or line is None:
            continue
        if prop not in buckets:
            buckets[prop] = {"proj": [], "actual": [], "line": []}
        buckets[prop]["proj"].append(proj)
        buckets[prop]["actual"].append(actual)
        buckets[prop]["line"].append(line)

    print(f"\n{'Prop Type':<25} {'Count':>7} {'Mean Proj':>10} {'Mean Actual':>12} {'Mean Line':>10} {'Proj Bias':>10}")
    print("-" * 77)

    for prop in sorted(buckets):
        b = buckets[prop]
        n = len(b["proj"])
        mean_proj = sum(b["proj"]) / n
        mean_actual = sum(b["actual"]) / n
        mean_line = sum(b["line"]) / n
        bias = mean_proj - mean_actual
        print(f"{prop:<25} {n:>7,} {mean_proj:>10.2f} {mean_actual:>12.2f} {mean_line:>10.2f} {bias:>+10.2f}")

    print()


def line_distribution(data):
    """Most common lines per prop type."""
    print("=" * 70)
    print("3. LINE DISTRIBUTION BY PROP TYPE (top 5 lines each)")
    print("=" * 70)

    buckets = {}
    for row in data:
        prop = row.get("prop_type", "unknown")
        line = row.get("line")
        if line is None:
            continue
        if prop not in buckets:
            buckets[prop] = []
        buckets[prop].append(line)

    for prop in sorted(buckets):
        counts = Counter(buckets[prop])
        top = counts.most_common(5)
        total = sum(counts.values())
        line_str = ", ".join(f"{line}({cnt}, {cnt/total:.0%})" for line, cnt in top)
        print(f"\n  {prop} (n={total:,})")
        print(f"    {line_str}")

    print()


def top_combos(data):
    """Top 30 prop × line × direction combos by volume with accuracy."""
    print("=" * 70)
    print("4. TOP 30 COMBOS BY VOLUME (prop × line × direction)")
    print("=" * 70)

    buckets = {}
    for row in data:
        prop = row.get("prop_type", "unknown")
        line = row.get("line")
        pick = row.get("pick", "")
        result = row.get("result", "")
        if line is None or result not in ("W", "L"):
            continue
        key = (prop, line, pick)
        if key not in buckets:
            buckets[key] = {"W": 0, "L": 0}
        buckets[key][result] += 1

    ranked = []
    for (prop, line, pick), wl in buckets.items():
        total = wl["W"] + wl["L"]
        acc = wl["W"] / total if total > 0 else 0
        ranked.append((prop, line, pick, total, wl["W"], wl["L"], acc))

    ranked.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'Prop Type':<25} {'Line':>6} {'Pick':>6} {'Total':>7} {'W-L':>8} {'Accuracy':>9}")
    print("-" * 65)

    for prop, line, pick, total, w, l, acc in ranked[:30]:
        print(f"{prop:<25} {line:>6.1f} {pick:>6} {total:>7,} {w}-{l:>5} {acc:>9.1%}")

    print()


def main():
    data = load_backtest()
    cross_tab(data)
    bias_table(data)
    line_distribution(data)
    top_combos(data)


if __name__ == "__main__":
    main()
