#!/usr/bin/env python3
"""
Re-process existing backtest records by applying new offsets to stored
projections, recalculating probabilities, and reporting improvements.
"""

import json
import sys
sys.path.insert(0, '.')

from src.predictor import calculate_over_under_probability

def reprocess():
    # Load weights manually to avoid caching issues
    with open('data/weights/current.json') as f:
        weights = json.load(f)

    offsets = weights.get('prop_type_offsets', {})

    print("=" * 80)
    print("REPROCESSING BACKTEST WITH NEW OFFSETS")
    print("=" * 80)
    print(f"Offsets: {offsets}")

    # Load backtest
    print("\nLoading backtest...")
    with open('data/backtest/backtest_2025.json') as f:
        records = json.load(f)

    print(f"Loaded {len(records):,} records")

    before_stats = {}
    after_stats = {}

    # Process in batches to manage memory
    for i, rec in enumerate(records):
        if i % 5000 == 0:
            print(f"  Processing {i:,} / {len(records):,}")

        prop = rec['prop_type']
        if prop not in before_stats:
            before_stats[prop] = {'W': 0, 'L': 0, 'push': 0}
            after_stats[prop] = {'W': 0, 'L': 0, 'push': 0}

        # Original projection
        orig_proj = rec['projection']

        # Apply new offset
        offset = offsets.get(prop, 0.0)
        new_proj = orig_proj + offset

        # Recalculate probability with new projection
        try:
            prob = calculate_over_under_probability(new_proj, rec['line'], prop)
            new_pick = prob['pick']
            new_rating = prob['rating']
            new_conf = prob['confidence']
        except Exception as e:
            print(f"Error on record {i}: {e}")
            continue

        # Grade
        actual = rec['actual']
        line = rec['line']
        if actual > line:
            result = 'W' if new_pick == 'MORE' else 'L'
        elif actual < line:
            result = 'W' if new_pick == 'LESS' else 'L'
        else:
            result = 'push'

        # Count
        before_result = rec.get('result', 'L')
        before_stats[prop][before_result if before_result in ['W', 'L', 'push'] else 'L'] += 1
        after_stats[prop][result] += 1

        # Update record in-place to minimize memory
        rec['projection'] = round(new_proj, 3)
        rec['pick'] = new_pick
        rec['rating'] = new_rating
        rec['confidence'] = round(new_conf, 4)
        rec['result'] = result

    # Save
    print("\nSaving reprocessed backtest...")
    with open('data/backtest/backtest_2025_v5.json', 'w') as f:
        json.dump(records, f, default=str)

    # Report
    print(f"Saved to: data/backtest/backtest_2025_v5.json\n")

    print("=" * 80)
    print("IMPROVEMENT SUMMARY BY PROP TYPE")
    print("=" * 80)

    for prop in sorted(before_stats.keys()):
        before = before_stats[prop]
        after = after_stats[prop]

        before_total = before['W'] + before['L']
        after_total = after['W'] + after['L']

        before_acc = before['W'] / before_total if before_total > 0 else 0
        after_acc = after['W'] / after_total if after_total > 0 else 0

        delta = (after_acc - before_acc) * 100

        print(f"\n{prop}")
        print(f"  Before: {before['W']:,}W - {before['L']:,}L  ({before_acc:.1%})")
        print(f"  After:  {after['W']:,}W - {after['L']:,}L  ({after_acc:.1%})")
        print(f"  Delta:  {delta:+.1f}% pts")

        offset = offsets.get(prop, 0.0)
        print(f"  Offset applied: {offset:+.2f}")

if __name__ == "__main__":
    reprocess()
