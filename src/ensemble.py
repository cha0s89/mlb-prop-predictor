"""
Hedge-Style Ensemble Weight Updater

Dynamically adjusts the blend weights between signal sources (sharp_odds,
projection, recent_form) based on recent performance. Uses multiplicative
weight updates inspired by the Hedge/Multiplicative Weights algorithm.

The idea: after each day's results, signal sources that were more accurate
get their weights increased, and less accurate sources get decreased.
This lets the model automatically adapt to which signals are working.

Usage:
    from src.ensemble import update_ensemble_weights, get_current_ensemble_weights
"""

import json
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional

from src.database import get_connection


def init_ensemble_history_table():
    """Create the ensemble_history table for tracking weight evolution."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ensemble_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            sharp_weight REAL NOT NULL,
            projection_weight REAL NOT NULL,
            recent_form_weight REAL NOT NULL,
            sharp_accuracy REAL,
            projection_accuracy REAL,
            recent_form_accuracy REAL,
            num_graded INTEGER,
            learning_rate REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ensemble_date
            ON ensemble_history(date);

        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            sharp_correct INTEGER,
            projection_correct INTEGER,
            outcome INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_signal_date
            ON signal_outcomes(date);
    """)
    conn.commit()
    conn.close()


def record_signal_outcome(player_name: str, stat_type: str,
                          signal_type: str, sharp_correct: bool = None,
                          projection_correct: bool = None,
                          outcome: int = None,
                          game_date: str = None):
    """Record whether each signal source was correct for a pick.

    Args:
        player_name: Player name
        stat_type: Stat type
        signal_type: CONFIRMED, SHARP_ONLY, or PROJECTION_ONLY
        sharp_correct: Did the sharp signal predict correctly?
        projection_correct: Did the projection predict correctly?
        outcome: 1 = win, 0 = loss
        game_date: Date (default today)
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    conn.execute("""
        INSERT INTO signal_outcomes
        (date, player_name, stat_type, signal_type, sharp_correct,
         projection_correct, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        game_date, player_name, stat_type, signal_type,
        1 if sharp_correct else (0 if sharp_correct is not None else None),
        1 if projection_correct else (0 if projection_correct is not None else None),
        outcome,
    ))
    conn.commit()
    conn.close()


def compute_source_accuracy(days: int = 14) -> Dict:
    """Compute accuracy of each signal source over the last N days.

    Returns:
        Dict with sharp_accuracy, projection_accuracy, and sample counts
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT sharp_correct, projection_correct, outcome
        FROM signal_outcomes
        WHERE date >= date('now', ? || ' days')
        AND outcome IS NOT NULL
    """, conn, params=(f"-{days}",))
    conn.close()

    result = {
        "sharp_accuracy": 0.5,
        "projection_accuracy": 0.5,
        "sharp_n": 0,
        "projection_n": 0,
        "total_n": len(df),
    }

    if df.empty:
        return result

    # Sharp accuracy (where we had sharp signal)
    sharp_df = df[df["sharp_correct"].notna()]
    if len(sharp_df) > 0:
        result["sharp_accuracy"] = float(sharp_df["sharp_correct"].mean())
        result["sharp_n"] = len(sharp_df)

    # Projection accuracy (where we had projection signal)
    proj_df = df[df["projection_correct"].notna()]
    if len(proj_df) > 0:
        result["projection_accuracy"] = float(proj_df["projection_correct"].mean())
        result["projection_n"] = len(proj_df)

    return result


def update_ensemble_weights(learning_rate: float = 0.1,
                            min_samples: int = 20,
                            lookback_days: int = 14) -> Dict:
    """Run the hedge-style weight update.

    Uses Multiplicative Weights Update:
        w_i(t+1) = w_i(t) * (1 + eta * (accuracy_i - 0.5))
    Then renormalize so weights sum to 1.

    The intuition: sources that beat 50% get upweighted, those below get
    downweighted. The learning rate controls how fast adaptation happens.

    Args:
        learning_rate: eta — how fast to adapt (0.05 to 0.20 recommended)
        min_samples: Minimum graded picks before updating
        lookback_days: Days of data to consider

    Returns:
        Dict with old_weights, new_weights, accuracies, and whether update was applied
    """
    # Load current weights
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "weights", "current.json"
    )
    with open(weights_path) as f:
        weights_data = json.load(f)

    current = weights_data.get("ensemble_weights", {
        "sharp_odds": 0.65,
        "projection": 0.30,
        "recent_form": 0.05,
    })

    old_weights = dict(current)

    # Compute recent accuracy
    accuracy = compute_source_accuracy(days=lookback_days)

    result = {
        "old_weights": old_weights,
        "accuracies": accuracy,
        "updated": False,
        "reason": "",
    }

    # Check if we have enough data
    total_n = accuracy["total_n"]
    if total_n < min_samples:
        result["reason"] = f"Insufficient data ({total_n} < {min_samples} min)"
        result["new_weights"] = old_weights
        return result

    # Multiplicative weight update
    sharp_acc = accuracy["sharp_accuracy"]
    proj_acc = accuracy["projection_accuracy"]

    # Recent form accuracy: proxy from overall win rate
    form_acc = 0.50  # Neutral by default (no separate tracking yet)

    # Apply multiplicative update
    w_sharp = current["sharp_odds"] * (1 + learning_rate * (sharp_acc - 0.5))
    w_proj = current["projection"] * (1 + learning_rate * (proj_acc - 0.5))
    w_form = current["recent_form"] * (1 + learning_rate * (form_acc - 0.5))

    # Enforce minimum weights (don't let any source go to zero)
    MIN_WEIGHT = 0.05
    w_sharp = max(MIN_WEIGHT, w_sharp)
    w_proj = max(MIN_WEIGHT, w_proj)
    w_form = max(MIN_WEIGHT, w_form)

    # Renormalize to sum to 1
    total = w_sharp + w_proj + w_form
    new_weights = {
        "sharp_odds": round(w_sharp / total, 4),
        "projection": round(w_proj / total, 4),
        "recent_form": round(w_form / total, 4),
    }

    # Enforce maximum weights too (no single source > 0.80)
    MAX_WEIGHT = 0.80
    for key in new_weights:
        if new_weights[key] > MAX_WEIGHT:
            excess = new_weights[key] - MAX_WEIGHT
            new_weights[key] = MAX_WEIGHT
            # Redistribute excess proportionally
            others = [k for k in new_weights if k != key]
            other_total = sum(new_weights[k] for k in others)
            for ok in others:
                new_weights[ok] += excess * (new_weights[ok] / max(other_total, 0.01))

    # Save to weights file
    weights_data["ensemble_weights"] = new_weights
    with open(weights_path, "w") as f:
        json.dump(weights_data, f, indent=2)

    # Log to history
    conn = get_connection()
    conn.execute("""
        INSERT INTO ensemble_history
        (date, sharp_weight, projection_weight, recent_form_weight,
         sharp_accuracy, projection_accuracy, recent_form_accuracy,
         num_graded, learning_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date.today().isoformat(),
        new_weights["sharp_odds"],
        new_weights["projection"],
        new_weights["recent_form"],
        sharp_acc, proj_acc, form_acc,
        total_n, learning_rate,
    ))
    conn.commit()
    conn.close()

    result["new_weights"] = new_weights
    result["updated"] = True
    result["reason"] = f"Updated from {total_n} graded picks"

    return result


def get_current_ensemble_weights() -> Dict:
    """Load the current ensemble weights from the weights file."""
    try:
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "weights", "current.json"
        )
        with open(weights_path) as f:
            w = json.load(f)
        return w.get("ensemble_weights", {
            "sharp_odds": 0.65,
            "projection": 0.30,
            "recent_form": 0.05,
        })
    except Exception:
        return {"sharp_odds": 0.65, "projection": 0.30, "recent_form": 0.05}


def get_weight_history(days: int = 30) -> pd.DataFrame:
    """Get the ensemble weight evolution over time."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT date, sharp_weight, projection_weight, recent_form_weight,
               sharp_accuracy, projection_accuracy, num_graded
        FROM ensemble_history
        WHERE date >= date('now', ? || ' days')
        ORDER BY date ASC
    """, conn, params=(f"-{days}",))
    conn.close()
    return df


# Wire the ensemble weights into combined.py scoring
def get_blend_weights() -> tuple:
    """Get the current sharp/projection blend weights for score_single_pick.

    Returns:
        (sharp_weight, proj_weight) tuple that sums to ~1.0
    """
    w = get_current_ensemble_weights()
    s = w.get("sharp_odds", 0.65)
    p = w.get("projection", 0.30)
    # Normalize to just sharp + proj (recent_form is applied elsewhere)
    total = s + p
    if total <= 0:
        return 0.65, 0.35
    return round(s / total, 3), round(p / total, 3)


# Initialize table on import
init_ensemble_history_table()
