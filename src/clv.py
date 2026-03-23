"""
Closing line value helpers.

Tracks actual bet-time probabilities for saved slip legs, then updates those
records with the latest PrizePicks line snapshot before game time.
"""

from __future__ import annotations

import pandas as pd
from datetime import date, datetime
from typing import Dict

from src.database import get_connection, init_clv_table
from src.line_snapshots import get_closing_line
from src.predictor import calculate_over_under_probability


CLV_EPSILON = 0.001


def record_opening_line(player_name: str, prop_type: str, direction: str,
                        pp_line: float, bet_probability: float,
                        edge_source: str = "combined",
                        game_date: str | None = None,
                        prediction_id: int | None = None,
                        projection: float | None = None) -> int:
    """Insert a CLV record for a saved bet."""
    init_clv_table()
    game_date = game_date or date.today().isoformat()

    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO clv_tracking
        (game_date, player_name, stat_type, line, pick, opening_prob, our_prob,
         prediction_id, projection, edge_source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_date,
        player_name,
        prop_type,
        pp_line,
        direction,
        bet_probability,
        bet_probability,
        prediction_id,
        projection,
        edge_source,
        datetime.now().isoformat(),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def record_prediction_clv(pred: dict, edge_source: str = "slip") -> int | None:
    """Convenience wrapper to track CLV from a prediction payload."""
    player_name = pred.get("player_name")
    prop_type = pred.get("stat_internal") or pred.get("stat_type")
    direction = pred.get("pick")
    line = pred.get("line")
    win_prob = pred.get("win_prob")

    if not player_name or not prop_type or not direction or line is None or win_prob is None:
        return None

    return record_opening_line(
        player_name=player_name,
        prop_type=prop_type,
        direction=direction,
        pp_line=float(line),
        bet_probability=float(win_prob),
        edge_source=edge_source,
        game_date=pred.get("game_date"),
        prediction_id=pred.get("prediction_id"),
        projection=pred.get("projection"),
    )


def _closing_prob_for_direction(projection: float | None, closing_line: float,
                                prop_type: str, direction: str) -> float | None:
    """Recompute our side's outright hit probability at the closing line."""
    if projection is None:
        return None

    try:
        result = calculate_over_under_probability(float(projection), float(closing_line), prop_type)
    except Exception:
        return None

    if direction == "MORE":
        return float(result.get("p_over", 0.0))
    return float(result.get("p_under", 0.0))


def update_closing_lines(game_date: str | None = None, days_back: int = 7) -> dict:
    """Update open CLV records with the latest available closing-line snapshot."""
    init_clv_table()
    conn = get_connection()

    if game_date:
        rows = conn.execute("""
            SELECT id, game_date, player_name, stat_type, pick, line, opening_prob, projection
            FROM clv_tracking
            WHERE game_date = ?
              AND (closing_prob IS NULL OR closing_line IS NULL)
        """, (game_date,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, game_date, player_name, stat_type, pick, line, opening_prob, projection
            FROM clv_tracking
            WHERE game_date >= date('now', '-' || ? || ' days')
              AND (closing_prob IS NULL OR closing_line IS NULL)
        """, (days_back,)).fetchall()

    updated = 0
    missing = 0

    for row_id, row_game_date, player_name, stat_type, direction, line, opening_prob, projection in rows:
        closing_line = get_closing_line(player_name, stat_type, row_game_date)
        if closing_line is None:
            missing += 1
            continue

        closing_prob = _closing_prob_for_direction(projection, closing_line, stat_type, direction)
        if closing_prob is None:
            # Fall back to opening probability when we lack the inputs to recompute.
            closing_prob = float(opening_prob or 0.0)

        clv_points = float(opening_prob or 0.0) - float(closing_prob)
        beat_close = 1 if clv_points > CLV_EPSILON else 0

        conn.execute("""
            UPDATE clv_tracking
            SET closing_line = ?, closing_prob = ?, clv_points = ?, beat_close = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            float(closing_line),
            float(closing_prob),
            float(clv_points),
            beat_close,
            datetime.now().isoformat(),
            row_id,
        ))
        updated += 1

    conn.commit()
    conn.close()
    return {"updated": updated, "missing_closing_line": missing}


def _normalize_outcome(outcome) -> int | None:
    """Normalize outcome to 1/0 when possible."""
    if outcome in (1, 0):
        return int(outcome)
    if isinstance(outcome, str):
        key = outcome.strip().upper()
        if key == "W":
            return 1
        if key == "L":
            return 0
    return None


def record_outcome(player_name: str, prop_type: str, outcome,
                   game_date: str | None = None,
                   prediction_id: int | None = None) -> int:
    """Update matching CLV rows with the actual result."""
    norm = _normalize_outcome(outcome)
    if norm is None:
        return 0

    game_date = game_date or date.today().isoformat()
    conn = get_connection()

    if prediction_id is not None:
        cur = conn.execute("""
            UPDATE clv_tracking
            SET outcome = ?, updated_at = ?
            WHERE prediction_id = ?
        """, (norm, datetime.now().isoformat(), prediction_id))
    else:
        cur = conn.execute("""
            UPDATE clv_tracking
            SET outcome = ?, updated_at = ?
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
        """, (norm, datetime.now().isoformat(), game_date, player_name, prop_type))

    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated


def compute_clv_stats(days: int = 30) -> Dict:
    """Compute CLV summary stats for recent tracked bets."""
    init_clv_table()
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT *
        FROM clv_tracking
        WHERE game_date >= date('now', '-' || ? || ' days')
          AND clv_points IS NOT NULL
    """, conn, params=(days,))
    conn.close()

    if df.empty:
        return {"total_bets": 0, "mean_clv": 0, "by_source": {}, "by_prop": {}}

    result = {
        "total_bets": int(len(df)),
        "mean_clv": round(float(df["clv_points"].mean()), 4),
        "median_clv": round(float(df["clv_points"].median()), 4),
        "positive_clv_rate": round(float((df["clv_points"] > 0).mean()), 4),
    }

    result["by_source"] = {}
    if "edge_source" in df.columns:
        for source, sdf in df.groupby(df["edge_source"].fillna("unknown")):
            graded = sdf[sdf["outcome"].notna()]
            result["by_source"][source] = {
                "count": int(len(sdf)),
                "mean_clv": round(float(sdf["clv_points"].mean()), 4),
                "win_rate": round(float(graded["outcome"].mean()), 4) if len(graded) else 0.0,
                "positive_clv_rate": round(float((sdf["clv_points"] > 0).mean()), 4),
            }

    result["by_prop"] = {}
    for prop_type, pdf in df.groupby("stat_type"):
        graded = pdf[pdf["outcome"].notna()]
        result["by_prop"][prop_type] = {
            "count": int(len(pdf)),
            "mean_clv": round(float(pdf["clv_points"].mean()), 4),
            "win_rate": round(float(graded["outcome"].mean()), 4) if len(graded) else 0.0,
        }

    graded = df[df["outcome"].notna()]
    if len(graded) >= 10:
        pos_clv = graded[graded["clv_points"] > 0]
        neg_clv = graded[graded["clv_points"] <= 0]
        result["clv_correlation"] = {
            "positive_clv_win_rate": round(float(pos_clv["outcome"].mean()), 4) if len(pos_clv) else 0.0,
            "negative_clv_win_rate": round(float(neg_clv["outcome"].mean()), 4) if len(neg_clv) else 0.0,
        }

    return result
