"""
Database Module
SQLite-based prediction logging, auto-grading, and accuracy tracking.
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path


DB_PATH = Path("data/predictions.db")


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            game_date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            stat_internal TEXT NOT NULL,
            line REAL NOT NULL,
            projection REAL NOT NULL,
            pick TEXT NOT NULL,
            confidence REAL NOT NULL,
            rating TEXT,
            p_over REAL,
            p_under REAL,
            edge REAL,
            park_team TEXT,
            weather_temp REAL,
            weather_wind REAL,
            model_version TEXT DEFAULT 'v1.0',
            -- Result fields (filled after game)
            actual_result REAL,
            result TEXT,  -- 'W', 'L', 'push'
            graded_at TEXT
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result);
    """)
    conn.commit()
    conn.close()


def init_projected_stats_table():
    """Create the projected_stats table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projected_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT,
            stat_type TEXT NOT NULL,
            projected_value REAL NOT NULL,
            actual_value REAL,
            line REAL,
            pick TEXT,
            confidence REAL,
            rating TEXT,
            was_correct INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_date ON projected_stats(game_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_player ON projected_stats(player_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_stat ON projected_stats(stat_type)")
    conn.commit()
    conn.close()


def log_prediction(pred: dict, game_date: str = None):
    """Save a prediction to the database."""
    conn = get_connection()
    game_date = game_date or date.today().isoformat()

    conn.execute("""
        INSERT INTO predictions
        (game_date, player_name, stat_type, stat_internal, line, projection,
         pick, confidence, rating, p_over, p_under, edge, park_team,
         weather_temp, weather_wind, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_date,
        pred.get("player_name", ""),
        pred.get("stat_type", ""),
        pred.get("stat_internal", ""),
        pred.get("line", 0),
        pred.get("projection", 0),
        pred.get("pick", ""),
        pred.get("confidence", 0),
        pred.get("rating", ""),
        pred.get("p_over", 0),
        pred.get("p_under", 0),
        pred.get("edge", 0),
        pred.get("park_team", ""),
        pred.get("weather_temp", None),
        pred.get("weather_wind", None),
        pred.get("model_version", "v1.0"),
    ))
    conn.commit()
    conn.close()


def log_batch_predictions(predictions: list, game_date: str = None):
    """Save multiple predictions at once."""
    for pred in predictions:
        log_prediction(pred, game_date)


def grade_prediction(pred_id: int, actual_result: float):
    """Grade a single prediction given the actual outcome."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT line, pick FROM predictions WHERE id = ?", (pred_id,)
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return

    line, pick = row

    if actual_result > line:
        result = "W" if pick == "MORE" else "L"
    elif actual_result < line:
        result = "W" if pick == "LESS" else "L"
    else:
        result = "push"

    conn.execute("""
        UPDATE predictions
        SET actual_result = ?, result = ?, graded_at = datetime('now')
        WHERE id = ?
    """, (actual_result, result, pred_id))
    conn.commit()
    conn.close()
    return result


def get_ungraded_predictions(game_date: str = None) -> pd.DataFrame:
    """Get all predictions that haven't been graded yet."""
    conn = get_connection()
    query = "SELECT * FROM predictions WHERE result IS NULL"
    params = ()
    if game_date:
        query += " AND game_date = ?"
        params = (game_date,)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_all_predictions(limit: int = 500) -> pd.DataFrame:
    """Get all predictions, most recent first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_graded_predictions(limit: int = 500) -> pd.DataFrame:
    """Get only graded predictions."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE result IS NOT NULL ORDER BY game_date DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_accuracy_stats() -> dict:
    """Calculate overall accuracy and breakdown by stat type, rating, etc."""
    conn = get_connection()
    graded = pd.read_sql_query(
        "SELECT * FROM predictions WHERE result IS NOT NULL", conn
    )
    conn.close()

    if graded.empty:
        return {"total": 0, "accuracy": 0, "breakdown": {}}

    total = len(graded[graded["result"].isin(["W", "L"])])
    wins = len(graded[graded["result"] == "W"])
    accuracy = wins / total if total > 0 else 0

    # By stat type
    by_stat = {}
    for stat in graded["stat_type"].unique():
        subset = graded[(graded["stat_type"] == stat) & (graded["result"].isin(["W", "L"]))]
        w = len(subset[subset["result"] == "W"])
        t = len(subset)
        by_stat[stat] = {"wins": w, "total": t, "accuracy": w / t if t > 0 else 0}

    # By rating
    by_rating = {}
    for rating in ["A", "B", "C", "D"]:
        subset = graded[(graded["rating"] == rating) & (graded["result"].isin(["W", "L"]))]
        w = len(subset[subset["result"] == "W"])
        t = len(subset)
        by_rating[rating] = {"wins": w, "total": t, "accuracy": w / t if t > 0 else 0}

    # By pick direction
    by_direction = {}
    for pick in ["MORE", "LESS"]:
        subset = graded[(graded["pick"] == pick) & (graded["result"].isin(["W", "L"]))]
        w = len(subset[subset["result"] == "W"])
        t = len(subset)
        by_direction[pick] = {"wins": w, "total": t, "accuracy": w / t if t > 0 else 0}

    return {
        "total": total,
        "wins": wins,
        "losses": total - wins,
        "pushes": len(graded[graded["result"] == "push"]),
        "accuracy": accuracy,
        "by_stat": by_stat,
        "by_rating": by_rating,
        "by_direction": by_direction,
    }


def save_projected_stats(predictions: list):
    """Bulk insert projected stats for a day.

    Args:
        predictions: List of dicts with keys:
            - game_date (str)
            - player_name (str)
            - team (str, optional)
            - stat_type (str)
            - projected_value (float)
            - actual_value (float, optional)
            - line (float, optional)
            - pick (str, optional: 'MORE'/'LESS')
            - confidence (float, optional)
            - rating (str, optional: 'A'/'B'/'C'/'D')
    """
    conn = get_connection()
    for pred in predictions:
        conn.execute("""
            INSERT INTO projected_stats
            (game_date, player_name, team, stat_type, projected_value,
             actual_value, line, pick, confidence, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.get("game_date"),
            pred.get("player_name"),
            pred.get("team"),
            pred.get("stat_type"),
            pred.get("projected_value"),
            pred.get("actual_value"),
            pred.get("line"),
            pred.get("pick"),
            pred.get("confidence"),
            pred.get("rating"),
        ))
    conn.commit()
    conn.close()


def grade_projected_stats(game_date: str, actuals: dict):
    """Update actual_value and was_correct for a game date.

    Args:
        game_date (str): Date in YYYY-MM-DD format
        actuals (dict): Mapping of (player_name, stat_type) -> actual_value
    """
    conn = get_connection()

    for (player_name, stat_type), actual_value in actuals.items():
        # Get the projected row
        cur = conn.execute("""
            SELECT id, projected_value, line, pick FROM projected_stats
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
        """, (game_date, player_name, stat_type))

        row = cur.fetchone()
        if not row:
            continue

        pred_id, projected_value, line, pick = row

        # Determine if correct
        was_correct = None
        if line is not None and pick is not None:
            if actual_value > line:
                was_correct = 1 if pick == "MORE" else 0
            elif actual_value < line:
                was_correct = 1 if pick == "LESS" else 0
            else:
                # Push scenario - could treat as correct or neutral
                was_correct = None

        conn.execute("""
            UPDATE projected_stats
            SET actual_value = ?, was_correct = ?
            WHERE id = ?
        """, (actual_value, was_correct, pred_id))

    conn.commit()
    conn.close()


def get_projection_accuracy(days_back: int = 30) -> dict:
    """Calculate accuracy stats for projected stats over last N days.

    Args:
        days_back (int): Number of days to look back

    Returns:
        dict with per-prop accuracy stats
    """
    conn = get_connection()

    # Get graded projections from last N days
    graded = pd.read_sql_query("""
        SELECT * FROM projected_stats
        WHERE was_correct IS NOT NULL
        AND game_date >= date('now', '-' || ? || ' days')
    """, conn, params=(days_back,))
    conn.close()

    if graded.empty:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0,
            "by_stat_type": {},
            "by_rating": {},
            "by_pick": {}
        }

    total = len(graded)
    correct = len(graded[graded["was_correct"] == 1])
    accuracy = correct / total if total > 0 else 0

    # By stat type
    by_stat_type = {}
    for stat_type in graded["stat_type"].unique():
        if pd.isna(stat_type):
            continue
        subset = graded[graded["stat_type"] == stat_type]
        c = len(subset[subset["was_correct"] == 1])
        t = len(subset)
        by_stat_type[stat_type] = {
            "correct": c,
            "total": t,
            "accuracy": c / t if t > 0 else 0
        }

    # By rating
    by_rating = {}
    for rating in ["A", "B", "C", "D"]:
        subset = graded[graded["rating"] == rating]
        if len(subset) > 0:
            c = len(subset[subset["was_correct"] == 1])
            t = len(subset)
            by_rating[rating] = {
                "correct": c,
                "total": t,
                "accuracy": c / t if t > 0 else 0
            }

    # By pick direction
    by_pick = {}
    for pick in ["MORE", "LESS"]:
        subset = graded[graded["pick"] == pick]
        if len(subset) > 0:
            c = len(subset[subset["was_correct"] == 1])
            t = len(subset)
            by_pick[pick] = {
                "correct": c,
                "total": t,
                "accuracy": c / t if t > 0 else 0
            }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_stat_type": by_stat_type,
        "by_rating": by_rating,
        "by_pick": by_pick
    }


def get_projection_history(player_name: str, stat_type: str, limit: int = 20) -> list:
    """Get recent projections vs actuals for a player and stat type.

    Args:
        player_name (str): Player name
        stat_type (str): Stat type (e.g., 'hits', 'total_bases')
        limit (int): Max number of records to return

    Returns:
        list of dicts with projection and actual data
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT game_date, team, projected_value, actual_value, line, pick,
               confidence, rating, was_correct
        FROM projected_stats
        WHERE player_name = ? AND stat_type = ?
        ORDER BY game_date DESC
        LIMIT ?
    """, (player_name, stat_type, limit)).fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "game_date": row[0],
            "team": row[1],
            "projected_value": row[2],
            "actual_value": row[3],
            "line": row[4],
            "pick": row[5],
            "confidence": row[6],
            "rating": row[7],
            "was_correct": row[8]
        })

    return result


def get_daily_projection_summary(game_date: str) -> dict:
    """Get summary of all projections for a date with actuals if available.

    Args:
        game_date (str): Date in YYYY-MM-DD format

    Returns:
        dict with summary stats and list of projections
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT player_name, team, stat_type, projected_value, actual_value,
               line, pick, confidence, rating, was_correct
        FROM projected_stats
        WHERE game_date = ?
        ORDER BY player_name, stat_type
    """, (game_date,)).fetchall()
    conn.close()

    projections = []
    for row in rows:
        projections.append({
            "player_name": row[0],
            "team": row[1],
            "stat_type": row[2],
            "projected_value": row[3],
            "actual_value": row[4],
            "line": row[5],
            "pick": row[6],
            "confidence": row[7],
            "rating": row[8],
            "was_correct": row[9]
        })

    # Calculate summary stats
    total_count = len(projections)
    graded_count = len([p for p in projections if p["actual_value"] is not None])
    correct_count = len([p for p in projections if p["was_correct"] == 1])

    summary = {
        "game_date": game_date,
        "total_projections": total_count,
        "graded_projections": graded_count,
        "correct_projections": correct_count,
        "grading_complete": graded_count == total_count,
        "accuracy": correct_count / graded_count if graded_count > 0 else None,
        "projections": projections
    }

    return summary


# Initialize DB on import
init_db()
init_projected_stats_table()
