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


# Initialize DB on import
init_db()
