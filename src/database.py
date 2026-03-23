"""
Database Module
SQLite-based prediction logging, auto-grading, and accuracy tracking.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path


DB_PATH = Path("data/predictions.db")
PACIFIC_TZ_NAME = "America/Los_Angeles"


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def resolve_game_date(pred: dict = None, game_date: str = None) -> str:
    """Resolve the best available game date for a prediction payload."""
    if game_date:
        return str(game_date)

    pred = pred or {}
    explicit = pred.get("game_date")
    if explicit:
        return str(explicit)

    for key in ("game_time_utc", "start_time", "game_time"):
        raw_value = pred.get(key)
        if not raw_value:
            continue
        try:
            ts = pd.Timestamp(raw_value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.tz_convert(PACIFIC_TZ_NAME).date().isoformat()
        except Exception:
            continue

    return date.today().isoformat()


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
    # Older app versions inserted the full board on every rerun. Collapse those
    # duplicates so the uniqueness rule below can be enforced safely.
    conn.execute("""
        DELETE FROM projected_stats
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM projected_stats
            GROUP BY game_date, player_name, stat_type
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_date ON projected_stats(game_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_player ON projected_stats(player_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projected_stats_stat ON projected_stats(stat_type)")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_projected_stats_unique
        ON projected_stats(game_date, player_name, stat_type)
    """)
    conn.commit()
    conn.close()


def init_clv_table():
    """Create the CLV tracking table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            pick TEXT NOT NULL,
            opening_prob REAL,
            closing_prob REAL,
            our_prob REAL,
            clv_points REAL,
            beat_close INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clv_date ON clv_tracking(game_date)")
    conn.commit()
    conn.close()


def log_prediction(pred: dict, game_date: str = None):
    """Save a prediction to the database."""
    conn = get_connection()
    resolved_game_date = resolve_game_date(pred, game_date)

    cur = conn.execute("""
        INSERT INTO predictions
        (game_date, player_name, stat_type, stat_internal, line, projection,
         pick, confidence, rating, p_over, p_under, edge, park_team,
         weather_temp, weather_wind, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resolved_game_date,
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
    row_id = cur.lastrowid
    conn.close()
    return row_id


def log_batch_predictions(predictions: list, game_date: str = None):
    """Save multiple predictions at once."""
    inserted_ids = []
    for pred in predictions:
        inserted_ids.append(log_prediction(pred, game_date))
    return inserted_ids


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
    if not predictions:
        return

    conn = get_connection()
    rows = [(
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
    ) for pred in predictions]
    conn.executemany("""
        INSERT INTO projected_stats
        (game_date, player_name, team, stat_type, projected_value,
         actual_value, line, pick, confidence, rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_date, player_name, stat_type) DO UPDATE SET
            team = excluded.team,
            projected_value = excluded.projected_value,
            line = excluded.line,
            pick = excluded.pick,
            confidence = excluded.confidence,
            rating = excluded.rating
    """, rows)
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
            SELECT projected_value, line, pick FROM projected_stats
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
            ORDER BY id DESC
            LIMIT 1
        """, (game_date, player_name, stat_type))

        row = cur.fetchone()
        if not row:
            continue

        projected_value, line, pick = row

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
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
        """, (actual_value, was_correct, game_date, player_name, stat_type))

    conn.commit()
    conn.close()


def get_projection_accuracy(days_back: int = 30) -> dict:
    """Calculate accuracy stats for projected stats over last N days.

    Args:
        days_back (int): Number of days to look back

    Returns:
        dict with per-prop accuracy stats including Brier score, log loss, and calibration
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
            "brier_score": None,
            "log_loss": None,
            "calibration_by_bucket": [],
            "by_stat_type": {},
            "by_rating": {},
            "by_pick": {}
        }

    total = len(graded)
    correct = len(graded[graded["was_correct"] == 1])
    accuracy = correct / total if total > 0 else 0

    # Compute Brier score: mean of (predicted_probability - outcome)^2
    brier_score = None
    log_loss = None
    calibration_by_bucket = []

    if not graded["confidence"].isna().all() and not graded["was_correct"].isna().all():
        # Brier score
        brier_score = np.mean((graded["confidence"] - graded["was_correct"]) ** 2)

        # Log loss: -[y*log(p) + (1-y)*log(1-p)]
        # Clamp confidence to avoid log(0)
        confidence_clamped = np.clip(graded["confidence"], 1e-10, 1 - 1e-10)
        log_loss = -np.mean(
            graded["was_correct"] * np.log(confidence_clamped) +
            (1 - graded["was_correct"]) * np.log(1 - confidence_clamped)
        )

        # Calibration by bucket
        buckets = [
            (0.50, 0.55, "50-55%"),
            (0.55, 0.60, "55-60%"),
            (0.60, 0.65, "60-65%"),
            (0.65, 0.70, "65-70%"),
            (0.70, 1.01, "70%+")
        ]

        for lower, upper, label in buckets:
            mask = (graded["confidence"] >= lower) & (graded["confidence"] < upper)
            bucket_data = graded[mask]
            if len(bucket_data) > 0:
                predicted_mean = bucket_data["confidence"].mean()
                actual_rate = bucket_data["was_correct"].mean()
                calibration_by_bucket.append({
                    "bucket": label,
                    "predicted_mean": float(predicted_mean),
                    "actual_rate": float(actual_rate),
                    "count": int(len(bucket_data))
                })

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
        "brier_score": float(brier_score) if brier_score is not None else None,
        "log_loss": float(log_loss) if log_loss is not None else None,
        "calibration_by_bucket": calibration_by_bucket,
        "by_stat_type": by_stat_type,
        "by_rating": by_rating,
        "by_pick": by_pick
    }


def get_projection_diagnostics(days_back: int = 30) -> dict:
    """Return residual-based diagnostics for graded projected stats."""
    conn = get_connection()
    graded = pd.read_sql_query(
        """
        SELECT game_date, player_name, team, stat_type, projected_value, actual_value,
               line, pick, confidence, rating, was_correct
        FROM projected_stats
        WHERE actual_value IS NOT NULL
          AND game_date >= date('now', '-' || ? || ' days')
        ORDER BY game_date DESC, player_name ASC
        """,
        conn,
        params=(days_back,),
    )
    conn.close()

    if graded.empty:
        return {
            "total": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "accuracy": None,
            "by_stat_type": {},
            "worst_misses": [],
            "daily_summary": [],
        }

    graded = graded.copy()
    graded["error"] = graded["actual_value"] - graded["projected_value"]
    graded["abs_error"] = graded["error"].abs()
    graded["sq_error"] = graded["error"] ** 2

    overall = {
        "total": int(len(graded)),
        "mae": float(graded["abs_error"].mean()),
        "rmse": float(np.sqrt(graded["sq_error"].mean())),
        "bias": float(graded["error"].mean()),
        "accuracy": float(graded["was_correct"].dropna().mean()) if graded["was_correct"].notna().any() else None,
    }

    by_stat_type = {}
    for stat_type, subset in graded.groupby("stat_type"):
        by_stat_type[stat_type] = {
            "count": int(len(subset)),
            "mae": float(subset["abs_error"].mean()),
            "rmse": float(np.sqrt(subset["sq_error"].mean())),
            "bias": float(subset["error"].mean()),
            "accuracy": float(subset["was_correct"].dropna().mean()) if subset["was_correct"].notna().any() else None,
        }

    worst = (
        graded.sort_values("abs_error", ascending=False)
        .head(15)[[
            "game_date", "player_name", "team", "stat_type",
            "projected_value", "actual_value", "error", "line", "pick", "confidence"
        ]]
        .to_dict("records")
    )

    daily_summary = []
    for game_date, subset in graded.groupby("game_date"):
        daily_summary.append({
            "game_date": game_date,
            "count": int(len(subset)),
            "mae": float(subset["abs_error"].mean()),
            "rmse": float(np.sqrt(subset["sq_error"].mean())),
            "bias": float(subset["error"].mean()),
            "accuracy": float(subset["was_correct"].dropna().mean()) if subset["was_correct"].notna().any() else None,
        })

    daily_summary.sort(key=lambda row: row["game_date"], reverse=True)

    return {
        **overall,
        "by_stat_type": by_stat_type,
        "worst_misses": worst,
        "daily_summary": daily_summary,
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


def get_calibration_data(days_back: int = 30) -> dict:
    """Get calibration data for reliability diagrams.

    Groups predictions into 5 buckets by confidence and computes actual win rate
    per bucket. Also returns overall Brier score and log loss.

    Args:
        days_back (int): Number of days to look back

    Returns:
        dict with 'buckets' (list of dicts) and 'brier_score' and 'log_loss'
    """
    conn = get_connection()

    # Get graded projections from last N days
    graded = pd.read_sql_query("""
        SELECT confidence, was_correct FROM projected_stats
        WHERE was_correct IS NOT NULL AND confidence IS NOT NULL
        AND game_date >= date('now', '-' || ? || ' days')
    """, conn, params=(days_back,))
    conn.close()

    if graded.empty:
        return {
            "buckets": [],
            "brier_score": None,
            "log_loss": None,
            "total_samples": 0
        }

    # Compute Brier score
    brier_score = np.mean((graded["confidence"] - graded["was_correct"]) ** 2)

    # Compute log loss
    confidence_clamped = np.clip(graded["confidence"], 1e-10, 1 - 1e-10)
    log_loss = -np.mean(
        graded["was_correct"] * np.log(confidence_clamped) +
        (1 - graded["was_correct"]) * np.log(1 - confidence_clamped)
    )

    # Calibration buckets
    buckets = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 1.01, "70%+")
    ]

    bucket_list = []
    for lower, upper, label in buckets:
        mask = (graded["confidence"] >= lower) & (graded["confidence"] < upper)
        bucket_data = graded[mask]
        if len(bucket_data) > 0:
            predicted_mean = bucket_data["confidence"].mean()
            actual_rate = bucket_data["was_correct"].mean()
            bucket_list.append({
                "bucket": label,
                "predicted_mean": float(predicted_mean),
                "actual_rate": float(actual_rate),
                "count": int(len(bucket_data))
            })

    return {
        "buckets": bucket_list,
        "brier_score": float(brier_score),
        "log_loss": float(log_loss),
        "total_samples": int(len(graded))
    }


def save_clv_record(records: list):
    """Save CLV tracking records (bulk insert).

    Args:
        records (list[dict]): List of CLV records with keys:
            - game_date (str)
            - player_name (str)
            - stat_type (str)
            - line (float)
            - pick (str: 'MORE' or 'LESS')
            - opening_prob (float, optional)
            - closing_prob (float, optional)
            - our_prob (float, optional)
            - clv_points (float, optional)
            - beat_close (int, optional: 0 or 1)
    """
    conn = get_connection()
    for record in records:
        conn.execute("""
            INSERT INTO clv_tracking
            (game_date, player_name, stat_type, line, pick, opening_prob,
             closing_prob, our_prob, clv_points, beat_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("game_date"),
            record.get("player_name"),
            record.get("stat_type"),
            record.get("line"),
            record.get("pick"),
            record.get("opening_prob"),
            record.get("closing_prob"),
            record.get("our_prob"),
            record.get("clv_points"),
            record.get("beat_close", 0),
        ))
    conn.commit()
    conn.close()


def get_clv_summary(days_back: int = 30) -> dict:
    """Get CLV summary stats for the last N days.

    Args:
        days_back (int): Number of days to look back

    Returns:
        dict with overall CLV stats and breakdown by prop type:
            - total_tracked (int): Number of tracked bets
            - avg_clv (float): Average CLV points per bet
            - beat_close_pct (float): % of bets that beat the closing line
            - by_prop (dict): Breakdown by stat_type with avg_clv, beat_pct, count
    """
    conn = get_connection()

    # Get CLV records from last N days
    clv_data = pd.read_sql_query("""
        SELECT stat_type, clv_points, beat_close FROM clv_tracking
        WHERE game_date >= date('now', '-' || ? || ' days')
    """, conn, params=(days_back,))
    conn.close()

    if clv_data.empty:
        return {
            "total_tracked": 0,
            "avg_clv": None,
            "beat_close_pct": None,
            "by_prop": {}
        }

    total_tracked = len(clv_data)
    avg_clv = float(clv_data["clv_points"].mean())

    beat_close_count = len(clv_data[clv_data["beat_close"] == 1])
    beat_close_pct = (beat_close_count / total_tracked * 100) if total_tracked > 0 else 0

    # Breakdown by stat type
    by_prop = {}
    for stat_type in clv_data["stat_type"].unique():
        if pd.isna(stat_type):
            continue
        subset = clv_data[clv_data["stat_type"] == stat_type]
        prop_avg_clv = float(subset["clv_points"].mean())
        prop_beat_count = len(subset[subset["beat_close"] == 1])
        prop_beat_pct = (prop_beat_count / len(subset) * 100) if len(subset) > 0 else 0
        by_prop[stat_type] = {
            "avg_clv": prop_avg_clv,
            "beat_pct": prop_beat_pct,
            "count": int(len(subset))
        }

    return {
        "total_tracked": total_tracked,
        "avg_clv": avg_clv,
        "beat_close_pct": beat_close_pct,
        "by_prop": by_prop
    }


# Initialize DB on import
init_db()
init_projected_stats_table()
init_clv_table()
