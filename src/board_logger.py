"""
Daily Board Logger — Logs EVERY evaluated prop for bias-free learning.

Stores a complete snapshot of every prop the model evaluates each day,
not just the ones selected for slips. This prevents selection bias in
the learning loop and enables CLV tracking.

Tables:
  daily_board — one row per evaluated prop/direction
"""

import hashlib
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone
from typing import List, Dict

from src.database import get_connection, resolve_game_date


def _ensure_column(conn, table_name: str, column_name: str, column_def: str) -> None:
    cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _confidence_bucket(confidence: float) -> str:
    if confidence is None:
        return "50-55%"
    if confidence < 0.55:
        return "50-55%"
    if confidence < 0.60:
        return "55-60%"
    if confidence < 0.65:
        return "60-65%"
    if confidence < 0.70:
        return "65-70%"
    return "70%+"


def _shadow_seed(game_date: str) -> int:
    digest = hashlib.md5(str(game_date).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def init_board_table():
    """Create the daily_board table if it doesn't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_board (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT,
            prop_type TEXT NOT NULL,
            line REAL NOT NULL,
            direction TEXT NOT NULL,
            model_probability REAL,
            sharp_implied_probability REAL,
            pp_line REAL,
            edge REAL,
            grade TEXT,
            confidence REAL,
            projection REAL,
            was_bet INTEGER DEFAULT 0,
            actual_stat REAL,
            outcome INTEGER,
            model_version TEXT,
            line_type TEXT DEFAULT 'standard',
            edge_source TEXT,
            is_shadow_sample INTEGER DEFAULT 0,
            shadow_bucket TEXT,
            shadow_selected_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_board_date ON daily_board(date);
        CREATE INDEX IF NOT EXISTS idx_board_player ON daily_board(player_name);
        CREATE INDEX IF NOT EXISTS idx_board_prop ON daily_board(prop_type);
        CREATE INDEX IF NOT EXISTS idx_board_outcome ON daily_board(outcome);
    """)
    _ensure_column(conn, "daily_board", "is_shadow_sample", "INTEGER DEFAULT 0")
    _ensure_column(conn, "daily_board", "shadow_bucket", "TEXT")
    _ensure_column(conn, "daily_board", "shadow_selected_at", "TEXT")
    conn.execute("""
        DELETE FROM daily_board
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM daily_board
            GROUP BY date, player_name, prop_type, line, direction, line_type
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_board_unique
        ON daily_board(date, player_name, prop_type, line, direction, line_type)
    """)
    conn.commit()
    conn.close()


def log_board_snapshot(predictions: List[Dict], edges: List[Dict] = None,
                       model_version: str = "v018") -> int:
    """Log all evaluated props to the daily_board table.

    Args:
        predictions: List of prediction dicts from generate_prediction
        edges: Optional list of sharp edge dicts for matching
        model_version: Current model version string

    Returns:
        Number of rows inserted
    """
    if not predictions:
        return 0

    # Build a lookup for sharp edges
    edge_lookup = {}
    if edges:
        for e in edges:
            key = f"{e.get('player_name', '')}_{e.get('stat_type', '')}"
            edge_lookup[key] = e

    conn = get_connection()
    rows = []
    for p in predictions:
        player = p.get("player_name", "")
        prop_type = p.get("stat_internal", p.get("stat_type", ""))
        key = f"{player}_{prop_type}"
        sharp = edge_lookup.get(key, {})
        row_date = resolve_game_date(p)

        rows.append((
            row_date,
            player,
            p.get("team", ""),
            prop_type,
            p.get("line", 0),
            p.get("pick", "MORE"),
            p.get("confidence", 0.5),
            sharp.get("sharp_implied_prob"),
            p.get("line", 0),
            p.get("edge", 0),
            p.get("rating", "D"),
            p.get("confidence", 0.5),
            p.get("projection", 0),
            0,  # was_bet — updated when slips are created
            None,  # actual_stat — filled after grading
            None,  # outcome — filled after grading
            model_version,
            p.get("line_type", "standard"),
            sharp.get("edge_source", classify_edge_source(p, sharp)),
        ))

    if rows:
        conn.executemany("""
            INSERT INTO daily_board
            (date, player_name, team, prop_type, line, direction,
             model_probability, sharp_implied_probability, pp_line,
             edge, grade, confidence, projection, was_bet,
             actual_stat, outcome, model_version, line_type, edge_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, player_name, prop_type, line, direction, line_type) DO UPDATE SET
                team = excluded.team,
                model_probability = excluded.model_probability,
                sharp_implied_probability = excluded.sharp_implied_probability,
                pp_line = excluded.pp_line,
                edge = excluded.edge,
                grade = excluded.grade,
                confidence = excluded.confidence,
                projection = excluded.projection,
                model_version = excluded.model_version,
                edge_source = excluded.edge_source
        """, rows)
        conn.commit()

    conn.close()
    return len(rows)


def mark_as_bet(player_name: str, prop_type: str, game_date: str = None):
    """Mark a board entry as included in a slip."""
    if game_date is None:
        game_date = date.today().isoformat()
    conn = get_connection()
    conn.execute("""
        UPDATE daily_board SET was_bet = 1
        WHERE player_name = ? AND prop_type = ? AND date = ?
    """, (player_name, prop_type, game_date))
    conn.commit()
    conn.close()


def grade_board_entry(player_name: str, prop_type: str, actual_stat: float,
                      game_date: str = None):
    """Grade a board entry with the actual stat value."""
    if game_date is None:
        game_date = date.today().isoformat()
    conn = get_connection()
    cur = conn.execute("""
        SELECT id, line, direction FROM daily_board
        WHERE player_name = ? AND prop_type = ? AND date = ?
    """, (player_name, prop_type, game_date))
    rows = cur.fetchall()
    updated = 0
    for board_id, line, direction in rows:
        if actual_stat > line:
            outcome = 1 if direction == "MORE" else 0
        elif actual_stat < line:
            outcome = 1 if direction == "LESS" else 0
        else:
            outcome = None  # Push
        conn.execute("""
            UPDATE daily_board SET actual_stat = ?, outcome = ? WHERE id = ?
        """, (actual_stat, outcome, board_id))
        updated += 1
    if updated:
        conn.commit()
    conn.close()
    return updated


def get_board_stats(days: int = 30) -> Dict:
    """Get board-level statistics for the last N days."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM daily_board
        WHERE date >= date('now', ? || ' days')
        AND outcome IS NOT NULL
    """, conn, params=(f"-{days}",))
    conn.close()

    if df.empty:
        return {"total": 0, "graded": 0}

    total = len(df)
    wins = len(df[df["outcome"] == 1])
    accuracy = wins / total if total > 0 else 0

    # Break down by was_bet
    bet_df = df[df["was_bet"] == 1]
    nobet_df = df[df["was_bet"] == 0]

    return {
        "total": total,
        "graded": total,
        "accuracy_all": round(accuracy, 4),
        "accuracy_bet": round(len(bet_df[bet_df["outcome"] == 1]) / max(len(bet_df), 1), 4),
        "accuracy_nobet": round(len(nobet_df[nobet_df["outcome"] == 1]) / max(len(nobet_df), 1), 4),
        "selection_bias": round(
            (len(bet_df[bet_df["outcome"] == 1]) / max(len(bet_df), 1)) -
            (len(nobet_df[nobet_df["outcome"] == 1]) / max(len(nobet_df), 1)),
            4
        ),
        "by_grade": {
            grade: {
                "total": len(g_df),
                "wins": len(g_df[g_df["outcome"] == 1]),
                "accuracy": round(len(g_df[g_df["outcome"] == 1]) / max(len(g_df), 1), 4),
            }
            for grade, g_df in df.groupby("grade")
        },
        "by_prop": {
            prop: {
                "total": len(p_df),
                "wins": len(p_df[p_df["outcome"] == 1]),
                "accuracy": round(len(p_df[p_df["outcome"] == 1]) / max(len(p_df), 1), 4),
            }
            for prop, p_df in df.groupby("prop_type")
        },
    }


def ensure_shadow_sample(game_date: str, sample_size: int = 50) -> Dict:
    """Select a stable, stratified random QA sample from the daily board."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT id, prop_type, direction, confidence, line, line_type, is_shadow_sample
        FROM daily_board
        WHERE date = ?
        ORDER BY id
        """,
        conn,
        params=(game_date,),
    )

    if df.empty:
        conn.close()
        return {
            "game_date": game_date,
            "available": 0,
            "shadow_sample_size": 0,
            "selected_now": 0,
            "status": "no_rows",
        }

    target = min(sample_size, len(df))
    existing = df[df["is_shadow_sample"] == 1].copy()
    if len(existing) >= target:
        conn.close()
        return {
            "game_date": game_date,
            "available": int(len(df)),
            "shadow_sample_size": int(len(existing)),
            "selected_now": 0,
            "status": "already_selected",
        }

    remaining = df[df["is_shadow_sample"] != 1].copy()
    if remaining.empty:
        conn.close()
        return {
            "game_date": game_date,
            "available": int(len(df)),
            "shadow_sample_size": int(len(existing)),
            "selected_now": 0,
            "status": "no_remaining_rows",
        }

    seed = _shadow_seed(game_date)
    rng = np.random.default_rng(seed)
    remaining["confidence_bucket"] = remaining["confidence"].apply(_confidence_bucket)
    remaining["line_bucket"] = remaining["line"].apply(lambda x: "0.5" if x <= 0.5 else ("1.5-2.5" if x <= 2.5 else "3.0+"))
    remaining["shadow_bucket"] = remaining.apply(
        lambda row: f"{row['prop_type']}|{row['direction']}|{row['confidence_bucket']}|{row['line_bucket']}",
        axis=1,
    )
    remaining["_rand"] = rng.random(len(remaining))

    needed = target - len(existing)
    selected_parts = []

    strata = (
        remaining.groupby("shadow_bucket", dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not strata.empty:
        strata["_rand"] = rng.random(len(strata))
        strata = strata.sort_values(["count", "_rand"], ascending=[False, True])
        for stratum in strata["shadow_bucket"].tolist():
            if sum(len(part) for part in selected_parts) >= needed:
                break
            group = remaining[remaining["shadow_bucket"] == stratum].sort_values("_rand")
            if not group.empty:
                selected_parts.append(group.head(1))

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else remaining.iloc[0:0].copy()
    if len(selected_df) < needed:
        selected_ids = set(selected_df["id"].tolist())
        pool = remaining[~remaining["id"].isin(selected_ids)].copy()
        if not pool.empty:
            stratum_sizes = pool.groupby("shadow_bucket")["id"].transform("count").clip(lower=1)
            weights = 1.0 / stratum_sizes
            fill_n = min(needed - len(selected_df), len(pool))
            fill_df = pool.sample(
                n=fill_n,
                weights=weights,
                random_state=seed,
                replace=False,
            )
            selected_df = pd.concat([selected_df, fill_df], ignore_index=True)

    selected_df = selected_df.drop_duplicates(subset=["id"]).head(needed)
    if selected_df.empty:
        conn.close()
        return {
            "game_date": game_date,
            "available": int(len(df)),
            "shadow_sample_size": int(len(existing)),
            "selected_now": 0,
            "status": "selection_failed",
        }

    selected_at = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        UPDATE daily_board
        SET is_shadow_sample = 1,
            shadow_bucket = ?,
            shadow_selected_at = ?
        WHERE id = ?
        """,
        [
            (row["shadow_bucket"], selected_at, int(row["id"]))
            for _, row in selected_df.iterrows()
        ],
    )
    conn.commit()
    conn.close()

    return {
        "game_date": game_date,
        "available": int(len(df)),
        "shadow_sample_size": int(len(existing) + len(selected_df)),
        "selected_now": int(len(selected_df)),
        "status": "selected",
    }


def get_shadow_sample_stats(days: int = 30) -> Dict:
    """Return QA metrics for the stratified random shadow sample."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT date, player_name, team, prop_type, line, direction, confidence,
               grade, projection, actual_stat, outcome, shadow_bucket, was_bet
        FROM daily_board
        WHERE date >= date('now', ? || ' days')
          AND is_shadow_sample = 1
        ORDER BY date DESC, id DESC
        """,
        conn,
        params=(f"-{days}",),
    )
    conn.close()

    if df.empty:
        return {
            "total_sampled": 0,
            "graded": 0,
            "pending": 0,
            "accuracy": None,
            "by_prop": {},
            "by_confidence_bucket": [],
            "recent_rows": [],
        }

    graded = df[df["outcome"].notna()].copy()
    graded["confidence_bucket"] = graded["confidence"].apply(_confidence_bucket)
    accuracy = float(graded["outcome"].mean()) if not graded.empty else None

    by_prop = {}
    if not graded.empty:
        for prop_type, subset in graded.groupby("prop_type"):
            by_prop[prop_type] = {
                "count": int(len(subset)),
                "accuracy": float(subset["outcome"].mean()),
            }

    by_conf = []
    if not graded.empty:
        for bucket, subset in graded.groupby("confidence_bucket"):
            by_conf.append({
                "bucket": bucket,
                "count": int(len(subset)),
                "predicted_mean": float(subset["confidence"].mean()),
                "actual_rate": float(subset["outcome"].mean()),
            })
        by_conf.sort(key=lambda row: row["bucket"])

    recent_cols = ["date", "player_name", "team", "prop_type", "line", "direction", "confidence", "actual_stat", "outcome", "was_bet"]
    recent_rows = df[recent_cols].head(25).to_dict("records")

    return {
        "total_sampled": int(len(df)),
        "graded": int(len(graded)),
        "pending": int(len(df) - len(graded)),
        "accuracy": accuracy,
        "by_prop": by_prop,
        "by_confidence_bucket": by_conf,
        "recent_rows": recent_rows,
    }


def classify_edge_source(prediction: Dict, sharp: Dict = None) -> str:
    """Classify the primary source of edge for a pick.

    Edge sources:
      stale_line — sharp books moved but PP hasn't
      projection_disagreement — model proj diverges from market
      lineup_shock — unexpected lineup change (pitcher or batter)
      weather_mismatch — weather not reflected in lines
      matchup_edge — Log5 matchup creates favorable rate
      umpire_edge — umpire zone shape creates favorable K/BB rate
      combined — multiple small edges
    """
    sources = []

    # Check for sharp vs PP divergence (stale line)
    if sharp:
        sharp_prob = sharp.get("sharp_implied_prob", 0.5)
        pp_implied = 0.5  # PP lines are roughly 50/50
        if abs(sharp_prob - pp_implied) > 0.03:
            sources.append(("stale_line", abs(sharp_prob - pp_implied)))

    # Projection disagreement
    proj = prediction.get("projection", 0)
    line = prediction.get("line", 0)
    if line > 0 and abs(proj - line) / line > 0.08:
        sources.append(("projection_disagreement", abs(proj - line) / line))

    # Weather mismatch
    wx_mult = prediction.get("weather_mult", 1.0)
    if abs(wx_mult - 1.0) > 0.03:
        sources.append(("weather_mismatch", abs(wx_mult - 1.0)))

    # Matchup edge (Log5 or PA multiplier)
    if prediction.get("pa_multiplier", 1.0) > 1.05:
        sources.append(("matchup_edge", prediction["pa_multiplier"] - 1.0))

    # Lineup shock — late lineup change or unexpected starter
    if prediction.get("lineup_changed", False):
        sources.append(("lineup_shock", 0.05))

    # Umpire edge — umpire zone notably favors the pick
    ump_k_adj = prediction.get("ump_k_adjustment", 0)
    if abs(ump_k_adj) > 0.3:
        sources.append(("umpire_edge", abs(ump_k_adj) / 2.0))

    if not sources:
        return "combined"

    # Return the strongest edge source
    sources.sort(key=lambda x: x[1], reverse=True)
    return sources[0][0]


def classify_all_edge_sources(prediction: Dict, sharp: Dict = None) -> List[Dict]:
    """Return ALL edge sources with their magnitudes (not just the primary).

    Useful for understanding which edges compound on a single pick,
    and for tracking CLV by source over time.

    Returns:
        List of dicts with 'source' and 'magnitude' keys, sorted by magnitude
    """
    sources = []

    if sharp:
        sharp_prob = sharp.get("sharp_implied_prob", 0.5)
        if abs(sharp_prob - 0.5) > 0.03:
            sources.append({"source": "stale_line", "magnitude": round(abs(sharp_prob - 0.5), 4)})

    proj = prediction.get("projection", 0)
    line = prediction.get("line", 0)
    if line > 0 and abs(proj - line) / line > 0.08:
        sources.append({"source": "projection_disagreement", "magnitude": round(abs(proj - line) / line, 4)})

    wx_mult = prediction.get("weather_mult", 1.0)
    if abs(wx_mult - 1.0) > 0.03:
        sources.append({"source": "weather_mismatch", "magnitude": round(abs(wx_mult - 1.0), 4)})

    if prediction.get("pa_multiplier", 1.0) > 1.05:
        sources.append({"source": "matchup_edge", "magnitude": round(prediction["pa_multiplier"] - 1.0, 4)})

    if prediction.get("lineup_changed", False):
        sources.append({"source": "lineup_shock", "magnitude": 0.05})

    ump_k_adj = prediction.get("ump_k_adjustment", 0)
    if abs(ump_k_adj) > 0.3:
        sources.append({"source": "umpire_edge", "magnitude": round(abs(ump_k_adj) / 2.0, 4)})

    sources.sort(key=lambda x: x["magnitude"], reverse=True)
    return sources if sources else [{"source": "combined", "magnitude": 0.0}]


# Initialize table on import
init_board_table()
