"""
Daily Board Logger — Logs EVERY evaluated prop for bias-free learning.

Stores a complete snapshot of every prop the model evaluates each day,
not just the ones selected for slips. This prevents selection bias in
the learning loop and enables CLV tracking.

Tables:
  daily_board — one row per evaluated prop/direction
"""

import sqlite3
import pandas as pd
from datetime import date, datetime
from typing import Optional, List, Dict

from src.database import get_connection, resolve_game_date


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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_board_date ON daily_board(date);
        CREATE INDEX IF NOT EXISTS idx_board_player ON daily_board(player_name);
        CREATE INDEX IF NOT EXISTS idx_board_prop ON daily_board(prop_type);
        CREATE INDEX IF NOT EXISTS idx_board_outcome ON daily_board(outcome);
    """)
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
