"""
Closing Line Value (CLV) Tracker

Tracks whether our bets beat the closing line — the single best predictor
of long-term profitability in sports betting. Positive CLV means we're
getting better prices than the market settles at.

Usage:
    from src.clv import record_opening_line, record_closing_line, compute_clv_stats

The idea: capture the line at the time of bet placement (opening), then
capture the line at game time (closing). CLV = opening_edge - closing_edge.
"""

import sqlite3
import pandas as pd
from datetime import date, datetime
from typing import Optional, Dict, List

from src.database import get_connection


def init_clv_table():
    """Create the CLV tracking table."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            prop_type TEXT NOT NULL,
            direction TEXT NOT NULL,
            pp_line REAL,
            bet_probability REAL,
            closing_probability REAL,
            clv REAL,
            edge_source TEXT,
            outcome INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_clv_date ON clv_tracking(date);
        CREATE INDEX IF NOT EXISTS idx_clv_source ON clv_tracking(edge_source);
    """)
    conn.commit()
    conn.close()


def record_opening_line(player_name: str, prop_type: str, direction: str,
                        pp_line: float, bet_probability: float,
                        edge_source: str = "combined",
                        game_date: str = None) -> int:
    """Record a bet's opening line for future CLV calculation.

    Args:
        player_name: Player name
        prop_type: Internal stat type
        direction: MORE or LESS
        pp_line: PrizePicks line at time of bet
        bet_probability: Model's estimated probability at bet time
        edge_source: Why this edge exists (stale_line, projection, etc.)
        game_date: Game date (default today)

    Returns:
        Row ID
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO clv_tracking
        (date, player_name, prop_type, direction, pp_line,
         bet_probability, edge_source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (game_date, player_name, prop_type, direction,
          pp_line, bet_probability, edge_source))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def record_closing_line(player_name: str, prop_type: str,
                        closing_probability: float,
                        game_date: str = None):
    """Record the closing probability and compute CLV.

    Call this near game time to capture the market's final assessment.
    CLV = bet_probability - closing_probability (positive = beat the close).
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    cur = conn.execute("""
        SELECT id, bet_probability FROM clv_tracking
        WHERE player_name = ? AND prop_type = ? AND date = ?
        AND closing_probability IS NULL
        ORDER BY id DESC LIMIT 1
    """, (player_name, prop_type, game_date))
    row = cur.fetchone()

    if row:
        row_id, bet_prob = row
        clv = bet_prob - closing_probability
        conn.execute("""
            UPDATE clv_tracking
            SET closing_probability = ?, clv = ?, updated_at = ?
            WHERE id = ?
        """, (closing_probability, clv, datetime.now().isoformat(), row_id))
        conn.commit()

    conn.close()


def record_outcome(player_name: str, prop_type: str, outcome: int,
                   game_date: str = None):
    """Record whether the bet won or lost (1/0)."""
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    conn.execute("""
        UPDATE clv_tracking SET outcome = ?
        WHERE player_name = ? AND prop_type = ? AND date = ?
    """, (outcome, player_name, prop_type, game_date))
    conn.commit()
    conn.close()


def compute_clv_stats(days: int = 30) -> Dict:
    """Compute CLV statistics for the last N days.

    Returns:
        Dict with overall CLV, CLV by edge source, CLV by prop type,
        and correlation between CLV and win rate.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM clv_tracking
        WHERE date >= date('now', ? || ' days')
        AND clv IS NOT NULL
    """, conn, params=(f"-{days}",))
    conn.close()

    if df.empty:
        return {"total_bets": 0, "mean_clv": 0, "by_source": {}, "by_prop": {}}

    result = {
        "total_bets": len(df),
        "mean_clv": round(df["clv"].mean(), 4),
        "median_clv": round(df["clv"].median(), 4),
        "positive_clv_rate": round(len(df[df["clv"] > 0]) / len(df), 4),
    }

    # CLV by edge source
    result["by_source"] = {}
    for source, sdf in df.groupby("edge_source"):
        graded = sdf[sdf["outcome"].notna()]
        result["by_source"][source] = {
            "count": len(sdf),
            "mean_clv": round(sdf["clv"].mean(), 4),
            "win_rate": round(len(graded[graded["outcome"] == 1]) / max(len(graded), 1), 4),
            "positive_clv_rate": round(len(sdf[sdf["clv"] > 0]) / len(sdf), 4),
        }

    # CLV by prop type
    result["by_prop"] = {}
    for prop, pdf in df.groupby("prop_type"):
        graded = pdf[pdf["outcome"].notna()]
        result["by_prop"][prop] = {
            "count": len(pdf),
            "mean_clv": round(pdf["clv"].mean(), 4),
            "win_rate": round(len(graded[graded["outcome"] == 1]) / max(len(graded), 1), 4),
        }

    # Overall: does positive CLV correlate with wins?
    graded = df[df["outcome"].notna()]
    if len(graded) >= 10:
        pos_clv = graded[graded["clv"] > 0]
        neg_clv = graded[graded["clv"] <= 0]
        result["clv_correlation"] = {
            "positive_clv_win_rate": round(
                len(pos_clv[pos_clv["outcome"] == 1]) / max(len(pos_clv), 1), 4
            ),
            "negative_clv_win_rate": round(
                len(neg_clv[neg_clv["outcome"] == 1]) / max(len(neg_clv), 1), 4
            ),
        }

    return result


# Initialize table on import
init_clv_table()
