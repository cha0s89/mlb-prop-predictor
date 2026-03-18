"""
PrizePicks Slip Tracker
Track multi-pick entries (slips), calculate payouts, and manage P&L.

PrizePicks payout structure:
  5-pick Flex: 10x (all 5), 2x (4/5), 0.4x (3/5) — break-even ~54.2%
  6-pick Flex: 25x (all 6), 5x (5/6), 1.5x (4/6), 0.25x (3/6) — break-even ~52.9%
  3-pick Power: 5x (all 3) — break-even ~59.8%
  2-pick Power: 3x (all 2) — break-even ~57.7%
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from src.database import get_connection, grade_prediction


# ── Payout tables ──
PAYOUTS = {
    "5_flex": {5: 10.0, 4: 2.0, 3: 0.4, 2: 0, 1: 0, 0: 0},
    "6_flex": {6: 25.0, 5: 5.0, 4: 1.5, 3: 0.25, 2: 0, 1: 0, 0: 0},
    "3_power": {3: 5.0, 2: 0, 1: 0, 0: 0},
    "2_power": {2: 3.0, 1: 0, 0: 0},
    "4_flex": {4: 5.0, 3: 1.5, 2: 0.25, 1: 0, 0: 0},
}

BREAKEVEN = {
    "5_flex": 0.542, "6_flex": 0.529, "3_power": 0.598,
    "2_power": 0.577, "4_flex": 0.555,
}


def init_slips_table():
    """Create slips and slip_picks tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS slips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            game_date TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            entry_amount REAL NOT NULL DEFAULT 5.0,
            num_picks INTEGER NOT NULL,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            pushes INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            payout_mult REAL DEFAULT 0,
            payout_amount REAL DEFAULT 0,
            net_profit REAL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS slip_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slip_id INTEGER NOT NULL,
            prediction_id INTEGER,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line REAL NOT NULL,
            pick TEXT NOT NULL,
            result TEXT,
            actual_result REAL,
            FOREIGN KEY (slip_id) REFERENCES slips(id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_slips_date ON slips(game_date);
        CREATE INDEX IF NOT EXISTS idx_slip_picks_slip ON slip_picks(slip_id);
    """)
    conn.commit()
    conn.close()


def create_slip(game_date: str, entry_type: str, entry_amount: float,
                picks: list[dict], notes: str = "") -> int:
    """
    Create a new slip with picks.

    picks: list of dicts with keys: player_name, stat_type, line, pick, prediction_id (optional)
    Returns: slip ID
    """
    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO slips (game_date, entry_type, entry_amount, num_picks, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (game_date, entry_type, entry_amount, len(picks), notes))
    slip_id = cur.lastrowid

    for pick in picks:
        conn.execute("""
            INSERT INTO slip_picks (slip_id, prediction_id, player_name, stat_type, line, pick)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            slip_id,
            pick.get("prediction_id"),
            pick["player_name"],
            pick["stat_type"],
            pick["line"],
            pick["pick"],
        ))

    conn.commit()
    conn.close()
    return slip_id


def grade_slip_pick(slip_pick_id: int, actual_result: float) -> str:
    """Grade a single pick within a slip."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT line, pick FROM slip_picks WHERE id = ?", (slip_pick_id,)
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    line, pick = row
    if actual_result > line:
        result = "W" if pick == "MORE" else "L"
    elif actual_result < line:
        result = "W" if pick == "LESS" else "L"
    else:
        result = "push"

    conn.execute("""
        UPDATE slip_picks SET result = ?, actual_result = ? WHERE id = ?
    """, (result, actual_result, slip_pick_id))
    conn.commit()
    conn.close()

    # Check if all picks graded, then finalize slip
    _try_finalize_slip_for_pick(slip_pick_id)
    return result


def _try_finalize_slip_for_pick(slip_pick_id: int):
    """After grading a pick, check if the whole slip can be finalized."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT slip_id FROM slip_picks WHERE id = ?", (slip_pick_id,)
    )
    row = cur.fetchone()
    if row:
        finalize_slip(row[0])
    conn.close()


def finalize_slip(slip_id: int) -> Optional[dict]:
    """
    Finalize a slip once all picks are graded.
    Calculates payout based on entry type and number of wins.
    Pushes reduce the pick count (PrizePicks rules).
    """
    conn = get_connection()
    picks = pd.read_sql_query(
        "SELECT * FROM slip_picks WHERE slip_id = ?", conn, params=(slip_id,)
    )

    if picks.empty:
        conn.close()
        return None

    # Check if all picks are graded
    if picks["result"].isna().any():
        conn.close()
        return None

    wins = len(picks[picks["result"] == "W"])
    losses = len(picks[picks["result"] == "L"])
    pushes = len(picks[picks["result"] == "push"])

    # PrizePicks: pushes reduce the entry (5-pick flex with 1 push = 4-pick flex)
    effective_picks = wins + losses
    if effective_picks == 0:
        # All pushes
        status = "push"
        payout_mult = 1.0  # Money back
    else:
        # Look up payout
        slip_row = conn.execute(
            "SELECT entry_type, entry_amount FROM slips WHERE id = ?", (slip_id,)
        ).fetchone()
        if not slip_row:
            conn.close()
            return None

        entry_type, entry_amount = slip_row

        # Find the right payout table based on effective picks
        effective_type = f"{effective_picks}_flex"
        payout_table = PAYOUTS.get(entry_type, PAYOUTS.get(effective_type, {}))

        payout_mult = payout_table.get(wins, 0)
        status = "win" if payout_mult > 1 else ("loss" if payout_mult == 0 else "partial")

    slip_info = conn.execute(
        "SELECT entry_amount FROM slips WHERE id = ?", (slip_id,)
    ).fetchone()
    entry_amount = slip_info[0] if slip_info else 5.0

    payout_amount = entry_amount * payout_mult
    net_profit = payout_amount - entry_amount

    conn.execute("""
        UPDATE slips SET wins = ?, losses = ?, pushes = ?, status = ?,
               payout_mult = ?, payout_amount = ?, net_profit = ?
        WHERE id = ?
    """, (wins, losses, pushes, status, payout_mult, payout_amount, net_profit, slip_id))
    conn.commit()
    conn.close()

    return {
        "slip_id": slip_id, "wins": wins, "losses": losses, "pushes": pushes,
        "status": status, "payout_mult": payout_mult,
        "payout_amount": payout_amount, "net_profit": net_profit,
    }


def get_slips(limit: int = 50) -> pd.DataFrame:
    """Get all slips, most recent first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM slips ORDER BY created_at DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_slip_picks(slip_id: int) -> pd.DataFrame:
    """Get all picks for a specific slip."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM slip_picks WHERE slip_id = ?", conn, params=(slip_id,)
    )
    conn.close()
    return df


def get_slip_pnl(days: int = 30) -> dict:
    """Calculate P&L summary for recent slips."""
    conn = get_connection()
    slips = pd.read_sql_query("""
        SELECT * FROM slips
        WHERE status != 'pending'
        AND game_date >= date('now', ? || ' days')
    """, conn, params=(f"-{days}",))
    conn.close()

    if slips.empty:
        return {
            "total_wagered": 0, "total_returned": 0, "net_profit": 0,
            "roi": 0, "slips_won": 0, "slips_lost": 0, "slips_total": 0,
        }

    total_wagered = slips["entry_amount"].sum()
    total_returned = slips["payout_amount"].sum()
    net_profit = total_returned - total_wagered
    roi = (net_profit / total_wagered * 100) if total_wagered > 0 else 0

    return {
        "total_wagered": round(total_wagered, 2),
        "total_returned": round(total_returned, 2),
        "net_profit": round(net_profit, 2),
        "roi": round(roi, 1),
        "slips_won": len(slips[slips["status"] == "win"]),
        "slips_lost": len(slips[slips["status"] == "loss"]),
        "slips_partial": len(slips[slips["status"] == "partial"]),
        "slips_total": len(slips),
    }


# Initialize tables on import
init_slips_table()
