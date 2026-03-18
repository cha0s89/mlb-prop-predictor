"""
Pick Slip Tracker
Group individual picks into PrizePicks entries (2-6 pick slips).
Track results at the SLIP level, not just individual legs.

This matters because:
- A 6-pick flex that goes 5/6 still pays 2x
- A 6-pick flex that goes 4/6 pays 0.4x
- Tracking individual leg accuracy is useful for model tuning
- Tracking SLIP results is what tells you if you're making money
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import json


DB_PATH = Path("data/predictions.db")


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_slips_table():
    """Create the slips table if it doesn't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS slips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            game_date TEXT NOT NULL,
            slip_type TEXT NOT NULL,          -- '2-pick', '3-pick', '4-pick', '5-pick', '6-pick'
            entry_mode TEXT DEFAULT 'flex',   -- 'flex' or 'power'
            entry_amount REAL DEFAULT 0,      -- how much you wagered
            pick_ids TEXT NOT NULL,           -- JSON array of prediction IDs
            picks_summary TEXT,              -- human-readable summary
            total_picks INTEGER NOT NULL,
            correct_picks INTEGER,           -- filled after grading
            payout_multiplier REAL,          -- what PrizePicks pays
            profit REAL,                     -- actual profit/loss
            result TEXT,                     -- 'win', 'partial', 'loss'
            graded_at TEXT,
            notes TEXT                       -- user notes
        );

        CREATE INDEX IF NOT EXISTS idx_slips_date ON slips(game_date);
        CREATE INDEX IF NOT EXISTS idx_slips_result ON slips(result);
    """)
    conn.commit()
    conn.close()


# PrizePicks payout tables
FLEX_PAYOUTS = {
    2: {2: 3.0, 1: 1.5, 0: 0},
    3: {3: 5.0, 2: 1.5, 1: 0, 0: 0},
    4: {4: 10.0, 3: 2.0, 2: 0.4, 1: 0, 0: 0},
    5: {5: 10.0, 4: 2.0, 3: 0.4, 2: 0, 1: 0, 0: 0},
    6: {6: 25.0, 5: 2.0, 4: 0.4, 3: 0, 2: 0, 1: 0, 0: 0},
}

POWER_PAYOUTS = {
    2: {2: 3.0, 1: 0, 0: 0},
    3: {3: 5.0, 2: 0, 1: 0, 0: 0},
    4: {4: 10.0, 3: 0, 2: 0, 1: 0, 0: 0},
    5: {5: 20.0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    6: {6: 25.0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
}


def save_slip(pick_ids: list, picks_data: list, slip_type: str = None,
              entry_mode: str = "flex", entry_amount: float = 0,
              game_date: str = None, notes: str = "") -> int:
    """
    Save a pick slip (grouped entry).

    Args:
        pick_ids: list of prediction IDs from the predictions table
        picks_data: list of dicts with pick details for summary
        slip_type: '2-pick', '3-pick', etc. Auto-detected if None
        entry_mode: 'flex' or 'power'
        entry_amount: wager amount in dollars
        game_date: date string
        notes: optional user notes

    Returns: slip ID
    """
    total = len(pick_ids)
    if total < 2 or total > 6:
        raise ValueError(f"PrizePicks requires 2-6 picks. Got {total}.")

    if slip_type is None:
        slip_type = f"{total}-pick"

    # Build summary string
    summary_parts = []
    for p in picks_data:
        name = p.get("player_name", "Unknown")
        stat = p.get("stat_type", "")
        line = p.get("line", 0)
        pick = p.get("pick", "")
        summary_parts.append(f"{name} {stat} {pick} {line}")
    summary = " | ".join(summary_parts)

    game_date = game_date or date.today().isoformat()

    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO slips
        (game_date, slip_type, entry_mode, entry_amount, pick_ids,
         picks_summary, total_picks, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_date, slip_type, entry_mode, entry_amount,
        json.dumps(pick_ids), summary, total, notes,
    ))
    slip_id = cur.lastrowid
    conn.commit()
    conn.close()
    return slip_id


def grade_slip(slip_id: int) -> dict:
    """
    Grade a slip based on its individual pick results.
    Pulls results from the predictions table for each pick in the slip.
    """
    conn = get_connection()

    # Get slip info
    cur = conn.execute("SELECT * FROM slips WHERE id = ?", (int(slip_id),))
    columns = [desc[0] for desc in cur.description]
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"error": "Slip not found"}

    slip = dict(zip(columns, row))
    pick_ids = json.loads(slip["pick_ids"])
    total = slip["total_picks"]
    entry_mode = slip.get("entry_mode", "flex")
    entry_amount = slip.get("entry_amount", 0)

    # Check each pick's result
    correct = 0
    ungraded = 0
    for pid in pick_ids:
        cur2 = conn.execute("SELECT result FROM predictions WHERE id = ?", (int(pid),))
        r = cur2.fetchone()
        if r and r[0] == "W":
            correct += 1
        elif r and r[0] == "push":
            total -= 1  # Pushes reduce the slip size
        elif r is None or r[0] is None:
            ungraded += 1

    if ungraded > 0:
        conn.close()
        return {
            "status": "incomplete",
            "correct": correct,
            "ungraded": ungraded,
            "total": total,
            "message": f"{ungraded} picks not yet graded",
        }

    # Calculate payout
    payouts = FLEX_PAYOUTS if entry_mode == "flex" else POWER_PAYOUTS
    payout_table = payouts.get(total, {})
    multiplier = payout_table.get(correct, 0)
    profit = (entry_amount * multiplier) - entry_amount if entry_amount > 0 else 0

    if correct == total:
        result = "win"
    elif multiplier > 1.0:
        result = "partial_win"
    elif multiplier > 0:
        result = "partial_loss"  # Got money back but less than wagered
    else:
        result = "loss"

    # Update slip
    conn.execute("""
        UPDATE slips
        SET correct_picks = ?, payout_multiplier = ?, profit = ?,
            result = ?, graded_at = datetime('now')
        WHERE id = ?
    """, (correct, multiplier, profit, result, int(slip_id)))
    conn.commit()
    conn.close()

    return {
        "status": "graded",
        "slip_id": slip_id,
        "correct": correct,
        "total": total,
        "multiplier": multiplier,
        "entry_amount": entry_amount,
        "payout": entry_amount * multiplier if entry_amount > 0 else 0,
        "profit": profit,
        "result": result,
    }


def get_all_slips(limit: int = 100) -> pd.DataFrame:
    """Get all slips, most recent first."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM slips ORDER BY created_at DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_slip_stats() -> dict:
    """Calculate slip-level performance stats — THIS IS YOUR P&L."""
    conn = get_connection()
    graded = pd.read_sql_query(
        "SELECT * FROM slips WHERE result IS NOT NULL", conn
    )
    conn.close()

    if graded.empty:
        return {"total_slips": 0}

    total = len(graded)
    wins = len(graded[graded["result"] == "win"])
    partial_wins = len(graded[graded["result"] == "partial_win"])
    losses = len(graded[graded["result"] == "loss"])

    total_wagered = graded["entry_amount"].sum()
    total_profit = graded["profit"].sum()
    roi = total_profit / total_wagered * 100 if total_wagered > 0 else 0

    # By slip type
    by_type = {}
    for st in graded["slip_type"].unique():
        subset = graded[graded["slip_type"] == st]
        by_type[st] = {
            "count": len(subset),
            "wins": len(subset[subset["result"] == "win"]),
            "partial": len(subset[subset["result"].isin(["partial_win", "partial_loss"])]),
            "losses": len(subset[subset["result"] == "loss"]),
            "profit": round(subset["profit"].sum(), 2),
            "roi": round(subset["profit"].sum() / subset["entry_amount"].sum() * 100, 1) if subset["entry_amount"].sum() > 0 else 0,
        }

    # Average correct picks per slip
    avg_correct = graded["correct_picks"].mean()

    return {
        "total_slips": total,
        "wins": wins,
        "partial_wins": partial_wins,
        "losses": losses,
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 1),
        "avg_correct": round(avg_correct, 1) if pd.notna(avg_correct) else 0,
        "by_type": by_type,
    }


def get_ungraded_slips() -> pd.DataFrame:
    """Get slips that haven't been fully graded."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM slips WHERE result IS NULL ORDER BY game_date DESC",
        conn,
    )
    conn.close()
    return df


# Initialize on import
init_slips_table()

