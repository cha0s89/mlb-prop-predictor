"""
PrizePicks Slip Tracker
Track multi-pick entries (slips), calculate payouts, and manage P&L.

Official PrizePicks payout structure (March 2026):
  Power Play (must be perfect):
    2-Pick = 3x, 3-Pick = 6x, 4-Pick = 10x, 5-Pick = 20x, 6-Pick = 25x
  Flex Play (partial payouts):
    3-Pick: 3/3=3x, 2/3=1x
    4-Pick: 4/4=6x, 3/4=1.5x
    5-Pick: 5/5=10x, 4/5=2x, 3/5=0.4x
    6-Pick: 6/6=12.5x, 5/6=2x, 4/6=0.4x

  Ties: revert payout down one level (not removed like DNP)
  2-Pick Power special: 1 correct + 1 tie = 1.5x; 1 loss + 1 tie = loss
  DNP/Reboot: pick removed, lineup reverts (e.g., 3-pick Power → 2-pick Power)
  2-Pick Power DNP → refund
  If DNP removals leave all remaining picks on same team → refund (ineligible)

  Note: Some payouts may differ for discounted projections, same-game combos,
  specific leagues, and Demon/Goblin projections. In-app detail is authoritative.
"""

import sqlite3
import math
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from src.database import get_connection, grade_prediction


# ── Official PrizePicks Payout Tables (March 2026) ──
# Source: PrizePicks Help Center, verified March 2026
PAYOUTS = {
    # Power Play (must be perfect)
    "2_power": {2: 3.0, 1: 0, 0: 0},
    "3_power": {3: 6.0, 2: 0, 1: 0, 0: 0},
    "4_power": {4: 10.0, 3: 0, 2: 0, 1: 0, 0: 0},
    "5_power": {5: 20.0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "6_power": {6: 25.0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    # Flex Play (partial payouts)
    "3_flex": {3: 3.0, 2: 1.0, 1: 0, 0: 0},
    "4_flex": {4: 6.0, 3: 1.5, 2: 0, 1: 0, 0: 0},
    "5_flex": {5: 10.0, 4: 2.0, 3: 0.4, 2: 0, 1: 0, 0: 0},
    "6_flex": {6: 12.5, 5: 2.0, 4: 0.4, 3: 0, 2: 0, 1: 0, 0: 0},
}

# Special tie payout for 2-pick Power
TIE_SPECIAL = {
    "2_power": {(1, 1): 1.5},  # 1 correct + 1 tie = 1.5x
}

# Goblin/Demon payout tables — reduced payouts for promo lines
# Source: PrizePicks in-app, approximate values (March 2026)
# Goblin lines are "easy" (e.g., HR 0.5 over) with much lower payouts
# Demon lines are "hard" with higher payouts
GOBLIN_PAYOUTS = {
    "2_power": {2: 1.5, 1: 0, 0: 0},
    "3_power": {3: 2.5, 2: 0, 1: 0, 0: 0},
    "4_power": {4: 5.0, 3: 0, 2: 0, 1: 0, 0: 0},
    "5_power": {5: 10.0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "6_power": {6: 12.5, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "3_flex": {3: 1.5, 2: 0.5, 1: 0, 0: 0},
    "4_flex": {4: 3.0, 3: 0.75, 2: 0, 1: 0, 0: 0},
    "5_flex": {5: 5.0, 4: 1.0, 3: 0.2, 2: 0, 1: 0, 0: 0},
    "6_flex": {6: 6.25, 5: 1.0, 4: 0.2, 3: 0, 2: 0, 1: 0, 0: 0},
}

DEMON_PAYOUTS = {
    "2_power": {2: 6.0, 1: 0, 0: 0},
    "3_power": {3: 12.0, 2: 0, 1: 0, 0: 0},
    "4_power": {4: 20.0, 3: 0, 2: 0, 1: 0, 0: 0},
    "5_power": {5: 40.0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "6_power": {6: 50.0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    "3_flex": {3: 6.0, 2: 2.0, 1: 0, 0: 0},
    "4_flex": {4: 12.0, 3: 3.0, 2: 0, 1: 0, 0: 0},
    "5_flex": {5: 20.0, 4: 4.0, 3: 0.8, 2: 0, 1: 0, 0: 0},
    "6_flex": {6: 25.0, 5: 4.0, 4: 0.8, 3: 0, 2: 0, 1: 0, 0: 0},
}


def get_payout_table(entry_type: str, line_type: str = "standard") -> dict:
    """Get the appropriate payout table based on line type.

    Args:
        entry_type: e.g. '5_flex', '3_power'
        line_type: 'standard', 'promo' (goblin), 'demon', etc.

    Returns:
        Payout dict mapping num_wins -> multiplier
    """
    if line_type in ("promo", "goblin", "discounted", "flash_sale"):
        return GOBLIN_PAYOUTS.get(entry_type, PAYOUTS.get(entry_type, {}))
    elif line_type == "demon":
        return DEMON_PAYOUTS.get(entry_type, PAYOUTS.get(entry_type, {}))
    return PAYOUTS.get(entry_type, {})


# Per-leg iid break-even probabilities (exact, from research report)
BREAKEVEN = {
    "2_power": 0.57735,
    "3_power": 0.55032,
    "4_power": 0.56234,
    "5_power": 0.54928,
    "6_power": 0.58480,
    "3_flex":  0.57735,
    "4_flex":  0.55032,
    "5_flex":  0.54253,
    "6_flex":  0.58984,
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

        # PrizePicks tie/push handling (March 2026 rules):
        # Ties revert payout down one level (not removed like DNP).
        # Special: 2-Pick Power with 1 correct + 1 tie = 1.5x
        # Special: 2-Pick Power with 1 loss + 1 tie = loss
        is_power = "power" in entry_type

        if pushes > 0:
            # Check special tie cases first
            special = TIE_SPECIAL.get(entry_type, {})
            special_key = (wins, pushes)
            if special_key in special:
                payout_mult = special[special_key]
            else:
                # Revert down: ties reduce effective lineup size
                effective_type = f"{effective_picks}_{'power' if is_power else 'flex'}"
                payout_table = PAYOUTS.get(effective_type, PAYOUTS.get(entry_type, {}))
                payout_mult = payout_table.get(wins, 0)
        else:
            payout_table = PAYOUTS.get(entry_type, {})
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
