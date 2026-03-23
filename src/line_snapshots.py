"""
PrizePicks Line Snapshot Storage

Stores timestamped PrizePicks line history for:
1. CLV calculation (compare bet-time line to closing line)
2. Stale line detection (flag lines that haven't moved when sharps have)
3. Line movement tracking (detect steam moves or reverse line movement)

Usage:
    from src.line_snapshots import snapshot_pp_lines, get_line_history, detect_stale_lines
"""

import sqlite3
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List

from src.database import get_connection


def _resolve_snapshot_game_date(row: pd.Series, fallback_date: str | None) -> str:
    """Resolve a snapshot row's game date from row data before falling back."""
    explicit = row.get("game_date")
    if explicit:
        return str(explicit)

    start_time = row.get("start_time")
    if start_time is not None and str(start_time) not in ("", "NaT"):
        try:
            ts = pd.Timestamp(start_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.tz_convert("America/Los_Angeles").date().isoformat()
        except Exception:
            pass

    return fallback_date or date.today().isoformat()


def init_line_snapshots_table():
    """Create the line_snapshots table."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS line_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_time TEXT NOT NULL,
            game_date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            pp_line REAL NOT NULL,
            line_type TEXT DEFAULT 'standard',
            start_time TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_snap_player
            ON line_snapshots(player_name, stat_type, game_date);
        CREATE INDEX IF NOT EXISTS idx_snap_time
            ON line_snapshots(snapshot_time);
        CREATE INDEX IF NOT EXISTS idx_snap_date
            ON line_snapshots(game_date);
    """)
    conn.commit()
    conn.close()


def snapshot_pp_lines(pp_lines_df: pd.DataFrame, game_date: str = None):
    """Take a timestamped snapshot of all current PrizePicks lines.

    Call this every time PP lines are fetched. The snapshot_time lets us
    track movement over the course of a day.

    Args:
        pp_lines_df: DataFrame with columns: player_name, stat_type, line,
                     and optionally line_type, start_time
        game_date: Game date (default today)
    """
    if pp_lines_df is None or pp_lines_df.empty:
        return 0

    if game_date is None:
        game_date = date.today().isoformat()

    now = datetime.now().isoformat()
    conn = get_connection()

    rows_inserted = 0
    for _, row in pp_lines_df.iterrows():
        player = row.get("player_name", "")
        stat = row.get("stat_internal") or row.get("stat_type", "")
        line = row.get("line")
        row_game_date = _resolve_snapshot_game_date(row, game_date)

        if not player or not stat or line is None:
            continue

        # Check if this exact line already exists in the last 15 minutes
        # to avoid flooding the DB with duplicate snapshots
        existing = conn.execute("""
            SELECT id FROM line_snapshots
            WHERE player_name = ? AND stat_type = ? AND game_date = ?
            AND pp_line = ? AND snapshot_time > datetime(?, '-15 minutes')
        """, (player, stat, row_game_date, line, now)).fetchone()

        if existing:
            continue

        conn.execute("""
            INSERT INTO line_snapshots
            (snapshot_time, game_date, player_name, stat_type, pp_line,
             line_type, start_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            now, row_game_date, player, stat, line,
            row.get("line_type", "standard"),
            row.get("start_time", ""),
        ))
        rows_inserted += 1

    conn.commit()
    conn.close()
    return rows_inserted


def get_line_history(player_name: str, stat_type: str,
                     game_date: str = None) -> List[Dict]:
    """Get the full line movement history for a player/prop on a given date.

    Args:
        player_name: Player name
        stat_type: Stat type
        game_date: Game date (default today)

    Returns:
        List of dicts with snapshot_time and pp_line, ordered chronologically
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    rows = conn.execute("""
        SELECT snapshot_time, pp_line, line_type
        FROM line_snapshots
        WHERE player_name = ? AND stat_type = ? AND game_date = ?
        ORDER BY snapshot_time ASC
    """, (player_name, stat_type, game_date)).fetchall()
    conn.close()

    return [
        {"snapshot_time": r[0], "pp_line": r[1], "line_type": r[2]}
        for r in rows
    ]


def get_opening_line(player_name: str, stat_type: str,
                     game_date: str = None) -> Optional[float]:
    """Get the earliest recorded PP line for a player/prop (the opening line).

    Returns:
        The opening line value, or None if no snapshots exist
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    row = conn.execute("""
        SELECT pp_line FROM line_snapshots
        WHERE player_name = ? AND stat_type = ? AND game_date = ?
        ORDER BY snapshot_time ASC LIMIT 1
    """, (player_name, stat_type, game_date)).fetchone()
    conn.close()

    return row[0] if row else None


def get_closing_line(player_name: str, stat_type: str,
                     game_date: str = None) -> Optional[float]:
    """Get the most recent PP line for a player/prop (the closing line).

    Returns:
        The closing line value, or None if no snapshots exist
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    row = conn.execute("""
        SELECT pp_line FROM line_snapshots
        WHERE player_name = ? AND stat_type = ? AND game_date = ?
        ORDER BY snapshot_time DESC LIMIT 1
    """, (player_name, stat_type, game_date)).fetchone()
    conn.close()

    return row[0] if row else None


def detect_stale_lines(pp_lines_df: pd.DataFrame,
                       sharp_lines: list,
                       hours_threshold: float = 2.0) -> List[Dict]:
    """Detect PrizePicks lines that haven't moved even though sharp books have.

    A stale line is one where:
    1. Sharp books have moved the line (sharp_line != pp_line)
    2. The PP line hasn't changed in the last N hours

    These are prime candidates for +EV bets — PP is slow to adjust.

    Args:
        pp_lines_df: Current PP lines
        sharp_lines: Current sharp book lines (from find_ev_edges format)
        hours_threshold: How long a line must be unchanged to be "stale"

    Returns:
        List of stale line alerts with player, prop, staleness info
    """
    stale_alerts = []
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(hours=hours_threshold)).isoformat()

    for sharp in sharp_lines:
        player = sharp.get("player", "")
        market = sharp.get("market", "")
        sharp_line_val = sharp.get("line", 0)

        # Find matching PP line
        matching = pp_lines_df[
            pp_lines_df["player_name"].str.contains(
                player.split()[-1], case=False, na=False
            )
        ]
        if matching.empty:
            continue

        for _, pp_row in matching.iterrows():
            pp_line = pp_row["line"]
            stat_type = pp_row.get("stat_internal", pp_row.get("stat_type", ""))
            row_game_date = _resolve_snapshot_game_date(pp_row, None)

            if abs(pp_line - sharp_line_val) < 0.25:
                continue  # Lines match, not stale

            # Check if PP line has moved since cutoff
            movement = conn.execute("""
                SELECT COUNT(DISTINCT pp_line) as distinct_lines,
                       MIN(pp_line) as min_line,
                       MAX(pp_line) as max_line
                FROM line_snapshots
                WHERE player_name = ? AND stat_type = ? AND game_date = ?
                AND snapshot_time >= ?
            """, (pp_row["player_name"], stat_type, row_game_date, cutoff)).fetchone()

            if movement and movement[0] <= 1:
                # PP line hasn't moved — it's stale
                stale_alerts.append({
                    "player_name": pp_row["player_name"],
                    "stat_type": stat_type,
                    "pp_line": pp_line,
                    "sharp_line": sharp_line_val,
                    "line_diff": round(sharp_line_val - pp_line, 2),
                    "hours_unchanged": hours_threshold,
                    "edge_direction": "MORE" if pp_line < sharp_line_val else "LESS",
                })

    conn.close()
    return stale_alerts


def get_line_movement_summary(game_date: str = None) -> Dict:
    """Get a summary of all line movements for a date.

    Returns:
        Dict with total_props, moved_count, biggest_movers, etc.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT player_name, stat_type,
               MIN(pp_line) as min_line,
               MAX(pp_line) as max_line,
               COUNT(DISTINCT pp_line) as distinct_lines,
               COUNT(*) as snapshot_count,
               MIN(snapshot_time) as first_seen,
               MAX(snapshot_time) as last_seen
        FROM line_snapshots
        WHERE game_date = ?
        GROUP BY player_name, stat_type
    """, conn, params=(game_date,))
    conn.close()

    if df.empty:
        return {"total_props": 0, "moved_count": 0, "movers": []}

    df["movement"] = df["max_line"] - df["min_line"]
    movers = df[df["distinct_lines"] > 1].sort_values("movement", ascending=False)

    return {
        "total_props": len(df),
        "moved_count": len(movers),
        "unchanged_count": len(df) - len(movers),
        "biggest_movers": movers.head(10).to_dict("records") if not movers.empty else [],
        "avg_snapshots_per_prop": round(df["snapshot_count"].mean(), 1),
    }


# Initialize table on import
init_line_snapshots_table()
