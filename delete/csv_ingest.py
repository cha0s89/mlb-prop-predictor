"""
CSV Ingestion Module
Load and process prop prediction CSVs from external sources.

Supports:
1. RotoWire PrizePicks daily prediction CSVs
2. Generic prop CSVs (player, prop_type, line, side)
3. Custom export from our own predictions for record-keeping

This lets you compare Rotowire's picks against our model's picks
and see who's more accurate over time.
"""

import pandas as pd
import os
from datetime import date, datetime
from pathlib import Path


# RotoWire market name → our internal key mapping
ROTOWIRE_MARKET_MAP = {
    "Hits": "hits",
    "Total Bases": "total_bases",
    "Home Runs": "home_runs",
    "RBIs": "rbis",
    "Runs": "runs",
    "Stolen Bases": "stolen_bases",
    "Pitcher Strikeouts": "pitcher_strikeouts",
    "Strikeouts": "pitcher_strikeouts",
    "Earned Runs": "earned_runs",
    "Earned Runs Allowed": "earned_runs",
    "Pitching Outs": "pitching_outs",
    "Outs Recorded": "pitching_outs",
    "Walks": "walks",
    "Walks Allowed": "walks_allowed",
    "Hits Allowed": "hits_allowed",
    "Batter Strikeouts": "batter_strikeouts",
    "Hits+Runs+RBIs": "hits_runs_rbis",
    "Singles": "singles",
    "Doubles": "doubles",
}

# Generic CSV column aliases
COLUMN_ALIASES = {
    # Player name
    "player": "player_name", "Player": "player_name",
    "player_name": "player_name", "Player Name": "player_name",
    "name": "player_name", "Name": "player_name",
    # Prop type
    "prop_type": "stat_type", "Market Name": "stat_type",
    "market": "stat_type", "Market": "stat_type",
    "Prop": "stat_type", "prop": "stat_type",
    "Stat Type": "stat_type", "stat_type": "stat_type",
    # Line
    "line": "line", "Line": "line",
    "Line Score": "line", "projection": "line",
    "Projection": "line", "Over/Under": "line",
    # Side / lean
    "side": "pick", "Side": "pick",
    "lean": "pick", "Lean": "pick",
    "Pick": "pick", "pick": "pick",
    "Direction": "pick", "direction": "pick",
    # Probability
    "probability": "probability", "Probability": "probability",
    "prob": "probability", "Prob": "probability",
    "Hit Rate": "probability", "Win Prob": "probability",
}


def load_rotowire_csv(filepath: str = None, game_date: date = None) -> pd.DataFrame:
    """
    Load a RotoWire PrizePicks prediction CSV.

    If no filepath given, looks for today's date-stamped file in common locations:
    - rotowire_YYYY-MM-DD.csv
    - RotoWire_PrizePicks_YYYY-MM-DD.csv
    - prizepicks_predictions_YYYY-MM-DD.csv
    """
    if filepath and os.path.exists(filepath):
        return _parse_rotowire(pd.read_csv(filepath))

    # Auto-detect today's file
    game_date = game_date or date.today()
    date_str = game_date.isoformat()

    search_patterns = [
        f"rotowire_{date_str}.csv",
        f"RotoWire_PrizePicks_{date_str}.csv",
        f"prizepicks_predictions_{date_str}.csv",
        f"rotowire_predictions_{date_str}.csv",
    ]

    search_dirs = [".", "data", "downloads", os.path.expanduser("~/Downloads")]

    for dir_path in search_dirs:
        for pattern in search_patterns:
            full_path = os.path.join(dir_path, pattern)
            if os.path.exists(full_path):
                return _parse_rotowire(pd.read_csv(full_path))

    return pd.DataFrame()  # Not found


def _parse_rotowire(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalize a RotoWire CSV into our standard format."""
    if df.empty:
        return df

    # Normalize column names
    df = df.rename(columns=COLUMN_ALIASES)

    # Ensure required columns exist
    required = ["player_name", "stat_type", "line"]
    for col in required:
        if col not in df.columns:
            # Try to find it with different casing
            for orig, mapped in COLUMN_ALIASES.items():
                if mapped == col and orig in df.columns:
                    df = df.rename(columns={orig: col})
                    break

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    # Map market names to internal keys
    if "stat_type" in df.columns:
        df["stat_internal"] = df["stat_type"].map(ROTOWIRE_MARKET_MAP).fillna(
            df["stat_type"].str.lower().str.replace(" ", "_")
        )

    # Normalize pick direction
    if "pick" in df.columns:
        df["pick"] = df["pick"].str.upper().str.strip()
        df["pick"] = df["pick"].replace({
            "OVER": "MORE", "UNDER": "LESS",
            "O": "MORE", "U": "LESS",
            "HIGHER": "MORE", "LOWER": "LESS",
        })

    # Ensure line is numeric
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])

    # Add source tag
    df["source"] = "rotowire"

    return df


def load_generic_csv(filepath: str) -> pd.DataFrame:
    """
    Load any generic prop CSV and normalize it.
    Supports various column naming conventions.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Normalize columns
    df = df.rename(columns=COLUMN_ALIASES)

    # Map market names if present
    if "stat_type" in df.columns:
        df["stat_internal"] = df["stat_type"].map(ROTOWIRE_MARKET_MAP).fillna(
            df["stat_type"].str.lower().str.replace(" ", "_")
        )

    if "pick" in df.columns:
        df["pick"] = df["pick"].str.upper().str.strip()
        df["pick"] = df["pick"].replace({
            "OVER": "MORE", "UNDER": "LESS",
            "O": "MORE", "U": "LESS",
        })

    df["line"] = pd.to_numeric(df.get("line", pd.Series()), errors="coerce")
    df["source"] = "csv_upload"

    return df


def compare_sources(our_picks: pd.DataFrame, external_picks: pd.DataFrame) -> pd.DataFrame:
    """
    Compare our model's picks to an external source (RotoWire, etc.).

    Shows where we agree, disagree, and the confidence difference.
    Agreement between our model and RotoWire = higher confidence.
    Disagreement = worth investigating why.
    """
    if our_picks.empty or external_picks.empty:
        return pd.DataFrame()

    # Merge on player + prop type
    merged = pd.merge(
        our_picks,
        external_picks,
        on=["player_name", "stat_type"],
        how="inner",
        suffixes=("_ours", "_rw"),
    )

    if merged.empty:
        return pd.DataFrame()

    # Compare picks
    if "pick_ours" in merged.columns and "pick_rw" in merged.columns:
        merged["agreement"] = merged["pick_ours"] == merged["pick_rw"]
        merged["status"] = merged["agreement"].map({
            True: "✅ Both agree",
            False: "⚠️ Disagree",
        })

    # Compare lines
    if "line_ours" in merged.columns and "line_rw" in merged.columns:
        merged["line_diff"] = merged["line_ours"] - merged["line_rw"]

    return merged


def export_picks_csv(predictions: list, filepath: str = None) -> str:
    """Export our predictions to CSV for record-keeping."""
    if not predictions:
        return ""

    df = pd.DataFrame(predictions)

    if filepath is None:
        Path("data/exports").mkdir(parents=True, exist_ok=True)
        filepath = f"data/exports/picks_{date.today().isoformat()}.csv"

    export_cols = ["player_name", "team", "stat_type", "line", "projection",
                   "pick", "confidence", "edge", "rating"]
    available = [c for c in export_cols if c in df.columns]

    df[available].to_csv(filepath, index=False)
    return filepath
