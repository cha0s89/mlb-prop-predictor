"""
Sharp Odds Module
Pulls player prop odds from sharp sportsbooks via The Odds API,
devigs them, and compares to PrizePicks lines to find +EV edges.

The Odds API free tier: 500 requests/month. Each event-odds call = 1 credit.
MLB ~15 games/day = ~450/month. Tight but workable.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional
import os
import json
import unicodedata
from datetime import datetime


def _strip_accents(s: str) -> str:
    """Remove accents/diacritics: Ramírez → Ramirez"""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_name_match(pp_names: pd.Series, sharp_name: str) -> pd.Series:
    """
    Match player names using both first AND last name.
    Handles accents (Ramírez), suffixes (Jr./III), and common variations.
    Returns a boolean Series for filtering.
    """
    sharp_clean = _strip_accents(sharp_name).lower().strip()
    sharp_parts = sharp_clean.replace(".", "").replace(",", "").split()
    # Remove suffixes
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    sharp_parts = [p for p in sharp_parts if p not in suffixes]

    if len(sharp_parts) < 2:
        # Fallback to last-name contains if we can't parse
        return pp_names.str.contains(sharp_name.split()[-1], case=False, na=False)

    sharp_first = sharp_parts[0]
    sharp_last = sharp_parts[-1]

    def _match(pp_name):
        if not isinstance(pp_name, str):
            return False
        pp_clean = _strip_accents(pp_name).lower().strip()
        pp_parts = pp_clean.replace(".", "").replace(",", "").split()
        pp_parts = [p for p in pp_parts if p not in suffixes]
        if len(pp_parts) < 2:
            return False
        pp_first = pp_parts[0]
        pp_last = pp_parts[-1]
        # Require last name exact match + first name starts-with (handles nicknames)
        return (pp_last == sharp_last and
                (pp_first == sharp_first or
                 pp_first.startswith(sharp_first[:3]) or
                 sharp_first.startswith(pp_first[:3])))

    return pp_names.apply(_match)


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"

# Prop market keys supported by The Odds API for MLB
PROP_MARKETS = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_hits": "batter_hits",
    "batter_total_bases": "batter_total_bases",
    "batter_home_runs": "batter_home_runs",
    "batter_rbis": "batter_rbis",
    "batter_runs_scored": "batter_runs_scored",
    "batter_stolen_bases": "batter_stolen_bases",
    "batter_singles": "batter_singles",
    "batter_doubles": "batter_doubles",
    "batter_walks": "batter_walks",
    "batter_strikeouts": "batter_strikeouts",
    "pitcher_hits_allowed": "pitcher_hits_allowed",
    "pitcher_walks": "pitcher_walks",
    "pitcher_earned_runs": "pitcher_earned_runs",
    "pitcher_outs": "pitcher_outs",
}

# Map PrizePicks stat names to Odds API market keys
PP_TO_ODDS_API = {
    "Pitcher Strikeouts": "pitcher_strikeouts",
    "Strikeouts": "pitcher_strikeouts",
    "Hits": "batter_hits",
    "Total Bases": "batter_total_bases",
    "Home Runs": "batter_home_runs",
    "RBIs": "batter_rbis",
    "Runs": "batter_runs_scored",
    "Stolen Bases": "batter_stolen_bases",
    "Hits Allowed": "pitcher_hits_allowed",
    "Walks Allowed": "pitcher_walks",
    "Earned Runs Allowed": "pitcher_earned_runs",
    "Pitching Outs": "pitcher_outs",
    "Batter Strikeouts": "batter_strikeouts",
    "Singles": "batter_singles",
    "Doubles": "batter_doubles",
    "Walks": "batter_walks",
}

# Book sharpness weights for MLB props (from Pikkit research)
BOOK_WEIGHTS = {
    "fanduel": 1.236,       # Sharpest for MLB props by far
    "pinnacle": 0.918,
    "draftkings": 0.859,
    "betmgm": 0.700,
    "caesars": 0.650,
    "bet365": 0.600,
    "bovada": 0.550,
    "betrivers": 0.500,
    "pointsbet": 0.480,
    "unibet": 0.450,
}

# Priority order for which books to trust
SHARP_BOOKS = ["fanduel", "pinnacle", "draftkings", "betmgm", "caesars"]


def get_api_key() -> str:
    """Get The Odds API key from env var or Streamlit secrets."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
    return key


def fetch_mlb_events(api_key: str = None) -> list:
    """Fetch current MLB events (games) list."""
    api_key = api_key or get_api_key()
    if not api_key:
        return []

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events",
            params={"apiKey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def fetch_event_props(event_id: str, markets: list = None, api_key: str = None) -> dict:
    """
    Fetch player prop odds for a specific MLB game.
    Each call costs 1 API credit.
    """
    api_key = api_key or get_api_key()
    if not api_key:
        return {}

    if markets is None:
        # Fetch ALL supported prop types — not just 5
        markets = list(PROP_MARKETS.values())

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "american",
            },
            timeout=20,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        # Record freshness
        try:
            from src.freshness import record_data_pull
            record_data_pull("sharp_odds", f"event {event_id}, {remaining} credits left")
        except Exception:
            pass
        return {"data": resp.json(), "remaining_credits": remaining}
    except requests.RequestException:
        return {}


def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def devig_two_way(over_odds: int, under_odds: int, method: str = "power") -> dict:
    """
    Remove the vig from a two-way line to get fair probabilities.

    Methods:
    - 'multiplicative': Scale proportionally (most common)
    - 'power': Power method (Pinnacle-preferred, more accurate)
    - 'additive': Split vig equally (simplest)
    """
    p_over = american_to_implied_prob(over_odds)
    p_under = american_to_implied_prob(under_odds)
    total = p_over + p_under  # > 1.0 due to vig

    if method == "power":
        # Power devig: find c where p_over^(1/c) + p_under^(1/c) = 1
        # Use binary search — more robust than closed-form approximation
        lo, hi = 1.0, 2.0
        for _ in range(50):  # converges fast
            mid = (lo + hi) / 2
            val = p_over ** (1 / mid) + p_under ** (1 / mid)
            if val > 1.0:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2
        fair_over = p_over ** (1 / c)
        fair_under = p_under ** (1 / c)
        # Normalize for any remaining rounding
        norm = fair_over + fair_under
        fair_over /= norm
        fair_under /= norm
    elif method == "additive":
        vig = total - 1.0
        fair_over = max(p_over - vig / 2, 0.01)
        fair_under = max(p_under - vig / 2, 0.01)
        norm = fair_over + fair_under
        fair_over /= norm
        fair_under /= norm
    else:  # multiplicative (default — what OddsShopper/OddsJam use)
        fair_over = p_over / total
        fair_under = p_under / total

    return {
        "fair_over": round(fair_over, 4),
        "fair_under": round(fair_under, 4),
        "vig_pct": round((total - 1.0) * 100, 2),
        "raw_over_odds": over_odds,
        "raw_under_odds": under_odds,
    }


def extract_sharp_lines(event_data: dict) -> list:
    """
    Parse The Odds API event response and extract devigged lines
    from sharp books, weighted by book sharpness.

    Returns list of dicts with player, market, line, fair probs, and consensus.
    """
    if not event_data:
        return []

    bookmakers = event_data.get("bookmakers", [])
    if not bookmakers:
        return []

    # Collect all lines by (player, market, line_value)
    lines_by_player = {}

    for book in bookmakers:
        book_key = book.get("key", "").lower()
        book_weight = BOOK_WEIGHTS.get(book_key, 0.3)

        for market in book.get("markets", []):
            market_key = market.get("key", "")

            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                name = outcome.get("name", "")  # "Over" or "Under"
                price = outcome.get("price", 0)
                point = outcome.get("point", 0)

                if not player or not price:
                    continue

                key = (player, market_key, point)
                if key not in lines_by_player:
                    lines_by_player[key] = {
                        "player": player,
                        "market": market_key,
                        "line": point,
                        "books": {},
                    }

                if book_key not in lines_by_player[key]["books"]:
                    lines_by_player[key]["books"][book_key] = {
                        "weight": book_weight,
                    }

                if name == "Over":
                    lines_by_player[key]["books"][book_key]["over_odds"] = price
                elif name == "Under":
                    lines_by_player[key]["books"][book_key]["under_odds"] = price

    # Now devig each book's line and compute weighted consensus
    results = []
    for key, data in lines_by_player.items():
        fair_probs = []

        for book_key, book_data in data["books"].items():
            over_odds = book_data.get("over_odds")
            under_odds = book_data.get("under_odds")
            weight = book_data.get("weight", 0.3)

            if over_odds is not None and under_odds is not None:
                devigged = devig_two_way(over_odds, under_odds, method="power")
                fair_probs.append({
                    "book": book_key,
                    "weight": weight,
                    "fair_over": devigged["fair_over"],
                    "fair_under": devigged["fair_under"],
                    "vig_pct": devigged["vig_pct"],
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                })

        if not fair_probs:
            continue

        # Weighted average fair probability
        total_weight = sum(fp["weight"] for fp in fair_probs)
        if total_weight == 0:
            continue

        consensus_over = sum(fp["fair_over"] * fp["weight"] for fp in fair_probs) / total_weight
        consensus_under = sum(fp["fair_under"] * fp["weight"] for fp in fair_probs) / total_weight

        # FanDuel-specific line (most important for MLB)
        fanduel_line = next((fp for fp in fair_probs if fp["book"] == "fanduel"), None)

        results.append({
            "player": data["player"],
            "market": data["market"],
            "line": data["line"],
            "consensus_fair_over": round(consensus_over, 4),
            "consensus_fair_under": round(consensus_under, 4),
            "fanduel_fair_over": round(fanduel_line["fair_over"], 4) if fanduel_line else None,
            "fanduel_fair_under": round(fanduel_line["fair_under"], 4) if fanduel_line else None,
            "num_books": len(fair_probs),
            "book_details": fair_probs,
        })

    return results


def find_ev_edges(pp_lines: pd.DataFrame, sharp_lines: list,
                  min_ev_pct: float = 0.25) -> list:
    """
    Compare PrizePicks lines to sharp devigged consensus.
    Flag props where PrizePicks offers better odds than fair value.

    PrizePicks treats all picks as ~50/50, so any prop where the
    fair probability of one side exceeds ~54.2% (5-pick flex breakeven)
    is potentially +EV.

    Args:
        pp_lines: DataFrame of PrizePicks lines
        sharp_lines: List of devigged sharp book lines
        min_ev_pct: Minimum edge % to flag (default 0.25%)

    Returns:
        List of +EV edge dicts sorted by edge size
    """
    edges = []

    for sharp in sharp_lines:
        # Find matching PrizePicks line
        player = sharp["player"]
        market = sharp["market"]
        sharp_line_val = sharp["line"]

        # Map Odds API market key back to PrizePicks stat names
        odds_to_pp = {v: k for k, v in PP_TO_ODDS_API.items()}
        pp_stat_names = [k for k, v in PP_TO_ODDS_API.items() if v == market]

        matching_pp = pp_lines[
            (_normalize_name_match(pp_lines["player_name"], player)) &
            (pp_lines["stat_type"].isin(pp_stat_names)) &
            (abs(pp_lines["line"] - sharp_line_val) < 1.0)  # Allow small line differences
        ]

        if matching_pp.empty:
            continue

        for _, pp_row in matching_pp.iterrows():
            pp_line = pp_row["line"]

            # Use FanDuel line if available, otherwise consensus
            fair_over = sharp.get("fanduel_fair_over") or sharp["consensus_fair_over"]
            fair_under = sharp.get("fanduel_fair_under") or sharp["consensus_fair_under"]

            # PrizePicks breakeven thresholds (implied odds of each entry type)
            # 5-pick flex: ~54.2% per leg
            # 6-pick flex: ~52.9% per leg
            pp_implied = 0.50  # PrizePicks treats each leg as 50/50

            # Determine which side has edge
            if pp_line == sharp_line_val:
                # Same line — direct comparison
                if fair_over > pp_implied + (min_ev_pct / 100):
                    pick = "MORE"
                    edge = fair_over - pp_implied
                    fair_prob = fair_over
                elif fair_under > pp_implied + (min_ev_pct / 100):
                    pick = "LESS"
                    edge = fair_under - pp_implied
                    fair_prob = fair_under
                else:
                    continue
            elif pp_line < sharp_line_val:
                # PP line is lower → MORE is easier on PP
                pick = "MORE"
                edge = fair_over + (sharp_line_val - pp_line) * 0.08  # rough adjustment
                edge = min(edge, 0.85)
                fair_prob = edge
                edge = fair_prob - pp_implied
            elif pp_line > sharp_line_val:
                # PP line is higher → LESS is easier on PP
                pick = "LESS"
                edge = fair_under + (pp_line - sharp_line_val) * 0.08
                edge = min(edge, 0.85)
                fair_prob = edge
                edge = fair_prob - pp_implied
            else:
                continue

            if edge < min_ev_pct / 100:
                continue

            # Rating based on edge size
            if edge >= 0.08:
                rating = "A"
            elif edge >= 0.05:
                rating = "B"
            elif edge >= 0.03:
                rating = "C"
            else:
                rating = "D"

            # EV calculation for different entry types
            ev_5pick = (fair_prob ** 5 * 10 + 5 * fair_prob ** 4 * (1 - fair_prob) * 2 +
                        10 * fair_prob ** 3 * (1 - fair_prob) ** 2 * 0.4) - 1
            ev_6pick = (fair_prob ** 6 * 25 + 6 * fair_prob ** 5 * (1 - fair_prob) * 2 +
                        15 * fair_prob ** 4 * (1 - fair_prob) ** 2 * 0.4) - 1

            edges.append({
                "player_name": pp_row["player_name"],
                "team": pp_row.get("team", ""),
                "stat_type": pp_row["stat_type"],
                "stat_internal": pp_row.get("stat_internal", market),
                "pp_line": pp_line,
                "line": pp_line,  # DB field — CRITICAL for correct grading
                "sharp_line": sharp_line_val,
                "pick": pick,
                "projection": pp_line * (1 + edge) if pick == "MORE" else pp_line * (1 - edge),
                "fair_prob": round(fair_prob, 4),
                "confidence": round(fair_prob, 4),  # DB field
                "edge_pct": round(edge * 100, 2),
                "edge": round(edge, 4),  # DB field (decimal, not pct)
                "p_over": round(fair_prob if pick == "MORE" else 1 - fair_prob, 4),
                "p_under": round(fair_prob if pick == "LESS" else 1 - fair_prob, 4),
                "rating": rating,
                "num_books": sharp["num_books"],
                "fanduel_agrees": sharp.get("fanduel_fair_over") is not None,
                "consensus_over": sharp["consensus_fair_over"],
                "consensus_under": sharp["consensus_fair_under"],
                "ev_5pick_roi": round(ev_5pick * 100, 1),
                "ev_6pick_roi": round(ev_6pick * 100, 1),
                "start_time": pp_row.get("start_time", ""),
                "market": market,
            })

    # Sort by edge size descending
    edges.sort(key=lambda x: x["edge_pct"], reverse=True)
    return edges


def get_api_usage(api_key: str = None) -> dict:
    """Check remaining API credits."""
    api_key = api_key or get_api_key()
    if not api_key:
        return {"remaining": "No API key", "used": "N/A"}

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports",
            params={"apiKey": api_key},
            timeout=10,
        )
        return {
            "remaining": resp.headers.get("x-requests-remaining", "?"),
            "used": resp.headers.get("x-requests-used", "?"),
        }
    except Exception:
        return {"remaining": "Error", "used": "Error"}
