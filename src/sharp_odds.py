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
from datetime import datetime
from scipy.optimize import brentq
from src.distributions import compute_probabilities


import logging

_log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
SPORT_KEY_PRESEASON = "baseball_mlb_preseason"

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

# Book sharpness weights for MLB PLAYER PROPS specifically
# Source: Bettor Odds/Pikkit study (Aug 2025) — sharpness is market-dependent!
# These differ from main-market sharpness (where Pinnacle leads).
BOOK_WEIGHTS = {
    "circa": 1.278,         # #1 for MLB props (not available on Odds API, but future-proof)
    "fanduel": 1.182,       # #2 — when FanDuel moves first, others follow
    "propbuilder": 1.116,   # #3
    "pinnacle": 0.962,      # #4 — drops from #1 on mains to #4 on props
    "draftkings": 0.910,    # #5
    "bet365": 0.888,        # #6
    "betmgm": 0.700,
    "caesars": 0.650,
    "bovada": 0.550,
    "betrivers": 0.500,
    "kambi": 0.400,         # Consistently softest — Kambi-powered books are targets
    "pointsbet": 0.480,
    "unibet": 0.450,
}

# Priority order for which books to trust for MLB props
SHARP_BOOKS = ["fanduel", "pinnacle", "draftkings", "betmgm", "caesars"]

# ── Distribution-based repricing engine ──────────────────────────────
# Maps Odds API market keys → distribution_params keys in current.json
_MARKET_TO_DIST_KEY = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_hits": "hits",
    "batter_total_bases": "total_bases",
    "batter_home_runs": "home_runs",
    "batter_rbis": "hits_runs_rbis",  # closest proxy
    "batter_runs_scored": "hits_runs_rbis",
    "batter_stolen_bases": "stolen_bases",
    "batter_singles": "hits",
    "batter_doubles": "hits",
    "batter_walks": "walks_allowed",
    "batter_strikeouts": "batter_strikeouts",
    "pitcher_hits_allowed": "hits_allowed",
    "pitcher_walks": "walks_allowed",
    "pitcher_earned_runs": "earned_runs",
    "pitcher_outs": "pitching_outs",
}

# Load distribution params from weights once
_DIST_PARAMS_CACHE = {}


def _load_dist_params() -> dict:
    """Load distribution_params from current.json (cached)."""
    if _DIST_PARAMS_CACHE:
        return _DIST_PARAMS_CACHE
    try:
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "weights", "current.json"
        )
        with open(weights_path) as f:
            w = json.load(f)
        _DIST_PARAMS_CACHE.update(w.get("distribution_params", {}))
    except Exception:
        pass
    return _DIST_PARAMS_CACHE


def _prob_over_at_line(line: float, mu: float, dist_type: str,
                       vr: float = 1.5, phi: float = 25.0) -> float:
    """Compute P(X >= ceil(line)) for a given mean mu and distribution type.

    Delegates to distributions.compute_probabilities() as single source of truth.

    Args:
        line: The threshold line value
        mu: Distribution mean
        dist_type: One of betabinom, negbin, normal, gamma, binary
        vr: Variance ratio (for negbin/normal/gamma)
        phi: Precision parameter (for betabinom)
    """
    result = compute_probabilities(
        line=line, mu=mu, dist_type=dist_type,
        var_ratio=vr, phi=phi,
    )
    return result["p_over"]


def _solve_mu_from_fair_over(fair_over: float, line: float,
                              dist_type: str, vr: float = 1.5,
                              phi: float = 25.0) -> float:
    """Numerically solve for the distribution mean mu such that
    P(X >= ceil(line)) = fair_over.

    Returns mu, or line as fallback if solver fails.
    """
    if fair_over <= 0.01 or fair_over >= 0.99:
        return max(0.1, line)

    def objective(mu):
        return _prob_over_at_line(line, mu, dist_type, vr, phi) - fair_over

    try:
        # Search range: mu from 0.1 to 3x the line
        lo, hi = 0.1, max(line * 3, 10.0)
        # Ensure bracketing — P(over) decreases as mu decreases
        f_lo = objective(hi)   # high mu → high P(over) → positive
        f_hi = objective(lo)   # low mu → low P(over) → negative
        if f_lo * f_hi > 0:
            # Can't bracket — fall back to line as mu
            return max(0.1, line)
        mu_solved = brentq(objective, lo, hi, xtol=0.001, maxiter=50)
        return max(0.1, mu_solved)
    except Exception:
        return max(0.1, line)


def distribution_reprice(market: str, sharp_line: float, pp_line: float,
                         fair_over: float, fair_under: float) -> tuple:
    """Replace the heuristic `* 0.08` line-difference adjustment with
    distribution-based repricing.

    Given sharp book fair probabilities at sharp_line, compute the fair
    probabilities at pp_line using the prop's statistical distribution.

    Args:
        market: Odds API market key (e.g. "pitcher_strikeouts")
        sharp_line: Sharp book consensus line
        pp_line: PrizePicks line
        fair_over: Sharp fair P(over) at sharp_line
        fair_under: Sharp fair P(under) at sharp_line

    Returns:
        (fair_over_at_pp, fair_under_at_pp) — adjusted probabilities
    """
    if pp_line == sharp_line:
        return fair_over, fair_under

    # Look up distribution type for this market
    dist_params = _load_dist_params()
    dist_key = _MARKET_TO_DIST_KEY.get(market, "")
    params = dist_params.get(dist_key, {})
    dist_type = params.get("type", "negbin")
    vr = params.get("vr", 1.5)
    phi = params.get("phi", 25.0)

    # Step 1: solve for mu from the sharp fair_over at sharp_line
    mu = _solve_mu_from_fair_over(fair_over, sharp_line, dist_type, vr, phi)

    # Step 2: compute P(over) at pp_line using that mu
    new_fair_over = _prob_over_at_line(pp_line, mu, dist_type, vr, phi)

    # Sanity bounds
    new_fair_over = min(0.95, max(0.05, new_fair_over))
    new_fair_under = min(0.95, max(0.05, 1.0 - new_fair_over))

    return new_fair_over, new_fair_under


def get_api_key() -> str:
    """Get The Odds API key from env var, .env file, or Streamlit secrets."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        # Try loading from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("ODDS_API_KEY") and "=" in line:
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
    return key


## ── Disk-based cache for API responses ──────────────────────────────
#  Saves credits by persisting responses to JSON files with timestamps.
#  Survives Streamlit reruns, app restarts, and dyno cycling.

_ODDS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "odds_cache"
)
os.makedirs(_ODDS_CACHE_DIR, exist_ok=True)

# Default TTL: 8 hours — one fetch per session, refresh manually via button.
# Prop odds move slowly for player props; no need to burn credits re-fetching.
# User can hit "Refresh Odds" in sidebar anytime for fresh data.
ODDS_CACHE_TTL_SECONDS = 28800  # 8 hours

# Budget guard: stop fetching when below this threshold
MIN_CREDITS_THRESHOLD = 50

# Track remaining credits in-memory (updated on every API response)
_last_known_credits: dict = {"remaining": None}


def _cache_path(key: str) -> str:
    """Get filesystem path for a cache entry."""
    safe_key = key.replace("/", "_").replace(":", "_")
    return os.path.join(_ODDS_CACHE_DIR, f"{safe_key}.json")


def _read_cache(key: str, ttl: int = ODDS_CACHE_TTL_SECONDS):
    """Read a cached API response if it exists and is fresh enough."""
    path = _cache_path(key)
    try:
        with open(path) as f:
            entry = json.load(f)
        cached_at = entry.get("cached_at", 0)
        age = datetime.now().timestamp() - cached_at
        if age < ttl:
            _log.info("[sharp_odds] CACHE HIT for %s (age: %ds)", key, int(age))
            return entry.get("data")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _write_cache(key: str, data) -> None:
    """Write an API response to disk cache."""
    path = _cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump({"cached_at": datetime.now().timestamp(), "data": data}, f)
    except Exception as e:
        _log.warning("[sharp_odds] Failed to write cache for %s: %s", key, e)


def get_credits_remaining() -> int:
    """Return last known credits remaining, or -1 if unknown."""
    val = _last_known_credits.get("remaining")
    if val is None:
        return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1


def clear_odds_cache() -> int:
    """Clear all cached odds files. Returns number of files removed."""
    count = 0
    try:
        for fname in os.listdir(_ODDS_CACHE_DIR):
            if fname.endswith(".json"):
                os.remove(os.path.join(_ODDS_CACHE_DIR, fname))
                count += 1
    except Exception:
        pass
    _log.info("[sharp_odds] Cleared %d cache files", count)
    return count


def fetch_mlb_events(api_key: str = None, sport_key: str = None) -> list:
    """Fetch current MLB events (games) list. Uses disk cache."""
    api_key = api_key or get_api_key()
    if not api_key:
        _log.warning("No Odds API key configured — skipping sharp odds")
        return []

    sport = sport_key or SPORT_KEY
    cache_key = f"events_{sport}_{datetime.now().strftime('%Y-%m-%d')}"

    # Check disk cache first
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    # Budget guard
    creds = get_credits_remaining()
    if 0 <= creds < MIN_CREDITS_THRESHOLD:
        _log.warning("[sharp_odds] Only %d credits left — skipping events fetch", creds)
        return []

    url = f"{ODDS_API_BASE}/sports/{sport}/events"
    _log.info("[sharp_odds] GET %s", url)

    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        credits = resp.headers.get("x-requests-remaining", "?")
        _last_known_credits["remaining"] = credits
        _log.info("[sharp_odds] Got %d events for %s, credits remaining: %s",
                  len(events), sport, credits)

        _write_cache(cache_key, events)
        return events
    except requests.RequestException as e:
        _log.error("[sharp_odds] ERROR fetching events from %s: %s", url, e)
        return []


def fetch_event_props(event_id: str, markets: list = None, api_key: str = None) -> dict:
    """
    Fetch player prop odds for a specific MLB game.
    Each call costs 1 API credit. Uses disk cache to avoid repeat calls.
    """
    api_key = api_key or get_api_key()
    if not api_key:
        _log.warning("[sharp_odds] No API key — skipping props for event %s", event_id)
        return {}

    cache_key = f"props_{event_id}_{datetime.now().strftime('%Y-%m-%d')}"

    # Check disk cache first
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    # Budget guard
    creds = get_credits_remaining()
    if 0 <= creds < MIN_CREDITS_THRESHOLD:
        _log.warning("[sharp_odds] Only %d credits left — skipping event %s", creds, event_id)
        return {}

    if markets is None:
        markets = list(PROP_MARKETS.values())

    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    _log.info("[sharp_odds] GET %s (%d markets)", url, len(markets))

    try:
        resp = requests.get(
            url,
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
        _last_known_credits["remaining"] = remaining
        data = resp.json()
        n_books = len(data.get("bookmakers", []))
        _log.info("[sharp_odds] Event %s: %d bookmakers, credits remaining: %s",
                  event_id, n_books, remaining)

        # Record freshness
        try:
            from src.freshness import record_data_pull
            record_data_pull("sharp_odds", f"event {event_id}, {remaining} credits left")
        except Exception:
            pass

        result = {"data": data, "remaining_credits": remaining}
        _write_cache(cache_key, result)
        return result
    except requests.RequestException as e:
        _log.error("[sharp_odds] ERROR fetching props for event %s: %s", event_id, e)
        return {}


def has_cached_odds_today() -> bool:
    """Check if we already have cached odds data for today.
    Used by app.py to decide whether to auto-fetch or skip."""
    today = datetime.now().strftime('%Y-%m-%d')
    events_key = f"events_{SPORT_KEY}_{today}"
    cached_events = _read_cache(events_key)
    if cached_events and len(cached_events) > 0:
        # Check if we also have at least one event's props cached
        for event in cached_events[:3]:
            eid = event.get("id", "")
            if eid:
                props_key = f"props_{eid}_{today}"
                if _read_cache(props_key) is not None:
                    return True
    return False


def get_cache_age_minutes() -> int:
    """Return age of the oldest cache entry in minutes, or -1 if no cache."""
    today = datetime.now().strftime('%Y-%m-%d')
    events_key = f"events_{SPORT_KEY}_{today}"
    path = _cache_path(events_key)
    try:
        with open(path) as f:
            entry = json.load(f)
        age = datetime.now().timestamp() - entry.get("cached_at", 0)
        return int(age / 60)
    except (FileNotFoundError, json.JSONDecodeError):
        return -1


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


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching: lowercase, strip suffixes and accents."""
    import unicodedata
    # Remove accents (ñ -> n, é -> e, etc.)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    # Strip common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def _names_match(sharp_name: str, pp_name: str) -> bool:
    """Check if two player names refer to the same person."""
    s = _normalize_name(sharp_name)
    p = _normalize_name(pp_name)
    if s == p:
        return True
    # Last name match + first initial match
    s_parts = s.split()
    p_parts = p.split()
    if len(s_parts) >= 2 and len(p_parts) >= 2:
        if s_parts[-1] == p_parts[-1] and s_parts[0][0] == p_parts[0][0]:
            return True
    return False


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
        pp_stat_names = [k for k, v in PP_TO_ODDS_API.items() if v == market]

        # Match players using robust name comparison
        name_mask = pp_lines["player_name"].apply(lambda pn: _names_match(player, pn))

        matching_pp = pp_lines[
            name_mask &
            (pp_lines["stat_type"].isin(pp_stat_names)) &
            (abs(pp_lines["line"] - sharp_line_val) <= 2.0)  # Allow line differences up to 2
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
                # Distribution-based repricing: compute exact P(over) at PP line
                repriced_over, repriced_under = distribution_reprice(
                    market, sharp_line_val, pp_line, fair_over, fair_under
                )
                pick = "MORE"
                fair_prob = repriced_over
                edge = fair_prob - pp_implied
            elif pp_line > sharp_line_val:
                # PP line is higher → LESS is easier on PP
                repriced_over, repriced_under = distribution_reprice(
                    market, sharp_line_val, pp_line, fair_over, fair_under
                )
                pick = "LESS"
                fair_prob = repriced_under
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

            # EV calculation using official March 2026 PrizePicks payouts
            # 5-pick Flex: 10x (5/5), 2x (4/5), 0.4x (3/5)
            from scipy.special import comb
            p, q = fair_prob, 1 - fair_prob
            ev_5pick = (p**5 * 10 + comb(5,4)*p**4*q * 2 +
                        comb(5,3)*p**3*q**2 * 0.4) - 1
            # 6-pick Flex: 12.5x (6/6), 2x (5/6), 0.4x (4/6)
            ev_6pick = (p**6 * 12.5 + comb(6,5)*p**5*q * 2 +
                        comb(6,4)*p**4*q**2 * 0.4) - 1

            edges.append({
                "player_name": pp_row["player_name"],
                "team": pp_row.get("team", ""),
                "stat_type": pp_row["stat_type"],
                "pp_line": pp_line,
                "sharp_line": sharp_line_val,
                "pick": pick,
                "fair_prob": round(fair_prob, 4),
                "edge_pct": round(edge * 100, 2),
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
    except Exception as e:
        _log.error("[sharp_odds] ERROR checking API usage: %s", e)
        return {"remaining": "Error", "used": "Error"}


# ═══════════════════════════════════════════════════════
# Hedge-Style Online Book-Weight Learning
# ═══════════════════════════════════════════════════════
# Instead of static book weights, learn which books are most
# reliable for each prop market over time using multiplicative
# weights (Hedge algorithm) with time decay.
#
# Source: Report 4 (Sierra) — continuous book-weight learning
# w_{b}^{t+1} ∝ w_{b}^{t} * exp(-η * loss_t(b))

import math as _math

_LEARNED_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "book_weights.json"
)


def load_learned_book_weights() -> dict:
    """Load per-market learned book weights from disk.

    Returns dict of {market_key: {book_key: weight}}.
    Falls back to static BOOK_WEIGHTS if no learned weights exist.
    """
    try:
        with open(_LEARNED_WEIGHTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_learned_book_weights(weights: dict) -> None:
    """Persist learned book weights to disk."""
    os.makedirs(os.path.dirname(_LEARNED_WEIGHTS_PATH), exist_ok=True)
    with open(_LEARNED_WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)


def hedge_update_book_weights(
    book_probs: dict,
    outcome: int,
    market_key: str,
    eta: float = 0.05,
    decay_half_life_hours: float = 24.0,
    dt_hours: float = 1.0,
) -> dict:
    """Update book weights for a market using Hedge algorithm with time decay.

    Args:
        book_probs: {book_key: devigged_probability} for the winning side
        outcome: 1 if over hit, 0 if under hit
        market_key: e.g. "pitcher_strikeouts", "batter_hits"
        eta: learning rate (lower = more conservative)
        decay_half_life_hours: time decay toward uniform
        dt_hours: time since last update

    Returns:
        Updated weights dict for this market
    """
    all_weights = load_learned_book_weights()
    market_weights = all_weights.get(market_key, {})

    # Initialize from static weights if empty
    if not market_weights:
        for book in book_probs:
            market_weights[book] = BOOK_WEIGHTS.get(book, 0.5)

    # Add any new books not seen before
    for book in book_probs:
        if book not in market_weights:
            market_weights[book] = 0.5

    # Step 1: Time decay toward uniform
    books = list(market_weights.keys())
    if books:
        decay = 0.5 ** (dt_hours / decay_half_life_hours)
        uniform = 1.0 / len(books)
        for b in books:
            market_weights[b] = decay * market_weights[b] + (1 - decay) * uniform

    # Step 2: Compute per-book log loss and apply multiplicative update
    eps = 1e-6
    for b in books:
        if b in book_probs:
            p = min(max(book_probs[b], eps), 1 - eps)
            loss = -(outcome * _math.log(p) + (1 - outcome) * _math.log(1 - p))
            market_weights[b] *= _math.exp(-eta * loss)

    # Step 3: Renormalize
    total = sum(market_weights.values())
    if total > 0:
        for b in books:
            market_weights[b] = market_weights[b] / total

    # Save
    all_weights[market_key] = market_weights
    save_learned_book_weights(all_weights)

    return market_weights


def get_effective_book_weight(book_key: str, market_key: str = None) -> float:
    """Get the effective weight for a book, blending static and learned.

    If learned weights exist for this market, use them.
    Otherwise fall back to static BOOK_WEIGHTS.
    """
    if market_key:
        learned = load_learned_book_weights()
        market_weights = learned.get(market_key, {})
        if book_key in market_weights:
            # Blend: 60% learned, 40% static (prevent runaway drift)
            learned_w = market_weights[book_key]
            static_w = BOOK_WEIGHTS.get(book_key, 0.5)
            return 0.6 * learned_w + 0.4 * static_w

    return BOOK_WEIGHTS.get(book_key, 0.3)
