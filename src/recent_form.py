"""Rolling recent form with exponential decay (L7/L14 hot/cold tracking).

Computes a per-player hot/cold multiplier by comparing exponential-decay-weighted
per-game performance over the last 7 and 14 days against the player's season average.

Formula:
    blended_rate = 0.60 * L7_weighted_rate + 0.40 * L14_weighted_rate
    multiplier   = blended_rate / season_rate
    final        = clamp(multiplier, 0.85, 1.15)

Decay weights (yesterday = 1.0):
    L7  decay = 0.85/day  → 7 days ago weight ≈ 0.38
    L14 decay = 0.92/day  → 14 days ago weight ≈ 0.30

Fallback: returns 1.0 (no adjustment) when:
    - Player cannot be found in MLB Stats API
    - Fewer than MIN_GAMES_REQUIRED games in the recent window
    - Season baseline has fewer than SEASON_MIN_GAMES games
    - Any API call fails

Data source: MLB Stats API (statsapi.mlb.com) — free, no auth required.
Endpoints:
    - All players: /api/v1/sports/1/players?season=YYYY
    - Game log:    /api/v1/people/{personId}/stats?stats=gameLog&season=YYYY&group=hitting|pitching
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10

# Exponential decay per day back (yesterday = weight 1.0)
L7_DECAY = 0.85
L14_DECAY = 0.92

L7_WINDOW = 7
L14_WINDOW = 14

BLEND_L7 = 0.60
BLEND_L14 = 0.40

# Multiplier bounds — prevent overreaction to small samples
MULT_FLOOR = 0.85
MULT_CAP = 1.15

# Minimum games needed in a window to trust the rate estimate
MIN_GAMES_REQUIRED = 3
# Minimum season games before the baseline is meaningful
SEASON_MIN_GAMES = 10

# Props that use the pitching game log group
_PITCHING_PROPS = frozenset({
    "pitcher_strikeouts", "pitching_outs", "earned_runs",
    "walks_allowed", "hits_allowed",
})

# ── Module-level caches ──────────────────────────────────────────────────────
# season → {normalized_name: player_id}
_PLAYER_ID_CACHE: dict[int, dict[str, int]] = {}
# (player_id, season, group) → list of {"date": date, "stat": dict}
_GAME_LOG_CACHE: dict[tuple, list] = {}
# track fetch date for daily cache invalidation
_GAME_LOG_CACHE_DATE: dict[tuple, date] = {}


# ── Name normalization ───────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Strip accents, suffixes, punctuation and lowercase."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.lower().strip()
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name)
    ascii_name = re.sub(r"[.\'\-]", "", ascii_name)
    return re.sub(r"\s+", " ", ascii_name).strip()


# ── Player ID lookup ─────────────────────────────────────────────────────────

def _load_player_id_map(season: int) -> dict[str, int]:
    """Fetch all active players for a season. Returns {norm_name: player_id}."""
    url = f"{MLB_API_BASE}/sports/1/players"
    params = {"season": season, "fields": "people,id,fullName,active"}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("recent_form: failed to load player map for %d: %s", season, exc)
        return {}

    result: dict[str, int] = {}
    for person in data.get("people", []):
        pid = person.get("id")
        full_name = person.get("fullName", "")
        if pid and full_name:
            result[_normalize_name(full_name)] = int(pid)
    return result


def _get_player_id(player_name: str, season: int) -> Optional[int]:
    """Return MLB player ID for a name, or None if not found."""
    if season not in _PLAYER_ID_CACHE:
        _PLAYER_ID_CACHE[season] = _load_player_id_map(season)

    id_map = _PLAYER_ID_CACHE[season]
    if not id_map:
        return None

    norm = _normalize_name(player_name)

    # Exact normalized match
    if norm in id_map:
        return id_map[norm]

    # Last name + first-initial fallback
    parts = norm.split()
    if len(parts) >= 2:
        for stored_name, pid in id_map.items():
            sp = stored_name.split()
            if (len(sp) >= 2
                    and sp[-1] == parts[-1]
                    and sp[0][:1] == parts[0][:1]):
                return pid

    logger.debug("recent_form: player not found: %s", player_name)
    return None


# ── Game log fetching ────────────────────────────────────────────────────────

def _fetch_game_logs(player_id: int, season: int, group: str) -> list[dict]:
    """
    Fetch full-season game log for a player.

    Returns list of {"date": date, "stat": dict} sorted ascending by date.
    Cached per (player_id, season, group) with daily invalidation.
    """
    cache_key = (player_id, season, group)
    today = date.today()

    if (cache_key in _GAME_LOG_CACHE
            and _GAME_LOG_CACHE_DATE.get(cache_key) == today):
        return _GAME_LOG_CACHE[cache_key]

    url = f"{MLB_API_BASE}/people/{player_id}/stats"
    params = {"stats": "gameLog", "season": season, "group": group}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning(
            "recent_form: game log fetch failed for player %d: %s", player_id, exc
        )
        return []

    games: list[dict] = []
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            date_str = split.get("date", "")
            try:
                game_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            games.append({"date": game_date, "stat": split.get("stat", {})})

    games.sort(key=lambda g: g["date"])
    _GAME_LOG_CACHE[cache_key] = games
    _GAME_LOG_CACHE_DATE[cache_key] = today
    return games


# ── Per-game stat extraction ─────────────────────────────────────────────────

def _batter_stat(stat: dict, prop: str) -> tuple[float, float]:
    """
    Extract (value, opportunity) from one batter game log stat dict.

    opportunity is plate appearances for rate stats, at-bats for hit-quality
    stats.  Returns (0.0, 0.0) for unsupported props or games with 0 PA.
    """
    ab = float(stat.get("atBats") or 0)
    pa = float(stat.get("plateAppearances") or ab)

    if ab == 0 and pa == 0:
        return 0.0, 0.0

    h = float(stat.get("hits") or 0)
    d2 = float(stat.get("doubles") or 0)
    t3 = float(stat.get("triples") or 0)
    hr = float(stat.get("homeRuns") or 0)
    rbi = float(stat.get("rbi") or 0)
    r = float(stat.get("runs") or 0)
    bb = float(stat.get("baseOnBalls") or 0)
    hbp = float(stat.get("hitByPitch") or 0)
    sb = float(stat.get("stolenBases") or 0)
    k = float(stat.get("strikeOuts") or 0)
    tb = float(stat.get("totalBases") or 0)
    singles = max(0.0, h - d2 - t3 - hr)

    if prop == "hits":
        return h, ab
    if prop == "total_bases":
        return tb, ab
    if prop == "home_runs":
        return hr, pa
    if prop == "rbis":
        return rbi, pa
    if prop == "runs":
        return r, pa
    if prop == "stolen_bases":
        return sb, pa
    if prop in ("batter_strikeouts", "hitter_strikeouts"):
        return k, pa
    if prop == "walks":
        return bb, pa
    if prop == "singles":
        return singles, ab
    if prop == "doubles":
        return d2, ab
    if prop == "hits_runs_rbis":
        return h + r + rbi, pa
    if prop == "hitter_fantasy_score":
        fs = (singles * 3 + d2 * 5 + t3 * 8 + hr * 10
              + rbi * 2 + r * 2 + (bb + hbp) * 2 + sb * 5)
        return fs, pa

    return 0.0, 0.0


def _pitcher_stat(stat: dict, prop: str) -> tuple[float, float]:
    """
    Extract (value, 1.0) from one pitcher game log stat dict.

    opportunity is always 1.0 (per start/appearance).
    Returns (0.0, 0.0) if the pitcher did not appear (0 outs recorded).
    """
    ip_str = str(stat.get("inningsPitched") or "0")
    try:
        parts = ip_str.split(".")
        outs = int(parts[0]) * 3 + (int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        try:
            outs = int(float(ip_str) * 3)
        except (ValueError, TypeError):
            outs = 0

    if outs == 0:
        return 0.0, 0.0

    if prop == "pitcher_strikeouts":
        return float(stat.get("strikeOuts") or 0), 1.0
    if prop == "pitching_outs":
        return float(outs), 1.0
    if prop == "earned_runs":
        return float(stat.get("earnedRuns") or 0), 1.0
    if prop == "walks_allowed":
        return float(stat.get("baseOnBalls") or 0), 1.0
    if prop == "hits_allowed":
        return float(stat.get("hits") or 0), 1.0

    return 0.0, 0.0


# ── Rate computation ─────────────────────────────────────────────────────────

def _weighted_rate(
    games: list[dict],
    prop: str,
    is_pitcher: bool,
    decay: float,
    window: int,
    today: date,
) -> Optional[float]:
    """
    Compute exponential-decay weighted rate for games in (today-window, today).

    Each game N days ago gets weight = decay^(N-1), so yesterday = 1.0.
    Returns None when fewer than MIN_GAMES_REQUIRED valid games fall in window.
    """
    extract = _pitcher_stat if is_pitcher else _batter_stat
    total_wv = 0.0
    total_wo = 0.0
    count = 0

    for game in games:
        days_ago = (today - game["date"]).days
        if days_ago < 1 or days_ago > window:
            continue
        val, opp = extract(game["stat"], prop)
        if opp == 0:
            continue
        w = decay ** (days_ago - 1)
        total_wv += w * val
        total_wo += w * opp
        count += 1

    if count < MIN_GAMES_REQUIRED or total_wo == 0:
        return None
    return total_wv / total_wo


def _season_rate(
    games: list[dict],
    prop: str,
    is_pitcher: bool,
    before: date,
) -> Optional[float]:
    """
    Compute unweighted season-average rate from all games before `before`.

    Returns None when fewer than SEASON_MIN_GAMES valid games are available.
    """
    extract = _pitcher_stat if is_pitcher else _batter_stat
    total_v = 0.0
    total_o = 0.0
    count = 0

    for game in games:
        if game["date"] >= before:
            continue
        val, opp = extract(game["stat"], prop)
        if opp == 0:
            continue
        total_v += val
        total_o += opp
        count += 1

    if count < SEASON_MIN_GAMES or total_o == 0:
        return None
    return total_v / total_o


# ── Public API ───────────────────────────────────────────────────────────────

def compute_recent_form_multiplier(
    player_name: str,
    stat_internal: str,
    game_date=None,
) -> float:
    """
    Compute L7/L14 exponential-decay hot/cold multiplier for a player.

    Compares the player's recent weighted per-game rate against their season
    average rate for the same stat.  Hot stretches produce multipliers > 1.0;
    cold stretches produce multipliers < 1.0.

    Args:
        player_name:   Full player name matched against MLB Stats API roster.
        stat_internal: Internal stat key (e.g. "hits", "pitcher_strikeouts").
        game_date:     Game date as a date object, datetime, or "YYYY-MM-DD"
                       string.  Defaults to date.today().

    Returns:
        Float in [MULT_FLOOR, MULT_CAP].  Returns 1.0 on any error or when
        insufficient data is available (graceful fallback).
    """
    # Resolve game_date to a date object
    if game_date is None:
        today = date.today()
    elif isinstance(game_date, str):
        try:
            today = date.fromisoformat(game_date[:10])
        except ValueError:
            today = date.today()
    elif hasattr(game_date, "date"):
        today = game_date.date()
    else:
        today = game_date  # type: ignore[assignment]

    is_pitcher = stat_internal in _PITCHING_PROPS
    group = "pitching" if is_pitcher else "hitting"
    season = today.year

    # Resolve player ID via MLB Stats API roster
    player_id = _get_player_id(player_name, season)
    if player_id is None:
        return 1.0

    # Fetch full-season game log (cached)
    games = _fetch_game_logs(player_id, season, group)
    if not games:
        return 1.0

    # Season baseline — requires SEASON_MIN_GAMES games before today
    base = _season_rate(games, stat_internal, is_pitcher, today)
    if base is None or base <= 0:
        return 1.0

    # Recent weighted rates
    l7 = _weighted_rate(games, stat_internal, is_pitcher, L7_DECAY, L7_WINDOW, today)
    l14 = _weighted_rate(games, stat_internal, is_pitcher, L14_DECAY, L14_WINDOW, today)

    if l7 is None and l14 is None:
        return 1.0

    # Blend L7 (60%) and L14 (40%); use whichever is available if only one qualifies
    if l7 is not None and l14 is not None:
        recent = BLEND_L7 * l7 + BLEND_L14 * l14
    elif l7 is not None:
        recent = l7
    else:
        recent = l14  # type: ignore[assignment]

    raw_mult = recent / base
    return max(MULT_FLOOR, min(MULT_CAP, raw_mult))


def clear_cache() -> None:
    """Clear all cached player IDs and game logs.  Useful in tests."""
    _PLAYER_ID_CACHE.clear()
    _GAME_LOG_CACHE.clear()
    _GAME_LOG_CACHE_DATE.clear()
