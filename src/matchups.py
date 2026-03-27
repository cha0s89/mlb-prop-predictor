"""
Matchups Module
Batter vs. Pitcher historical matchups, lineup data, and platoon splits.
These are the "hidden edge" factors that PrizePicks rarely prices correctly.

Data sources:
- pybaseball statcast: pitch-level BvP data
- MLB Stats API: lineups, probable pitchers, rosters
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

try:
    from pybaseball import (
        statcast,
        statcast_batter,
        statcast_pitcher,
        playerid_lookup,
        playerid_reverse_lookup,
        cache,
    )
    cache.enable()
    PYBASEBALL_OK = True
except ImportError:
    PYBASEBALL_OK = False


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


# ─────────────────────────────────────────────
# BATTER VS PITCHER MATCHUP HISTORY
# ─────────────────────────────────────────────

def get_bvp_matchup(batter_id: int, pitcher_id: int, years_back: int = 3) -> dict:
    """
    Pull historical batter vs. pitcher matchup data from Statcast.

    Returns aggregated stats: PA, hits, HR, K, BB, AVG, SLG, wOBA,
    barrel rate, exit velo, whiff rate — all specific to this matchup.

    This is gold for props. A batter who crushes a specific pitcher
    (or gets owned by one) is a massive edge that PrizePicks doesn't
    adjust for at the individual matchup level.
    """
    if not PYBASEBALL_OK:
        return _empty_bvp()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    try:
        # Pull all pitch-level data for this batter
        batter_data = statcast_batter(start_date, end_date, batter_id)
        if batter_data.empty:
            return _empty_bvp()

        # Filter to at-bats against this specific pitcher
        matchup = batter_data[batter_data["pitcher"] == pitcher_id]
        if matchup.empty:
            return _empty_bvp()

        # Count plate appearances (unique at-bat IDs)
        pa = matchup["at_bat_number"].nunique() if "at_bat_number" in matchup.columns else len(matchup.drop_duplicates(subset=["game_pk", "at_bat_number"])) if "game_pk" in matchup.columns else 0

        # Events (final pitch of each PA)
        if "events" in matchup.columns:
            events = matchup.dropna(subset=["events"])
        else:
            events = pd.DataFrame()

        if events.empty:
            return _empty_bvp(pa=pa)

        # Count outcomes
        hits = len(events[events["events"].isin(["single", "double", "triple", "home_run"])])
        singles = len(events[events["events"] == "single"])
        doubles = len(events[events["events"] == "double"])
        triples = len(events[events["events"] == "triple"])
        home_runs = len(events[events["events"] == "home_run"])
        strikeouts = len(events[events["events"].isin(["strikeout", "strikeout_double_play"])])
        walks = len(events[events["events"].isin(["walk", "hit_by_pitch"])])
        total_pa = len(events)

        # At-bats (PA minus walks, HBP, sac flies, sac bunts)
        non_ab = len(events[events["events"].isin(["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play"])])
        ab = total_pa - non_ab

        # Batting average
        avg = hits / ab if ab > 0 else 0.0

        # Slugging
        total_bases = singles + (doubles * 2) + (triples * 3) + (home_runs * 4)
        slg = total_bases / ab if ab > 0 else 0.0

        # ISO
        iso = slg - avg

        # K rate and BB rate
        k_rate = strikeouts / total_pa if total_pa > 0 else 0.0
        bb_rate = walks / total_pa if total_pa > 0 else 0.0

        # Statcast quality metrics for this matchup
        batted = matchup[matchup["type"] == "X"] if "type" in matchup.columns else pd.DataFrame()
        exit_velo = float(batted["launch_speed"].mean()) if not batted.empty and "launch_speed" in batted.columns else 0.0
        barrel_rate = float((batted.get("barrel", pd.Series([0])) == 1).mean() * 100) if not batted.empty and "barrel" in batted.columns else 0.0
        hard_hit = float((batted["launch_speed"] >= 95).mean() * 100) if not batted.empty and "launch_speed" in batted.columns else 0.0

        # Whiff rate for this matchup
        swings = matchup[matchup["description"].str.contains("swing|foul", case=False, na=False)] if "description" in matchup.columns else pd.DataFrame()
        whiffs = matchup[matchup["description"].str.contains("swinging_strike", case=False, na=False)] if "description" in matchup.columns else pd.DataFrame()
        whiff_rate = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0.0

        # Determine matchup quality
        if total_pa < 5:
            quality = "⚠️ Small sample"
        elif avg >= 0.350 and total_pa >= 10:
            quality = "🔥 Batter owns this pitcher"
        elif avg >= 0.300:
            quality = "✅ Batter has edge"
        elif avg <= 0.150 and total_pa >= 10:
            quality = "❄️ Pitcher dominates"
        elif avg <= 0.200:
            quality = "↘️ Pitcher has edge"
        else:
            quality = "➖ Neutral matchup"

        return {
            "has_data": True,
            "pa": total_pa,
            "ab": ab,
            "hits": hits,
            "home_runs": home_runs,
            "strikeouts": strikeouts,
            "walks": walks,
            "avg": round(avg, 3),
            "slg": round(slg, 3),
            "iso": round(iso, 3),
            "total_bases": total_bases,
            "tb_per_pa": round(total_bases / total_pa, 3) if total_pa > 0 else 0.0,
            "k_rate": round(k_rate * 100, 1),
            "bb_rate": round(bb_rate * 100, 1),
            "exit_velo": round(exit_velo, 1),
            "barrel_rate": round(barrel_rate, 1),
            "hard_hit_pct": round(hard_hit, 1),
            "whiff_rate": round(whiff_rate, 1),
            "quality": quality,
        }

    except Exception as e:
        return _empty_bvp(error=str(e))


def _empty_bvp(pa: int = 0, error: str = None) -> dict:
    return {
        "has_data": False,
        "pa": pa,
        "ab": 0, "hits": 0, "home_runs": 0,
        "strikeouts": 0, "walks": 0,
        "avg": 0.0, "slg": 0.0, "iso": 0.0,
        "total_bases": 0, "tb_per_pa": 0.0,
        "k_rate": 0.0, "bb_rate": 0.0,
        "exit_velo": 0.0, "barrel_rate": 0.0,
        "hard_hit_pct": 0.0, "whiff_rate": 0.0,
        "quality": "No matchup data" if not error else f"Error: {error}",
    }


# ─────────────────────────────────────────────
# PITCHER VS LINEUP (aggregate)
# ─────────────────────────────────────────────

def get_pitcher_vs_lineup(pitcher_id: int, lineup_batter_ids: list, years_back: int = 3) -> dict:
    """
    Aggregate BvP matchup data for a pitcher against an entire lineup.
    This tells you if a lineup historically crushes or struggles vs this pitcher.
    
    Critical for pitcher K props: if half the lineup has 30%+ K rates
    against this pitcher, K over becomes very attractive.
    """
    if not PYBASEBALL_OK or not lineup_batter_ids:
        return {"has_data": False, "batters_with_data": 0, "summary": "No data"}

    results = []
    for batter_id in lineup_batter_ids:
        bvp = get_bvp_matchup(batter_id, pitcher_id, years_back)
        if bvp["has_data"] and bvp["pa"] >= 3:
            results.append(bvp)

    if not results:
        return {"has_data": False, "batters_with_data": 0, "summary": "No matchup history"}

    total_pa = sum(r["pa"] for r in results)
    total_hits = sum(r["hits"] for r in results)
    total_ks = sum(r["strikeouts"] for r in results)
    total_hr = sum(r["home_runs"] for r in results)
    total_tb = sum(r["total_bases"] for r in results)
    total_ab = sum(r["ab"] for r in results)

    agg_avg = total_hits / total_ab if total_ab > 0 else 0.0
    agg_k_rate = total_ks / total_pa if total_pa > 0 else 0.0
    agg_slg = total_tb / total_ab if total_ab > 0 else 0.0

    # K prop insight: if lineup K rate vs this pitcher is high, K over is better
    if agg_k_rate >= 0.28:
        k_insight = "🔥 Lineup strikes out a LOT vs this pitcher — K OVER"
    elif agg_k_rate >= 0.23:
        k_insight = "↗️ Lineup K-prone vs this pitcher"
    elif agg_k_rate <= 0.15:
        k_insight = "❄️ Lineup makes great contact vs this pitcher — K UNDER"
    elif agg_k_rate <= 0.18:
        k_insight = "↘️ Lineup handles this pitcher well"
    else:
        k_insight = "➖ Neutral K matchup"

    return {
        "has_data": True,
        "batters_with_data": len(results),
        "total_pa": total_pa,
        "agg_avg": round(agg_avg, 3),
        "agg_k_rate": round(agg_k_rate * 100, 1),
        "agg_slg": round(agg_slg, 3),
        "total_hr": total_hr,
        "k_insight": k_insight,
        "summary": f"{len(results)} batters, {total_pa} PA: .{int(agg_avg*1000):03d}/{int(agg_slg*1000):03d}, {agg_k_rate*100:.0f}% K",
    }


# ─────────────────────────────────────────────
# PLATOON SPLITS (L/R)
# ─────────────────────────────────────────────

def get_platoon_split_adjustment(
    batter_hand: str,
    pitcher_hand: str,
    mlbam_id: int = None,
    season: int = None,
) -> dict:
    """
    Calculate platoon split adjustment.

    When mlbam_id is provided, fetches actual per-player L/R split data from
    the MLB Stats API and applies Bayesian shrinkage toward league-average
    splits (see src/platoon_splits.py).

    Falls back to the generic ±8% adjustment when:
    - mlbam_id is not provided
    - API call fails
    - Player has fewer than 30 PA against that pitcher hand
    """
    try:
        from src.platoon_splits import get_batter_platoon_adjustment
        return get_batter_platoon_adjustment(
            batter_hand=batter_hand,
            pitcher_hand=pitcher_hand,
            mlbam_id=mlbam_id,
            season=season,
        )
    except Exception:
        pass

    # Hard fallback: compute inline without the module
    batter_hand = batter_hand.upper().strip() if batter_hand else ""
    pitcher_hand = pitcher_hand.upper().strip() if pitcher_hand else ""

    if not batter_hand or not pitcher_hand:
        return {"adjustment": 1.0, "description": "Unknown handedness", "favorable": None}

    if batter_hand == "S":
        return {
            "adjustment": 1.02,
            "k_adjustment": 0.98,
            "description": "Switch hitter — always opposite side",
            "favorable": True,
            "batter_hand": "S",
            "pitcher_hand": pitcher_hand,
            "source": "generic",
        }

    is_favorable = (
        (batter_hand == "L" and pitcher_hand == "R")
        or (batter_hand == "R" and pitcher_hand == "L")
    )

    if is_favorable:
        return {
            "adjustment": 1.08,
            "k_adjustment": 0.92,
            "iso_adjustment": 1.14,
            "description": f"✅ Favorable platoon ({batter_hand}HB vs {pitcher_hand}HP)",
            "favorable": True,
            "batter_hand": batter_hand,
            "pitcher_hand": pitcher_hand,
            "source": "generic",
        }

    return {
        "adjustment": 0.92,
        "k_adjustment": 1.08,
        "iso_adjustment": 0.877,
        "description": f"❌ Same-side platoon ({batter_hand}HB vs {pitcher_hand}HP)",
        "favorable": False,
        "batter_hand": batter_hand,
        "pitcher_hand": pitcher_hand,
        "source": "generic",
    }


# ─────────────────────────────────────────────
# MLB LINEUP & PROBABLE PITCHER DATA
# ─────────────────────────────────────────────

def fetch_todays_schedule() -> list:
    """
    Fetch today's MLB schedule with probable pitchers.
    Uses the free MLB Stats API (no key needed).
    """
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "sportId": 1,  # MLB
                "date": today,
                "hydrate": "probablePitcher,linescore,team",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            games.append({
                "game_pk": game.get("gamePk"),
                "game_time": game.get("gameDate", ""),
                "status": game.get("status", {}).get("detailedState", ""),
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "home_team_name": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "away_team_name": away.get("team", {}).get("name", ""),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_hand": home_pitcher.get("pitchHand", {}).get("code", ""),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_hand": away_pitcher.get("pitchHand", {}).get("code", ""),
            })

    return games


def fetch_game_lineup(game_pk: int) -> dict:
    """
    Fetch confirmed lineup for a specific game.
    Lineups are usually available ~90 minutes before first pitch.
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/game/{game_pk}/boxscore",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return {"home": [], "away": []}

    result = {"home": [], "away": []}

    for side in ["home", "away"]:
        team_data = data.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        for player_id in batting_order:
            player_key = f"ID{player_id}"
            player_info = players.get(player_key, {})
            person = player_info.get("person", {})

            result[side].append({
                "id": player_id,
                "name": person.get("fullName", "Unknown"),
                "position": player_info.get("position", {}).get("abbreviation", ""),
                "bat_side": player_info.get("batSide", {}).get("code", ""),
                "batting_order": len(result[side]) + 1,
            })

    return result


def get_lineup_k_rate(lineup: list, years_back: int = 1) -> dict:
    """
    Calculate the aggregate K rate for a lineup.
    This is critical for pitcher K props.

    A lineup with a 26% aggregate K rate vs. one with 19% is
    nearly a 40% relative difference — massive for K prop prediction.
    """
    if not PYBASEBALL_OK or not lineup:
        return {"has_data": False, "avg_k_rate": 22.7}  # League average default

    # This would normally pull season stats for each batter
    # For now, return structure that the app can populate
    return {
        "has_data": False,
        "avg_k_rate": 22.7,
        "lineup_size": len(lineup),
        "note": "Full lineup K-rate integration available with pybaseball season stats",
    }


# ─────────────────────────────────────────────
# PLAYER ID LOOKUP
# ─────────────────────────────────────────────

def lookup_player_id(player_name: str) -> dict:
    """
    Look up a player's MLB ID from their name.
    Used to connect PrizePicks names to Statcast IDs for BvP data.
    """
    if not PYBASEBALL_OK:
        return {"found": False}

    try:
        parts = player_name.strip().split()
        if len(parts) < 2:
            return {"found": False}

        last = parts[-1]
        first = parts[0]

        result = playerid_lookup(last, first)
        if result.empty:
            return {"found": False}

        # Take the most recent player (highest key_mlbam)
        row = result.sort_values("key_mlbam", ascending=False).iloc[0]

        return {
            "found": True,
            "mlbam_id": int(row.get("key_mlbam", 0)),
            "fangraphs_id": int(row.get("key_fangraphs", 0)) if pd.notna(row.get("key_fangraphs")) else None,
            "name": f"{row.get('name_first', '')} {row.get('name_last', '')}",
        }
    except Exception:
        return {"found": False}


# ─────────────────────────────────────────────
# COMPOSITE MATCHUP SCORE
# ─────────────────────────────────────────────

def compute_matchup_confidence(
    bvp: dict,
    platoon: dict,
    weather: dict = None,
    umpire: dict = None,
    sharp_edge: dict = None,
    prop_type: str = "general",
) -> dict:
    """
    Combine all signals into a single matchup confidence score.
    
    The more independent signals that agree on a direction,
    the higher the confidence. This is where we stack edges.
    
    Returns a composite score and breakdown of contributing factors.
    """
    signals = []
    total_weight = 0
    weighted_sum = 0

    # 1. Sharp odds edge (most important — weight 40%)
    if sharp_edge and sharp_edge.get("edge_pct", 0) > 0:
        edge = sharp_edge["edge_pct"] / 100
        signals.append({
            "factor": "Sharp books",
            "direction": sharp_edge.get("pick", ""),
            "strength": min(edge * 5, 1.0),  # Normalize to 0-1
            "weight": 0.40,
            "detail": f"+{sharp_edge['edge_pct']:.1f}% edge vs PrizePicks",
        })
        weighted_sum += min(edge * 5, 1.0) * 0.40
        total_weight += 0.40

    # 2. BvP matchup history (weight 25%)
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 5:
        if prop_type in ("hits", "total_bases", "home_runs"):
            bvp_signal = (bvp["avg"] - 0.250) / 0.150  # Normalize around league avg
        elif prop_type in ("pitcher_strikeouts",):
            bvp_signal = (bvp["k_rate"] - 22.7) / 15.0  # Positive = more Ks
        else:
            bvp_signal = (bvp["avg"] - 0.250) / 0.150

        bvp_signal = max(-1.0, min(1.0, bvp_signal))
        direction = "MORE" if bvp_signal > 0 else "LESS"
        if prop_type in ("pitcher_strikeouts",):
            direction = "MORE" if bvp_signal > 0 else "LESS"  # High K rate = MORE Ks

        signals.append({
            "factor": "BvP matchup",
            "direction": direction,
            "strength": abs(bvp_signal),
            "weight": 0.25,
            "detail": f"{bvp['pa']} PA: .{int(bvp['avg']*1000):03d} AVG, {bvp['k_rate']:.0f}% K rate",
        })
        weighted_sum += abs(bvp_signal) * 0.25
        total_weight += 0.25

    # 3. Platoon advantage (weight 15%)
    if platoon and platoon.get("favorable") is not None:
        plat_strength = 0.6 if platoon["favorable"] else -0.6
        signals.append({
            "factor": "Platoon",
            "direction": "MORE" if platoon["favorable"] else "LESS",
            "strength": abs(plat_strength),
            "weight": 0.15,
            "detail": platoon.get("description", ""),
        })
        weighted_sum += abs(plat_strength) * 0.15
        total_weight += 0.15

    # 4. Weather (weight 10%)
    if weather:
        if prop_type in ("home_runs", "total_bases"):
            wx_mult = weather.get("weather_hr_mult", 1.0)
        elif prop_type in ("pitcher_strikeouts",):
            wx_mult = weather.get("weather_k_mult", 1.0)
        else:
            wx_mult = weather.get("weather_offense_mult", 1.0)

        if abs(wx_mult - 1.0) > 0.02:
            wx_signal = (wx_mult - 1.0) * 5
            direction = "MORE" if wx_signal > 0 else "LESS"
            signals.append({
                "factor": "Weather",
                "direction": direction,
                "strength": min(abs(wx_signal), 1.0),
                "weight": 0.10,
                "detail": f"{weather.get('temp_f', 72)}°F, {weather.get('wind_mph', 0)} mph wind",
            })
            weighted_sum += min(abs(wx_signal), 1.0) * 0.10
            total_weight += 0.10

    # 5. Umpire (weight 10% — K props only)
    if umpire and umpire.get("known") and prop_type in ("pitcher_strikeouts",):
        k_adj = umpire.get("k_adjustment", 0)
        if abs(k_adj) > 0.2:
            ump_signal = k_adj / 1.0  # Normalize
            direction = "MORE" if ump_signal > 0 else "LESS"
            signals.append({
                "factor": "Umpire",
                "direction": direction,
                "strength": min(abs(ump_signal), 1.0),
                "weight": 0.10,
                "detail": umpire.get("impact", ""),
            })
            weighted_sum += min(abs(ump_signal), 1.0) * 0.10
            total_weight += 0.10

    # Compute composite
    if total_weight == 0:
        return {"composite_score": 0.5, "signals": [], "agreement": "No data"}

    composite = weighted_sum / total_weight

    # Check signal agreement
    if signals:
        directions = [s["direction"] for s in signals if s["direction"]]
        if directions:
            more_count = directions.count("MORE")
            less_count = directions.count("LESS")
            total_signals = len(directions)

            if more_count == total_signals:
                agreement = f"✅ ALL {total_signals} signals agree: MORE"
            elif less_count == total_signals:
                agreement = f"✅ ALL {total_signals} signals agree: LESS"
            elif more_count > less_count:
                agreement = f"⚠️ {more_count}/{total_signals} signals say MORE"
            else:
                agreement = f"⚠️ {less_count}/{total_signals} signals say LESS"
        else:
            agreement = "No directional signals"
    else:
        agreement = "No signals available"

    return {
        "composite_score": round(composite, 3),
        "signals": signals,
        "signal_count": len(signals),
        "agreement": agreement,
    }
