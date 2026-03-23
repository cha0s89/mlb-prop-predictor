"""
Weather Module
Fetches game-time weather forecasts from Open-Meteo (free, no API key).
Calculates weather impact adjustments for prop predictions.
"""

import math
from datetime import datetime

import pandas as pd
import requests


# MLB stadium coordinates and dome status
STADIUMS = {
    "ARI": {"name": "Chase Field", "lat": 33.4455, "lon": -112.0667, "dome": True},
    "ATL": {"name": "Truist Park", "lat": 33.8907, "lon": -84.4677, "dome": False},
    "BAL": {"name": "Camden Yards", "lat": 39.2838, "lon": -76.6216, "dome": False},
    "BOS": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "dome": False},
    "CHC": {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "dome": False},
    "CWS": {"name": "Guaranteed Rate", "lat": 41.8299, "lon": -87.6338, "dome": False},
    "CIN": {"name": "Great American", "lat": 39.0974, "lon": -84.5082, "dome": False},
    "CLE": {"name": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "dome": False},
    "COL": {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942, "dome": False},
    "DET": {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "dome": False},
    "HOU": {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555, "dome": True},
    "KC":  {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "dome": False},
    "LAA": {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "dome": False},
    "LAD": {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "dome": False},
    "MIA": {"name": "LoanDepot Park", "lat": 25.7781, "lon": -80.2196, "dome": True},
    "MIL": {"name": "American Family", "lat": 43.0280, "lon": -87.9712, "dome": True},
    "MIN": {"name": "Target Field", "lat": 44.9818, "lon": -93.2775, "dome": False},
    "NYM": {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458, "dome": False},
    "NYY": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "dome": False},
    "OAK": {"name": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "dome": False},
    "PHI": {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "dome": False},
    "PIT": {"name": "PNC Park", "lat": 40.4469, "lon": -80.0057, "dome": False},
    "SD":  {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570, "dome": False},
    "SF":  {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "dome": False},
    "SEA": {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "dome": True},
    "STL": {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "dome": False},
    "TB":  {"name": "Tropicana Field", "lat": 27.7682, "lon": -82.6534, "dome": True},
    "TEX": {"name": "Globe Life Field", "lat": 32.7512, "lon": -97.0832, "dome": True},
    "TOR": {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "dome": True},
    "WSH": {"name": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "dome": False},
}

# Approximate home-plate-to-center-field bearings in degrees.
# Source: publicly documented ballpark orientation maps. Values are coarse but
# sufficient to distinguish tailwind, headwind, and crosswind relative to play.
STADIUM_CENTER_FIELD_AZIMUTH = {
    "ARI": 0.0,
    "ATL": 22.5,
    "BAL": 22.5,
    "BOS": 45.0,
    "CHC": 22.5,
    "CWS": 135.0,
    "CIN": 112.5,
    "CLE": 0.0,
    "COL": 0.0,
    "DET": 157.5,
    "HOU": 337.5,
    "KC": 45.0,
    "LAA": 45.0,
    "LAD": 22.5,
    "MIA": 135.0,
    "MIL": 135.0,
    "MIN": 90.0,
    "NYM": 22.5,
    "NYY": 67.5,
    "OAK": 67.5,
    "PHI": 13.0,
    "PIT": 112.5,
    "SD": 0.0,
    "SEA": 45.0,
    "SF": 90.0,
    "STL": 67.5,
    "TB": 45.0,
    "TEX": 135.0,
    "TOR": 0.0,
    "WSH": 22.5,
    # Temporary Sacramento home for the Athletics.
    "ATH": 45.0,
}

# Common team abbreviation aliases
TEAM_ALIASES = {
    "AZ": "ARI", "CHW": "CWS", "CWS": "CWS", "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "LAD": "LAD", "NYM": "NYM", "NYY": "NYY",
    "SD": "SD", "SDP": "SD", "SF": "SF", "SFG": "SF", "SEA": "SEA",
    "STL": "STL", "TB": "TB", "TBR": "TB", "TEX": "TEX", "TOR": "TOR",
    "WAS": "WSH", "WSH": "WSH",
}

# Team timezone mapping for Open-Meteo API
TEAM_TIMEZONE = {
    "LAD": "America/Los_Angeles", "LAA": "America/Los_Angeles",
    "SF": "America/Los_Angeles", "OAK": "America/Los_Angeles",
    "SD": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "ARI": "America/Phoenix",
    "COL": "America/Denver",
    "MIN": "America/Chicago", "MIL": "America/Chicago",
    "CHC": "America/Chicago", "CWS": "America/Chicago",
    "STL": "America/Chicago", "KC": "America/Chicago",
    "TEX": "America/Chicago", "HOU": "America/Chicago",
}
# Default to "America/New_York" for all eastern teams


def resolve_team(abbr: str) -> str:
    """Normalize team abbreviation."""
    abbr = abbr.upper().strip()
    return TEAM_ALIASES.get(abbr, abbr)


def is_dome(team: str) -> bool:
    """Check if team's home stadium has a roof/dome."""
    team = resolve_team(team)
    stadium = STADIUMS.get(team, {})
    return stadium.get("dome", False)


def fetch_game_weather(team: str, game_time: datetime = None) -> dict:
    """
    Fetch weather forecast for a team's stadium at game time.

    Returns dict with:
        temp_f, wind_mph, wind_dir, humidity, is_dome, precip_chance,
        weather_adjustment (multiplier for offensive props)
    """
    team = resolve_team(team)
    stadium = STADIUMS.get(team)

    if stadium is None:
        return _default_weather()

    # Dome stadiums get neutral weather
    if stadium["dome"]:
        return {
            "temp_f": 72.0,
            "wind_mph": 0.0,
            "wind_dir": "N/A",
            "humidity": 50.0,
            "is_dome": True,
            "precip_chance": 0.0,
            "stadium": stadium["name"],
            "weather_offense_mult": 1.0,
            "weather_hr_mult": 1.0,
            "weather_k_mult": 1.0,
        }

    if game_time is None:
        game_time = datetime.now()

    try:
        tz = TEAM_TIMEZONE.get(team, "America/New_York")
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": stadium["lat"],
                "longitude": stadium["lon"],
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": tz,
                "forecast_days": 3,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return _default_weather(stadium_name=stadium["name"])

    # Find the closest hour to game time
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return _default_weather(stadium_name=stadium["name"])

    game_hour_str = game_time.strftime("%Y-%m-%dT%H:00")
    idx = 0
    for i, t in enumerate(times):
        if t >= game_hour_str:
            idx = i
            break

    temp = hourly.get("temperature_2m", [72])[idx]
    humidity = hourly.get("relative_humidity_2m", [50])[idx]
    wind_mph = hourly.get("wind_speed_10m", [5])[idx]
    wind_dir = hourly.get("wind_direction_10m", [0])[idx]
    precip = hourly.get("precipitation_probability", [0])[idx]

    field_azimuth = STADIUM_CENTER_FIELD_AZIMUTH.get(team)
    wind_context = _classify_field_relative_wind(wind_mph, wind_dir, field_azimuth)

    # Calculate weather adjustments
    offense_mult, hr_mult, k_mult = _calculate_weather_impact(
        temp, wind_mph, wind_dir, humidity, stadium, field_azimuth
    )

    wind_label = _wind_direction_label(wind_dir)

    return {
        "temp_f": round(temp, 1),
        "wind_mph": round(wind_mph, 1),
        "wind_dir": wind_label,
        "wind_degrees": wind_dir,
        "humidity": round(humidity, 1),
        "is_dome": False,
        "precip_chance": round(precip, 1),
        "stadium": stadium["name"],
        "center_field_azimuth": field_azimuth,
        "weather_offense_mult": round(offense_mult, 4),
        "weather_hr_mult": round(hr_mult, 4),
        "weather_k_mult": round(k_mult, 4),
        **wind_context,
    }


def _angle_delta(a: float, b: float) -> float:
    """Return the signed shortest-path difference between two bearings."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _classify_field_relative_wind(
    wind_mph: float,
    wind_dir: float,
    field_azimuth: float | None,
) -> dict:
    """
    Convert meteorological wind direction into baseball terms.

    Open-Meteo reports the direction the wind is coming FROM. Baseball impact
    depends on where the wind is blowing TO relative to center field.
    """
    if field_azimuth is None:
        return {
            "wind_to_center_mph": 0.0,
            "wind_in_from_center_mph": 0.0,
            "crosswind_mph": 0.0,
            "wind_field_relation": "unknown",
            "wind_relative_degrees": None,
        }

    try:
        wind_to_deg = (float(wind_dir) + 180.0) % 360.0
        rel = _angle_delta(wind_to_deg, float(field_azimuth))
        tail_component = float(wind_mph) * math.cos(math.radians(rel))
        cross_component = float(wind_mph) * math.sin(math.radians(rel))
    except (TypeError, ValueError):
        return {
            "wind_to_center_mph": 0.0,
            "wind_in_from_center_mph": 0.0,
            "crosswind_mph": 0.0,
            "wind_field_relation": "unknown",
            "wind_relative_degrees": None,
        }

    if abs(rel) <= 35:
        relation = "out"
    elif abs(rel) >= 145:
        relation = "in"
    else:
        relation = "cross"

    return {
        "wind_to_center_mph": round(max(tail_component, 0.0), 2),
        "wind_in_from_center_mph": round(max(-tail_component, 0.0), 2),
        "crosswind_mph": round(abs(cross_component), 2),
        "wind_field_relation": relation,
        "wind_relative_degrees": round(rel, 1),
    }


def _calculate_weather_impact(temp: float, wind_mph: float, wind_dir: float,
                                humidity: float, stadium: dict, field_azimuth: float | None) -> tuple:
    """
    Calculate weather multipliers for offense, HR, and K props.

    Based on research:
    - Every 1.8°F above 72°F increases HR rate by ~2%
    - Wind out (to CF) adds ~19ft per 5mph
    - Humidity has minimal effect (myth debunked)
    """
    # Temperature effect (baseline 72°F)
    temp_delta = (temp - 72.0) / 1.8  # Convert to Celsius delta
    temp_offense_adj = 1.0 + (temp_delta * 0.02 * 0.5)  # ~1% per degree C for general offense
    temp_hr_adj = 1.0 + (temp_delta * 0.02)  # ~2% per degree C for HR

    wind_ctx = _classify_field_relative_wind(wind_mph, wind_dir, field_azimuth)
    wind_out_factor = min(wind_ctx["wind_to_center_mph"] / 18.0, 1.0)
    wind_in_factor = min(wind_ctx["wind_in_from_center_mph"] / 18.0, 1.0)
    cross_factor = min(wind_ctx["crosswind_mph"] / 20.0, 1.0)

    wind_offense_adj = 1.0 + (wind_out_factor * 0.035) - (wind_in_factor * 0.03) + (cross_factor * 0.005)
    wind_hr_adj = 1.0 + (wind_out_factor * 0.08) - (wind_in_factor * 0.08) - (cross_factor * 0.01)
    wind_k_adj = 1.0 - (wind_out_factor * 0.015) + (wind_in_factor * 0.02)

    # Cold weather slightly increases K rate (less grip, tighter zone)
    k_adj = 1.0
    if temp < 55:
        k_adj = 1.0 + (55 - temp) * 0.003

    offense_mult = max(0.85, min(1.20, temp_offense_adj * wind_offense_adj))
    hr_mult = max(0.80, min(1.35, temp_hr_adj * wind_hr_adj))
    k_mult = max(0.95, min(1.10, k_adj * wind_k_adj))

    return offense_mult, hr_mult, k_mult


def _wind_direction_label(degrees: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(degrees / 22.5) % 16
    return dirs[idx]


def get_stat_specific_weather_adjustment(weather: dict, stat_internal: str) -> float:
    """Return weather multiplier specific to the stat being predicted.

    Research-backed adjustments:
    - Temperature: HR/TB/SLG most affected (+1.5% per 10°F above 72°F)
    - Wind out: HR/TB boosted (~3-5%), Hits slightly boosted (~1%)
    - Wind in: HR/TB suppressed (~3-5%)
    - Rain/dome: minimal effect (games get postponed or played in dome)

    Args:
        weather (dict): Weather data with temperature, wind_speed, wind_direction, dome status
        stat_internal (str): Internal stat name (e.g., 'home_runs', 'hits', 'pitcher_strikeouts')

    Returns:
        float: Multiplier to apply to the stat (clamped to 0.85-1.15 range)
    """
    if not weather:
        return 1.0

    temp = weather.get("temperature") or weather.get("temp") or weather.get("temp_f")
    wind_speed = weather.get("wind_speed", 0) or weather.get("wind_mph", 0)
    wind_dir = weather.get("wind_direction", "").lower() or weather.get("wind_dir", "").lower()
    is_dome = weather.get("dome", False) or weather.get("is_dome", False)

    if is_dome:
        return 1.0  # Dome = controlled environment

    mult = 1.0

    # Temperature effect (baseline 72°F)
    if temp is not None:
        try:
            temp = float(temp)
            temp_delta = (temp - 72) / 10  # per 10°F above 72

            # Stats most affected by temperature
            if stat_internal in ("total_bases", "home_runs"):
                mult *= 1.0 + temp_delta * 0.015  # 1.5% per 10°F
            elif stat_internal in ("hits", "hits_runs_rbis", "rbis", "runs"):
                mult *= 1.0 + temp_delta * 0.008  # 0.8% per 10°F
            elif stat_internal in ("pitcher_strikeouts",):
                mult *= 1.0 - temp_delta * 0.003  # Slightly fewer K's in heat (more offense)
            elif stat_internal in ("earned_runs",):
                mult *= 1.0 + temp_delta * 0.012  # More runs in heat
        except (ValueError, TypeError):
            pass

    # Wind effect: prefer field-relative components derived from stadium orientation.
    if wind_speed and wind_speed > 5:
        try:
            wind_speed = float(wind_speed)
            tailwind = float(weather.get("wind_to_center_mph", 0.0) or 0.0)
            headwind = float(weather.get("wind_in_from_center_mph", 0.0) or 0.0)
            crosswind = float(weather.get("crosswind_mph", 0.0) or 0.0)
            if not (tailwind or headwind or crosswind):
                wind_factor = min(wind_speed / 15.0, 1.0)
                tailwind = wind_speed if any(d in wind_dir for d in ["out", "left", "right", "center"]) else 0.0
                headwind = wind_speed if "in" in wind_dir else 0.0
                crosswind = wind_speed if not tailwind and not headwind else 0.0
                tail_factor = min(tailwind / 15.0, 1.0)
                head_factor = min(headwind / 15.0, 1.0)
                cross_factor = min(crosswind / 20.0, 1.0)
            else:
                tail_factor = min(tailwind / 18.0, 1.0)
                head_factor = min(headwind / 18.0, 1.0)
                cross_factor = min(crosswind / 20.0, 1.0)

            if stat_internal in ("total_bases", "home_runs"):
                mult *= 1.0 + tail_factor * 0.08 - head_factor * 0.08 - cross_factor * 0.01
            elif stat_internal in ("hits", "singles", "doubles"):
                mult *= 1.0 + tail_factor * 0.03 - head_factor * 0.025 + cross_factor * 0.01
            elif stat_internal in ("hits_runs_rbis", "rbis", "runs", "hitter_fantasy_score"):
                mult *= 1.0 + tail_factor * 0.045 - head_factor * 0.04 + cross_factor * 0.012
            elif stat_internal in ("earned_runs", "hits_allowed"):
                mult *= 1.0 + tail_factor * 0.055 - head_factor * 0.045 + cross_factor * 0.015
            elif stat_internal in ("pitcher_strikeouts", "batter_strikeouts"):
                mult *= 1.0 - tail_factor * 0.02 + head_factor * 0.025
            elif stat_internal in ("pitching_outs",):
                mult *= 1.0 - tail_factor * 0.015 + head_factor * 0.02
        except (ValueError, TypeError):
            pass

    return round(max(0.85, min(1.15, mult)), 4)  # Cap at ±15%


def _default_weather(stadium_name: str = "Unknown") -> dict:
    return {
        "temp_f": 72.0,
        "wind_mph": 5.0,
        "wind_dir": "N/A",
        "humidity": 50.0,
        "is_dome": False,
        "precip_chance": 0.0,
        "stadium": stadium_name,
        "center_field_azimuth": None,
        "wind_to_center_mph": 0.0,
        "wind_in_from_center_mph": 0.0,
        "crosswind_mph": 0.0,
        "wind_field_relation": "unknown",
        "wind_relative_degrees": None,
        "weather_offense_mult": 1.0,
        "weather_hr_mult": 1.0,
        "weather_k_mult": 1.0,
    }
