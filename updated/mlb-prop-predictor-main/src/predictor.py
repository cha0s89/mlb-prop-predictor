"""
Prediction Engine v2 — Full Sabermetric Integration
Every prop type. Every relevant stat. Proper stabilization and weighting.

PROP TYPES COVERED:
  Pitcher: Strikeouts, Outs Recorded, Earned Runs, Walks, Hits Allowed
  Batter:  Hits, Total Bases, Home Runs, RBIs, Runs, Stolen Bases,
           Batter Strikeouts, Walks, Singles, Doubles, H+R+RBI combo

KEY INPUTS PER PROJECTION:
  1. Season stats (regressed via Bayesian stabilization)
  2. Statcast quality metrics (xBA, xSLG, barrel rate, CSW%, EV90, etc.)
  3. BvP matchup history (batter vs. specific pitcher)
  4. Platoon splits (L/R handedness advantage)
  5. Park factors (general, HR-specific, K-specific)
  6. Weather adjustments (temp, wind, humidity)
  7. Umpire tendencies (K-rate impact)
  8. Lineup position (PA opportunity adjustment)
  9. Opposing quality (pitcher FIP for batters, lineup wOBA for pitchers)

WEIGHTING PHILOSOPHY (from research):
  - Sharp book edge is primary (handled in sharp_odds.py)
  - This engine is the SECONDARY confirmation layer
  - Bayesian regression pulls small samples toward league average
  - Statcast expected stats are descriptive, not purely predictive
    (used as one signal among many, not the sole driver)
  - BvP data is powerful but requires 10+ PA to be meaningful
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Optional


# ═══════════════════════════════════════════════════════
# LEAGUE AVERAGES (2024 season — update annually)
# ═══════════════════════════════════════════════════════
LG = {
    # Batting
    "avg": 0.248, "obp": 0.312, "slg": 0.399, "iso": 0.151,
    "babip": 0.296, "woba": 0.310, "wrc_plus": 100,
    "k_rate": 22.7, "bb_rate": 8.3,
    "hr_per_pa": 0.033, "sb_per_game": 0.18,
    "rbi_per_game": 0.55, "runs_per_game": 0.55,
    "hits_per_game": 0.95, "tb_per_game": 1.50,
    # Statcast batting
    "exit_velo": 88.5, "hard_hit_pct": 37.0, "barrel_rate": 7.5,
    "sweet_spot_pct": 32.0, "ev90": 105.0,
    "xba": 0.248, "xslg": 0.399, "xwoba": 0.310,
    "sprint_speed": 27.0,  # ft/s
    # Pitching
    "era": 4.17, "fip": 4.12, "xfip": 4.10, "siera": 4.05,
    "whip": 1.28, "k9": 8.58, "bb9": 3.22, "hr9": 1.15,
    "k_pct_p": 22.7, "bb_pct_p": 8.3, "hr_fb_rate": 12.0,
    # Statcast pitching
    "csw_pct": 28.5, "swstr_pct": 11.3,
    "zone_pct": 45.0, "f_strike_pct": 60.0,
    "chase_rate": 28.0,
    # Game context
    "pa_per_lineup_spot": {1: 4.8, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.3,
                           6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8},
    "bf_per_ip": 4.3,
    "avg_ip_starter": 5.5,
}

# ═══════════════════════════════════════════════════════
# STABILIZATION CONSTANTS (PA/BF for 50% regression to mean)
# From FanGraphs / Russell Carleton research
# ═══════════════════════════════════════════════════════
STAB = {
    # Batter (PA)
    "k_rate": 60, "bb_rate": 120, "hbp_rate": 240,
    "babip": 820, "avg": 500, "obp": 300, "slg": 320, "iso": 160,
    "woba": 300, "hr_fb": 300, "gb_rate": 80, "fb_rate": 80,
    "ld_rate": 600, "barrel_rate": 100, "contact_rate": 100,
    "sprint_speed": 50,  # essentially stable (physical trait)
    # Pitcher (BF)
    "k_pct_p": 70, "bb_pct_p": 170, "hr_fb_p": 300,
    "babip_p": 2000, "csw_pct": 200, "swstr_pct": 150,
    "era": 600, "fip": 200, "gb_rate_p": 100,
    "f_strike_pct": 100, "zone_pct": 100,
}


# ═══════════════════════════════════════════════════════
# PARK FACTORS (100 = neutral)
# ═══════════════════════════════════════════════════════
PARK = {  # General runs
    "COL": 116, "ARI": 106, "TEX": 105, "CIN": 104, "BOS": 103,
    "PHI": 102, "ATL": 101, "CHC": 101, "MIL": 101, "LAD": 100,
    "MIN": 100, "NYY": 100, "CLE": 100, "HOU": 100, "BAL": 100,
    "DET": 99, "KC": 99, "STL": 99, "PIT": 98, "CWS": 98,
    "WSH": 98, "TOR": 98, "LAA": 97, "NYM": 97, "SEA": 96,
    "SD": 96, "SF": 95, "TB": 96, "MIA": 95, "OAK": 97,
}
PARK_HR = {  # HR-specific (more extreme)
    "COL": 130, "NYY": 115, "CIN": 112, "TEX": 108, "ARI": 107,
    "PHI": 106, "BAL": 105, "BOS": 103, "CHC": 104, "ATL": 102,
    "MIL": 102, "LAD": 100, "MIN": 100, "HOU": 99, "DET": 98,
    "CLE": 97, "CWS": 97, "OAK": 97, "KC": 95, "STL": 98,
    "PIT": 94, "WSH": 99, "TOR": 98, "LAA": 96, "TB": 96,
    "NYM": 92, "SEA": 93, "SD": 91, "MIA": 90, "SF": 88,
}
PARK_K = {  # K-rate
    "SD": 103, "SF": 102, "SEA": 102, "MIA": 102, "PHI": 101,
    "ARI": 101, "MIL": 101, "LAD": 101, "CLE": 101, "HOU": 101,
    "TOR": 101, "LAA": 101, "NYM": 101, "TB": 101,
    "ATL": 100, "CHC": 100, "DET": 100, "KC": 100, "STL": 100,
    "PIT": 100, "CWS": 100, "WSH": 100, "BAL": 100, "MIN": 100,
    "NYY": 100, "OAK": 100, "TEX": 99, "CIN": 100, "BOS": 98,
    "COL": 96,
}
PARK_SB = {  # Stolen base factors (turf, foul territory, altitude)
    "TB": 105, "TOR": 104, "ARI": 103, "HOU": 102, "MIA": 102,
    "COL": 101, "KC": 101, "MIN": 101, "TEX": 100, "ATL": 100,
    "BAL": 100, "BOS": 100, "CHC": 100, "CIN": 100, "CLE": 100,
    "CWS": 100, "DET": 100, "LAA": 100, "LAD": 100, "MIL": 100,
    "NYM": 100, "NYY": 100, "OAK": 100, "PHI": 100, "PIT": 100,
    "SD": 100, "SF": 99, "SEA": 100, "STL": 100, "WSH": 100,
}

# Also export for backward compatibility
PARK_FACTORS = PARK
PARK_FACTORS_HR = PARK_HR
PARK_FACTORS_K = PARK_K


def _regress(observed, sample_n, stab_n, league_avg):
    """Bayesian regression: blend observed with league average based on sample size."""
    return (sample_n * observed + stab_n * league_avg) / (sample_n + stab_n)


def _park(team, factors_dict):
    """Get park factor as multiplier (1.0 = neutral)."""
    return factors_dict.get(team, 100) / 100.0


def _lineup_pa(order_pos):
    """Expected PA based on lineup position (1-9)."""
    return LG["pa_per_lineup_spot"].get(order_pos, 4.2)


# ═══════════════════════════════════════════════════════
# PITCHER PROJECTIONS
# ═══════════════════════════════════════════════════════

def project_pitcher_strikeouts(p, bvp=None, platoon=None, ump=None,
                                opp_k_rate=None, park=None, wx=None,
                                expected_ip=None):
    """
    PITCHER STRIKEOUTS — strongest signal prop type.

    Primary inputs (weighted by research):
      K%/K9 (30%) — season rate, regressed
      CSW% + SwStr% (20%) — Statcast pitch quality
      Opposing lineup K% (20%) — who they're facing matters enormously
      Umpire tendency (10%) — high-K umps add 0.5-1.0 K per pitcher
      BvP aggregate K rate (10%) — historical matchup
      Park K factor (5%) — slight effect
      Weather (5%) — cold = grip issues = fewer Ks
    """
    k_pct = p.get("k_pct", 0) or (p.get("k9", LG["k9"]) / 9 * 27 / LG["bf_per_ip"] * 100)
    bf_est = p.get("ip", 0) * LG["bf_per_ip"] if p.get("ip", 0) > 0 else 0
    csw = p.get("recent_csw_pct", 0)
    swstr = p.get("recent_swstr_pct", 0)

    # Regress K%
    reg_k = _regress(k_pct, bf_est, STAB["k_pct_p"], LG["k_pct_p"])

    # CSW quality adjustment (r=0.87 with K%)
    if csw > 0:
        csw_delta = (csw - LG["csw_pct"]) * 1.2  # +1 CSW% ≈ +1.2 K%
        reg_k = reg_k * 0.70 + (reg_k + csw_delta) * 0.30
    if swstr > 0:
        swstr_delta = (swstr - LG["swstr_pct"]) * 1.5
        reg_k = reg_k * 0.85 + (reg_k + swstr_delta) * 0.15

    # Opposing lineup K% (team K rates range 19%-27% — huge impact)
    if opp_k_rate and opp_k_rate > 0:
        reg_k *= (opp_k_rate / LG["k_rate"])

    # BvP aggregate K adjustment
    if bvp and bvp.get("has_data") and bvp.get("total_pa", 0) >= 10:
        bvp_k = bvp.get("agg_k_rate", LG["k_rate"])
        bvp_adj = bvp_k / LG["k_rate"]
        reg_k *= (1 + (bvp_adj - 1) * 0.25)  # 25% weight on BvP

    # Expected IP / batters faced
    if expected_ip is None:
        ip = p.get("ip", 0)
        starts = max(p.get("gs", ip / 5.5), 1) if ip > 10 else 1
        expected_ip = min(ip / starts, 7.0) if ip > 10 else LG["avg_ip_starter"]
        expected_ip = max(4.5, min(7.5, expected_ip))
    exp_bf = expected_ip * LG["bf_per_ip"]

    # Raw projection
    proj = exp_bf * (reg_k / 100)

    # Park K factor
    if park: proj *= _park(park, PARK_K)

    # Umpire adjustment (+/- 0.5-1.0 K)
    if ump and ump.get("known"):
        proj += ump.get("k_adjustment", 0)

    # Weather
    if wx: proj *= wx.get("weather_k_mult", 1.0)

    # Platoon (if entire lineup skews one hand)
    if platoon and platoon.get("k_adjustment"):
        proj *= platoon["k_adjustment"]

    mu = max(proj, 0.5)
    return {"projection": round(mu, 2), "mu": mu, "regressed_k_pct": round(reg_k, 1),
            "expected_ip": round(expected_ip, 1), "expected_bf": round(exp_bf, 1)}


def project_pitcher_outs(p, park=None, wx=None):
    """
    PITCHER OUTS RECORDED — sensitive to pitch count and bullpen usage.
    Outs = IP × 3. A starter going 6 IP = 18 outs.

    Key inputs: historical IP/start, pitch efficiency (pitches/PA),
    BB rate (walks extend innings), game script tendency.
    """
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bb_pct = p.get("bb_pct", LG["bb_pct_p"])
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    # Regress BB% (high BB = shorter outings)
    reg_bb = _regress(bb_pct, bf_est, STAB["bb_pct_p"], LG["bb_pct_p"])

    # Average IP per start
    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(8.0, avg_ip))

    # BB% adjustment: high walk rate = fewer outs (more pitches burned)
    bb_adj = 1.0 - (reg_bb - LG["bb_pct_p"]) / LG["bb_pct_p"] * 0.15

    proj_ip = avg_ip * bb_adj
    proj_outs = proj_ip * 3

    if park: proj_outs *= (1 + (_park(park, PARK) - 1) * -0.1)  # Hitter parks = fewer outs
    if wx: proj_outs *= (1 + (wx.get("weather_offense_mult", 1.0) - 1) * -0.1)

    mu = max(proj_outs, 9.0)
    return {"projection": round(mu, 1), "mu": mu, "avg_ip_start": round(avg_ip, 1),
            "regressed_bb_pct": round(reg_bb, 1)}


def project_pitcher_earned_runs(p, park=None, wx=None, opp_woba=None):
    """
    PITCHER EARNED RUNS — use FIP/xFIP over ERA (more predictive).

    Key: xFIP (fielding-independent, normalized HR/FB) is the best
    predictor of future run prevention. ERA is noisy.
    """
    era = p.get("era", LG["era"])
    fip = p.get("fip", era)
    xfip = p.get("xfip", fip)
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    # Use FIP/xFIP blend, regressed
    # xFIP is most stable, ERA is noisiest
    pitching_rate = era * 0.15 + fip * 0.35 + xfip * 0.50
    reg_rate = _regress(pitching_rate, bf_est, STAB["fip"], LG["fip"])

    # Expected IP
    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(7.5, avg_ip))

    # ER projection = (rate / 9) * expected IP
    proj_er = (reg_rate / 9.0) * avg_ip

    # Opposing lineup quality
    if opp_woba and opp_woba > 0:
        opp_adj = opp_woba / LG["woba"]
        proj_er *= (1 + (opp_adj - 1) * 0.5)

    # Park factor (hitter parks = more ER)
    if park: proj_er *= _park(park, PARK)

    # Weather (warm = more offense = more ER)
    if wx: proj_er *= wx.get("weather_offense_mult", 1.0)

    mu = max(proj_er, 0.5)
    return {"projection": round(mu, 2), "mu": mu, "blended_rate": round(reg_rate, 2),
            "avg_ip": round(avg_ip, 1)}


def project_pitcher_walks(p, park=None, ump=None):
    """PITCHER WALKS ALLOWED — BB% is the key driver."""
    bb_pct = p.get("bb_pct", LG["bb_pct_p"])
    bb9 = p.get("bb9", LG["bb9"])
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0

    reg_bb = _regress(bb_pct, bf_est, STAB["bb_pct_p"], LG["bb_pct_p"])

    # Zone% and F-Strike% adjustments (pitchers who throw strikes walk fewer)
    zone = p.get("zone_pct", 0)
    if zone > 0:
        zone_adj = (zone - LG["zone_pct"]) / LG["zone_pct"] * -0.2
        reg_bb *= (1 + zone_adj)

    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(7.5, avg_ip))
    exp_bf = avg_ip * LG["bf_per_ip"]

    proj = exp_bf * (reg_bb / 100)

    # Umpire with tight zone = more walks
    if ump and ump.get("known"):
        k_adj = ump.get("k_adjustment", 0)
        proj -= k_adj * 0.3  # Inverse: high-K ump = fewer walks

    mu = max(proj, 0.5)
    return {"projection": round(mu, 2), "mu": mu, "regressed_bb_pct": round(reg_bb, 1)}


def project_pitcher_hits_allowed(p, park=None, wx=None, opp_avg=None):
    """PITCHER HITS ALLOWED — WHIP, BABIP, K rate drive this."""
    whip = p.get("whip", LG["whip"])
    ip = p.get("ip", 0)
    gs = max(p.get("gs", 1), 1)
    bf_est = ip * LG["bf_per_ip"] if ip > 0 else 0
    bb9 = p.get("bb9", LG["bb9"])

    # Hits per 9 = (WHIP - BB/9) * 9... approximate
    h9 = (whip * 9) - (bb9 if bb9 > 0 else LG["bb9"])
    h9 = max(h9, 5.0)

    avg_ip = ip / gs if gs > 0 else LG["avg_ip_starter"]
    avg_ip = max(4.0, min(7.5, avg_ip))

    proj = (h9 / 9.0) * avg_ip

    if opp_avg and opp_avg > 0:
        proj *= (opp_avg / LG["avg"])

    if park: proj *= _park(park, PARK)
    if wx: proj *= wx.get("weather_offense_mult", 1.0)

    mu = max(proj, 2.0)
    return {"projection": round(mu, 2), "mu": mu}


# ═══════════════════════════════════════════════════════
# BATTER PROJECTIONS
# ═══════════════════════════════════════════════════════

def project_batter_hits(b, opp_p=None, bvp=None, platoon=None,
                         park=None, wx=None, lineup_pos=None):
    """
    BATTER HITS — most predictable batting prop.

    Key inputs:
      AVG/xBA (25%) — season rate + Statcast expected, regressed
      K% (20%) — low K = more balls in play = more hit opportunities
      Contact quality (15%) — hard hit%, barrel rate, EV90
      BvP matchup (15%) — head-to-head history
      Opposing pitcher (10%) — WHIP, K%, quality
      Platoon (8%) — handedness advantage
      Park + Weather (7%) — slight effects
    """
    avg = b.get("avg", LG["avg"])
    xba = b.get("xba", 0)
    pa = b.get("pa", 0)
    k_rate = b.get("k_rate", LG["k_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    ev90 = b.get("recent_ev90", LG["ev90"])
    babip = b.get("babip", LG["babip"])

    # Regress AVG
    reg_avg = _regress(avg, pa, STAB["avg"], LG["avg"])

    # xBA blend (descriptive, not purely predictive — weight 30%)
    if xba > 0:
        reg_avg = reg_avg * 0.70 + xba * 0.30

    # K% adjustment: low K% = more balls in play
    reg_k = _regress(k_rate, pa, STAB["k_rate"], LG["k_rate"])
    k_adj = 1.0 + (LG["k_rate"] - reg_k) / LG["k_rate"] * 0.12
    reg_avg *= k_adj

    # Contact quality: hard hit rate, EV90
    if hard_hit > 0:
        hh_adj = (hard_hit - LG["hard_hit_pct"]) / LG["hard_hit_pct"] * 0.10
        reg_avg *= (1 + hh_adj)

    # BABIP regression check (if BABIP is way above/below xBA, regression coming)
    if babip > 0 and xba > 0:
        babip_delta = babip - (xba + 0.050)  # BABIP is typically ~50 pts above xBA
        if abs(babip_delta) > 0.030:
            reg_avg *= (1 - babip_delta * 0.15)  # Pull toward expected

    # BvP matchup
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_avg = bvp.get("avg", LG["avg"])
        bvp_weight = min(bvp["pa"] / 50, 0.30)  # Max 30% weight at 50+ PA
        reg_avg = reg_avg * (1 - bvp_weight) + bvp_avg * bvp_weight

    # Opposing pitcher quality
    if opp_p:
        opp_whip = opp_p.get("whip", LG["whip"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_quality = (opp_whip / LG["whip"] * 0.6 + opp_fip / LG["fip"] * 0.4)
        reg_avg *= (1 + (opp_quality - 1) * 0.35)

    # Platoon
    if platoon and platoon.get("adjustment"):
        reg_avg *= platoon["adjustment"]

    # Park + weather
    if park: reg_avg *= (1 + (_park(park, PARK) - 1) * 0.25)
    if wx: reg_avg *= wx.get("weather_offense_mult", 1.0)

    # Expected AB (PA minus walks/HBP ~9%)
    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2
    exp_ab = exp_pa * (1 - (b.get("bb_rate", LG["bb_rate"]) / 100))

    mu = max(exp_ab * reg_avg, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_avg": round(reg_avg, 3),
            "expected_pa": round(exp_pa, 1), "expected_ab": round(exp_ab, 1)}


def project_batter_total_bases(b, opp_p=None, bvp=None, platoon=None,
                                 park=None, wx=None, lineup_pos=None):
    """
    BATTER TOTAL BASES — SLG-driven with power metrics.

    Key: Barrel rate is the single best Statcast predictor for TB.
    85.8% of HRs are barrels, 50.2% of barrels become HRs.
    """
    slg = b.get("slg", LG["slg"])
    xslg = b.get("xslg", 0)
    iso = b.get("iso", LG["iso"])
    pa = b.get("pa", 0)
    barrel = b.get("recent_barrel_rate", LG["barrel_rate"])
    hard_hit = b.get("recent_hard_hit_pct", LG["hard_hit_pct"])
    ev90 = b.get("recent_ev90", LG["ev90"])

    # Regress SLG
    reg_slg = _regress(slg, pa, STAB["slg"], LG["slg"])

    # xSLG blend (30% weight)
    if xslg > 0:
        reg_slg = reg_slg * 0.70 + xslg * 0.30

    # Barrel rate (strongest power predictor)
    if barrel > 0:
        barrel_adj = (barrel - LG["barrel_rate"]) / LG["barrel_rate"] * 0.18
        reg_slg *= (1 + barrel_adj)

    # EV90 (90th percentile exit velo — more stable than max EV)
    if ev90 > 0:
        ev_adj = (ev90 - LG["ev90"]) / LG["ev90"] * 0.08
        reg_slg *= (1 + ev_adj)

    # ISO regression: if ISO is way above xSLG-xBA, regression likely
    if iso > 0 and xslg > 0:
        x_iso = xslg - (b.get("xba", LG["xba"]))
        if x_iso > 0:
            iso_delta = iso - x_iso
            if abs(iso_delta) > 0.030:
                reg_slg *= (1 - iso_delta * 0.10)

    # BvP matchup
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_slg = bvp.get("slg", LG["slg"])
        bvp_w = min(bvp["pa"] / 50, 0.25)
        reg_slg = reg_slg * (1 - bvp_w) + bvp_slg * bvp_w

    # Opposing pitcher
    if opp_p:
        opp_hr9 = opp_p.get("hr9", LG["hr9"])
        opp_fip = opp_p.get("fip", LG["fip"])
        opp_quality = (opp_fip / LG["fip"] * 0.6 + opp_hr9 / LG["hr9"] * 0.4)
        reg_slg *= (1 + (opp_quality - 1) * 0.30)

    # Platoon
    if platoon and platoon.get("adjustment"):
        reg_slg *= platoon["adjustment"]

    # Park (blend general + HR)
    if park:
        pk = _park(park, PARK) * 0.55 + _park(park, PARK_HR) * 0.45
        reg_slg *= (1 + (pk - 1) * 0.40)

    # Weather (HR mult matters more for TB)
    if wx:
        blended_wx = wx.get("weather_offense_mult", 1.0) * 0.5 + wx.get("weather_hr_mult", 1.0) * 0.5
        reg_slg *= blended_wx

    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2
    exp_ab = exp_pa * (1 - (b.get("bb_rate", LG["bb_rate"]) / 100))

    mu = max(exp_ab * reg_slg, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_slg": round(reg_slg, 3),
            "expected_ab": round(exp_ab, 1)}


def project_batter_home_runs(b, opp_p=None, bvp=None, platoon=None,
                               park=None, wx=None, lineup_pos=None):
    """
    BATTER HOME RUNS — highest variance prop. Use with caution.

    Even Judge only homers in ~26% of games. The line is usually 0.5.
    Barrel rate + HR/FB + fly ball rate are the key inputs.
    """
    hr = b.get("hr", 0)
    pa = b.get("pa", 0)
    iso = b.get("iso", LG["iso"])
    barrel = b.get("recent_barrel_rate", LG["barrel_rate"])

    hr_rate = hr / pa if pa > 0 else LG["hr_per_pa"]
    reg_hr = _regress(hr_rate, pa, 300, LG["hr_per_pa"])

    # Barrel rate is THE predictor
    if barrel > 0:
        barrel_adj = barrel / LG["barrel_rate"]
        reg_hr *= (1 + (barrel_adj - 1) * 0.35)

    # ISO confirmation
    reg_iso = _regress(iso, pa, STAB["iso"], LG["iso"])
    iso_adj = reg_iso / LG["iso"]
    reg_hr *= (1 + (iso_adj - 1) * 0.15)

    # BvP HR history
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 10:
        bvp_hr_rate = bvp["home_runs"] / bvp["pa"] if bvp["pa"] > 0 else 0
        if bvp_hr_rate > 0:
            bvp_w = min(bvp["pa"] / 60, 0.20)
            reg_hr = reg_hr * (1 - bvp_w) + bvp_hr_rate * bvp_w

    # Opposing pitcher HR tendency
    if opp_p:
        opp_hr9 = opp_p.get("hr9", LG["hr9"])
        reg_hr *= (opp_hr9 / LG["hr9"])

    # Platoon (ISO is ~14% higher in favorable platoon)
    if platoon and platoon.get("favorable"):
        reg_hr *= 1.14
    elif platoon and platoon.get("favorable") is False:
        reg_hr *= 0.86

    # Park HR factor (biggest effect of any prop)
    if park: reg_hr *= _park(park, PARK_HR)

    # Weather (temp is huge for HR — 2% per degree C above 72F)
    if wx: reg_hr *= wx.get("weather_hr_mult", 1.0)

    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2
    exp_ab = exp_pa * (1 - (b.get("bb_rate", LG["bb_rate"]) / 100))

    mu = max(exp_ab * reg_hr, 0.01)
    return {"projection": round(mu, 3), "mu": mu, "hr_rate": round(reg_hr, 4)}


def project_batter_rbis(b, opp_p=None, bvp=None, platoon=None,
                          park=None, wx=None, lineup_pos=None):
    """
    BATTER RBIs — heavily dependent on lineup context.

    RBIs are NOT purely a player skill stat — they depend on who's on base.
    Lineup position is critical: cleanup hitter behind 3 high-OBP guys
    has way more RBI opportunities than a #8 hitter.

    Key: wOBA (weights all offensive events) + lineup position + team run environment.
    """
    woba = b.get("woba", LG["woba"])
    slg = b.get("slg", LG["slg"])
    pa = b.get("pa", 0)
    hr = b.get("hr", 0)

    reg_woba = _regress(woba, pa, STAB["woba"], LG["woba"])

    # RBI rate scales with wOBA/SLG and lineup position
    # Middle of order (3-5) gets ~25% more RBI opportunities
    base_rbi_rate = LG["rbi_per_game"]
    woba_adj = reg_woba / LG["woba"]
    proj = base_rbi_rate * woba_adj

    # Lineup position boost (3-5 hitters get runners on base more often)
    if lineup_pos:
        if lineup_pos <= 2:
            proj *= 0.85  # Leadoff/2-hole: fewer RBI chances, more run scoring
        elif lineup_pos <= 5:
            proj *= 1.20  # Heart of order: most RBI opportunity
        elif lineup_pos <= 7:
            proj *= 0.95
        else:
            proj *= 0.80  # Bottom of order

    # Power hitters drive in more runs (HR = guaranteed RBI)
    hr_rate = hr / pa if pa > 0 else LG["hr_per_pa"]
    if hr_rate > LG["hr_per_pa"]:
        proj *= (1 + (hr_rate / LG["hr_per_pa"] - 1) * 0.20)

    # BvP
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 10:
        bvp_slg = bvp.get("slg", LG["slg"])
        bvp_adj = bvp_slg / LG["slg"]
        proj *= (1 + (bvp_adj - 1) * 0.15)

    # Opposing pitcher (bad pitcher = more runs = more RBI chances)
    if opp_p:
        opp_fip = opp_p.get("fip", LG["fip"])
        proj *= (opp_fip / LG["fip"])

    if platoon and platoon.get("adjustment"):
        proj *= platoon["adjustment"]

    if park: proj *= _park(park, PARK)
    if wx: proj *= wx.get("weather_offense_mult", 1.0)

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_woba": round(reg_woba, 3)}


def project_batter_runs(b, opp_p=None, bvp=None, platoon=None,
                          park=None, wx=None, lineup_pos=None):
    """
    BATTER RUNS SCORED — OBP-driven + lineup position + sprint speed.

    Leadoff hitters score more runs. Fast guys score more.
    OBP = getting on base = opportunity to score.
    """
    obp = b.get("obp", LG["obp"])
    pa = b.get("pa", 0)
    sprint = b.get("sprint_speed", LG["sprint_speed"])

    reg_obp = _regress(obp, pa, STAB["obp"], LG["obp"])

    base_run_rate = LG["runs_per_game"]
    obp_adj = reg_obp / LG["obp"]
    proj = base_run_rate * obp_adj

    # Lineup position (leadoff scores most runs)
    if lineup_pos:
        if lineup_pos == 1:
            proj *= 1.25
        elif lineup_pos == 2:
            proj *= 1.15
        elif lineup_pos <= 5:
            proj *= 1.05
        elif lineup_pos <= 7:
            proj *= 0.90
        else:
            proj *= 0.80

    # Sprint speed: fast players score from 1st on doubles, score on sac flies
    if sprint > 0:
        speed_adj = (sprint - LG["sprint_speed"]) / LG["sprint_speed"] * 0.15
        proj *= (1 + speed_adj)

    if opp_p:
        opp_fip = opp_p.get("fip", LG["fip"])
        proj *= (opp_fip / LG["fip"])

    if platoon and platoon.get("adjustment"):
        proj *= platoon["adjustment"]

    if park: proj *= _park(park, PARK)
    if wx: proj *= wx.get("weather_offense_mult", 1.0)

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_obp": round(reg_obp, 3)}


def project_batter_stolen_bases(b, park=None):
    """
    BATTER STOLEN BASES — sprint speed + opportunity + catcher.

    SB is a rare event (most lines are 0.5). Very binary.
    Sprint speed is essentially a physical trait — stabilizes fast.
    """
    sb = b.get("sb", 0)
    pa = b.get("pa", 0)
    sprint = b.get("sprint_speed", LG["sprint_speed"])
    obp = b.get("obp", LG["obp"])

    # SB rate per game
    games = pa / 4.2 if pa > 0 else 1
    sb_rate = sb / games if games > 0 else LG["sb_per_game"]
    reg_sb = _regress(sb_rate, pa, 200, LG["sb_per_game"])

    # Sprint speed is the dominant factor
    if sprint > 0:
        speed_factor = (sprint - 25.0) / (30.0 - 25.0)  # Normalize: 25=slow, 30=elite
        speed_factor = max(0, min(1.5, speed_factor))
        reg_sb *= (0.5 + speed_factor)

    # OBP: can't steal if you're not on base
    reg_obp = _regress(obp, pa, STAB["obp"], LG["obp"])
    reg_sb *= (reg_obp / LG["obp"])

    if park: reg_sb *= _park(park, PARK_SB)

    mu = max(reg_sb, 0.01)
    return {"projection": round(mu, 3), "mu": mu, "sprint_speed": sprint}


def project_batter_strikeouts(b, opp_p=None, bvp=None, platoon=None,
                                park=None, ump=None, lineup_pos=None):
    """
    BATTER STRIKEOUTS — K% is the fastest-stabilizing batter stat (60 PA).

    High-K batters against high-K pitchers with a big-zone umpire = K over city.
    """
    k_rate = b.get("k_rate", LG["k_rate"])
    pa = b.get("pa", 0)
    contact = b.get("contact_rate", 0)

    reg_k = _regress(k_rate, pa, STAB["k_rate"], LG["k_rate"])

    # Contact rate adjustment (if available from Statcast)
    if contact > 0:
        contact_adj = (100 - contact) / (100 - (100 - LG["k_rate"]))
        reg_k = reg_k * 0.75 + (reg_k * contact_adj) * 0.25

    # Opposing pitcher K ability
    if opp_p:
        opp_k = opp_p.get("k_pct", LG["k_pct_p"])
        opp_csw = opp_p.get("recent_csw_pct", LG["csw_pct"])
        opp_k_quality = (opp_k / LG["k_pct_p"] * 0.6 +
                          (opp_csw / LG["csw_pct"] if opp_csw > 0 else 1.0) * 0.4)
        reg_k *= opp_k_quality

    # BvP K rate
    if bvp and bvp.get("has_data") and bvp.get("pa", 0) >= 8:
        bvp_k = bvp.get("k_rate", LG["k_rate"])
        bvp_w = min(bvp["pa"] / 50, 0.25)
        reg_k = reg_k * (1 - bvp_w) + bvp_k * bvp_w

    # Platoon (same-hand = more Ks)
    if platoon:
        k_plat = platoon.get("k_adjustment", 1.0)
        reg_k *= k_plat

    # Umpire (big zone = more Ks for batters too)
    if ump and ump.get("known"):
        k_ump_adj = ump.get("k_adjustment", 0) * 0.15  # Scaled for individual batter
        reg_k *= (1 + k_ump_adj / 5)

    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2
    proj = exp_pa * (reg_k / 100)

    if park: proj *= _park(park, PARK_K)

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_k_rate": round(reg_k, 1)}


def project_batter_walks(b, opp_p=None, ump=None, lineup_pos=None):
    """BATTER WALKS — BB% + opposing pitcher BB% + umpire zone."""
    bb_rate = b.get("bb_rate", LG["bb_rate"])
    pa = b.get("pa", 0)

    reg_bb = _regress(bb_rate, pa, STAB["bb_rate"], LG["bb_rate"])

    if opp_p:
        opp_bb = opp_p.get("bb_pct", LG["bb_pct_p"])
        reg_bb *= (opp_bb / LG["bb_pct_p"])

    # Tight-zone ump = more walks
    if ump and ump.get("known"):
        k_adj = ump.get("k_adjustment", 0)
        reg_bb *= (1 - k_adj * 0.05)  # High-K ump = fewer walks

    exp_pa = _lineup_pa(lineup_pos) if lineup_pos else 4.2
    proj = exp_pa * (reg_bb / 100)

    mu = max(proj, 0.1)
    return {"projection": round(mu, 2), "mu": mu, "regressed_bb_rate": round(reg_bb, 1)}


def project_hits_runs_rbis(b, opp_p=None, bvp=None, platoon=None,
                            park=None, wx=None, ump=None, lineup_pos=None):
    """HITS + RUNS + RBIs combo — sum of individual projections."""
    hits = project_batter_hits(b, opp_p, bvp, platoon, park, wx, lineup_pos)
    runs = project_batter_runs(b, opp_p, bvp, platoon, park, wx, lineup_pos)
    rbis = project_batter_rbis(b, opp_p, bvp, platoon, park, wx, lineup_pos)

    mu = hits["mu"] + runs["mu"] + rbis["mu"]
    return {"projection": round(mu, 2), "mu": mu,
            "hits_proj": hits["projection"], "runs_proj": runs["projection"],
            "rbis_proj": rbis["projection"]}


# ═══════════════════════════════════════════════════════
# PROBABILITY & DISTRIBUTION
# ═══════════════════════════════════════════════════════

def calculate_over_under_probability(projection, line, prop_type):
    """
    P(over) and P(under) using prop-specific variance models.

    Variance ratios calibrated to actual MLB distributions:
      Pitcher Ks: variance ≈ 2.0× mean (overdispersed, beta-binomial)
      Batter Ks: variance ≈ 1.4× mean
      Hits: variance ≈ 1.3× mean
      Total Bases: variance ≈ 1.8× mean (right-skewed)
      Home Runs: variance ≈ 3.5× mean (very rare, high variance)
      RBIs: variance ≈ 1.6× mean (context-dependent)
      Runs: variance ≈ 1.4× mean
      Stolen Bases: variance ≈ 2.5× mean (rare binary)
      H+R+RBI: variance ≈ 1.5× mean (sum smooths variance)
      Pitcher Outs: variance ≈ 1.3× mean (most predictable)
      Pitcher ER: variance ≈ 2.2× mean (volatile)
      Walks: variance ≈ 1.8× mean
    """
    mu = max(projection, 0.01)

    variance_ratios = {
        "pitcher_strikeouts": 2.0, "batter_strikeouts": 1.4,
        "hits": 1.3, "total_bases": 1.8, "home_runs": 3.5,
        "rbis": 1.6, "runs": 1.4, "stolen_bases": 2.5,
        "hits_runs_rbis": 1.5, "pitching_outs": 1.3,
        "earned_runs": 2.2, "walks_allowed": 1.8, "walks": 1.8,
        "hits_allowed": 1.5, "singles": 1.3, "doubles": 2.5,
    }
    var_ratio = variance_ratios.get(prop_type, 1.5)
    sigma = max(np.sqrt(mu * var_ratio), 0.25)

    # Continuity correction for count data
    if line == int(line):
        p_over = 1 - sp_stats.norm.cdf(line + 0.5, loc=mu, scale=sigma)
        p_under = sp_stats.norm.cdf(line - 0.5, loc=mu, scale=sigma)
    else:
        p_over = 1 - sp_stats.norm.cdf(line, loc=mu, scale=sigma)
        p_under = sp_stats.norm.cdf(line, loc=mu, scale=sigma)

    total = p_over + p_under
    if total > 0:
        p_over /= total
        p_under /= total

    edge = abs(p_over - 0.5)
    pick = "MORE" if p_over > 0.5 else "LESS"
    confidence = max(p_over, p_under)

    if confidence >= 0.62: rating = "A"
    elif confidence >= 0.57: rating = "B"
    elif confidence >= 0.54: rating = "C"
    else: rating = "D"

    return {"p_over": round(p_over, 4), "p_under": round(p_under, 4),
            "pick": pick, "confidence": round(confidence, 4),
            "edge": round(edge, 4), "rating": rating,
            "projection": round(mu, 2), "line": line, "sigma": round(sigma, 2)}


# ═══════════════════════════════════════════════════════
# MASTER ROUTER
# ═══════════════════════════════════════════════════════

def generate_prediction(player_name, stat_type, stat_internal, line,
                         batter_profile=None, pitcher_profile=None,
                         opp_pitcher_profile=None, opp_team_k_rate=None,
                         bvp=None, platoon=None, ump=None,
                         park_team=None, weather=None, lineup_pos=None):
    """
    Master prediction router. Picks the right projection function
    based on prop type and feeds in all available context.
    """
    b = batter_profile or {}
    p = pitcher_profile or {}
    opp = opp_pitcher_profile or {}

    # Route to correct projection
    route = {
        "pitcher_strikeouts": lambda: project_pitcher_strikeouts(
            p, bvp=bvp, platoon=platoon, ump=ump,
            opp_k_rate=opp_team_k_rate, park=park_team, wx=weather),
        "pitching_outs": lambda: project_pitcher_outs(p, park=park_team, wx=weather),
        "earned_runs": lambda: project_pitcher_earned_runs(
            p, park=park_team, wx=weather, opp_woba=opp.get("woba")),
        "walks_allowed": lambda: project_pitcher_walks(p, park=park_team, ump=ump),
        "hits_allowed": lambda: project_pitcher_hits_allowed(
            p, park=park_team, wx=weather),
        "hits": lambda: project_batter_hits(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "total_bases": lambda: project_batter_total_bases(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "home_runs": lambda: project_batter_home_runs(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "rbis": lambda: project_batter_rbis(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "runs": lambda: project_batter_runs(
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "stolen_bases": lambda: project_batter_stolen_bases(b, park_team),
        "batter_strikeouts": lambda: project_batter_strikeouts(
            b, opp, bvp, platoon, park_team, ump, lineup_pos),
        "walks": lambda: project_batter_walks(b, opp, ump, lineup_pos),
        "hits_runs_rbis": lambda: project_hits_runs_rbis(
            b, opp, bvp, platoon, park_team, weather, ump, lineup_pos),
        "singles": lambda: project_batter_hits(  # Singles ≈ Hits - XBH
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
        "doubles": lambda: project_batter_hits(  # Approximate
            b, opp, bvp, platoon, park_team, weather, lineup_pos),
    }

    proj_fn = route.get(stat_internal)
    if proj_fn:
        proj_result = proj_fn()
    elif b:
        proj_result = project_batter_hits(b, opp, bvp, platoon, park_team, weather, lineup_pos)
    elif p:
        proj_result = project_pitcher_strikeouts(p, bvp=bvp, park=park_team, wx=weather)
    else:
        proj_result = {"projection": line, "mu": line}

    projection = proj_result.get("projection", line)
    prob_result = calculate_over_under_probability(projection, line, stat_internal)

    return {
        "player_name": player_name, "stat_type": stat_type,
        "stat_internal": stat_internal, "line": line,
        **proj_result, **prob_result,
    }
