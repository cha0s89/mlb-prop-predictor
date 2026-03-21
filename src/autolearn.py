"""
Self-Learning Weight Adjustment System — autolearn.py

Analyzes graded predictions (from backtest or live results), detects systematic
biases and calibration problems, and automatically adjusts model weights to
improve future accuracy.

Design philosophy:
  - Conservative: only adjusts after sufficient sample size (25+ graded picks)
  - Every adjustment is versioned and logged with full reasoning
  - Kill switch: if accuracy drops below 45% after any adjustment, auto-rollback
    to previous version AND log what was tried so it is not repeated
  - Learns from failures: rolled-back adjustments are recorded as "do not retry"

Weight categories (the knobs):
  1. Confidence thresholds — Grade boundary cutoffs (A/B/C/D)
  2. Direction bias correction — MORE vs LESS multiplier
  3. Prop type multipliers — Per-stat-type projection offset
  4. Factor weights — How much to trust each input signal
  5. Variance ratios — For over/under probability calculations

Usage:
  from src.autolearn import run_adjustment_cycle, load_current_weights
  result = run_adjustment_cycle(min_sample=25)
  weights = load_current_weights()

CLI:
  python -m src.autolearn
"""

import json
import copy
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.database import get_graded_predictions

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════

WEIGHTS_DIR = Path("data/weights")
CURRENT_WEIGHTS_PATH = WEIGHTS_DIR / "current.json"
WEIGHT_HISTORY_PATH = WEIGHTS_DIR / "weight_history.json"

# Maximum percentage change per adjustment cycle (±10%)
MAX_ADJUSTMENT_PCT = 0.10

# Minimum graded picks required before any adjustment
MIN_SAMPLE_DEFAULT = 25

# Kill switch: rollback if accuracy falls below this after an adjustment
KILL_SWITCH_THRESHOLD = 0.48

# Minimum picks to evaluate kill switch after a new weight version
KILL_SWITCH_EVAL_SIZE = 25


# ═══════════════════════════════════════════════════════
# BASELINE WEIGHTS (mirrors predictor.py constants)
# ═══════════════════════════════════════════════════════

def get_baseline_weights() -> dict:
    """
    Return the default model weights derived from predictor.py constants.

    These are the starting point before any learning occurs. Every tunable
    parameter that autolearn can adjust is represented here.

    Returns:
        dict with keys: confidence_thresholds, direction_bias,
        prop_type_offsets, factor_weights, variance_ratios, version, metadata.
    """
    return {
        "version": "v003",
        "description": "Baseline weights from predictor.py defaults",
        "created_at": datetime.now(timezone.utc).isoformat(),

        # Grade boundary cutoffs (confidence thresholds)
        # A pick gets grade X if confidence >= threshold
        "confidence_thresholds": {
            "A": 0.70,
            "B": 0.62,
            "C": 0.57,
            "D": 0.00,  # Everything below C
        },

        # Direction bias correction multiplier
        # Applied to projection before probability calc
        # >1.0 nudges toward MORE, <1.0 nudges toward LESS
        "direction_bias": {
            "more_multiplier": 1.0,
            "less_multiplier": 1.0,
        },

        # Per-prop-type projection offsets (added to raw projection)
        # Positive = model was projecting too low, Negative = too high
        "prop_type_offsets": {
            "pitcher_strikeouts": 0.0,
            "batter_strikeouts": 0.0,
            "hits": 0.0,
            "total_bases": 0.0,
            "home_runs": 0.0,
            "rbis": 0.0,
            "runs": 0.0,
            "stolen_bases": 0.0,
            "hitter_fantasy_score": 0.0,
            "hits_runs_rbis": 0.0,
            "pitching_outs": 0.0,
            "earned_runs": 0.0,
            "walks_allowed": 0.0,
            "walks": 0.0,
            "hits_allowed": 0.0,
            "singles": 0.0,
            "doubles": 0.0,
        },

        # Factor weights: how much to trust each input signal
        # 1.0 = default trust. Adjusted based on whether the factor
        # improves or hurts prediction accuracy.
        "factor_weights": {
            "statcast_blend": 1.0,   # xBA/xSLG blend weight
            "bvp_matchup": 1.0,     # Batter vs pitcher history
            "platoon_split": 1.0,   # Handedness advantage
            "park_factor": 1.0,     # Park effects
            "weather": 1.0,         # Weather adjustments
            "umpire": 1.0,          # Umpire tendencies
            "opposing_quality": 1.0, # Opposing pitcher/lineup quality
            "barrel_rate": 1.0,     # Statcast barrel rate signal
        },

        # Variance ratios for probability calculation
        # Higher = wider distribution = less confident predictions
        "variance_ratios": {
            "pitcher_strikeouts": 2.0,
            "batter_strikeouts": 1.4,
            "hits": 1.3,
            "total_bases": 1.8,
            "home_runs": 3.5,
            "rbis": 1.6,
            "runs": 1.4,
            "stolen_bases": 2.5,
            "hitter_fantasy_score": 1.6,
            "hits_runs_rbis": 1.5,
            "pitching_outs": 1.3,
            "earned_runs": 2.2,
            "walks_allowed": 1.8,
            "walks": 1.8,
            "hits_allowed": 1.5,
            "singles": 1.3,
            "doubles": 2.5,
        },

        # Metadata for tracking
        "metadata": {
            "parent_version": None,
            "adjustment_count": 0,
            "total_picks_analyzed": 0,
            "accuracy_at_creation": None,
        },
    }


# ═══════════════════════════════════════════════════════
# WEIGHT I/O
# ═══════════════════════════════════════════════════════

def _ensure_dirs() -> None:
    """Create the weights directory structure if it does not exist."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def load_current_weights() -> dict:
    """
    Load the active weight configuration from data/weights/current.json.

    If no current.json exists, creates one from baseline defaults.

    Returns:
        dict: The active weight configuration.
    """
    _ensure_dirs()
    if CURRENT_WEIGHTS_PATH.exists():
        try:
            with open(CURRENT_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                weights = json.load(f)
            # Validate structure — fill in any missing keys from baseline
            baseline = get_baseline_weights()
            for top_key in baseline:
                if top_key not in weights:
                    weights[top_key] = baseline[top_key]
                elif isinstance(baseline[top_key], dict) and isinstance(weights.get(top_key), dict):
                    for sub_key in baseline[top_key]:
                        if sub_key not in weights[top_key]:
                            weights[top_key][sub_key] = baseline[top_key][sub_key]
            return weights
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load current.json, reverting to baseline: %s", e)

    # No current weights — initialize from baseline
    weights = get_baseline_weights()
    save_weights(weights, "v001", "Baseline weights initialized")
    return weights


def save_weights(weights: dict, version: str, description: str) -> str:
    """
    Save a new weight version and update current.json.

    Creates a versioned snapshot file (e.g., v002_post_backtest.json)
    and copies it to current.json as the active configuration.

    Args:
        weights: The weight configuration dict.
        version: Version tag (e.g., 'v002').
        description: Human-readable description of this version.

    Returns:
        str: The file path of the saved versioned weights.
    """
    _ensure_dirs()
    weights = copy.deepcopy(weights)
    weights["version"] = version
    weights["description"] = description
    weights["created_at"] = datetime.now(timezone.utc).isoformat()

    # Save versioned copy
    safe_desc = description.lower().replace(" ", "_")[:40]
    safe_desc = "".join(c for c in safe_desc if c.isalnum() or c == "_")
    version_filename = f"{version}_{safe_desc}.json"
    version_path = WEIGHTS_DIR / version_filename

    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, default=str)

    # Update current.json
    with open(CURRENT_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, default=str)

    logger.info("Saved weights %s: %s -> %s", version, description, version_path)
    return str(version_path)


def _next_version() -> str:
    """
    Determine the next version number by scanning existing weight files.

    Returns:
        str: Next version tag like 'v003'.
    """
    _ensure_dirs()
    max_v = 1
    for p in WEIGHTS_DIR.glob("v*.json"):
        name = p.stem
        # Extract version number from filenames like v002_post_backtest
        parts = name.split("_")
        if parts and parts[0].startswith("v"):
            try:
                num = int(parts[0][1:])
                max_v = max(max_v, num)
            except ValueError:
                pass
    return f"v{max_v + 1:03d}"


# ═══════════════════════════════════════════════════════
# WEIGHT HISTORY LOG
# ═══════════════════════════════════════════════════════

def get_weight_history() -> list[dict]:
    """
    Load the full adjustment history log.

    Returns:
        list[dict]: Each entry has timestamp, version, changes, accuracy,
        sample_size, status (applied/rolled_back), and reason.
    """
    _ensure_dirs()
    if WEIGHT_HISTORY_PATH.exists():
        try:
            with open(WEIGHT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _append_history(entry: dict) -> None:
    """Append an entry to the weight history log."""
    _ensure_dirs()
    history = get_weight_history()
    history.append(entry)
    with open(WEIGHT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


def _get_failed_adjustments() -> list[dict]:
    """
    Return all adjustments that were rolled back, so they are not retried.

    The system learns from its failures by never repeating an adjustment
    that caused accuracy to drop.

    Returns:
        list[dict]: Entries from history where status == 'rolled_back'.
    """
    history = get_weight_history()
    return [h for h in history if h.get("status") == "rolled_back"]


# ═══════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════

def analyze_direction_bias(graded: pd.DataFrame) -> dict:
    """
    Analyze whether the model has a systematic bias toward MORE or LESS picks.

    If MORE picks win at 58% but LESS wins at 47%, the model is biased toward
    over-projecting, and a correction multiplier is needed.

    Args:
        graded: DataFrame of graded predictions with 'pick' and 'result' columns.

    Returns:
        dict with keys: more_accuracy, less_accuracy, more_total, less_total,
        bias_detected (bool), suggested_correction.
    """
    result = {
        "more_accuracy": None, "less_accuracy": None,
        "more_total": 0, "less_total": 0,
        "bias_detected": False, "suggested_correction": None,
    }

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return result

    for direction in ["MORE", "LESS"]:
        subset = wl[wl["pick"] == direction]
        total = len(subset)
        wins = len(subset[subset["result"] == "W"])
        acc = wins / total if total > 0 else 0.0

        if direction == "MORE":
            result["more_accuracy"] = round(acc, 4)
            result["more_total"] = total
        else:
            result["less_accuracy"] = round(acc, 4)
            result["less_total"] = total

    # Detect bias if both directions have enough data and differ by >5%
    if result["more_total"] >= 10 and result["less_total"] >= 10:
        more_acc = result["more_accuracy"]
        less_acc = result["less_accuracy"]
        if more_acc is not None and less_acc is not None:
            diff = more_acc - less_acc
            if abs(diff) > 0.05:
                result["bias_detected"] = True
                # If MORE wins more, projections may be too high —
                # reduce more_multiplier (pull projections down slightly)
                # so fewer picks land on MORE, or equivalently nudge LESS up
                if diff > 0:
                    # MORE is winning more: projections are already good for
                    # overs but bad for unders. Nudge less_multiplier up.
                    correction = min(abs(diff) * 0.5, MAX_ADJUSTMENT_PCT)
                    result["suggested_correction"] = {
                        "target": "less_multiplier",
                        "direction": "increase",
                        "amount": round(correction, 4),
                        "reason": (
                            f"MORE picks hitting {more_acc:.1%} vs LESS at "
                            f"{less_acc:.1%}. Increasing LESS multiplier by "
                            f"{correction:.1%} to balance."
                        ),
                    }
                else:
                    correction = min(abs(diff) * 0.5, MAX_ADJUSTMENT_PCT)
                    result["suggested_correction"] = {
                        "target": "more_multiplier",
                        "direction": "increase",
                        "amount": round(correction, 4),
                        "reason": (
                            f"LESS picks hitting {less_acc:.1%} vs MORE at "
                            f"{more_acc:.1%}. Increasing MORE multiplier by "
                            f"{correction:.1%} to balance."
                        ),
                    }

    return result


def analyze_prop_type_accuracy(graded: pd.DataFrame) -> dict:
    """
    Analyze accuracy and projection bias per prop type.

    Detects which prop types the model consistently over- or under-projects,
    and suggests per-type offset corrections.

    Args:
        graded: DataFrame of graded predictions with 'stat_internal',
                'projection', 'actual_result', and 'result' columns.

    Returns:
        dict keyed by stat_internal with accuracy, mean_error, sample_size,
        and suggested_offset for each prop type.
    """
    result = {}
    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return result

    for stat_type in wl["stat_internal"].unique():
        subset = wl[wl["stat_internal"] == stat_type]
        total = len(subset)
        wins = len(subset[subset["result"] == "W"])
        acc = wins / total if total > 0 else 0.0

        # Calculate mean projection error (projection - actual)
        has_actual = subset.dropna(subset=["actual_result", "projection"])
        mean_error = 0.0
        if not has_actual.empty:
            errors = has_actual["projection"] - has_actual["actual_result"]
            mean_error = float(errors.mean())

        suggested_offset = None
        # Only suggest offset if we have enough data and error is significant
        if total >= 10 and abs(mean_error) > 0.25:
            # Offset goes opposite direction of error: if projecting too high,
            # subtract. Cap at ±10% of the mean projection for that type.
            mean_proj = float(subset["projection"].mean()) if not subset.empty else 1.0
            max_offset = abs(mean_proj) * MAX_ADJUSTMENT_PCT
            offset = -mean_error * 0.5  # Apply half the error as correction
            offset = np.clip(offset, -max_offset, max_offset)
            suggested_offset = {
                "offset": round(float(offset), 3),
                "reason": (
                    f"{stat_type}: mean error {mean_error:+.2f} over {total} picks "
                    f"(accuracy {acc:.1%}). Applying offset {offset:+.3f}."
                ),
            }

        result[stat_type] = {
            "accuracy": round(acc, 4),
            "total": total,
            "wins": wins,
            "mean_error": round(mean_error, 3),
            "suggested_offset": suggested_offset,
        }

    return result


def analyze_grade_calibration(graded: pd.DataFrame) -> dict:
    """
    Check whether grade thresholds are actually predictive.

    If A-grade picks only hit 52%, the A threshold is too loose.
    If C-grade picks hit 60%, the C threshold is too tight.
    Grades should be monotonically ordered: A > B > C > D in accuracy.

    Args:
        graded: DataFrame of graded predictions with 'rating' and 'result'.

    Returns:
        dict with per-grade accuracy and suggested threshold adjustments.
    """
    result = {"by_grade": {}, "calibrated": True, "suggestions": []}

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return result

    accuracies = {}
    for grade in ["A", "B", "C", "D"]:
        subset = wl[wl["rating"] == grade]
        total = len(subset)
        wins = len(subset[subset["result"] == "W"])
        acc = wins / total if total > 0 else None

        result["by_grade"][grade] = {
            "accuracy": round(acc, 4) if acc is not None else None,
            "total": total,
            "wins": wins,
        }
        if total >= 10 and acc is not None:
            accuracies[grade] = acc

    # Check monotonicity: A should beat B should beat C should beat D
    grade_order = ["A", "B", "C", "D"]
    for i in range(len(grade_order) - 1):
        g1, g2 = grade_order[i], grade_order[i + 1]
        if g1 in accuracies and g2 in accuracies:
            if accuracies[g2] > accuracies[g1] + 0.02:
                # Lower grade is outperforming higher grade — thresholds wrong
                result["calibrated"] = False
                result["suggestions"].append({
                    "type": "threshold_swap",
                    "higher_grade": g1,
                    "lower_grade": g2,
                    "higher_acc": round(accuracies[g1], 4),
                    "lower_acc": round(accuracies[g2], 4),
                    "reason": (
                        f"Grade {g2} ({accuracies[g2]:.1%}) outperforms "
                        f"{g1} ({accuracies[g1]:.1%}). Tightening {g1} threshold."
                    ),
                })

    # Check if A-grade is too loose (accuracy below 55%)
    if "A" in accuracies and accuracies["A"] < 0.55:
        result["calibrated"] = False
        result["suggestions"].append({
            "type": "tighten_a",
            "current_acc": round(accuracies["A"], 4),
            "reason": (
                f"A-grade accuracy is only {accuracies['A']:.1%}. "
                "Raising A threshold to be more selective."
            ),
        })

    return result


def analyze_variance_calibration(graded: pd.DataFrame) -> dict:
    """
    Check whether the model is over-confident or under-confident.

    If the model says 62% confidence but only hits 55%, variance is too low
    (the distribution is too narrow, making the model overconfident).
    If the model says 55% but hits 62%, variance is too high.

    Uses binned confidence calibration: group picks by confidence level
    and compare predicted vs actual hit rate.

    Args:
        graded: DataFrame with 'confidence' and 'result' columns.

    Returns:
        dict with calibration_error, overconfident (bool),
        per_prop_variance_suggestions.
    """
    result = {
        "calibration_error": None,
        "overconfident": None,
        "per_prop_suggestions": {},
    }

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if len(wl) < MIN_SAMPLE_DEFAULT:
        return result

    # Bin by confidence and check actual hit rate
    wl = wl.copy()
    wl["is_win"] = (wl["result"] == "W").astype(int)
    wl["conf_bin"] = pd.cut(
        wl["confidence"], bins=[0.50, 0.54, 0.57, 0.62, 0.70, 1.0],
        labels=["50-54", "54-57", "57-62", "62-70", "70+"],
    )

    # Overall calibration: mean(confidence) vs mean(actual win rate)
    mean_confidence = float(wl["confidence"].mean())
    actual_win_rate = float(wl["is_win"].mean())
    cal_error = mean_confidence - actual_win_rate

    result["calibration_error"] = round(cal_error, 4)
    result["overconfident"] = cal_error > 0.02  # >2% overconfident

    # Per-prop-type variance suggestions
    for stat_type in wl["stat_internal"].unique():
        subset = wl[wl["stat_internal"] == stat_type]
        if len(subset) < 15:
            continue

        type_mean_conf = float(subset["confidence"].mean())
        type_actual_wr = float(subset["is_win"].mean())
        type_cal_error = type_mean_conf - type_actual_wr

        if abs(type_cal_error) > 0.03:
            # Overconfident: increase variance. Underconfident: decrease.
            # Variance adjustment proportional to calibration error
            var_adjustment = type_cal_error * 2.0  # Scale factor
            var_adjustment = np.clip(var_adjustment, -MAX_ADJUSTMENT_PCT, MAX_ADJUSTMENT_PCT)

            result["per_prop_suggestions"][stat_type] = {
                "mean_confidence": round(type_mean_conf, 4),
                "actual_win_rate": round(type_actual_wr, 4),
                "calibration_error": round(type_cal_error, 4),
                "variance_adjustment_pct": round(float(var_adjustment), 4),
                "reason": (
                    f"{stat_type}: predicted {type_mean_conf:.1%} confidence, "
                    f"actual {type_actual_wr:.1%}. "
                    f"{'Increasing' if var_adjustment > 0 else 'Decreasing'} "
                    f"variance by {abs(var_adjustment):.1%}."
                ),
            }

    return result


def analyze_per_prop_direction(graded: pd.DataFrame) -> dict:
    """
    Per-prop-type direction analysis. Checks if specific prop types
    have a MORE/LESS accuracy imbalance that could be corrected with
    a per-prop projection offset (pushing projection up = more MORE picks,
    pushing down = more LESS picks).

    Unlike the global direction_bias (disabled for v003), this targets
    specific props where the projection model itself may be systematically
    biased in one direction.

    Returns dict keyed by stat_internal, each with:
      more_accuracy, less_accuracy, more_count, less_count,
      imbalance, suggested_offset (or None)
    """
    result = {}
    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return result

    for stat_type in wl["stat_internal"].unique():
        subset = wl[wl["stat_internal"] == stat_type]
        more_sub = subset[subset["pick"] == "MORE"]
        less_sub = subset[subset["pick"] == "LESS"]

        more_count = len(more_sub)
        less_count = len(less_sub)

        # Need 20+ graded picks in BOTH directions
        if more_count < 20 or less_count < 20:
            continue

        more_wins = len(more_sub[more_sub["result"] == "W"])
        less_wins = len(less_sub[less_sub["result"] == "W"])
        more_acc = more_wins / more_count
        less_acc = less_wins / less_count
        imbalance = more_acc - less_acc

        suggested_offset = None
        # One direction below 45% AND other above 55% → clear bias
        if (more_acc < 0.45 and less_acc > 0.55) or (less_acc < 0.45 and more_acc > 0.55):
            if more_acc < less_acc:
                # MORE is bad → projection too high → negative offset
                bad_acc = more_acc
                offset = 0.3 * (bad_acc - 0.50)  # negative
            else:
                # LESS is bad → projection too low → positive offset
                bad_acc = less_acc
                offset = -0.3 * (bad_acc - 0.50)  # positive

            offset = float(np.clip(offset, -0.5, 0.5))
            suggested_offset = {
                "offset": round(offset, 3),
                "reason": (
                    f"{stat_type}: MORE={more_acc:.1%} ({more_count}), "
                    f"LESS={less_acc:.1%} ({less_count}). "
                    f"Per-prop direction offset {offset:+.3f}."
                ),
            }

        result[stat_type] = {
            "more_accuracy": round(more_acc, 4),
            "less_accuracy": round(less_acc, 4),
            "more_count": more_count,
            "less_count": less_count,
            "imbalance": round(imbalance, 4),
            "suggested_offset": suggested_offset,
        }

    return result


# Prop types whose overdispersion/variance params are tunable
TUNABLE_OVERDISPERSION = {"home_runs", "stolen_bases", "hitter_fantasy_score"}


def analyze_overdispersion(graded: pd.DataFrame) -> dict:
    """
    Check if NegBin/Gamma distribution parameters produce well-calibrated
    probabilities. Groups picks by prop_type and confidence bucket, then
    compares predicted P(correct) vs actual accuracy.

    If predicted confidence consistently exceeds actual accuracy, the
    distribution is too tight (overdispersion parameter too low → increase it).
    If predicted confidence is below actual, distribution is too wide (decrease it).

    Only tunes: home_runs, stolen_bases (NegBin r param),
                hitter_fantasy_score (Gamma var_ratio)
    """
    result = {}

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return result

    wl = wl.copy()
    wl["is_win"] = (wl["result"] == "W").astype(int)

    for stat_type in TUNABLE_OVERDISPERSION:
        subset = wl[wl["stat_internal"] == stat_type]
        if len(subset) < 30:
            continue

        # Bin by confidence
        bins = [0.50, 0.55, 0.62, 0.70, 1.0]
        labels = ["50-55", "55-62", "62-70", "70+"]
        subset = subset.copy()
        subset["conf_bin"] = pd.cut(subset["confidence"], bins=bins, labels=labels)

        bucket_errors = []
        for label in labels:
            bucket = subset[subset["conf_bin"] == label]
            if len(bucket) < 10:
                continue
            mean_conf = float(bucket["confidence"].mean())
            actual_wr = float(bucket["is_win"].mean())
            bucket_errors.append(mean_conf - actual_wr)

        if not bucket_errors:
            continue

        cal_error = float(np.mean(bucket_errors))

        suggested_adjustment = None
        if abs(cal_error) > 0.05:
            # Overconfident (cal_error > 0): increase variance param
            # Underconfident (cal_error < 0): decrease variance param
            adj_pct = cal_error * 0.3
            adj_pct = float(np.clip(adj_pct, -0.15, 0.15))  # Cap at ±15%
            suggested_adjustment = {
                "adjustment_pct": round(adj_pct, 4),
                "reason": (
                    f"{stat_type}: mean calibration error {cal_error:+.3f} "
                    f"({'overconfident' if cal_error > 0 else 'underconfident'}). "
                    f"{'Increasing' if adj_pct > 0 else 'Decreasing'} "
                    f"variance param by {abs(adj_pct):.1%}."
                ),
            }

        result[stat_type] = {
            "calibration_error": round(cal_error, 4),
            "sample_size": len(subset),
            "overconfident": cal_error > 0.05,
            "suggested_adjustment": suggested_adjustment,
        }

    return result


def build_calibration_curve(graded: pd.DataFrame) -> dict:
    """
    Build a piecewise linear calibration curve that maps raw model
    confidence to actual observed accuracy. Saved in weights file
    as 'calibration_curve'.

    After enough data (100+ graded picks), this enables the app to show
    "Calibrated confidence: 62%" instead of raw model confidence which
    may be overconfident or underconfident.

    Returns dict with:
      points: list of [raw_confidence, actual_accuracy] pairs
      enough_data: bool
      total_picks: int
    """
    result = {"points": [], "enough_data": False, "total_picks": 0}

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    result["total_picks"] = len(wl)

    if len(wl) < 100:
        return result

    wl = wl.copy()
    wl["is_win"] = (wl["result"] == "W").astype(int)

    bins = [0.50, 0.55, 0.60, 0.65, 0.75, 1.0]
    labels = ["50-55", "55-60", "60-65", "65-75", "75+"]
    wl["conf_bin"] = pd.cut(wl["confidence"], bins=bins, labels=labels)

    points = []
    for label in labels:
        bucket = wl[wl["conf_bin"] == label]
        if len(bucket) < 10:
            continue
        mean_conf = round(float(bucket["confidence"].mean()), 4)
        actual_acc = round(float(bucket["is_win"].mean()), 4)
        points.append([mean_conf, actual_acc])

    result["points"] = points
    result["enough_data"] = len(points) >= 3

    return result


# ═══════════════════════════════════════════════════════
# CALIBRATION TABLE REBUILD (v017+)
# ═══════════════════════════════════════════════════════

CALIBRATION_PATH = WEIGHTS_DIR / "calibration_v015.json"
CALIBRATION_PROPS = {
    "hits": {"line": 1.5, "bin_width": 0.05, "min_proj": 0.5, "max_proj": 1.5},
    "total_bases": {"line": 1.5, "bin_width": 0.10, "min_proj": 0.8, "max_proj": 2.5},
    "pitcher_strikeouts": {"line": 4.5, "bin_width": 0.25, "min_proj": 3.0, "max_proj": 9.0},
    "hitter_fantasy_score": {"line": 7.5, "bin_width": 0.25, "min_proj": 5.5, "max_proj": 12.0},
}


def rebuild_calibration_tables(graded: pd.DataFrame, min_per_bin: int = 30) -> dict:
    """
    Rebuild empirical P(over)/P(under) calibration tables from graded results.

    This is the core self-learning mechanism: as the model generates predictions
    and they get graded (W/L), we accumulate evidence of the TRUE probability
    at each projection level. Over time, this replaces the static backtest
    calibration with live-data calibration that adapts to the current season.

    Args:
        graded: DataFrame with columns: stat_internal, projection, line,
                actual_result (numeric), result ('W'/'L'/'push').
        min_per_bin: Minimum observations per bin to include in calibration.

    Returns:
        dict: Calibration tables in the same format as calibration_v015.json.
        Also saves to disk if enough data exists.
    """
    result = {}

    if graded.empty:
        return result

    for prop_type, cfg in CALIBRATION_PROPS.items():
        line = cfg["line"]
        bin_width = cfg["bin_width"]

        # Filter to this prop type with matching line
        col_name = "stat_internal" if "stat_internal" in graded.columns else "prop_type"
        mask = (graded[col_name] == prop_type) & (graded["result"].isin(["W", "L"]))
        if "line" in graded.columns:
            mask = mask & (abs(graded["line"] - line) < 0.01)
        subset = graded[mask].copy()

        if len(subset) < min_per_bin:
            continue

        # Need actual_result (numeric) for P(over) calculation
        actual_col = None
        for c in ["actual_result", "actual"]:
            if c in subset.columns:
                actual_col = c
                break
        if actual_col is None:
            continue

        # Build bins
        points = []
        proj_lo = cfg["min_proj"]
        while proj_lo < cfg["max_proj"]:
            proj_hi = proj_lo + bin_width
            proj_mid = round((proj_lo + proj_hi) / 2, 4)

            bin_mask = (subset["projection"] >= proj_lo) & (subset["projection"] < proj_hi)
            bin_data = subset[bin_mask]

            if len(bin_data) >= min_per_bin:
                # Calculate empirical P(over) = fraction where actual > line
                if line == int(line):
                    n_over = int((bin_data[actual_col] > line).sum())
                    n_under = int((bin_data[actual_col] < line).sum())
                else:
                    n_over = int((bin_data[actual_col] > line).sum())
                    n_under = int((bin_data[actual_col] <= line).sum())

                n = n_over + n_under
                if n > 0:
                    p_over = round(n_over / n, 4)
                    p_under = round(n_under / n, 4)
                    points.append({
                        "proj_mid": proj_mid,
                        "proj_lo": round(proj_lo, 4),
                        "proj_hi": round(proj_hi, 4),
                        "p_over": p_over,
                        "p_under": p_under,
                        "n": n,
                        "n_over": n_over,
                        "n_under": n_under,
                    })

            proj_lo = proj_hi

        if points:
            result[prop_type] = {
                "line": line,
                "points": points,
                "total_predictions": len(subset),
            }

    # Save if we have meaningful data for at least one prop
    if result:
        try:
            # Merge with existing calibration: keep existing bins,
            # update with new data where we have enough observations
            existing = {}
            if CALIBRATION_PATH.exists():
                with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)

            for prop_type, new_data in result.items():
                if prop_type in existing:
                    # Merge: for each bin, use the one with more observations
                    existing_pts = {p["proj_mid"]: p for p in existing[prop_type].get("points", [])}
                    new_pts = {p["proj_mid"]: p for p in new_data["points"]}

                    merged = {}
                    for mid in set(list(existing_pts.keys()) + list(new_pts.keys())):
                        old = existing_pts.get(mid)
                        new = new_pts.get(mid)
                        if old and new:
                            # Weighted merge: combine observations
                            total_n = old["n"] + new["n"]
                            merged_over = old["n_over"] + new["n_over"]
                            merged_under = old["n_under"] + new["n_under"]
                            merged[mid] = {
                                "proj_mid": mid,
                                "proj_lo": old.get("proj_lo", new.get("proj_lo")),
                                "proj_hi": old.get("proj_hi", new.get("proj_hi")),
                                "p_over": round(merged_over / total_n, 4),
                                "p_under": round(merged_under / total_n, 4),
                                "n": total_n,
                                "n_over": merged_over,
                                "n_under": merged_under,
                            }
                        elif old:
                            merged[mid] = old
                        else:
                            merged[mid] = new

                    existing[prop_type] = {
                        "line": new_data["line"],
                        "points": sorted(merged.values(), key=lambda x: x["proj_mid"]),
                        "total_predictions": existing[prop_type].get("total_predictions", 0) + new_data["total_predictions"],
                    }
                else:
                    existing[prop_type] = new_data

            with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

            logger.info("Calibration tables updated: %s", list(result.keys()))
        except Exception as e:
            logger.error("Failed to save calibration tables: %s", e)

    return result


def reoptimize_floors(graded: pd.DataFrame, min_sample: int = 50) -> dict:
    """
    Re-optimize per-prop confidence floors from graded results.

    Analyzes accuracy at different confidence thresholds for each prop+direction
    and suggests tighter or looser floors to maximize hit rate.

    Returns dict of suggested floor changes.
    """
    suggestions = {}

    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if wl.empty:
        return suggestions

    col_name = "stat_internal" if "stat_internal" in wl.columns else "prop_type"

    for prop_type in CALIBRATION_PROPS:
        for direction in ["MORE", "LESS"]:
            mask = (wl[col_name] == prop_type) & (wl["pick"] == direction)
            subset = wl[mask]

            if len(subset) < min_sample:
                continue

            # Test floors from 0.55 to 0.80 in 0.02 steps
            best_acc = 0
            best_floor = 0.55
            best_n = 0

            for floor in [f / 100 for f in range(55, 81, 2)]:
                above = subset[subset["confidence"] >= floor]
                if len(above) < 20:
                    continue
                wins = len(above[above["result"] == "W"])
                acc = wins / len(above)
                # Prefer higher accuracy, but penalize extreme volume loss
                if acc > best_acc:
                    best_acc = acc
                    best_floor = floor
                    best_n = len(above)

            key = f"{prop_type}_{direction.lower()}"
            suggestions[key] = {
                "suggested_floor": best_floor,
                "accuracy_at_floor": round(best_acc, 4),
                "picks_at_floor": best_n,
            }

    return suggestions


# ═══════════════════════════════════════════════════════
# ADJUSTMENT PROPOSAL AND APPLICATION
# ═══════════════════════════════════════════════════════

def suggest_adjustments(analysis: dict, current_weights: dict) -> list[dict]:
    """
    Generate a list of proposed weight adjustments from the analysis results.

    Each proposal specifies what to change, by how much, and why.
    Proposals are filtered against previously failed adjustments to avoid
    repeating mistakes.

    Args:
        analysis: Combined output from all analyze_* functions.
        current_weights: The currently active weight configuration.

    Returns:
        list[dict]: Each entry has category, key, old_value, new_value,
        change_pct, reason.
    """
    proposals = []
    failed = _get_failed_adjustments()
    failed_keys = set()
    for f in failed:
        for change in f.get("changes", []):
            # Build a signature of the failed adjustment to avoid repeating it
            sig = f"{change.get('category', '')}:{change.get('key', '')}:{change.get('direction', '')}"
            failed_keys.add(sig)

    # 1. Direction bias correction
    # DISABLED for v003: NegBin CDFs make LESS-heavy direction balance
    # correct for discrete stats.
    if False:
        dir_analysis = analysis.get("direction_bias", {})
        if dir_analysis.get("bias_detected") and dir_analysis.get("suggested_correction"):
            corr = dir_analysis["suggested_correction"]
            target = corr["target"]
            direction = corr["direction"]
            amount = corr["amount"]
            sig = f"direction_bias:{target}:{direction}"

            if sig not in failed_keys:
                old_val = current_weights.get("direction_bias", {}).get(target, 1.0)
                if direction == "increase":
                    new_val = old_val * (1 + amount)
                else:
                    new_val = old_val * (1 - amount)
                new_val = round(np.clip(new_val, 0.85, 1.15), 4)

                proposals.append({
                    "category": "direction_bias",
                    "key": target,
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_pct": round((new_val - old_val) / old_val, 4) if old_val else 0,
                    "direction": direction,
                    "reason": corr["reason"],
                })

    # 2. Prop type offset corrections
    prop_analysis = analysis.get("prop_type_accuracy", {})
    for stat_type, info in prop_analysis.items():
        if info.get("suggested_offset") is None:
            continue
        offset_info = info["suggested_offset"]
        offset = offset_info["offset"]
        direction = "increase" if offset > 0 else "decrease"
        sig = f"prop_type_offsets:{stat_type}:{direction}"

        if sig not in failed_keys:
            old_val = current_weights.get("prop_type_offsets", {}).get(stat_type, 0.0)
            new_val = round(old_val + offset, 3)

            proposals.append({
                "category": "prop_type_offsets",
                "key": stat_type,
                "old_value": old_val,
                "new_value": new_val,
                "change_pct": round(abs(offset), 4),
                "direction": direction,
                "reason": offset_info["reason"],
            })

    # 3. Grade threshold adjustments
    grade_analysis = analysis.get("grade_calibration", {})
    if not grade_analysis.get("calibrated", True):
        for suggestion in grade_analysis.get("suggestions", []):
            if suggestion["type"] == "tighten_a":
                sig = "confidence_thresholds:A:increase"
                if sig not in failed_keys:
                    old_val = current_weights.get("confidence_thresholds", {}).get("A", 0.62)
                    new_val = round(min(old_val + 0.02, 0.78), 4)
                    proposals.append({
                        "category": "confidence_thresholds",
                        "key": "A",
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_pct": round((new_val - old_val) / old_val, 4) if old_val else 0,
                        "direction": "increase",
                        "reason": suggestion["reason"],
                    })

            elif suggestion["type"] == "threshold_swap":
                higher_g = suggestion["higher_grade"]
                sig = f"confidence_thresholds:{higher_g}:increase"
                if sig not in failed_keys:
                    old_val = current_weights.get("confidence_thresholds", {}).get(higher_g, 0.57)
                    new_val = round(min(old_val + 0.015, 0.70), 4)
                    proposals.append({
                        "category": "confidence_thresholds",
                        "key": higher_g,
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_pct": round((new_val - old_val) / old_val, 4) if old_val else 0,
                        "direction": "increase",
                        "reason": suggestion["reason"],
                    })

    # 4. Variance ratio adjustments (general)
    # DISABLED for v003: NegBin props already have variance ratios baked in.
    if False:
        var_analysis = analysis.get("variance_calibration", {})
        for stat_type, info in var_analysis.get("per_prop_suggestions", {}).items():
            var_adj = info["variance_adjustment_pct"]
            direction = "increase" if var_adj > 0 else "decrease"
            sig = f"variance_ratios:{stat_type}:{direction}"

            if sig not in failed_keys:
                old_val = current_weights.get("variance_ratios", {}).get(stat_type, 1.5)
                new_val = round(old_val * (1 + var_adj), 3)
                new_val = max(new_val, 0.5)  # Floor at 0.5

                proposals.append({
                    "category": "variance_ratios",
                    "key": stat_type,
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_pct": round(abs(var_adj), 4),
                    "direction": direction,
                    "reason": info["reason"],
                })

    # 5. Per-prop direction offset corrections
    # Stacks with section 2 offsets — if both fire for the same prop, they sum
    ppd_analysis = analysis.get("per_prop_direction", {})
    # Track which props already have offset proposals from section 2
    existing_offset_props = {
        p["key"] for p in proposals if p["category"] == "prop_type_offsets"
    }
    for stat_type, info in ppd_analysis.items():
        if info.get("suggested_offset") is None:
            continue
        offset_info = info["suggested_offset"]
        offset = offset_info["offset"]
        direction = "increase" if offset > 0 else "decrease"
        sig = f"prop_type_offsets:{stat_type}:ppd_{direction}"

        if sig not in failed_keys:
            old_val = current_weights.get("prop_type_offsets", {}).get(stat_type, 0.0)
            # If section 2 already proposed an offset, stack on top of it
            if stat_type in existing_offset_props:
                for p in proposals:
                    if p["category"] == "prop_type_offsets" and p["key"] == stat_type:
                        stacked = p["new_value"] + offset
                        mean_proj = abs(old_val) + 1.0  # rough scale
                        max_total = mean_proj * MAX_ADJUSTMENT_PCT
                        stacked = float(np.clip(stacked, -max_total, max_total))
                        p["new_value"] = round(stacked, 3)
                        p["reason"] += f" [+ppd: {offset_info['reason']}]"
                        break
            else:
                new_val = round(old_val + offset, 3)
                proposals.append({
                    "category": "prop_type_offsets",
                    "key": stat_type,
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_pct": round(abs(offset), 4),
                    "direction": direction,
                    "reason": f"[per-prop direction] {offset_info['reason']}",
                })

    # 6. Overdispersion tuning (NegBin r / Gamma var_ratio only)
    od_analysis = analysis.get("overdispersion", {})
    for stat_type, info in od_analysis.items():
        if info.get("suggested_adjustment") is None:
            continue
        adj_info = info["suggested_adjustment"]
        adj_pct = adj_info["adjustment_pct"]
        direction = "increase" if adj_pct > 0 else "decrease"
        sig = f"variance_ratios:{stat_type}:od_{direction}"

        if sig not in failed_keys:
            old_val = current_weights.get("variance_ratios", {}).get(stat_type, 1.5)
            new_val = round(old_val * (1 + adj_pct), 3)
            new_val = max(new_val, 0.5)

            proposals.append({
                "category": "variance_ratios",
                "key": stat_type,
                "old_value": old_val,
                "new_value": new_val,
                "change_pct": round(abs(adj_pct), 4),
                "direction": direction,
                "reason": f"[overdispersion] {adj_info['reason']}",
            })

    return proposals


def apply_adjustments(weights: dict, adjustments: list[dict]) -> dict:
    """
    Apply a list of proposed adjustments to a weight configuration.

    Each adjustment modifies one specific value within the weight dict.
    Changes are capped at ±10% per cycle to prevent wild swings.

    Args:
        weights: Current weight dict (will be deep-copied, not mutated).
        adjustments: List of adjustment proposals from suggest_adjustments().

    Returns:
        dict: New weight configuration with adjustments applied.
    """
    new_weights = copy.deepcopy(weights)

    for adj in adjustments:
        category = adj["category"]
        key = adj["key"]
        new_value = adj["new_value"]

        if category in new_weights and isinstance(new_weights[category], dict):
            new_weights[category][key] = new_value
        else:
            logger.warning("Unknown weight category %s, skipping", category)

    # Update metadata
    new_weights["metadata"]["adjustment_count"] = (
        new_weights.get("metadata", {}).get("adjustment_count", 0) + 1
    )

    return new_weights


# ═══════════════════════════════════════════════════════
# KILL SWITCH
# ═══════════════════════════════════════════════════════

def check_kill_switch(weights_version: str) -> dict:
    """
    Check whether the current weight version should be rolled back.

    Loads graded predictions made AFTER the weight version was applied,
    and checks if accuracy has dropped below the kill switch threshold (45%).

    Args:
        weights_version: The version string to evaluate (e.g., 'v003').

    Returns:
        dict with keys: should_rollback (bool), picks_evaluated (int),
        accuracy (float or None), reason (str).
    """
    result = {
        "should_rollback": False,
        "picks_evaluated": 0,
        "accuracy": None,
        "reason": "",
    }

    try:
        graded = get_graded_predictions(limit=500)
    except Exception as e:
        logger.error("Failed to load graded predictions for kill switch: %s", e)
        result["reason"] = f"Failed to load predictions: {e}"
        return result

    if graded.empty:
        result["reason"] = "No graded predictions available"
        return result

    # Filter to picks made with this weight version
    if "model_version" in graded.columns:
        version_picks = graded[graded["model_version"] == weights_version]
    else:
        # If model_version not tracked, use the most recent picks
        version_picks = graded.head(KILL_SWITCH_EVAL_SIZE)

    wl = version_picks[version_picks["result"].isin(["W", "L"])]
    total = len(wl)
    result["picks_evaluated"] = total

    if total < KILL_SWITCH_EVAL_SIZE:
        result["reason"] = (
            f"Only {total} graded picks with version {weights_version}, "
            f"need {KILL_SWITCH_EVAL_SIZE} to evaluate."
        )
        return result

    wins = len(wl[wl["result"] == "W"])
    accuracy = wins / total
    result["accuracy"] = round(accuracy, 4)

    if accuracy < KILL_SWITCH_THRESHOLD:
        result["should_rollback"] = True
        result["reason"] = (
            f"KILL SWITCH TRIGGERED: Version {weights_version} accuracy is "
            f"{accuracy:.1%} ({wins}W-{total - wins}L) over {total} picks, "
            f"below {KILL_SWITCH_THRESHOLD:.0%} threshold."
        )
    else:
        result["reason"] = (
            f"Version {weights_version} accuracy is {accuracy:.1%} "
            f"({wins}W-{total - wins}L) over {total} picks. Above threshold."
        )

    return result


def rollback_weights() -> dict:
    """
    Revert to the previous weight version and log the failure.

    Finds the most recent non-current versioned weight file and restores it
    as current.json. Records the rolled-back version in history so its
    adjustments are not repeated.

    Returns:
        dict with keys: rolled_back (bool), from_version (str),
        to_version (str), reason (str).
    """
    result = {
        "rolled_back": False,
        "from_version": None,
        "to_version": None,
        "reason": "",
    }

    current = load_current_weights()
    current_version = current.get("version", "unknown")
    result["from_version"] = current_version

    # Find all versioned weight files, sorted by version number
    version_files = []
    for p in WEIGHTS_DIR.glob("v*.json"):
        if p.name == "current.json" or p.name == "weight_history.json":
            continue
        name = p.stem
        parts = name.split("_")
        if parts and parts[0].startswith("v"):
            try:
                num = int(parts[0][1:])
                version_files.append((num, parts[0], p))
            except ValueError:
                pass

    version_files.sort(key=lambda x: x[0], reverse=True)

    # Find the most recent version that is NOT the current one
    previous_path = None
    previous_version = None
    for num, ver, path in version_files:
        if ver != current_version:
            previous_path = path
            previous_version = ver
            break

    if previous_path is None:
        # No previous version — rollback to baseline
        baseline = get_baseline_weights()
        save_weights(baseline, "v001", "Rollback to baseline (no previous version)")
        result["rolled_back"] = True
        result["to_version"] = "v001"
        result["reason"] = "No previous version found, rolled back to baseline"
    else:
        try:
            with open(previous_path, "r", encoding="utf-8") as f:
                prev_weights = json.load(f)
            # Restore as current
            with open(CURRENT_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(prev_weights, f, indent=2, default=str)
            result["rolled_back"] = True
            result["to_version"] = previous_version
            result["reason"] = (
                f"Rolled back from {current_version} to {previous_version}"
            )
        except (json.JSONDecodeError, OSError) as e:
            result["reason"] = f"Failed to read previous weights: {e}"
            return result

    # Log the rollback in history
    _append_history({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "rollback",
        "from_version": current_version,
        "to_version": result["to_version"],
        "status": "rolled_back",
        "reason": result["reason"],
        "changes": current.get("metadata", {}).get("last_changes", []),
    })

    logger.warning(
        "ROLLBACK: %s -> %s. Reason: %s",
        current_version, result["to_version"], result["reason"],
    )
    return result


# ═══════════════════════════════════════════════════════
# MAIN ADJUSTMENT CYCLE
# ═══════════════════════════════════════════════════════

def run_adjustment_cycle(min_sample: int = MIN_SAMPLE_DEFAULT) -> dict:
    """
    Run a full model weight adjustment cycle.

    This is the main entry point. It:
    1. Loads all graded predictions
    2. Checks if we have enough data (min_sample)
    3. Runs all analysis functions (direction bias, prop accuracy,
       grade calibration, variance calibration)
    4. Generates adjustment proposals
    5. Filters out previously failed adjustments
    6. Applies adjustments (if any)
    7. Saves the new weight version
    8. Checks the kill switch on the previous version
    9. Logs everything

    Args:
        min_sample: Minimum number of graded picks required. Default 25.

    Returns:
        dict with keys:
          - adjusted (bool): Whether any weights were changed
          - changes (list): What was changed and why
          - accuracy_before (float): Overall accuracy before adjustment
          - accuracy_by_grade (dict): Per-grade accuracy
          - version_old (str): Previous weight version
          - version_new (str): New weight version (or same if no change)
          - kill_switch (dict): Kill switch evaluation result
          - analysis (dict): Full analysis results
          - reason (str): Summary of what happened
    """
    result = {
        "adjusted": False,
        "changes": [],
        "accuracy_before": None,
        "accuracy_by_grade": {},
        "version_old": None,
        "version_new": None,
        "kill_switch": {},
        "analysis": {},
        "reason": "",
    }

    # Load current weights
    try:
        current_weights = load_current_weights()
    except Exception as e:
        result["reason"] = f"Failed to load weights: {e}"
        logger.error(result["reason"])
        return result

    current_version = current_weights.get("version", "v001")
    result["version_old"] = current_version

    # Load graded predictions
    try:
        graded = get_graded_predictions(limit=2000)
    except Exception as e:
        result["reason"] = f"Failed to load graded predictions: {e}"
        logger.error(result["reason"])
        return result

    wl = graded[graded["result"].isin(["W", "L"])] if not graded.empty else graded
    total = len(wl)

    if total < min_sample:
        result["reason"] = (
            f"Insufficient data: {total} graded picks, need {min_sample}. "
            "No adjustments made."
        )
        result["version_new"] = current_version
        logger.info(result["reason"])
        return result

    # Calculate overall accuracy before adjustment
    wins = len(wl[wl["result"] == "W"])
    accuracy_before = wins / total if total > 0 else 0.0
    result["accuracy_before"] = round(accuracy_before, 4)

    # v018: Compute Brier Score and Log Loss for proper calibration assessment
    try:
        confidences = wl["confidence"].astype(float).values
        outcomes = (wl["result"] == "W").astype(int).values
        # Brier Score: E[(p - y)^2], lower is better (0 = perfect, 0.25 = coin flip)
        brier_score = float(np.mean((confidences - outcomes) ** 2))
        # Log Loss: -E[y*log(p) + (1-y)*log(1-p)], lower is better
        eps = 1e-7
        clipped = np.clip(confidences, eps, 1 - eps)
        log_loss = float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))
        result["brier_score"] = round(brier_score, 4)
        result["log_loss"] = round(log_loss, 4)
        logger.info("Scoring: Brier=%.4f, LogLoss=%.4f, Accuracy=%.1f%%",
                     brier_score, log_loss, accuracy_before * 100)
    except Exception as e:
        logger.warning("Scoring metrics failed (non-fatal): %s", e)
        result["brier_score"] = None
        result["log_loss"] = None

    # ── Run all analyses ──
    analysis = {
        "direction_bias": analyze_direction_bias(graded),
        "prop_type_accuracy": analyze_prop_type_accuracy(graded),
        "grade_calibration": analyze_grade_calibration(graded),
        "variance_calibration": analyze_variance_calibration(graded),
        "per_prop_direction": analyze_per_prop_direction(graded),
        "overdispersion": analyze_overdispersion(graded),
    }
    result["analysis"] = analysis
    result["accuracy_by_grade"] = analysis["grade_calibration"].get("by_grade", {})

    # ── v017+: Rebuild empirical calibration tables from live results ──
    # This is the self-learning mechanism: as more graded predictions accumulate,
    # the calibration tables are updated with merged live + backtest data.
    if total >= 100:
        try:
            cal_result = rebuild_calibration_tables(graded)
            if cal_result:
                logger.info(
                    "Calibration tables updated from %d live results: %s",
                    total, list(cal_result.keys()),
                )
                analysis["calibration_rebuild"] = {
                    "props_updated": list(cal_result.keys()),
                    "total_picks": total,
                }
                # Clear predictor's calibration cache so it loads fresh tables
                try:
                    from src.predictor import _clear_weights_cache
                    _clear_weights_cache()
                except ImportError:
                    pass
        except Exception as e:
            logger.warning("Calibration rebuild failed (non-fatal): %s", e)

    # ── v017+: Suggest floor re-optimizations ──
    if total >= 200:
        try:
            floor_suggestions = reoptimize_floors(graded)
            if floor_suggestions:
                analysis["floor_suggestions"] = floor_suggestions
                logger.info("Floor re-optimization suggestions: %s", floor_suggestions)
        except Exception as e:
            logger.warning("Floor optimization failed (non-fatal): %s", e)

    # ── Check kill switch on current version first ──
    ks_result = check_kill_switch(current_version)
    result["kill_switch"] = ks_result

    if ks_result.get("should_rollback"):
        logger.warning("Kill switch triggered before adjustment: %s", ks_result["reason"])
        rb_result = rollback_weights()
        result["reason"] = (
            f"Kill switch triggered on {current_version}. "
            f"Rolled back to {rb_result.get('to_version', 'unknown')}. "
            f"No new adjustments applied."
        )
        result["version_new"] = rb_result.get("to_version", current_version)
        return result

    # ── Generate adjustment proposals ──
    proposals = suggest_adjustments(analysis, current_weights)

    if not proposals:
        result["reason"] = (
            f"No adjustments needed. Accuracy: {accuracy_before:.1%} "
            f"over {total} picks. Model is well-calibrated."
        )
        result["version_new"] = current_version
        logger.info(result["reason"])
        return result

    # ── Apply adjustments ──
    new_weights = apply_adjustments(current_weights, proposals)
    new_version = _next_version()

    # Store the changes in metadata so rollback can log them
    new_weights["metadata"]["last_changes"] = proposals
    new_weights["metadata"]["total_picks_analyzed"] = total
    new_weights["metadata"]["accuracy_at_creation"] = round(accuracy_before, 4)
    new_weights["metadata"]["parent_version"] = current_version

    # Build and store calibration curve
    new_weights["calibration_curve"] = build_calibration_curve(graded)

    # Build description
    change_summary = ", ".join(
        f"{p['category']}.{p['key']}: {p['old_value']}->{p['new_value']}"
        for p in proposals[:5]
    )
    if len(proposals) > 5:
        change_summary += f" (+{len(proposals) - 5} more)"

    description = (
        f"Auto-adjustment from {total} picks ({accuracy_before:.1%} accuracy). "
        f"Changes: {change_summary}"
    )

    # Save new weights
    try:
        save_path = save_weights(new_weights, new_version, description)
    except Exception as e:
        result["reason"] = f"Failed to save new weights: {e}"
        logger.error(result["reason"])
        result["version_new"] = current_version
        return result

    result["adjusted"] = True
    result["changes"] = proposals
    result["version_new"] = new_version
    result["reason"] = (
        f"Applied {len(proposals)} adjustment(s). "
        f"Version {current_version} -> {new_version}. "
        f"Accuracy was {accuracy_before:.1%} over {total} picks. "
        f"Monitoring for kill switch ({KILL_SWITCH_EVAL_SIZE} picks)."
    )

    # Log to history (with proper scoring metrics)
    _append_history({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "adjustment",
        "from_version": current_version,
        "to_version": new_version,
        "status": "applied",
        "sample_size": total,
        "accuracy_before": round(accuracy_before, 4),
        "brier_score": result.get("brier_score"),
        "log_loss": result.get("log_loss"),
        "changes": proposals,
        "description": description,
    })

    # v018: Trigger hedge-style ensemble weight update after grading
    # This reweights sharp_odds vs projection vs recent_form based on
    # which signal sources have been most accurate recently.
    try:
        from src.ensemble import update_ensemble_weights
        ens_result = update_ensemble_weights(learning_rate=0.1, min_samples=20)
        if ens_result.get("updated"):
            logger.info("Ensemble weights updated: %s", ens_result.get("new_weights"))
            result["ensemble_updated"] = True
            result["ensemble_weights"] = ens_result.get("new_weights")
        else:
            result["ensemble_updated"] = False
    except Exception as e:
        logger.warning("Ensemble weight update failed: %s", e)
        result["ensemble_updated"] = False

    logger.info(result["reason"])
    return result


# ═══════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════

def main() -> None:
    """Run an adjustment cycle from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("MLB Prop Edge — Self-Learning Weight Adjustment")
    print("=" * 60)

    # Show current state
    weights = load_current_weights()
    print(f"\nCurrent version: {weights.get('version', 'unknown')}")
    print(f"Description: {weights.get('description', 'N/A')}")

    history = get_weight_history()
    print(f"Adjustment history: {len(history)} entries")

    # Run cycle
    print("\nRunning adjustment cycle...")
    result = run_adjustment_cycle()

    print(f"\nResult: {result['reason']}")
    print(f"Adjusted: {result['adjusted']}")

    if result.get("accuracy_before") is not None:
        print(f"Accuracy before: {result['accuracy_before']:.1%}")

    if result["changes"]:
        print(f"\nChanges applied ({len(result['changes'])}):")
        for change in result["changes"]:
            print(f"  {change['category']}.{change['key']}: "
                  f"{change['old_value']} -> {change['new_value']}")
            print(f"    Reason: {change['reason']}")

    if result.get("version_new"):
        print(f"\nNew version: {result['version_new']}")

    ks = result.get("kill_switch", {})
    if ks.get("should_rollback"):
        print(f"\n*** KILL SWITCH: {ks['reason']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
