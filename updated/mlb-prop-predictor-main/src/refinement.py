"""
Model Refinement Engine
The self-improvement loop: predict → track → analyze → adjust → repeat.

This module analyzes your graded prediction history and:
1. Identifies which prop types, grades, and situations you're strongest/weakest on
2. Detects systematic biases (e.g., consistently overvaluing overs on TB props)
3. Measures calibration (are 60% confidence picks actually hitting 60%?)
4. Computes feature importance (which inputs are helping vs hurting?)
5. Suggests and applies weight adjustments to improve future accuracy

Run this after you have 100+ graded picks. The more data, the better the analysis.
250+ picks = reliable. 500+ = very reliable. Under 50 = too noisy to trust.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path

from src.database import get_graded_predictions, get_all_predictions, get_connection


# ═══════════════════════════════════════════════════════
# CORE ANALYSIS
# ═══════════════════════════════════════════════════════

def run_full_analysis(min_sample: int = 30) -> dict:
    """
    Run complete analysis on all graded predictions.
    Returns a comprehensive report with actionable insights.
    """
    graded = get_graded_predictions(limit=5000)
    if graded.empty:
        return {"status": "no_data", "message": "No graded predictions yet."}

    # Filter to W/L only (exclude pushes for accuracy calc)
    wl = graded[graded["result"].isin(["W", "L"])].copy()
    if len(wl) < min_sample:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(wl)} graded picks. Need {min_sample}+ for reliable analysis.",
            "total_graded": len(wl),
        }

    wl["win"] = (wl["result"] == "W").astype(int)
    wl["game_date"] = pd.to_datetime(wl["game_date"], errors="coerce")

    report = {
        "status": "ok",
        "total_graded": len(wl),
        "overall_accuracy": round(wl["win"].mean(), 4),
        "total_wins": int(wl["win"].sum()),
        "total_losses": int(len(wl) - wl["win"].sum()),
        "date_range": f"{wl['game_date'].min().strftime('%Y-%m-%d')} to {wl['game_date'].max().strftime('%Y-%m-%d')}",
    }

    # Run each analysis
    report["by_prop_type"] = _analyze_by_column(wl, "stat_type", min_n=10)
    report["by_rating"] = _analyze_by_column(wl, "rating", min_n=10)
    report["by_direction"] = _analyze_by_column(wl, "pick", min_n=15)
    report["by_model_version"] = _analyze_by_column(wl, "model_version", min_n=10)
    report["calibration"] = _analyze_calibration(wl)
    report["bias_detection"] = _detect_biases(wl)
    report["trend_analysis"] = _analyze_trends(wl)
    report["edge_accuracy"] = _analyze_edge_vs_accuracy(wl)
    report["recommendations"] = _generate_recommendations(report)

    return report


def _analyze_by_column(df, col, min_n=10) -> list:
    """Break down accuracy by a categorical column."""
    results = []
    for val in df[col].dropna().unique():
        subset = df[df[col] == val]
        n = len(subset)
        if n < min_n:
            continue
        wins = subset["win"].sum()
        acc = wins / n
        # Wilson confidence interval (better than simple proportion for small n)
        z = 1.96  # 95% CI
        center = (wins + z**2/2) / (n + z**2)
        spread = z * np.sqrt((wins * (n - wins) / n + z**2/4) / (n + z**2))
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)

        results.append({
            "value": val,
            "wins": int(wins),
            "losses": int(n - wins),
            "total": n,
            "accuracy": round(acc, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "profitable": acc >= 0.542,  # 5-pick flex breakeven
        })

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results


def _analyze_calibration(df) -> dict:
    """
    Calibration analysis: are confidence scores actually predictive?
    
    A well-calibrated model has 60% confidence picks winning ~60% of the time.
    If 60% confidence picks only win 50%, the model is overconfident.
    If they win 70%, the model is underconfident (leaving edge on the table).
    """
    if "confidence" not in df.columns or df["confidence"].isna().all():
        return {"status": "no_confidence_data"}

    bins = [
        (0.50, 0.54, "50-54%"),
        (0.54, 0.57, "54-57%"),
        (0.57, 0.60, "57-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 1.00, "65%+"),
    ]

    calibration_data = []
    for low, high, label in bins:
        subset = df[(df["confidence"] >= low) & (df["confidence"] < high)]
        if len(subset) < 5:
            continue
        actual_acc = subset["win"].mean()
        expected_acc = subset["confidence"].mean()
        gap = actual_acc - expected_acc

        calibration_data.append({
            "bin": label,
            "expected_accuracy": round(expected_acc, 4),
            "actual_accuracy": round(actual_acc, 4),
            "gap": round(gap, 4),
            "n": len(subset),
            "status": "overconfident" if gap < -0.03 else ("underconfident" if gap > 0.03 else "well_calibrated"),
        })

    # Brier score (lower = better, 0 = perfect, 0.25 = coin flip)
    brier = ((df["confidence"] - df["win"]) ** 2).mean()
    # Compare to baseline (always predict 50%)
    baseline_brier = ((0.5 - df["win"]) ** 2).mean()

    return {
        "bins": calibration_data,
        "brier_score": round(brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "brier_skill": round(1 - brier / baseline_brier, 4) if baseline_brier > 0 else 0,
        "interpretation": (
            "Model adds value over coin flip" if brier < baseline_brier
            else "Model is not adding value — needs recalibration"
        ),
    }


def _detect_biases(df) -> list:
    """
    Detect systematic biases in predictions.
    
    Common biases:
    - Over/under bias: consistently better on one direction
    - Prop type blind spots: great on Ks, terrible on TB
    - Rating inflation: A-rated picks performing like C-rated
    - Line-level bias: better on high lines vs low lines
    """
    biases = []

    # More vs Less bias
    more = df[df["pick"] == "MORE"]
    less = df[df["pick"] == "LESS"]
    if len(more) >= 15 and len(less) >= 15:
        more_acc = more["win"].mean()
        less_acc = less["win"].mean()
        gap = abs(more_acc - less_acc)
        if gap > 0.05:
            better = "MORE" if more_acc > less_acc else "LESS"
            worse = "LESS" if better == "MORE" else "MORE"
            biases.append({
                "type": "direction_bias",
                "severity": "high" if gap > 0.10 else "medium",
                "description": f"Significantly better on {better} picks ({more_acc:.1%}) vs {worse} ({less_acc:.1%})",
                "recommendation": f"Consider increasing confidence threshold for {worse} picks, or weight {worse} projections more conservatively",
                "gap": round(gap, 4),
            })

    # Rating calibration check
    for rating in ["A", "B", "C", "D"]:
        subset = df[df["rating"] == rating]
        if len(subset) < 15:
            continue
        acc = subset["win"].mean()
        expected = {"A": 0.62, "B": 0.57, "C": 0.54, "D": 0.50}
        exp = expected.get(rating, 0.50)
        gap = acc - exp
        if gap < -0.05:
            biases.append({
                "type": "rating_inflation",
                "severity": "high" if gap < -0.08 else "medium",
                "description": f"{rating}-rated picks hitting {acc:.1%} but expected {exp:.1%} (gap: {gap:+.1%})",
                "recommendation": f"Tighten {rating}-grade threshold — currently overrating these picks",
                "gap": round(gap, 4),
            })
        elif gap > 0.05:
            biases.append({
                "type": "rating_conservative",
                "severity": "info",
                "description": f"{rating}-rated picks hitting {acc:.1%} vs expected {exp:.1%} — model is underconfident here",
                "recommendation": f"Could be more aggressive with {rating}-grade picks",
                "gap": round(gap, 4),
            })

    # Prop type analysis
    for prop in df["stat_type"].unique():
        subset = df[df["stat_type"] == prop]
        if len(subset) < 15:
            continue
        acc = subset["win"].mean()
        if acc < 0.48:
            biases.append({
                "type": "prop_weakness",
                "severity": "high",
                "description": f"{prop}: only {acc:.1%} accuracy over {len(subset)} picks — losing money",
                "recommendation": f"Consider avoiding {prop} props or reworking that projection model",
            })
        elif acc > 0.58:
            biases.append({
                "type": "prop_strength",
                "severity": "info",
                "description": f"{prop}: {acc:.1%} accuracy over {len(subset)} picks — strong edge",
                "recommendation": f"Prioritize {prop} props — this is where your model excels",
            })

    return biases


def _analyze_trends(df) -> dict:
    """
    Detect if accuracy is improving, declining, or stable over time.
    """
    if len(df) < 30:
        return {"status": "insufficient_data"}

    df_sorted = df.sort_values("game_date").copy()

    # Rolling 30-pick accuracy
    df_sorted["rolling_acc"] = df_sorted["win"].rolling(30, min_periods=15).mean()

    # Split into first half / second half
    mid = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid]
    second_half = df_sorted.iloc[mid:]

    first_acc = first_half["win"].mean()
    second_acc = second_half["win"].mean()
    trend_delta = second_acc - first_acc

    if trend_delta > 0.03:
        trend = "improving"
        emoji = "📈"
    elif trend_delta < -0.03:
        trend = "declining"
        emoji = "📉"
    else:
        trend = "stable"
        emoji = "➖"

    # Last 50 picks vs overall
    last_50 = df_sorted.tail(50)
    recent_acc = last_50["win"].mean() if len(last_50) >= 20 else None

    return {
        "trend": trend,
        "emoji": emoji,
        "first_half_acc": round(first_acc, 4),
        "second_half_acc": round(second_acc, 4),
        "trend_delta": round(trend_delta, 4),
        "recent_50_acc": round(recent_acc, 4) if recent_acc else None,
        "interpretation": f"{emoji} Model is {trend}: {first_acc:.1%} → {second_acc:.1%}",
    }


def _analyze_edge_vs_accuracy(df) -> dict:
    """
    Analyze whether higher-edge picks actually perform better.
    This validates that the edge calculation is meaningful.
    """
    if "edge" not in df.columns or df["edge"].isna().all():
        return {"status": "no_edge_data"}

    # Bucket by edge size
    bins = [
        (0.00, 0.03, "0-3%"),
        (0.03, 0.06, "3-6%"),
        (0.06, 0.10, "6-10%"),
        (0.10, 1.00, "10%+"),
    ]

    results = []
    for low, high, label in bins:
        subset = df[(df["edge"] >= low) & (df["edge"] < high)]
        if len(subset) < 10:
            continue
        acc = subset["win"].mean()
        results.append({
            "bin": label,
            "accuracy": round(acc, 4),
            "n": len(subset),
        })

    # Check if edge is actually predictive (higher edge = higher accuracy?)
    if len(results) >= 2:
        accs = [r["accuracy"] for r in results]
        is_monotonic = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
        correlation = "Edge is predictive ✅" if is_monotonic else "Edge may need recalibration ⚠️"
    else:
        correlation = "Need more data"

    return {
        "bins": results,
        "edge_is_predictive": correlation,
    }


# ═══════════════════════════════════════════════════════
# RECOMMENDATIONS ENGINE
# ═══════════════════════════════════════════════════════

def _generate_recommendations(report: dict) -> list:
    """
    Synthesize all analysis into prioritized, actionable recommendations.
    """
    recs = []
    overall = report.get("overall_accuracy", 0)

    # Overall performance check
    if overall >= 0.57:
        recs.append({
            "priority": "info",
            "title": "Strong overall performance",
            "detail": f"At {overall:.1%} accuracy, you're well above the 54.2% breakeven. Stay disciplined.",
        })
    elif overall >= 0.542:
        recs.append({
            "priority": "medium",
            "title": "Profitable but thin margin",
            "detail": f"At {overall:.1%}, you're barely above breakeven. Focus on high-grade picks only.",
        })
    elif overall >= 0.50:
        recs.append({
            "priority": "high",
            "title": "Below breakeven — need adjustments",
            "detail": f"At {overall:.1%}, you're losing money on 5-pick flex. See bias analysis below.",
        })
    else:
        recs.append({
            "priority": "critical",
            "title": "Significantly below 50% — model needs rework",
            "detail": f"At {overall:.1%}, worse than coin flip. Focus on what's working and cut what isn't.",
        })

    # Prop type recommendations
    by_prop = report.get("by_prop_type", [])
    strong_props = [p for p in by_prop if p["accuracy"] >= 0.56 and p["total"] >= 15]
    weak_props = [p for p in by_prop if p["accuracy"] < 0.50 and p["total"] >= 15]

    if strong_props:
        names = ", ".join(p["value"] for p in strong_props[:3])
        recs.append({
            "priority": "info",
            "title": f"Your strongest props: {names}",
            "detail": "Double down on these — higher volume on your best prop types will increase ROI.",
        })

    if weak_props:
        names = ", ".join(p["value"] for p in weak_props[:3])
        recs.append({
            "priority": "high",
            "title": f"Cut or rework: {names}",
            "detail": "These props are losing money. Either avoid them or investigate why the model is wrong.",
        })

    # Calibration recommendations
    cal = report.get("calibration", {})
    if cal.get("brier_skill", 0) < 0:
        recs.append({
            "priority": "high",
            "title": "Model calibration is worse than baseline",
            "detail": "Your confidence scores aren't predictive. Consider using only sharp book edge (not projections) for pick selection.",
        })

    overconfident = [b for b in cal.get("bins", []) if b.get("status") == "overconfident"]
    if overconfident:
        bins = ", ".join(b["bin"] for b in overconfident)
        recs.append({
            "priority": "medium",
            "title": f"Overconfident in {bins} range",
            "detail": "Picks in these confidence buckets perform worse than the model expects. Raise the threshold for including these picks.",
        })

    # Direction bias
    biases = report.get("bias_detection", [])
    dir_biases = [b for b in biases if b["type"] == "direction_bias"]
    if dir_biases:
        recs.append({
            "priority": "medium",
            "title": dir_biases[0]["description"],
            "detail": dir_biases[0]["recommendation"],
        })

    # Trend
    trend = report.get("trend_analysis", {})
    if trend.get("trend") == "declining":
        recs.append({
            "priority": "high",
            "title": "Accuracy is declining over time",
            "detail": f"First half: {trend.get('first_half_acc', 0):.1%} → Second half: {trend.get('second_half_acc', 0):.1%}. Markets may be adjusting. Review if your edge sources are still valid.",
        })
    elif trend.get("trend") == "improving":
        recs.append({
            "priority": "info",
            "title": "Accuracy is improving — model is learning",
            "detail": f"First half: {trend.get('first_half_acc', 0):.1%} → Second half: {trend.get('second_half_acc', 0):.1%}. Keep doing what you're doing.",
        })

    # Volume check
    total = report.get("total_graded", 0)
    if total < 100:
        recs.append({
            "priority": "medium",
            "title": f"Only {total} graded picks — small sample warning",
            "detail": "Results may not be statistically significant yet. Need 250+ picks before trusting trends. Keep logging and grading.",
        })

    recs.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "info": 3}.get(x["priority"], 4))
    return recs


# ═══════════════════════════════════════════════════════
# WEIGHT ADJUSTMENT SUGGESTIONS
# ═══════════════════════════════════════════════════════

def suggest_weight_adjustments(report: dict) -> list:
    """
    Based on the analysis, suggest specific weight changes to the prediction engine.
    
    These are conservative — each adjustment is small (±5-15%) to avoid
    overcorrecting on limited data.
    """
    adjustments = []

    # If MORE/LESS is biased, adjust the projection direction
    biases = report.get("bias_detection", [])
    for bias in biases:
        if bias["type"] == "direction_bias" and bias.get("gap", 0) > 0.06:
            adjustments.append({
                "target": "projection_direction",
                "description": bias["description"],
                "action": bias["recommendation"],
                "magnitude": "±5-10% on the weaker direction's projections",
            })

    # If specific prop types are weak, suggest model changes
    weak = [p for p in report.get("by_prop_type", []) if p["accuracy"] < 0.48 and p["total"] >= 20]
    for prop in weak:
        adjustments.append({
            "target": f"prop_model:{prop['value']}",
            "description": f"{prop['value']} at {prop['accuracy']:.1%} over {prop['total']} picks",
            "action": f"Increase regression weight for {prop['value']} (pull projections closer to the line)",
            "magnitude": "+10-15% regression toward line for this prop type",
        })

    # If calibration shows overconfidence, suggest threshold changes
    cal = report.get("calibration", {})
    for bucket in cal.get("bins", []):
        if bucket.get("status") == "overconfident" and bucket.get("n", 0) >= 15:
            adjustments.append({
                "target": "confidence_threshold",
                "description": f"{bucket['bin']} confidence bucket: expected {bucket['expected_accuracy']:.1%}, actual {bucket['actual_accuracy']:.1%}",
                "action": f"Raise minimum confidence threshold for this range by ~{abs(bucket['gap'])*100:.0f}%",
                "magnitude": f"+{abs(bucket['gap'])*100:.0f}% threshold increase",
            })

    # If edge isn't predictive, suggest shifting to pure market-based
    edge_analysis = report.get("edge_accuracy", {})
    if "Edge may need recalibration" in str(edge_analysis.get("edge_is_predictive", "")):
        adjustments.append({
            "target": "edge_calculation",
            "description": "Higher edge picks aren't performing better than lower edge picks",
            "action": "Increase weight on sharp book consensus (currently 40%), decrease projection model weight",
            "magnitude": "Sharp books: 40% → 55%, Projections: reduce proportionally",
        })

    return adjustments


# ═══════════════════════════════════════════════════════
# SAVE/LOAD ANALYSIS HISTORY
# ═══════════════════════════════════════════════════════

ANALYSIS_DIR = Path("data/analysis")

def save_analysis(report: dict):
    """Save analysis report for historical comparison."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = ANALYSIS_DIR / f"analysis_{timestamp}.json"

    # Convert non-serializable types
    def _clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return obj

    clean_report = json.loads(json.dumps(report, default=_clean))
    with open(filepath, "w") as f:
        json.dump(clean_report, f, indent=2)

    return str(filepath)


def load_latest_analysis() -> dict:
    """Load the most recent analysis report."""
    if not ANALYSIS_DIR.exists():
        return {}
    files = sorted(ANALYSIS_DIR.glob("analysis_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


def compare_analyses() -> dict:
    """Compare current analysis to the previous one to see if adjustments helped."""
    if not ANALYSIS_DIR.exists():
        return {"status": "no_history"}

    files = sorted(ANALYSIS_DIR.glob("analysis_*.json"), reverse=True)
    if len(files) < 2:
        return {"status": "need_two_analyses"}

    with open(files[0]) as f:
        current = json.load(f)
    with open(files[1]) as f:
        previous = json.load(f)

    curr_acc = current.get("overall_accuracy", 0)
    prev_acc = previous.get("overall_accuracy", 0)
    delta = curr_acc - prev_acc

    return {
        "status": "ok",
        "current_accuracy": curr_acc,
        "previous_accuracy": prev_acc,
        "delta": round(delta, 4),
        "improving": delta > 0,
        "current_n": current.get("total_graded", 0),
        "previous_n": previous.get("total_graded", 0),
        "current_date": files[0].stem.replace("analysis_", ""),
        "previous_date": files[1].stem.replace("analysis_", ""),
    }
