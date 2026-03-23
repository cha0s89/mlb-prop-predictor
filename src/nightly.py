"""
Nightly self-improvement cycle.

Triggered by the "Run Nightly Update" button in the Dashboard tab,
or can be called externally (e.g. cron, scheduled task).
Run AFTER games finish (~11 PM ET) so box scores are available.

Phases:
  1. Auto-grade — pull box scores, grade all pending predictions
  2. Compute metrics — Brier score, log loss, hit rate (via autolearn)
  3. Update ensemble weights — Hedge-style reweighting of signal sources
  4. Drift detection — ADWIN / CUSUM regime change checks
  5. Calibration check — binned predicted-vs-actual, isotonic fit
  6. Log results — persist to nightly_logs table for Dashboard display
"""

import datetime
import json
import logging
import numpy as np
from pathlib import Path

from src.autograder import auto_grade_date
from src.autolearn import run_adjustment_cycle
from src.ensemble import update_ensemble_weights
from src.drift import check_model_health
from src.clv import compute_clv_stats, update_closing_lines
from src.board_logger import get_board_stats, ensure_shadow_sample, get_shadow_sample_stats
from src.database import get_connection, get_graded_predictions, get_projection_diagnostics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def run_nightly_cycle(target_date: str = None) -> dict:
    """
    Complete nightly self-improvement cycle.

    Args:
        target_date: YYYY-MM-DD string. Defaults to today.

    Returns:
        Dict with all phase results for Dashboard display.
    """
    if target_date is None:
        target_date = datetime.date.today().isoformat()

    logger.info("Starting nightly cycle for %s", target_date)

    results = {
        "date": target_date,
        "phase_results": {},
        "errors": [],
        "weights_updated": False,
        "drift_alerts": [],
        "calibration_warnings": [],
    }

    # Shadow QA sample should exist for every board date so we can track an
    # unbiased daily slice even if the app was only run once.
    try:
        results["phase_results"]["shadow_sample"] = ensure_shadow_sample(target_date, sample_size=50)
    except Exception as e:
        logger.error("Shadow sample selection failed: %s", e, exc_info=True)
        results["errors"].append(f"Shadow sample selection failed: {e}")

    # ── PHASE 1: AUTO-GRADE ──────────────────────────────────
    try:
        grade_results = _phase1_autograde(target_date)
        results["phase_results"]["grading"] = grade_results
        logger.info("Phase 1 done: %s graded", grade_results.get("total_graded", 0))
    except Exception as e:
        logger.error("Phase 1 (grading) failed: %s", e, exc_info=True)
        results["errors"].append(f"Grading failed: {e}")

    # ── PHASE 2: COMPUTE METRICS (via autolearn) ─────────────
    try:
        metrics = _phase2_compute_metrics()
        results["phase_results"]["metrics"] = metrics
        logger.info("Phase 2 done: Brier=%s, LogLoss=%s",
                     metrics.get("brier_score", "N/A"), metrics.get("log_loss", "N/A"))
    except Exception as e:
        logger.error("Phase 2 (metrics) failed: %s", e, exc_info=True)
        results["errors"].append(f"Metrics failed: {e}")

    # ── PHASE 3: UPDATE ENSEMBLE WEIGHTS ─────────────────────
    try:
        weight_results = _phase3_update_weights()
        results["phase_results"]["weights"] = weight_results
        results["weights_updated"] = weight_results.get("updated", False)
        logger.info("Phase 3 done: updated=%s", weight_results.get("updated"))
    except Exception as e:
        logger.error("Phase 3 (weights) failed: %s", e, exc_info=True)
        results["errors"].append(f"Weight update failed: {e}")

    # ── PHASE 4: DRIFT DETECTION ─────────────────────────────
    try:
        drift_results = _phase4_check_drift()
        results["drift_alerts"] = drift_results.get("alerts", [])
        results["phase_results"]["drift"] = drift_results
        logger.info("Phase 4 done: %d alerts", len(results["drift_alerts"]))
    except Exception as e:
        logger.error("Phase 4 (drift) failed: %s", e, exc_info=True)
        results["errors"].append(f"Drift detection failed: {e}")

    # ── PHASE 5: CALIBRATION CHECK ───────────────────────────
    try:
        cal_results = _phase5_calibration_check()
        results["calibration_warnings"] = cal_results.get("warnings", [])
        results["phase_results"]["calibration"] = cal_results
        logger.info("Phase 5 done: %d calibration warnings",
                     len(results["calibration_warnings"]))
    except Exception as e:
        logger.error("Phase 5 (calibration) failed: %s", e, exc_info=True)
        results["errors"].append(f"Calibration check failed: {e}")

    # ── PHASE 6: CLV STATS ───────────────────────────────────
    try:
        clv_sync = update_closing_lines(target_date, days_back=7)
        clv = compute_clv_stats(days=30)
        clv["sync"] = clv_sync
        results["phase_results"]["clv"] = clv
        logger.info("Phase 6 done: mean CLV=%.3f", clv.get("mean_clv", 0))
    except Exception as e:
        logger.error("Phase 6 (CLV) failed: %s", e, exc_info=True)
        results["errors"].append(f"CLV stats failed: {e}")

    # ── PHASE 7: LOG RESULTS ─────────────────────────────────
    try:
        _phase7_log_results(results)
    except Exception as e:
        logger.error("Phase 7 (logging) failed: %s", e, exc_info=True)
        results["errors"].append(f"Result logging failed: {e}")

    logger.info("Nightly cycle complete: %d errors", len(results["errors"]))
    return results


# ─────────────────────────────────────────────
# PHASE IMPLEMENTATIONS
# ─────────────────────────────────────────────

def _phase1_autograde(target_date: str) -> dict:
    """Pull box scores and grade all predictions from target_date."""
    raw = auto_grade_date(target_date)
    # auto_grade_date returns:
    #   graded, not_matched, skipped_not_final, slip_picks_graded, results, errors
    wins = sum(1 for r in raw.get("results", []) if r.get("result") == "W")
    losses = sum(1 for r in raw.get("results", []) if r.get("result") == "L")
    pushes = sum(1 for r in raw.get("results", []) if r.get("result") == "push")

    return {
        "total_graded": raw.get("graded", 0),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "tracking_repairs": raw.get("tracking_repairs", {}),
        "projected_stats_graded": raw.get("projected_stats_graded", 0),
        "board_entries_graded": raw.get("board_entries_graded", 0),
        "not_matched": raw.get("not_matched", 0),
        "pending": raw.get("skipped_not_final", 0),
        "slip_picks_graded": raw.get("slip_picks_graded", 0),
        "grader_errors": raw.get("errors", []),
    }


def _phase2_compute_metrics() -> dict:
    """Run autolearn adjustment cycle which computes Brier, LogLoss, accuracy."""
    raw = run_adjustment_cycle()
    # run_adjustment_cycle returns:
    #   adjusted, changes, accuracy_before, accuracy_by_grade, brier_score,
    #   log_loss, version_old, version_new, kill_switch, analysis, reason,
    #   ensemble_updated, ensemble_weights

    # Also get board-level stats (all props, not just bets)
    board = {}
    projection_diag = {}
    shadow = {}
    try:
        board = get_board_stats(days=30)
    except Exception:
        pass
    try:
        projection_diag = get_projection_diagnostics(days_back=30)
    except Exception:
        pass
    try:
        shadow = get_shadow_sample_stats(days=30)
    except Exception:
        pass

    return {
        "adjusted": raw.get("adjusted", False),
        "accuracy_before": raw.get("accuracy_before"),
        "accuracy_by_grade": raw.get("accuracy_by_grade", {}),
        "brier_score": raw.get("brier_score"),
        "log_loss": raw.get("log_loss"),
        "version_old": raw.get("version_old"),
        "version_new": raw.get("version_new"),
        "reason": raw.get("reason", ""),
        "kill_switch": raw.get("kill_switch", {}),
        "changes": raw.get("changes", []),
        "board_stats": board,
        "projection_diagnostics": projection_diag,
        "shadow_sample": shadow,
    }


def _phase3_update_weights() -> dict:
    """Run Hedge-style ensemble weight update with safety checks.

    Uses conservative defaults: low learning rate (0.05), high min samples (50),
    requires data from 3+ distinct game days, and uses per-day averaging to
    prevent outlier blowout games from skewing the signal.
    """
    # Check if we should skip — only update weights every 3+ days to avoid
    # overfitting to small sample noise (e.g. one 17-4 blowout day)
    conn = get_connection()
    try:
        last_update = conn.execute("""
            SELECT date FROM ensemble_history
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        if last_update:
            from datetime import datetime
            last_dt = datetime.strptime(last_update[0], "%Y-%m-%d").date()
            days_since = (datetime.now().date() - last_dt).days
            if days_since < 3:
                conn.close()
                return {
                    "old_weights": {},
                    "new_weights": {},
                    "updated": False,
                    "reason": f"Too soon since last update ({days_since}d < 3d min interval)",
                }
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    raw = update_ensemble_weights(
        learning_rate=0.05,
        min_samples=50,
        lookback_days=14,
        min_days=3,
    )
    return raw


def _phase4_check_drift() -> dict:
    """Run drift detection on recent graded predictions."""
    # Get recent graded predictions as list of dicts for check_model_health
    try:
        df = get_graded_predictions(limit=500)
        if df.empty:
            return {"alerts": [], "healthy": True, "reason": "No graded predictions yet"}

        # check_model_health expects list of dicts with stat_type, confidence, result
        preds = []
        for _, row in df.iterrows():
            preds.append({
                "stat_type": row.get("stat_type", row.get("stat_internal", "")),
                "confidence": row.get("confidence", 0.5),
                "result": row.get("result", ""),
            })

        health = check_model_health(preds, min_sample=50)
        # Returns: healthy, overall_brier, overall_accuracy, overall_logloss,
        #          by_prop, alerts, regime_change
        return {
            "healthy": health.get("healthy", True),
            "alerts": health.get("alerts", []),
            "overall_brier": health.get("overall_brier"),
            "overall_accuracy": health.get("overall_accuracy"),
            "regime_change": health.get("regime_change", False),
            "by_prop": health.get("by_prop", {}),
        }
    except Exception as e:
        logger.warning("Drift detection failed: %s", e)
        return {"alerts": [f"Drift check error: {e}"], "healthy": None}


def _phase5_calibration_check() -> dict:
    """Check if predicted probabilities match actual outcomes."""
    df = get_graded_predictions(limit=2000)
    if df.empty or len(df) < 50:
        return {"warnings": [], "n_total": len(df),
                "reason": "Not enough graded predictions for calibration"}

    # Need confidence (predicted prob) and result (W/L)
    if "confidence" not in df.columns or "result" not in df.columns:
        return {"warnings": [], "reason": "Missing confidence or result columns"}

    probs = df["confidence"].values.astype(float)
    outcomes = (df["result"] == "W").values.astype(float)

    # Bin predictions into 5 buckets
    warnings = []
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]

    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n_in_bin = int(mask.sum())

        if n_in_bin >= 20:
            predicted_avg = float(probs[mask].mean())
            actual_avg = float(outcomes[mask].mean())
            gap = abs(predicted_avg - actual_avg)

            if gap > 0.10:  # 10+ pct point miscalibration
                warnings.append({
                    "bin": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
                    "predicted": round(predicted_avg, 3),
                    "actual": round(actual_avg, 3),
                    "gap": round(gap, 3),
                    "n": n_in_bin,
                })

    # Fit isotonic calibration if enough data (200+ picks)
    calibrator_fitted = False
    if len(df) >= 200:
        try:
            from sklearn.isotonic import IsotonicRegression
            import pickle

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, outcomes)

            cal_path = Path("data/calibration/isotonic_overall.pkl")
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cal_path, "wb") as f:
                pickle.dump(ir, f)
            calibrator_fitted = True

            # Per prop type if enough data
            if "stat_type" in df.columns:
                for pt in df["stat_type"].unique():
                    pt_mask = (df["stat_type"] == pt).values
                    if pt_mask.sum() >= 100:
                        ir_pt = IsotonicRegression(out_of_bounds="clip")
                        ir_pt.fit(probs[pt_mask], outcomes[pt_mask])
                        with open(f"data/calibration/isotonic_{pt}.pkl", "wb") as f:
                            pickle.dump(ir_pt, f)

            logger.info("Isotonic calibrator fitted on %d picks", len(df))
        except ImportError:
            logger.info("sklearn not available — skipping isotonic calibration")
        except Exception as e:
            logger.warning("Isotonic calibration failed: %s", e)

    return {
        "warnings": warnings,
        "n_total": len(df),
        "calibrator_fitted": calibrator_fitted,
    }


def _phase7_log_results(results: dict):
    """Store nightly cycle results in SQLite for historical tracking."""
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nightly_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_graded INTEGER,
            overall_accuracy REAL,
            overall_brier REAL,
            overall_logloss REAL,
            weights_updated INTEGER,
            drift_alerts TEXT,
            calibration_warnings TEXT,
            errors TEXT,
            full_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    metrics = results.get("phase_results", {}).get("metrics", {})
    grading = results.get("phase_results", {}).get("grading", {})

    conn.execute("""
        INSERT INTO nightly_logs
        (date, total_graded, overall_accuracy, overall_brier, overall_logloss,
         weights_updated, drift_alerts, calibration_warnings, errors, full_results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        results["date"],
        grading.get("total_graded", 0),
        metrics.get("accuracy_before"),
        metrics.get("brier_score"),
        metrics.get("log_loss"),
        1 if results.get("weights_updated") else 0,
        json.dumps(results.get("drift_alerts", [])),
        json.dumps(results.get("calibration_warnings", [])),
        json.dumps(results.get("errors", [])),
        json.dumps(results, default=str),
    ))
    conn.commit()
    logger.info("Nightly results logged to nightly_logs table")


# ─────────────────────────────────────────────
# HISTORY ACCESS (for Dashboard)
# ─────────────────────────────────────────────

def get_nightly_history(days: int = 14) -> list:
    """
    Get recent nightly log entries for Dashboard display.

    Returns list of dicts with date, accuracy, brier, etc.
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT date, total_graded, overall_accuracy, overall_brier,
                   overall_logloss, weights_updated, drift_alerts
            FROM nightly_logs
            ORDER BY date DESC
            LIMIT ?
        """, (days,)).fetchall()
    except Exception:
        # Table doesn't exist yet
        return []

    return [
        {
            "date": r[0],
            "total_graded": r[1],
            "accuracy": r[2],
            "brier": r[3],
            "logloss": r[4],
            "weights_updated": bool(r[5]),
            "drift_alerts": json.loads(r[6]) if r[6] else [],
        }
        for r in rows
    ]
