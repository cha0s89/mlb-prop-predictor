#!/usr/bin/env python3
"""
Nightly Cycle CLI — auto-grade, compute metrics, update weights, drift check.

Scheduled: daily at 11:45 PM Pacific (after games finish).
Can also be run manually: python scripts/nightly_cycle.py [--date 2026-03-27]

Exit codes:
    0 — success
    1 — partial failure (some phases errored)
    2 — total failure
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    now = datetime.now(PACIFIC_TZ)
    log_file = LOG_DIR / f"nightly_cycle_{now.strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log file: %s", log_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly auto-grade and model update cycle")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (defaults to today)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("nightly_cycle")

    target_date = args.date or date.today().isoformat()

    logger.info("=" * 60)
    logger.info("Nightly Cycle starting at %s",
                datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %I:%M %p %Z"))
    logger.info("Target date: %s", target_date)
    logger.info("=" * 60)

    from src.database import init_db
    from src.nightly import run_nightly_cycle

    try:
        init_db()
    except Exception as exc:
        logger.error("DB init failed: %s", exc)

    result = run_nightly_cycle(target_date=target_date)

    n_errors = len(result.get("errors", []))
    grading = result.get("phase_results", {}).get("grading", {})
    metrics = result.get("phase_results", {}).get("metrics", {})

    logger.info("-" * 60)
    logger.info("GRADING: %d total, %d W, %d L, %d push",
                grading.get("total_graded", 0),
                grading.get("wins", 0),
                grading.get("losses", 0),
                grading.get("pushes", 0))
    if metrics.get("brier_score") is not None:
        logger.info("METRICS: Brier=%.4f, LogLoss=%s, Accuracy=%s",
                     metrics.get("brier_score", 0),
                     metrics.get("log_loss", "N/A"),
                     metrics.get("accuracy_before", "N/A"))
    logger.info("WEIGHTS UPDATED: %s", result.get("weights_updated", False))
    if result.get("drift_alerts"):
        for alert in result["drift_alerts"]:
            logger.warning("DRIFT ALERT: %s", alert)
    if result.get("calibration_warnings"):
        for warn in result["calibration_warnings"]:
            logger.warning("CALIBRATION: %s", warn)
    if result["errors"]:
        for err in result["errors"]:
            logger.error("  ERROR: %s", err)
    logger.info("-" * 60)

    # Persist summary
    summary_path = LOG_DIR / "last_nightly_cycle.json"
    try:
        summary_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass

    if n_errors > 0 and grading.get("total_graded", 0) == 0:
        logger.error("Total failure — exiting with code 2")
        return 2
    if n_errors > 0:
        logger.warning("Partial success — exiting with code 1")
        return 1

    logger.info("Nightly cycle completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
