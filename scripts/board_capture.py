#!/usr/bin/env python3
"""
Board Capture CLI — runs the full headless board-build pipeline.

Scheduled: every 2 hours from 8 AM to 6 PM Pacific daily.
Can also be run manually: python scripts/board_capture.py [--dry-run] [--skip-sharp]

Exit codes:
    0 — success
    1 — partial failure (some steps errored but predictions were generated)
    2 — total failure (no predictions generated)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    now = datetime.now(PACIFIC_TZ)
    log_file = LOG_DIR / f"board_capture_{now.strftime('%Y%m%d_%H%M%S')}.log"
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
    parser = argparse.ArgumentParser(description="Headless MLB board capture")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate predictions but don't persist to DB")
    parser.add_argument("--skip-sharp", action="store_true",
                        help="Skip sharp-odds fetch (saves API credits)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("board_capture")

    logger.info("=" * 60)
    logger.info("Board Capture starting at %s",
                datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %I:%M %p %Z"))
    logger.info("Flags: dry_run=%s, skip_sharp=%s", args.dry_run, args.skip_sharp)
    logger.info("=" * 60)

    from src.headless_board import build_board

    result = build_board(
        skip_sharp=args.skip_sharp,
        dry_run=args.dry_run,
    )

    n_preds = result["stats"].get("predictions", 0)
    n_edges = result["stats"].get("edges", 0)
    n_errors = len(result["errors"])

    logger.info("-" * 60)
    logger.info("RESULTS: %d predictions, %d edges, %d errors", n_preds, n_edges, n_errors)
    if result["errors"]:
        for err in result["errors"]:
            logger.error("  ERROR: %s", err)
    logger.info("-" * 60)

    # Write a compact JSON summary for the nightly cycle to reference
    summary_path = LOG_DIR / "last_board_capture.json"
    summary = {
        "timestamp": datetime.now(PACIFIC_TZ).isoformat(),
        "predictions": n_preds,
        "edges": n_edges,
        "errors": result["errors"],
        "dry_run": args.dry_run,
    }
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass

    if n_preds == 0:
        logger.error("No predictions generated — exiting with code 2")
        return 2
    if n_errors > 0:
        logger.warning("Partial success — exiting with code 1")
        return 1

    logger.info("Board capture completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
