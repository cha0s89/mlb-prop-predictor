#!/usr/bin/env python3
"""
Weekly Offline Tune CLI — optimizes confidence floors and model parameters
from accumulated backtest data.

Scheduled: Sundays at 4:00 AM Pacific.
Can also be run manually: python scripts/weekly_tune.py [--dry-run]

This writes updated weights to data/weights/runtime_override.json so
automation-generated weights don't dirty the git repo.  The app checks
for this file at startup and merges it with the base current.json.

Exit codes:
    0 — success (or no-op if insufficient data)
    1 — partial failure
    2 — total failure
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RUNTIME_WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights" / "runtime_override.json"


def setup_logging(verbose: bool = False) -> None:
    now = datetime.now(PACIFIC_TZ)
    log_file = LOG_DIR / f"weekly_tune_{now.strftime('%Y%m%d_%H%M%S')}.log"
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
    parser = argparse.ArgumentParser(description="Weekly offline model tuning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute optimal params but don't save")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("weekly_tune")

    logger.info("=" * 60)
    logger.info("Weekly Tune starting at %s",
                datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %I:%M %p %Z"))
    logger.info("Flags: dry_run=%s", args.dry_run)
    logger.info("=" * 60)

    from src.database import init_db

    try:
        init_db()
    except Exception as exc:
        logger.error("DB init failed: %s", exc)

    errors = []
    floor_result = None
    model_result = None

    # ── Phase 1: Confidence floor optimization ───────────────────────────────
    try:
        from src.offline_tuner import optimize_floors
        logger.info("Phase 1: Optimizing confidence floors...")
        floor_result = optimize_floors()
        if floor_result.get("status") == "updated":
            logger.info("  Floor optimization produced updates: %s",
                        json.dumps(floor_result.get("changes", {}), indent=2))
        else:
            logger.info("  Floor optimization: %s", floor_result.get("status", "no change"))
    except Exception as exc:
        errors.append(f"Floor optimization: {exc}")
        logger.error("Phase 1 failed: %s", exc, exc_info=True)

    # ── Phase 2: Model parameter tuning ──────────────────────────────────────
    try:
        from src.offline_tuner import tune_model_parameters
        logger.info("Phase 2: Tuning model parameters...")
        model_result = tune_model_parameters()
        if model_result.get("status") == "updated":
            logger.info("  Model tuning produced updates for: %s",
                        list(model_result.get("changes", {}).keys()))
        else:
            logger.info("  Model tuning: %s", model_result.get("status", "no change"))
    except ImportError:
        logger.info("Phase 2: tune_model_parameters not available, skipping")
    except Exception as exc:
        errors.append(f"Model tuning: {exc}")
        logger.error("Phase 2 failed: %s", exc, exc_info=True)

    # ── Save runtime override (unless dry_run) ───────────────────────────────
    if not args.dry_run:
        try:
            from src.autolearn import load_current_weights
            current = load_current_weights()
            # Merge floor and model changes into runtime override
            override = {}
            if RUNTIME_WEIGHTS_PATH.exists():
                try:
                    override = json.loads(RUNTIME_WEIGHTS_PATH.read_text(encoding="utf-8"))
                except Exception:
                    override = {}

            if floor_result and floor_result.get("status") == "updated":
                floors = override.get("per_prop_confidence_floors",
                                      current.get("per_prop_confidence_floors", {}))
                floors.update(floor_result.get("changes", {}))
                override["per_prop_confidence_floors"] = floors

            if model_result and model_result.get("status") == "updated":
                override.update(model_result.get("changes", {}))

            if override:
                RUNTIME_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                RUNTIME_WEIGHTS_PATH.write_text(
                    json.dumps(override, indent=2), encoding="utf-8"
                )
                logger.info("Runtime override saved: %s", RUNTIME_WEIGHTS_PATH)
            else:
                logger.info("No changes to save to runtime override")
        except Exception as exc:
            errors.append(f"Save runtime override: {exc}")
            logger.error("Save failed: %s", exc, exc_info=True)
    else:
        logger.info("Dry run — skipping save")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now(PACIFIC_TZ).isoformat(),
        "floor_result": floor_result,
        "model_result": model_result,
        "errors": errors,
        "dry_run": args.dry_run,
    }
    summary_path = LOG_DIR / "last_weekly_tune.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    logger.info("-" * 60)
    if errors:
        for err in errors:
            logger.error("  ERROR: %s", err)
        logger.warning("Completed with %d errors", len(errors))
        return 1

    logger.info("Weekly tune completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
