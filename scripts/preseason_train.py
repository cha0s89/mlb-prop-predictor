#!/usr/bin/env python3
"""
One-shot preseason training orchestration.

Rebuilds the 2025 backtest with the current projection code, then runs the
offline tuner against that freshly generated dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOCK_PATH = LOG_DIR / "preseason_train.lock.json"
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def setup_logging(verbose: bool = False) -> Path:
    now = datetime.now(PACIFIC_TZ)
    log_file = LOG_DIR / f"preseason_train_{now.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log file: %s", log_file)
    return log_file


def _pid_is_running(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(backtest_path: str) -> None:
    if LOCK_PATH.exists():
        try:
            existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing_pid = int(existing.get("pid") or 0)
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"Another preseason training run is already active (pid={existing_pid}). "
                f"Lock file: {LOCK_PATH}"
            )
        LOCK_PATH.unlink(missing_ok=True)

    LOCK_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(PACIFIC_TZ).isoformat(),
                "backtest_path": backtest_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(existing.get("pid") or 0) == os.getpid():
                LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preseason backtest + tuning")
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument("--backtest-path", default="data/backtest/backtest_2025.json")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Resume an existing backtest file instead of rebuilding it")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run tuning in dry-run mode after backtest")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("preseason_train")
    acquire_lock(args.backtest_path)
    logger.info("Starting preseason training run at %s", datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %I:%M %p %Z"))
    logger.info("Backtest file: %s", args.backtest_path)
    logger.info("Lock file: %s", LOCK_PATH)
    logger.info(
        "Progress file: %s",
        Path(args.backtest_path).with_name(f"{Path(args.backtest_path).stem}_progress.json"),
    )

    try:
        if not args.skip_backtest:
            from src.backtester import generate_backtest_report, load_results, run_backtest

            logger.info("Phase 1: rebuilding backtest (%s -> %s)", args.start, args.end)
            run_backtest(
                start_date=args.start,
                end_date=args.end,
                filepath=args.backtest_path,
                clear_existing=not args.keep_existing,
            )
            results = load_results(args.backtest_path)
            logger.info("Phase 1 complete: %s graded rows", len(results))
            if not results:
                logger.error("Backtest rebuild produced 0 rows; aborting tuning.")
                return 2
            report = generate_backtest_report(results)
            overall = report.get("overall", {})
            logger.info(
                "Backtest summary: %s wins / %s losses / %s pushes (%.2f%%)",
                overall.get("wins", 0),
                overall.get("losses", 0),
                overall.get("pushes", 0),
                float(overall.get("win_pct", 0.0)),
            )
        else:
            logger.info("Phase 1: skipped backtest rebuild")

        logger.info("Phase 2: running weekly_tune against rebuilt backtest")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "weekly_tune.py"),
            "--backtest-path",
            args.backtest_path,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.verbose:
            cmd.append("--verbose")

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        logger.info("weekly_tune exited with code %s", result.returncode)
        return int(result.returncode)
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
