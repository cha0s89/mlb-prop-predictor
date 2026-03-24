import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "preseason_train.py"
SPEC = importlib.util.spec_from_file_location("preseason_train_script", SCRIPT_PATH)
preseason_train = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(preseason_train)


class PreseasonTrainScriptTests(unittest.TestCase):
    def test_acquire_and_release_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_lock = preseason_train.LOCK_PATH
            preseason_train.LOCK_PATH = Path(tmpdir) / "preseason.lock.json"
            try:
                preseason_train.acquire_lock("data/backtest/backtest_2025.json")
                self.assertTrue(preseason_train.LOCK_PATH.exists())
                preseason_train.release_lock()
                self.assertFalse(preseason_train.LOCK_PATH.exists())
            finally:
                preseason_train.LOCK_PATH = original_lock

    def test_acquire_lock_rejects_live_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_lock = preseason_train.LOCK_PATH
            preseason_train.LOCK_PATH = Path(tmpdir) / "preseason.lock.json"
            preseason_train.LOCK_PATH.write_text(
                '{\n'
                f'  "pid": {os.getpid()},\n'
                '  "started_at": "2026-03-23T19:00:00",\n'
                '  "backtest_path": "data/backtest/backtest_2025.json"\n'
                '}',
                encoding="utf-8",
            )
            try:
                with self.assertRaises(RuntimeError):
                    preseason_train.acquire_lock("data/backtest/backtest_2025.json")
            finally:
                preseason_train.LOCK_PATH = original_lock


if __name__ == "__main__":
    unittest.main()
