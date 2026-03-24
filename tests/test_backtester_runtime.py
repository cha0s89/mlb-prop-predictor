import json
import tempfile
import unittest
from pathlib import Path

from src import backtester


class BacktesterRuntimeTests(unittest.TestCase):
    def test_progress_path_tracks_results_filename(self):
        progress = backtester._progress_path_for_results("data/backtest/backtest_2025.json")
        self.assertEqual(progress.name, "backtest_2025_progress.json")

    def test_save_results_and_progress_write_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            progress_path = Path(tmpdir) / "results_progress.json"
            rows = [{"game_date": "2025-04-01", "player_name": "Test", "result": "W"}]

            backtester.save_results(rows, filepath=str(results_path))
            backtester._write_backtest_progress(
                str(results_path),
                status="running",
                current_date="2025-04-01",
                processed_results=1,
            )

            loaded_rows = json.loads(results_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))

            self.assertEqual(loaded_rows, rows)
            self.assertEqual(progress["status"], "running")
            self.assertEqual(progress["current_date"], "2025-04-01")
            self.assertEqual(progress["processed_results"], 1)


if __name__ == "__main__":
    unittest.main()
