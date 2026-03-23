import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import board_logger, database


class ModelQAMetricsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmpdir.name) / "predictions.db"
        database.init_db()
        database.init_projected_stats_table()
        board_logger.init_board_table()

    def tearDown(self):
        database.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()

    def _fetchall(self, query, params=()):
        conn = sqlite3.connect(str(database.DB_PATH))
        try:
            cur = conn.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def test_projection_diagnostics_compute_mae_rmse_and_bias(self):
        database.save_projected_stats([
            {
                "game_date": "2026-03-20",
                "player_name": "Aaron Judge",
                "team": "NYY",
                "stat_type": "hits",
                "projected_value": 1.5,
                "line": 1.5,
                "pick": "MORE",
                "confidence": 0.60,
                "rating": "B",
            },
            {
                "game_date": "2026-03-20",
                "player_name": "Paul Skenes",
                "team": "PIT",
                "stat_type": "pitcher_strikeouts",
                "projected_value": 7.0,
                "line": 7.5,
                "pick": "LESS",
                "confidence": 0.58,
                "rating": "C",
            },
        ])
        database.grade_projected_stats(
            "2026-03-20",
            {
                ("Aaron Judge", "hits"): 2.0,
                ("Paul Skenes", "pitcher_strikeouts"): 6.0,
            },
        )

        diag = database.get_projection_diagnostics(days_back=30)

        self.assertEqual(diag["total"], 2)
        self.assertAlmostEqual(diag["mae"], 0.75, places=3)
        self.assertAlmostEqual(diag["rmse"], ((0.5 ** 2 + (-1.0) ** 2) / 2) ** 0.5, places=3)
        self.assertAlmostEqual(diag["bias"], -0.25, places=3)
        self.assertIn("hits", diag["by_stat_type"])
        self.assertIn("pitcher_strikeouts", diag["by_stat_type"])

    def test_shadow_sample_selection_and_stats(self):
        board_rows = []
        for idx in range(8):
            board_rows.append({
                "game_date": "2026-03-21",
                "player_name": f"Player {idx}",
                "team": "NYY" if idx % 2 == 0 else "BOS",
                "stat_internal": "hits" if idx < 4 else "runs",
                "stat_type": "Hits" if idx < 4 else "Runs",
                "line": 0.5 if idx % 2 == 0 else 1.5,
                "pick": "MORE" if idx % 3 else "LESS",
                "confidence": 0.52 + idx * 0.02,
                "rating": "A" if idx % 2 == 0 else "B",
                "projection": 1.0 + idx * 0.1,
            })
        board_logger.log_board_snapshot(board_rows)

        first = board_logger.ensure_shadow_sample("2026-03-21", sample_size=4)
        second = board_logger.ensure_shadow_sample("2026-03-21", sample_size=4)

        self.assertEqual(first["shadow_sample_size"], 4)
        self.assertEqual(first["selected_now"], 4)
        self.assertEqual(second["selected_now"], 0)

        shadow_rows = self._fetchall(
            """
            SELECT id FROM daily_board
            WHERE date = ? AND is_shadow_sample = 1
            """,
            ("2026-03-21",),
        )
        self.assertEqual(len(shadow_rows), 4)

        for board_id, in shadow_rows[:2]:
            conn = sqlite3.connect(str(database.DB_PATH))
            conn.execute(
                "UPDATE daily_board SET actual_stat = ?, outcome = ? WHERE id = ?",
                (1.0, 1, board_id),
            )
            conn.commit()
            conn.close()

        stats = board_logger.get_shadow_sample_stats(days=30)
        self.assertEqual(stats["total_sampled"], 4)
        self.assertEqual(stats["graded"], 2)
        self.assertEqual(stats["pending"], 2)
        self.assertAlmostEqual(stats["accuracy"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
