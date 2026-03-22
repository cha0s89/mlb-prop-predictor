import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import database


class ProjectedStatsPersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmpdir.name) / "predictions.db"
        database.init_db()
        database.init_projected_stats_table()

    def tearDown(self):
        database.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()

    def _fetchone(self, query, params=()):
        conn = sqlite3.connect(str(database.DB_PATH))
        try:
            cur = conn.execute(query, params)
            return cur.fetchone()
        finally:
            conn.close()

    def test_save_projected_stats_upserts_instead_of_inserting_duplicates(self):
        first = {
            "game_date": "2026-03-22",
            "player_name": "Aaron Judge",
            "team": "NYY",
            "stat_type": "total_bases",
            "projected_value": 1.7,
            "line": 1.5,
            "pick": "MORE",
            "confidence": 0.61,
            "rating": "B",
        }
        second = dict(first)
        second["projected_value"] = 1.9
        second["confidence"] = 0.64
        second["rating"] = "A"

        database.save_projected_stats([first])
        database.save_projected_stats([second])

        row = self._fetchone(
            """
            SELECT COUNT(*), projected_value, confidence, rating
            FROM projected_stats
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
            """,
            ("2026-03-22", "Aaron Judge", "total_bases"),
        )

        self.assertEqual(row[0], 1)
        self.assertAlmostEqual(row[1], 1.9, places=3)
        self.assertAlmostEqual(row[2], 0.64, places=3)
        self.assertEqual(row[3], "A")

    def test_grade_projected_stats_updates_the_unique_row(self):
        prediction = {
            "game_date": "2026-03-22",
            "player_name": "Aaron Judge",
            "team": "NYY",
            "stat_type": "total_bases",
            "projected_value": 1.8,
            "line": 1.5,
            "pick": "MORE",
            "confidence": 0.62,
            "rating": "B",
        }
        database.save_projected_stats([prediction])

        database.grade_projected_stats(
            "2026-03-22",
            {("Aaron Judge", "total_bases"): 2.0},
        )

        row = self._fetchone(
            """
            SELECT actual_value, was_correct
            FROM projected_stats
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
            """,
            ("2026-03-22", "Aaron Judge", "total_bases"),
        )

        self.assertEqual(row[0], 2.0)
        self.assertEqual(row[1], 1)


if __name__ == "__main__":
    unittest.main()
