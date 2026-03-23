import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import clv, database, line_snapshots, slips


class ClvTrackingTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmpdir.name) / "predictions.db"
        database.init_db()
        database.init_clv_table()
        line_snapshots.init_line_snapshots_table()
        slips.init_slips_table()

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

    def test_create_slip_tracks_clv_and_grading_updates_outcome(self):
        pred_id = database.log_prediction({
            "game_date": "2026-03-26",
            "player_name": "Paul Skenes",
            "stat_type": "Pitcher Strikeouts",
            "stat_internal": "pitcher_strikeouts",
            "line": 6.5,
            "projection": 7.4,
            "pick": "MORE",
            "confidence": 0.66,
            "win_prob": 0.66,
            "rating": "A",
        })

        slip_id = slips.create_slip(
            "2026-03-26",
            "2_power",
            5.0,
            [{
                "prediction_id": pred_id,
                "player_name": "Paul Skenes",
                "stat_type": "Pitcher Strikeouts",
                "line": 6.5,
                "pick": "MORE",
                "game_date": "2026-03-26",
            }],
        )
        self.assertEqual(slip_id, 1)

        clv_row = self._fetchone(
            """
            SELECT prediction_id, opening_prob, projection
            FROM clv_tracking
            WHERE prediction_id = ?
            """,
            (pred_id,),
        )
        self.assertEqual(clv_row[0], pred_id)
        self.assertAlmostEqual(clv_row[1], 0.66, places=3)
        self.assertAlmostEqual(clv_row[2], 7.4, places=3)

        conn = sqlite3.connect(str(database.DB_PATH))
        conn.execute("""
            INSERT INTO line_snapshots
            (snapshot_time, game_date, player_name, stat_type, pp_line, line_type, start_time)
            VALUES
            ('2026-03-26T10:00:00', '2026-03-26', 'Paul Skenes', 'pitcher_strikeouts', 6.5, 'standard', '2026-03-26T20:10:00Z'),
            ('2026-03-26T18:00:00', '2026-03-26', 'Paul Skenes', 'pitcher_strikeouts', 7.5, 'standard', '2026-03-26T20:10:00Z')
        """)
        conn.commit()
        conn.close()

        sync = clv.update_closing_lines("2026-03-26")
        self.assertEqual(sync["updated"], 1)

        updated = self._fetchone(
            """
            SELECT closing_line, closing_prob, clv_points, beat_close
            FROM clv_tracking
            WHERE prediction_id = ?
            """,
            (pred_id,),
        )
        self.assertAlmostEqual(updated[0], 7.5, places=3)
        self.assertIsNotNone(updated[1])
        self.assertGreater(updated[2], 0.0)
        self.assertEqual(updated[3], 1)

        slip_pick_id = self._fetchone(
            "SELECT id FROM slip_picks WHERE slip_id = ?",
            (slip_id,),
        )[0]
        result = slips.grade_slip_pick(slip_pick_id, 8.0)
        self.assertEqual(result, "W")

        outcome = self._fetchone(
            "SELECT outcome FROM clv_tracking WHERE prediction_id = ?",
            (pred_id,),
        )[0]
        self.assertEqual(outcome, 1)


if __name__ == "__main__":
    unittest.main()
