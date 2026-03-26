import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import autograder, board_logger, database


class AutograderWiringTests(unittest.TestCase):
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

    def test_auto_grade_prediction_supports_hits_allowed_pitchers(self):
        pred_id = database.log_prediction(
            {
                "game_date": "2026-03-26",
                "player_name": "Jacob deGrom",
                "stat_type": "Hits Allowed",
                "stat_internal": "hits_allowed",
                "line": 5.5,
                "projection": 4.8,
                "pick": "LESS",
                "confidence": 0.61,
                "rating": "B",
            }
        )
        pred_row = database.get_ungraded_predictions("2026-03-26").iloc[0]
        player_stats_list = [
            {
                "player_name": "Jacob deGrom",
                "player_type": "pitcher",
                "hits_allowed": 4,
                "pitcher_strikeouts": 8,
                "pitching_outs": 18,
                "earned_runs": 2,
                "walks_allowed": 1,
            }
        ]

        result = autograder.auto_grade_prediction(pred_row, player_stats_list)

        self.assertEqual(pred_id, 1)
        self.assertEqual(result, "W")
        row = self._fetchall(
            "SELECT actual_result, result FROM predictions WHERE id = ?",
            (pred_id,),
        )[0]
        self.assertEqual(row[0], 4.0)
        self.assertEqual(row[1], "W")

    def test_grade_tracking_tables_updates_projected_stats_and_all_board_rows(self):
        database.save_projected_stats([
            {
                "game_date": "2026-03-26",
                "player_name": "Aaron Judge",
                "team": "NYY",
                "stat_type": "hits",
                "projected_value": 1.7,
                "line": 1.5,
                "pick": "MORE",
                "confidence": 0.61,
                "rating": "B",
            }
        ])
        board_logger.log_board_snapshot([
            {
                "game_date": "2026-03-26",
                "player_name": "Aaron Judge",
                "team": "NYY",
                "stat_internal": "hits",
                "stat_type": "Hits",
                "line": 1.5,
                "pick": "MORE",
                "confidence": 0.61,
                "rating": "B",
                "projection": 1.7,
            },
            {
                "game_date": "2026-03-26",
                "player_name": "Aaron Judge",
                "team": "NYY",
                "stat_internal": "hits",
                "stat_type": "Hits",
                "line": 1.5,
                "pick": "LESS",
                "confidence": 0.55,
                "rating": "C",
                "projection": 1.2,
            },
        ])

        player_stats_list = [
            {
                "player_name": "Aaron Judge",
                "player_type": "batter",
                "hits": 2,
                "singles": 1,
                "doubles": 1,
                "triples": 0,
                "home_runs": 0,
                "rbi": 1,
                "runs": 1,
                "stolen_bases": 0,
                "walks": 1,
                "hbp": 0,
                "strikeouts": 1,
                "at_bats": 4,
                "total_bases": 3,
                "hits_runs_rbis": 4,
                "fantasy_score": 12.0,
            }
        ]

        result = autograder._grade_tracking_tables_for_date("2026-03-26", player_stats_list)

        self.assertEqual(result["projected_stats_graded"], 1)
        self.assertEqual(result["board_entries_graded"], 1)

        projected_row = self._fetchall(
            """
            SELECT actual_value, was_correct
            FROM projected_stats
            WHERE game_date = ? AND player_name = ? AND stat_type = ?
            """,
            ("2026-03-26", "Aaron Judge", "hits"),
        )[0]
        self.assertEqual(projected_row[0], 2.0)
        self.assertEqual(projected_row[1], 1)

        board_rows = self._fetchall(
            """
            SELECT actual_stat, outcome
            FROM daily_board
            WHERE date = ? AND player_name = ? AND prop_type = ?
            ORDER BY id
            """,
            ("2026-03-26", "Aaron Judge", "hits"),
        )
        self.assertEqual(len(board_rows), 1)
        self.assertEqual(board_rows[0][0], 2.0)
        self.assertEqual(board_rows[0][1], 1)

    def test_log_batch_predictions_uses_per_prediction_game_date(self):
        ids = database.log_batch_predictions([
            {
                "game_date": "2026-03-26",
                "player_name": "Aaron Judge",
                "stat_type": "Hits",
                "stat_internal": "hits",
                "line": 1.5,
                "projection": 1.8,
                "pick": "MORE",
                "confidence": 0.61,
                "rating": "B",
            },
            {
                "game_date": "2026-03-27",
                "player_name": "Paul Skenes",
                "stat_type": "Pitcher Strikeouts",
                "stat_internal": "pitcher_strikeouts",
                "line": 7.5,
                "projection": 8.1,
                "pick": "MORE",
                "confidence": 0.59,
                "rating": "C",
            },
        ])

        self.assertEqual(ids, [1, 2])
        rows = self._fetchall(
            "SELECT player_name, game_date FROM predictions ORDER BY id"
        )
        self.assertEqual(
            rows,
            [
                ("Aaron Judge", "2026-03-26"),
                ("Paul Skenes", "2026-03-27"),
            ],
        )

    def test_log_board_snapshot_upserts_duplicate_rows_on_rerun(self):
        snapshot = {
            "game_date": "2026-03-26",
            "player_name": "Aaron Judge",
            "team": "NYY",
            "stat_internal": "hits",
            "stat_type": "Hits",
            "line": 1.5,
            "pick": "MORE",
            "confidence": 0.61,
            "rating": "B",
            "projection": 1.7,
            "line_type": "standard",
        }
        updated_snapshot = dict(snapshot)
        updated_snapshot["confidence"] = 0.66
        updated_snapshot["rating"] = "A"
        updated_snapshot["projection"] = 1.9

        board_logger.log_board_snapshot([snapshot])
        board_logger.log_board_snapshot([updated_snapshot])

        rows = self._fetchall(
            """
            SELECT COUNT(*), confidence, grade, projection
            FROM daily_board
            WHERE date = ? AND player_name = ? AND prop_type = ?
            """,
            ("2026-03-26", "Aaron Judge", "hits"),
        )[0]
        self.assertEqual(rows[0], 1)
        self.assertAlmostEqual(rows[1], 0.66, places=3)
        self.assertEqual(rows[2], "A")
        self.assertAlmostEqual(rows[3], 1.9, places=3)

    def test_auto_grade_prediction_supports_plural_rbis_internal_name(self):
        pred_id = database.log_prediction(
            {
                "game_date": "2026-03-26",
                "player_name": "Shohei Ohtani",
                "stat_type": "RBIs",
                "stat_internal": "rbis",
                "line": 0.5,
                "projection": 0.9,
                "pick": "MORE",
                "confidence": 0.59,
                "rating": "C",
            }
        )
        pred_row = database.get_ungraded_predictions("2026-03-26").iloc[0]
        player_stats_list = [
            {
                "player_name": "Shohei Ohtani",
                "player_type": "batter",
                "rbi": 2,
                "hits": 1,
                "runs": 1,
                "walks": 1,
                "hbp": 0,
                "stolen_bases": 0,
                "strikeouts": 1,
                "at_bats": 4,
                "total_bases": 4,
                "hits_runs_rbis": 4,
                "fantasy_score": 14.0,
            }
        ]

        result = autograder.auto_grade_prediction(pred_row, player_stats_list)

        self.assertEqual(pred_id, 1)
        self.assertEqual(result, "W")
        row = self._fetchall(
            "SELECT actual_result, result FROM predictions WHERE id = ?",
            (pred_id,),
        )[0]
        self.assertEqual(row[0], 2.0)
        self.assertEqual(row[1], "W")

    def test_auto_grade_prediction_supports_hitter_strikeouts_alias(self):
        pred_id = database.log_prediction(
            {
                "game_date": "2026-03-26",
                "player_name": "Jung Lee",
                "stat_type": "Hitter Strikeouts",
                "stat_internal": "hitter_strikeouts",
                "line": 0.5,
                "projection": 0.94,
                "pick": "MORE",
                "confidence": 0.52,
                "rating": "D",
            }
        )
        pred_row = database.get_ungraded_predictions("2026-03-26").iloc[0]
        player_stats_list = [
            {
                "player_name": "Jung Lee",
                "player_type": "batter",
                "strikeouts": 1,
                "hits": 0,
                "runs": 0,
                "walks": 0,
                "hbp": 0,
                "stolen_bases": 0,
                "at_bats": 4,
                "total_bases": 0,
                "hits_runs_rbis": 0,
                "fantasy_score": 0.0,
            }
        ]

        result = autograder.auto_grade_prediction(pred_row, player_stats_list)

        self.assertEqual(pred_id, 1)
        self.assertEqual(result, "W")

    def test_repair_preopening_tracking_rows_removes_future_duplicate_legacy_rows(self):
        database.save_projected_stats([
            {
                "game_date": "2026-03-22",
                "player_name": "Max Fried",
                "team": "NYY",
                "stat_type": "pitcher_strikeouts",
                "projected_value": 4.6,
                "line": 5.0,
                "pick": "LESS",
                "confidence": 0.58,
                "rating": "C",
            },
            {
                "game_date": "2026-03-27",
                "player_name": "Max Fried",
                "team": "NYY",
                "stat_type": "pitcher_strikeouts",
                "projected_value": 5.0,
                "line": 5.0,
                "pick": "LESS",
                "confidence": 0.61,
                "rating": "B",
            },
        ])
        board_logger.log_board_snapshot([
            {
                "game_date": "2026-03-22",
                "player_name": "Max Fried",
                "team": "NYY",
                "stat_internal": "pitcher_strikeouts",
                "stat_type": "Pitcher Strikeouts",
                "line": 5.0,
                "pick": "LESS",
                "confidence": 0.58,
                "rating": "C",
                "projection": 4.6,
                "line_type": "standard",
            },
            {
                "game_date": "2026-03-27",
                "player_name": "Max Fried",
                "team": "NYY",
                "stat_internal": "pitcher_strikeouts",
                "stat_type": "Pitcher Strikeouts",
                "line": 5.0,
                "pick": "LESS",
                "confidence": 0.61,
                "rating": "B",
                "projection": 5.0,
                "line_type": "standard",
            },
        ])

        repair = autograder._repair_preopening_tracking_rows("2026-03-22")

        self.assertEqual(repair["projected_stats_removed"], 1)
        self.assertEqual(repair["board_rows_removed"], 1)
        projected_rows = self._fetchall(
            """
            SELECT game_date
            FROM projected_stats
            WHERE player_name = ? AND stat_type = ?
            ORDER BY game_date
            """,
            ("Max Fried", "pitcher_strikeouts"),
        )
        board_rows = self._fetchall(
            """
            SELECT date
            FROM daily_board
            WHERE player_name = ? AND prop_type = ?
            ORDER BY date
            """,
            ("Max Fried", "pitcher_strikeouts"),
        )
        self.assertEqual(projected_rows, [("2026-03-27",)])
        self.assertEqual(board_rows, [("2026-03-27",)])


if __name__ == "__main__":
    unittest.main()
