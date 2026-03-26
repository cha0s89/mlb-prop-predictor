import json
import tempfile
import unittest
from pathlib import Path

from src import autolearn


class AutolearnWeightsTests(unittest.TestCase):
    def setUp(self):
        self.original_weights_dir = autolearn.WEIGHTS_DIR
        self.original_current_path = autolearn.CURRENT_WEIGHTS_PATH
        self.original_history_path = autolearn.WEIGHT_HISTORY_PATH
        self.original_runtime_override_path = autolearn.RUNTIME_OVERRIDE_PATH

    def tearDown(self):
        autolearn.WEIGHTS_DIR = self.original_weights_dir
        autolearn.CURRENT_WEIGHTS_PATH = self.original_current_path
        autolearn.WEIGHT_HISTORY_PATH = self.original_history_path
        autolearn.RUNTIME_OVERRIDE_PATH = self.original_runtime_override_path

    def _point_paths(self, tmpdir: str) -> None:
        weights_dir = Path(tmpdir)
        autolearn.WEIGHTS_DIR = weights_dir
        autolearn.CURRENT_WEIGHTS_PATH = weights_dir / "current.json"
        autolearn.WEIGHT_HISTORY_PATH = weights_dir / "weight_history.json"
        autolearn.RUNTIME_OVERRIDE_PATH = weights_dir / "runtime_override.json"

    def test_corrupt_current_recovers_from_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._point_paths(tmpdir)
            snapshot = autolearn.get_baseline_weights()
            snapshot["version"] = "v027"
            snapshot["description"] = "good snapshot"
            snapshot_path = autolearn.WEIGHTS_DIR / "v027_good_snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            autolearn.CURRENT_WEIGHTS_PATH.write_text("{", encoding="utf-8")

            weights = autolearn.load_current_weights()
            repaired = json.loads(autolearn.CURRENT_WEIGHTS_PATH.read_text(encoding="utf-8"))

            self.assertEqual(weights["version"], "v027")
            self.assertEqual(repaired["version"], "v027")
            self.assertFalse((autolearn.WEIGHTS_DIR / "v001_baseline_weights_initialized.json").exists())

    def test_missing_current_initializes_baseline_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._point_paths(tmpdir)

            weights = autolearn.load_current_weights()

            self.assertTrue(autolearn.CURRENT_WEIGHTS_PATH.exists())
            saved = json.loads(autolearn.CURRENT_WEIGHTS_PATH.read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], "v001")
            self.assertEqual(weights["version"], "v001")


if __name__ == "__main__":
    unittest.main()
