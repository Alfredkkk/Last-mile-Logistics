"""Regression checks for interrupted atomic writes and advisory progress."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "training_persistence.py").is_file())
sys.path.insert(0, str(ROOT))
import training_persistence as persistence


class FileLockTests(unittest.TestCase):
    def test_transient_lock_retries_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "result.json"
            persistence.atomic_json(target, {"old": True})
            replace = persistence.os.replace
            calls = []

            def locked_once(source, destination):
                calls.append(source)
                if len(calls) == 1:
                    self.assertEqual(json.loads(target.read_text()), {"old": True})
                    raise PermissionError("locked")
                return replace(source, destination)

            with patch.object(persistence.os, "replace", side_effect=locked_once), patch.object(persistence.time, "sleep"):
                persistence.atomic_json(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"new": True})
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_persistent_lock_preserves_critical_file_and_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "latest.pt"
            target.write_bytes(b"previous checkpoint")
            with patch.object(persistence.os, "replace", side_effect=PermissionError("locked")) as replace, patch.object(persistence.time, "sleep"):
                with self.assertRaises(PermissionError):
                    persistence.atomic_write(target, lambda f: f.write(b"new checkpoint"))
            self.assertEqual(replace.call_count, 7)
            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_both_progress_files_are_nonfatal_and_recover(self):
        with tempfile.TemporaryDirectory() as folder:
            store = persistence.SweepStore(Path(folder) / "results.csv", [], {})
            progress = persistence.TrainingProgress(50, path=store.directory / "combo_progress.json",
                                                    enabled=False, callback=lambda s: store.status("training", training=s))
            with patch.object(persistence.os, "replace", side_effect=PermissionError("locked")), patch.object(persistence.time, "sleep"), self.assertLogs("training_persistence", level="WARNING") as logs:
                progress.report(5, "training", saved_update=5)
                progress.report(6, "training", saved_update=5)
            self.assertEqual(len(logs.output), 2)
            progress.report(7, "training", saved_update=5)
            self.assertEqual(json.loads(progress.path.read_text())["update"], 7)
            self.assertEqual(json.loads((store.directory / "progress.json").read_text())["training"]["update"], 7)

    def test_unrelated_error_is_not_retried(self):
        with patch.object(persistence.time, "sleep") as sleep:
            with self.assertRaises(FileNotFoundError):
                persistence._retry_file_operation(lambda: (_ for _ in ()).throw(FileNotFoundError()))
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
