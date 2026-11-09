"""Exercise the notebook's checkpoint selection without loading Whisper or a GPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class CheckpointResumeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        notebook = Path(__file__).resolve().parents[1] / "notebook/whisper-small/train.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        source = next("".join(c["source"]) for c in cells
                      if "def latest_complete_checkpoint():" in "".join(c["source"]))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "latest_complete_checkpoint")
        namespace = {"RUN_DIR": self.root, "json": json, "zipfile": zipfile}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
        self.select = namespace["latest_complete_checkpoint"]

    def checkpoint(self, step):
        path = self.root / f"checkpoint-{step}"
        path.mkdir()
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "pytorch_model.bin"):
            with zipfile.ZipFile(path / name, "w") as archive:
                archive.writestr("archive/data.pkl", b"fixture")
        return path

    def test_empty_run_starts_fresh(self):
        self.assertIsNone(self.select())

    def test_newest_intact_archive_selected(self):
        self.checkpoint(1650)
        newest = self.checkpoint(1700)
        self.assertEqual(self.select(), newest)

    def test_crc_damage_falls_back(self):
        older = self.checkpoint(1650)
        newest = self.checkpoint(1700)
        path = newest / "optimizer.pt"
        path.write_bytes(path.read_bytes().replace(b"fixture", b"damaged"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.select(), older)

    def test_truncated_resume_archives_fall_back_without_deletion(self):
        older = self.checkpoint(1650)
        newest = self.checkpoint(1700)
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "pytorch_model.bin", "scaler.pt"):
            with self.subTest(name=name):
                path = newest / name
                original = path.read_bytes() if path.exists() else None
                path.write_bytes(b"PK\x03\x04incomplete")
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(self.select(), older)
                self.assertIn(name, output.getvalue())
                self.assertTrue(path.exists())
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_no_usable_checkpoint_stops(self):
        path = self.checkpoint(1700)
        (path / "optimizer.pt").write_bytes(b"truncated")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "No checkpoint"):
            self.select()


if __name__ == "__main__":
    unittest.main()
