import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/download_commonvoice.py"
spec = importlib.util.spec_from_file_location("yodas_download", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class YodasArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "00000000.tar.gz"
        with tarfile.open(self.archive, "w:gz") as archive:
            content = b"audio fixture"
            info = tarfile.TarInfo("../../utterance.wav")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        self.text = self.root / "00000000.txt"
        self.text.write_text("utterance မင်္ဂလာပါ သူငယ်ချင်း\n", encoding="utf-8")

    def test_pairs_text_and_embeds_audio_without_extraction(self):
        examples = list(module.archive_examples(self.archive, self.text))
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["sentence"], "မင်္ဂလာပါ သူငယ်ချင်း")
        self.assertEqual(examples[0]["audio"], {"path": None, "bytes": b"audio fixture"})
        self.assertEqual(examples[0]["source"], "yodas_my")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["00000000.tar.gz", "00000000.txt"])

    def test_missing_audio_fails(self):
        self.text.write_text("missing transcript\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no audio"):
            list(module.archive_examples(self.archive, self.text))

    def test_duplicate_transcript_fails(self):
        self.text.write_text("utterance first\nutterance second\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            list(module.archive_examples(self.archive, self.text))


if __name__ == "__main__":
    unittest.main()
