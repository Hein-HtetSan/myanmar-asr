import csv
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import wave

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/data/build_combined_catalog.py'
spec = importlib.util.spec_from_file_location('combined_catalog', SCRIPT)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def row(name, split, text, digest):
    return dict(sample_id=name, split=split, text=text, sha256=digest,
                has_transcript=int(bool(text)), included=1, exclusion_reason='')


class CatalogCurationTest(unittest.TestCase):
    def test_duplicate_audio_prefers_evaluation_and_labeled_records(self):
        rows = [row('train', 'train', 'same words', 'one'),
                row('test', 'test_asr', 'same words', 'one'),
                row('unlabeled', 'unlabeled', '', 'one')]
        catalog.curate(rows)
        self.assertEqual([r['sample_id'] for r in rows if r['included']], ['test'])

    def test_conflicting_transcripts_quarantine_every_copy(self):
        rows = [row('a', 'train', 'first label', 'one'),
                row('b', 'test_asr', 'different label', 'one')]
        catalog.curate(rows)
        self.assertTrue(all(r['exclusion_reason'] == 'identical_audio_conflicting_transcripts'
                            for r in rows))

    def test_text_overlap_excludes_train_without_merging_distinct_audio(self):
        rows = [row('a', 'train', 'စမ်း သပ်', 'one'),
                row('b', 'validation', 'စမ်းသပ်', 'two'),
                row('c', 'train', 'other sentence', 'three')]
        catalog.curate(rows)
        self.assertEqual([r['sample_id'] for r in rows if r['included']], ['b', 'c'])
        self.assertEqual(rows[0]['exclusion_reason'], 'train_eval_normalized_text_overlap')

    def test_original_manifest_is_not_mutated_and_audio_resolves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'asr_work').mkdir()
            audio = root / 'sample.wav'
            with wave.open(str(audio), 'wb') as output:
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(struct.pack('<h', 10) * 1600)
            fields = ['sample_id','audio','text','source','speaker_group','record_group']
            for split in catalog.SPLITS:
                with (root/'asr_work'/f'{split}.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    if split == 'train':
                        writer.writerow(dict(zip(fields, ['sample', 'sample.wav', 'test', 'asr', 'spk', 'rec'])))
            original = (root/'asr_work/train.csv').read_bytes()
            rows = list(catalog.local_rows(root))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]['duration_s'], .1)
            self.assertEqual(rows[0]['sample_rate_hz'], 16000)
            self.assertEqual(original, (root/'asr_work/train.csv').read_bytes())
            catalog_root = root/'catalog'
            catalog_root.mkdir()
            (catalog_root/'audio_data').symlink_to(root, target_is_directory=True)
            waveform, rate = catalog.read_audio(rows[0], catalog_root)
            self.assertEqual((len(waveform), rate), (1600, 16000))


if __name__ == '__main__':
    unittest.main()
