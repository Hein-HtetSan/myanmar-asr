"""Check the combined-catalog selection without downloading weights or audio."""
import ast
import csv
import io
import json
from collections import Counter
import contextlib
from pathlib import Path
import random
import tempfile
import unittest
import unicodedata

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebook/whisper-small/train.ipynb'


class WhisperSmallNotebookTest(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.code = [''.join(c['source']) for c in self.notebook['cells'] if c['cell_type'] == 'code']

    def test_clean_code_compiles(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile('\n'.join('pass' if line.lstrip().startswith('%') else line
                                  for line in ''.join(cell['source']).splitlines()), str(NOTEBOOK), 'exec')
        code = '\n'.join(self.code)
        self.assertIn('QUICK_RUN = False', code)
        self.assertNotIn('CACHE_DIR', code)
        self.assertNotIn('cache_path', code)
        self.assertIn('include_labels=False', code)

    def test_resume_configuration_reuses_identical_and_rejects_changed_runs(self):
        source = next(c for c in self.code if 'config_path = RUN_DIR /' in c)
        guard = source[source.index('config_path = RUN_DIR /'):]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = dict(RUN_DIR=root, json=json, configuration={'manifest_sha256': {'train': 'abc'}})
            exec(guard, namespace)
            exec(guard, namespace)
            namespace['configuration'] = {'manifest_sha256': {'train': 'changed'}}
            with self.assertRaisesRegex(ValueError, 'Settings/data changed'):
                exec(guard, namespace)
            (root / 'run_config.json').unlink()
            (root / 'checkpoint-50').mkdir()
            with self.assertRaisesRegex(ValueError, 'no configuration'):
                exec(guard, namespace)

    def test_training_and_prediction_share_tracked_run_contexts(self):
        guarded_calls = []
        def visit(node, tracked=False):
            if isinstance(node, ast.With):
                tracked = tracked or any(isinstance(item.context_expr, ast.Call)
                                         and isinstance(item.context_expr.func, ast.Name)
                                         and item.context_expr.func.id == 'tracked_run'
                                         for item in node.items)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'trainer'
                    and node.func.attr in ('train', 'evaluate', 'predict')):
                self.assertTrue(tracked, f'{node.func.attr} requires an active MLflow run')
                guarded_calls.append(node.func.attr)
            for child in ast.iter_child_nodes(node):
                visit(child, tracked)
        for cell in self.code:
            visit(ast.parse('\n'.join('pass' if line.lstrip().startswith('%') else line
                                      for line in cell.splitlines())))
        self.assertEqual(set(guarded_calls), {'train', 'evaluate', 'predict'})
        self.assertIn('MLflowMetricsCallback()', '\n'.join(self.code))

    def selection(self, quick=False, overlap=False):
        source = next(c for c in self.code if 'def load_split(name):' in c)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ('train', 'validation', 'test_asr', 'test_user_study', 'test_ucsy',
                     'test_fleurs', 'test_yodas_my')
            for name in names:
                with (root / f'{name}_whisper.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['sample_id', 'text', 'source', 'duration_s', 'sha256'])
                    writer.writeheader()
                    count = 300 if name == 'train' else 80 if name == 'validation' else 1
                    for i in range(count):
                        writer.writerow(dict(sample_id=f'{name}:{i}', text=f'{name}:{i}',
                                             source=('asr', 'ucsy', 'fleurs', 'yodas_my')[i % 4],
                                             duration_s='1', sha256=f'{name}:{i}'))
            def load_rows(path):
                with path.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                if overlap and path.name == 'test_fleurs_whisper.csv':
                    rows[0]['text'] = ' t r a i n : 0 '
                return rows
            namespace = dict(CATALOG_DIR=root, QUICK_RUN=quick, load_rows=load_rows,
                             text_key=lambda s: ''.join(unicodedata.normalize('NFC', s).split()),
                             Counter=Counter, random=random)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            return namespace

    def test_full_keeps_every_training_source_and_five_tests(self):
        data = self.selection()
        self.assertEqual(len(data['train_rows']), 300)
        self.assertEqual(len(data['validation_rows']), 80)
        self.assertEqual(len(data['all_test_splits']), 5)
        self.assertEqual({row['source'] for row in data['train_rows']},
                         {'asr', 'ucsy', 'fleurs', 'yodas_my'})

    def test_quick_selection_is_bounded_and_repeatable(self):
        first, second = self.selection(quick=True), self.selection(quick=True)
        self.assertEqual(len(first['train_rows']), 256)
        self.assertEqual(len(first['validation_rows']), 64)
        self.assertEqual(first['train_rows'], second['train_rows'])

    def test_normalized_train_test_text_overlap_stops(self):
        with self.assertRaisesRegex(AssertionError, 'text crosses train and test_fleurs'):
            self.selection(overlap=True)


if __name__ == '__main__':
    unittest.main()
