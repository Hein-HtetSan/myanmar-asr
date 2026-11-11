"""Public MLflow export and deploy bundle checks, without a live tracker."""
import json
from pathlib import Path
from types import SimpleNamespace as Object
import tempfile
import unittest

from scripts.deploy.export_mlflow_results import export_results, completed_runs
from scripts.deploy.build_research_site import build

RUN_ID = 'a' * 32


class FakeClient:
    def get_run(self, run_id):
        return Object(info=Object(status='FINISHED', start_time=100, end_time=200),
                      data=Object(tags={'mlflow.runName': '<script>alert(1)</script>',
                                        'private_token': 'SECRET'},
                                  metrics={'eval_cer_no_space': 12.5, 'loss': float('nan')},
                                  params={'password': 'SECRET'}))

    def list_artifacts(self, run_id, path):
        return {'': [Object(path='configuration', is_dir=True),
                     Object(path='best_model', is_dir=True)],
                'configuration': [Object(path='configuration/run.json', is_dir=False, file_size=100)],
                'best_model': [Object(path='best_model/model.safetensors', is_dir=False, file_size=1000000)]}[path]

    def download_artifacts(self, run_id, artifact, directory):
        path = Path(directory) / 'run.json'
        path.write_text(json.dumps({'model_id': 'openai/whisper-small', 'family': 'whisper',
                                    'tracking_uri': 'https://user:SECRET@private.example',
                                    'training_arguments': {'learning_rate': 1e-5, 'output_dir': '/private/path',
                                                           'hub_token': 'SECRET'}}))
        return str(path)

    def get_metric_history(self, run_id, key):
        return [Object(step=2, timestamp=200, value=12.5), Object(step=1, timestamp=100, value=14)]


class ResearchResultsTest(unittest.TestCase):
    def test_export_has_full_inventory_and_histories_without_private_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            export_results(FakeClient(), [RUN_ID, RUN_ID], path, 'commit')
            text = path.read_text()
            data = json.loads(text)
            self.assertEqual(len(data['runs']), 1)
            run = data['runs'][0]
            self.assertEqual(len(run['artifacts']), 2)
            self.assertEqual([p['step'] for p in run['metric_history']['eval_cer_no_space']], [1, 2])
            self.assertIsNone(run['latest_metrics']['loss'])
            self.assertEqual(run['configuration']['training_arguments'], {'learning_rate': 1e-5})
            for private in ('SECRET', 'private.example', '/private/path', 'hub_token'):
                self.assertNotIn(private, text)

    def test_failed_export_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text('previous snapshot')
            client = FakeClient()
            def fail(*args):
                raise ConnectionError('Tracker unavailable')
            client.get_metric_history = fail
            with self.assertRaises(ConnectionError):
                export_results(client, [RUN_ID], path)
            self.assertEqual(path.read_text(), 'previous snapshot')

    def test_no_unreviewed_or_invalid_runs_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            for ids in ([], ['invalid-run']):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    export_results(FakeClient(), ids, Path(directory) / 'results.json')

    def test_reads_numeric_result_json_from_mlflow_artifacts(self):
        class ResultClient(FakeClient):
            def list_artifacts(self, run_id, path):
                files = super().list_artifacts(run_id, path)
                if path == '':
                    files.append(Object(path='metrics/test_metrics.json', is_dir=False, file_size=100))
                return files

            def download_artifacts(self, run_id, artifact, directory):
                if artifact.startswith('metrics/'):
                    path = Path(directory) / 'test_metrics.json'
                    path.write_text(json.dumps({'test_asr': {'wer': 12.5, 'clips': 193},
                                                'private_path': '/private/path', 'transcript': 'private speech'}))
                    return str(path)
                return super().download_artifacts(run_id, artifact, directory)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            export_results(ResultClient(), [RUN_ID], output, completed_only=True)
            data = json.loads(output.read_text())
            result = data['runs'][0]['result_artifacts']['metrics/test_metrics.json']
            self.assertEqual(result, {'test_asr': {'wer': 12.5, 'clips': 193}})

    def test_completed_full_run_waits_for_test_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            export_results(FakeClient(), [RUN_ID], output, completed_only=True)
            self.assertEqual(json.loads(output.read_text())['runs'], [])

    def test_quick_run_publishes_when_finished_without_test_results(self):
        class QuickClient(FakeClient):
            def download_artifacts(self, run_id, artifact, directory):
                path = Path(super().download_artifacts(run_id, artifact, directory))
                data = json.loads(path.read_text())
                data['quick_run'] = True
                path.write_text(json.dumps(data))
                return str(path)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            export_results(QuickClient(), [RUN_ID], output, completed_only=True)
            self.assertEqual(len(json.loads(output.read_text())['runs']), 1)

    def test_running_run_is_not_published(self):
        class RunningClient(FakeClient):
            def get_run(self, run_id):
                run = super().get_run(run_id)
                run.info.status = 'RUNNING'
                return run
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            export_results(RunningClient(), [RUN_ID], output, completed_only=True)
            self.assertEqual(json.loads(output.read_text())['runs'], [])

    def test_completed_run_discovery_reads_all_pages(self):
        class Page(list):
            pass
        class Client:
            def get_experiment_by_name(self, name):
                return Object(experiment_id='1')

            def search_runs(self, ids, filter_string, max_results, page_token, order_by):
                self_filter = "attributes.status = 'FINISHED'"
                assert filter_string == self_filter
                page = Page([Object(info=Object(run_id='b'*32 if page_token else RUN_ID))])
                page.token = None if page_token else 'next'
                return page
        self.assertEqual(completed_runs(Client(), 'experiment'), [RUN_ID, 'b'*32])

    def test_static_bundle_contains_only_public_site_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build(output)
            self.assertEqual({p.name for p in (output / 'static').iterdir()},
                             {'index.html', 'app.js', 'styles.css', 'results.json'})
            self.assertEqual(json.loads((output / 'config.json').read_text())['version'], 3)


if __name__ == '__main__':
    unittest.main()
