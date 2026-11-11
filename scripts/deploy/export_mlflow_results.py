"""Export explicitly selected MLflow runs to a public research summary."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile

CONFIG_KEYS = ('model_id', 'model_revision', 'resolved_revision', 'family', 'quick_run',
               'seed', 'precision', 'manifest_sha256', 'selected', 'rejected_labels',
               'freeze_encoder', 'train_clips', 'validation_clips', 'train_hours', 'validation_hours',
               'rejected_train_labels', 'rejected_validation_labels', 'selection_sha256',
               'catalog_summary_sha256', 'learning_rate', 'epochs', 'batch_size',
               'gradient_accumulation', 'max_steps', 'eval_save_steps', 'generation_max_length',
               'warmup_ratio', 'weight_decay', 'label_smoothing', 'metric_for_best_model', 'scoring')
TRAINING_KEYS = ('learning_rate', 'num_train_epochs', 'max_steps', 'seed',
                 'per_device_train_batch_size', 'per_device_eval_batch_size',
                 'gradient_accumulation_steps', 'warmup_steps', 'warmup_ratio',
                 'fp16', 'bf16', 'gradient_checkpointing', 'metric_for_best_model',
                 'greater_is_better', 'generation_max_length', 'eval_steps', 'save_steps')


def finite(value):
    return float(value) if math.isfinite(value) else None


def artifact_index(client, run_id, path=''):
    files = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.extend(artifact_index(client, run_id, item.path))
        else:
            files.append({'path': item.path, 'size_bytes': item.file_size})
    return sorted(files, key=lambda item: item['path'])


def numeric_results(value):
    """Publish numeric aggregate results, excluding text and private paths."""
    if isinstance(value, dict):
        return {key: numeric_results(item) for key, item in value.items()
                if isinstance(item, (dict, int, float)) and not isinstance(item, bool)}
    return finite(value)


def completed_runs(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f'Unknown MLflow experiment: {experiment_name}')
    run_ids, token = [], None
    while True:
        page = client.search_runs([experiment.experiment_id],
                                  filter_string="attributes.status = 'FINISHED'",
                                  max_results=1000, page_token=token,
                                  order_by=['attributes.start_time ASC'])
        run_ids.extend(run.info.run_id for run in page)
        token = page.token
        if not token:
            return run_ids


def export_run(client, run_id):
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('Expected a 32-character MLflow run ID')
    run = client.get_run(run_id)
    artifacts = artifact_index(client, run_id)
    configuration = None
    if any(item['path'] == 'configuration/run.json' for item in artifacts):
        with tempfile.TemporaryDirectory() as directory:
            path = client.download_artifacts(run_id, 'configuration/run.json', directory)
            raw = json.loads(Path(path).read_text(encoding='utf-8'))
        configuration = {key: raw[key] for key in CONFIG_KEYS if key in raw}
        configuration['training_arguments'] = {
            key: raw.get('training_arguments', {}).get(key) for key in TRAINING_KEYS
            if key in raw.get('training_arguments', {})}
    result_artifacts = {}
    for item in artifacts:
        if item['path'].startswith('metrics/') and item['path'].endswith('.json'):
            with tempfile.TemporaryDirectory() as directory:
                path = client.download_artifacts(run_id, item['path'], directory)
                result_artifacts[item['path']] = numeric_results(json.loads(Path(path).read_text(encoding='utf-8')))
    histories = {}
    for key in sorted(run.data.metrics):
        histories[key] = [{'step': item.step, 'timestamp': item.timestamp, 'value': finite(item.value)}
                          for item in client.get_metric_history(run_id, key)]
        histories[key].sort(key=lambda item: (item['step'], item['timestamp']))
    return {'run_id': run_id, 'name': run.data.tags.get('mlflow.runName', run_id),
            'status': run.info.status, 'start_time': run.info.start_time,
            'end_time': run.info.end_time, 'configuration': configuration,
            'latest_metrics': {key: finite(value) for key, value in sorted(run.data.metrics.items())},
            'metric_history': histories, 'result_artifacts': result_artifacts, 'artifacts': artifacts}


def export_results(client, run_ids, output, commit=None, completed_only=False):
    if not run_ids and not completed_only:
        raise ValueError('Select runs using MLFLOW_RUN_IDS or an experiment name')
    # Collect everything first: a failed tracker request must not replace a good snapshot.
    runs = [export_run(client, rid) for rid in dict.fromkeys(run_ids)]
    if completed_only:
        # Notebook training and test evaluation use separate run contexts. Wait for full test results.
        runs = [run for run in runs if run['status'] == 'FINISHED'
                and run['configuration'] is not None
                and (run['configuration'].get('quick_run') is True
                     or 'metrics/test_metrics.json' in run['result_artifacts'])]
    snapshot = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
                'source_commit': commit, 'runs': runs,
                'artifact_policy': 'Inventory only. Raw artifacts, text, audio, logs and weights are not published.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-ids', default=os.environ.get('MLFLOW_RUN_IDS', ''))
    parser.add_argument('--experiment-name', default=os.environ.get('MLFLOW_EXPERIMENT_NAME', 'myanmar-asr-notebooks'))
    parser.add_argument('--output', type=Path, default=Path('web/results.json'))
    args = parser.parse_args()
    run_ids = [value.strip() for value in args.run_ids.split(',') if value.strip()]
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    if not run_ids:
        run_ids = completed_runs(client, args.experiment_name)
    export_results(client, run_ids, args.output, os.environ.get('GITHUB_SHA'), completed_only=True)
    print(f'Exported completed runs to {args.output}')


if __name__ == '__main__':
    main()
