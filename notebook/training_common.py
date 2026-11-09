"""Shared audio, scoring, checkpoint and MLflow helpers for the three notebooks."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import numbers
from pathlib import Path
import platform
import random
import unicodedata
import zipfile

import jiwer
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from safetensors import SafetensorError
from transformers import Seq2SeqTrainer, TrainerCallback

from scripts.data.build_combined_catalog import read_audio


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, limit=None, seed=42):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty catalog view: {path}')
    for row in rows:
        if (row.get('included') != '1' or row.get('has_transcript') != '1'
                or not row.get('text', '').strip() or row['source'] == 'voa_myanmar'
                or not 0 < float(row['duration_s']) <= 30):
            raise ValueError(f"Not a labeled <=30-second candidate: {row.get('sample_id')}")
    if limit is not None:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def label_ids(processor, text, family, decoder_start_token_id):
    tokenizer = processor.tokenizer
    ids = (tokenizer(text_target=text).input_ids if family == 'seamless'
           else tokenizer(text).input_ids)
    # Whisper uses <startoftranscript>; Seamless target text starts with EOS.
    # The decoder adds this token itself, so labels must not contain it twice.
    if ids and ids[0] == decoder_start_token_id:
        ids = ids[1:]
    return ids


def filter_labels(rows, processor, family, decoder_start_token_id, max_length):
    accepted, rejected = [], []
    for row in rows:
        ids = label_ids(processor, row['text'], family, decoder_start_token_id)
        if ids and len(ids) <= max_length - 1:
            accepted.append({**row, 'labels': ids})
        else:
            rejected.append({'sample_id': row['sample_id'], 'label_tokens': len(ids),
                             'reason': 'label_length', 'text': row['text']})
    if not accepted:
        raise ValueError('No labels fit the configured generation/model length')
    return accepted, rejected


class CatalogSpeechDataset(Dataset):
    def __init__(self, rows, catalog_root, feature_extractor, include_labels=True):
        self.rows = rows
        self.root = Path(catalog_root)
        self.extractor = feature_extractor
        self.include_labels = include_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import librosa
        row = self.rows[index]
        audio, rate = read_audio(row, self.root)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        if not np.isfinite(audio).all() or not 0 < len(audio) <= 16000 * 30:
            raise ValueError(f"Invalid waveform or >30 seconds: {row['sample_id']}")
        features = self.extractor(audio, sampling_rate=16000, return_tensors='np',
                                  return_attention_mask=True)
        result = {'input_features': features.input_features[0],
                  'attention_mask': features.attention_mask[0]}
        if self.include_labels:
            result['labels'] = row['labels']
        return result


class SpeechCollator:
    def __init__(self, family, pad_token_id, decoder_start_token_id):
        self.family = family
        self.pad = pad_token_id
        self.start = decoder_start_token_id

    def __call__(self, features):
        values = [torch.as_tensor(f['input_features'], dtype=torch.float32) for f in features]
        masks = [torch.as_tensor(f['attention_mask'], dtype=torch.long) for f in features]
        batch = {
            'input_features': pad_sequence(values, batch_first=True) if self.family == 'seamless'
                              else torch.stack(values),
            'attention_mask': pad_sequence(masks, batch_first=True),
        }
        if 'labels' in features[0]:
            labels = pad_sequence([torch.tensor(f['labels'], dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
            decoder = torch.full_like(labels, self.pad)
            decoder[:, 0] = self.start
            decoder[:, 1:] = labels[:, :-1]
            decoder.masked_fill_(decoder == -100, self.pad)
            batch.update(labels=labels, decoder_input_ids=decoder)
        return batch


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def score_texts(references, hypotheses):
    references = [normalize_text(text) for text in references]
    hypotheses = [normalize_text(text) for text in hypotheses]
    return {
        'wer': 100 * jiwer.wer(references, hypotheses),
        'cer_raw': 100 * jiwer.cer(references, hypotheses),
        'cer_no_space': 100 * jiwer.cer([''.join(s.split()) for s in references],
                                      [''.join(s.split()) for s in hypotheses]),
    }


def decode_predictions(processor, predictions):
    ids = predictions[0] if isinstance(predictions, tuple) else predictions
    ids = np.where(ids < 0, processor.tokenizer.pad_token_id, ids)
    return processor.batch_decode(ids, skip_special_tokens=True)


def metric_function(processor):
    def compute(result):
        references = decode_predictions(processor, result.label_ids)
        hypotheses = decode_predictions(processor, result.predictions)
        return score_texts(references, hypotheses)
    return compute


class MLflowMetricsCallback(TrainerCallback):
    """The notebook owns the run; a tracking failure stops training visibly."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        import mlflow
        if mlflow.active_run() is None:
            raise RuntimeError('Start the MLflow run before training or evaluation')
        metrics = {key: float(value) for key, value in (logs or {}).items()
                   if isinstance(value, numbers.Real) and math.isfinite(value)}
        if torch.cuda.is_available():
            metrics['gpu_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)


class SpeechTrainer(Seq2SeqTrainer):
    """Supply the ASR language during scheduled and explicit evaluation alike."""
    def __init__(self, *args, generation_kwargs=None, **kwargs):
        self.speech_generation_kwargs = generation_kwargs or {}
        super().__init__(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **{**self.speech_generation_kwargs, **kwargs})

    def predict(self, *args, **kwargs):
        return super().predict(*args, **{**self.speech_generation_kwargs, **kwargs})


def latest_checkpoint(run_dir):
    checkpoints = sorted((p for p in Path(run_dir).glob('checkpoint-*')
                          if p.is_dir() and p.name[11:].isdigit()),
                         key=lambda p: int(p.name[11:]), reverse=True)
    for path in checkpoints:
        try:
            state = json.loads((path/'trainer_state.json').read_text())
            if state.get('global_step') != int(path.name[11:]):
                raise ValueError('trainer step differs from folder name')
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                with zipfile.ZipFile(path/name) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f'CRC failure: {name}')
            if (path/'scaler.pt').exists():
                with zipfile.ZipFile(path/'scaler.pt') as archive:
                    if archive.testzip() is not None:
                        raise ValueError('CRC failure: scaler.pt')
            weights = [path/'model.safetensors']
            index = path/'model.safetensors.index.json'
            if index.exists():
                mapping = json.loads(index.read_text())['weight_map']
                weights = [path/name for name in sorted(set(mapping.values()))]
            from safetensors import safe_open
            for weight in weights:
                if not weight.resolve().is_relative_to(path.resolve()):
                    raise ValueError('weight path escapes checkpoint')
                with safe_open(weight, framework='pt', device='cpu'):
                    pass
            return path
        except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, SafetensorError) as error:
            print(f'Skipping {path.name}: {error}')
    if checkpoints:
        raise RuntimeError('No usable checkpoint found; restore a backup or choose a new RUN_LABEL')
    return None


@contextmanager
def tracked_run(run_dir, tracking_uri, experiment_name, run_name, configuration):
    import mlflow
    run_dir = Path(run_dir)
    config_path = run_dir/'run_config.json'
    run_path = run_dir/'mlflow_run_id.txt'
    if mlflow.active_run() is not None:
        raise RuntimeError('Another MLflow run is active; finish it before this notebook run')
    canonical = json.dumps(configuration, sort_keys=True)
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Run settings/data changed. Use a new RUN_LABEL or restore the saved settings.')
    if run_path.exists() and not config_path.exists():
        raise ValueError('MLflow run ID exists without its configuration; use a new RUN_LABEL')
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    run_id = run_path.read_text().strip() if run_path.exists() else None
    run_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_id=run_id, experiment_id=experiment.experiment_id,
                          run_name=run_name) as run:
        run_path.write_text(run.info.run_id)
        config_path.write_text(canonical+'\n')
        mlflow.log_params({key: value for key, value in configuration.items()
                           if isinstance(value, (str, int, float, bool))})
        # Upload before the first optimizer step to verify artifact connectivity.
        mlflow.log_dict(configuration, 'configuration/run.json')
        mlflow.set_tags({'model_family': configuration['family'],
                         'quick_run': str(configuration['quick_run']).lower()})
        mlflow.log_dict({'torch': torch.__version__, 'python': platform.python_version(),
                         'packages': {name: importlib.metadata.version(name) for name in
                                      ('transformers', 'datasets', 'accelerate', 'sentencepiece', 'mlflow-skinny')},
                         'cuda_available': torch.cuda.is_available(),
                         'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps',
                         'recorded_at': datetime.now(timezone.utc).isoformat()}, 'configuration/environment.json')
        yield run


def save_predictions(path, rows, hypotheses):
    if len(rows) != len(hypotheses):
        raise ValueError('Prediction and reference counts differ')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'source', 'reference', 'prediction'])
        writer.writeheader()
        writer.writerows({'sample_id': row['sample_id'], 'source': row['source'],
                          'reference': row['text'], 'prediction': hypothesis}
                         for row, hypothesis in zip(rows, hypotheses))
