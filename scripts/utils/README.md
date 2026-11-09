# Shared training and remote data utilities

These helpers support [training](../training/README.md) and remote Linux GPU experiments. Most paths target `/workspace`, rather than the local notebook. Run commands from the repository root and review configuration constants before execution.

- `mlflow_callback.py`: shared Hugging Face Trainer callbacks and helpers for metrics, system statistics, final results, and model artifacts. Trainers add this directory to their Python import path. Uses MLflow and optional system-monitoring dependencies; its default tracking URI is `http://localhost:5050`, normally reached through the experiment's SSH tunnel. Import the helpers from training code rather than running this file as a job.
- `remote_setup.py`: installs remote Python packages, authenticates to Hugging Face using `HF_TOKEN` when supplied, checks GPU/package information, and checks the MLflow endpoint. It changes the remote environment and uses broad dependency versions; it is not a reproducible environment lock.
- `build_dataset_full.py`: downloads configured FLEURS, OpenSLR-80, and YODAS sources, normalizes, filters, adds 1.1x training speed variants, and writes `/workspace/data/myanmar_asr`. Requires datasets, NumPy, librosa, and SoundFile. Failed source downloads can be skipped, so verify the printed source/split counts. Existing output is moved to `_old`; an existing `_old` backup is deleted.
- `rebuild_dataset_remote.py`: older reconstruction workflow that tries Common Voice, FLEURS, and OpenSLR. It may reuse an existing saved dataset. Its OpenSLR fallback loads FLEURS, which can mislabel or duplicate data; review that behavior before using it for a study. Prefer inspecting the explicit source handling in `build_dataset_full.py` for new remote preparation.
- `deep_clean.py`: applies stricter duration, text, silence, and duplication checks, merges and recreates splits, and replaces `/workspace/data/myanmar_asr` with cleaned data. It manages a backup, but may delete a previous backup. Re-splitting changes the evaluation population; preserve the original dataset and split definitions before using it for a new experiment.

For a remote machine with dependencies already installed and configured sources reviewed:

```sh
python scripts/utils/build_dataset_full.py
```

This performs real downloads and writes data. It is a separate pipeline from [local dataset preparation](../data/README.md), and does not produce the CSV manifests required by [the Whisper Small notebook](../../notebook/whisper-small/README.md). Older Hub loading code and current dataset availability are not verified here.
