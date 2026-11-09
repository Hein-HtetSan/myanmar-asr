# Whisper Large v3 Turbo training notebook

[train.ipynb](train.ipynb) fine-tunes `openai/whisper-large-v3-turbo` from the combined catalog and logs to Docker MLflow. Follow [the shared setup and training guide](../README.md).

The matching script is [scripts/training/train_whisper_turbo.py](../../scripts/training/train_whisper_turbo.py). The notebook adapts the experiment for reference-backed audio, checks labels instead of truncating them, selects checkpoints using whitespace-free CER, and evaluates each test source separately. It uses Transformers 4.57.1 rather than the historical scripts' evolving dependencies.

Start with `QUICK_RUN=True`. Full runs target CUDA; defaults freeze the acoustic encoder and use batch size one. Output is `notebook/runs/whisper-turbo/quick/baseline/` or `notebook/runs/whisper-turbo/full/baseline/`, including `best_model/`, checkpoints, predictions, configuration, and the MLflow run ID. Change `RUN_LABEL` for another experiment.

Quick mode still loads the full model. The pre-download free-space floor is 30 GiB, followed by a parameter-based checkpoint estimate. These are storage checks, not a guarantee that the model fits your GPU. See the shared guide for the actual validation performed and remaining hardware limits.
