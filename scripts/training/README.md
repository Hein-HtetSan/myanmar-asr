# Model training

Run commands from the repository root. These are GPU training experiments with defaults for a Linux `/workspace` environment. For the separate Windows notebook workflow, see [Whisper Small](../../notebook/whisper-small/README.md).

The Hugging Face trainers require PyTorch, Transformers, datasets, Accelerate, evaluate/jiwer, and their tracking dependencies. They load a saved `DatasetDict` with `train`, `validation`, and `test` splits and `audio`/`sentence` columns. See [dataset preparation](../data/README.md). Canary instead requires NeMo/PyTorch Lightning and audio manifests. Use the [K80 runbook](../../docs/CentOS7_K80_Compatibility_Decision.md) for that older GPU; the large-model defaults are not K80 settings.

## Available trainers

- `train_whisper_turbo.py`: Whisper Large v3 Turbo by default; supports another Whisper checkpoint through `--model_id`. Supports `--no_push` to disable Hub publishing.
- `train_best.py`: an alternative Turbo experiment with encoder freezing, label smoothing, cosine scheduling, and MLflow logging.
- `train_dolphin.py`: fine-tunes `openai/whisper-large-v2`; “Dolphin” is this project's experiment name.
- `train_seamless.py`: fine-tunes `facebook/seamless-m4t-v2-large` for Myanmar ASR.
- `train_canary.py`: experimental NVIDIA Canary-1B training with NeMo, using train/validation/test JSONL manifests. Verify model language support and tokenizer setup before treating this as a validated Myanmar recipe.

Inspect each trainer's `--help` in its configured environment for arguments. The Hugging Face trainers accept `--model_id`, `--dataset_path`, `--output_dir`, `--batch_size`, `--grad_accum`, `--lr`, and `--epochs`. Canary accepts manifest paths instead of `--dataset_path`.

## Example: Whisper Small using a prepared dataset

```sh
python scripts/training/train_whisper_turbo.py --model_id openai/whisper-small --dataset_path combined/myanmar_asr_50h_clean --output_dir models/whisper-small --batch_size 1 --grad_accum 16 --no_push
```

This launches training; it is not a dry run. For a short check, the Turbo trainer also accepts `--max_steps`. Model and processor output is saved under `<output_dir>/final/`, with results in `whisper_turbo_results.json` under the output directory.

## Paths, tracking, and evaluation

Default datasets live at `/workspace/data/myanmar_asr`; model outputs live under `/workspace/models/`. Some trainers also use hardcoded `/workspace/logs/` and `/workspace/results/` paths even when `--output_dir` changes. Review those paths for local or Windows runs. The trainers use shared helpers from [utils](../utils/README.md), and several force MLflow to `http://localhost:5050`; remote runs may require the project's SSH tunnels and artifact-store configuration.

These trainers compute WER/CER using their own scoring code. Match the text normalization, dataset splits, and generation settings before comparing with the notebook's whitespace-normalized CER. See [evaluation](../evaluation/README.md) for separate test evaluation. Checkpoint/resume behavior differs from the notebook; do not assume these scripts inherit its corrupted-checkpoint fallback.
