# Whisper Small: full combined Myanmar ASR dataset

[train.ipynb](train.ipynb) fine-tunes `openai/whisper-small` using the combined audio-backed catalog at `combined/myanmar_asr_all/`. It uses local ASR, UCSY, FLEURS, and YODAS training data, selects the best checkpoint on validation no-space CER, and evaluates five held-out test views separately. This is a training recipe; no new accuracy or runtime result is claimed.

## Prepare the data

From the repository root, follow [dataset preparation](../../scripts/data/README.md#combine-downloads-with-the-original-audio-data-mappings). Obtain the original ASR/UCSY collections and downloaded saved datasets, then build:

```sh
python scripts/data/build_combined_catalog.py --audio-data-root /path/to/audio-data
```

The builder refuses to overwrite an existing catalog. If it already exists, use that snapshot or choose a new output with `--output-dir` and change `CATALOG_DIR` in the notebook. Keep `audio_data/` and `raw/` links resolvable. On Windows, copy the source directories and use junctions or real directories at those locations. Embedded audio is read directly from the saved Hugging Face datasets; copying CSVs alone is insufficient.

The local snapshot generated on 2026-11-09 contains 37,296 curated clips / 94.190 hours, of which 26,465 clips / 64.147 hours are labeled. VOA contributes 10,831 unlabeled clips / 30.043 hours and is excluded from supervised training and scoring. The duration-only Whisper views contain:

- `test_asr_whisper.csv`: 193 clips / 0.532 hours.
- `test_fleurs_whisper.csv`: 870 clips / 3.717 hours.
- `test_ucsy_whisper.csv`: 1,171 clips / 1.903 hours.
- `test_user_study_whisper.csv`: 153 clips / 0.504 hours.
- `test_yodas_my_whisper.csv`: 129 clips / 0.119 hours.
- `train_whisper.csv`: 21,805 clips / 50.679 hours.
- `validation_whisper.csv`: 1,894 clips / 4.084 hours.

These counts precede the model-specific label-token filter. Audio longer than 30 seconds is excluded, not truncated. Long training/validation labels are recorded in `long_labels.csv`; full test references are retained for inference scoring. “Full” uses all eligible training rows while keeping validation and tests held out.

## Install and run

Install a GPU-appropriate PyTorch >=2.6 build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the shared requirements in a separate environment. See [the environment instructions](../README.md#2-create-a-separate-jupyter-training-environment) for Windows and POSIX commands.

```sh
python -m pip install -r notebook/requirements-training.txt
python -m jupyter lab
```

Select that environment's kernel and open `notebook/whisper-small/train.ipynb`. Run cells top to bottom. The first cell installs the same requirements; restart the kernel if packages changed. Whisper Small uses the existing MLflow tracker: set `MLFLOW_TRACKING_URI` before launching Jupyter (default `http://127.0.0.1:5050`). Reuse your running service; see [tracking setup](../README.md#1-start-docker-mlflow) if needed. CUDA uses FP16; MPS/CPU use full precision. A full run is intended for a GPU, and fit/runtime must be checked on your hardware.

`QUICK_RUN = False` is the default: all eligible training/validation clips, three epochs, then all five test views. `True` selects up to 256 train and 64 validation clips using seed `20261030`, trains for 20 optimizer steps, and skips held-out evaluation. Training batch size is one with 16 gradient accumulation steps. Encoder parameters remain trainable.

Audio is downmixed and resampled to 16 kHz on demand using the shared reader in [training_common.py](../training_common.py). No duplicate WAV or feature cache is created. This trades disk usage for repeated decoding/resampling. Model downloads still use Hugging Face's normal cache.

## Outputs and interruption recovery

Quick outputs go to `run_combined_quick/`; full outputs go to `run_combined_full/`, both beside this README. They are ignored by Git and separate from legacy mapping-only runs. Checkpoints are saved every five quick steps or 50 full steps, retaining up to three including the best evaluated checkpoint. Evaluation occurs every 20 quick steps or 500 full steps.

Each run reuses its saved `mlflow_run_id.txt` in the `myanmar-asr-notebooks` experiment. MLflow records training/validation metrics, baseline metrics, run configuration, catalog audit, rejected labels, and full-mode test metrics/predictions. Tracker or artifact-upload failures stop the run visibly. Model weights stay local.

Each run records `run_config.json` (manifest SHA-256, resolved model revision, training arguments, versions and selected counts), `long_labels.csv`, `baseline_metrics.json`, `training_metrics.json`, and `best_model/`. Full runs add `test_metrics.json` and five `test_*_predictions.csv` files with sample IDs, sources, full references and predictions.

After interruption, restart the kernel and rerun with the same settings and catalog. The notebook validates checkpoint archives from newest to oldest, skips damaged saves, and stops if none is usable. Back up outputs separately. A configuration mismatch stops resume; choose a new `RUN_DIR` for changed settings/data. Set `MODEL_REVISION` to the recorded Hub commit SHA for a pinned repeat run. Manifest fingerprints describe metadata; keep original audio and its catalog hashes unchanged too.

No cache or run directory is required in a clone. To start fresh, choose a new output directory. Delete old local caches/runs only after backing up checkpoints and predictions you need.

## Interpret, share, and cite

WER uses whitespace-delimited tokens, raw CER retains spaces, and no-space CER removes whitespace. Scores are percentages; lower is better. Text is normalized with NFC and collapsed whitespace, retaining punctuation. Best-model selection uses validation no-space CER. Compare the pretrained baseline with final validation on the same selected rows; publish each held-out test score separately with predictions and run configuration.

Exact audio duplicates and training/evaluation normalized-text overlap are checked. Existing YODAS video overlap and uncertain UCSY speaker labels mean these are not guaranteed speaker/video-disjoint benchmarks. Evaluation text can repeat across held-out views. Review the catalog's generated audit and listen to representative clips before drawing conclusions. Historical thesis scores use different data/scoring and are not a baseline for this recipe.

See [reproducibility and provenance](../../docs/REPRODUCIBILITY.md), [contribution guidance](../../CONTRIBUTING.md), and [CITATION.cff](../../CITATION.cff). Cite this repository plus each upstream dataset/model used, and report the repository commit, model commit, manifest hashes, exclusions, and scoring policy. Source recordings retain their own licenses; local ASR/UCSY redistribution permissions are not established here.
