# Myanmar ASR training notebooks

Three notebooks reproduce the model families used by the training scripts, using the combined audio-backed catalog and Docker MLflow tracking:

- [Whisper Large v3 Turbo](whisper-turbo/train.ipynb): `openai/whisper-large-v3-turbo`.
- [Dolphin / Whisper Large v2](dolphin/train.ipynb): `openai/whisper-large-v2`.
- [SeamlessM4T v2 Large](seamless/train.ipynb): `facebook/seamless-m4t-v2-large`, speech-to-text only.

The [Whisper Small notebook](whisper-small/README.md) uses the same combined catalog and shared audio/scoring helpers, defaults to a full run, and tracks metrics and prediction artifacts in the same MLflow experiment. Start there for the smaller model; its setup and outputs are documented separately. The three new notebooks use [training_common.py](training_common.py) for catalog audio access, collation, scoring, checkpoint validation, and MLflow run management. The historical scripts' default model IDs are preserved; their `/workspace` paths, silent tracking fallbacks, label truncation, and Transformers 5.x assumptions are replaced for this notebook workflow.

## 1. Start Docker MLflow

Run from the repository root with Docker Desktop or Docker Engine running:

```sh
docker compose -f services/mlflow-notebooks.compose.yml up -d
docker compose -f services/mlflow-notebooks.compose.yml ps
docker compose -f services/mlflow-notebooks.compose.yml logs --tail=50 mlflow
```

Open **http://127.0.0.1:5050**. This is a dedicated tracking service with SQLite metadata and server-proxied artifacts in a persistent Docker volume. It does not need the annotation services, PostgreSQL, MinIO, or credentials from the main service stack. Jupyter and GPU training run outside this container.

The default port overlaps the main stack's MLflow service. Run one tracker on that port, or change the new service's port. In PowerShell:

```powershell
$env:MLFLOW_NOTEBOOK_PORT = "5051"
docker compose -f services/mlflow-notebooks.compose.yml up -d
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5051"
```

In a POSIX shell:

```sh
MLFLOW_NOTEBOOK_PORT=5051 docker compose -f services/mlflow-notebooks.compose.yml up -d
export MLFLOW_TRACKING_URI=http://127.0.0.1:5051
```

Set the tracking URI before launching Jupyter, or edit `MLFLOW_URI` in the notebook. Each notebook checks connectivity and uploads its configuration before the first optimizer step. An unavailable tracker or failing artifact upload stops the run visibly.

Stop the service without deleting its database or artifacts:

```sh
docker compose -f services/mlflow-notebooks.compose.yml down
```

Keep the named volume for run history; `down -v` deletes it. The compose service binds only to the host loopback interface. For a remote GPU kernel, provide a reachable tracking URI or an SSH tunnel. For example, run `ssh -R 5050:127.0.0.1:5050 user@gpu-host` from the machine hosting Docker, then use `http://127.0.0.1:5050` from the remote kernel while the tunnel stays connected.

The notebooks also embed the MLflow UI in an iframe and print a direct run URL. If the browser cannot resolve the kernel's tracking address or blocks embedding, open the UI in another tab.

## 2. Create a separate Jupyter training environment

Use a separate environment from the download scripts: the notebook stack pins Transformers 4.57.1 and Datasets 4.3.0. Python 3.11 is a practical starting choice; the isolated verification used Python 3.13. Install a **PyTorch >=2.6 build appropriate to your CUDA GPU** using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before the notebook dependencies. The requirements file does not replace your PyTorch build.

PowerShell:

```powershell
py -3.11 -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
# Install the PyTorch command selected for your GPU here.
python -m pip install -r notebook/requirements-training.txt
python -m ipykernel install --user --name myanmar-asr-training --display-name "Myanmar ASR training"
python -m jupyter lab
```

Linux/macOS:

```sh
python3 -m venv .venv-training
source .venv-training/bin/activate
# Install the PyTorch command selected for your GPU here.
python -m pip install -r notebook/requirements-training.txt
python -m ipykernel install --user --name myanmar-asr-training --display-name "Myanmar ASR training"
python -m jupyter lab
```

Choose the **Myanmar ASR training** kernel. Run cells top to bottom. Restart the kernel if the dependency cell changed packages. Run one model notebook at a time; changing notebooks while keeping a previous model loaded does not free GPU memory.

Full runs target CUDA. Quick mode still downloads and loads the full checkpoint, so it is not a CPU-sized model test. Large v2 and Seamless need substantially more memory than Whisper Small; encoder freezing, batch size one, and gradient checkpointing reduce memory but do not guarantee that every model fits a 16 GB GPU. Use the short run to establish that your actual GPU can run the selected model before starting a full experiment.

## 3. Check the combined catalog and storage

Build [the combined catalog](../scripts/data/README.md#combine-downloads-with-the-original-audio-data-mappings) first. The notebooks expect `combined/myanmar_asr_all/` with curated CSV manifests, `summary.json`, and working `audio_data/` and `raw/` links/directories.

For Windows, copy the actual source directories and replace those links with junctions or real directories at the same relative locations. Embedded Hugging Face audio stays in the original saved datasets. The shared reader handles file paths and embedded bytes; audio is downmixed and resampled to 16 kHz on demand. No full feature cache or duplicate waveform collection is generated, and TorchCodec is not required.

The pre-download disk floors are 30 GiB for Turbo, 50 GiB for Large v2, and 60 GiB for Seamless. A second check estimates space from loaded parameter counts for two resumable checkpoints and final weights. These are conservative checks, not a precise disk budget; allow additional space for Hugging Face caches, Docker, artifacts, and later runs. This local machine currently has insufficient space for these full model/checkpoint workflows.

## 4. Run a quick check, then a full experiment

Each notebook starts with `QUICK_RUN=True`: up to 64 training clips, 16 validation clips, and 10 optimizer steps. Evaluation/checkpointing occurs every five steps, and held-out tests are skipped. This verifies a real model's training pipeline; it does not produce a final quality estimate.

For a full run, restart the kernel, set `QUICK_RUN=False`, choose a `RUN_LABEL`, and rerun. Full mode uses all eligible training/validation clips, trains for three epochs with early stopping, and evaluates all named test views separately after selecting the best validation checkpoint. Encoder freezing is enabled by default. The notebooks expose learning rate, batch size, accumulation, maximum generation length, epoch count, and save/evaluation interval.

Model-specific output folders are:

```text
notebook/runs/whisper-turbo/{quick,full}/{RUN_LABEL}/
notebook/runs/dolphin/{quick,full}/{RUN_LABEL}/
notebook/runs/seamless/{quick,full}/{RUN_LABEL}/
```

Each stores resumable checkpoints, `best_model/`, `run_config.json`, and `mlflow_run_id.txt`. Full runs additionally store per-test prediction CSVs and `test_metrics.json`. Outputs are ignored by Git. Back up model files and the tracking volume separately.

After an interruption, keep the same model revision, catalog, settings, and `RUN_LABEL`. Restart and rerun: the training cell chooses the newest checkpoint whose resume archives and model headers pass validation, and reuses the saved MLflow run. Interrupted ZIP saves are skipped. If no checkpoint passes, it stops; validation cannot repair files or prove every optimizer state will load. Work after the latest usable checkpoint is repeated. Use a new `RUN_LABEL` for changed settings or another experiment.

## 5. What MLflow records

All three models share the `myanmar-asr-notebooks` experiment, with separate named runs for model, run label, and quick/full mode. The notebook's explicit run context and callback share one run ID.

- Training loss, learning rate, gradient norm, runtime, and validation loss.
- Validation WER, raw CER, and CER with whitespace removed, as percentages.
- CUDA peak allocated memory when running on CUDA.
- Model ID and resolved revision, dataset manifest/selection fingerprints, hyperparameters, precision, and accepted/rejected label counts.
- Configuration JSON, the catalog audit summary, rejected-label details, and final metrics.
- Full-mode prediction CSVs and WER/CER for each named held-out test set.

The best checkpoint is selected using validation `cer_no_space`, with lower being better. WER uses whitespace-delimited tokens; it is not a Myanmar word-segmentation score. Text is normalized with NFC and collapsed whitespace, and the additional no-space CER removes whitespace entirely. This scoring policy differs from some historical script reports.

Labels that exceed the configured model/generation limit are excluded from training/validation and reported, not truncated. Different tokenizers may retain different samples, so inspect the logged selections before comparing models. Held-out inference uses complete references without teacher-forced labels; all clips in each test duration view are scored. No VOA records enter supervised training or test scoring. Existing YODAS video overlap and inferred UCSY speaker-label overlap remain documented in the dataset report.

Model weights stay local by default. `LOG_MODEL_ARTIFACTS=True` also uploads `best_model/`, duplicating its storage on the MLflow server. Prefer leaving it false until storage is provisioned.

## Validation and limits

Seventeen tests passed in an isolated environment, including notebook JSON/code, label handling, collation, scoring, actual tiny random Whisper and Seamless forward/backward training with frozen encoders, generation, checkpoint checks, run reuse, and artifact upload/download through a temporary HTTP MLflow server. The real processors for all three model IDs also loaded and produced batches from sampled catalog audio. Full model weights and real GPU training were not tested. The Docker compose configuration was validated separately. During setup, the actual Docker image pull failed with `no space left on device`; resolve Docker's storage capacity before starting the service. Existing Docker images, containers, and volumes were not pruned. The temporary HTTP server was stopped after verification.

References: [Transformers 4.57.1 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [SeamlessM4T v2](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/seamless_m4t_v2), and [MLflow 2.19 tracking](https://mlflow.org/docs/2.19.0/tracking.html).
