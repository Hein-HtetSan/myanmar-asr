# Infrastructure and annotation services

Run commands from the repository root. Python helpers use the dependencies for their target service; Bash orchestration additionally needs SSH, rsync, the Vast.ai CLI, and a configured account. Bash scripts require a POSIX environment, such as Linux, macOS, or WSL.

## GPU checks

```sh
python scripts/infra/check_k80.py
python scripts/infra/check_k80.py --model openai/whisper-small
```

`check_k80.py` checks CUDA execution and optionally loads a Whisper model for an inference smoke test. Follow the [K80 environment runbook](../../docs/CentOS7_K80_Compatibility_Decision.md) for installation and interpretation.

## Annotation helpers

Start the required services using [the service stack](../../services/docker-compose.yml) and configure credentials in your environment. Helpers read environment variables directly; do not assume they load `.env` automatically.

- `setup_label_studio.py`: creates a transcription-review project. Requires `LABEL_STUDIO_API_KEY`; `LABEL_STUDIO_URL` defaults to `http://localhost:8081`. Uses `label_studio_sdk`.
- `upload_to_label_studio.py`: exports audio from `combined/myanmar_asr_50h_clean` and uploads tasks. Review its hardcoded `PROJECT_ID = 1` and export directory `~/myanmar-asr/exports/audio` first. Uses the same Label Studio environment variables plus datasets, SoundFile, and requests.
- `setup_argilla.py`: legacy REST workflow that loads `combined/{train,test,validation}_manifest.jsonl` and uploads transcript records. Requires `ARGILLA_USERNAME` and `ARGILLA_PASSWORD`; `ARGILLA_URL` defaults to `http://localhost:6900`.
- `setup_argilla_dataset.py`: alternative Argilla SDK feedback workflow using `ARGILLA_API_KEY`. Reads `combined/train_manifest.jsonl`, uploads up to 500 records, and constructs MinIO audio links. Its SDK imports target an older Argilla API. Configure `MINIO_BASE` and provide real audio paths in the manifest: the metadata-only output of `data/export_manifest.py` lacks those paths.

These helpers create remote projects or records when run; choose the workflow matching your service version and existing project.

## Vast.ai helpers

- `vastai_setup.sh`: installs packages and prepares directories on the remote GPU machine.
- `sync_to_vastai.sh`: takes an instance IP and SSH port, uploads prepared data and scripts, and invokes remote setup.
- `vastai_run.sh`: older orchestration for `search`, `rent`, `sync`, `train`, `status`, `logs`, `results`, and `destroy`. Contains a machine-specific CLI path.
- `oneclick_train.sh`: orchestrates GPU rental, uploads, training, tunnels, results, and instance management; model commands include `turbo`, `dolphin`, `seamless`, and `canary`. Uses `.vastai_state` for instance state.

These scripts mix legacy flattened `/workspace/scripts/` locations with the current `training/`, `evaluation/`, and `utils/` layout. In particular, the one-click uploader preserves subdirectories while its training launcher uses a flattened path. Review and reconcile upload and launch paths before starting a paid run. Rental commands incur charges; `destroy` deletes the instance, so retrieve needed models and results first. This documentation does not validate the cloud pipeline.
