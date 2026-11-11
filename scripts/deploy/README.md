# Inference API and demo

Run commands from the repository root. The API uses FastAPI, Uvicorn, python-multipart, PyTorch, Transformers, SoundFile, and librosa. The demo additionally uses Streamlit, requests, and MLflow; see [demo dependencies](../../requirements-demo.txt). Model downloads require Hub access when weights are not local.

## Inference server

`inference_server.py` serves these endpoints:

- `GET /health`: service health.
- `GET /models`: configured model registry.
- `POST /transcribe`: multipart audio transcription; select a registry key using `?model=whisper-small`, `whisper-turbo`, `dolphin`, `seamless`, or `whisper-baseline`.

For a local server using pretrained Whisper Small, set the environment and launch it. In PowerShell:

```powershell
$env:SERVE_HOST = "127.0.0.1"
$env:DEFAULT_MODEL = "whisper-small"
python scripts/deploy/inference_server.py
```

In a POSIX shell:

```sh
SERVE_HOST=127.0.0.1 DEFAULT_MODEL=whisper-small python scripts/deploy/inference_server.py
```

The default port is 8080 (`SERVE_PORT`). For your trained notebook model, set `WHISPER_SMALL_PATH` to its `best_model/` folder before launch. Script-trained models normally use their `final/` folder. Other path overrides are `WHISPER_TURBO_PATH`, `DOLPHIN_PATH`, and `SEAMLESS_PATH`; their defaults point to `/workspace/models/` experiments. The server's default host is `0.0.0.0`; use a trusted network or appropriate deployment access controls when exposing it.

## Streamlit demo

```sh
streamlit run scripts/deploy/streamlit_app.py
```

The demo supports local/cloud inference, uploaded or recorded audio, model comparison, and MLflow inspection. `CLOUD_INFERENCE_URL` selects the cloud API, `CLOUD_SERVER_PORT` defaults to 8080, `MLFLOW_TRACKING_URI` defaults to `http://localhost:5050`, and `APP_ROOT` overrides the detected repository root.

`_inspect_mlflow.py` is a diagnostic helper that lists experiments, runs, and artifacts; inspect its tracking configuration before running.

For the older CentOS 7 / K80 environment and its float32 inference requirements, follow the [deployment runbook](../../docs/CentOS7_K80_Compatibility_Decision.md).

## Research results export and deployment

`export_mlflow_results.py` discovers completed runs in your configured MLflow experiment (or filters by optional run IDs), reading numeric aggregate result JSON from its artifacts, and exports them to `web/results.json`. `build_research_site.py` packages only the public static site for Vercel. See [the site and CI/CD setup](../../web/README.md) for exported fields, credentials, runner connectivity, and publishing controls. Inference stays on its separate GPU/API host.
