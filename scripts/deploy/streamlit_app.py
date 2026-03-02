#!/usr/bin/env python3
"""
Myanmar ASR — Streamlit Demo App

Interactive web interface for testing Burmese speech recognition models.
Supports **Local** (Apple Silicon / CPU) and **Cloud** (Vast.ai GPU) inference,
audio upload, microphone recording, model comparison, and MLflow run inspection.

Usage:
    streamlit run scripts/deploy/streamlit_app.py

    # With custom MLflow URI (default: http://localhost:5050):
    MLFLOW_TRACKING_URI=http://localhost:5050 streamlit run scripts/deploy/streamlit_app.py

Requirements:
    pip install streamlit torch transformers soundfile librosa numpy mlflow requests
"""

import os
import io
import json
import time
import tempfile
import pathlib
import numpy as np
import soundfile as sf
import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Myanmar ASR — Burmese Speech Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────
SAMPLING_RATE = 16000
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")

# MinIO / S3 credentials (needed for MLflow artifact downloads)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9002")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")

# Project root — works both locally (scripts/deploy/) and in Docker (/app/)
_PROJECT_ROOT = pathlib.Path(os.environ.get("APP_ROOT", pathlib.Path(__file__).resolve().parent.parent.parent))

# Local cache directory for models downloaded from MLflow/MinIO
MODEL_CACHE_DIR = _PROJECT_ROOT / "models" / "mlflow_cache"

# ── Vast.ai Cloud Inference ──────────────────────────
VASTAI_STATE_FILE = _PROJECT_ROOT / ".vastai_state"
CLOUD_SERVER_PORT = int(os.environ.get("CLOUD_SERVER_PORT", "8080"))

# Model key mapping — Streamlit display name → server model key
CLOUD_MODEL_KEYS = {
    "Whisper Large-v3 Turbo (Fine-tuned)": "whisper-turbo",
    "Whisper Large-v3 Turbo (Baseline)": "whisper-baseline",
    "Dolphin ASR (Fine-tuned)": "dolphin",
    "SeamlessM4T v2 (Fine-tuned)": "seamless",
}

# Available models — resolved at runtime from MLflow artifacts or HF Hub
MODELS = {
    "Whisper Large-v3 Turbo (Fine-tuned)": {
        "id": "devhnhts/whisper-large-v3-turbo-myanmar",  # HF Hub fallback
        "type": "whisper",
        "description": "OpenAI Whisper v3 Turbo fine-tuned on 50h Myanmar speech data",
        "mlflow_experiment": "myanmar-asr-whisper-turbo",
        "mlflow_run_name": "whisper-turbo-myanmar-v3-frozen-enc",
    },
    "Whisper Large-v3 Turbo (Baseline)": {
        "id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "description": "Original OpenAI Whisper v3 Turbo (zero-shot Myanmar)",
        "mlflow_experiment": None,
        "mlflow_run_name": None,
    },
    "Dolphin ASR (Fine-tuned)": {
        "id": "devhnhts/dolphin-asr-myanmar",  # HF Hub fallback (not yet pushed)
        "type": "whisper",
        "description": "Whisper Large-v2 with decoder-only fine-tuning for Myanmar",
        "mlflow_experiment": "myanmar-asr-dolphin",
        "mlflow_run_name": "dolphin-myanmar-v1-frozen-enc",
    },
    "SeamlessM4T v2 (Fine-tuned)": {
        "id": "devhnhts/seamless-m4t-v2-myanmar",  # HF Hub fallback (not yet pushed)
        "type": "seamless",
        "description": "Meta SeamlessM4T v2 Large fine-tuned for Myanmar ASR",
        "mlflow_experiment": "myanmar-asr-seamless",
        "mlflow_run_name": "seamless-myanmar-v1-frozen-enc",
    },
}


# ── MLflow Helpers ────────────────────────────────────
@st.cache_resource(ttl=60)
def get_mlflow_client():
    """Connect to MLflow server and return client, or None on failure."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        # Quick connectivity test
        client.search_experiments()
        return client
    except Exception:
        return None


def get_mlflow_runs(client, experiment_name="myanmar-asr"):
    """Fetch all runs from the given experiment."""
    try:
        import mlflow
        exps = client.search_experiments(filter_string=f"name = '{experiment_name}'")
        if not exps:
            return []
        exp_id = exps[0].experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["metrics.eval_cer ASC"],
            max_results=50,
        )
        return runs
    except Exception:
        return []


def get_best_model_path_from_mlflow(client, run_name_hint, experiment_name="myanmar-asr"):
    """Try to find a local artifact path from MLflow for a given run."""
    try:
        import mlflow
        exps = client.search_experiments(filter_string=f"name = '{experiment_name}'")
        if not exps:
            return None
        exp_id = exps[0].experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.runName LIKE '%{run_name_hint}%'",
            order_by=["metrics.eval_cer ASC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        artifact_uri = run.info.artifact_uri
        # If local path (file://), return it directly
        if artifact_uri.startswith("file://"):
            return artifact_uri.replace("file://", "")
        return None
    except Exception:
        return None


@st.cache_resource(show_spinner="Downloading model from MLflow/MinIO...")
def download_model_from_mlflow(experiment_name: str, run_name: str) -> str | None:
    """Download a model artifact directory from MLflow (MinIO S3) to local cache.

    Returns the local path to the downloaded model directory, or None on failure.
    """
    try:
        from mlflow.tracking import MlflowClient
        from mlflow.artifacts import download_artifacts

        client = MlflowClient(tracking_uri=MLFLOW_URI)

        exps = client.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )
        if not exps:
            return None
        exp_id = exps[0].experiment_id

        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["metrics.eval_cer ASC"],
            max_results=1,
        )
        if not runs:
            return None

        run = runs[0]
        run_id = run.info.run_id

        # Download 'model' artifact folder to local cache
        dest = str(MODEL_CACHE_DIR / experiment_name / run_name)
        local_path = download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=dest,
            tracking_uri=MLFLOW_URI,
        )
        return local_path
    except Exception as exc:
        st.warning(f"Could not download model from MLflow: {exc}")
        return None


def resolve_model_id(model_info: dict) -> str:
    """Resolve the effective model path — MLflow artifact first, then HF Hub fallback."""
    exp = model_info.get("mlflow_experiment")
    run_name = model_info.get("mlflow_run_name")

    if exp and run_name:
        # Check if already cached locally
        cached = MODEL_CACHE_DIR / exp / run_name / "model"
        if cached.is_dir() and any(cached.iterdir()):
            return str(cached)

        # Try downloading from MLflow/MinIO
        local_path = download_model_from_mlflow(exp, run_name)
        if local_path and os.path.isdir(local_path):
            return local_path

    # Fallback to HF Hub ID
    return model_info["id"]


# ── Cloud Inference Helpers ───────────────────────────
def _read_vastai_state() -> dict | None:
    """Read Vast.ai instance info from .vastai_state file."""
    if not VASTAI_STATE_FILE.exists():
        return None
    try:
        parts = VASTAI_STATE_FILE.read_text().strip().split("|")
        if len(parts) >= 3:
            return {"inst_id": parts[0], "host": parts[1], "ssh_port": parts[2]}
    except Exception:
        pass
    return None


def _get_cloud_url() -> str | None:
    """Build the cloud server URL from session state or environment."""
    # User override in session state
    if st.session_state.get("cloud_url"):
        return st.session_state["cloud_url"].rstrip("/")
    # Env var override
    env_url = os.environ.get("CLOUD_INFERENCE_URL")
    if env_url:
        return env_url.rstrip("/")
    return None


def cloud_health_check(url: str) -> dict | None:
    """Check if the cloud inference server is reachable."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def cloud_transcribe(audio_array: np.ndarray, model_name: str, cloud_url: str):
    """Send audio to the cloud server for transcription.

    Returns (text, elapsed_seconds) tuple.
    """
    model_key = CLOUD_MODEL_KEYS.get(model_name)
    if not model_key:
        return f"❌ Model '{model_name}' not available for cloud inference.", 0.0

    # Encode audio as WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio_array, SAMPLING_RATE, format="WAV", subtype="FLOAT")
    buf.seek(0)

    try:
        t0 = time.time()
        resp = requests.post(
            f"{cloud_url}/transcribe",
            params={"model": model_key},
            files={"file": ("audio.wav", buf, "audio/wav")},
            timeout=120,
        )
        elapsed = time.time() - t0

        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            return f"❌ Cloud error ({resp.status_code}): {detail}", elapsed

        data = resp.json()
        return data.get("text", ""), elapsed
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to cloud server. Is the inference server running?", 0.0
    except requests.exceptions.Timeout:
        return "❌ Cloud request timed out (>120s).", 0.0
    except Exception as exc:
        return f"❌ Cloud inference error: {exc}", 0.0


# ── Model Loading (Cached) — Local Mode ──────────────
def _get_device_and_dtype(model_type: str = "whisper"):
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU.

    - Whisper / Dolphin → MPS + float32 (works on Apple Silicon)
    - SeamlessM4T → CPU + float32 (MPS causes OverflowError even with fp32)
    - CUDA always uses fp16 for all models
    """
    import torch

    if torch.cuda.is_available():
        return 0, torch.float16                      # CUDA GPU — fp16 OK

    # Apple Silicon — SeamlessM4T is NOT compatible with MPS
    if torch.backends.mps.is_available() and model_type != "seamless":
        return "mps", torch.float32                   # Apple Silicon GPU — fp32 safe

    return -1, torch.float32                          # CPU fallback (or SeamlessM4T)


@st.cache_resource(show_spinner="Loading Whisper model — first run takes a minute…")
def load_whisper_pipeline(model_id):
    """Load a Whisper ASR pipeline."""
    from transformers import pipeline

    device, dtype = _get_device_and_dtype("whisper")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        dtype=dtype,
        generate_kwargs={"language": "my", "task": "transcribe"},
    )
    return pipe


@st.cache_resource(show_spinner="Loading SeamlessM4T model — first run takes a minute…")
def load_seamless_pipeline(model_id):
    """Load a SeamlessM4T ASR pipeline — always on CPU (MPS incompatible)."""
    from transformers import pipeline

    device, dtype = _get_device_and_dtype("seamless")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        dtype=dtype,
        generate_kwargs={"tgt_lang": "mya"},
    )
    return pipe


def load_audio(uploaded_file):
    """Load audio from uploaded file, resample to 16kHz mono."""
    import librosa

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    audio, sr = sf.read(tmp_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)

    os.unlink(tmp_path)
    return audio


def transcribe(audio_array, model_name, model_info, *, mode="local", cloud_url=None):
    """Run transcription — locally or via cloud server."""
    # ── Cloud Mode ──
    if mode == "cloud" and cloud_url:
        return cloud_transcribe(audio_array, model_name, cloud_url)

    # ── Local Mode ──
    t0 = time.time()

    # Resolve model path: MLflow/MinIO first, then HF Hub fallback
    model_id = resolve_model_id(model_info)

    try:
        if model_info["type"] == "whisper":
            pipe = load_whisper_pipeline(model_id)
        elif model_info["type"] == "seamless":
            pipe = load_seamless_pipeline(model_id)
        else:
            return "Unsupported model type", 0.0
    except OSError as exc:
        return f"❌ Model not available: {exc}", 0.0
    except Exception as exc:
        return f"❌ Error loading model: {exc}", 0.0

    result = pipe(
        {"array": audio_array, "sampling_rate": SAMPLING_RATE},
    )
    elapsed = time.time() - t0

    return result["text"], elapsed


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Flag_of_Myanmar.svg/200px-Flag_of_Myanmar.svg.png", width=80)
    st.title("🎙️ Myanmar ASR")
    st.markdown("**Burmese Speech Recognition**")
    st.markdown("---")

    # ━━ Inference Mode Toggle ━━
    st.subheader("Inference Mode")

    # Detect if torch is installed (local inference requires it)
    _torch_available = False
    try:
        import torch as _torch_check  # noqa: F401
        _torch_available = True
    except ImportError:
        pass

    if _torch_available:
        inference_mode = st.radio(
            "Where to run models:",
            ["🖥️ Local (Apple Silicon)", "☁️ Cloud (Vast.ai GPU)"],
            index=0,
            key="inference_mode_radio",
            help="**Local**: Run on this Mac using MPS/CPU. "
                 "**Cloud**: Send audio to Vast.ai RTX 4090 for fast CUDA inference.",
        )
    else:
        inference_mode = "☁️ Cloud (Vast.ai GPU)"
        st.info("☁️ Cloud mode only (torch not installed)")

    is_cloud = inference_mode.startswith("☁️")

    # Cloud server configuration
    cloud_url = None
    cloud_ok = False
    if is_cloud:
        vastai_state = _read_vastai_state()
        default_url = os.environ.get("CLOUD_INFERENCE_URL", "")
        if not default_url and vastai_state:
            # Vast.ai uses SSH tunneling; default to localhost with port-forward
            default_url = f"http://localhost:{CLOUD_SERVER_PORT}"

        cloud_url = st.text_input(
            "Cloud server URL",
            value=default_url,
            placeholder="http://your-vastai-ip:8080",
            key="cloud_url",
        )

        if cloud_url:
            with st.spinner("Checking cloud server..."):
                health = cloud_health_check(cloud_url)
            if health:
                gpu_name = health.get("gpu_name", "Unknown")
                loaded = health.get("models_loaded", [])
                available = health.get("models_available", [])
                st.success(f"✅ Cloud connected\n**GPU:** {gpu_name}")
                st.caption(f"Models loaded: {len(loaded)}/{len(available)}")
                cloud_ok = True
            else:
                st.error(
                    f"❌ Cannot reach `{cloud_url}`\n\n"
                    "**Setup on Vast.ai:**\n"
                    "```bash\n"
                    "pip install fastapi uvicorn python-multipart soundfile librosa\n"
                    "python scripts/deploy/inference_server.py\n"
                    "```"
                )
        else:
            st.warning("Enter the cloud server URL to use cloud inference.")
            if vastai_state:
                st.caption(
                    f"Vast.ai instance: `{vastai_state['host']}` "
                    f"(SSH port {vastai_state['ssh_port']})"
                )
    else:
        # Local mode — MLflow + MinIO status
        mlflow_client = get_mlflow_client()
        if mlflow_client:
            st.success(f"✅ MLflow connected\n`{MLFLOW_URI}`")
            st.caption(f"MinIO S3: `{os.environ.get('MLFLOW_S3_ENDPOINT_URL')}`")
        else:
            st.warning(
                f"⚠️ MLflow offline `{MLFLOW_URI}`\n\n"
                "Models will load from HF Hub fallback.\n\n"
                "Start Docker:\n"
                "```\ncd services && docker compose up -d\n```"
            )

    st.markdown("---")

    # Model selection
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Choose a model:",
        list(MODELS.keys()),
        index=0,
    )
    model_info = MODELS[selected_model]
    st.caption(model_info["description"])

    # Show model source info
    if is_cloud:
        cloud_key = CLOUD_MODEL_KEYS.get(selected_model, "?")
        st.caption(f"☁️ Cloud model key: `{cloud_key}`")
    else:
        exp = model_info.get("mlflow_experiment")
        run_name = model_info.get("mlflow_run_name")
        if exp and run_name:
            cached = MODEL_CACHE_DIR / exp / run_name / "model"
            if cached.is_dir() and any(cached.iterdir()):
                st.caption(f"📦 Cached locally: `{cached}`")
            else:
                st.caption(f"☁️ Will download from MLflow: `{exp}/{run_name}`")
        else:
            st.caption(f"🤗 HuggingFace Hub: `{model_info['id']}`")

        # Show device info for local mode
        if model_info["type"] == "seamless":
            st.caption("🐢 Device: CPU (SeamlessM4T — MPS incompatible)")
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    st.caption(f"🖥️ Device: CUDA ({torch.cuda.get_device_name(0)})")
                elif torch.backends.mps.is_available():
                    st.caption("🍎 Device: Apple Silicon GPU (MPS, float32)")
                else:
                    st.caption("🐢 Device: CPU")
            except ImportError:
                st.caption("🐢 Device: N/A (torch not installed — use Cloud mode)")

    # Comparison mode
    st.markdown("---")
    compare_mode = st.checkbox("🔄 Compare Models", value=False)
    compare_models = []
    if compare_mode:
        compare_models = st.multiselect(
            "Select models to compare:",
            [m for m in MODELS.keys() if m != selected_model],
            default=[list(MODELS.keys())[1]] if len(MODELS) > 1 else [],
        )

    # Info
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This demo tests fine-tuned ASR models for Myanmar (Burmese) language.

    **Dataset:** ~50 hours of Myanmar speech
    - FLEURS (Google)
    - OpenSLR-80
    - Common Voice (Mozilla)
    - VOA Burmese

    **Models trained on:** Vast.ai RTX 4090
    """)

    st.markdown("---")
    st.markdown("*Myanmar ASR M5 Project*")


# ── Main Content ──────────────────────────────────────
st.title("🇲🇲 Myanmar Speech Recognition")

# Mode banner
if is_cloud:
    if cloud_ok:
        st.info("☁️ **Cloud Mode** — Audio will be sent to Vast.ai RTX 4090 for fast GPU inference.")
    else:
        st.warning("☁️ **Cloud Mode** — Server not connected. Configure the URL in the sidebar.")
else:
    st.caption("🖥️ **Local Mode** — Running inference on this machine.")

st.markdown("Upload an audio file or record from your microphone to transcribe Burmese speech.")

# Tabs
tab_upload, tab_record, tab_mlflow, tab_results = st.tabs(["📁 Upload Audio", "🎤 Record Audio", "📈 MLflow Runs", "📊 Results"])

with tab_upload:
    st.markdown("### Upload an audio file")
    st.caption("Supported formats: WAV, MP3, FLAC, OGG, M4A")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
        key="upload",
    )

    if uploaded_file is not None:
        # Show audio player
        st.audio(uploaded_file, format="audio/wav")

        # Load audio
        uploaded_file.seek(0)  # Reset after audio player
        with st.spinner("Loading audio..."):
            audio_array = load_audio(uploaded_file)

        duration = len(audio_array) / SAMPLING_RATE
        st.info(f"Audio duration: {duration:.1f}s | Sample rate: {SAMPLING_RATE} Hz")

        # Transcribe button
        if st.button("🚀 Transcribe", key="btn_upload", type="primary"):
            cur_mode = "cloud" if is_cloud else "local"

            # Primary model
            with st.spinner(f"Transcribing with {selected_model} ({cur_mode})..."):
                text, elapsed = transcribe(audio_array, selected_model, model_info, mode=cur_mode, cloud_url=cloud_url)

            st.success(f"Completed in {elapsed:.2f}s (RTF: {elapsed/duration:.3f})")

            st.markdown("### Transcription")
            st.markdown(f"""
            <div style="background-color: #1e1e2e; padding: 20px; border-radius: 10px;
                        font-size: 1.3em; line-height: 1.8; color: #cdd6f4;
                        border-left: 4px solid #89b4fa;">
                {text}
            </div>
            """, unsafe_allow_html=True)

            # Comparison mode
            if compare_mode and compare_models:
                st.markdown("### Model Comparison")
                cols = st.columns(len(compare_models))
                for i, cmp_model in enumerate(compare_models):
                    with cols[i]:
                        cmp_info = MODELS[cmp_model]
                        with st.spinner(f"Running {cmp_model}..."):
                            cmp_text, cmp_elapsed = transcribe(audio_array, cmp_model, cmp_info, mode=cur_mode, cloud_url=cloud_url)

                        st.markdown(f"**{cmp_model}**")
                        st.markdown(f"*{cmp_elapsed:.2f}s (RTF: {cmp_elapsed/duration:.3f})*")
                        st.markdown(f"""
                        <div style="background-color: #313244; padding: 15px; border-radius: 8px;
                                    font-size: 1.1em; line-height: 1.6; color: #cdd6f4;">
                            {cmp_text}
                        </div>
                        """, unsafe_allow_html=True)

with tab_record:
    st.markdown("### Record from Microphone")
    st.caption("Click the button below to start recording.")

    # Streamlit's built-in audio input
    audio_bytes = st.audio_input("🎤 Record audio", key="mic")

    if audio_bytes is not None:
        with st.spinner("Processing recorded audio..."):
            # Read the recorded audio
            audio_bytes.seek(0)
            audio_data, sr = sf.read(io.BytesIO(audio_bytes.read()))

            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample if needed
            if sr != SAMPLING_RATE:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLING_RATE)

        duration = len(audio_data) / SAMPLING_RATE
        st.info(f"Recorded: {duration:.1f}s | Sample rate: {SAMPLING_RATE} Hz")

        if st.button("🚀 Transcribe Recording", key="btn_record", type="primary"):
            cur_mode = "cloud" if is_cloud else "local"
            with st.spinner(f"Transcribing with {selected_model} ({cur_mode})..."):
                text, elapsed = transcribe(audio_data, selected_model, model_info, mode=cur_mode, cloud_url=cloud_url)

            st.success(f"Completed in {elapsed:.2f}s")
            st.markdown("### Transcription")
            st.markdown(f"""
            <div style="background-color: #1e1e2e; padding: 20px; border-radius: 10px;
                        font-size: 1.3em; line-height: 1.8; color: #cdd6f4;
                        border-left: 4px solid #a6e3a1;">
                {text}
            </div>
            """, unsafe_allow_html=True)

with tab_mlflow:
    st.markdown("### MLflow Experiment Runs")

    # MLflow client — always try to connect (independent of inference mode)
    _mlflow_client = get_mlflow_client()

    if not _mlflow_client:
        st.error(
            f"**Cannot connect to MLflow at `{MLFLOW_URI}`.**\n\n"
            "Make sure Docker services are running:\n"
            "```bash\n"
            "cd services && docker compose up -d\n"
            "```\n"
            "Then refresh this page."
        )
    else:
        import pandas as pd

        experiment_name = st.text_input("Experiment name", value="myanmar-asr")
        runs = get_mlflow_runs(_mlflow_client, experiment_name)

        if not runs:
            st.warning(f"No runs found in experiment **{experiment_name}**.")
        else:
            # Build a summary DataFrame
            rows = []
            for r in runs:
                metrics = r.data.metrics
                tags = r.data.tags
                rows.append({
                    "Run Name": tags.get("mlflow.runName", r.info.run_id[:8]),
                    "Status": r.info.status,
                    "WER": round(metrics.get("eval_wer", metrics.get("test_wer", float("nan"))), 2),
                    "CER": round(metrics.get("eval_cer", metrics.get("test_cer", float("nan"))), 2),
                    "Eval Loss": round(metrics.get("eval_loss", float("nan")), 4),
                    "Train Loss": round(metrics.get("train_loss", float("nan")), 4),
                    "Steps": int(metrics.get("step", 0)),
                    "Run ID": r.info.run_id[:12],
                })

            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            # Best run summary
            valid = df_runs.dropna(subset=["CER"])
            if not valid.empty:
                best = valid.loc[valid["CER"].idxmin()]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🏆 Best Run", best["Run Name"])
                col2.metric("CER", f"{best['CER']}%")
                col3.metric("WER", f"{best['WER']}%")
                col4.metric("Eval Loss", best["Eval Loss"])

            # Drill into a specific run
            st.markdown("---")
            st.markdown("#### Inspect Run Metrics")
            run_names = [r["Run Name"] for r in rows]
            selected_run_name = st.selectbox("Select run", run_names)

            sel_run = next(
                (r for r in runs if r.data.tags.get("mlflow.runName", r.info.run_id[:8]) == selected_run_name),
                None,
            )
            if sel_run:
                # Params
                with st.expander("Hyperparameters", expanded=False):
                    params = sel_run.data.params
                    if params:
                        param_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No params logged.")

                # Metric history chart
                with st.expander("Training Curves", expanded=True):
                    try:
                        metric_keys = list(sel_run.data.metrics.keys())
                        chosen_metrics = st.multiselect(
                            "Plot metrics",
                            metric_keys,
                            default=[m for m in ["eval_cer", "eval_wer", "eval_loss", "train_loss"] if m in metric_keys],
                        )
                        if chosen_metrics:
                            chart_data = {}
                            for mk in chosen_metrics:
                                history = _mlflow_client.get_metric_history(sel_run.info.run_id, mk)
                                if history:
                                    chart_data[mk] = {h.step: h.value for h in sorted(history, key=lambda x: x.step)}

                            if chart_data:
                                all_steps = sorted(set(s for v in chart_data.values() for s in v))
                                plot_rows = {mk: [chart_data[mk].get(s, None) for s in all_steps] for mk in chart_data}
                                chart_df = pd.DataFrame(plot_rows, index=all_steps)
                                st.line_chart(chart_df)
                    except Exception as e:
                        st.warning(f"Could not load metric history: {e}")

                # Artifacts
                with st.expander("Artifacts", expanded=False):
                    try:
                        artifacts = _mlflow_client.list_artifacts(sel_run.info.run_id)
                        if artifacts:
                            for a in artifacts:
                                st.text(f"  📄 {a.path}  ({a.file_size or 0} bytes)")
                        else:
                            st.info("No artifacts logged.")
                    except Exception as e:
                        st.warning(f"Could not list artifacts: {e}")

            # Link to full MLflow UI
            st.markdown("---")
            st.markdown(f"🔗 **[Open MLflow Dashboard]({MLFLOW_URI})** *(requires SSH tunnel)*")


with tab_results:
    st.markdown("### Model Comparison Results")
    st.caption("Results from the evaluation on the test set (loaded from results files if available).")

    # Try to load comparison results
    results_paths = [
        str(_PROJECT_ROOT / "results" / "model_comparison.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "results", "model_comparison.json"),
        "/workspace/results/model_comparison.json",
    ]

    comparison_data = None
    for rp in results_paths:
        if os.path.exists(rp):
            with open(rp) as f:
                comparison_data = json.load(f)
            break

    if comparison_data:
        import pandas as pd

        df = pd.DataFrame(comparison_data)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        best = df.loc[df["wer"].idxmin()] if "wer" in df.columns else None
        if best is not None:
            col1.metric("🏆 Best Model", best["model"])
            col2.metric("Best WER", f"{best['wer']}%")
            col3.metric("Best CER", f"{best.get('cer', 'N/A')}%")

        # Full table
        st.dataframe(
            df[["model", "wer", "cer", "rtf", "inference_time_sec", "num_samples"]],
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart
        st.markdown("#### WER Comparison")
        chart_df = df[["model", "wer"]].set_index("model")
        st.bar_chart(chart_df)

        if "cer" in df.columns:
            st.markdown("#### CER Comparison")
            cer_df = df[["model", "cer"]].set_index("model")
            st.bar_chart(cer_df)
    else:
        st.warning("No comparison results found. Run `evaluate_models.py` first to generate results.")

        # Show placeholder
        st.markdown("""
        **Expected results format** (`results/model_comparison.json`):
        ```json
        [
            {"model": "Whisper v3 Turbo", "wer": 45.2, "cer": 22.1, "rtf": 0.12},
            {"model": "Dolphin ASR", "wer": 48.5, "cer": 25.3, "rtf": 0.15},
            {"model": "SeamlessM4T v2", "wer": 52.1, "cer": 28.7, "rtf": 0.18}
        ]
        ```
        """)


# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c7293;'>"
    "Myanmar ASR M5 Project | Fine-tuned on ~50h Burmese speech | "
    "Powered by 🤗 Transformers + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
