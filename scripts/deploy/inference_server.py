#!/usr/bin/env python3
"""
Myanmar ASR — FastAPI Inference Server (runs on Vast.ai GPU)

Lightweight API server that loads models once and serves transcription requests.
Designed to run on Vast.ai RTX 4090 for fast fp16 inference.

Usage (on Vast.ai):
    pip install fastapi uvicorn python-multipart soundfile librosa torch transformers
    python scripts/deploy/inference_server.py

    # Or with custom port:
    SERVE_PORT=8080 python scripts/deploy/inference_server.py

Endpoints:
    GET  /health              — Health check
    GET  /models              — List available models
    POST /transcribe          — Transcribe audio file
    POST /transcribe?model=dolphin  — Transcribe with specific model
"""

import os
import io
import time
import logging

import numpy as np
import torch
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("myanmar-asr-server")

# ── Config ────────────────────────────────────────────
SAMPLING_RATE = 16000
SERVE_PORT = int(os.environ.get("SERVE_PORT", "8080"))

# Model registry — maps short names to model configs
MODEL_REGISTRY = {
    "whisper-turbo": {
        "id": os.environ.get(
            "WHISPER_TURBO_PATH",
            "/workspace/models/whisper-turbo-myanmar-v3/final",
        ),
        "type": "whisper",
        "name": "Whisper Large-v3 Turbo (Fine-tuned)",
    },
    "whisper-baseline": {
        "id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "name": "Whisper Large-v3 Turbo (Baseline)",
    },
    "dolphin": {
        "id": os.environ.get(
            "DOLPHIN_PATH",
            "/workspace/models/dolphin-myanmar-v1/final",
        ),
        "type": "whisper",
        "name": "Dolphin ASR (Fine-tuned)",
    },
    "seamless": {
        "id": os.environ.get(
            "SEAMLESS_PATH",
            "/workspace/models/seamless-myanmar-v1/final",
        ),
        "type": "seamless",
        "name": "SeamlessM4T v2 (Fine-tuned)",
    },
}

# ── Globals (loaded lazily) ───────────────────────────
_pipelines: dict = {}

app = FastAPI(
    title="Myanmar ASR Inference Server",
    version="1.0.0",
    description="GPU-accelerated transcription for Myanmar/Burmese speech",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model Loading ─────────────────────────────────────
def _get_pipeline(model_key: str):
    """Load a model pipeline (cached in memory)."""
    if model_key in _pipelines:
        return _pipelines[model_key]

    if model_key not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model: {model_key}")

    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg["id"]
    model_type = cfg["type"]

    logger.info(f"Loading model '{model_key}' from {model_id} ...")
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if model_type == "whisper":
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=dtype,
            generate_kwargs={"language": "my", "task": "transcribe"},
        )
    elif model_type == "seamless":
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=dtype,
            generate_kwargs={"tgt_lang": "mya"},
        )
    else:
        raise HTTPException(400, f"Unsupported model type: {model_type}")

    _pipelines[model_key] = pipe
    logger.info(f"Model '{model_key}' loaded on {'CUDA' if device == 0 else 'CPU'} ({dtype})")
    return pipe


def _load_audio_bytes(raw: bytes) -> np.ndarray:
    """Convert raw audio bytes to 16kHz mono float32 numpy array."""
    audio, sr = sf.read(io.BytesIO(raw))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    return audio.astype(np.float32)


# ── Endpoints ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(_pipelines.keys()),
        "models_available": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models")
def list_models():
    return {
        key: {
            "name": cfg["name"],
            "type": cfg["type"],
            "id": cfg["id"],
            "loaded": key in _pipelines,
        }
        for key, cfg in MODEL_REGISTRY.items()
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("whisper-turbo", description="Model key: whisper-turbo, dolphin, seamless, whisper-baseline"),
):
    """Transcribe an uploaded audio file."""
    if model not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY.keys())}")

    # Read & decode audio
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "Empty audio file")

    try:
        audio_array = _load_audio_bytes(raw)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode audio: {exc}")

    duration = len(audio_array) / SAMPLING_RATE

    # Run inference
    pipe = _get_pipeline(model)
    t0 = time.time()
    result = pipe({"array": audio_array, "sampling_rate": SAMPLING_RATE})
    elapsed = time.time() - t0

    return {
        "text": result["text"],
        "model": model,
        "model_name": MODEL_REGISTRY[model]["name"],
        "duration_sec": round(duration, 2),
        "inference_sec": round(elapsed, 3),
        "rtf": round(elapsed / max(duration, 0.01), 4),
        "gpu": torch.cuda.is_available(),
    }


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Myanmar ASR server on port {SERVE_PORT}")
    logger.info(f"GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")

    # Pre-load default model
    try:
        _get_pipeline("whisper-turbo")
    except Exception as exc:
        logger.warning(f"Could not pre-load whisper-turbo: {exc}")

    uvicorn.run(app, host="0.0.0.0", port=SERVE_PORT, log_level="info")
