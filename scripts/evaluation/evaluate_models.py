#!/usr/bin/env python3
"""
Compare all fine-tuned Myanmar ASR models on the test set.
Run on Vast.ai AFTER all training jobs complete.

Usage:
    python3 /workspace/scripts/evaluate_models.py

Outputs:
    /workspace/results/model_comparison.csv
    /workspace/results/model_comparison.json
    Logs to W&B summary table
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk, Audio
from transformers import pipeline
import evaluate

# ── Config ────────────────────────────────────────────
DATASET_PATH = "/workspace/data/myanmar_asr"
RESULTS_DIR  = "/workspace/results"
SAMPLING_RATE = 16000

# Whisper-family models (path → label)
HF_MODELS = {
    "Whisper v3 Turbo": "devhnhts/whisper-large-v3-turbo-myanmar",
    "Dolphin ASR":      "devhnhts/dolphin-asr-myanmar",
}

# SeamlessM4T models (different generate_kwargs)
SEAMLESS_MODELS = {
    "SeamlessM4T v2":   "devhnhts/seamless-m4t-v2-myanmar",
}

# Local models (non-HF)
LOCAL_MODELS = {
    "Whisper v3 Turbo (local)": "/workspace/models/whisper-turbo-myanmar/final",
    "Dolphin ASR (local)":      "/workspace/models/dolphin-myanmar/final",
}
LOCAL_SEAMLESS_MODELS = {
    "SeamlessM4T v2 (local)":   "/workspace/models/seamless-myanmar/final",
}

# Canary needs separate NeMo evaluation (handled separately)
CANARY_RESULTS = "/workspace/results/canary_results.json"


def evaluate_whisper_model(model_path, test_ds, wer_metric, cer_metric, device=0):
    """Evaluate a Whisper-family checkpoint on the test set."""
    print(f"  Loading pipeline from {model_path}...")
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            generate_kwargs={"language": "my", "task": "transcribe"},
        )
    except Exception as e:
        print(f"  ⚠ Failed to load {model_path}: {e}")
        return None

    print(f"  Running inference on {len(test_ds)} samples...")
    t0 = time.time()

    predictions = []
    references = test_ds["sentence"]

    # Process in batches for speed
    batch_size = 16
    audio_arrays = [s["array"] for s in test_ds["audio"]]

    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i:i+batch_size]
        batch_inputs = [{"array": a, "sampling_rate": SAMPLING_RATE} for a in batch]
        results = asr(batch_inputs, batch_size=batch_size)
        predictions.extend([r["text"] for r in results])

        if (i + batch_size) % 200 == 0:
            print(f"    {min(i+batch_size, len(audio_arrays))}/{len(audio_arrays)}")

    inference_time = time.time() - t0

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)

    # Compute RTF (Real-Time Factor)
    total_audio_sec = sum(len(a) / SAMPLING_RATE for a in audio_arrays)
    rtf = inference_time / total_audio_sec

    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "inference_time_sec": round(inference_time, 1),
        "total_audio_sec": round(total_audio_sec, 1),
        "rtf": round(rtf, 4),
        "num_samples": len(test_ds),
    }


def evaluate_seamless_model(model_path, test_ds, wer_metric, cer_metric, device=0):
    """Evaluate a SeamlessM4T checkpoint on the test set."""
    print(f"  Loading SeamlessM4T pipeline from {model_path}...")
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            generate_kwargs={"tgt_lang": "mya"},
        )
    except Exception as e:
        print(f"  ⚠ Failed to load {model_path}: {e}")
        return None

    print(f"  Running inference on {len(test_ds)} samples...")
    t0 = time.time()

    predictions = []
    references = test_ds["sentence"]

    batch_size = 8  # SeamlessM4T uses more VRAM
    audio_arrays = [s["array"] for s in test_ds["audio"]]

    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i:i+batch_size]
        batch_inputs = [{"array": a, "sampling_rate": SAMPLING_RATE} for a in batch]
        results = asr(batch_inputs, batch_size=batch_size)
        predictions.extend([r["text"] for r in results])

        if (i + batch_size) % 200 == 0:
            print(f"    {min(i+batch_size, len(audio_arrays))}/{len(audio_arrays)}")

    inference_time = time.time() - t0

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)
    total_audio_sec = sum(len(a) / SAMPLING_RATE for a in audio_arrays)
    rtf = inference_time / total_audio_sec

    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "inference_time_sec": round(inference_time, 1),
        "total_audio_sec": round(total_audio_sec, 1),
        "rtf": round(rtf, 4),
        "num_samples": len(test_ds),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Myanmar ASR — Model Comparison")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu}")

    # Load test set
    print(f"\nLoading test set from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    test_ds = ds["test"]
    print(f"  Test samples: {len(test_ds)}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    results = []

    # ── Evaluate local Whisper models first (faster, no download) ──
    for name, path in LOCAL_MODELS.items():
        if os.path.exists(path):
            print(f"\n{'─'*40}")
            print(f"Evaluating: {name}")
            metrics = evaluate_whisper_model(path, test_ds, wer_metric, cer_metric)
            if metrics:
                metrics["model"] = name
                metrics["source"] = "local"
                results.append(metrics)
                print(f"  ✓ WER: {metrics['wer']}% | CER: {metrics['cer']}% | RTF: {metrics['rtf']}")

    # ── Evaluate local SeamlessM4T models ─────────────
    for name, path in LOCAL_SEAMLESS_MODELS.items():
        if os.path.exists(path):
            print(f"\n{'─'*40}")
            print(f"Evaluating: {name}")
            metrics = evaluate_seamless_model(path, test_ds, wer_metric, cer_metric)
            if metrics:
                metrics["model"] = name
                metrics["source"] = "local"
                results.append(metrics)
                print(f"  ✓ WER: {metrics['wer']}% | CER: {metrics['cer']}% | RTF: {metrics['rtf']}")

    # ── Evaluate HF Hub Whisper models ────────────────
    for name, repo in HF_MODELS.items():
        local_name = f"{name} (local)"
        if any(r["model"] == local_name for r in results):
            print(f"\n  Skipping {name} — already evaluated local checkpoint")
            continue

        print(f"\n{'─'*40}")
        print(f"Evaluating: {name}")
        try:
            metrics = evaluate_whisper_model(repo, test_ds, wer_metric, cer_metric)
            if metrics:
                metrics["model"] = name
                metrics["source"] = "hub"
                results.append(metrics)
                print(f"  ✓ WER: {metrics['wer']}% | CER: {metrics['cer']}% | RTF: {metrics['rtf']}")
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    # ── Evaluate HF Hub SeamlessM4T models ────────────
    for name, repo in SEAMLESS_MODELS.items():
        local_name = f"{name} (local)"
        if any(r["model"] == local_name for r in results):
            print(f"\n  Skipping {name} — already evaluated local checkpoint")
            continue

        print(f"\n{'─'*40}")
        print(f"Evaluating: {name}")
        try:
            metrics = evaluate_seamless_model(repo, test_ds, wer_metric, cer_metric)
            if metrics:
                metrics["model"] = name
                metrics["source"] = "hub"
                results.append(metrics)
                print(f"  ✓ WER: {metrics['wer']}% | CER: {metrics['cer']}% | RTF: {metrics['rtf']}")
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    # ── Include Canary results if available ────────────
    if os.path.exists(CANARY_RESULTS):
        print(f"\nLoading Canary results from {CANARY_RESULTS}...")
        with open(CANARY_RESULTS) as f:
            canary = json.load(f)
        if "test_results" in canary and canary["test_results"]:
            tr = canary["test_results"]
            if isinstance(tr, list):
                tr = tr[0]
            results.append({
                "model": "Canary-1B",
                "source": "nemo",
                "wer": tr.get("test_wer", tr.get("val_wer", "N/A")),
                "cer": tr.get("test_cer", "N/A"),
                "inference_time_sec": canary.get("train_time_seconds", "N/A"),
                "rtf": "N/A",
                "num_samples": "N/A",
            })
            print(f"  ✓ Canary-1B included")

    # ── Summary ───────────────────────────────────────
    if not results:
        print("\n⚠ No models evaluated successfully!")
        return

    df = pd.DataFrame(results).sort_values("wer")

    print(f"\n{'='*60}")
    print("  📊 FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(df[["model", "wer", "cer", "rtf"]].to_string(index=False))

    # Find best model
    best = df.iloc[0]
    print(f"\n  🏆 Best model: {best['model']} (WER: {best['wer']}%, CER: {best['cer']}%)")

    # Save results
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    json_path = os.path.join(RESULTS_DIR, "model_comparison.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to:")
    print(f"    {csv_path}")
    print(f"    {json_path}")

    # ── Log to MLflow ────────────────────────────────
    try:
        import mlflow
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("myanmar-asr")
        with mlflow.start_run(run_name="model-comparison"):
            for r in results:
                name = r["model"].replace(" ", "_").lower()
                for k in ["wer", "cer", "rtf"]:
                    if isinstance(r.get(k), (int, float)):
                        mlflow.log_metric(f"compare/{name}_{k}", r[k])
            # Log comparison table as artifact
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(json_path)
        print(f"  ✓ Logged to MLflow")
    except Exception as e:
        print(f"  ⚠ MLflow logging skipped: {e}")

    # ── Log to W&B ───────────────────────────────────
    try:
        import wandb
        run = wandb.init(project="myanmar-asr", name="model-comparison", job_type="eval")
        table = wandb.Table(dataframe=df)
        run.log({"model_comparison": table})
        run.finish()
        print(f"  ✓ Logged to W&B")
    except Exception as e:
        print(f"  ⚠ W&B logging skipped: {e}")


if __name__ == "__main__":
    main()
