"""Upload ALL clean dataset samples to Label Studio for review.
Exports WAV files to local dir, creates tasks via API.
"""
import os
import json
import time
import soundfile as sf
import numpy as np
import requests
from datasets import load_from_disk

EXPORT_DIR = os.path.expanduser("~/myanmar-asr/exports/audio")
os.makedirs(EXPORT_DIR, exist_ok=True)

LS_URL = "http://localhost:8080"
TOKEN = "f28275bebd3cdbafa071687ffbc50d183a40a696"
HEADERS = {"Authorization": f"Token {TOKEN}", "Content-Type": "application/json"}
PROJECT_ID = 1

# Check existing task count
resp = requests.get(f"{LS_URL}/api/projects/{PROJECT_ID}/", headers=HEADERS)
resp.raise_for_status()
existing_tasks = resp.json()["task_number"]
print(f"Existing tasks in project: {existing_tasks}")

# Load clean dataset (original, not augmented — augmented copies don't need review)
ds = load_from_disk("combined/myanmar_asr_50h_clean")

total_exported = 0
total_uploaded = 0
t0 = time.time()

for split_name in ds:
    split_ds = ds[split_name]
    print(f"\nProcessing {split_name} ({len(split_ds):,} samples)...")

    tasks = []
    for i, example in enumerate(split_ds):
        audio_array = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        global_idx = total_exported + i
        filename = f"{global_idx:06d}_{split_name}_{example['source']}.wav"
        filepath = os.path.join(EXPORT_DIR, filename)

        # Only write WAV if it doesn't exist yet
        if not os.path.exists(filepath):
            sf.write(filepath, np.array(audio_array, dtype=np.float32), sr)

        tasks.append({
            "data": {
                "audio": f"/data/local-files/?d=audio/{filename}",
                "sentence": example["sentence"],
                "source": example["source"],
                "speaker_id": example["speaker_id"],
                "split": split_name,
            }
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  Exported {i+1}/{len(split_ds)} WAVs | {elapsed:.0f}s")

    # Upload tasks in batches of 500
    BATCH = 500
    for batch_start in range(0, len(tasks), BATCH):
        batch = tasks[batch_start:batch_start+BATCH]
        resp = requests.post(
            f"{LS_URL}/api/projects/{PROJECT_ID}/import",
            headers=HEADERS,
            json=batch,
        )
        if resp.ok:
            total_uploaded += len(batch)
            print(f"  Uploaded batch {batch_start}-{batch_start+len(batch)} | total: {total_uploaded}")
        else:
            print(f"  ⚠️ Upload failed: {resp.status_code} {resp.text[:200]}")

    total_exported += len(split_ds)

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"✅ Uploaded {total_uploaded:,} tasks to Label Studio ({elapsed:.0f}s)")
print(f"   Review at: {LS_URL}/projects/{PROJECT_ID}/")