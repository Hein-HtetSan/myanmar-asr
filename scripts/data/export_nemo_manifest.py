#!/usr/bin/env python3
"""
Export NeMo-format JSONL manifests + WAV files for Canary-1B training.
NeMo requires: {"audio_filepath": ..., "text": ..., "duration": ..., "lang": "my"}

Run on MacBook:
    python3 scripts/export_nemo_manifest.py

Outputs:
    exports/nemo_audio/{train,val,test}/*.wav
    exports/nemo_train_manifest.jsonl
    exports/nemo_val_manifest.jsonl
    exports/nemo_test_manifest.jsonl
"""

import json
import os
import time
import numpy as np
import soundfile as sf
from datasets import load_from_disk

# Use augmented dataset (58h) for training
DS_PATH   = "combined/myanmar_asr_augmented"
AUDIO_DIR = "exports/nemo_audio"
OUT_DIR   = "exports"

SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}

def export_split(ds, split_name, folder_name):
    """Export one split to WAV files + NeMo manifest."""
    audio_dir = os.path.join(AUDIO_DIR, folder_name)
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, f"nemo_{folder_name}_manifest.jsonl")

    t0 = time.time()
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split_name]):
            # Save WAV
            filename = f"{i:06d}.wav"
            audio_arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            sf.write(os.path.join(audio_dir, filename), audio_arr, sr)

            duration = len(audio_arr) / sr

            # NeMo manifest row — audio_filepath uses Vast.ai server path
            # Canary-1B requires: source_lang, target_lang, taskname, pnc
            row = {
                "audio_filepath": f"/workspace/data/nemo_audio/{folder_name}/{filename}",
                "text":           ex["sentence"],
                "duration":       round(duration, 4),
                "lang":           "my",
                "source_lang":    "my",
                "target_lang":    "my",
                "taskname":       "asr",
                "pnc":            "no",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  {split_name}: {i+1:,}/{len(ds[split_name]):,} | {elapsed:.0f}s")

    total = len(ds[split_name])
    elapsed = time.time() - t0
    print(f"✅ {split_name}: {total:,} samples → {manifest_path} ({elapsed:.0f}s)")
    return total


def main():
    print(f"Loading dataset from {DS_PATH}...")
    ds = load_from_disk(DS_PATH)

    total_samples = 0
    for split_name, folder_name in SPLIT_MAP.items():
        if split_name in ds:
            total_samples += export_split(ds, split_name, folder_name)
        else:
            print(f"⚠ Split '{split_name}' not found, skipping")

    print(f"\n{'='*60}")
    print(f"✅ Exported {total_samples:,} total samples")
    print(f"   Audio:     {AUDIO_DIR}/")
    print(f"   Manifests: {OUT_DIR}/nemo_*_manifest.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
