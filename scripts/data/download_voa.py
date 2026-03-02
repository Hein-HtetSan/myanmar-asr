"""
download_voa.py
Download VOA Myanmar speech data from HuggingFace (freococo/voa_myanmar_asr_audio_1).

Streams data, saves WAV files to disk incrementally to avoid OOM.
Then builds HuggingFace DatasetDict from the saved files.
"""

import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, Audio

TARGET_HOURS = 30.0
MAX_SAMPLES  = 15_000
AUDIO_DIR    = Path("raw/voa_myanmar_wav")
META_PATH    = Path("raw/voa_myanmar_meta.jsonl")
SAVE_PATH    = "raw/voa_myanmar_hf"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Check if we already downloaded enough
existing_count = 0
existing_dur = 0.0
if META_PATH.exists():
    with open(META_PATH) as f:
        for line in f:
            rec = json.loads(line)
            existing_dur += rec.get("duration", 0)
            existing_count += 1
    print(f"Found {existing_count} existing samples ({existing_dur/3600:.2f}h)")
    if existing_dur / 3600 >= TARGET_HOURS:
        print("Already have enough data, skipping download.")
        # Skip to building dataset
        rows = []
        with open(META_PATH) as f:
            for line in f:
                rows.append(json.loads(line))
        print(f"Building dataset from {len(rows)} existing samples...")
        # Jump to dataset building below
        import sys
        # We'll handle this in the build section
    else:
        print(f"Need {TARGET_HOURS - existing_dur/3600:.1f}h more, resuming download...")

print(f"Downloading VOA Myanmar — target {TARGET_HOURS}h, max {MAX_SAMPLES} samples")
print("=" * 60)

ds_stream = load_dataset(
    "freococo/voa_myanmar_asr_audio_1",
    trust_remote_code=True,
    streaming=True,
)

meta_file = open(META_PATH, "a" if existing_count > 0 else "w")
count = existing_count
total_dur = existing_dur
errors = 0
skip = existing_count  # Skip already-downloaded samples
t0 = time.time()

for i, ex in enumerate(ds_stream["train"]):
    if i < skip:
        continue
    
    try:
        meta = ex.get("json", {}) or {}
        duration = float(meta.get("duration", 0))
        
        sentence = ""
        if "txt" in ex and ex["txt"]:
            sentence = ex["txt"]
        elif "text" in meta:
            sentence = meta["text"]
        elif "transcription" in meta:
            sentence = meta["transcription"]
        
        audio_data = ex.get("mp3")
        if audio_data is None or not isinstance(audio_data, dict):
            errors += 1
            continue
        
        arr = audio_data.get("array")
        sr = audio_data.get("sampling_rate", 16000)
        if arr is None or len(arr) == 0:
            errors += 1
            continue
        
        computed_dur = len(arr) / sr
        if duration == 0:
            duration = computed_dur
        
        # Save WAV file to disk
        wav_path = AUDIO_DIR / f"voa_{count:06d}.wav"
        arr_np = np.array(arr, dtype=np.float32)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            arr_np = librosa.resample(arr_np, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        sf.write(str(wav_path), arr_np, sr)
        
        # Save metadata
        rec = {
            "wav_path": str(wav_path),
            "sentence": sentence.strip() if sentence else "",
            "duration": round(duration, 3),
            "source": "voa_myanmar",
        }
        meta_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta_file.flush()
        
        count += 1
        total_dur += duration
        
        # Free memory
        del arr, arr_np, audio_data, ex
        
        if (count - existing_count) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {count:>6} samples | {total_dur/3600:.2f}h | {errors} errors | {elapsed:.0f}s")
        
        if total_dur / 3600 >= TARGET_HOURS:
            print(f"\n  Reached target: {total_dur/3600:.2f}h")
            break
        if count >= MAX_SAMPLES:
            print(f"\n  Reached max samples: {MAX_SAMPLES}")
            break
            
    except Exception as e:
        errors += 1
        if errors <= 10:
            print(f"  ⚠️ Error on sample {i}: {e}")
        continue

meta_file.close()
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Downloaded {count:,} samples | {total_dur/3600:.2f}h | {errors} errors | {elapsed:.0f}s")

# ── Build HF DatasetDict from saved WAV files ────────────────────────────────
print(f"\nBuilding HuggingFace dataset from {count} WAV files...")

rows = []
with open(META_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if os.path.exists(rec["wav_path"]):
            rows.append({
                "audio": rec["wav_path"],
                "sentence": rec["sentence"],
                "source": rec["source"],
                "speaker_id": "voa_unknown",
                "locale": "my",
            })

print(f"  {len(rows)} valid samples")

# Build in chunks to control memory
CHUNK = 3000
from datasets import concatenate_datasets

chunks = []
for start in range(0, len(rows), CHUNK):
    batch = rows[start:start+CHUNK]
    ds_chunk = Dataset.from_list(batch)
    ds_chunk = ds_chunk.cast_column("audio", Audio(sampling_rate=16000))
    chunks.append(ds_chunk)
    print(f"  Chunk {start//CHUNK + 1}: {len(batch)} samples")

ds = concatenate_datasets(chunks)

# 90/5/5 split
print("Splitting 90/5/5 train/val/test...")
split1 = ds.train_test_split(test_size=0.10, seed=42)
test_val = split1["test"].train_test_split(test_size=0.50, seed=42)

dd = DatasetDict({
    "train": split1["train"],
    "validation": test_val["test"],
    "test": test_val["train"],
})

dd.save_to_disk(SAVE_PATH)

print(f"\n✅ Saved VOA Myanmar dataset → {SAVE_PATH}")
for name, split in dd.items():
    print(f"  {name:12s}: {len(split):>6,} samples")
print(f"  Total hours: {total_dur/3600:.2f}h")