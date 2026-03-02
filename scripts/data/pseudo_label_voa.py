"""
pseudo_label_voa.py
Generate transcriptions for VOA Myanmar audio using OpenAI Whisper.

Uses Apple Silicon MPS GPU for acceleration.
Saves transcriptions incrementally to JSONL (resumable).
"""

import json
import os
import time
from pathlib import Path

import torch
import whisper

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "medium"              # Good Burmese support, faster than large-v3
AUDIO_DIR  = Path("raw/voa_myanmar_wav")
META_IN    = Path("raw/voa_myanmar_meta.jsonl")
META_OUT   = Path("raw/voa_myanmar_meta_labeled.jsonl")
LANGUAGE   = "my"
LOG_EVERY  = 50

# ── Check for resume ────────────────────────────────────────────────────────
done_set = set()
if META_OUT.exists():
    with open(META_OUT) as f:
        for line in f:
            rec = json.loads(line)
            done_set.add(rec["wav_path"])
    print(f"Resuming: {len(done_set)} already labeled")

# ── Load metadata ────────────────────────────────────────────────────────────
all_records = []
with open(META_IN) as f:
    for line in f:
        rec = json.loads(line)
        all_records.append(rec)

todo = [r for r in all_records if r["wav_path"] not in done_set]
print(f"Total: {len(all_records)}, Already done: {len(done_set)}, Remaining: {len(todo)}")

if not todo:
    print("All done!")
    exit(0)

# ── Load model ───────────────────────────────────────────────────────────────
device = "cpu"  # openai-whisper on MPS can be buggy; CPU is reliable
print(f"Loading Whisper {MODEL_NAME} on {device}...")
t0 = time.time()
model = whisper.load_model(MODEL_NAME, device=device)
print(f"Model loaded in {time.time()-t0:.1f}s")

# ── Transcribe ───────────────────────────────────────────────────────────────
out_file = open(META_OUT, "a")
labeled = len(done_set)
errors = 0
t0 = time.time()

for i, rec in enumerate(todo):
    wav_path = rec["wav_path"]
    try:
        if not os.path.exists(wav_path):
            errors += 1
            continue

        result = model.transcribe(
            wav_path,
            language=LANGUAGE,
            task="transcribe",
            fp16=False,
        )
        text = result["text"].strip()

        out_rec = {
            "wav_path": wav_path,
            "sentence": text,
            "duration": rec["duration"],
            "source": rec["source"],
        }
        out_file.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        out_file.flush()
        labeled += 1

        if (i + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(todo) - i - 1) / rate / 60
            print(f"  [{labeled}/{len(all_records)}] {elapsed/60:.1f}m | {rate:.2f} samp/s | ETA {eta_min:.0f}m | {text[:60]}")

    except Exception as e:
        errors += 1
        # Still write with empty sentence so we don't re-try
        out_rec = {
            "wav_path": wav_path,
            "sentence": "",
            "duration": rec["duration"],
            "source": rec["source"],
        }
        out_file.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        out_file.flush()
        labeled += 1
        if errors <= 10:
            print(f"  ⚠️ Error {wav_path}: {e}")

out_file.close()
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Done: {labeled} labeled | {errors} errors | {elapsed/60:.1f} min")

# ── Stats ────────────────────────────────────────────────────────────────────
empty = 0
with_text = 0
with open(META_OUT) as f:
    for line in f:
        r = json.loads(line)
        if r["sentence"].strip():
            with_text += 1
        else:
            empty += 1
print(f"With text: {with_text}/{with_text+empty} ({100*with_text/(with_text+empty):.1f}%)")
