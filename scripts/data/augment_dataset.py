"""
augment_dataset.py
Apply data augmentation to the clean Myanmar ASR dataset to reach ~50h.

Augmentation techniques:
  1. Speed perturbation (0.9x and 1.1x) — doubles the data with valid labels
  2. The original data is kept as-is

Schema preserved: audio, sentence, source, speaker_id, locale
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict, Audio, concatenate_datasets
import librosa
import time

CLEAN_PATH = "combined/myanmar_asr_50h_clean"
OUTPUT_PATH = "combined/myanmar_asr_augmented"
SPEED_FACTORS = [0.9, 1.1]  # Will create 2 additional copies

print("=" * 60)
print("  MYANMAR ASR — DATA AUGMENTATION")
print("=" * 60)

# ── Load clean dataset ───────────────────────────────────────────────────────
print("\nLoading clean dataset...")
ds = load_from_disk(CLEAN_PATH)
for split_name, split_ds in ds.items():
    print(f"  {split_name}: {len(split_ds):,} samples")

# ── Only augment training data ───────────────────────────────────────────────
train_ds = ds["train"]
original_len = len(train_ds)
print(f"\nAugmenting train split ({original_len:,} samples)...")

# Create temp dir for augmented audio
aug_dir = Path("raw/augmented_audio")
aug_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
all_augmented = []

for speed_factor in SPEED_FACTORS:
    speed_label = f"sp{speed_factor:.1f}".replace(".", "")
    print(f"\n  Speed perturbation {speed_factor}x...")
    
    aug_rows = []
    errors = 0
    
    # Only augment ORIGINAL samples, not previously augmented ones
    for idx in range(original_len):
        try:
            example = train_ds[idx]
            audio_data = example["audio"]
            arr = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]
            
            # Apply speed perturbation via resampling
            new_sr = int(sr * speed_factor)
            augmented = librosa.resample(arr, orig_sr=new_sr, target_sr=sr)
            
            # Save augmented audio
            out_path = aug_dir / f"{speed_label}_{idx:06d}.wav"
            sf.write(str(out_path), augmented, sr)
            
            aug_rows.append({
                "audio": str(out_path),
                "sentence": example["sentence"],
                "source": example["source"] + f"_{speed_label}",
                "speaker_id": example["speaker_id"],
                "locale": example["locale"],
            })
            
            if (idx + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"    {idx+1}/{original_len} | {elapsed:.0f}s")
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    ⚠️ Error idx={idx}: {e}")
    
    print(f"    Generated {len(aug_rows):,} augmented samples ({errors} errors)")
    
    if aug_rows:
        aug_ds = Dataset.from_list(aug_rows)
        aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=16000))
        all_augmented.append(aug_ds)

# Combine: original + all augmented
all_parts = [train_ds] + all_augmented
train_ds = concatenate_datasets(all_parts)
print(f"\n  Final train: {len(train_ds):,} samples (original {original_len} + augmented {len(train_ds)-original_len})")

# ── Shuffle augmented training set ──────────────────────────────────────────
print(f"\nShuffling augmented training set ({len(train_ds):,} samples)...")
train_ds = train_ds.shuffle(seed=42)

# ── Build final DatasetDict ─────────────────────────────────────────────────
augmented = DatasetDict({
    "train": train_ds,
    "validation": ds["validation"] if "validation" in ds else None,
    "test": ds["test"] if "test" in ds else None,
})
# Remove None splits
augmented = DatasetDict({k: v for k, v in augmented.items() if v is not None})

# ── Compute total hours ─────────────────────────────────────────────────────
total_hours = 0
print(f"\n{'='*60}")
print("AUGMENTED DATASET SUMMARY:")
for split_name, split_ds in augmented.items():
    # Sample duration calculation
    total_dur = 0
    for i in range(len(split_ds)):
        audio = split_ds[i]["audio"]
        total_dur += len(audio["array"]) / audio["sampling_rate"]
        if i >= 499 and split_name == "train":
            # Estimate from sample for large splits
            avg_dur = total_dur / (i + 1)
            total_dur = avg_dur * len(split_ds)
            break
    hours = total_dur / 3600
    total_hours += hours
    print(f"  {split_name:12s}: {len(split_ds):>6,} samples | ~{hours:.2f}h")

print(f"  {'TOTAL':12s}: {sum(len(s) for s in augmented.values()):>6,} samples | ~{total_hours:.2f}h")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_PATH}...")
augmented.save_to_disk(OUTPUT_PATH)
print(f"✅ Saved augmented dataset → {OUTPUT_PATH}")
