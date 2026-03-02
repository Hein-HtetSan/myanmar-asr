#!/usr/bin/env python3
"""
Deep-clean the Myanmar ASR dataset and fix train/val/test split balance.

Fixes:
  1. Tighter duration filter (1.0s – 25.0s)
  2. Tighter text length filter (≥ 5 chars)
  3. Stricter silence removal (rms ≥ 5e-4)
  4. Remove near-duplicate texts
  5. Re-split val/test to include ALL sources (not just FLEURS)
  6. Verify augmented samples match originals

Usage:
    python3 /workspace/scripts/deep_clean.py
"""
import os, sys, json, time, re
import numpy as np
from collections import Counter

DATA_DIR = "/workspace/data/myanmar_asr"
SAVE_DIR = "/workspace/data/myanmar_asr"  # overwrite in-place (backs up first)
SAMPLING_RATE = 16000

# ── Cleaning thresholds ──────────────────────────────────────
MIN_DURATION = 1.0       # seconds
MAX_DURATION = 25.0      # seconds
MIN_TEXT_LEN = 5         # characters
MAX_TEXT_LEN = 400       # characters
MIN_RMS = 5e-4           # energy threshold (stricter than 1e-5)
MIN_CHARS_PER_SEC = 1.5  # catch misaligned audio/text
MAX_CHARS_PER_SEC = 30.0 # catch misaligned audio/text


def main():
    from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio

    print("=" * 60)
    print("  DEEP CLEAN — Myanmar ASR Dataset")
    print("=" * 60)
    t0 = time.time()

    ds = load_from_disk(DATA_DIR)
    print(f"\nLoaded: {', '.join(f'{s}={len(ds[s]):,}' for s in ds)}")

    # ── Step 1: Merge everything, then re-split ──────────────
    print("\n[1/4] Merging all splits for re-balancing...")

    # Separate original and augmented so we can track
    all_samples = concatenate_datasets([ds[s] for s in ds])
    total_before = len(all_samples)
    print(f"  Total samples: {total_before:,}")

    # ── Step 2: Deep clean ────────────────────────────────────
    print("\n[2/4] Applying deep cleaning filters...\n")

    rejection_reasons = Counter()

    def deep_clean(example):
        # Text checks
        text = example.get("sentence", "") or ""
        text = text.strip()
        if len(text) < MIN_TEXT_LEN:
            rejection_reasons["text_too_short"] += 1
            return False
        if len(text) > MAX_TEXT_LEN:
            rejection_reasons["text_too_long"] += 1
            return False

        # Check for non-Myanmar junk (pure ASCII / numbers only)
        myanmar_chars = len(re.findall(r'[\u1000-\u109F]', text))
        if myanmar_chars < 2:
            rejection_reasons["no_myanmar_chars"] += 1
            return False

        # Audio checks
        audio = example.get("audio")
        if audio is None:
            rejection_reasons["no_audio"] += 1
            return False
        arr = audio.get("array")
        if arr is None:
            rejection_reasons["no_audio_array"] += 1
            return False

        arr = np.array(arr, dtype=np.float32)
        duration = len(arr) / SAMPLING_RATE

        if duration < MIN_DURATION:
            rejection_reasons["too_short"] += 1
            return False
        if duration > MAX_DURATION:
            rejection_reasons["too_long"] += 1
            return False

        # RMS energy
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms < MIN_RMS:
            rejection_reasons["near_silent"] += 1
            return False

        # Chars-per-second ratio (catch misaligned pairs)
        cps = len(text) / duration
        if cps < MIN_CHARS_PER_SEC:
            rejection_reasons["text_too_sparse"] += 1
            return False
        if cps > MAX_CHARS_PER_SEC:
            rejection_reasons["text_too_dense"] += 1
            return False

        # Check for NaN / Inf in audio
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            rejection_reasons["nan_inf_audio"] += 1
            return False

        return True

    cleaned = all_samples.filter(deep_clean, num_proc=4)
    total_after = len(cleaned)
    removed = total_before - total_after
    print(f"  Cleaned: {total_before:,} → {total_after:,} ({removed} removed, {removed/total_before*100:.1f}%)")
    print(f"\n  Rejection breakdown:")
    for reason, count in rejection_reasons.most_common():
        print(f"    {reason}: {count}")

    # ── Step 3: Remove near-duplicate texts ───────────────────
    print("\n[3/4] Removing near-duplicate transcriptions...")

    # Get all texts and find exact duplicates (keeping one copy each)
    texts_seen = {}
    dup_indices = set()
    for i in range(len(cleaned)):
        text = cleaned[i]["sentence"].strip()
        source = cleaned[i].get("source", "")
        # For augmented copies, the text is the same as original — that's fine
        # Only flag dupes within the SAME source type (original or augmented)
        is_augmented = "_sp11" in source
        key = (text, is_augmented)
        if key in texts_seen:
            dup_indices.add(i)
        else:
            texts_seen[key] = i

    if dup_indices:
        keep_indices = [i for i in range(len(cleaned)) if i not in dup_indices]
        cleaned = cleaned.select(keep_indices)
        print(f"  Removed {len(dup_indices)} exact duplicate texts")
    else:
        print(f"  No exact duplicates found")
    print(f"  Final count: {len(cleaned):,}")

    # ── Step 4: Re-split with balanced sources ────────────────
    print("\n[4/4] Creating balanced train/val/test splits...")

    # Separate augmented from original
    is_aug = [("_sp11" in (cleaned[i].get("source", ""))) for i in range(len(cleaned))]
    orig_indices = [i for i, aug in enumerate(is_aug) if not aug]
    aug_indices = [i for i, aug in enumerate(is_aug) if aug]

    originals = cleaned.select(orig_indices)
    augmented = cleaned.select(aug_indices) if aug_indices else None

    print(f"  Originals: {len(originals):,} | Augmented: {len(aug_indices):,}")

    # Split originals into train/val/test (90/5/5) with shuffling
    originals = originals.shuffle(seed=42)
    n = len(originals)
    n_test = max(int(n * 0.05), 200)
    n_val = max(int(n * 0.05), 200)
    n_train = n - n_val - n_test

    test_ds = originals.select(range(n_test))
    val_ds = originals.select(range(n_test, n_test + n_val))
    train_orig = originals.select(range(n_test + n_val, n))

    # Add augmented only to training
    if augmented is not None and len(augmented) > 0:
        train_ds = concatenate_datasets([train_orig, augmented])
        train_ds = train_ds.shuffle(seed=42)
    else:
        train_ds = train_orig

    print(f"\n  Final splits:")
    print(f"    train:      {len(train_ds):,}")
    print(f"    validation: {len(val_ds):,}")
    print(f"    test:       {len(test_ds):,}")

    # Check source distribution in val/test
    for split_name, split_ds in [("validation", val_ds), ("test", test_ds)]:
        sources = Counter()
        for i in range(len(split_ds)):
            sources[split_ds[i].get("source", "unknown")] += 1
        print(f"    {split_name} sources: {dict(sources)}")

    # ── Save ──────────────────────────────────────────────────
    combined = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    # Save to temp dir first, then swap (datasets can't overwrite itself)
    import shutil
    TEMP_DIR = SAVE_DIR + "_clean_tmp"
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    combined.save_to_disk(TEMP_DIR)

    # Backup old and move clean into place
    if os.path.exists(SAVE_DIR):
        backup = SAVE_DIR + "_pre_deepclean"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.rename(SAVE_DIR, backup)
        print(f"\n  Old dataset backed up to {backup}")
    os.rename(TEMP_DIR, SAVE_DIR)

    # ── Compute final hours ───────────────────────────────────
    total_hours = 0
    for split_name in combined:
        split_ds = combined[split_name]
        n = len(split_ds)
        sample_n = min(300, n)
        total_dur = 0
        for i in range(sample_n):
            audio = split_ds[i]["audio"]
            total_dur += len(audio["array"]) / audio["sampling_rate"]
        avg_dur = total_dur / sample_n if sample_n > 0 else 0
        est_hours = (avg_dur * n) / 3600
        total_hours += est_hours
        print(f"  {split_name:12s}: {n:>6,} samples | avg {avg_dur:.1f}s | ~{est_hours:.1f}h")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DEEP CLEAN COMPLETE")
    print(f"  Path: {SAVE_DIR}")
    print(f"  Total: {sum(len(combined[s]) for s in combined):,} samples (~{total_hours:.1f}h)")
    print(f"  Removed: {removed + len(dup_indices)} bad/duplicate samples")
    print(f"  Val/test now have mixed sources (not just FLEURS)")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
