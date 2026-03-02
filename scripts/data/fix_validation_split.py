#!/usr/bin/env python3
"""
Fix the undersized validation split (129 → ~630 samples).

Problem:  Cleaning removed most validation samples (671 → 129).
          Only yodas_my remains — no fleurs or openslr80 validation data.

Solution: Move ~500 original (non-augmented) train samples into validation,
          balanced across sources. Remove those + their augmented copies
          from training to prevent data leakage.

Run:
    python3 scripts/fix_validation_split.py

Outputs:
    combined/myanmar_asr_augmented/   (updated in-place, backup created first)
"""

import os
import shutil
import collections
from datasets import load_from_disk, DatasetDict, concatenate_datasets

DS_PATH = "combined/myanmar_asr_augmented"
BACKUP_PATH = "combined/myanmar_asr_augmented_backup"
VAL_SAMPLES_PER_SOURCE = 170  # ~170 × 3 sources = ~510 new val samples
SEED = 42


def main():
    print("=" * 60)
    print("  Fix Validation Split — Myanmar ASR")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────
    print(f"\nLoading {DS_PATH}...")
    ds = load_from_disk(DS_PATH)

    for split in ds:
        sources = collections.Counter(ds[split]["source"])
        print(f"  {split}: {len(ds[split]):,} samples — {dict(sources)}")

    # ── Identify original (non-augmented) sources ─────
    # Augmented sources have _sp09 or _sp11 suffix
    original_sources = sorted(set(
        s for s in ds["train"]["source"]
        if not s.endswith("_sp09") and not s.endswith("_sp11")
    ))
    print(f"\nOriginal sources in train: {original_sources}")

    # ── Select samples to move from train → val ──────
    train = ds["train"]
    move_indices = []

    for source in original_sources:
        # Get indices for this source
        source_indices = [i for i, s in enumerate(train["source"]) if s == source]
        print(f"  {source}: {len(source_indices)} original samples")

        # Shuffle deterministically and take VAL_SAMPLES_PER_SOURCE
        import random
        rng = random.Random(SEED)
        rng.shuffle(source_indices)
        n_move = min(VAL_SAMPLES_PER_SOURCE, len(source_indices))
        move_indices.extend(source_indices[:n_move])

    print(f"\nMoving {len(move_indices)} original samples from train → val")

    # ── Also find their augmented copies ──────────────
    # For each original sample at index i, its augmented copies are at
    # different indices with source = "{original_source}_sp09" and "_sp11"
    # We need to identify them by matching the sentence text
    move_set = set(move_indices)
    move_sentences = set(train[i]["sentence"] for i in move_indices)

    # Find augmented copies by matching sentences from moved originals
    augmented_remove = []
    for i in range(len(train)):
        source = train[i]["source"]
        if (source.endswith("_sp09") or source.endswith("_sp11")):
            if train[i]["sentence"] in move_sentences:
                augmented_remove.append(i)

    total_remove = set(move_indices) | set(augmented_remove)
    print(f"  Also removing {len(augmented_remove)} augmented copies")
    print(f"  Total removed from train: {len(total_remove)}")

    # ── Build new splits ──────────────────────────────
    keep_indices = sorted(set(range(len(train))) - total_remove)
    new_val_indices = sorted(move_indices)

    new_train = train.select(keep_indices)
    new_val_from_train = train.select(new_val_indices)

    # Combine with existing validation
    if len(ds["validation"]) > 0:
        new_val = concatenate_datasets([ds["validation"], new_val_from_train])
    else:
        new_val = new_val_from_train

    # ── Create new DatasetDict ────────────────────────
    new_ds = DatasetDict({
        "train": new_train,
        "validation": new_val,
        "test": ds["test"],
    })

    print(f"\n{'─' * 40}")
    print("New split sizes:")
    for split in new_ds:
        sources = collections.Counter(new_ds[split]["source"])
        print(f"  {split}: {len(new_ds[split]):,} samples")
        for s, c in sorted(sources.items()):
            print(f"    {s}: {c}")

    # ── Backup and save ───────────────────────────────
    TEMP_PATH = DS_PATH + "_fixed"

    if os.path.exists(BACKUP_PATH):
        print(f"\n  Backup already exists at {BACKUP_PATH}")
    else:
        print(f"\n  Backing up → {BACKUP_PATH}")
        shutil.copytree(DS_PATH, BACKUP_PATH)

    print(f"  Saving fixed dataset → {TEMP_PATH}")
    new_ds.save_to_disk(TEMP_PATH)

    # Replace original with fixed version
    shutil.rmtree(DS_PATH)
    shutil.move(TEMP_PATH, DS_PATH)
    print(f"  Moved → {DS_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Validation fixed: {len(ds['validation'])} → {len(new_ds['validation'])} samples")
    print(f"  ✅ Train adjusted:   {len(ds['train'])} → {len(new_ds['train'])} samples")
    print(f"  ✅ Test unchanged:   {len(new_ds['test'])} samples")
    print(f"{'=' * 60}")
    print(f"\n  Next: Re-export NeMo manifests:")
    print(f"    python3 scripts/export_nemo_manifest.py")


if __name__ == "__main__":
    main()
