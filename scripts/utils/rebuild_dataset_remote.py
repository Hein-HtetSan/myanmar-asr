#!/usr/bin/env python3
"""
Reconstruct the Myanmar ASR augmented dataset on a remote GPU server
by downloading original datasets from HuggingFace Hub.

This avoids slow uploads from local machines.
Runs on the Vast.ai instance directly.
"""
import os
import sys

def main():
    from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
    import numpy as np

    save_path = "/workspace/data/myanmar_asr"

    # Check if already exists
    if os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        from datasets import load_from_disk
        ds = load_from_disk(save_path)
        total = sum(len(ds[s]) for s in ds)
        if total > 1000:
            print(f"Dataset already exists at {save_path} ({total} total samples)")
            for s in ds:
                print(f"  {s}: {len(ds[s]):,}")
            return

    print("=== Downloading datasets from HuggingFace Hub ===\n")

    SCHEMA = ["audio", "sentence", "source", "speaker_id", "locale"]

    # ── 1. Common Voice Myanmar ──
    print("[1/3] Loading Common Voice Myanmar...")
    cv = None
    for cv_version in ["mozilla-foundation/common_voice_17_0",
                       "mozilla-foundation/common_voice_16_1",
                       "mozilla-foundation/common_voice_13_0"]:
        try:
            cv = load_dataset(cv_version, "my", trust_remote_code=True)
            print(f"  {cv_version}: {', '.join(f'{s}={len(cv[s])}' for s in cv)}")
            break
        except Exception as e:
            print(f"  {cv_version} failed: {e}")
    if cv is None:
        print("  WARNING: All Common Voice versions failed. Skipping.")

    # ── 2. FLEURS Myanmar ──
    print("[2/3] Loading FLEURS Myanmar...")
    fl = None
    try:
        fl = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
        print(f"  FLEURS: {', '.join(f'{s}={len(fl[s])}' for s in fl)}")
    except Exception as e:
        print(f"  FLEURS via google/fleurs failed: {e}")
        try:
            fl = load_dataset("google/xtreme_s", "fleurs.my_mm", trust_remote_code=True)
            print(f"  FLEURS via xtreme_s: {', '.join(f'{s}={len(fl[s])}' for s in fl)}")
        except Exception as e2:
            print(f"  FLEURS all methods failed: {e2}")

    # ── 3. OpenSLR-80 ──
    print("[3/3] Loading OpenSLR-80...")
    slr = None
    try:
        slr = load_dataset("openslr", "SLR80", trust_remote_code=True)
        print(f"  OpenSLR-80: {', '.join(f'{s}={len(slr[s])}' for s in slr)}")
    except Exception as e:
        print(f"  OpenSLR-80 failed: {e}")
        try:
            # Try the direct URL approach
            slr = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
        except:
            pass

    # ── Normalize to common schema ──
    print("\n=== Normalizing schemas ===\n")

    def normalize_cv(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=16000))
        d = d.rename_column("sentence", "sentence")
        d = d.map(lambda x: {
            "source": "common_voice",
            "speaker_id": x.get("client_id", "unknown"),
            "locale": "my",
        })
        d = d.select_columns(SCHEMA)
        return d

    def normalize_fl(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=16000))
        d = d.rename_column("transcription", "sentence") if "transcription" in d.column_names else d
        if "raw_transcription" in d.column_names and "sentence" not in d.column_names:
            d = d.rename_column("raw_transcription", "sentence")
        d = d.map(lambda x: {
            "source": "fleurs",
            "speaker_id": str(x.get("id", "unknown")),
            "locale": "my",
        })
        d = d.select_columns(SCHEMA)
        return d

    def normalize_slr(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=16000))
        if "sentence" not in d.column_names and "text" in d.column_names:
            d = d.rename_column("text", "sentence")
        d = d.map(lambda x: {
            "source": "openslr80",
            "speaker_id": "unknown",
            "locale": "my",
        })
        d = d.select_columns(SCHEMA)
        return d

    # ── Combine splits ──
    print("=== Combining datasets ===\n")
    all_train = []
    all_val = []
    all_test = []

    for name, norm_fn, ds in [("cv", normalize_cv, cv), ("fl", normalize_fl, fl), ("slr", normalize_slr, slr)]:
        for split_name, target_list in [("train", all_train), ("validation", all_val), ("test", all_test)]:
            d = norm_fn(ds, split_name)
            if d is not None:
                print(f"  {name}/{split_name}: {len(d)} samples")
                target_list.append(d)

    if not all_train:
        print("ERROR: No training data found!")
        sys.exit(1)

    train = concatenate_datasets(all_train) if all_train else None
    val = concatenate_datasets(all_val) if all_val else None
    test = concatenate_datasets(all_test) if all_test else None

    # If no validation, split from train
    if val is None or len(val) < 100:
        print("  Creating validation split from train (5%)...")
        split = train.train_test_split(test_size=0.05, seed=42)
        train = split["train"]
        val = split["test"]

    combined = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    print(f"\n=== Combined dataset ===")
    for s in combined:
        print(f"  {s}: {len(combined[s]):,}")

    # ── Filter bad samples ──
    print("\n=== Filtering ===")
    def is_valid(example):
        s = example.get("sentence", "")
        if not s or len(s.strip()) < 2:
            return False
        audio = example.get("audio", {})
        if audio is None:
            return False
        arr = audio.get("array", None)
        if arr is None or len(arr) < 1600:  # < 0.1s
            return False
        if len(arr) > 480000:  # > 30s
            return False
        return True

    for split in combined:
        before = len(combined[split])
        combined[split] = combined[split].filter(is_valid, num_proc=4)
        after = len(combined[split])
        print(f"  {split}: {before} -> {after} ({before - after} removed)")

    # ── Save ──
    print(f"\n=== Saving to {save_path} ===")
    combined.save_to_disk(save_path)

    total = sum(len(combined[s]) for s in combined)
    print(f"\nDone! Total: {total:,} samples")
    for s in combined:
        print(f"  {s}: {len(combined[s]):,}")


if __name__ == "__main__":
    main()
