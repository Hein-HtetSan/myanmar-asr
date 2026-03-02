#!/usr/bin/env python3
"""
Build Myanmar ASR dataset on Vast.ai from publicly available HF sources.

Sources (no VOA — public only):
  1. FLEURS Myanmar      (~3,938 samples, ~8h)
  2. OpenSLR-80           (~2,530 samples, ~10h)
  3. YODAS Myanmar        (~2,585 samples, ~5h)

Pipeline: Download → Normalize → Combine → Clean → Augment (1.1x speed) → Save

Usage (on Vast.ai):
    python3 /workspace/scripts/utils/build_dataset_full.py
"""

import os
import sys
import time
import numpy as np

SAVE_PATH = "/workspace/data/myanmar_asr"
SAMPLING_RATE = 16000
SCHEMA = ["audio", "sentence", "source", "speaker_id", "locale"]
SPEED_FACTOR = 1.1  # single speed augmentation


def main():
    from datasets import (
        load_dataset, concatenate_datasets,
        DatasetDict, Dataset, Audio,
    )

    print("=" * 60)
    print("  MYANMAR ASR — DATASET BUILD (FLEURS + OpenSLR + YODAS)")
    print("=" * 60)
    t0 = time.time()

    # ── Phase 1: Download ─────────────────────────────────────

    print("\n[PHASE 1] Downloading datasets from HuggingFace Hub\n")

    # 1. FLEURS Myanmar
    print("[1/3] Loading FLEURS Myanmar...")
    fl = None
    try:
        fl = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
        print(f"  ✓ FLEURS: {', '.join(f'{s}={len(fl[s])}' for s in fl)}")
    except Exception as e:
        print(f"  ✗ FLEURS failed: {e}")

    # 2. OpenSLR-80
    print("[2/3] Loading OpenSLR-80...")
    slr = None
    try:
        slr = load_dataset("openslr", "SLR80", trust_remote_code=True)
        print(f"  ✓ OpenSLR-80: {', '.join(f'{s}={len(slr[s])}' for s in slr)}")
    except Exception as e:
        print(f"  ✗ OpenSLR-80 failed: {e}")

    # 3. YODAS Myanmar
    print("[3/3] Loading YODAS Myanmar...")
    yodas = None
    try:
        yodas = load_dataset("espnet/yodas", "my000", trust_remote_code=True)
        print(f"  ✓ YODAS: {', '.join(f'{s}={len(yodas[s])}' for s in yodas)}")
    except Exception as e:
        print(f"  ✗ YODAS failed: {e}")
        for config in ["my", "my_mm", "mya"]:
            try:
                yodas = load_dataset("espnet/yodas", config, trust_remote_code=True)
                print(f"  ✓ YODAS ({config}): {', '.join(f'{s}={len(yodas[s])}' for s in yodas)}")
                break
            except:
                continue

    dl_time = time.time() - t0
    print(f"\n  Downloads complete in {dl_time/60:.1f} min")

    # ── Phase 2: Normalize to common schema ────────────────────

    print("\n[PHASE 2] Normalizing schemas\n")

    def normalize_fleurs(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if "transcription" in d.column_names:
            d = d.rename_column("transcription", "sentence")
        elif "raw_transcription" in d.column_names:
            d = d.rename_column("raw_transcription", "sentence")
        d = d.map(lambda x: {
            "source": "fleurs",
            "speaker_id": str(x.get("id", "unknown")),
            "locale": "my",
        }, num_proc=1)
        d = d.select_columns(SCHEMA)
        return d

    def normalize_openslr(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if "sentence" not in d.column_names and "text" in d.column_names:
            d = d.rename_column("text", "sentence")
        d = d.map(lambda x: {
            "source": "openslr80",
            "speaker_id": "unknown",
            "locale": "my",
        }, num_proc=1)
        d = d.select_columns(SCHEMA)
        return d

    def normalize_yodas(ds, split):
        if ds is None or split not in ds:
            return None
        d = ds[split]
        d = d.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if "sentence" not in d.column_names:
            for candidate in ["text", "transcription", "transcript"]:
                if candidate in d.column_names:
                    d = d.rename_column(candidate, "sentence")
                    break
        d = d.map(lambda x: {
            "source": "yodas_my",
            "speaker_id": str(x.get("utt_id", x.get("id", "unknown"))),
            "locale": "my",
        }, num_proc=1)
        d = d.select_columns(SCHEMA)
        return d

    # ── Phase 3: Combine ─────────────────────────────────────

    print("[PHASE 3] Combining datasets\n")

    sources = [
        ("fleurs",   normalize_fleurs,  fl),
        ("openslr",  normalize_openslr, slr),
        ("yodas",    normalize_yodas,   yodas),
    ]

    all_train, all_val, all_test = [], [], []

    for name, norm_fn, ds in sources:
        for split_name, target in [("train", all_train), ("validation", all_val), ("test", all_test)]:
            d = norm_fn(ds, split_name)
            if d is not None and len(d) > 0:
                print(f"  {name}/{split_name}: {len(d):,} samples")
                target.append(d)

    if not all_train:
        print("ERROR: No training data!")
        sys.exit(1)

    train = concatenate_datasets(all_train)
    val = concatenate_datasets(all_val) if all_val else None
    test = concatenate_datasets(all_test) if all_test else None

    # Create val/test from train if missing
    if val is None or len(val) < 100:
        print("  Creating validation split from train (5%)...")
        sp = train.train_test_split(test_size=0.05, seed=42)
        train, val = sp["train"], sp["test"]

    if test is None or len(test) < 50:
        print("  Creating test split from train (5%)...")
        sp = train.train_test_split(test_size=0.05, seed=42)
        train, test = sp["train"], sp["test"]

    print(f"\n  Combined: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    # ── Phase 4: Clean ────────────────────────────────────────

    print("\n[PHASE 4] Cleaning dataset\n")

    def is_valid(example):
        # Text checks
        s = example.get("sentence", "")
        if not s or len(s.strip()) < 2:
            return False
        if len(s.strip()) > 500:
            return False

        # Audio checks
        audio = example.get("audio", {})
        if audio is None:
            return False
        arr = audio.get("array", None)
        if arr is None or len(arr) < 800:  # < 0.05s
            return False

        duration = len(arr) / SAMPLING_RATE
        if duration < 0.5 or duration > 30.0:
            return False

        # Silence check (very low energy)
        rms = np.sqrt(np.mean(np.array(arr, dtype=np.float32) ** 2))
        if rms < 1e-5:
            return False

        return True

    for split_name, ds_split in [("train", train), ("val", val), ("test", test)]:
        before = len(ds_split)
        ds_split = ds_split.filter(is_valid, num_proc=4)
        after = len(ds_split)
        removed = before - after
        pct = (removed / before * 100) if before else 0
        print(f"  {split_name}: {before:,} → {after:,} ({removed} removed, {pct:.1f}%)")
        if split_name == "train":
            train = ds_split
        elif split_name == "val":
            val = ds_split
        else:
            test = ds_split

    # ── Phase 5: Augment training data (1.1x speed) ──────────

    print(f"\n[PHASE 5] Augmenting training data ({SPEED_FACTOR}x speed perturbation)\n")

    try:
        import librosa
        import soundfile as sf

        speed_label = f"sp{SPEED_FACTOR:.1f}".replace(".", "")
        aug_dir = "/workspace/data/augmented_audio"
        os.makedirs(aug_dir, exist_ok=True)

        original_len = len(train)
        aug_rows = []
        errors = 0
        t_aug = time.time()

        for idx in range(original_len):
            try:
                example = train[idx]
                audio_data = example["audio"]
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]

                # Speed perturbation: resample to change tempo without pitch shift
                new_sr = int(sr * SPEED_FACTOR)
                augmented = librosa.resample(arr, orig_sr=new_sr, target_sr=sr)

                # Save augmented audio
                out_path = os.path.join(aug_dir, f"{speed_label}_{idx:06d}.wav")
                sf.write(out_path, augmented, sr)

                aug_rows.append({
                    "audio": out_path,
                    "sentence": example["sentence"],
                    "source": example["source"] + f"_{speed_label}",
                    "speaker_id": example["speaker_id"],
                    "locale": example["locale"],
                })

                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - t_aug
                    rate = (idx + 1) / elapsed
                    eta = (original_len - idx - 1) / rate / 60
                    print(f"    {idx+1}/{original_len} ({rate:.0f}/s, ETA {eta:.1f}m)")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    ⚠ Error idx={idx}: {e}")

        aug_time = time.time() - t_aug
        print(f"  ✓ Generated {len(aug_rows):,} augmented samples ({errors} errors) in {aug_time/60:.1f}m")

        if aug_rows:
            aug_ds = Dataset.from_list(aug_rows)
            aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
            train = concatenate_datasets([train, aug_ds])
            train = train.shuffle(seed=42)
            print(f"  Final train: {len(train):,} (original {original_len:,} + augmented {len(aug_rows):,})")

    except ImportError:
        print("  ⚠ librosa/soundfile not available — skipping augmentation")
        print("    Install with: pip install librosa soundfile")

    # ── Phase 6: Save final dataset ───────────────────────────

    print(f"\n[PHASE 6] Saving to {SAVE_PATH}\n")

    combined = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    # Compute total hours
    total_hours = 0
    for split_name in combined:
        split_ds = combined[split_name]
        n = len(split_ds)
        sample_n = min(200, n)
        total_dur = 0
        for i in range(sample_n):
            audio = split_ds[i]["audio"]
            total_dur += len(audio["array"]) / audio["sampling_rate"]
        avg_dur = total_dur / sample_n if sample_n > 0 else 0
        est_hours = (avg_dur * n) / 3600
        total_hours += est_hours
        print(f"  {split_name:12s}: {n:>6,} samples | avg {avg_dur:.1f}s | ~{est_hours:.1f}h")

    print(f"  {'TOTAL':12s}: {sum(len(combined[s]) for s in combined):>6,} samples | ~{total_hours:.1f}h")

    # Backup old dataset
    if os.path.exists(SAVE_PATH):
        import shutil
        backup = SAVE_PATH + "_old"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.rename(SAVE_PATH, backup)
        print(f"  Old dataset backed up to {backup}")

    combined.save_to_disk(SAVE_PATH)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DATASET BUILD COMPLETE")
    print(f"  Path: {SAVE_PATH}")
    print(f"  Total: {sum(len(combined[s]) for s in combined):,} samples (~{total_hours:.1f}h)")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
