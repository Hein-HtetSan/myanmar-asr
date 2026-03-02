"""
build_local_datasets.py
Converts already-downloaded raw files into HuggingFace DatasetDict format.

Handles:
  - FLEURS my_mm  : raw/fleurs_my_mm/{train,test}/*.wav + train.tsv / test.tsv
  - OpenSLR-80    : raw/openslr80/*.wav + line_index.tsv
"""

import csv
import os
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

# ─────────────────────────────────────────────────────────────────────────────
#  FLEURS
# ─────────────────────────────────────────────────────────────────────────────
def build_fleurs():
    root = Path("raw/fleurs_my_mm")
    save_path = "raw/fleurs_my_mm_hf"

    splits = {}
    for split_name, tsv_name, audio_dir in [
        ("train",      "train.tsv",      root / "train"),
        ("test",       "test.tsv",       root / "test"),
        ("validation", "validation.tsv", root / "validation"),
    ]:
        tsv_path = root / tsv_name
        if not tsv_path.exists():
            print(f"  [FLEURS] {split_name}: TSV not found — skipping")
            continue
        if tsv_path.stat().st_size == 0:
            print(f"  [FLEURS] {split_name}: TSV is empty — skipping (audio not downloaded)")
            continue
        if not audio_dir.exists():
            print(f"  [FLEURS] {split_name}: audio folder missing — skipping")
            continue

        rows = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                fname       = parts[1].strip()           # e.g. 10286965.wav
                transcription = parts[2].strip()         # raw_transcription col
                wav_path = audio_dir / fname
                if not wav_path.exists():
                    continue
                rows.append({
                    "audio":      str(wav_path),
                    "sentence":   transcription,
                    "source":     "fleurs",
                    "speaker_id": parts[6].strip() if len(parts) > 6 else "unknown",
                    "locale":     "my",
                })

        if not rows:
            print(f"  [FLEURS] {split_name}: no matching WAV files found — skipping")
            continue

        ds = Dataset.from_list(rows)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        splits[split_name] = ds
        hours = len(ds) * 16000 / 16000 / 3600  # rough: avg ~7s = not exact
        print(f"  [FLEURS] {split_name}: {len(ds):,} samples")

    if not splits:
        print("  [FLEURS] No splits built — nothing saved.")
        return

    DatasetDict(splits).save_to_disk(save_path)
    print(f"  [FLEURS] ✅ Saved → {save_path}")
    print(f"           splits: {list(splits.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
#  OpenSLR-80
# ─────────────────────────────────────────────────────────────────────────────
def build_openslr():
    audio_dir = Path("raw/openslr80")
    tsv_path  = audio_dir / "line_index.tsv"
    save_path = "raw/openslr80_hf"

    if not tsv_path.exists():
        print("  [OpenSLR] line_index.tsv not found — skipping")
        return

    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            stem        = parts[0].strip()          # bur_XXXX_XXXXXXXXXX (no .wav)
            transcription = parts[1].strip()
            wav_path = audio_dir / f"{stem}.wav"
            if not wav_path.exists():
                continue
            # speaker_id = bur_XXXX part (second field)
            speaker_id = stem.split("_")[1] if "_" in stem else "unknown"
            rows.append({
                "audio":      str(wav_path),
                "sentence":   transcription,
                "source":     "openslr80",
                "speaker_id": speaker_id,
                "locale":     "my",
            })

    if not rows:
        print("  [OpenSLR] No matching WAV files — skipping")
        return

    print(f"  [OpenSLR] {len(rows):,} samples loaded")
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 90/10 train/test split (no pre-defined splits in OpenSLR-80)
    split = ds.train_test_split(test_size=0.1, seed=42)
    ddict = DatasetDict({"train": split["train"], "test": split["test"]})
    ddict.save_to_disk(save_path)
    print(f"  [OpenSLR] ✅ Saved → {save_path}")
    print(f"            train: {len(ddict['train']):,} | test: {len(ddict['test']):,}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building FLEURS dataset from local files...")
    build_fleurs()
    print()
    print("Building OpenSLR-80 dataset from local files...")
    build_openslr()
    print()
    print("Done. Next step: run scripts/combine_datasets.py")
