"""
combine_datasets.py
Merges all downloaded datasets (already normalized to common schema) into one.

Expected datasets — all must have columns: audio, sentence, source, speaker_id, locale
  raw/openslr80_hf      — train / test  (built by build_local_datasets.py)
  raw/fleurs_my_mm_hf   — train / test  (built by build_local_datasets.py)
  raw/common_voice_my   — train / validation / test
  raw/voa_myanmar_hf    — train / validation / test  (built by download_voa.py)
"""

from datasets import load_from_disk, concatenate_datasets, DatasetDict

SCHEMA = ["audio", "sentence", "source", "speaker_id", "locale"]

def load_split(path, split):
    """Load a single split; return None if path or split doesn't exist."""
    try:
        dd = load_from_disk(path)
        if split in dd:
            return dd[split]
    except Exception as e:
        print(f"  ⚠️  Could not load {path}/{split}: {e}")
    return None

print("Loading datasets...")
openslr_train = load_split("raw/openslr80_hf",    "train")
openslr_test  = load_split("raw/openslr80_hf",    "test")
fl_train      = load_split("raw/fleurs_my_mm_hf", "train")
fl_val        = load_split("raw/fleurs_my_mm_hf", "validation")
fl_test       = load_split("raw/fleurs_my_mm_hf", "test")
cv_train      = load_split("raw/common_voice_my",  "train")
cv_val        = load_split("raw/common_voice_my",  "validation")
cv_test       = load_split("raw/common_voice_my",  "test")
voa_train     = load_split("raw/voa_myanmar_hf",   "train")
voa_val       = load_split("raw/voa_myanmar_hf",   "validation")
voa_test      = load_split("raw/voa_myanmar_hf",   "test")

def merge(*splits):
    """Concatenate non-None splits."""
    valid = [s for s in splits if s is not None]
    if not valid:
        return None
    return concatenate_datasets(valid)

print("Merging...")
train_ds = merge(openslr_train, fl_train, cv_train, voa_train)
val_ds   = merge(fl_val, cv_val, voa_val)
test_ds  = merge(openslr_test, fl_test, cv_test, voa_test)

if train_ds is None:
    raise RuntimeError("No training data found — run build_local_datasets.py first.")

train_ds = train_ds.shuffle(seed=42)

splits = {"train": train_ds}
if val_ds:
    splits["validation"] = val_ds
if test_ds:
    splits["test"] = test_ds

combined = DatasetDict(splits)

print(f"\n✅ Combined Dataset:")
for name, ds in combined.items():
    print(f"  {name:12s}: {len(ds):>6,} samples")

import os
os.makedirs("combined", exist_ok=True)
combined.save_to_disk("combined/myanmar_asr_50h")
print("\n✅ Saved to combined/myanmar_asr_50h")