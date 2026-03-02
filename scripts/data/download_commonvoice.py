# ─────────────────────────────────────────────────────────────────────────────
# Common Voice Burmese replacement:
# Mozilla removed all CV datasets from HuggingFace in October 2025.
#
# Replacement: YODAS (espnet/yodas) — YouTube-derived Burmese ASR dataset
#   • subset "my000" — ~2.4 hrs of manually-captioned Burmese speech
#   • Free, CC-BY-3.0, directly loadable from HuggingFace
#   • Utterance-level segments, 16kHz, with transcriptions
# ─────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset, DatasetDict

SAVE_PATH = "raw/common_voice_my"

print("Downloading YODAS Burmese (my000) — replaces Common Voice...")
print("Source: espnet/yodas | subset: my000 | ~2.4 hrs")

ds_raw = load_dataset("espnet/yodas", "my000", trust_remote_code=True)

# Normalize to the same schema used by the rest of the pipeline:
# audio, sentence, source, speaker_id, locale
def normalize(example):
    return {
        "audio":      example["audio"],
        "sentence":   example["text"].strip(),
        "source":     "yodas_my",
        "speaker_id": example.get("utt_id", "unknown"),
        "locale":     "my",
    }

keep_cols = set(ds_raw["train"].column_names) - {"audio"}
ds_norm = ds_raw.map(normalize, remove_columns=list(keep_cols))

# YODAS only has a "train" split — carve out validation/test (90/5/5)
full = ds_norm["train"].shuffle(seed=42)
n = len(full)
n_val  = max(1, int(n * 0.05))
n_test = max(1, int(n * 0.05))

combined = DatasetDict({
    "train":      full.select(range(n_val + n_test, n)),
    "validation": full.select(range(n_val)),
    "test":       full.select(range(n_val, n_val + n_test)),
})

combined.save_to_disk(SAVE_PATH)

print(f"\n✅ Saved YODAS Burmese → {SAVE_PATH}")
for split, d in combined.items():
    hrs = sum(len(a["array"]) / a["sampling_rate"] for a in d["audio"]) / 3600
    print(f"  {split:12s}: {len(d):>5,} samples | {hrs:.2f} hrs")