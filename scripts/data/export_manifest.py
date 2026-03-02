import json
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h_clean")

for split in ds:
    out_path = f"combined/{split}_manifest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split]):
            row = {
                "id":         f"{split}_{i:06d}",
                "sentence":   ex["sentence"],
                "source":     ex["source"],
                "speaker_id": ex["speaker_id"],
                "duration":   round(len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"], 2),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ {split}_manifest.jsonl → {i+1:,} rows")