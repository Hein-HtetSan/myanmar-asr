from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h")

def is_valid(example):
    audio    = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    text     = example["sentence"].strip()
    return (
        0.5 <= duration <= 30.0 and
        len(text) >= 2 and
        len(text) <= 500 and
        not text.isspace()
    )

original_sizes = {k: len(v) for k, v in ds.items()}

print("Filtering bad samples...")
ds_filtered = ds.filter(is_valid, num_proc=8)  # Uses all M5 efficiency cores

print("\n🧹 Filtering Results:")
for split in ds_filtered:
    removed = original_sizes[split] - len(ds_filtered[split])
    print(f"  {split}: removed {removed} bad samples → {len(ds_filtered[split]):,} remaining")

ds_filtered.save_to_disk("combined/myanmar_asr_50h_clean")
print("✅ Saved filtered dataset to combined/myanmar_asr_50h_clean")