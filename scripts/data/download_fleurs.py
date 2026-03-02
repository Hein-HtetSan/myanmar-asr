from datasets import load_dataset

print("Downloading FLEURS my_mm...")
ds = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
ds.save_to_disk("raw/fleurs_my_mm")

for split in ds:
    hours = sum(len(a["array"]) / a["sampling_rate"] for a in ds[split]["audio"]) / 3600
    print(f"  {split}: {len(ds[split])} samples (~{hours:.1f} hrs)")