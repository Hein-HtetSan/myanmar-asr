from datasets import load_dataset

print("Downloading FLEURS my_mm...")
ds = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
ds.save_to_disk("raw/fleurs_my_mm")
print("Saved FLEURS to raw/fleurs_my_mm")

for split in ds:
    # Counting rows does not require an audio decoder or loading every waveform.
    print(f"  {split}: {len(ds[split])} samples")
