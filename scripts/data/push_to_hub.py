#!/usr/bin/env python3
"""
Push Myanmar ASR datasets to HuggingFace Hub (private).

Usage:
    python3 scripts/push_to_hub.py               # Push augmented (default)
    python3 scripts/push_to_hub.py --clean        # Push clean (non-augmented)
    python3 scripts/push_to_hub.py --both         # Push both
"""

import argparse
from datasets import load_from_disk


def push_dataset(path, repo_id):
    print(f"\nLoading {path}...")
    ds = load_from_disk(path)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} samples")

    print(f"\nPushing to {repo_id} (private)...")
    ds.push_to_hub(repo_id, private=True, max_shard_size="500MB")
    print(f"✅ https://huggingface.co/datasets/{repo_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", action="store_true", help="Push clean dataset only")
    p.add_argument("--both",  action="store_true", help="Push both clean and augmented")
    args = p.parse_args()

    if args.both:
        push_dataset("combined/myanmar_asr_50h_clean", "devhnhts/myanmar-asr-50h-clean")
        push_dataset("combined/myanmar_asr_augmented",  "devhnhts/myanmar-asr-augmented")
    elif args.clean:
        push_dataset("combined/myanmar_asr_50h_clean", "devhnhts/myanmar-asr-50h-clean")
    else:
        push_dataset("combined/myanmar_asr_augmented",  "devhnhts/myanmar-asr-augmented")


if __name__ == "__main__":
    main()
