# Combined datasets

Generated CSV catalogs and saved datasets live here and remain local. The [current catalog report](myanmar_asr_all/README.md) is included for researchers; CSVs, audio links, and runtime artifacts are ignored by Git.

Build a catalog from the repository root:

```sh
python scripts/data/build_combined_catalog.py --audio-data-root /path/to/audio-data
```

See [data preparation](../scripts/data/README.md) for required downloaded inputs and [Whisper Small](../notebook/whisper-small/README.md) for full supervised training. The builder refuses to overwrite existing snapshots. Keep referenced audio directories available; CSV metadata alone cannot train a model.
