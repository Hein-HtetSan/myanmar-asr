# Dataset preparation

Run commands from the repository root in your Python environment. These scripts use Hugging Face `datasets`, NumPy, SoundFile, and, for resampling or augmentation, librosa. Analysis also uses pandas and matplotlib. Downloads require network access; pseudo-labeling additionally uses PyTorch, `openai-whisper`, and FFmpeg. There is no single tested dependency lock for this entire directory.

Most scripts execute immediately when run or imported. Paths are generally constants inside each file, rather than command-line arguments.

## Prepare local FLEURS and OpenSLR-80 files

Arrange downloaded files under the repository root:

```text
raw/
  fleurs_my_mm/
    train.tsv
    validation.tsv
    test.tsv
    train/*.wav
    validation/*.wav
    test/*.wav
  openslr80/
    line_index.tsv
    *.wav
```

Then run:

```sh
python scripts/data/build_local_datasets.py
python scripts/data/combine_datasets.py
python scripts/data/filter_dataset.py
```

The builder writes `raw/fleurs_my_mm_hf/` and `raw/openslr80_hf/`. It skips missing FLEURS splits and creates a random 90/10 train/test split for OpenSLR; it does not create OpenSLR validation data or enforce speaker separation. The combiner includes whichever configured sources load successfully, so inspect its output for missing sources and splits. Filtering keeps clips from 0.5 to 30 seconds with stripped transcripts of 2–500 characters. The final output is `combined/myanmar_asr_50h_clean/`; the name does not guarantee 50 hours of audio.

The shared dataset schema is `audio`, `sentence`, `source`, `speaker_id`, and `locale`. Training scripts expect `train`, `validation`, and `test` splits.

## Script reference

- `download_fleurs.py`: downloads `google/fleurs`, configuration `my_mm`, to `raw/fleurs_my_mm/` as a saved Hugging Face dataset. Prints split counts without decoding audio, so the download summary does not require TorchCodec. This is a different format from the local WAV/TSV layout above. It does not produce the normalized `raw/fleurs_my_mm_hf/` expected by the combiner; normalize its schema and save to that path before combining. Do not mix the two layouts in one directory.
- `download_commonvoice.py`: despite its filename, downloads YODAS `espnet/yodas`, subset `my000`, directly from its audio archives and transcript files using `huggingface_hub`. It does not execute `yodas.py` or use `trust_remote_code`. Pairs audio/transcript IDs, normalizes the schema, and creates deterministic 90/5/5 utterance splits under `raw/common_voice_my/`. Downloads and summaries do not decode audio or require TorchCodec; later audio processing may. The `speaker_id` field contains an utterance ID, not a verified speaker identity; these splits do not ensure speaker/video separation. Run it before combining if you want that source included.
- `download_voa.py`: streams encoded VOA MP3s, decodes with SoundFile, writes 16 kHz mono WAVs and metadata under `raw/`, and builds `raw/voa_myanmar_hf/`. It does not require TorchCodec for downloading. Its configured limits are 30 hours or 15,000 samples; actual output depends on available data. This source is audio-only: `sentence` values remain empty until labels are supplied. Do not use it for supervised training or WER/CER evaluation without reviewed transcripts.
- `pseudo_label_voa.py`: uses Whisper medium to write resumable machine transcriptions to `raw/voa_myanmar_meta_labeled.jsonl`. Review its device selection before running outside Apple Silicon. This is a separate output; running it does not automatically rebuild the combined dataset with those labels.
- `normalize_datasets.py`: defines normalization helpers; running the file alone does not write a dataset.
- `build_local_datasets.py`: builds normalized datasets from the WAV/TSV layout above.
- `combine_datasets.py`: combines local OpenSLR, FLEURS, YODAS, and VOA datasets into `combined/myanmar_asr_50h/`.
- `filter_dataset.py`: filters the combined dataset into `combined/myanmar_asr_50h_clean/`, using eight worker processes.
- `augment_dataset.py`: adds 0.9x and 1.1x speed variants to training only, keeps evaluation splits, and writes `combined/myanmar_asr_augmented/` plus `raw/augmented_audio/`.
- `fix_validation_split.py`: historical split repair that moves selected originals and removes associated training variants. It replaces `combined/myanmar_asr_augmented/` and creates a backup if one does not already exist. Review the selection rules before applying it to a new corpus.
- `analyze_dataset.py`: writes plots and statistics to `viz/`. Accepts a dataset path as its first positional argument; otherwise chooses augmented data if present, then clean data.
- `export_manifest.py`: writes `combined/{split}_manifest.jsonl` from the clean dataset. These contain transcript metadata, but no audio path or exported audio.
- `export_nemo_manifest.py`: exports augmented data to `exports/nemo_audio/` and `exports/nemo_{train,val,test}_manifest.jsonl`. Manifest audio paths target `/workspace/data/nemo_audio/`; copy the WAVs there or change the exporter for your machine.
- `push_to_hub.py`: uploads augmented data by default; `--clean` selects clean data and `--both` uploads both. Destination repository IDs are hardcoded, and uploads are private. Set your destinations and authenticate before running.

Optional local steps after filtering:

```sh
python scripts/data/analyze_dataset.py combined/myanmar_asr_50h_clean
python scripts/data/augment_dataset.py
python scripts/data/export_manifest.py
```

## Combine downloads with the original audio-data mappings

`build_combined_catalog.py` reads the original `audio-data/asr_work/*.csv` mappings, downloaded FLEURS, YODAS, and VOA data. It measures audio headers, hashes encoded audio, preserves source split assignments, and writes a combined CSV catalog and detailed report without duplicating recordings:

```sh
python scripts/data/build_combined_catalog.py --audio-data-root /path/to/audio-data
```

Default output is `combined/myanmar_asr_all/`, including `all.csv`, curated supervised splits, duration-only Whisper views, `unlabeled.csv`, `excluded.csv`, `summary.json`, and `README.md`. Existing output is never overwritten; use `--output-dir` for another snapshot. The `audio_data/` and `raw/` symlinks must continue to resolve to their source directories. The catalog combines audio references and metadata, not copied WAVs or an embedded Hugging Face DatasetDict. Its Python `read_audio` helper handles both file paths and embedded dataset audio. See the generated README for usage, exact measurements, curation rules, and speaker/video overlap limitations.

VOA remains unlabeled. Synthetic speech and alternate audio copies excluded by the original mappings are not added. This catalog does not download OpenSLR; it includes the sources already present in the configured inputs.

## Using the Whisper Small notebook

These outputs are not the CSV manifests consumed by [the Whisper Small notebook](../../notebook/whisper-small/README.md). That notebook expects `sample_id`, `audio`, `source`, `duration_s`, `sha256`, and `text`, with audio paths relative to `dataset/`. An additional exporter and split/leakage checks are needed. `export_manifest.py` does not perform that conversion.

Keep evaluation splits held out, check duplicates and speaker grouping, and review audio/transcript pairs before training. Source availability and older `trust_remote_code` loaders depend on the installed library version; downloads have not been verified by this documentation. See [data provenance](../../docs/REPRODUCIBILITY.md).
