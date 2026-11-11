# Historical Myanmar ASR training mappings

The current [Whisper Small workflow](../whisper-small/README.md) reads `combined/myanmar_asr_all/` directly. This directory documents older local mappings. The `prepare_asr.py` generator mentioned below belongs to the original external audio collection and is not included in this repository. Use [the combined-catalog builder](../../scripts/data/README.md#combine-downloads-with-the-original-audio-data-mappings) for the current workflow.

Generated from the original collections by `python3 prepare_asr.py` in the parent directory. The script uses only Python's standard library and `ffprobe` for WAV variants that Python cannot read. Run it again after source files change. It overwrites the generated CSV and JSON files here; it does not alter source WAVs or workbooks.

## Use the manifests

All `audio` paths are relative to the **audio-data root**, one directory above this file. When moving to the RTX 5060 Ti machine, copy `asr_work/`, `MyanmarASRScriptsAndData/waves_data/`, `MyanmarASRScriptsAndData/waves_data5/`, and `NLP Project1/audio_files/` under the same parent directory. The unused `audioNew/` and `gemini/` folders are not needed for this baseline. Each row has a transcription in `text`, an audio path, source and speaker grouping, duration, sample rate, channel count, and SHA-256 hash. Read the CSV as UTF-8.

| File | Role |
| --- | --- |
| `train.csv` | Combined natural-speech training pool. |
| `validation.csv` | ASR dev plus UCSY validation; use for model selection. |
| `test_asr.csv` | Original ASR test; keep held out until final evaluation. |
| `test_user_study.csv` | Original user-study test; keep held out. |
| `test_ucsy.csv` | Held-out UCSY human speech. |
| `*_whisper.csv` | Same split with clips longer than 30 seconds omitted. |
| `whisper_long_clips.csv` | Omitted long clips, for later alignment or long-form evaluation. |
| `exclusions.csv` | Every omitted workbook reference or rejected audio candidate and its reason. |
| `summary.json` | Current counts, hours, and exclusion totals. |

For a first Whisper experiment, use `train_whisper.csv` and `validation_whisper.csv`. Read each WAV from the `audio` path and resample to **16 kHz mono in the data loader**; the original files have mixed rates and channels. Do not truncate a clip while retaining its full transcript. The full manifests remain available for models or pipelines that support longer audio.

The `text` column preserves the supplied wording, with Unicode NFC and repeated whitespace normalized. The ASR source has spaces between many Myanmar words while UCSY text uses a different spacing style. Pin a single scoring normalization before comparing character error rates across sources.

The UCSY split is deterministic (seed `20261030`). All recordings of one master-sheet row, including question and answer sides, stay together. Rows with identical transcriptions stay together as well. Exact duplicate audio is kept once. The ASR train/dev speaker overlap and train/evaluation text overlap are removed from training. `audioNew/` is an alternate copy of ASR audio and is not included. `gemini/` speech is synthetic and is reserved for a separate experiment; it is not in any training, validation, or test manifest.

These are **metadata-checked candidates**. The script checks file presence, WAV metadata, and exact audio hashes; it does not listen to clips, verify that speech matches each transcript, or detect near-duplicate recordings. The UCSY filename prefix is an inferred recorder label, not a verified speaker identity. Review a representative sample with a Myanmar-fluent listener before relying on final accuracy numbers. The source files do not include a clear training or redistribution license.
