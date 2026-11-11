# Combined Myanmar speech dataset

Generated 2026-11-09T14:55:01.961413+00:00 from the local files listed in `summary.json`.

## Contents and scope

The complete catalog has **37,487 records / 94.373 hours** before combined curation. There are **37,482 unique encoded audio files / 94.370 hours** when each byte-identical file is counted once. Unique inventory hours still include recordings quarantined for conflicting labels; use the curated manifests for training.

The catalog includes 26,656 labeled records (64.330 hours) and 10,831 unlabeled records (30.043 hours). VOA is audio-only, and its original random train/test assignments are recorded as provenance rather than supervised evaluation splits.

This is an audio-backed CSV catalog. It does not copy or concatenate waveforms, and it is not a self-contained Hugging Face DatasetDict. `audio_data/` and `raw/` are symlinks to the original audio root and downloaded datasets. Keep their targets available. For transfer to Windows, copy those source directories and recreate the links or equivalent directories under this catalog.

## Sources

### asr

Local MyanmarASRScriptsAndData speech, with supplied Kaldi transcripts and speaker labels. Existing train/dev/test/user-study assignments are preserved.

- 7,879 records; 21.813 hours; 7,879 labeled records.
- Duration: min 0.885 s, median 9.107 s, mean 9.967 s, p95 18.376 s, max 47.912 s.
- Longer than 30 s: 23 records / 0.219 hours.
- Sample rates (Hz → clips): {"16000": 7879}; channels: {"1": 7879}.
- Encoding subtypes: {"PCM_16": 7879}.
- Unique normalized transcripts: 7,787; mean transcript length 194.3 Unicode characters, max 930.

### fleurs

Downloaded google/fleurs my_mm. Original train/validation/test splits are retained. Text uses the supplied transcription column, which can differ from raw_transcription in punctuation and numbers. Speaker identities are not supplied in this catalog.

- 4,322 records; 17.592 hours; 4,322 labeled records.
- Duration: min 3.660 s, median 14.160 s, mean 14.654 s, p95 23.160 s, max 41.640 s.
- Longer than 30 s: 32 records / 0.295 hours.
- Sample rates (Hz → clips): {"16000": 4322}; channels: {"1": 4322}.
- Encoding subtypes: {"FLOAT": 4322}.
- Unique normalized transcripts: 1,986; mean transcript length 154.0 Unicode characters, max 425.

### ucsy

Human speech referenced by the UCSY master workbook through the existing asr_work mappings. Recorder names inferred from filenames are not verified speaker identities. Existing record/text grouping is retained.

- 11,870 records; 22.569 hours; 11,870 labeled records.
- Duration: min 0.697 s, median 5.360 s, mean 6.845 s, p95 15.813 s, max 65.120 s.
- Longer than 30 s: 195 records / 2.094 hours.
- Sample rates (Hz → clips): {"48000": 2044, "16000": 9383, "8000": 330, "44100": 112, "10000": 1}; channels: {"2": 1640, "1": 10230}.
- Encoding subtypes: {"PCM_16": 11869, "PCM_24": 1}.
- Unique normalized transcripts: 4,343; mean transcript length 83.2 Unicode characters, max 737.

### voa_myanmar

Downloaded freococo/voa_myanmar_asr_audio_1 subset, capped at approximately 30 hours. No transcripts are provided. Use for labeling or unsupervised work, not supervised ASR training or WER/CER evaluation.

- 10,831 records; 30.043 hours; 0 labeled records.
- Duration: min 0.034 s, median 9.993 s, mean 9.986 s, p95 14.538 s, max 15.008 s.
- Longer than 30 s: 0 records / 0.000 hours.
- Sample rates (Hz → clips): {"16000": 10831}; channels: {"1": 10831}.
- Encoding subtypes: {"PCM_16": 10831}.
- Unique normalized transcripts: 0; mean transcript length 0.0 Unicode characters, max 0.

### yodas_my

Downloaded espnet/yodas my000 YouTube captioned speech. Captions are not guaranteed human transcriptions or verified Burmese. The downloader split utterances randomly; inferred video IDs reveal possible video overlap across splits. Utterance IDs are not speakers.

- 2,585 records; 2.356 hours; 2,585 labeled records.
- Duration: min 0.500 s, median 2.930 s, mean 3.282 s, p95 6.990 s, max 25.070 s.
- Longer than 30 s: 0 records / 0.000 hours.
- Sample rates (Hz → clips): {"16000": 2585}; channels: {"1": 2585}.
- Encoding subtypes: {"PCM_16": 2585}.
- Unique normalized transcripts: 2,267; mean transcript length 31.7 Unicode characters, max 219.

## Curated splits

The curated training pool has **22,011 clips / 52.845 hours** before the 30-second duration view. Validation and each named test remain separate. No random global re-split is performed.

- `test_asr.csv`: 193 clips / 0.532 hours.
- `test_fleurs.csv`: 880 clips / 3.814 hours.
- `test_ucsy.csv`: 1,186 clips / 2.051 hours.
- `test_user_study.csv`: 154 clips / 0.513 hours.
- `test_yodas_my.csv`: 129 clips / 0.119 hours.
- `train.csv`: 22,011 clips / 52.845 hours.
- `unlabeled.csv`: 10,831 clips / 30.043 hours.
- `validation.csv`: 1,912 clips / 4.272 hours.

Duration-only Whisper views (labeled clips at most 30 seconds):

- `test_asr_whisper.csv`: 193 clips / 0.532 hours.
- `test_fleurs_whisper.csv`: 870 clips / 3.717 hours.
- `test_ucsy_whisper.csv`: 1,171 clips / 1.903 hours.
- `test_user_study_whisper.csv`: 153 clips / 0.504 hours.
- `test_yodas_my_whisper.csv`: 129 clips / 0.119 hours.
- `train_whisper.csv`: 21,805 clips / 50.679 hours.
- `validation_whisper.csv`: 1,894 clips / 4.084 hours.

## Curation and leakage audit

The source CSVs are read without modification. The combined catalog recalculates audio headers and hashes. Byte-identical audio with conflicting nonempty normalized transcripts is quarantined. Other exact duplicates prefer labeled over unlabeled records and test over validation over training. Training records whose NFC/whitespace-free text appears in retained evaluation records are excluded. Evaluation-to-evaluation text repeats remain for audit; they do not constitute independent text coverage.

Combined exclusions by reason: `{"train_eval_normalized_text_overlap": 185, "identical_audio_conflicting_transcripts": 6}`.

Retained normalized transcript overlap across splits (distinct texts per pair): `{"test_yodas_my / validation": 5}`.

Retained record/video overlap across splits (distinct groups per pair): `{"test_yodas_my / train": 20, "test_yodas_my / validation": 17, "train / validation": 27}`.

Retained supplied/inferred speaker-label overlap (distinct labels per pair): `{"test_ucsy / train": 21, "test_ucsy / validation": 20, "train / validation": 20}`. Unknown and utterance-ID pseudo-speakers are omitted from that check; inferred UCSY labels remain uncertain.

These checks do not detect re-encoded or near-duplicate audio, prove that speech matches text, establish language correctness, or guarantee speaker/video-disjoint evaluation. YODAS video overlap requires a new grouped split before claiming video-disjoint scores.

## Files and audio access

- `all.csv`: every input record, including inclusion flags and exclusion reasons.
- `excluded.csv`: records removed by combined curation; original audio is retained.
- `unlabeled.csv`: retained unlabeled audio, excluded from all supervised views.
- `train.csv`, `validation.csv`, and `test_*.csv`: curated labeled views.
- `*_whisper.csv`: duration-only subsets of those labeled views.
- `summary.json`: totals, source/split cross-tabs, metadata distributions, and overlap counts.
- `input_provenance.json`: paths and SHA-256 of local mappings and saved-dataset metadata.

`audio_storage=file` resolves `audio` relative to this folder. `audio_storage=hf_dataset` resolves `audio_dataset`, `audio_split`, and zero-based `audio_index` to the audio bytes already embedded in the saved dataset. Use the included Python reader; the `audio` field is empty for embedded-audio references.

```python
import csv
from scripts.data.build_combined_catalog import read_audio
root = "combined/myanmar_asr_all"
with open(f"{root}/train.csv", encoding="utf-8", newline="") as handle:
    row = next(csv.DictReader(handle))
waveform, sample_rate = read_audio(row, root)
print(waveform.shape, sample_rate, row["text"])
```

Run the example from the repository root with datasets and SoundFile installed. Audio is returned in float32 at its original sample rate/channel layout. Downmix and resample in your trainer. The Whisper Small notebook at notebook/whisper-small/train.ipynb consumes these views directly using the shared reference-aware reader, without a duplicate waveform cache.

## Measurement and limitations

Durations are measured from audio header frames divided by sample rate, not download progress or filenames. Totals are summed before rounding; p95 uses the nearest-rank definition. Encoded byte totals count each catalog record and are not a measurement of physical disk use. Text character counts are Unicode code points, not words, graphemes, or tokenizer tokens. Whisper views check duration only; the model token limit and minimum usable duration still require validation. No augmentation, transcript generation, silence detection, manual listening, or model evaluation is performed.

The original asr_work exclusions and seed are embedded in summary.json. Those exclusions are outside this catalog; synthetic gemini audio and alternate audioNew copies remain excluded. A source label is provenance, not a quality guarantee.

The source cards declare [FLEURS](https://huggingface.co/datasets/google/fleurs) as CC BY 4.0, [YODAS](https://huggingface.co/datasets/espnet/yodas) as CC BY 3.0, and [VOA](https://huggingface.co/datasets/freococo/voa_myanmar_asr_audio_1) as PDDL. The local ASR/UCSY mappings do not establish a redistribution license. These materials therefore do not acquire a single combined license; retain source attribution and confirm permissions before redistribution.
