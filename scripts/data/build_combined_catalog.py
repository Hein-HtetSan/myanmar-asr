"""Combine local ASR mappings and downloaded speech without copying audio.

Run from the repository root. Audio references resolve through audio_data/ and
raw/ symlinks inside the output folder; keep their target directories available.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import unicodedata

SPLITS = ('train', 'validation', 'test_asr', 'test_user_study', 'test_ucsy')
FIELDS = ('sample_id', 'source', 'split', 'original_split', 'text', 'has_transcript',
          'audio_storage', 'audio', 'audio_dataset', 'audio_split', 'audio_index',
          'speaker_group', 'speaker_identity', 'record_group', 'duration_s',
          'sample_rate_hz', 'channels', 'frames', 'audio_format', 'audio_subtype',
          'encoded_bytes', 'sha256', 'included', 'exclusion_reason', 'whisper_duration_ok')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text or '').split())


def text_key(text):
    return ''.join(normalize(text).split())


def inspect_audio(audio):
    import soundfile as sf
    if isinstance(audio, bytes):
        info = sf.info(io.BytesIO(audio))
        digest = hashlib.sha256(audio).hexdigest()
        size = len(audio)
    else:
        info = sf.info(audio)
        h = hashlib.sha256()
        with open(audio, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        digest, size = h.hexdigest(), Path(audio).stat().st_size
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError('Empty audio')
    return dict(duration_s=info.frames / info.samplerate, sample_rate_hz=info.samplerate,
                channels=info.channels, frames=info.frames, audio_format=info.format,
                audio_subtype=info.subtype, encoded_bytes=size, sha256=digest)


def finish_row(row, audio):
    row['text'] = normalize(row.get('text', ''))
    row['has_transcript'] = int(bool(row['text']))
    row.update(included=1, exclusion_reason='')
    row.update(inspect_audio(audio))
    row['whisper_duration_ok'] = int(0 < row['duration_s'] <= 30)
    return {key: row.get(key, '') for key in FIELDS}


def local_rows(audio_root):
    for split in SPLITS:
        path = audio_root / 'asr_work' / f'{split}.csv'
        with path.open(encoding='utf-8', newline='') as handle:
            for original in csv.DictReader(handle):
                audio = (audio_root / original['audio']).resolve()
                if not audio.is_relative_to(audio_root.resolve()):
                    raise ValueError(f'Audio path escapes root: {audio}')
                row = dict(sample_id=original['sample_id'], source=original['source'],
                           split=split, original_split=split, text=original['text'],
                           audio_storage='file', audio=f"audio_data/{original['audio']}",
                           speaker_group=original['speaker_group'],
                           speaker_identity='supplied' if original['source'] == 'asr' else 'inferred_filename',
                           record_group=original['record_group'])
                yield finish_row(row, audio)


def downloaded_rows(raw_root):
    from datasets import Audio, load_from_disk
    for folder, source in (('fleurs_my_mm', 'fleurs'), ('common_voice_my', 'yodas_my'),
                           ('voa_myanmar_hf', 'voa_myanmar')):
        dataset = load_from_disk(str(raw_root / folder))
        for split, data in dataset.items():
            data = data.cast_column('audio', Audio(decode=False))
            for index, original in enumerate(data):
                encoded = original['audio']
                audio = encoded['bytes'] or encoded['path']
                if not isinstance(audio, bytes):
                    audio = Path(audio)
                speaker = original.get('speaker_id', '')
                row = dict(sample_id=f'{source}:{split}:{index}', source=source,
                           original_split=split,
                           split='unlabeled' if source == 'voa_myanmar' else
                                 f'test_{source}' if split == 'test' else split,
                           text=original.get('sentence', original.get('transcription', '')),
                           audio_storage='hf_dataset', audio_dataset=f'raw/{folder}',
                           audio_split=split, audio_index=index,
                           speaker_group=f'{source}:{speaker}' if speaker else '',
                           speaker_identity='utterance_id' if source == 'yodas_my' else 'unknown',
                           record_group=f'yodas_video:{speaker[:11]}' if source == 'yodas_my' else '')
                if source == 'voa_myanmar':
                    # Use the existing WAV when available, avoiding Arrow lookup during training.
                    filename = Path(encoded.get('path') or '').name
                    wav = raw_root / 'voa_myanmar_wav' / filename
                    if filename and wav.is_file():
                        row.update(audio_storage='file', audio=f'raw/voa_myanmar_wav/{filename}')
                yield finish_row(row, audio)
        print(f'Inspected {source}', flush=True)


def curate(rows):
    """Keep evaluation precedence; quarantine label conflicts and training overlap."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['sha256']].append(row)
    for group in by_hash.values():
        labeled_texts = {text_key(row['text']) for row in group if row['has_transcript']}
        if len(labeled_texts) > 1:
            for row in group:
                row.update(included=0, exclusion_reason='identical_audio_conflicting_transcripts')
            continue
        ordered = sorted(group, key=lambda row: (
            not bool(row['has_transcript']),
            0 if row['split'].startswith('test_') else 1 if row['split'] == 'validation' else 2,
            row['sample_id'],
        ))
        for row in ordered[1:]:
            row.update(included=0, exclusion_reason='identical_audio_duplicate')
    heldout = {text_key(row['text']) for row in rows
               if row['included'] and row['has_transcript'] and row['split'] != 'train'}
    for row in rows:
        if row['included'] and row['split'] == 'train' and text_key(row['text']) in heldout:
            row.update(included=0, exclusion_reason='train_eval_normalized_text_overlap')


def describe(rows):
    if not rows:
        return {'clips': 0, 'hours': 0}
    durations = sorted(row['duration_s'] for row in rows)
    texts = [row['text'] for row in rows if row['has_transcript']]
    lengths = [len(text) for text in texts]
    seconds = sum(durations)
    return dict(clips=len(rows), seconds=seconds, hours=seconds / 3600,
                labeled_clips=len(texts), labeled_hours=sum(r['duration_s'] for r in rows if r['has_transcript']) / 3600,
                unlabeled_clips=len(rows)-len(texts),
                duration_min_s=durations[0], duration_max_s=durations[-1],
                duration_mean_s=statistics.mean(durations), duration_median_s=statistics.median(durations),
                duration_p95_s=durations[max(0, (95 * len(durations) + 99) // 100 - 1)],
                clips_over_30s=sum(d > 30 for d in durations),
                hours_over_30s=sum(d for d in durations if d > 30) / 3600,
                clips_under_half_second=sum(d < .5 for d in durations),
                sample_rates_hz=dict(Counter(str(row['sample_rate_hz']) for row in rows)),
                channels=dict(Counter(str(row['channels']) for row in rows)),
                formats=dict(Counter(row['audio_format'] for row in rows)),
                subtypes=dict(Counter(row['audio_subtype'] for row in rows)),
                encoded_audio_bytes=sum(row['encoded_bytes'] for row in rows),
                unique_audio_hashes=len({row['sha256'] for row in rows}),
                unique_normalized_transcripts=len({text_key(text) for text in texts}),
                transcript_mean_characters=statistics.mean(lengths) if lengths else 0,
                transcript_max_characters=max(lengths, default=0))


def grouped_stats(rows, key):
    return {value: describe([r for r in rows if r[key] == value])
            for value in sorted({r[key] for r in rows})}


def overlaps(rows, key):
    groups = defaultdict(set)
    for row in rows:
        value = text_key(row['text']) if key == 'text' else row[key]
        if value:
            groups[value].add(row['split'])
    pairs = Counter()
    for splits in groups.values():
        ordered = sorted(splits)
        for i, first in enumerate(ordered):
            for second in ordered[i+1:]:
                pairs[f'{first} / {second}'] += 1
    return dict(sorted(pairs.items()))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=16)
def encoded_dataset(path, split):
    from datasets import Audio, load_from_disk
    return load_from_disk(path)[split].cast_column('audio', Audio(decode=False))


def read_audio(row, catalog_root):
    """Return (float32 waveform, sample_rate), preserving original channels/rate."""
    import soundfile as sf
    root = Path(catalog_root)
    if row['audio_storage'] == 'file':
        source = str(root / row['audio'])
    elif row['audio_storage'] == 'hf_dataset':
        data = encoded_dataset(str((root / row['audio_dataset']).resolve()), row['audio_split'])
        audio = data[int(row['audio_index'])]['audio']
        source = io.BytesIO(audio['bytes']) if audio['bytes'] else audio['path']
    else:
        raise ValueError(f"Unknown audio storage: {row['audio_storage']}")
    return sf.read(source, dtype='float32')


def link_directory(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to replace {path}')
    path.symlink_to(target.resolve(), target_is_directory=True)


def write_report(path, summary):
    s = summary
    total, unique = s['inventory'], s['unique_inventory']
    train = s['curated_by_split'].get('train', {'clips': 0, 'hours': 0})
    lines = [
        '# Combined Myanmar speech dataset', '',
        f"Generated {s['generated_at']} from the local files listed in `summary.json`.", '',
        '## Contents and scope', '',
        f"The complete catalog has **{total['clips']:,} records / {total['hours']:.3f} hours** before combined curation. "
        f"There are **{unique['clips']:,} unique encoded audio files / {unique['hours']:.3f} hours** when each byte-identical file is counted once. "
        'Unique inventory hours still include recordings quarantined for conflicting labels; use the curated manifests for training.', '',
        f"The catalog includes {total['labeled_clips']:,} labeled records ({total['labeled_hours']:.3f} hours) and "
        f"{total['unlabeled_clips']:,} unlabeled records ({total['hours']-total['labeled_hours']:.3f} hours). "
        'VOA is audio-only, and its original random train/test assignments are recorded as provenance rather than supervised evaluation splits.', '',
        'This is an audio-backed CSV catalog. It does not copy or concatenate waveforms, and it is not a self-contained Hugging Face DatasetDict. '
        '`audio_data/` and `raw/` are symlinks to the original audio root and downloaded datasets. Keep their targets available. '
        'For transfer to Windows, copy those source directories and recreate the links or equivalent directories under this catalog.', '',
        '## Sources', '',
    ]
    descriptions = {
        'asr': 'Local MyanmarASRScriptsAndData speech, with supplied Kaldi transcripts and speaker labels. Existing train/dev/test/user-study assignments are preserved.',
        'ucsy': 'Human speech referenced by the UCSY master workbook through the existing asr_work mappings. Recorder names inferred from filenames are not verified speaker identities. Existing record/text grouping is retained.',
        'fleurs': 'Downloaded google/fleurs my_mm. Original train/validation/test splits are retained. Text uses the supplied transcription column, which can differ from raw_transcription in punctuation and numbers. Speaker identities are not supplied in this catalog.',
        'yodas_my': 'Downloaded espnet/yodas my000 YouTube captioned speech. Captions are not guaranteed human transcriptions or verified Burmese. The downloader split utterances randomly; inferred video IDs reveal possible video overlap across splits. Utterance IDs are not speakers.',
        'voa_myanmar': 'Downloaded freococo/voa_myanmar_asr_audio_1 subset, capped at approximately 30 hours. No transcripts are provided. Use for labeling or unsupervised work, not supervised ASR training or WER/CER evaluation.',
    }
    for source, stats in s['by_source'].items():
        lines += [f'### {source}', '', descriptions.get(source, ''), '',
                  f"- {stats['clips']:,} records; {stats['hours']:.3f} hours; {stats['labeled_clips']:,} labeled records.",
                  f"- Duration: min {stats['duration_min_s']:.3f} s, median {stats['duration_median_s']:.3f} s, "
                  f"mean {stats['duration_mean_s']:.3f} s, p95 {stats['duration_p95_s']:.3f} s, max {stats['duration_max_s']:.3f} s.",
                  f"- Longer than 30 s: {stats['clips_over_30s']:,} records / {stats['hours_over_30s']:.3f} hours.",
                  f"- Sample rates (Hz → clips): {json.dumps(stats['sample_rates_hz'])}; channels: {json.dumps(stats['channels'])}.",
                  f"- Encoding subtypes: {json.dumps(stats['subtypes'])}.",
                  f"- Unique normalized transcripts: {stats['unique_normalized_transcripts']:,}; mean transcript length "
                  f"{stats['transcript_mean_characters']:.1f} Unicode characters, max {stats['transcript_max_characters']:,}.", '']
    lines += ['## Curated splits', '',
              f"The curated training pool has **{train['clips']:,} clips / {train['hours']:.3f} hours** before the 30-second duration view. "
              'Validation and each named test remain separate. No random global re-split is performed.', '']
    for split, stats in s['curated_by_split'].items():
        lines += [f"- `{split}.csv`: {stats['clips']:,} clips / {stats['hours']:.3f} hours."]
    lines += ['', 'Duration-only Whisper views (labeled clips at most 30 seconds):', '']
    for split, stats in s['whisper_by_split'].items():
        lines += [f"- `{split}_whisper.csv`: {stats['clips']:,} clips / {stats['hours']:.3f} hours."]
    lines += ['', '## Curation and leakage audit', '',
              'The source CSVs are read without modification. The combined catalog recalculates audio headers and hashes. '
              'Byte-identical audio with conflicting nonempty normalized transcripts is quarantined. Other exact duplicates '
              'prefer labeled over unlabeled records and test over validation over training. '
              'Training records whose NFC/whitespace-free text appears in retained evaluation records are excluded. '
              'Evaluation-to-evaluation text repeats remain for audit; they do not constitute independent text coverage.', '',
              f"Combined exclusions by reason: `{json.dumps(s['combined_exclusions'])}`.", '',
              f"Retained normalized transcript overlap across splits (distinct texts per pair): `{json.dumps(s['curated_overlap']['text'])}`.", '',
              f"Retained record/video overlap across splits (distinct groups per pair): `{json.dumps(s['curated_overlap']['record_group'])}`.", '',
              f"Retained supplied/inferred speaker-label overlap (distinct labels per pair): `{json.dumps(s['speaker_overlap'])}`. "
              'Unknown and utterance-ID pseudo-speakers are omitted from that check; inferred UCSY labels remain uncertain.', '',
              'These checks do not detect re-encoded or near-duplicate audio, prove that speech matches text, establish language correctness, '
              'or guarantee speaker/video-disjoint evaluation. YODAS video overlap requires a new grouped split before claiming video-disjoint scores.', '',
              '## Files and audio access', '',
              '- `all.csv`: every input record, including inclusion flags and exclusion reasons.',
              '- `excluded.csv`: records removed by combined curation; original audio is retained.',
              '- `unlabeled.csv`: retained unlabeled audio, excluded from all supervised views.',
              '- `train.csv`, `validation.csv`, and `test_*.csv`: curated labeled views.',
              '- `*_whisper.csv`: duration-only subsets of those labeled views.',
              '- `summary.json`: totals, source/split cross-tabs, metadata distributions, and overlap counts.',
              '- `input_provenance.json`: paths and SHA-256 of local mappings and saved-dataset metadata.', '',
              '`audio_storage=file` resolves `audio` relative to this folder. `audio_storage=hf_dataset` resolves '
              '`audio_dataset`, `audio_split`, and zero-based `audio_index` to the audio bytes already embedded in the saved dataset. '
              'Use the included Python reader; the `audio` field is empty for embedded-audio references.', '',
              '```python', 'import csv', 'from scripts.data.build_combined_catalog import read_audio',
              'root = "combined/myanmar_asr_all"',
              'with open(f"{root}/train.csv", encoding="utf-8", newline="") as handle:',
              '    row = next(csv.DictReader(handle))',
              'waveform, sample_rate = read_audio(row, root)', 'print(waveform.shape, sample_rate, row["text"])', '```', '',
              'Run the example from the repository root with datasets and SoundFile installed. Audio is returned in float32 '
              'at its original sample rate/channel layout. Downmix and resample in your trainer. '
              'The Whisper Small notebook at notebook/whisper-small/train.ipynb consumes these views directly using the shared reference-aware reader, without a duplicate waveform cache.', '',
              '## Measurement and limitations', '',
              'Durations are measured from audio header frames divided by sample rate, not download progress or filenames. '
              'Totals are summed before rounding; p95 uses the nearest-rank definition. Encoded byte totals count each catalog record '
              'and are not a measurement of physical disk use. Text character counts are Unicode code points, not words, graphemes, or tokenizer tokens. '
              'Whisper views check duration only; the model token limit and minimum usable duration still require validation. '
              'No augmentation, transcript generation, silence detection, manual listening, or model evaluation is performed.', '',
              'The original asr_work exclusions and seed are embedded in summary.json. Those exclusions are outside this catalog; '
              'synthetic gemini audio and alternate audioNew copies remain excluded. A source label is provenance, not a quality guarantee.', '',
              'The source cards declare [FLEURS](https://huggingface.co/datasets/google/fleurs) as CC BY 4.0, '
              '[YODAS](https://huggingface.co/datasets/espnet/yodas) as CC BY 3.0, and '
              '[VOA](https://huggingface.co/datasets/freococo/voa_myanmar_asr_audio_1) as PDDL. '
              'The local ASR/UCSY mappings do not establish a redistribution license. These materials therefore do not acquire a single '
              'combined license; retain source attribution and confirm permissions before redistribution.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-data-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, default=Path('raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('combined/myanmar_asr_all'))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {output}')
    print('Inspecting original ASR/UCSY audio...', flush=True)
    rows = list(local_rows(args.audio_data_root))
    print(f'Inspected {len(rows):,} local mapped clips', flush=True)
    rows.extend(downloaded_rows(args.raw_root))
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    curate(rows)
    retained = [row for row in rows if row['included']]
    unique = list({row['sha256']: row for row in rows}.values())
    labeled = [row for row in retained if row['has_transcript']]
    whisper = [row for row in labeled if row['whisper_duration_ok']]
    summary = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        inventory=describe(rows), unique_inventory=describe(unique), curated=describe(retained),
        by_source=grouped_stats(rows, 'source'), by_catalog_split_before_curation=grouped_stats(rows, 'split'),
        curated_by_source=grouped_stats(retained, 'source'), curated_by_split=grouped_stats(retained, 'split'),
        whisper=describe(whisper), whisper_by_source=grouped_stats(whisper, 'source'),
        whisper_by_split=grouped_stats(whisper, 'split'),
        source_split={source: grouped_stats([r for r in retained if r['source']==source], 'split')
                      for source in sorted({r['source'] for r in rows})},
        combined_exclusions=dict(Counter(r['exclusion_reason'] for r in rows if not r['included'])),
        inventory_overlap={key: overlaps(rows, key) for key in ('sha256','text','record_group')},
        curated_overlap={key: overlaps(retained, key) for key in ('sha256','text','record_group')},
        speaker_overlap=overlaps([r for r in retained if r['speaker_identity'] in ('supplied','inferred_filename')], 'speaker_group'),
        original_mapping_summary=json.loads((args.audio_data_root/'asr_work/summary.json').read_text()),
    )
    assert not summary['curated_overlap']['sha256'], 'Audio duplicates crossed retained splits'
    assert not any('train / ' in pair or ' / train' in pair for pair in summary['curated_overlap']['text']), 'Training text overlap'
    provenance = {}
    inputs = list((args.audio_data_root/'asr_work').glob('*.csv'))
    inputs += list(args.raw_root.glob('*/dataset_dict.json')) + list(args.raw_root.glob('*/*/state.json'))
    for path in sorted(inputs):
        provenance[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    link_directory(output/'audio_data', args.audio_data_root)
    link_directory(output/'raw', args.raw_root)
    csv_write(output/'all.csv', rows)
    csv_write(output/'excluded.csv', [r for r in rows if not r['included']])
    for split in sorted({row['split'] for row in retained}):
        subset = [row for row in retained if row['split'] == split]
        csv_write(output/f'{split}.csv', subset)
        if split != 'unlabeled':
            csv_write(output/f'{split}_whisper.csv', [r for r in subset if r['whisper_duration_ok']])
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (output/'input_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    write_report(output/'README.md', summary)
    print(json.dumps({'output': str(output), 'inventory': summary['inventory'],
                      'curated': summary['curated'], 'exclusions': summary['combined_exclusions']}, indent=2))


if __name__ == '__main__':
    main()
