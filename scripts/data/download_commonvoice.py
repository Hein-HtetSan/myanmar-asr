"""Download YODAS Myanmar archives without executing a Hub loading script."""

from pathlib import Path
import tarfile

REPO_ID = "espnet/yodas"
SUBSET = "my000"
SAVE_PATH = "raw/common_voice_my"


def archive_examples(archive_path, transcript_path):
    """Pair transcript IDs with archive members; never extract paths to disk."""
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[0] in transcripts:
                raise ValueError(f"Invalid or duplicate transcript at {transcript_path}:{number}")
            transcripts[fields[0]] = fields[1]
    if not transcripts:
        raise ValueError(f"No transcripts in {transcript_path}")

    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            utterance = Path(member.name).stem
            if not member.isfile() or utterance not in transcripts:
                continue
            with archive.extractfile(member) as audio:
                audio_bytes = audio.read()
            if not audio_bytes:
                raise ValueError(f"Empty audio: {member.name}")
            yield {
                "audio": {"path": None, "bytes": audio_bytes},
                "sentence": transcripts.pop(utterance),
                "source": "yodas_my",
                # An utterance ID is not a verified speaker identity.
                "speaker_id": utterance,
                "locale": "my",
            }
    if transcripts:
        raise ValueError(f"{len(transcripts)} transcripts have no audio in {archive_path}")


def download_examples(revision, archives):
    from huggingface_hub import hf_hub_download

    for filename in archives:
        shard = Path(filename).name.removesuffix(".tar.gz")
        transcript = f"data/{SUBSET}/text/{shard}.txt"
        paths = [hf_hub_download(REPO_ID, path, repo_type="dataset", revision=revision)
                 for path in (filename, transcript)]
        yield from archive_examples(*paths)


def main():
    from datasets import Audio, Dataset, DatasetDict, Features, Value
    from huggingface_hub import HfApi

    print(f"Downloading {REPO_ID} / {SUBSET} from audio archives and transcripts...")
    api = HfApi()
    revision = api.dataset_info(REPO_ID).sha
    archives = sorted(
        entry.path for entry in api.list_repo_tree(
            REPO_ID, path_in_repo=f"data/{SUBSET}/audio",
            repo_type="dataset", revision=revision,
        ) if entry.path.endswith(".tar.gz")
    )
    if not archives:
        raise RuntimeError(f"No audio archives found for {SUBSET}")
    features = Features({
        # Build encoded storage first: newer Audio encoders import TorchCodec.
        "audio": {"bytes": Value("binary"), "path": Value("string")},
        **{name: Value("string") for name in ("sentence", "source", "speaker_id", "locale")},
    })
    full = Dataset.from_generator(
        download_examples, gen_kwargs={"revision": revision, "archives": archives},
        features=features,
    ).cast_column("audio", Audio(sampling_rate=16000)).shuffle(seed=42)
    n = len(full)
    n_val = n_test = max(1, int(n * 0.05))
    if n <= n_val + n_test:
        raise ValueError(f"Not enough samples for train/validation/test: {n}")
    combined = DatasetDict({
        "train": full.select(range(n_val + n_test, n)),
        "validation": full.select(range(n_val)),
        "test": full.select(range(n_val, n_val + n_test)),
    })
    combined.save_to_disk(SAVE_PATH)
    print(f"Saved YODAS Myanmar to {SAVE_PATH} (source revision: {revision})")
    for split, dataset in combined.items():
        print(f"  {split:12s}: {len(dataset):,} samples")


if __name__ == "__main__":
    main()
