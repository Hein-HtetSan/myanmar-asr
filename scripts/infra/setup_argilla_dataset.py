#!/usr/bin/env python3
"""Create Argilla dataset for Myanmar ASR transcription review with MinIO audio links."""

import json
import argilla as rg
from argilla.client.feedback.schemas.fields import TextField
from argilla.client.feedback.schemas.questions import (
    LabelQuestion,
    TextQuestion,
)
from argilla.client.feedback.schemas.records import FeedbackRecord
from argilla.client.feedback.dataset.local.dataset import FeedbackDataset

ARGILLA_URL = "http://localhost:6900"
ARGILLA_KEY = "admin.apikey"
WORKSPACE = "admin"
MINIO_BASE = "http://localhost:9002/myanmar-asr-data"
MANIFEST = "combined/train_manifest.jsonl"
DATASET_NAME = "myanmar-asr-transcription"
MAX_RECORDS = 500  # Upload a sample for annotation

def main():
    rg.init(api_url=ARGILLA_URL, api_key=ARGILLA_KEY, workspace=WORKSPACE)

    # Define dataset schema
    dataset = FeedbackDataset(
        fields=[
            TextField(name="audio_url", title="Audio URL (open in browser)", required=True),
            TextField(name="transcript", title="Original Transcript", required=True),
            TextField(name="source", title="Data Source", required=False),
            TextField(name="duration", title="Duration (seconds)", required=False),
        ],
        questions=[
            TextQuestion(
                name="corrected_transcript",
                title="Corrected Transcript (Myanmar)",
                description="Edit the transcript if it contains errors. Leave unchanged if correct.",
                required=True,
            ),
            LabelQuestion(
                name="audio_quality",
                title="Audio Quality",
                labels={"clean": "🟢 Clean", "noisy": "🟡 Noisy", "unclear": "🔴 Unclear", "reject": "⛔ Reject"},
                required=True,
            ),
            LabelQuestion(
                name="transcript_accuracy",
                title="Transcript Accuracy",
                labels={"correct": "✅ Correct", "minor_errors": "⚠️ Minor Errors", "major_errors": "❌ Major Errors", "wrong": "🚫 Completely Wrong"},
                required=True,
            ),
        ],
        guidelines=(
            "# Myanmar ASR Transcription Review\n\n"
            "Review audio-transcript pairs from the Myanmar ASR dataset.\n\n"
            "## Steps:\n"
            "1. Open the audio URL in a new tab to listen\n"
            "2. Read the original transcript\n"
            "3. If the transcript has errors, provide a corrected version\n"
            "4. Rate the audio quality and transcript accuracy\n\n"
            "## Audio URL:\n"
            "Copy the URL from the 'Audio URL' field and open it in your browser to listen.\n"
            "Audio files are stored in MinIO at localhost:9002.\n\n"
            "## Notes:\n"
            "- Myanmar text should use Unicode NFC normalization\n"
            "- Listen for background noise, music, or unclear speech\n"
            "- Mark 'Reject' if audio is unusable"
        ),
    )

    # Load records from manifest
    records = []
    with open(MANIFEST, "r") as f:
        for i, line in enumerate(f):
            if i >= MAX_RECORDS:
                break
            entry = json.loads(line.strip())
            audio_path = entry.get("audio_filepath", entry.get("audio", ""))
            text = entry.get("text", entry.get("sentence", ""))
            source = entry.get("source", "unknown")
            duration = entry.get("duration", 0)

            # Build MinIO URL for the audio
            # NeMo audio files are in exports/nemo_audio/
            filename = audio_path.split("/")[-1]
            split = "train"
            if "/test/" in audio_path:
                split = "test"
            elif "/val/" in audio_path:
                split = "val"
            audio_url = f"{MINIO_BASE}/audio/nemo/{split}/{filename}"

            records.append(
                FeedbackRecord(
                    fields={
                        "audio_url": audio_url,
                        "transcript": text,
                        "source": source,
                        "duration": f"{duration:.1f}",
                    }
                )
            )

    print(f"Prepared {len(records)} records")

    # Push dataset to Argilla
    remote = dataset.push_to_argilla(name=DATASET_NAME, workspace=WORKSPACE)
    print(f"Created dataset: {DATASET_NAME}")

    # Add records
    remote.add_records(records)
    print(f"Added {len(records)} records to '{DATASET_NAME}'")
    print(f"View at: {ARGILLA_URL}/dataset/{remote.id}/annotation-mode")


if __name__ == "__main__":
    main()
