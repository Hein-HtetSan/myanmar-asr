"""Upload transcript records to Argilla using REST API directly.
Avoids argilla SDK dependency conflicts with numpy/pandas versions.
"""
import json
import requests

ARGILLA_URL = "http://localhost:6900"

# Get auth token (admin/12345678 are quickstart defaults)
token_resp = requests.post(
    f"{ARGILLA_URL}/api/security/token",
    data={"username": "admin", "password": "12345678"},
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
token_resp.raise_for_status()
TOKEN = token_resp.json()["access_token"]
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Read all manifests (train + test + validation)
records_data = []
for manifest in ["combined/train_manifest.jsonl", "combined/test_manifest.jsonl", "combined/validation_manifest.jsonl"]:
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["split"] = manifest.split("/")[-1].replace("_manifest.jsonl", "")
                records_data.append(rec)
    except FileNotFoundError:
        print(f"  Skipping {manifest} (not found)")

print(f"Loaded {len(records_data)} records from all manifests")

# Create a TextClassification dataset via legacy API
DATASET_NAME = "myanmar_asr_transcripts"
WORKSPACE = "admin"

# Create dataset
create_resp = requests.post(
    f"{ARGILLA_URL}/api/datasets",
    headers={**HEADERS, "X-Argilla-Workspace": WORKSPACE},
    json={
        "name": DATASET_NAME,
        "task": "TextClassification",
        "settings": {
            "label_schema": {
                "labels": ["correct", "wrong", "needs_fix", "reject"]
            }
        },
    },
)
if create_resp.status_code == 409:
    print(f"Dataset '{DATASET_NAME}' already exists, will append records.")
elif create_resp.status_code >= 400:
    print(f"Warning creating dataset: {create_resp.status_code} {create_resp.text}")
else:
    print(f"Created dataset: {DATASET_NAME}")

# Upload ALL records in batches
BATCH_SIZE = 100
total = len(records_data)
uploaded = 0

for batch_start in range(0, total, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total)
    batch_records = []
    for i in range(batch_start, batch_end):
        row = records_data[i]
        batch_records.append({
            "inputs": {"text": row["sentence"]},
            "metadata": {
                "source": row["source"],
                "speaker_id": row["speaker_id"],
                "duration": row["duration"],
                "idx": i,
            },
            "status": "Default",
        })

    log_resp = requests.post(
        f"{ARGILLA_URL}/api/datasets/{DATASET_NAME}/TextClassification:bulk",
        headers={**HEADERS, "X-Argilla-Workspace": WORKSPACE},
        json={"records": batch_records},
    )
    if log_resp.ok:
        result = log_resp.json()
        uploaded += result.get("processed", len(batch_records))
        print(f"  Batch {batch_start}-{batch_end}: OK ({result.get('processed', '?')} processed)")
    else:
        print(f"  Batch {batch_start}-{batch_end}: FAILED ({log_resp.status_code}: {log_resp.text[:200]})")

print(f"\n✅ {uploaded} records uploaded to Argilla")
print(f"   Review at: {ARGILLA_URL}")