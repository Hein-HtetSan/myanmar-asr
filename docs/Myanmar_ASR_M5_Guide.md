# 🇲🇲 Myanmar ASR — Complete Pipeline Guide
> MacBook Pro M5 (Apple Silicon) → Services Setup → Vast.ai Training
> Updated for Apple Silicon ARM64 Architecture

---

## 🍎 PHASE 0 — MAC M5 SYSTEM PREREQUISITES

### Step 0.1 — Install Homebrew (Mac Package Manager)

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# After install, add to PATH (M5 uses /opt/homebrew NOT /usr/local)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Verify
brew --version
```

---

### Step 0.2 — Install Core System Tools

```bash
# All tools you'll need
brew install git wget curl rsync htop tmux \
             ffmpeg portaudio libsndfile \
             postgresql redis tree

# Install Docker Desktop for Apple Silicon
brew install --cask docker

# Open Docker Desktop app and complete setup
open -a Docker

# Verify Docker is running
docker --version
docker run hello-world
```

---

### Step 0.3 — Install Python via Miniforge (CRITICAL for M5)

> ⚠️ **IMPORTANT:** Do NOT use the system Python or standard Anaconda on M5.
> Use **Miniforge** — it's the Apple Silicon-native conda distribution.
> Standard Anaconda and some pip packages ship x86 binaries that run via Rosetta,
> which is 3–5x slower and causes issues with audio libraries.

```bash
# Download Miniforge for Apple Silicon (arm64)
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh \
  -o Miniforge3-arm64.sh

# Install
bash Miniforge3-arm64.sh
# Press ENTER, type "yes" for all prompts

# Reload shell
source ~/.zshrc

# Verify it's arm64 native (NOT x86_64 via Rosetta)
python3 -c "import platform; print(platform.machine())"
# Should print: arm64
```

---

### Step 0.4 — Create Myanmar ASR Conda Environment

```bash
# If you see: "CondaError: Run 'conda init' before 'conda activate'"
# Do this once for zsh, then restart your shell:
conda init zsh
source ~/.zshrc
# (Alternative if you don’t want to modify shell config)
# source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Create dedicated environment with Python 3.11
conda create -n myanmar-asr python=3.11 -y
conda activate myanmar-asr

# Install PyTorch with MPS support (Apple Silicon GPU)
# MPS = Metal Performance Shaders — M5's GPU backend
pip install torch torchvision torchaudio

# Verify MPS (Apple GPU) is available
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"
# Expected: MPS available: True

# Install all ASR dependencies (arm64-compatible)
pip install datasets huggingface_hub soundfile librosa pandas \
            tqdm pyarrow transformers accelerate evaluate jiwer \
            matplotlib seaborn plotly jupyterlab ipywidgets \
            audiomentations psycopg2-binary redis

# Audio-specific
# If `pip install pyaudio` fails with: `fatal error: 'portaudio.h' file not found`
# install PortAudio into the conda env first, then build PyAudio against it:
conda install -y -c conda-forge portaudio
CFLAGS="-I$CONDA_PREFIX/include" LDFLAGS="-L$CONDA_PREFIX/lib" \
  pip install pyaudio sounddevice

# Hugging Face login
# Newer `huggingface_hub` installs provide the `hf` CLI (not `huggingface-cli`).
hf auth login
# → Paste your token from https://huggingface.co/settings/tokens
# Non-interactive option:
# hf auth login --token $HF_TOKEN
```

---

### Step 0.5 — Project Folder Structure

```bash
mkdir -p ~/myanmar-asr/{raw,processed,combined,models,logs,scripts,services,exports,viz}
cd ~/myanmar-asr

# Verify folder structure
tree -L 2 ~/myanmar-asr
```

```
myanmar-asr/
├── raw/              ← Downloaded datasets (original format)
│   ├── openslr80_hf/
│   ├── fleurs_my_mm/
│   ├── common_voice_my/
│   └── voa_burmese/
├── processed/        ← Cleaned, filtered audio
├── combined/         ← Final merged dataset
├── models/           ← Local model checkpoints (for testing)
├── logs/             ← Training logs
├── scripts/          ← All Python scripts
├── services/         ← Docker compose files for services
├── exports/          ← WAV files exported for Label Studio
└── viz/              ← Visualization outputs
```

---

## 🐳 PHASE 1 — SERVICES SETUP (Docker on Mac M5)

> All services run locally via Docker Compose. M5 handles this excellently.

### Step 1.1 — Create Master Docker Compose File

```bash
mkdir -p ~/myanmar-asr/services
cat > ~/myanmar-asr/services/docker-compose.yml << 'EOF'
version: "3.9"

# ─────────────────────────────────────────────
#  Myanmar ASR — Local Services Stack
#  Runs natively on Apple Silicon (arm64)
# ─────────────────────────────────────────────

services:

  # ── Label Studio (Audio Annotation) ──────────────────
  label-studio:
    image: heartexlabs/label-studio:latest
    platform: linux/arm64        # ← Native ARM for M5
    container_name: label-studio
    ports:
      - "8080:8080"
    volumes:
      - ./label_studio_data:/label-studio/data
      - ../exports:/label-studio/files  # Mount audio exports
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
      - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files
      - DJANGO_DB=default
      - POSTGRE_NAME=labelstudio
      - POSTGRE_USER=labelstudio
      - POSTGRE_PASSWORD=labelstudio123
      - POSTGRE_PORT=5432
      - POSTGRE_HOST=postgres
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  # ── Argilla (ML Team Dataset Review) ─────────────────
  argilla:
    image: argilla/argilla-quickstart:latest
    platform: linux/amd64        # ← Runs via Rosetta on M5 (no native arm64 image yet)
    container_name: argilla
    ports:
      - "6900:6900"
    volumes:
      - argilla_data:/var/lib/postgresql/data
      - argilla_elasticsearch:/usr/share/elasticsearch/data
    environment:
      - ARGILLA_HOME_PATH=/var/lib/argilla
      - ARGILLA_SERVER_WORKERS=2
    restart: unless-stopped

  # ── PostgreSQL (Database for Label Studio) ───────────
  postgres:
    image: postgres:15-alpine
    platform: linux/arm64
    container_name: asr-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=labelstudio
      - POSTGRES_USER=labelstudio
      - POSTGRES_PASSWORD=labelstudio123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U labelstudio"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ── MinIO (S3-compatible local storage for audio) ────
  minio:
    image: minio/minio:latest
    platform: linux/arm64
    container_name: asr-minio
    ports:
      - "9000:9000"    # API
      - "9001:9001"    # Web Console
    volumes:
      - minio_data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin123
    command: server /data --console-address ":9001"
    restart: unless-stopped

  # ── Redis (Task queue / caching) ─────────────────────
  redis:
    image: redis:7-alpine
    platform: linux/arm64
    container_name: asr-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # ── Jupyter Lab (Data exploration) ───────────────────
  jupyter:
    image: jupyter/scipy-notebook:latest
    platform: linux/arm64
    container_name: asr-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ~/myanmar-asr:/home/jovyan/myanmar-asr
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=myanmar123
    restart: unless-stopped

volumes:
  postgres_data:
  argilla_data:
  argilla_elasticsearch:
  minio_data:
  redis_data:
  label_studio_data:

EOF
```

---

### Step 1.2 — Start All Services

```bash
cd ~/myanmar-asr/services

# Pull all images first (first time only — may take 5–10 min)
docker compose pull

# Start all services in background
docker compose up -d

# Check all are running
docker compose ps

# View logs for a specific service
docker compose logs -f label-studio
docker compose logs -f argilla
```

**Access URLs after startup:**

| Service | URL | Login |
|---------|-----|-------|
| **Label Studio** | http://localhost:8080 | Register on first visit |
| **Argilla** | http://localhost:6900 | `argilla` / `12345678` |
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| **Jupyter Lab** | http://localhost:8888 | Token: `myanmar123` |

---

### Step 1.3 — Label Studio Full Setup (Audio ASR Project)

```bash
# Install Label Studio Python SDK
conda activate myanmar-asr
pip install label-studio-sdk
```

```python
# scripts/setup_label_studio.py
from label_studio_sdk import Client

# Connect to Label Studio
ls = Client(url="http://localhost:8080", api_key="YOUR_API_KEY")
# Get API key from: http://localhost:8080/user/account → Access Token

# Create Myanmar ASR project
project = ls.start_project(
    title="Myanmar ASR — Audio Transcription Review",
    label_config="""
<View style="display: flex; flex-direction: column; gap: 10px;">
  <Header value="🎙️ Myanmar Audio Transcription Review"/>

  <!-- Audio player with waveform -->
  <Audio name="audio" value="$audio"
         zoom="true" waveHeight="100"
         speed="true" volume="true"/>

  <!-- Show original auto-transcript -->
  <Header value="Original Transcript (auto):"/>
  <Text name="original_text" value="$sentence"
        style="background: #f0f0f0; padding: 10px; border-radius: 4px;"/>

  <!-- Editable corrected transcript -->
  <Header value="✏️ Corrected Transcript (Myanmar):"/>
  <TextArea name="corrected_text" toName="audio"
            placeholder="ပြင်ဆင်ထားသော မြန်မာဘာသာ စာသားကို ဤနေရာတွင် ရိုက်ထည့်ပါ..."
            rows="4" editable="true" maxSubmissions="1"/>

  <!-- Quality rating -->
  <Header value="Audio Quality:"/>
  <Choices name="quality" toName="audio"
           choice="single-radio" showInline="true">
    <Choice value="clean"   alias="🟢 Clean"/>
    <Choice value="noisy"   alias="🟡 Noisy"/>
    <Choice value="unclear" alias="🔴 Unclear"/>
    <Choice value="reject"  alias="⛔ Reject"/>
  </Choices>

  <!-- Source tag (read-only display) -->
  <Header value="Dataset Source:"/>
  <Text name="source_tag" value="$source"/>
</View>
"""
)

print(f"✅ Project created: ID={project.id}")
print(f"   URL: http://localhost:8080/projects/{project.id}/")
```

---

### Step 1.4 — Upload Audio Files to Label Studio

```python
# scripts/upload_to_label_studio.py
import os
import json
import soundfile as sf
import numpy as np
from datasets import load_from_disk
from label_studio_sdk import Client

EXPORT_DIR = os.path.expanduser("~/myanmar-asr/exports/audio")
os.makedirs(EXPORT_DIR, exist_ok=True)

ls = Client(url="http://localhost:8080", api_key="YOUR_API_KEY")
project = ls.get_project(PROJECT_ID)  # Replace with your project ID

ds = load_from_disk("combined/myanmar_asr_50h")["train"]

# Export first 200 samples as WAV files + create tasks
tasks = []
print("Exporting audio files...")
for i, example in enumerate(ds.select(range(200))):
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    filename = f"{i:06d}_{example['source']}.wav"
    filepath = os.path.join(EXPORT_DIR, filename)

    sf.write(filepath, audio_array, sr)

    tasks.append({
        "data": {
            "audio": f"/data/local-files/?d=audio/{filename}",
            "sentence": example["sentence"],
            "source": example["source"],
            "speaker_id": example["speaker_id"],
        }
    })

# Upload tasks to Label Studio
project.import_tasks(tasks)
print(f"✅ Uploaded {len(tasks)} tasks to Label Studio")
print(f"   Review at: http://localhost:8080/projects/{project.id}/")
```

---

### Step 1.5 — Argilla Setup (Team Dataset Review)

```python
# scripts/setup_argilla.py
import argilla as rg
import soundfile as sf
import numpy as np
from datasets import load_from_disk

# Connect to Argilla
rg.init(
    api_url="http://localhost:6900",
    api_key="admin.apikey",          # Default for quickstart image
    workspace="myanmar-asr-team"
)

ds = load_from_disk("combined/myanmar_asr_50h")["train"]

# Create a Text2Text dataset for ASR review
settings = rg.TextClassificationSettings(
    label_schema=["correct", "wrong", "needs_fix", "reject"]
)

# Create dataset
rg.configure_dataset_settings(
    name="myanmar_asr_transcripts",
    settings=settings
)

# Upload records for review
records = []
for i, example in enumerate(ds.select(range(500))):
    records.append(
        rg.TextClassificationRecord(
            text=example["sentence"],
            metadata={
                "source": example["source"],
                "speaker_id": example["speaker_id"],
                "idx": i,
            },
            status="Default",
        )
    )

rg.log(
    records=records,
    name="myanmar_asr_transcripts",
    batch_size=100
)
print("✅ 500 records uploaded to Argilla")
print("   Review at: http://localhost:6900")
```

---

### Step 1.6 — MinIO Storage Setup (for Large Audio Files)

```bash
# Install MinIO CLI
brew install minio/stable/mc

# Configure local MinIO connection
mc alias set local http://localhost:9000 minioadmin minioadmin123

# Create bucket for audio files
mc mb local/myanmar-asr-audio
mc mb local/myanmar-asr-models

# Upload audio exports to MinIO
mc cp --recursive ~/myanmar-asr/exports/audio/ local/myanmar-asr-audio/

# Verify
mc ls local/myanmar-asr-audio
```

---

### Step 1.7 — Stop / Restart Services

```bash
cd ~/myanmar-asr/services

# Stop all (data is preserved in volumes)
docker compose down

# Stop + remove all data volumes (DESTRUCTIVE - clean slate)
docker compose down -v

# Restart a single service
docker compose restart label-studio

# View resource usage (important on M5 to check RAM)
docker stats
```

---

## 📥 PHASE 2 — DOWNLOADING ALL DATASETS

### Step 2.1 — Download OpenSLR-80 (Google Burmese, ~6 hrs)

```bash
conda activate myanmar-asr
cd ~/myanmar-asr

# Method A: Direct download (use curl on Mac, not wget)
mkdir -p raw/openslr80
curl -L https://www.openslr.org/resources/80/my_mm_female.zip \
     -o raw/openslr80/my_mm_female.zip
unzip raw/openslr80/my_mm_female.zip -d raw/openslr80/

# Method B: Via HuggingFace (recommended)
python3 - <<'EOF'
from datasets import load_dataset
ds = load_dataset("chuuhtetnaing/myanmar-speech-dataset-openslr-80")
ds.save_to_disk("raw/openslr80_hf")
print(f"✅ OpenSLR-80: {len(ds['train'])} samples")
EOF
```

---

### Step 2.2 — Download Google FLEURS (~12 hrs)

```python
# scripts/download_fleurs.py
from datasets import load_dataset

print("Downloading FLEURS my_mm...")
ds = load_dataset("google/fleurs", "my_mm", trust_remote_code=True)
ds.save_to_disk("raw/fleurs_my_mm")

for split in ds:
    hours = sum(len(a["array"]) / a["sampling_rate"] for a in ds[split]["audio"]) / 3600
    print(f"  {split}: {len(ds[split])} samples (~{hours:.1f} hrs)")
```

```bash
python3 scripts/download_fleurs.py
```

---

### Step 2.3 — Download Mozilla Common Voice (~15–30 hrs)

```bash
# Requires HuggingFace login (already done in Phase 0)
```

```python
# scripts/download_commonvoice.py
from datasets import load_dataset

print("Downloading Common Voice Burmese...")
ds = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "my",
    trust_remote_code=True
)
ds.save_to_disk("raw/common_voice_my")

for split in ds:
    hours = len(ds[split]) * 7 / 3600
    print(f"  {split}: {len(ds[split])} samples (~{hours:.1f} hrs)")
```

---

### Step 2.4 — Download VOA Burmese (Unlabeled, Streaming)

```python
# scripts/download_voa.py
from datasets import load_dataset

# Stream only (full dataset = hundreds of GB)
print("Connecting to VOA Burmese (streaming mode)...")
ds = load_dataset("freococo/voa_myanmar_asr_audio_1", streaming=True)

# Preview first few samples
for i, sample in enumerate(ds["train"]):
    dur = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
    print(f"  Sample {i}: {dur:.1f}s")
    if i >= 4:
        break

print("ℹ️ Full VOA download = ~hundreds of GB. Download on Vast.ai server instead.")
```

---

## 🔧 PHASE 3 — NORMALIZE & COMBINE DATASETS

### Step 3.1 — Normalize Schema Across All Sources

```python
# scripts/normalize_datasets.py
from datasets import load_from_disk, Audio

def normalize_openslr(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["sentence"],
            "source":     "openslr80",
            "speaker_id": str(example.get("speaker_id", "unknown")),
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))

def normalize_fleurs(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["transcription"],
            "source":     "fleurs",
            "speaker_id": str(example.get("speaker_id", "unknown")),
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))

def normalize_commonvoice(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["sentence"],
            "source":     "commonvoice",
            "speaker_id": example.get("client_id", "unknown")[:8],
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))
```

---

### Step 3.2 — Combine Into One Dataset

```python
# scripts/combine_datasets.py
import sys
sys.path.insert(0, "scripts")
from normalize_datasets import normalize_openslr, normalize_fleurs, normalize_commonvoice
from datasets import load_from_disk, concatenate_datasets, DatasetDict

print("Loading datasets...")
openslr = load_from_disk("raw/openslr80_hf")["train"]
fleurs  = load_from_disk("raw/fleurs_my_mm")
cv      = load_from_disk("raw/common_voice_my")

print("Normalizing schemas...")
openslr_norm = normalize_openslr(openslr)
fl_train = normalize_fleurs(fleurs["train"])
fl_val   = normalize_fleurs(fleurs["validation"])
fl_test  = normalize_fleurs(fleurs["test"])
cv_train = normalize_commonvoice(cv["train"])
cv_val   = normalize_commonvoice(cv["validation"])
cv_test  = normalize_commonvoice(cv["test"])

print("Merging...")
combined = DatasetDict({
    "train":      concatenate_datasets([openslr_norm, fl_train, cv_train]).shuffle(seed=42),
    "validation": concatenate_datasets([fl_val, cv_val]),
    "test":       concatenate_datasets([fl_test, cv_test]),
})

print(f"\n✅ Combined Dataset:")
for split, ds in combined.items():
    total_secs = sum(len(a["array"]) / a["sampling_rate"] for a in ds["audio"])
    print(f"  {split:12s}: {len(ds):>6,} samples | {total_secs/3600:.1f} hours")

combined.save_to_disk("combined/myanmar_asr_50h")
print("\n✅ Saved to combined/myanmar_asr_50h")
```

```bash
cd ~/myanmar-asr
python3 scripts/combine_datasets.py
```

---

### Step 3.3 — Filter Bad Samples

```python
# scripts/filter_dataset.py
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h")

def is_valid(example):
    audio    = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    text     = example["sentence"].strip()
    return (
        0.5 <= duration <= 30.0 and
        len(text) >= 2 and
        len(text) <= 500 and
        not text.isspace()
    )

original_sizes = {k: len(v) for k, v in ds.items()}
ds_filtered = ds.filter(is_valid, num_proc=8)  # Uses all M5 efficiency cores

print("\n🧹 Filtering Results:")
for split in ds_filtered:
    removed = original_sizes[split] - len(ds_filtered[split])
    print(f"  {split}: removed {removed} bad samples → {len(ds_filtered[split]):,} remaining")

ds_filtered.save_to_disk("combined/myanmar_asr_50h_clean")
print("✅ Saved filtered dataset")
```

---

### Step 3.4 — Export JSONL Manifest

```python
# scripts/export_manifest.py
import json
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h_clean")

for split in ds:
    out_path = f"combined/{split}_manifest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split]):
            row = {
                "id":         f"{split}_{i:06d}",
                "sentence":   ex["sentence"],
                "source":     ex["source"],
                "speaker_id": ex["speaker_id"],
                "duration":   round(len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"], 2),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ {split}_manifest.jsonl → {i+1:,} rows")
```

---

## 📊 PHASE 4 — ANALYZE & VISUALIZE

### Step 4.1 — Dataset Analysis Script

```python
# scripts/analyze_dataset.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h_clean")["train"]

durations  = [len(a["array"]) / a["sampling_rate"] for a in ds["audio"]]
text_lens  = [len(s) for s in ds["sentence"]]
sources    = list(ds["source"])

df = pd.DataFrame({"duration": durations, "text_len": text_lens, "source": sources})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Myanmar ASR — Dataset Analysis", fontsize=16, fontweight="bold")

# 1. Duration distribution
axes[0,0].hist(df["duration"], bins=60, color="#5B8DEF", edgecolor="white")
axes[0,0].axvline(df["duration"].mean(), color="red", linestyle="--",
                   label=f"Mean: {df['duration'].mean():.1f}s")
axes[0,0].set_title("Audio Duration Distribution")
axes[0,0].set_xlabel("Duration (seconds)")
axes[0,0].legend()

# 2. Source pie
src_counts = df["source"].value_counts()
axes[0,1].pie(src_counts.values, labels=src_counts.index, autopct="%1.1f%%",
               colors=["#5B8DEF","#F5A623","#7ED321"])
axes[0,1].set_title("Samples by Dataset Source")

# 3. Text length histogram
axes[1,0].hist(df["text_len"], bins=60, color="#7ED321", edgecolor="white")
axes[1,0].set_title("Transcript Length (characters)")
axes[1,0].set_xlabel("Characters")

# 4. Hours by source
hours = df.groupby("source")["duration"].sum() / 3600
bars = axes[1,1].bar(hours.index, hours.values, color=["#5B8DEF","#F5A623","#7ED321"])
axes[1,1].set_title("Training Hours by Source")
axes[1,1].set_ylabel("Hours")
for bar, val in zip(bars, hours.values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                   f"{val:.1f}h", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("viz/dataset_analysis.png", dpi=150, bbox_inches="tight")
print("✅ Saved to viz/dataset_analysis.png")
print(f"\n📊 Total hours: {df['duration'].sum()/3600:.1f}h")
print(df.groupby("source")["duration"].agg(count="count", total_h=lambda x: x.sum()/3600))
```

```bash
mkdir -p viz
python3 scripts/analyze_dataset.py
open viz/dataset_analysis.png  # Opens in Preview on Mac
```

---

### Step 4.2 — Launch Jupyter Lab for Interactive Exploration

```bash
# Option A: Via Docker (already running on port 8888)
open http://localhost:8888
# Token: myanmar123

# Option B: Direct on Mac (faster, uses native M5 speed)
conda activate myanmar-asr
cd ~/myanmar-asr
jupyter lab --port 8889 --no-browser &
open http://localhost:8889
```

---

## 🖥️ PHASE 5 — LOCAL DEV STACK OVERVIEW (M5 Mac)

```
MacBook Pro M5 — Local Services Architecture
──────────────────────────────────────────────────

  ┌─────────────────────────────────────────────┐
  │           Docker Desktop (arm64)             │
  │  ┌──────────────┐  ┌──────────────────────┐ │
  │  │ Label Studio │  │      Argilla          │ │
  │  │ :8080        │  │      :6900            │ │
  │  └──────┬───────┘  └──────────┬───────────┘ │
  │         │                     │              │
  │  ┌──────▼─────────────────────▼───────────┐ │
  │  │   PostgreSQL :5432    Redis :6379       │ │
  │  └────────────────────────────────────────┘ │
  │  ┌─────────────┐   ┌──────────────────────┐ │
  │  │    MinIO    │   │     Jupyter Lab       │ │
  │  │  :9000/9001 │   │       :8888           │ │
  │  └─────────────┘   └──────────────────────┘ │
  └─────────────────────────────────────────────┘

  Native Python (conda myanmar-asr):
  ├── Data download scripts
  ├── Dataset normalization + combine
  ├── Audio filtering + analysis
  └── MPS-accelerated local testing

  External:
  ├── HuggingFace Hub (dataset + model storage)
  ├── Weights & Biases (experiment tracking)
  └── Vast.ai (GPU training server)
```

---

### Step 5.1 — M5 Mac Performance Tips for Data Processing

```python
# Use all M5 cores for dataset processing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For datasets.map(), M5 has 10+ cores — use them all
ds.map(process_fn, num_proc=10)    # M5 Pro: 12 cores
ds.filter(filter_fn, num_proc=10)

# Use MPS for local model inference/testing (NOT full training)
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Running on: {device}")  # mps
```

---

### Step 5.2 — Push Dataset to HuggingFace Hub (Private)

```python
# scripts/push_to_hub.py
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h_clean")

print("Pushing to HuggingFace Hub (private)...")
ds.push_to_hub(
    "YOUR_HF_USERNAME/myanmar-asr-50h",
    private=True,
    max_shard_size="500MB",   # Splits into manageable chunks
)
print("✅ Dataset available at: https://huggingface.co/datasets/YOUR_HF_USERNAME/myanmar-asr-50h")
```

---

## ☁️ PHASE 6 — VAST.AI SERVER SETUP

### Step 6.1 — Install Vast.ai CLI on Mac

```bash
conda activate myanmar-asr
pip install vastai

# Set your API key
vastai set api-key YOUR_VAST_API_KEY
# Get key from: https://vast.ai/account → API Key

# Search for good GPU instances
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 inet_down>500 disk_space>200 reliability>0.97'
```

**Recommended Vast.ai Instances:**

| Use Case | GPU | VRAM | Cost/hr |
|----------|-----|------|---------|
| Quick test | RTX 3090 | 24GB | ~$0.25 |
| **Best value** | **RTX 4090** | **24GB** | **~$0.40** |
| Fastest training | A100 40GB | 40GB | ~$1.20 |
| Large model | A100 80GB | 80GB | ~$2.00 |

```bash
# Rent an instance (replace OFFER_ID from search results)
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel \
  --disk 200 \
  --env '-e HF_TOKEN=hf_YOURTOKEN -e WANDB_API_KEY=YOUR_WANDB_KEY'

# List your running instances
vastai show instances

# SSH into instance
vastai ssh INSTANCE_ID
```

---

### Step 6.2 — Server Bootstrap Script (Run Once on Vast.ai)

```bash
# Save this as: scripts/vast_bootstrap.sh
# Upload and run it on the Vast.ai server

cat > scripts/vast_bootstrap.sh << 'BOOTSTRAP'
#!/bin/bash
set -e
echo "🚀 Setting up Myanmar ASR training environment..."

cd /workspace

# Install Python dependencies
pip install -q \
  transformers datasets evaluate accelerate \
  jiwer soundfile librosa audiomentations \
  tensorboard wandb tqdm pyarrow

# Authenticate HuggingFace
huggingface-cli login --token $HF_TOKEN

# Authenticate Weights & Biases
wandb login $WANDB_API_KEY

# Download dataset from HuggingFace Hub
python3 -c "
from datasets import load_dataset
print('Downloading Myanmar ASR dataset...')
ds = load_dataset('YOUR_HF_USERNAME/myanmar-asr-50h')
ds.save_to_disk('/workspace/data/myanmar_asr')
for split in ds:
    print(f'  {split}: {len(ds[split]):,} samples')
print('Dataset ready!')
"

# Set up tmux session for training
tmux new-session -d -s training
echo "✅ Server ready! Connect with: tmux attach -t training"
BOOTSTRAP

# Upload bootstrap script to server
scp -P PORT scripts/vast_bootstrap.sh root@VAST_IP:/workspace/
# Then on the server:
# bash /workspace/vast_bootstrap.sh
```

---

### Step 6.3 — Sync Local Scripts to Vast.ai

```bash
# From your MacBook M5 — sync scripts to Vast.ai server
rsync -avz --progress \
  ~/myanmar-asr/scripts/ \
  root@VAST_IP:/workspace/scripts/ \
  -e "ssh -p PORT"

# Or use the vastai CLI
vastai copy ./scripts INSTANCE_ID:/workspace/scripts
```

---

## 🚀 PHASE 7 — TRAINING (On Vast.ai Server)

### Step 7.1 — Whisper Fine-tuning Script

```python
# scripts/train_whisper.py
import torch
import wandb
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np

# ── Config ────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-small"   # swap to whisper-medium for better WER
DATASET_PATH  = "/workspace/data/myanmar_asr"
OUTPUT_DIR    = "/workspace/models/whisper-myanmar"
SAMPLING_RATE = 16000
HF_REPO       = "YOUR_HF_USERNAME/whisper-small-myanmar"

# ── Load ─────────────────────────────────────────────
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="my", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []

# ── Dataset ───────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare_dataset,
            remove_columns=["audio","sentence","source","speaker_id","locale"],
            num_proc=4)

# ── Data Collator ─────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch  = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        lbl = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lbl[:, 0] == self.processor.tokenizer.bos_token_id).all():
            lbl = lbl[:, 1:]
        batch["labels"] = lbl
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── Metrics ───────────────────────────────────────────
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": round(100 * wer_metric.compute(predictions=pred_str, references=label_str), 2),
        "cer": round(100 * cer_metric.compute(predictions=pred_str, references=label_str), 2),
    }

# ── Training Args ─────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,      # Effective batch = 32
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    bf16=True,                          # bf16 on A100 (use fp16=True for RTX 4090)
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard","wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_strategy="every_save",          # Auto-backup every checkpoint
)

# ── Train ─────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model()
trainer.push_to_hub()
print("✅ Training complete! Model saved to HuggingFace Hub.")
```

```bash
# On Vast.ai server — run inside tmux
tmux attach -t training
python3 /workspace/scripts/train_whisper.py
# Ctrl+B, D to detach (training keeps running)
```

---

## 💡 PHASE 8 — TIPS & TRICKS

### 🍎 M5 Mac-Specific Tips

```bash
# Check Docker memory allocation (increase if services are slow)
# Docker Desktop → Settings → Resources → Memory → Set to 12–16GB

# Monitor Mac resources while Docker is running
sudo powermetrics --samplers cpu_power,gpu_power -n 3
htop  # CPU/RAM view

# Free up memory if Docker is eating too much
docker system prune -f

# Use Rosetta 2 compatibility mode for x86 images
# Already handled automatically by Docker Desktop on M5
```

### 🎯 Audio Augmentation (Improve Model Robustness)

```python
# scripts/augment_data.py
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
    RoomSimulator(p=0.2),  # Simulate room acoustics
])

def augment_batch(example):
    audio = example["audio"]["array"].astype(np.float32)
    sr    = example["audio"]["sampling_rate"]
    example["audio"]["array"] = augment(samples=audio, sample_rate=sr)
    return example

# Apply ONLY to training set
ds["train"] = ds["train"].map(augment_batch, num_proc=4)
```

### 📈 Pseudo-Labeling VOA (Expand Beyond 50 Hours)

```python
# Run on Vast.ai after initial training
from transformers import pipeline
from datasets import load_dataset

asr = pipeline("automatic-speech-recognition",
               model="YOUR_HF_USERNAME/whisper-small-myanmar",
               device=0, chunk_length_s=30)

def pseudo_label_batch(examples):
    results = asr([a["array"] for a in examples["audio"]])
    examples["sentence"] = [r["text"] for r in results]
    examples["source"]   = ["voa_pseudo"] * len(results)
    return examples

voa = load_dataset("freococo/voa_myanmar_asr_audio_1", streaming=True)
# Process in chunks, filter by confidence, add to combined dataset
```

### 🏆 Model Selection Guide

| Model | VRAM | WER (expected) | Vast.ai Train Time | Best For |
|-------|------|---------------|-------------------|----------|
| whisper-tiny | 1GB | ~30% | ~1 hr RTX 4090 | Baseline sanity check |
| **whisper-small** | 4GB | ~18–22% | ~4 hrs RTX 4090 | **Quick start** |
| **whisper-medium** | 8GB | ~12–16% | ~10 hrs RTX 4090 | **Best balance** |
| whisper-large-v3 | 20GB | ~8–12% | ~20 hrs RTX 4090 | Production quality |
| wav2vec2-large | 6GB | ~14–20% | ~6 hrs RTX 4090 | CTC alternative |

---

## 📋 COMPLETE CHECKLIST

**Mac M5 Setup:**
- [ ] Homebrew installed
- [ ] Miniforge (arm64) installed — NOT standard Anaconda
- [ ] `myanmar-asr` conda env created
- [ ] PyTorch with MPS verified (`torch.backends.mps.is_available() == True`)
- [ ] Docker Desktop for Apple Silicon installed
- [ ] All brew system packages installed (ffmpeg, portaudio, etc.)

**Services (Docker):**
- [ ] `docker-compose.yml` created in `~/myanmar-asr/services/`
- [ ] `docker compose up -d` runs without errors
- [ ] Label Studio accessible at http://localhost:8080
- [ ] Argilla accessible at http://localhost:6900
- [ ] MinIO Console accessible at http://localhost:9001
- [ ] Jupyter Lab accessible at http://localhost:8888
- [ ] Label Studio project created with Myanmar ASR config

**Dataset Pipeline:**
- [ ] HuggingFace token configured
- [ ] OpenSLR-80 downloaded (~6h)
- [ ] FLEURS downloaded (~12h)
- [ ] Common Voice downloaded (~20h)
- [ ] Schemas normalized (all match)
- [ ] Datasets combined and saved to `combined/myanmar_asr_50h`
- [ ] Bad samples filtered out
- [ ] JSONL manifests exported
- [ ] Dataset analyzed + viz charts generated

**Collaboration & Review:**
- [ ] Audio files exported to Label Studio
- [ ] Sample records uploaded to Argilla
- [ ] Dataset pushed to HuggingFace Hub (private)

**Vast.ai Training:**
- [ ] Vast.ai CLI installed + API key set
- [ ] Instance rented (RTX 4090 or A100)
- [ ] Bootstrap script run on server
- [ ] WandB API key configured
- [ ] Scripts synced to server via rsync
- [ ] tmux session started
- [ ] Training launched
- [ ] Model auto-saved to HuggingFace Hub

---

*Guide v2.0 | Myanmar ASR Pipeline | MacBook Pro M5 (Apple Silicon) | Vast.ai GPU Training*
