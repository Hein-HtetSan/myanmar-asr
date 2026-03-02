# Myanmar Automatic Speech Recognition System

**Fine-tuning state-of-the-art transformer models for Burmese speech recognition**

> By **Hein Htet San** & **Ye Myat Kyaw** — March 2026

---

## Overview

Myanmar (Burmese) is a **low-resource language** where commercial ASR systems fail badly — global models produce **WER > 80%** due to complex script (no word boundaries), tonal distinctions, and extremely limited labeled data. This project fine-tunes **three multilingual transformer models** on a curated **54-hour Myanmar speech dataset** and achieves dramatic improvements:

| Metric         | Zero-shot (Baseline) | After Fine-tuning | Improvement |
| -------------- | -------------------- | ----------------- | ----------- |
| **WER (best)** | ~100%                | **33.02%**        | ↓ 67 pp     |
| **CER (best)** | ~88%                 | **13.04%**        | ↓ 75 pp     |

## Models

Three architectures were fine-tuned, each with a **frozen encoder** strategy optimized for low-resource settings:

| Model                    | Params | Approach                 | WER ↓    | CER ↓      | Train Time |
| ------------------------ | ------ | ------------------------ | -------- | ---------- | ---------- |
| **Whisper v3 Turbo**     | 809M   | Full fine-tune (distilled 4-layer decoder) | 54.49%   | 36.00%     | 159 min    |
| **Dolphin (Whisper-v2)** | 1.5B   | Frozen encoder, 32-layer decoder          | **33.02%** | 28.00%     | 335 min    |
| **SeamlessM4T v2 Large** | 2.3B   | Frozen w2v-BERT 2.0 encoder              | 49.12%   | **13.04%** | 239 min    |

**Recommendation**: Use **Dolphin** for word-level tasks (best WER) and **SeamlessM4T** for character-level accuracy (best CER).

## Dataset

| Source                        | Samples   | Notes                              |
| ----------------------------- | --------- | ---------------------------------- |
| Google FLEURS (my_mm)         | ~3,000    | Read speech, clean studio audio    |
| OpenSLR-80                    | ~5,000    | Crowdsourced Myanmar recordings    |
| YODAS / YouTube               | ~9,000    | Diverse real-world speech          |
| Speed Augmentation (0.9×/1.1×)| +10,515   | Data augmentation for robustness   |

**Final splits**: Train 20,814 (~49.5h) · Val 639 (~1.5h) · Test 1,252 (~3.2h) · **Total: 22,705 samples (~54.2h)**

All audio is 16kHz mono WAV. Text normalized to Myanmar Unicode NFC.

### Preprocessing Pipeline

```
Raw 17,552 samples
  → Audio validation     (−312 corrupted/unreadable)
  → Duration filter      (−1,847 too short/long)
  → Text validation      (−2,215 non-Myanmar/empty)
  → SNR filter           (−528 low signal-to-noise)
  → Deduplication        (−351 near-duplicate)
  → Clean dataset:       12,190 samples
  → Speed augmentation:  +10,515 samples
  → Final dataset:       22,705 samples
```

~30% of raw data removed as low-quality.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Project Architecture                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐   SSH Tunnel    ┌──────────────────────┐    │
│  │ Mac M-chip├───────────────►│  Vast.ai RTX 4090    │    │
│  │ (Local)   │   (autossh)    │  - Training (3 models)│   │
│  │           │◄───────────────┤  - FastAPI inference  │    │
│  └─────┬─────┘                └──────────────────────┘    │
│        │                                                   │
│  Docker Services (localhost)                               │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐       │
│  │ MLflow:5050  │  │ MinIO:9001 │  │ Label Studio │       │
│  │ Experiments  │  │ S3 Storage │  │    :8081     │       │
│  └─────────────┘  └────────────┘  └──────────────┘       │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐       │
│  │ Argilla:6900│  │ Redis:6379 │  │ Postgres:5432│       │
│  │ Data Review │  │ Cache/Queue│  │ Metadata DB  │       │
│  └─────────────┘  └────────────┘  └──────────────┘       │
│                                                            │
│  Presentation Container (:8090)                            │
│  ┌─────────────────────────────────────┐                  │
│  │  nginx → /       → Next.js (slides) │                  │
│  │       → /demo/   → Streamlit (demo) │                  │
│  └─────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5) or Linux
- Docker Desktop
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or Conda

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Hein-HtetSan/myanmar-asr.git
cd myanmar-asr

# Create conda environment
conda create -n myanmar-asr python=3.11 -y
conda activate myanmar-asr

# Install dependencies
pip install -r requirements-demo.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if you need to change default credentials
```

### 3. Start Services Stack

```bash
cd services
docker compose up -d
cd ..
```

This starts: **MLflow** (`:5050`), **MinIO** (`:9001`), **Label Studio** (`:8081`), **Argilla** (`:6900`), **Redis** (`:6379`), **PostgreSQL** (`:5432`).

### 4. Run the Presentation + Demo

```bash
# Build & run the all-in-one container
docker compose -f docker-compose.presentation.yml up -d --build
```

- **Slides**: [http://localhost:8090/](http://localhost:8090/) — Neo-Brutalism Next.js presentation
- **Demo**: [http://localhost:8090/demo/](http://localhost:8090/demo/) — Interactive Streamlit ASR demo

### 5. (Optional) Cloud Inference via Vast.ai

For GPU-accelerated inference using the models hosted on Vast.ai:

```bash
# SSH tunnel to Vast.ai instance (FastAPI inference server)
ssh -N -L 8080:localhost:8080 root@<vast-ai-host> -p <port>
```

Then select **"☁️ Cloud (Vast.ai)"** in the Streamlit demo sidebar.

## Project Structure

```
myanmar-asr/
├── Dockerfile                      # Multi-stage: Next.js + Streamlit + nginx
├── docker-compose.presentation.yml # Presentation container
├── requirements-demo.txt           # Python deps (torch CPU, transformers 5.x)
├── .env.example                    # Environment template
│
├── scripts/
│   ├── deploy/
│   │   └── streamlit_app.py        # Streamlit demo (local + cloud inference)
│   ├── training/
│   │   ├── train_whisper_turbo.py   # Whisper v3 Turbo fine-tuning
│   │   ├── train_dolphin.py         # Dolphin (Whisper-v2) fine-tuning
│   │   ├── train_seamless.py        # SeamlessM4T v2 fine-tuning
│   │   └── train_best.py           # Optimized Whisper variant
│   ├── analyze_dataset.py          # Dataset statistics & visualization
│   ├── augment_dataset.py          # Speed augmentation pipeline
│   ├── combine_datasets.py         # Merge FLEURS + OpenSLR + YODAS
│   ├── filter_dataset.py           # Quality filtering (SNR, duration, text)
│   ├── normalize_datasets.py       # Unicode NFC normalization
│   └── export_nemo_manifest.py     # NeMo format export
│
├── presentation/                   # Next.js 15 slides (Neo-Brutalism design)
│   ├── src/app/page.tsx            # 12-slide presentation
│   └── package.json                # React 19, Framer Motion, Tailwind
│
├── services/
│   └── docker-compose.yml          # MLflow, MinIO, Label Studio, Argilla, etc.
│
├── docker/                         # nginx, supervisord configs
└── docs/
    └── Myanmar_ASR_M5_Guide.md     # Full setup guide for Mac M-series
```

## Training Environment

| Component       | Details                                    |
| --------------- | ------------------------------------------ |
| **GPU**         | NVIDIA RTX 4090 (24GB VRAM) via Vast.ai   |
| **Framework**   | PyTorch 2.5.1 + CUDA 12.1                 |
| **Library**     | Transformers 5.2.0, Accelerate             |
| **Tracking**    | MLflow 2.19 + MinIO S3 artifact storage    |
| **Training**    | ~12.2 hours total across all 3 models      |
| **Strategy**    | Frozen encoder, cosine LR scheduler, bf16  |

## Services & Tools

| Service          | Port  | Purpose                                      |
| ---------------- | ----- | -------------------------------------------- |
| **MLflow**       | 5050  | Experiment tracking, model registry           |
| **MinIO**        | 9001  | S3-compatible storage (models, data, audio)   |
| **Label Studio** | 8081  | Audio transcription annotation                |
| **Argilla**      | 6900  | ML-assisted dataset review & validation       |
| **PostgreSQL**   | 5432  | Label Studio metadata                         |
| **Redis**        | 6379  | Task queue & caching                          |
| **Presentation** | 8090  | Next.js slides + Streamlit demo               |

## Key Findings

1. **Frozen encoder is critical** for low-resource fine-tuning — prevents catastrophic forgetting of pre-trained audio features
2. **CER is more appropriate than WER** for Myanmar — the language lacks clear word boundaries, making CER a fairer metric
3. **Deeper decoders improve WER** — Dolphin's 32-layer decoder significantly outperforms Whisper Turbo's 4-layer distilled decoder
4. **54 hours of clean data produces competitive results** — careful preprocessing is more valuable than raw data volume
5. **Speed augmentation (+0.9×/1.1×)** nearly doubled the training set and improved generalization

## Tech Stack

- **Models**: OpenAI Whisper v3 Turbo, Whisper Large-v2 (Dolphin), Meta SeamlessM4T v2
- **Training**: PyTorch, HuggingFace Transformers, Accelerate
- **Tracking**: MLflow, MinIO (S3)
- **Annotation**: Label Studio, Argilla
- **Frontend**: Next.js 15, React 19, Framer Motion, Tailwind CSS
- **Demo**: Streamlit, Python
- **Infra**: Docker, nginx, supervisord, Vast.ai (GPU)

## License

This project is for academic purposes (Master's thesis). Individual datasets and pre-trained models retain their original licenses.

---

<p align="center">
  <b>Myanmar ASR</b> — Advancing speech recognition for the Burmese language 🇲🇲
</p>
