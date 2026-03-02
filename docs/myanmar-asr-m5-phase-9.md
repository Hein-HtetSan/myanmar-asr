
## 🔥 PHASE 9 — ADVANCED MODEL FINE-TUNING ON VAST.AI

> These are the most powerful pretrained models you can fine-tune for Myanmar/Burmese ASR.
> All training runs on your rented **Vast.ai GPU server** — NOT your MacBook.

---

### Model Power Rankings at a Glance

| Model | Power | Best Use Case | Min VRAM | Vast.ai GPU | Est. Cost |
|-------|-------|--------------|----------|-------------|-----------|
| **Canary-1B (NVIDIA)** | 🏆 10/10 | Maximum Burmese accuracy | 24GB+ | A100 40GB | ~$12–20 |
| **Dolphin ASR** | ⚡ 9.5/10 | Asian dialect specialist | 16GB+ | RTX 4090 | ~$5–8 |
| **Whisper Large-v3 Turbo** | 🚀 9/10 | Fastest fine-tune, cheapest | 12GB+ | RTX 3090 | ~$3–5 |
| **SeamlessM4T v2 Large** | 🌐 8/10 | ASR + Translation together | 16GB+ | RTX 4090 | ~$6–10 |

> 💡 **Recommended starting order:** Whisper v3 Turbo first (cheap, fast baseline) → then Canary for production.

---

### Step 9.0 — Vast.ai Instance Selection Per Model

```bash
conda activate myanmar-asr

# ── For Whisper v3 Turbo (cheapest — RTX 3090 is enough) ──
vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 inet_down>500 disk_space>150 reliability>0.97'

# ── For Dolphin ASR / SeamlessM4T (RTX 4090 sweet spot) ──
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 inet_down>500 disk_space>200 reliability>0.97'

# ── For Canary-1B (NEEDS A100 — 24GB minimum) ──
vastai search offers \
  'gpu_ram>=24 gpu_name=A100 num_gpus=1 disk_space>250 reliability>0.97'

# Rent your chosen instance
vastai create instance OFFER_ID \
  --image nvcr.io/nvidia/nemo:24.01 \        # Use NeMo image for Canary
  --disk 250 \
  --env '-e HF_TOKEN=hf_YOURTOKEN -e WANDB_API_KEY=YOUR_KEY'

# OR use standard PyTorch image for non-Canary models
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel \
  --disk 200 \
  --env '-e HF_TOKEN=hf_YOURTOKEN -e WANDB_API_KEY=YOUR_KEY'
```

---

## 🏆 MODEL 1 — NVIDIA Canary-1B (Best Accuracy)

> NVIDIA's Canary is a FastConformer-based model trained on 85,000+ hours of multilingual speech.
> It uses the **NeMo framework** instead of HuggingFace Transformers.
> The 1B variant is the most capable open-source ASR model available as of 2025.

### Step 9.1A — Install NeMo on Vast.ai Server

```bash
# SSH into your A100 instance first
vastai ssh INSTANCE_ID

# If using the NeMo Docker image (recommended) — NeMo is pre-installed
python3 -c "import nemo; print(nemo.__version__)"

# If using standard PyTorch image — install NeMo manually
pip install nemo_toolkit['asr']
# OR for full install with all dependencies:
pip install nemo_toolkit['all']

# Verify CUDA + GPU
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

### Step 9.1B — Prepare Dataset for NeMo Format

> NeMo uses a specific JSONL manifest format different from HuggingFace datasets.

```python
# scripts/export_nemo_manifest.py
# Run this on your MacBook first, then rsync the manifests to Vast.ai

import json
import os
import soundfile as sf
import numpy as np
from datasets import load_from_disk

ds = load_from_disk("combined/myanmar_asr_50h_clean")

AUDIO_DIR = "exports/nemo_audio"
os.makedirs(f"{AUDIO_DIR}/train", exist_ok=True)
os.makedirs(f"{AUDIO_DIR}/val",   exist_ok=True)
os.makedirs(f"{AUDIO_DIR}/test",  exist_ok=True)

def export_nemo(split, folder):
    manifest_path = f"combined/nemo_{split}_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split]):
            # Save WAV file
            filename  = f"{folder}/{i:06d}.wav"
            audio_arr = ex["audio"]["array"].astype(np.float32)
            sr        = ex["audio"]["sampling_rate"]
            sf.write(f"{AUDIO_DIR}/{filename}", audio_arr, sr)

            duration = len(audio_arr) / sr

            # NeMo manifest format (REQUIRED fields)
            row = {
                "audio_filepath": f"/workspace/data/nemo_audio/{filename}",
                "text":           ex["sentence"],
                "duration":       round(duration, 4),
                "lang":           "my",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ {split}: exported {i+1:,} samples → {manifest_path}")

export_nemo("train",      "train")
export_nemo("validation", "val")
export_nemo("test",       "test")
```

```bash
# Sync audio + manifests to Vast.ai server
rsync -avz --progress exports/nemo_audio/  root@VAST_IP:/workspace/data/nemo_audio/ -e "ssh -p PORT"
rsync -avz --progress combined/nemo_*.jsonl root@VAST_IP:/workspace/data/ -e "ssh -p PORT"
```

---

### Step 9.1C — Fine-tune Canary-1B with NeMo

```python
# scripts/train_canary.py  (Run on Vast.ai A100 server)
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

# ── Load Canary-1B pretrained ─────────────────────────
print("Loading NVIDIA Canary-1B...")
model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
    "nvidia/canary-1b"
)

# ── Update config for Myanmar fine-tuning ─────────────
cfg = model.cfg

with open_dict(cfg):
    # Training data
    cfg.train_ds.manifest_filepath = "/workspace/data/nemo_train_manifest.jsonl"
    cfg.train_ds.batch_size        = 8          # Per GPU (A100 40GB)
    cfg.train_ds.num_workers       = 8
    cfg.train_ds.shuffle           = True
    cfg.train_ds.max_duration      = 30.0       # Skip clips > 30s
    cfg.train_ds.min_duration      = 0.5

    # Validation data
    cfg.validation_ds.manifest_filepath = "/workspace/data/nemo_val_manifest.jsonl"
    cfg.validation_ds.batch_size        = 8
    cfg.validation_ds.num_workers       = 4

    # Optimizer
    cfg.optim.lr           = 5e-6              # Low LR for fine-tuning
    cfg.optim.weight_decay = 1e-3

    # Set language to Burmese
    cfg.decoding.beam.ngram_lm_model = None    # No LM for now
    cfg.train_ds.lang = "my"

model.setup_training_data(cfg.train_ds)
model.setup_validation_data(cfg.validation_ds)

# ── Trainer ───────────────────────────────────────────
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=5,
    precision="bf16-mixed",           # bf16 on A100
    accumulate_grad_batches=4,        # Effective batch = 32
    gradient_clip_val=1.0,
    val_check_interval=0.25,          # Validate 4x per epoch
    log_every_n_steps=10,
    enable_checkpointing=True,
    default_root_dir="/workspace/models/canary-myanmar",
    logger=pl.loggers.WandbLogger(
        project="myanmar-asr",
        name="canary-1b-burmese"
    )
)

# ── Train ─────────────────────────────────────────────
print("Starting Canary-1B fine-tuning...")
trainer.fit(model)

# Save final model
model.save_to("/workspace/models/canary-myanmar-final.nemo")
print("✅ Saved canary-myanmar-final.nemo")

# ── Test WER ──────────────────────────────────────────
trainer.test(model)
```

```bash
# On Vast.ai server in tmux
tmux new -s canary
python3 /workspace/scripts/train_canary.py 2>&1 | tee /workspace/logs/canary_train.log
# Ctrl+B, D to detach
```

---

## ⚡ MODEL 2 — Dolphin ASR (Asian Dialect Specialist)

> Dolphin is a fine-tuned Whisper variant specifically optimized for Asian languages including
> Burmese, Thai, Vietnamese, and Chinese. Uses HuggingFace Transformers — familiar workflow.

### Step 9.2A — Install & Setup on Vast.ai (RTX 4090)

```bash
# SSH into RTX 4090 instance
pip install transformers datasets accelerate evaluate jiwer \
            soundfile librosa wandb tensorboard tqdm
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY
```

---

### Step 9.2B — Fine-tune Dolphin ASR

```python
# scripts/train_dolphin.py  (Run on Vast.ai RTX 4090)
import torch
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np

# ── Dolphin ASR model (Asian language specialist) ─────
# Primary choice: fine-tuned for Southeast Asian languages
MODEL_ID  = "openai/whisper-large-v2"          # Dolphin ASR base
# If a specific Dolphin checkpoint is available:
# MODEL_ID = "cognitivecomputations/dolphin-asr-burmese"

DATASET_PATH = "/workspace/data/myanmar_asr"
OUTPUT_DIR   = "/workspace/models/dolphin-myanmar"
HF_REPO      = "YOUR_HF_USERNAME/dolphin-asr-myanmar"
SAMPLING_RATE = 16000

# ── Load processor & model ────────────────────────────
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="burmese", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []

# ── Freeze encoder — only fine-tune decoder ───────────
# This is the KEY trick for Dolphin-style fine-tuning:
# keeps the powerful acoustic encoder intact
for param in model.model.encoder.parameters():
    param.requires_grad = False
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Dataset ───────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare,
            remove_columns=["audio","sentence","source","speaker_id","locale"],
            num_proc=4)

# ── Collator ──────────────────────────────────────────
@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )
        lbl = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lbl[:, 0] == self.processor.tokenizer.bos_token_id).all():
            lbl = lbl[:, 1:]
        inputs["labels"] = lbl
        return inputs

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

# ── Training Arguments ────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,      # RTX 4090 24GB
    gradient_accumulation_steps=4,      # Effective batch = 32
    learning_rate=5e-6,                 # Lower LR — decoder-only tuning
    warmup_ratio=0.05,
    num_train_epochs=8,
    gradient_checkpointing=True,
    fp16=True,                          # RTX 4090 uses fp16 (not bf16)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    report_to=["tensorboard","wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_strategy="every_save",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=DataCollator(processor=processor),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model()
print("✅ Dolphin ASR fine-tuning complete!")
```

---

## 🚀 MODEL 3 — Whisper Large-v3 Turbo (Fastest & Cheapest)

> Whisper Large-v3 Turbo is a distilled version of Whisper Large-v3 —
> same accuracy but **6x faster inference** and fits in **12GB VRAM**.
> Best starting point if you're on a budget.

### Step 9.3A — Vast.ai Instance (Budget — RTX 3090 is fine)

```bash
# Cheapest good option
vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 inet_down>300 disk_space>150 reliability>0.95'

# Instance setup
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel \
  --disk 150
```

---

### Step 9.3B — Fine-tune Whisper Large-v3 Turbo

```python
# scripts/train_whisper_turbo.py  (Run on Vast.ai — RTX 3090 or better)
import torch
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

MODEL_ID     = "openai/whisper-large-v3-turbo"   # ← Key difference
DATASET_PATH = "/workspace/data/myanmar_asr"
OUTPUT_DIR   = "/workspace/models/whisper-turbo-myanmar"
HF_REPO      = "YOUR_HF_USERNAME/whisper-large-v3-turbo-myanmar"
SAMPLING_RATE = 16000

# ── Load ─────────────────────────────────────────────
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="my", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Turbo model: forced decoder IDs MUST be set this way
model.generation_config.language = "my"
model.generation_config.task     = "transcribe"
model.generation_config.forced_decoder_ids = None

# ── Dataset ───────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare,
            remove_columns=["audio","sentence","source","speaker_id","locale"],
            num_proc=4)

# ── Collator ──────────────────────────────────────────
@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]):
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )
        lbl = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lbl[:, 0] == self.processor.tokenizer.bos_token_id).all():
            lbl = lbl[:, 1:]
        inputs["labels"] = lbl
        return inputs

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

# ── Training Args (optimised for RTX 3090 12GB) ───────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,      # 12GB VRAM safe batch
    gradient_accumulation_steps=8,      # Effective batch = 32
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard","wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_strategy="every_save",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=DataCollator(processor=processor),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model()
print("✅ Whisper v3 Turbo fine-tuning complete!")
```

> 💡 **Turbo vs regular Whisper-large-v3:** Turbo is 6x faster at inference, same WER.
> Fine-tune Turbo first to get a baseline WER, then try Large-v3 if you need lower WER.

---

## 🌐 MODEL 4 — SeamlessM4T v2 Large (ASR + Translation)

> Meta's SeamlessM4T v2 is a multi-task model that does:
> - **Speech → Text (ASR)** in Burmese
> - **Speech → Translation** (Burmese audio → English text)
> - **Text → Speech (TTS)**
>
> Best if you need BOTH transcription AND translation in one model.

### Step 9.4A — Vast.ai Instance (RTX 4090 recommended)

```bash
# SeamlessM4T v2 Large needs ~16GB VRAM minimum
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 inet_down>500 disk_space>200 reliability>0.97'

# Setup
pip install transformers datasets accelerate evaluate jiwer soundfile wandb
huggingface-cli login --token $HF_TOKEN
```

---

### Step 9.4B — Fine-tune SeamlessM4T v2

```python
# scripts/train_seamless.py  (Run on Vast.ai RTX 4090)
import torch
import numpy as np
from datasets import load_from_disk, Audio
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List
import evaluate

MODEL_ID     = "facebook/seamless-m4t-v2-large"
DATASET_PATH = "/workspace/data/myanmar_asr"
OUTPUT_DIR   = "/workspace/models/seamless-myanmar"
HF_REPO      = "YOUR_HF_USERNAME/seamless-m4t-v2-myanmar"
SAMPLING_RATE = 16000

# ── Load ─────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_ID)

# ── Freeze speech encoder (only fine-tune decoder) ────
# This saves VRAM and preserves acoustic representations
for name, param in model.named_parameters():
    if "speech_encoder" in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

# ── Dataset Preparation ───────────────────────────────
ds = load_from_disk(DATASET_PATH)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def prepare_seamless(batch):
    audio = batch["audio"]

    # SeamlessM4T uses its own processor
    inputs = processor(
        audios=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        src_lang="mya",                   # ISO 639-3 for Burmese
    )

    labels = processor(
        text=batch["sentence"],
        return_tensors="pt",
        tgt_lang="mya",
    ).input_ids

    batch["input_features"] = inputs.input_features[0]
    batch["labels"]         = labels[0]
    return batch

ds = ds.map(prepare_seamless,
            remove_columns=["audio","sentence","source","speaker_id","locale"],
            num_proc=4)

# ── Collator ──────────────────────────────────────────
@dataclass
class SeamlessCollator:
    processor: Any
    def __call__(self, features: List[Dict]):
        input_features = torch.stack([torch.tensor(f["input_features"]) for f in features])
        # Pad labels
        label_seqs  = [f["labels"] for f in features]
        max_len     = max(len(l) for l in label_seqs)
        padded      = torch.full((len(label_seqs), max_len), -100, dtype=torch.long)
        for i, seq in enumerate(label_seqs):
            padded[i, :len(seq)] = torch.tensor(seq)
        return {"input_features": input_features, "labels": padded}

# ── Metrics ───────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": round(100 * wer_metric.compute(
        predictions=pred_str, references=label_str), 2)}

# ── Training Args ─────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # Effective batch = 32
    learning_rate=5e-6,
    num_train_epochs=6,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to=["tensorboard","wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=SeamlessCollator(processor=processor),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
print("✅ SeamlessM4T v2 fine-tuning complete!")
```

```python
# ── Bonus: Use SeamlessM4T for BOTH ASR + Translation ──
from transformers import pipeline

# Transcribe Burmese audio → Burmese text
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=HF_REPO,
    device=0
)
result = asr_pipe("burmese_audio.wav")
print("Transcription:", result["text"])

# Translate Burmese audio → English text (zero-shot, no extra training!)
translate_pipe = pipeline(
    "translation",
    model="facebook/seamless-m4t-v2-large",   # Use original for translation
    device=0
)
result = translate_pipe("burmese_audio.wav", src_lang="mya", tgt_lang="eng")
print("Translation:", result[0]["translation_text"])
```

---

## 📊 MODEL COMPARISON & DECISION GUIDE

```
Which model should you train first?
─────────────────────────────────────────────────────────

 Budget < $5?
 └─► Whisper v3 Turbo on RTX 3090
     → Quick baseline WER in 4-6 hrs
     → cheapest way to validate your dataset

 Best WER for production?
 └─► Canary-1B on A100 40GB
     → Industry-leading accuracy
     → Requires NeMo framework (different from HF)
     → ~$15-20 for full run

 Need ASR + English translation in ONE model?
 └─► SeamlessM4T v2 on RTX 4090
     → Only model that does both tasks
     → Great for building bilingual apps

 Best Asian dialect accuracy, HF-compatible?
 └─► Dolphin ASR (Whisper-large-v2 based) on RTX 4090
     → Decoder-only fine-tuning = faster convergence
     → Best for Burmese dialect variation

RECOMMENDED TRAINING SEQUENCE:
  Step 1: Whisper v3 Turbo   → Get baseline WER (~$4)
  Step 2: Canary-1B          → Push WER as low as possible (~$18)
  Step 3: SeamlessM4T v2     → Add translation capability (~$8)
```

### Expected WER After Fine-tuning on 50h Myanmar Data

| Model | Expected WER | Expected CER | Inference Speed |
|-------|-------------|-------------|-----------------|
| Whisper v3 Turbo | ~12–18% | ~6–10% | Very Fast |
| Dolphin ASR | ~10–16% | ~5–9% | Fast |
| SeamlessM4T v2 | ~12–18% | ~6–10% | Medium |
| **Canary-1B** | **~7–13%** | **~4–8%** | Medium |

> 📌 WER = Word Error Rate. Lower = better. CER = Character Error Rate (more meaningful for Burmese script).

---

### Step 9.5 — Run Multiple Models & Compare WER

```bash
# Run all 4 training jobs one after another on Vast.ai
# using tmux windows for each

tmux new-session -d -s turbo     "python3 /workspace/scripts/train_whisper_turbo.py"
tmux new-session -d -s dolphin   "python3 /workspace/scripts/train_dolphin.py"
tmux new-session -d -s seamless  "python3 /workspace/scripts/train_seamless.py"
tmux new-session -d -s canary    "python3 /workspace/scripts/train_canary.py"

# List all sessions
tmux ls

# Switch to a session
tmux attach -t canary
```

```python
# scripts/compare_models.py — Run on Vast.ai after all training done
from transformers import pipeline
from datasets import load_from_disk
import evaluate
import pandas as pd

wer_metric = evaluate.load("wer")

MODELS = {
    "Whisper v3 Turbo": "YOUR_HF_USERNAME/whisper-large-v3-turbo-myanmar",
    "Dolphin ASR":      "YOUR_HF_USERNAME/dolphin-asr-myanmar",
    "SeamlessM4T v2":   "YOUR_HF_USERNAME/seamless-m4t-v2-myanmar",
    "Canary-1B":        "/workspace/models/canary-myanmar-final.nemo",  # NeMo local
}

ds_test = load_from_disk("/workspace/data/myanmar_asr")["test"].select(range(200))

results = []
for name, model_id in MODELS.items():
    if "nemo" in model_id:
        continue  # Canary uses different inference (NeMo)
    asr = pipeline("automatic-speech-recognition", model=model_id, device=0)
    preds = [r["text"] for r in asr([a["array"] for a in ds_test["audio"]])]
    refs  = ds_test["sentence"]
    wer   = 100 * wer_metric.compute(predictions=preds, references=refs)
    results.append({"Model": name, "WER (%)": round(wer, 2)})
    print(f"  {name}: WER = {wer:.2f}%")

df = pd.DataFrame(results).sort_values("WER (%)")
print("\n📊 Final Model Comparison:")
print(df.to_string(index=False))
df.to_csv("/workspace/results/model_comparison.csv", index=False)
```

---

## 📋 UPDATED CHECKLIST — Advanced Models

**Phase 9 — Model Fine-tuning:**
- [ ] Vast.ai CLI installed and API key configured
- [ ] NeMo Docker image pulled (for Canary)
- [ ] NeMo JSONL manifests exported (for Canary)
- [ ] Audio files synced to Vast.ai server
- [ ] **Model 1:** Whisper v3 Turbo trained on RTX 3090 (baseline)
- [ ] **Model 2:** Dolphin ASR trained on RTX 4090 (Asian specialist)
- [ ] **Model 3:** SeamlessM4T v2 trained on RTX 4090 (multi-task)
- [ ] **Model 4:** Canary-1B trained on A100 (best accuracy)
- [ ] All models pushed to HuggingFace Hub
- [ ] WER comparison script run
- [ ] Best model selected for production

---

*Guide v3.0 | Myanmar ASR Pipeline | MacBook Pro M5 (Apple Silicon) | 4-Model Comparison on Vast.ai*
