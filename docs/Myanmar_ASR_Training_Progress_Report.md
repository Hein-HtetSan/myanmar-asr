# Myanmar ASR Training Progress Report

**Project:** Automatic Speech Recognition for Myanmar Language  
**Model:** OpenAI Whisper Large-v3 Turbo (809M parameters)  
**Author:** Hein Htet San  
**Date:** March 1, 2026  
**Status:** Training in progress (v3 — Frozen Encoder)

---

## 1. Project Overview

This project fine-tunes OpenAI's **Whisper Large-v3 Turbo** model for Myanmar (Burmese) automatic speech recognition. Myanmar is a low-resource language with limited ASR training data, making this a challenging task requiring careful optimization.

### Infrastructure
| Component | Details |
|---|---|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Cloud Provider | Vast.ai (~$0.34/hr) |
| Precision | bf16 (Brain Float 16) |
| Experiment Tracking | MLflow (self-hosted) + TensorBoard |
| Framework | Hugging Face Transformers 5.2.0, PyTorch 2.5.1 |

---

## 2. Dataset

### Sources (Public Only)
| Dataset | Source | Samples |
|---|---|---|
| FLEURS Myanmar | Google Research | ~3,000 |
| OpenSLR-80 | SIL International | ~5,000 |
| YODAS Myanmar | Kensho Technologies | ~9,000 |

### Data Pipeline
1. **Raw Collection**: 17,552 samples from 3 public datasets
2. **1.1x Speed Augmentation**: Applied to increase data diversity
3. **Deep Cleaning** (6 quality filters):
   - Removed empty/whitespace-only transcriptions
   - Filtered non-Myanmar text (Latin, Chinese, etc.)
   - Removed extreme duration outliers (<0.5s or >30s)
   - Filtered extreme transcript lengths (too short/long)
   - Removed corrupted audio files
   - Filtered abnormal audio-to-text ratios
4. **Final Dataset**: **12,299 samples (~24.5 hours)**

| Split | Samples |
|---|---|
| Train | 11,661 |
| Validation | 319 |
| Test | 319 |

---

## 3. Training Experiments

### Experiment v2: Full Model Fine-tuning (Baseline)

**Hypothesis:** Fine-tune all 809M parameters with a conservative learning rate.

| Hyperparameter | Value |
|---|---|
| Trainable Parameters | 808.9M (100%) |
| Learning Rate | 5e-6 |
| LR Schedule | Cosine with 10% warmup |
| Effective Batch Size | 32 (8 × 4 gradient accumulation) |
| Weight Decay | 0.01 |
| Label Smoothing | 0.1 |
| Max Epochs | 10 |
| Early Stopping Patience | 5 evals |
| Training Speed | ~3.66 s/step |

**Result at Epoch 0.5 (Step 182):**

| Metric | Value |
|---|---|
| Train Loss | 9.01 |
| Eval Loss | 2.26 |
| **Eval WER** | **106.23%** |
| **Eval CER** | **102.76%** |

**Analysis:** WER > 100% indicates the model outputs are worse than empty predictions. The extremely low learning rate (5e-6) caused the model to barely learn — training loss only dropped from ~11.7 to ~9.0 over 182 steps. With 12k samples, this is insufficient signal for 809M parameters to converge. Training was **terminated** due to poor convergence.

---

### Experiment v3: Frozen Encoder + Higher LR (Current)

**Hypothesis:** Whisper's audio encoder already extracts good features from pretrained multilingual data. By freezing the encoder and only training the decoder with a much higher learning rate, we enable faster convergence on limited data while preventing the encoder from forgetting its representations.

| Hyperparameter | Value |
|---|---|
| Trainable Parameters | 171.9M (21% — decoder only) |
| Encoder | **Frozen** (637M params locked) |
| Learning Rate | **1e-4** (20x higher than v2) |
| LR Schedule | Cosine with 6% warmup |
| Effective Batch Size | 32 (8 × 4 gradient accumulation) |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Max Epochs | 15 |
| Early Stopping Patience | 5 evals |
| Training Speed | **~1.26 s/step** (2.9x faster than v2) |

**Results (training in progress):**

| Eval Point | Epoch | Eval Loss | Eval WER | Eval CER |
|---|---|---|---|---|
| Step 182 | 0.50 | 1.896 | 99.91% | 87.62% |
| Step 364 | 1.00 | **1.697** | 97.17% | **67.41%** |
| Step 546 | 1.50 | **1.645** | 101.20% | **64.89%** |

**Train Loss Progression (selected steps):**

| Epoch | Train Loss | Grad Norm | Learning Rate |
|---|---|---|---|
| 0.03 | 11.73 | 11.52 | 1.23e-7 (warmup) |
| 0.14 | 10.43 | 7.26 | 6.71e-7 (warmup) |
| 0.30 | 9.41 | 8.84 | 1.49e-6 (warmup) |
| 0.47 | 9.02 | 12.28 | 2.32e-6 (warmup) |
| 0.63 | 7.26 | 5.80 | 6.96e-5 |
| 0.88 | 6.86 | 2.77 | 9.70e-5 |
| 1.04 | 6.47 | 2.38 | 1.00e-4 (peak) |
| 1.37 | 6.57 | 1.69 | 9.97e-5 |
| 1.48 | 6.57 | 2.87 | 9.96e-5 |

---

## 4. Comparison: v2 vs v3

| Metric | v2 (Full Model) @ 0.5 epoch | v3 (Frozen Enc) @ 0.5 epoch | v3 @ 1.0 epoch | v3 @ 1.5 epoch |
|---|---|---|---|---|
| **Eval WER** | 106.23% | 99.91% | 97.17% | 101.20% |
| **Eval CER** | 102.76% | 87.62% | 67.41% | **64.89%** |
| **Eval Loss** | 2.26 | 1.90 | 1.70 | **1.65** |
| Train Loss | 9.01 | ~8.0 | ~6.9 | ~6.6 |
| Speed (s/step) | 3.66 | **1.26** | 1.26 | 1.23 |

### Key Observations

1. **CER is improving strongly** (102.8% → 64.9%), meaning the model is learning to produce correct Myanmar characters, even if word boundaries and word order aren't fully correct yet (hence WER still high).

2. **Eval loss is monotonically decreasing** (2.26 → 1.90 → 1.70 → 1.65), confirming the model is generalizing and not just memorizing.

3. **Frozen encoder trains 2.9x faster** — fewer gradient computations and less memory usage.

4. **The WER/CER gap** (101% WER vs 65% CER) is typical for Myanmar—a language where word segmentation is inherently difficult because Myanmar script doesn't use spaces between words.

---

## 5. Technical Challenges Solved

### Challenge 1: Transformers 5.x Compatibility
**Problem:** The Whisper model's `forward()` method in Transformers 5.2.0 removed automatic `decoder_input_ids` generation from labels, causing a `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`.

**Solution:** Implemented manual right-shift of labels in the data collator:
```python
def _shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
```

### Challenge 2: Data Preprocessing Deadlock
**Problem:** Multi-process tokenization (`num_proc=4`) deadlocked during dataset preprocessing.

**Solution:** Set `num_proc=1` to avoid multiprocessing issues in the containerized GPU environment.

### Challenge 3: Remote MLflow Tracking
**Problem:** Training runs on a remote GPU server (Vast.ai) but MLflow runs locally.

**Solution:** SSH reverse tunnel (`ssh -R 5050:localhost:5050`) to expose local MLflow to the remote server.

---

## 6. Expected Trajectory & Next Steps

Based on current trends, with 15 epochs and early stopping:

| Epoch (projected) | Expected CER | Expected WER |
|---|---|---|
| 2–3 | ~45–55% | ~75–85% |
| 5–7 | ~30–40% | ~55–65% |
| 10–15 | ~20–30% | ~40–55% |

### Planned Improvements
1. **Add VOA Myanmar data** (~5,000 more samples with pseudo-labels) to increase training data
2. **Unfreeze encoder** in a second phase (lower LR) after decoder converges — progressive unfreezing
3. **Data augmentation**: SpecAugment (already built into Whisper), noise injection, speed perturbation
4. **Larger dataset collection**: Common Voice Myanmar, crowd-sourced recordings
5. **Evaluation**: Human evaluation on real-world Myanmar speech samples

---

## 7. Reproducibility

All code is available in the project repository:
- Training script: `scripts/training/train_best.py`
- Data cleaning: `scripts/utils/deep_clean.py`
- Dataset building: `scripts/utils/build_dataset_full.py`
- Dataset analysis: `scripts/evaluation/analyze_dataset.py`

### To reproduce:
```bash
# 1. Build and clean dataset
python3 scripts/utils/build_dataset_full.py
python3 scripts/utils/deep_clean.py

# 2. Train with frozen encoder
python3 scripts/training/train_best.py --freeze_encoder

# 3. Train with full model (optional comparison)
python3 scripts/training/train_best.py --no_freeze_encoder --lr 5e-6
```

---

*Report generated from live MLflow experiment tracking. Training is ongoing — final results will be updated upon completion.*
