#!/usr/bin/env python3
"""
Fine-tune Meta SeamlessM4T v2 Large for Myanmar ASR.
Run on Vast.ai: RTX 4090 (24 GB) or better.

This model supports:
  - Burmese speech → Burmese text (ASR)
  - Burmese speech → English text (translation — zero-shot after ASR fine-tune)

Strategy (optimized from Whisper-turbo v3 experiment):
  1. Freeze speech encoder — fine-tune text decoder + adaptor
  2. Cosine LR schedule (5e-5) with warmup
  3. Label smoothing + weight decay for regularization
  4. bf16 on Ampere+ GPUs
  5. Gradient checkpointing for max batch usage
  6. Step-based eval every ~0.5 epoch
  7. Transformers 5.x compatible (manual decoder_input_ids)

Usage:
    python3 /workspace/scripts/train_seamless.py
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

# ── MLflow + MinIO config (reverse-tunneled to local) ──
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5050"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "myanmar-asr-seamless"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9002"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
# NOTE: HF_MLFLOW_LOG_ARTIFACTS disabled — pyfunc.log_model() crashes on older MLflow.
# Artifacts are logged manually post-training via MlflowClient.log_artifact().
# os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"

from datasets import load_from_disk, Audio
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate
import mlflow
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

# ── Configuration ─────────────────────────────────────
MODEL_ID       = "facebook/seamless-m4t-v2-large"
DATASET_PATH   = "/workspace/data/myanmar_asr"
OUTPUT_DIR     = "/workspace/models/seamless-myanmar-v1"
LOG_DIR        = "/workspace/logs/seamless-v1"
SAMPLING_RATE  = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SeamlessM4T v2 for Myanmar ASR")
    p.add_argument("--model_id",     default=MODEL_ID)
    p.add_argument("--dataset_path", default=DATASET_PATH)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    # ── Batch ──
    p.add_argument("--batch_size",   type=int, default=4,
                   help="Per-device batch (4 for seamless-large on 24GB)")
    p.add_argument("--grad_accum",   type=int, default=8,
                   help="Gradient accumulation → effective batch = 32")
    # ── LR & Schedule ──
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="Moderate LR for SeamlessM4T (larger model, more params)")
    p.add_argument("--lr_scheduler", default="cosine",
                   help="Cosine outperforms linear for ASR fine-tuning")
    p.add_argument("--warmup_ratio", type=float, default=0.08,
                   help="8%% warmup — slightly longer for larger model")
    p.add_argument("--weight_decay", type=float, default=0.05,
                   help="Regularization")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for regularization")
    # ── Epochs ──
    p.add_argument("--epochs",       type=int, default=12,
                   help="More epochs with early stopping")
    p.add_argument("--patience",     type=int, default=5,
                   help="Early stopping patience (in eval intervals)")
    # ── Eval/Save ──
    p.add_argument("--eval_steps",   type=int, default=182,
                   help="Eval every ~0.5 epoch (11661/32=364 steps/epoch)")
    p.add_argument("--save_steps",   type=int, default=182)
    p.add_argument("--save_total_limit", type=int, default=5)
    # ── Other ──
    p.add_argument("--freeze_speech_encoder", action="store_true", default=True,
                   help="Freeze speech encoder (default)")
    p.add_argument("--no_freeze_speech_encoder", dest="freeze_speech_encoder", action="store_false")
    p.add_argument("--max_length",   type=int, default=256,
                   help="Max generation length for eval")
    return p.parse_args()


# ── Transformers 5.x compatibility ────────────────────
def _shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Shift labels right for decoder input: prepend decoder_start_token_id, replace -100 with pad."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


@dataclass
class SeamlessDataCollator:
    """Data collator for SeamlessM4T v2 with variable-length spectrograms."""
    pad_token_id: int = 0
    decoder_start_token_id: int = 3

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ── Pad input features (variable length spectrograms) ──
        # SeamlessM4T input_features shape: (time_steps, n_mels) — pad along time axis (dim 0)
        input_feats = [f["input_features"] for f in features]

        # Convert to tensors
        if isinstance(input_feats[0], np.ndarray):
            input_feats = [torch.tensor(f) for f in input_feats]
        elif isinstance(input_feats[0], list):
            input_feats = [torch.tensor(f) for f in input_feats]

        max_time = max(f.shape[0] for f in input_feats)
        n_mels = input_feats[0].shape[1]
        padded_inputs = []
        attention_masks = []

        for feat in input_feats:
            t = feat.shape[0]
            pad_len = max_time - t
            if pad_len > 0:
                # Pad along time dimension (dim 0): (0, 0) = no mel pad, (0, pad_len) = time pad
                padded_inputs.append(
                    torch.nn.functional.pad(feat.float(), (0, 0, 0, pad_len))
                )
                mask = torch.cat([torch.ones(t, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
                attention_masks.append(mask)
            else:
                padded_inputs.append(feat.float())
                attention_masks.append(torch.ones(t, dtype=torch.long))

        # ── Pad labels ──
        label_seqs = [f["labels"] for f in features]
        if isinstance(label_seqs[0], list):
            label_seqs_t = [torch.tensor(s, dtype=torch.long) for s in label_seqs]
        else:
            label_seqs_t = [s.long() for s in label_seqs]

        max_label_len = max(len(s) for s in label_seqs_t)
        padded_labels = torch.full((len(label_seqs_t), max_label_len), -100, dtype=torch.long)
        for i, seq in enumerate(label_seqs_t):
            padded_labels[i, :len(seq)] = seq

        batch = {
            "input_features": torch.stack(padded_inputs),
            "attention_mask": torch.stack(attention_masks),
            "labels": padded_labels,
        }

        # Manually create decoder_input_ids (transformers 5.x compatibility)
        batch["decoder_input_ids"] = _shift_tokens_right(
            padded_labels, self.pad_token_id, self.decoder_start_token_id
        )

        return batch


def main():
    args = parse_args()
    t0 = time.time()

    # ── GPU info ──────────────────────────────────────
    device_name = "CPU"
    use_bf16 = False
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        print(f"GPU: {device_name} ({vram:.1f} GB)")
        print(f"  bf16: {'YES' if use_bf16 else 'NO (using fp16)'}")

    # ── Load model ────────────────────────────────────
    print(f"\nLoading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model_id, torch_dtype=torch.float32
    )

    # Freeze speech encoder to save VRAM and preserve acoustic features
    if args.freeze_speech_encoder:
        for name, param in model.named_parameters():
            if "speech_encoder" in name:
                param.requires_grad = False
        print("  Speech encoder FROZEN — fine-tuning text decoder + adaptor")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.1f}%)")

    # Get decoder config for data collator
    pad_token_id = getattr(model.config, "pad_token_id", 0)
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", 3)
    print(f"  pad_token_id={pad_token_id}, decoder_start_token_id={decoder_start_token_id}")

    # ── Load dataset ──────────────────────────────────
    print(f"\nLoading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    n_train = len(ds["train"])
    n_val   = len(ds["validation"])
    n_test  = len(ds["test"])
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    steps_per_epoch = n_train // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
    print(f"  Eval every {args.eval_steps} steps ({args.eval_steps/steps_per_epoch:.1f} epochs)")

    # ── Preprocessing ─────────────────────────────────
    def prepare_dataset(batch):
        audio = batch["audio"]
        # SeamlessM4T processor for audio
        inputs = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            src_lang="mya",  # ISO 639-3 for Burmese
        )
        # Text labels
        labels = processor(
            text=batch["sentence"],
            return_tensors="pt",
            tgt_lang="mya",
        ).input_ids

        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = labels[0].tolist()[:448]
        return batch

    remove_cols = [c for c in ds["train"].column_names if c not in ("input_features", "labels")]
    print("\nPreprocessing dataset...")
    ds = ds.map(
        prepare_dataset,
        remove_columns=remove_cols,
        num_proc=1,
        desc="Preprocessing",
    )

    # ── Metrics ───────────────────────────────────────
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Handle -100 padding
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        # Clip prediction IDs to valid tokenizer range to avoid OverflowError
        vocab_size = processor.tokenizer.vocab_size
        pred_ids = np.clip(pred_ids, 0, vocab_size - 1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 2), "cer": round(cer, 2)}

    # ── Training arguments ────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # no increase (model is large)
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        num_train_epochs=args.epochs,
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=(not use_bf16 and torch.cuda.is_available()),
        bf16=use_bf16,
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_length,
        # Logging
        logging_dir=LOG_DIR,
        logging_steps=10,
        report_to=["tensorboard", "mlflow"],
        run_name="seamless-myanmar-v1-frozen-enc",
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Dataloader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Hub — disabled (push manually)
        push_to_hub=False,
    )

    # ── Trainer ───────────────────────────────────────
    data_collator = SeamlessDataCollator(
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── Print config ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SeamlessM4T v2 Large — MYANMAR ASR")
    print(f"{'='*60}")
    print(f"  Speech encoder frozen: {args.freeze_speech_encoder}")
    print(f"  Effective batch size : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate       : {args.lr}")
    print(f"  LR scheduler        : {args.lr_scheduler}")
    print(f"  Warmup              : {args.warmup_ratio*100:.0f}%")
    print(f"  Weight decay        : {args.weight_decay}")
    print(f"  Label smoothing     : {args.label_smoothing}")
    print(f"  Epochs (max)        : {args.epochs}")
    print(f"  Early stop patience : {args.patience}")
    print(f"  Precision           : {'bf16' if use_bf16 else 'fp16'}")
    print(f"  Eval every          : {args.eval_steps} steps")
    print(f"  Total train steps   : ~{total_steps}")
    print(f"{'='*60}\n")

    # ── MLflow: Pre-create run so HF callback + system monitor share it ──
    experiment = mlflow.set_experiment("myanmar-asr-seamless")
    client = mlflow.MlflowClient()
    mlflow_run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name="seamless-myanmar-v1-frozen-enc",
    )
    run_id = mlflow_run.info.run_id
    os.environ["MLFLOW_RUN_ID"] = run_id   # HF callback will activate this run
    print(f"  MLflow run: {run_id}")
    print(f"  View at: http://localhost:5050/#/experiments/{experiment.experiment_id}/runs/{run_id}")

    # ── Train ─────────────────────────────────────────
    # Start MLflow's built-in system metrics monitor (appears in System Metrics tab)
    sys_monitor = SystemMetricsMonitor(
        run_id=run_id,
        sampling_interval=10,       # collect every 10s
        samples_before_logging=6,   # log every 60s (6 × 10s)
    )
    sys_monitor.start()
    print("  System metrics monitor started (MLflow native → System Metrics tab)")

    # Resume from checkpoint if available
    ckpt_dir = args.output_dir
    resume_ckpt = None
    if os.path.isdir(ckpt_dir):
        ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if ckpts:
            resume_ckpt = os.path.join(ckpt_dir, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])
            print(f"  Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Stop system metrics monitor
    sys_monitor.finish()
    print("  System metrics monitor stopped")

    # ── Save final model ──────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\n  Best model saved to {final_dir}")

    # ── Evaluate on test set ──────────────────────────
    print("\n  Evaluating on test set...")
    test_results = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(f"\n{'='*60}")
    print(f"  SEAMLESS — TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test WER : {test_results['test_wer']:.2f}%")
    print(f"  Test CER : {test_results['test_cer']:.2f}%")
    print(f"  Test Loss: {test_results['test_loss']:.4f}")
    print(f"{'='*60}")

    # Save results
    training_time_min = round((time.time() - t0) / 60, 1)
    os.makedirs("/workspace/results", exist_ok=True)
    results_file = "/workspace/results/seamless_v1_results.json"
    results_data = {
        "test_wer": test_results["test_wer"],
        "test_cer": test_results["test_cer"],
        "test_loss": test_results["test_loss"],
        "model": args.model_id,
        "dataset": args.dataset_path,
        "config": {
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "precision": "bf16" if use_bf16 else "fp16",
            "freeze_speech_encoder": args.freeze_speech_encoder,
        },
        "dataset_stats": {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
        },
        "training_time_min": training_time_min,
    }
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    # ── MLflow: Log artifacts & final info ────────────
    print("\n  Logging artifacts to MLflow...")
    try:
        client = mlflow.MlflowClient()

        # Log test metrics explicitly
        client.log_metric(run_id, "test_wer", test_results["test_wer"])
        client.log_metric(run_id, "test_cer", test_results["test_cer"])
        client.log_metric(run_id, "test_loss", test_results["test_loss"])
        client.log_metric(run_id, "training_time_min", training_time_min)

        # Log system info as tags
        client.set_tag(run_id, "model.base", args.model_id)
        client.set_tag(run_id, "model.type", "seamless-m4t-v2-large")
        client.set_tag(run_id, "training.strategy", "frozen-speech-encoder" if args.freeze_speech_encoder else "full")
        client.set_tag(run_id, "test.wer", f"{test_results['test_wer']:.2f}")
        client.set_tag(run_id, "test.cer", f"{test_results['test_cer']:.2f}")
        client.set_tag(run_id, "training.time_min", str(training_time_min))
        client.set_tag(run_id, "gpu", device_name)
        client.set_tag(run_id, "model.location", final_dir)

        # Log all hyperparameters as params
        for k, v in results_data["config"].items():
            client.log_param(run_id, k, v)
        client.log_param(run_id, "trainable_params_m", f"{trainable/1e6:.1f}")
        client.log_param(run_id, "total_params_m", f"{total/1e6:.1f}")

        # Log results JSON
        client.log_artifact(run_id, results_file)
        print(f"    -> Results JSON logged")

        # Log model config files (small)
        model_config_files = [
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "processor_config.json", "training_args.bin",
        ]
        for fname in model_config_files:
            fpath = os.path.join(final_dir, fname)
            if os.path.exists(fpath):
                client.log_artifact(run_id, fpath, artifact_path="model")
        print(f"    -> Model config files logged")

        # Log model weights (this takes a few minutes via tunnel)
        model_weights = os.path.join(final_dir, "model.safetensors")
        if os.path.exists(model_weights):
            sz_gb = os.path.getsize(model_weights) / (1024**3)
            print(f"    -> Uploading model.safetensors ({sz_gb:.1f} GB)...")
            upload_t0 = time.time()
            client.log_artifact(run_id, model_weights, artifact_path="model")
            upload_time = time.time() - upload_t0
            print(f"    -> model.safetensors uploaded in {upload_time:.0f}s")

        # Create and log training summary
        summary = f"""# SeamlessM4T v2 Large Myanmar - Training Summary

## Model
- Base Model: {args.model_id}
- Fine-tuned: {'Text decoder + adaptor (speech encoder frozen)' if args.freeze_speech_encoder else 'Full model'}
- Trainable Params: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)
- GPU: {device_name}

## Final Test Results
| Metric | Value |
|--------|-------|
| WER    | {test_results['test_wer']:.2f}% |
| CER    | {test_results['test_cer']:.2f}% |
| Loss   | {test_results['test_loss']:.4f} |

## Training Configuration
- LR: {args.lr} ({args.lr_scheduler}), Warmup: {args.warmup_ratio*100:.0f}%
- Weight Decay: {args.weight_decay}, Label Smoothing: {args.label_smoothing}
- Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective
- Precision: {'bf16' if use_bf16 else 'fp16'}
- Eval every {args.eval_steps} steps, Patience: {args.patience}

## Dataset
- Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}

## Training Time: {training_time_min} min
"""
        summary_path = "/tmp/seamless_training_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        client.log_artifact(run_id, summary_path)
        print(f"    -> Training summary logged")

        print(f"  MLflow run: http://localhost:5050/#/experiments/{experiment.experiment_id}/runs/{run_id}")

    except Exception as e:
        print(f"  WARNING: MLflow artifact logging failed: {e}")
        print(f"  (Metrics were still logged via HF callback)")

    print(f"\n  Results saved to {results_file}")
    print(f"  Total time: {training_time_min} min")
    print(f"\n  TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
