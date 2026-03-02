#!/usr/bin/env python3
"""
Fine-tune Whisper Large-v3 Turbo on Myanmar ASR data.
Run on Vast.ai: RTX 3090 (12 GB) or better.

Usage:
    python3 /workspace/scripts/train_whisper_turbo.py
    # or with custom args:
    python3 /workspace/scripts/train_whisper_turbo.py \
        --epochs 5 --batch_size 4 --lr 1e-5

Experiment tracking: W&B + TensorBoard
"""

import argparse
import os
import sys
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

# MLflow tracking (via SSH reverse tunnel from Vast.ai → local)
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_callback import (
    get_mlflow_callback, get_system_metrics_callback,
    log_final_results, log_model_artifacts,
    log_dataset_info, log_training_config,
)

# ── Defaults ──────────────────────────────────────────
MODEL_ID       = "openai/whisper-large-v3-turbo"
DATASET_PATH   = "/workspace/data/myanmar_asr"
OUTPUT_DIR     = "/workspace/models/whisper-turbo-myanmar"
LOG_DIR        = "/workspace/logs/whisper-turbo"
HF_REPO        = "devhnhts/whisper-large-v3-turbo-myanmar"
SAMPLING_RATE  = 16000
WANDB_PROJECT  = "myanmar-asr"
WANDB_RUN_NAME = "whisper-v3-turbo"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper v3 Turbo for Myanmar ASR")
    p.add_argument("--model_id",     default=MODEL_ID)
    p.add_argument("--dataset_path", default=DATASET_PATH)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    p.add_argument("--hf_repo",      default=HF_REPO)
    p.add_argument("--batch_size",   type=int, default=4,    help="Per-device batch size")
    p.add_argument("--grad_accum",   type=int, default=8,    help="Gradient accumulation steps")
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--max_steps",    type=int, default=-1,   help="Override epochs if > 0")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_steps",   type=int, default=500)
    p.add_argument("--save_steps",   type=int, default=500)
    p.add_argument("--patience",     type=int, default=3,    help="Early stopping patience")
    p.add_argument("--freeze_encoder", action="store_true",  help="Freeze encoder, train decoder only")
    p.add_argument("--no_push",      action="store_true",    help="Don't push to HF Hub")
    return p.parse_args()


# ── Data Collator ─────────────────────────────────────
@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present at start
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    args = parse_args()

    # ── GPU info ──────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("⚠ No GPU detected — training will be very slow!")

    # ── Load processor & model ────────────────────────
    print(f"Loading {args.model_id}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_id, language="my", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32
    )

    # Configure generation for Myanmar
    model.generation_config.language = "my"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # Optionally freeze encoder (saves VRAM, faster convergence)
    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen — decoder-only training")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # ── Load dataset ──────────────────────────────────
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    print(f"  Train: {len(ds['train']):,} | Val: {len(ds['validation']):,} | Test: {len(ds['test']):,}")

    # ── Preprocessing ─────────────────────────────────
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        # Truncate labels to Whisper's max length (448 tokens)
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids[:448]
        return batch

    # Determine columns to remove
    remove_cols = [c for c in ds["train"].column_names if c not in ("input_features", "labels")]
    print("Preprocessing dataset...")
    ds = ds.map(
        prepare_dataset,
        remove_columns=remove_cols,
        num_proc=1,
        desc="Tokenizing",
    )

    # ── Metrics ───────────────────────────────────────
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
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
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 100,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        bf16=False,
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        # Generation
        predict_with_generate=True,
        generation_max_length=225,
        # Logging
        logging_dir=LOG_DIR,
        logging_steps=25,
        report_to=["tensorboard", "mlflow"],
        run_name=WANDB_RUN_NAME,
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Hub
        push_to_hub=not args.no_push,
        hub_model_id=args.hf_repo if not args.no_push else None,
        hub_strategy="every_save",
    )

    # ── Trainer ───────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=WhisperDataCollator(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── MLflow setup ──────────────────────────────────
    mlflow_cb = get_mlflow_callback(WANDB_RUN_NAME)
    if mlflow_cb:
        trainer.add_callback(mlflow_cb)
    sys_cb = get_system_metrics_callback(log_every_n_steps=25)
    if sys_cb:
        trainer.add_callback(sys_cb)

    # Log dataset & config to MLflow
    log_dataset_info(ds)
    log_training_config(training_args.to_dict(), model.config, "whisper-v3-turbo")

    # ── Train ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting Whisper v3 Turbo fine-tuning")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"{'='*60}\n")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    # Resume from checkpoint if available
    ckpt_dir = args.output_dir
    resume_ckpt = None
    if os.path.isdir(ckpt_dir):
        ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if ckpts:
            resume_ckpt = os.path.join(ckpt_dir, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])
            print(f"  Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save final model ──────────────────────────────
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))

    # ── Evaluate on test set ──────────────────────────
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(f"  Test WER: {test_results['test_wer']:.2f}%")
    print(f"  Test CER: {test_results['test_cer']:.2f}%")

    # Save test results
    os.makedirs("/workspace/results", exist_ok=True)
    import json
    with open("/workspace/results/whisper_turbo_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Log to MLflow
    log_final_results(test_results, "whisper-v3-turbo", args.output_dir)
    log_model_artifacts(os.path.join(args.output_dir, "final"), "whisper-v3-turbo")

    print(f"\n✅ Whisper v3 Turbo training complete!")
    print(f"   Model saved to: {args.output_dir}/final")
    if not args.no_push:
        print(f"   HF Hub: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
