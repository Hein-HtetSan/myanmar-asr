#!/usr/bin/env python3
"""
Fine-tune Dolphin ASR (Whisper-large-v2 based) for Myanmar ASR.
Run on Vast.ai: RTX 4090 (24 GB) or better.

Strategy (optimized from Whisper-turbo v3 experiment):
  1. Freeze encoder — decoder-only fine-tuning (proven effective)
  2. Cosine LR schedule (1e-4) with warmup
  3. Label smoothing + weight decay for regularization
  4. bf16 on Ampere+ GPUs
  5. Gradient checkpointing for max batch usage
  6. Step-based eval every ~0.5 epoch
  7. Transformers 5.x compatible (manual decoder_input_ids)
  8. Full MLflow logging: metrics, artifacts, system info

Requires SSH reverse tunnels from Vast.ai → local Mac:
  - :5050 → MLflow tracking server
  - :9002 → MinIO (S3 artifact store)

Usage:
    python3 /workspace/scripts/train_dolphin.py
    # Full fine-tune (needs more VRAM):
    python3 /workspace/scripts/train_dolphin.py --no_freeze_encoder
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ── MLflow + MinIO config (reverse-tunneled to local) ──
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5050"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "myanmar-asr-dolphin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9002"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
# NOTE: HF_MLFLOW_LOG_ARTIFACTS disabled — pyfunc.log_model() crashes on older MLflow server.
# Artifacts are logged manually post-training via MlflowClient.log_artifact().
# os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate
import mlflow
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

# ── Configuration ─────────────────────────────────────
MODEL_ID       = "openai/whisper-large-v2"
DATASET_PATH   = "/workspace/data/myanmar_asr"
OUTPUT_DIR     = "/workspace/models/dolphin-myanmar-v1"
LOG_DIR        = "/workspace/logs/dolphin-v1"
SAMPLING_RATE  = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Dolphin ASR for Myanmar")
    p.add_argument("--model_id",     default=MODEL_ID)
    p.add_argument("--dataset_path", default=DATASET_PATH)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    # ── Batch ──
    p.add_argument("--batch_size",   type=int, default=6,
                   help="Per-device batch (6 for large-v2 on 24GB with grad ckpt)")
    p.add_argument("--grad_accum",   type=int, default=4,
                   help="Gradient accumulation → effective batch = 24")
    # ── LR & Schedule ──
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Higher LR for frozen-encoder decoder-only training")
    p.add_argument("--lr_scheduler", default="cosine",
                   help="Cosine generally outperforms linear for ASR")
    p.add_argument("--warmup_ratio", type=float, default=0.06,
                   help="6%% warmup")
    p.add_argument("--weight_decay", type=float, default=0.05,
                   help="Regularization for decoder-only")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for regularization")
    # ── Epochs ──
    p.add_argument("--epochs",       type=int, default=15,
                   help="More epochs with early stopping")
    p.add_argument("--patience",     type=int, default=5,
                   help="Early stopping patience (in eval intervals)")
    # ── Eval/Save ──
    p.add_argument("--eval_steps",   type=int, default=243,
                   help="Eval every ~0.5 epoch (11661/24=486 steps/epoch)")
    p.add_argument("--save_steps",   type=int, default=243)
    p.add_argument("--save_total_limit", type=int, default=5)
    # ── Other ──
    p.add_argument("--freeze_encoder", action="store_true", default=True,
                   help="Freeze encoder (default for low-resource)")
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--max_length",   type=int, default=225,
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
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int = 50258

    def __post_init__(self):
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Manually create decoder_input_ids (transformers 5.x removed auto-shift)
        batch["decoder_input_ids"] = _shift_tokens_right(
            labels, self.pad_token_id, self.decoder_start_token_id
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

    # ── Load processor & model ────────────────────────
    print(f"\nLoading {args.model_id}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_id, language="my", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32
    )

    # Configure for Myanmar
    model.generation_config.language = "my"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder FROZEN — decoder-only training (Dolphin style)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.1f}%)")

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
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids[:448]
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
        per_device_eval_batch_size=args.batch_size * 2,
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
        run_name="dolphin-myanmar-v1-frozen-enc",
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
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
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
    print(f"  DOLPHIN ASR (Whisper-large-v2) — MYANMAR")
    print(f"{'='*60}")
    print(f"  Encoder frozen      : {args.freeze_encoder}")
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

    # ── MLflow: Pre-create run so HF callback + system logger share it ──
    experiment = mlflow.set_experiment("myanmar-asr-dolphin")
    client = mlflow.MlflowClient()
    mlflow_run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name="dolphin-myanmar-v1-frozen-enc",
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

    trainer.train()

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
    print(f"  DOLPHIN — TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test WER : {test_results['test_wer']:.2f}%")
    print(f"  Test CER : {test_results['test_cer']:.2f}%")
    print(f"  Test Loss: {test_results['test_loss']:.4f}")
    print(f"{'='*60}")

    training_time_min = round((time.time() - t0) / 60, 1)

    # Save results to JSON
    os.makedirs("/workspace/results", exist_ok=True)
    results_file = "/workspace/results/dolphin_v1_results.json"
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
            "freeze_encoder": args.freeze_encoder,
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
        # Reuse the pre-created run_id (shared with HF callback + system logger)
        # HF callback already ended the run, but MlflowClient can still log artifacts
        client = mlflow.MlflowClient()

        # Log test metrics explicitly
        client.log_metric(run_id, "test_wer", test_results["test_wer"])
        client.log_metric(run_id, "test_cer", test_results["test_cer"])
        client.log_metric(run_id, "test_loss", test_results["test_loss"])
        client.log_metric(run_id, "training_time_min", training_time_min)

        # Log system info as tags
        client.set_tag(run_id, "model.base", args.model_id)
        client.set_tag(run_id, "model.type", "whisper-large-v2")
        client.set_tag(run_id, "training.strategy", "frozen-encoder" if args.freeze_encoder else "full")
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

        # Log model weights (3+ GB — this takes a few minutes via tunnel)
        model_weights = os.path.join(final_dir, "model.safetensors")
        if os.path.exists(model_weights):
            sz_gb = os.path.getsize(model_weights) / (1024**3)
            print(f"    -> Uploading model.safetensors ({sz_gb:.1f} GB)...")
            upload_t0 = time.time()
            client.log_artifact(run_id, model_weights, artifact_path="model")
            upload_time = time.time() - upload_t0
            print(f"    -> model.safetensors uploaded in {upload_time:.0f}s")

        # Create and log training summary
        summary = f"""# Dolphin ASR (Whisper-large-v2) Myanmar - Training Summary

## Model
- Base Model: {args.model_id}
- Fine-tuned: {'Decoder only (encoder frozen)' if args.freeze_encoder else 'Full model'}
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
        summary_path = "/tmp/dolphin_training_summary.md"
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
