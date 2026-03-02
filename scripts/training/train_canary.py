#!/usr/bin/env python3
"""
Fine-tune NVIDIA Canary-1B on Myanmar ASR data using NeMo.
Run on Vast.ai: RTX 4090 (24 GB) or A100.

Strategy (optimized from Whisper-turbo v3 experiment):
  1. Freeze encoder — fine-tune decoder + joint layers only
  2. Cosine LR schedule with warmup
  3. bf16-mixed precision
  4. Gradient accumulation for effective batch = 32
  5. Early stopping on val_wer
  6. MLflow tracking via PyTorch Lightning logger

Usage:
    python3 /workspace/scripts/train_canary.py
    # Full fine-tune (needs A100 40GB+):
    python3 /workspace/scripts/train_canary.py --no_freeze_encoder

Requires:
    pip install nemo_toolkit['asr'] pytorch-lightning
"""

import argparse
import os
import sys
import json
import time
from copy import deepcopy

# ── Check NeMo availability ──────────────────────────
try:
    import nemo
    import nemo.collections.asr as nemo_asr
    import pytorch_lightning as pl
    from omegaconf import OmegaConf, open_dict
    print(f"NeMo version: {nemo.__version__}")
except ImportError:
    print("ERROR: NeMo not installed!")
    print("  Option 1: Use Docker image nvcr.io/nvidia/nemo:24.01")
    print("  Option 2: pip install nemo_toolkit['asr']")
    sys.exit(1)

import torch

# ── Configuration ─────────────────────────────────────
TRAIN_MANIFEST = "/workspace/data/nemo_train_manifest.jsonl"
VAL_MANIFEST   = "/workspace/data/nemo_val_manifest.jsonl"
TEST_MANIFEST  = "/workspace/data/nemo_test_manifest.jsonl"
OUTPUT_DIR     = "/workspace/models/canary-myanmar-v1"
LOG_DIR        = "/workspace/logs/canary-v1"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Canary-1B for Myanmar ASR")
    # ── Data ──
    p.add_argument("--train_manifest", default=TRAIN_MANIFEST)
    p.add_argument("--val_manifest",   default=VAL_MANIFEST)
    p.add_argument("--test_manifest",  default=TEST_MANIFEST)
    p.add_argument("--output_dir",     default=OUTPUT_DIR)
    # ── Batch ──
    p.add_argument("--batch_size",   type=int, default=6,
                   help="Per-device batch (6 for RTX 4090 24GB)")
    p.add_argument("--grad_accum",   type=int, default=5,
                   help="Gradient accumulation → effective batch ≈ 30")
    # ── LR & Schedule ──
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Higher LR for frozen-encoder decoder-only training")
    p.add_argument("--min_lr",       type=float, default=1e-6,
                   help="Minimum LR for cosine schedule")
    p.add_argument("--warmup_ratio", type=float, default=0.06,
                   help="6%% of total steps for warmup")
    p.add_argument("--weight_decay", type=float, default=0.05)
    # ── Epochs ──
    p.add_argument("--epochs",       type=int, default=15,
                   help="Max epochs (early stopping will cut short)")
    p.add_argument("--patience",     type=int, default=5,
                   help="Early stopping patience (validation checks)")
    # ── Audio ──
    p.add_argument("--max_duration", type=float, default=30.0,
                   help="Max audio duration in seconds")
    p.add_argument("--min_duration", type=float, default=0.5,
                   help="Min audio duration in seconds")
    # ── Encoder ──
    p.add_argument("--freeze_encoder", action="store_true", default=True,
                   help="Freeze encoder (default for low-resource)")
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    return p.parse_args()


def verify_manifests(train_path, val_path, test_path):
    """Check that all manifests exist and count samples."""
    counts = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found!")
            print(f"  Run: python3 scripts/export_nemo_manifest.py")
            sys.exit(1)
        count = sum(1 for _ in open(path))
        counts[name] = count
        print(f"  {name}: {count:,} samples → {path}")
    return counts


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("  NVIDIA Canary-1B — Myanmar ASR Fine-tuning")
    print("=" * 60)

    # ── GPU check ─────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    print(f"GPU: {gpu} ({vram:.1f} GB)")
    print(f"  bf16: {'YES' if use_bf16 else 'NO (using fp16)'}")

    if vram < 20:
        print(f"  WARNING: Canary-1B needs 24GB+ VRAM. You have {vram:.1f}GB.")
        print("  Consider reducing batch_size to 2-4.")

    # ── Verify data ───────────────────────────────────
    print("\nVerifying manifests...")
    counts = verify_manifests(args.train_manifest, args.val_manifest, args.test_manifest)

    # ── Load pretrained model ─────────────────────────
    print("\nLoading NVIDIA Canary-1B (this may take a few minutes)...")
    model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    cfg = model.cfg

    # ── Freeze encoder ────────────────────────────────
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
        print("  Encoder FROZEN — decoder-only fine-tuning")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.1f}%)")

    # ── Update NeMo config ────────────────────────────
    with open_dict(cfg):
        # Training data
        cfg.train_ds.manifest_filepath = args.train_manifest
        cfg.train_ds.batch_size        = args.batch_size
        cfg.train_ds.num_workers       = 4
        cfg.train_ds.shuffle           = True
        cfg.train_ds.max_duration      = args.max_duration
        cfg.train_ds.min_duration      = args.min_duration

        # Validation data
        cfg.validation_ds.manifest_filepath = args.val_manifest
        cfg.validation_ds.batch_size        = args.batch_size
        cfg.validation_ds.num_workers       = 4

        # Optimizer — CosineAnnealing with warmup
        cfg.optim.name         = "adamw"
        cfg.optim.lr           = args.lr
        cfg.optim.weight_decay = args.weight_decay
        cfg.optim.betas        = [0.9, 0.98]

        # Cosine annealing scheduler
        cfg.optim.sched.name           = "CosineAnnealing"
        cfg.optim.sched.warmup_ratio   = args.warmup_ratio
        cfg.optim.sched.min_lr         = args.min_lr

        # Disable N-gram LM if present
        if hasattr(cfg, "decoding") and hasattr(cfg.decoding, "beam"):
            cfg.decoding.beam.ngram_lm_model = None

    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)

    # ── Callbacks ─────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="canary-{epoch:02d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=args.patience,
        verbose=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # ── Loggers ───────────────────────────────────────
    loggers = [
        pl.loggers.TensorBoardLogger(save_dir=LOG_DIR, name="canary"),
    ]

    # Add MLflow logger if available
    try:
        mlflow_logger = pl.loggers.MLFlowLogger(
            experiment_name="myanmar-asr-canary",
            tracking_uri="http://localhost:5050",
            run_name="canary-1b-myanmar-v1-frozen-enc",
            log_model=False,
        )
        loggers.append(mlflow_logger)
        print("  MLflow tracking: http://localhost:5050")
    except Exception as e:
        print(f"  MLflow not available: {e}")

    # ── Compute steps ─────────────────────────────────
    steps_per_epoch = counts["train"] // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    val_check = max(1, steps_per_epoch // 2)  # ~2x per epoch

    # ── Trainer ───────────────────────────────────────
    precision = "bf16-mixed" if use_bf16 else "16-mixed"

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        precision=precision,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
        val_check_interval=val_check,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=loggers,
    )

    # ── Print config ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CANARY-1B — MYANMAR ASR")
    print(f"{'='*60}")
    print(f"  Encoder frozen      : {args.freeze_encoder}")
    print(f"  Effective batch size : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate       : {args.lr}")
    print(f"  LR scheduler        : CosineAnnealing")
    print(f"  Warmup              : {args.warmup_ratio*100:.0f}%")
    print(f"  Weight decay        : {args.weight_decay}")
    print(f"  Epochs (max)        : {args.epochs}")
    print(f"  Early stop patience : {args.patience}")
    print(f"  Precision           : {precision}")
    print(f"  Steps/epoch         : ~{steps_per_epoch}")
    print(f"  Val check every     : {val_check} steps")
    print(f"  Total train steps   : ~{total_steps}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────
    trainer.fit(model)
    train_time = time.time() - t0

    # ── Save final .nemo model ────────────────────────
    final_path = os.path.join(args.output_dir, "canary-myanmar-v1-final.nemo")
    model.save_to(final_path)
    print(f"\n  Model saved to {final_path}")

    # ── Evaluate on test set ──────────────────────────
    print("\n  Evaluating on test set...")
    with open_dict(cfg):
        cfg.test_ds = deepcopy(cfg.validation_ds)
        cfg.test_ds.manifest_filepath = args.test_manifest
    model.setup_test_data(cfg.test_ds)
    test_results = trainer.test(model)

    # ── Save results ──────────────────────────────────
    os.makedirs("/workspace/results", exist_ok=True)
    results_file = "/workspace/results/canary_v1_results.json"
    results = {
        "model": "nvidia/canary-1b",
        "test_results": test_results,
        "config": {
            "lr": args.lr,
            "lr_scheduler": "CosineAnnealing",
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "precision": precision,
            "freeze_encoder": args.freeze_encoder,
        },
        "dataset_stats": counts,
        "training_time_min": round(train_time / 60, 1),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CANARY — TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training time : {train_time/60:.1f} min")
    print(f"  Model         : {final_path}")
    print(f"  Results       : {results_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
