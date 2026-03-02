#!/bin/bash
# ============================================================
# Vast.ai Server Bootstrap Script
# Run this ONCE after SSH-ing into your rented GPU instance.
# Usage:  bash /workspace/scripts/vastai_setup.sh
# ============================================================
set -euo pipefail

echo "=========================================="
echo "  Myanmar ASR — Vast.ai Server Setup"
echo "=========================================="

# ── 1. System packages ───────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq ffmpeg tmux htop git-lfs > /dev/null 2>&1
echo "  ✓ ffmpeg, tmux, htop, git-lfs"

# ── 2. Python packages ──────────────────────────────
echo "[2/6] Installing Python packages..."
pip install -q --upgrade pip
pip install -q \
    transformers>=4.44.0 \
    datasets>=3.0.0 \
    accelerate>=0.34.0 \
    evaluate>=0.4.0 \
    jiwer>=3.0.0 \
    soundfile \
    librosa \
    wandb \
    tensorboard \
    tqdm \
    safetensors \
    sentencepiece \
    protobuf
echo "  ✓ transformers, datasets, accelerate, evaluate, jiwer, wandb, etc."

# ── 3. HuggingFace CLI login ────────────────────────
echo "[3/6] Logging into HuggingFace..."
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "  ✓ HF authenticated"
else
    echo "  ⚠ HF_TOKEN not set — run: huggingface-cli login"
fi

# ── 4. W&B login ────────────────────────────────────
echo "[4/6] Logging into Weights & Biases..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY"
    echo "  ✓ W&B authenticated"
else
    echo "  ⚠ WANDB_API_KEY not set — run: wandb login"
fi

# ── 5. Verify GPU ───────────────────────────────────
echo "[5/6] Checking GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPU:     {torch.cuda.get_device_name(0)}')
print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ── 6. Directory structure ──────────────────────────
echo "[6/6] Creating workspace directories..."
mkdir -p /workspace/{data,models,logs,results,scripts}
echo "  ✓ /workspace/{data,models,logs,results,scripts}"

echo ""
echo "=========================================="
echo "  ✅ Server ready! Next steps:"
echo "  1. rsync data:    see scripts/sync_to_vastai.sh"
echo "  2. Train model:   python3 /workspace/scripts/train_*.py"
echo "  3. Monitor:       https://wandb.ai"
echo "=========================================="
