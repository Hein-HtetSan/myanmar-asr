#!/bin/bash
# ============================================================
# Sync dataset + scripts to Vast.ai GPU server
#
# Usage:
#   bash scripts/sync_to_vastai.sh <VAST_IP> <SSH_PORT>
#
# Example:
#   bash scripts/sync_to_vastai.sh 89.37.121.193 41222
# ============================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <VAST_IP> <SSH_PORT>"
    echo "  Get these from: vastai show instances"
    exit 1
fi

VAST_IP="$1"
SSH_PORT="$2"
SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"

echo "=========================================="
echo "  Syncing to ${VAST_IP}:${SSH_PORT}"
echo "=========================================="

# ── 1. Create remote directories ─────────────────────
echo "[1/5] Creating remote directories..."
$SSH_CMD root@${VAST_IP} "mkdir -p /workspace/{data,scripts,logs,results,models}"

# ── 2. Sync HuggingFace dataset (for Whisper/Dolphin/Seamless) ──
echo "[2/5] Syncing HF dataset (augmented, 58h)..."
echo "  This uploads the Arrow files — fastest for HF models."
rsync -avz --progress \
    combined/myanmar_asr_augmented/ \
    root@${VAST_IP}:/workspace/data/myanmar_asr/ \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"

# ── 3. Sync NeMo manifests (for Canary) ─────────────
echo "[3/5] Syncing NeMo audio + manifests..."
if [ -d "exports/nemo_audio" ]; then
    rsync -avz --progress \
        exports/nemo_audio/ \
        root@${VAST_IP}:/workspace/data/nemo_audio/ \
        -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"

    rsync -avz --progress \
        exports/nemo_*_manifest.jsonl \
        root@${VAST_IP}:/workspace/data/ \
        -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"
else
    echo "  ⚠ NeMo exports not found — run: python3 scripts/export_nemo_manifest.py"
fi

# ── 4. Sync training scripts ────────────────────────
echo "[4/5] Syncing training scripts..."
rsync -avz --progress \
    scripts/train_*.py \
    scripts/evaluate_models.py \
    scripts/vastai_setup.sh \
    root@${VAST_IP}:/workspace/scripts/ \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"

# ── 5. Run server setup ─────────────────────────────
echo "[5/5] Running server bootstrap..."
$SSH_CMD root@${VAST_IP} "bash /workspace/scripts/vastai_setup.sh"

echo ""
echo "=========================================="
echo "  ✅ Sync complete!"
echo "  SSH in:  ssh -p ${SSH_PORT} root@${VAST_IP}"
echo "  Train:   python3 /workspace/scripts/train_whisper_turbo.py"
echo "=========================================="
