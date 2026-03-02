#!/bin/bash
# ============================================================
# Myanmar ASR — Vast.ai Orchestrator
#
# Master script to search, rent, set up, and launch training
# on a Vast.ai GPU instance.
#
# Usage:
#   bash scripts/vastai_run.sh search          # Search for GPU offers
#   bash scripts/vastai_run.sh rent <OFFER_ID> <MODEL>  # Rent + launch
#   bash scripts/vastai_run.sh status          # Check running instances
#   bash scripts/vastai_run.sh ssh             # SSH into instance
#   bash scripts/vastai_run.sh sync            # Sync data to instance
#   bash scripts/vastai_run.sh train <MODEL>   # Start training remotely
#   bash scripts/vastai_run.sh logs            # Tail training logs
#   bash scripts/vastai_run.sh results         # Download results
#   bash scripts/vastai_run.sh destroy         # Destroy instance
# ============================================================
set -euo pipefail

VASTAI="/Users/heinhtetsan/miniforge3/envs/myanmar-asr/bin/vastai"

# Credentials (set as env vars or edit here)
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
WANDB_KEY="${WANDB_API_KEY:-$(grep 'api.wandb.ai' ~/.netrc 2>/dev/null | awk '{print $NF}' || echo '')}"

# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────

get_instance_info() {
    $VASTAI show instances --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data:
    print('NO_INSTANCE')
else:
    inst = data[0]
    print(f\"{inst['id']}|{inst.get('public_ipaddr','?')}|{inst.get('ssh_port','?')}|{inst.get('actual_status','?')}|{inst.get('gpu_name','?')}\")
" 2>/dev/null || echo "NO_INSTANCE"
}

# ──────────────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────────────

cmd_search() {
    echo "═══════════════════════════════════════"
    echo "  GPU Offers on Vast.ai"
    echo "═══════════════════════════════════════"

    echo ""
    echo "── RTX 3090 (Whisper Turbo — cheapest) ──"
    $VASTAI search offers \
        'gpu_name=RTX_3090 num_gpus=1 inet_down>300 disk_space>150 reliability>0.95' \
        --order 'dph_total' --limit 5 \
        --type on-demand 2>/dev/null || echo "  (no results)"

    echo ""
    echo "── RTX 4090 (Dolphin / SeamlessM4T) ──"
    $VASTAI search offers \
        'gpu_name=RTX_4090 num_gpus=1 inet_down>300 disk_space>200 reliability>0.95' \
        --order 'dph_total' --limit 5 \
        --type on-demand 2>/dev/null || echo "  (no results)"

    echo ""
    echo "── A100 (Canary-1B — best accuracy) ──"
    $VASTAI search offers \
        'gpu_ram>=40 num_gpus=1 disk_space>250 reliability>0.95 gpu_name=A100_SXM4' \
        --order 'dph_total' --limit 5 \
        --type on-demand 2>/dev/null || echo "  (no results)"

    echo ""
    echo "── Any GPU >= 24GB VRAM (flexible) ──"
    $VASTAI search offers \
        'gpu_ram>=24 num_gpus=1 inet_down>300 disk_space>200 reliability>0.95' \
        --order 'dph_total' --limit 10 \
        --type on-demand 2>/dev/null || echo "  (no results)"

    echo ""
    echo "═══════════════════════════════════════"
    echo "  Next: bash scripts/vastai_run.sh rent <OFFER_ID> <MODEL>"
    echo "  Models: turbo | dolphin | seamless | canary"
    echo "═══════════════════════════════════════"
}

cmd_rent() {
    local OFFER_ID="${1:?Usage: vastai_run.sh rent <OFFER_ID> <MODEL>}"
    local MODEL="${2:-turbo}"

    # Select Docker image based on model
    local IMAGE=""
    local DISK=200
    case "$MODEL" in
        canary)
            IMAGE="nvcr.io/nvidia/nemo:24.01"
            DISK=250
            ;;
        *)
            IMAGE="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
            DISK=200
            ;;
    esac

    echo "Renting offer ${OFFER_ID} for ${MODEL}..."
    echo "  Image: ${IMAGE}"
    echo "  Disk:  ${DISK}GB"

    $VASTAI create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK" \
        --env "-e HF_TOKEN=${HF_TOKEN} -e WANDB_API_KEY=${WANDB_KEY}" \
        --onstart-cmd "mkdir -p /workspace/{data,models,logs,results,scripts}" \
        2>&1

    echo ""
    echo "Instance created! Waiting for it to start..."
    echo "  Check status: bash scripts/vastai_run.sh status"
    echo "  Then sync:    bash scripts/vastai_run.sh sync"
}

cmd_status() {
    echo "═══════════════════════════════════════"
    echo "  Vast.ai Instance Status"
    echo "═══════════════════════════════════════"
    $VASTAI show instances 2>&1
    echo ""

    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "  No running instances."
    else
        IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"
        echo "  Instance ID: $ID"
        echo "  GPU:         $GPU"
        echo "  Status:      $STATUS"
        echo "  SSH:         ssh -p $PORT root@$IP"
    fi
}

cmd_ssh() {
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances!"
        exit 1
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"
    echo "Connecting to $GPU instance ($IP:$PORT)..."
    ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP"
}

cmd_sync() {
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances! Rent one first."
        exit 1
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"
    echo "Syncing to ${IP}:${PORT}..."
    bash scripts/sync_to_vastai.sh "$IP" "$PORT"
}

cmd_train() {
    local MODEL="${1:-turbo}"
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances!"
        exit 1
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"
    SSH_CMD="ssh -p $PORT -o StrictHostKeyChecking=no root@$IP"

    local SCRIPT=""
    case "$MODEL" in
        turbo)    SCRIPT="train_whisper_turbo.py" ;;
        dolphin)  SCRIPT="train_dolphin.py" ;;
        seamless) SCRIPT="train_seamless.py" ;;
        canary)   SCRIPT="train_canary.py" ;;
        eval)     SCRIPT="evaluate_models.py" ;;
        *)
            echo "Unknown model: $MODEL"
            echo "Options: turbo | dolphin | seamless | canary | eval"
            exit 1 ;;
    esac

    echo "Starting ${MODEL} training on ${GPU}..."
    echo "  Script: /workspace/scripts/${SCRIPT}"
    echo "  Logs:   /workspace/logs/${MODEL}_train.log"
    echo ""

    # Launch in tmux so it persists after SSH disconnect
    $SSH_CMD "tmux kill-session -t ${MODEL} 2>/dev/null; \
              tmux new-session -d -s ${MODEL} \
              'python3 /workspace/scripts/${SCRIPT} 2>&1 | tee /workspace/logs/${MODEL}_train.log; \
               echo DONE; sleep 3600'"

    echo "✅ Training launched in tmux session '${MODEL}'"
    echo "   Monitor:"
    echo "     bash scripts/vastai_run.sh ssh   → tmux attach -t ${MODEL}"
    echo "     https://wandb.ai → project myanmar-asr"
}

cmd_logs() {
    local MODEL="${1:-turbo}"
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances!"
        exit 1
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"
    ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" \
        "tail -50 /workspace/logs/${MODEL}_train.log 2>/dev/null || echo 'No log file yet'"
}

cmd_results() {
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances!"
        exit 1
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"

    echo "Downloading results..."
    mkdir -p results

    rsync -avz --progress \
        root@"$IP":/workspace/results/ \
        results/ \
        -e "ssh -p $PORT -o StrictHostKeyChecking=no" 2>&1

    echo ""
    echo "Results saved to results/"
    ls -la results/
}

cmd_destroy() {
    local INFO=$(get_instance_info)
    if [ "$INFO" = "NO_INSTANCE" ]; then
        echo "No running instances."
        exit 0
    fi
    IFS='|' read -r ID IP PORT STATUS GPU <<< "$INFO"

    echo "⚠ Destroying instance $ID ($GPU)..."
    echo "  Make sure you've downloaded results first!"
    read -p "  Continue? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy] ]]; then
        $VASTAI destroy instance "$ID"
        echo "✅ Instance destroyed"
    else
        echo "Cancelled."
    fi
}

# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

CMD="${1:-help}"
shift || true

case "$CMD" in
    search)   cmd_search ;;
    rent)     cmd_rent "$@" ;;
    status)   cmd_status ;;
    ssh)      cmd_ssh ;;
    sync)     cmd_sync ;;
    train)    cmd_train "$@" ;;
    logs)     cmd_logs "$@" ;;
    results)  cmd_results ;;
    destroy)  cmd_destroy ;;
    *)
        echo "Myanmar ASR — Vast.ai Orchestrator"
        echo ""
        echo "Usage: bash scripts/vastai_run.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  search              Search for GPU offers"
        echo "  rent <ID> <MODEL>   Rent instance (turbo|dolphin|seamless|canary)"
        echo "  status              Show running instances"
        echo "  ssh                 SSH into instance"
        echo "  sync                Sync data + scripts to instance"
        echo "  train <MODEL>       Start training (turbo|dolphin|seamless|canary|eval)"
        echo "  logs <MODEL>        Tail training logs"
        echo "  results             Download results to local"
        echo "  destroy             Destroy instance"
        echo ""
        echo "Recommended workflow:"
        echo "  1. bash scripts/vastai_run.sh search"
        echo "  2. bash scripts/vastai_run.sh rent <OFFER_ID> turbo"
        echo "  3. bash scripts/vastai_run.sh sync"
        echo "  4. bash scripts/vastai_run.sh train turbo"
        echo "  5. bash scripts/vastai_run.sh logs turbo"
        echo "  6. bash scripts/vastai_run.sh train eval"
        echo "  7. bash scripts/vastai_run.sh results"
        echo "  8. bash scripts/vastai_run.sh destroy"
        ;;
esac
