#!/bin/bash
# ============================================================
#  Myanmar ASR — One-Click Training on Vast.ai
#
#  Searches for cheapest GPU, rents it, syncs data,
#  sets up MLflow reverse tunnel, trains, evaluates,
#  downloads results, and destroys the instance.
#
#  Usage:
#    bash scripts/oneclick_train.sh turbo      # Whisper v3 Turbo
#    bash scripts/oneclick_train.sh dolphin    # Dolphin ASR
#    bash scripts/oneclick_train.sh seamless   # SeamlessM4T v2
#    bash scripts/oneclick_train.sh canary     # Canary-1B
#    bash scripts/oneclick_train.sh eval       # Evaluate all
#    bash scripts/oneclick_train.sh resume     # Resume on existing instance
#    bash scripts/oneclick_train.sh destroy    # Destroy instance
#
#  Tracks experiments to local MLflow (http://localhost:5050)
#  via SSH reverse tunnel.
# ============================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"  # scripts/infra → project root
VASTAI="$(which vastai 2>/dev/null || echo "$HOME/miniforge3/envs/myanmar-asr/bin/vastai")"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
MLFLOW_PORT=5050
STATE_FILE="$PROJECT_DIR/.vastai_state"  # Persists instance info between runs

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── State management (survives script restarts) ───────
save_state() { echo "$1|$2|$3" > "$STATE_FILE"; }
load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        IFS='|' read -r INST_ID VAST_IP SSH_PORT < "$STATE_FILE"
        return 0
    fi
    return 1
}
clear_state() { rm -f "$STATE_FILE"; }

# ── Helpers ───────────────────────────────────────────
check_deps() {
    command -v ssh >/dev/null || { err "ssh not found"; exit 1; }
    command -v rsync >/dev/null || { err "rsync not found"; exit 1; }
    [[ -x "$VASTAI" ]] || { err "vastai CLI not found at $VASTAI"; exit 1; }
    [[ -n "$HF_TOKEN" ]] || { warn "HF_TOKEN not set — models won't push to Hub"; }
}

get_instance() {
    # Try state file first
    if load_state && [[ -n "${INST_ID:-}" ]]; then
        # Verify instance still exists
        local status
        status=$($VASTAI show instance "$INST_ID" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('actual_status','unknown'))
" 2>/dev/null || echo "gone")
        if [[ "$status" != "gone" && "$status" != "unknown" ]]; then
            return 0
        fi
        warn "Saved instance $INST_ID is $status, clearing state"
        clear_state
    fi

    # Query Vast.ai for running instances
    local raw
    raw=$($VASTAI show instances --raw 2>/dev/null)
    local count
    count=$(echo "$raw" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [[ "$count" -gt 0 ]]; then
        eval $(echo "$raw" | python3 -c "
import sys,json
d=json.load(sys.stdin)[0]
print(f\"INST_ID={d['id']}\")
print(f\"VAST_IP={d.get('ssh_host','') or d.get('public_ipaddr','')}\")
print(f\"SSH_PORT={d.get('ssh_port','')}\")
")
        save_state "$INST_ID" "$VAST_IP" "$SSH_PORT"
        return 0
    fi
    return 1
}

wait_for_instance() {
    info "Waiting for instance $INST_ID to be ready..."
    local max_wait=300  # 5 min
    local elapsed=0
    while (( elapsed < max_wait )); do
        local raw
        raw=$($VASTAI show instance "$INST_ID" --raw 2>/dev/null)
        local status ip port
        eval $(echo "$raw" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"status={d.get('actual_status','unknown')}\")
print(f\"ip={d.get('ssh_host','') or d.get('public_ipaddr','')}\")
print(f\"port={d.get('ssh_port','')}\")
" 2>/dev/null)

        if [[ "$status" == "running" && -n "$ip" && -n "$port" ]]; then
            VAST_IP="$ip"
            SSH_PORT="$port"
            save_state "$INST_ID" "$VAST_IP" "$SSH_PORT"
            ok "Instance ready: $VAST_IP:$SSH_PORT"
            # Wait for SSH daemon to be fully ready, then test connection
            info "Waiting for SSH to come up..."
            local ssh_wait=0
            while (( ssh_wait < 120 )); do
                if ssh -p "$SSH_PORT" -i ~/.ssh/id_ed25519 \
                    -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                    root@"$VAST_IP" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
                    ok "SSH is ready"
                    return 0
                fi
                sleep 10
                ssh_wait=$((ssh_wait + 10))
                printf "\r  SSH: waiting... (%ds)" "$ssh_wait"
            done
            echo ""
            warn "SSH not responding after 120s, proceeding anyway..."
            return 0
        fi
        printf "\r  Status: %-12s (${elapsed}s / ${max_wait}s)" "$status"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo ""
    err "Instance did not become ready in ${max_wait}s"
    return 1
}

ssh_cmd() {
    ssh -p "$SSH_PORT" -i ~/.ssh/id_ed25519 \
        -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
        -o ServerAliveInterval=30 root@"$VAST_IP" "$@"
}

# ── GPU Search & Rent ─────────────────────────────────
find_and_rent() {
    local model="$1"
    local min_vram=20
    local image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
    local disk=200
    local max_price=0.40
    local gpu_filter="gpu_name=RTX_4090"

    case "$model" in
        turbo)
            min_vram=20; disk=150; max_price=0.40
            ;;
        dolphin|seamless)
            min_vram=20; disk=200; max_price=0.40
            ;;
        canary)
            image="nvcr.io/nvidia/nemo:24.01"
            min_vram=24; disk=250; max_price=0.60
            ;;
    esac

    info "Searching for RTX 4090: <\$${max_price}/hr..."

    # Search for cheapest RTX 4090 offer
    local offer_id
    offer_id=$($VASTAI search offers \
        "${gpu_filter} num_gpus=1 inet_down>200 disk_space>=${disk} reliability>0.90 dph_total<${max_price}" \
        --order 'dph_total' --limit 1 --type on-demand --raw 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if d else '')" 2>/dev/null)

    if [[ -z "$offer_id" ]]; then
        warn "No RTX 4090 under \$${max_price}/hr. Trying any GPU ≥${min_vram}GB..."
        offer_id=$($VASTAI search offers \
            "gpu_ram>=${min_vram} num_gpus=1 inet_down>200 disk_space>=${disk} reliability>0.85 dph_total<0.50" \
            --order 'dph_total' --limit 1 --type on-demand --raw 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if d else '')" 2>/dev/null)
    fi

    if [[ -z "$offer_id" ]]; then
        err "No GPU found meeting requirements. Check vast.ai manually."
        exit 1
    fi

    # Get offer details
    local gpu_name price
    eval $($VASTAI search offers \
        "${gpu_filter} num_gpus=1 inet_down>200 disk_space>=${disk} reliability>0.85 dph_total<${max_price}" \
        --order 'dph_total' --limit 1 --type on-demand --raw 2>/dev/null \
        | python3 -c "
import sys, json
d = json.load(sys.stdin)[0]
gn = d.get('gpu_name', '?')
pr = d.get('dph_total', '?')
print(f'gpu_name={gn!r}')
print(f'price={pr!r}')
")

    info "Best offer: $gpu_name @ \$$price/hr (ID: $offer_id)"
    info "Renting with image: $(basename $image)"

    INST_ID=$($VASTAI create instance "$offer_id" \
        --image "$image" \
        --disk "$disk" \
        --env "-e HF_TOKEN=${HF_TOKEN}" \
        --onstart-cmd "mkdir -p /workspace/{data,models,logs,results,scripts}" \
        --raw 2>&1 | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('new_contract',''))
" 2>/dev/null)

    if [[ -z "$INST_ID" ]]; then
        err "Failed to rent instance"
        exit 1
    fi

    ok "Rented instance: $INST_ID ($gpu_name @ \$$price/hr)"
    save_state "$INST_ID" "" ""
    wait_for_instance
}

# ── Sync Data ─────────────────────────────────────────
sync_data() {
    info "Syncing data to $VAST_IP:$SSH_PORT..."
    local SSH="ssh -p ${SSH_PORT} -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

    # Create dirs
    ssh_cmd "mkdir -p /workspace/{data,scripts,logs,results,models}"

    # Sync HF dataset (for Whisper/Dolphin/Seamless)
    info "[1/4] Syncing HF dataset (augmented)..."
    rsync -az --info=progress2 \
        "$PROJECT_DIR/combined/myanmar_asr_augmented/" \
        root@"$VAST_IP":/workspace/data/myanmar_asr/ \
        -e "$SSH"

    # Sync NeMo manifests + audio (for Canary)
    if [[ -d "$PROJECT_DIR/exports/nemo_audio" ]]; then
        info "[2/4] Syncing NeMo audio..."
        rsync -az --info=progress2 \
            "$PROJECT_DIR/exports/nemo_audio/" \
            root@"$VAST_IP":/workspace/data/nemo_audio/ \
            -e "$SSH"

        rsync -az "$PROJECT_DIR/exports/nemo_"*"_manifest.jsonl" \
            root@"$VAST_IP":/workspace/data/ \
            -e "$SSH"
    fi

    # Sync scripts
    info "[3/4] Syncing training scripts..."
    rsync -az \
        "$PROJECT_DIR/scripts/training/" \
        root@"$VAST_IP":/workspace/scripts/training/ \
        -e "$SSH"
    rsync -az \
        "$PROJECT_DIR/scripts/evaluation/" \
        root@"$VAST_IP":/workspace/scripts/evaluation/ \
        -e "$SSH"
    rsync -az \
        "$PROJECT_DIR/scripts/utils/" \
        root@"$VAST_IP":/workspace/scripts/utils/ \
        -e "$SSH"

    # Server setup
    info "[4/4] Installing packages..."
    ssh_cmd "pip install -q transformers>=4.44.0 datasets>=3.0.0 accelerate>=0.34.0 \
        evaluate>=0.4.0 jiwer>=3.0.0 soundfile librosa mlflow boto3 \
        tensorboard tqdm safetensors sentencepiece protobuf 2>&1 | tail -3"

    ok "Sync complete"
}

# ── SSH Reverse Tunnel for MLflow ────────────────────
start_tunnel() {
    info "Starting SSH reverse tunnel (MLflow :${MLFLOW_PORT})..."

    # Kill any existing tunnel
    pkill -f "ssh.*-R.*${MLFLOW_PORT}.*root@${VAST_IP}" 2>/dev/null || true
    sleep 1

    # Start tunnel in background
    # Remote localhost:5050 → Local localhost:5050 (MLflow)
    ssh -f -N \
        -R ${MLFLOW_PORT}:localhost:${MLFLOW_PORT} \
        -p "$SSH_PORT" \
        -i ~/.ssh/id_ed25519 \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ExitOnForwardFailure=yes \
        root@"$VAST_IP"

    ok "MLflow tunnel active (Vast.ai :${MLFLOW_PORT} → local :${MLFLOW_PORT})"
}

# ── Train Model ──────────────────────────────────────
train_model() {
    local model="$1"
    local script=""

    case "$model" in
        turbo)    script="training/train_whisper_turbo.py" ;;
        dolphin)  script="training/train_dolphin.py" ;;
        seamless) script="training/train_seamless.py" ;;
        canary)   script="training/train_canary.py" ;;
        eval)     script="evaluation/evaluate_models.py" ;;
        *)        err "Unknown model: $model"; exit 1 ;;
    esac

    info "Starting training: $model ($script)"

    # Launch in tmux with MLflow env vars
    ssh_cmd "tmux kill-session -t train 2>/dev/null || true; \
             tmux new-session -d -s train \
             'export MLFLOW_TRACKING_URI=http://localhost:${MLFLOW_PORT}; \
              export MLFLOW_EXPERIMENT_NAME=myanmar-asr; \
              export WANDB_MODE=disabled; \
              python3 /workspace/scripts/${script} 2>&1 | tee /workspace/logs/${model}_train.log; \
              echo TRAINING_DONE; sleep 86400'"

    ok "Training launched in tmux session 'train'"
    echo ""
    echo "  ┌──────────────────────────────────────────┐"
    echo "  │  Monitor training:                       │"
    echo "  │                                          │"
    echo "  │  MLflow UI:  http://localhost:${MLFLOW_PORT}       │"
    echo "  │  Tail logs:  bash $0 logs                │"
    echo "  │  SSH:        bash $0 ssh                 │"
    echo "  │                                          │"
    echo "  │  When done:  bash $0 results             │"
    echo "  │  Clean up:   bash $0 destroy             │"
    echo "  └──────────────────────────────────────────┘"
}

# ── Monitor Logs ─────────────────────────────────────
tail_logs() {
    local model="${1:-}"
    get_instance || { err "No running instance"; exit 1; }

    if [[ -z "$model" ]]; then
        # Find the latest log
        ssh_cmd "ls -t /workspace/logs/*_train.log 2>/dev/null | head -1 | xargs tail -50"
    else
        ssh_cmd "tail -50 /workspace/logs/${model}_train.log 2>/dev/null || echo 'No log file yet'"
    fi
}

# ── Check Training Status ────────────────────────────
check_status() {
    get_instance || { err "No running instance"; exit 1; }

    info "Instance: $INST_ID ($VAST_IP:$SSH_PORT)"

    # Check if training is still running
    local tmux_status
    tmux_status=$(ssh_cmd "tmux has-session -t train 2>&1 && echo running || echo stopped")

    if [[ "$tmux_status" == *"running"* ]]; then
        ok "Training is running"
        echo ""
        info "Last 10 lines:"
        ssh_cmd "ls -t /workspace/logs/*_train.log 2>/dev/null | head -1 | xargs tail -10"
    else
        warn "Training session not found (may have finished)"
        ssh_cmd "ls -t /workspace/logs/*_train.log 2>/dev/null | head -1 | xargs tail -5"
    fi

    # Check if TRAINING_DONE marker exists
    local done
    done=$(ssh_cmd "grep -c TRAINING_DONE /workspace/logs/*_train.log 2>/dev/null || echo 0")
    if [[ "$done" -gt 0 ]]; then
        ok "Training has completed!"
    fi
}

# ── Download Results ─────────────────────────────────
download_results() {
    get_instance || { err "No running instance"; exit 1; }
    local SSH="ssh -p ${SSH_PORT} -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

    info "Downloading results and models..."
    mkdir -p "$PROJECT_DIR/results"

    # Results JSON files
    rsync -az root@"$VAST_IP":/workspace/results/ \
        "$PROJECT_DIR/results/" \
        -e "$SSH" 2>/dev/null || true

    # Training logs
    rsync -az root@"$VAST_IP":/workspace/logs/ \
        "$PROJECT_DIR/logs/" \
        -e "$SSH" 2>/dev/null || true

    ok "Results downloaded to $PROJECT_DIR/results/"
    ls -la "$PROJECT_DIR/results/" 2>/dev/null

    echo ""
    info "To download trained model checkpoint, run:"
    echo "  rsync -az -e 'ssh -p $SSH_PORT' root@$VAST_IP:/workspace/models/<model>/final/ models/<model>/"
}

# ── SSH Into Instance ────────────────────────────────
ssh_into() {
    get_instance || { err "No running instance"; exit 1; }
    info "Connecting to $VAST_IP:$SSH_PORT..."
    ssh -p "$SSH_PORT" -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@"$VAST_IP"
}

# ── Destroy Instance ─────────────────────────────────
destroy_instance() {
    if ! get_instance; then
        ok "No running instances"
        clear_state
        return
    fi

    warn "Destroying instance $INST_ID..."
    echo "  Have you downloaded results? (bash $0 results)"
    read -p "  Continue? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy] ]]; then
        # Kill tunnel first
        pkill -f "ssh.*-R.*${MLFLOW_PORT}.*root@${VAST_IP}" 2>/dev/null || true
        $VASTAI destroy instance "$INST_ID" 2>&1
        clear_state
        ok "Instance destroyed"
    else
        info "Cancelled"
    fi
}

# ── Full Pipeline (one-click) ────────────────────────
run_full_pipeline() {
    local model="$1"

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Myanmar ASR — One-Click Training               ║"
    echo "║  Model: $(printf '%-40s' "$model")║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    check_deps

    # Step 1: Check if we already have an instance
    if get_instance; then
        ok "Using existing instance: $INST_ID ($VAST_IP:$SSH_PORT)"
    else
        # Step 2: Find and rent GPU
        find_and_rent "$model"
    fi

    # Step 3: Sync data + scripts
    sync_data

    # Step 4: Start MLflow reverse tunnel
    start_tunnel

    # Step 5: Launch training
    train_model "$model"
}

# ── Resume (reconnect tunnel + show status) ──────────
resume() {
    get_instance || { err "No running instance found"; exit 1; }
    ok "Instance: $INST_ID ($VAST_IP:$SSH_PORT)"
    start_tunnel
    check_status
}

# ── Main ─────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
    turbo|dolphin|seamless|canary)
        run_full_pipeline "$CMD"
        ;;
    eval)
        if get_instance; then
            start_tunnel
            train_model "eval"
        else
            err "No running instance. Start training first."
            exit 1
        fi
        ;;
    resume)   resume ;;
    status)   check_status ;;
    logs)     tail_logs "$@" ;;
    ssh)      ssh_into ;;
    results)  download_results ;;
    destroy)  destroy_instance ;;
    *)
        echo "Myanmar ASR — One-Click Training"
        echo ""
        echo "Usage: bash scripts/oneclick_train.sh <command>"
        echo ""
        echo "Training (full pipeline — rent, sync, train):"
        echo "  turbo       Whisper v3 Turbo (cheapest ~\$0.05-0.10/hr)"
        echo "  dolphin     Dolphin ASR (needs 16GB+ VRAM)"
        echo "  seamless    SeamlessM4T v2 (needs 16GB+ VRAM)"
        echo "  canary      Canary-1B (needs 24GB+ VRAM, NeMo)"
        echo ""
        echo "Management:"
        echo "  resume      Reconnect MLflow tunnel + show status"
        echo "  status      Check training progress"
        echo "  logs [m]    Tail training logs"
        echo "  ssh         SSH into instance"
        echo "  results     Download results to local"
        echo "  eval        Run model comparison on instance"
        echo "  destroy     Destroy instance (stop billing)"
        echo ""
        echo "MLflow UI: http://localhost:${MLFLOW_PORT}"
        echo ""
        echo "Example:"
        echo "  bash scripts/oneclick_train.sh turbo   # One click!"
        echo "  bash scripts/oneclick_train.sh logs    # Monitor"
        echo "  bash scripts/oneclick_train.sh results # Download"
        echo "  bash scripts/oneclick_train.sh destroy # Stop billing"
        ;;
esac
