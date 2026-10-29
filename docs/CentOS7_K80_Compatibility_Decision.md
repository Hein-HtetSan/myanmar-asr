# CentOS 7 / CUDA 11.4 / Tesla K80 compatibility decision

**Status:** Ready to deploy for VPS acceptance testing. GPU execution, installation, and model memory use remain unverified until the commands below pass on the target VPS.

## Decision

Run GPU inference and limited fine-tuning in a native Python 3.10 Conda environment using PyTorch 1.12.1 with its CUDA 11.3 wheel. Start K80 training with Whisper Small, batch size 1, gradient accumulation 32, a frozen encoder, and float32 precision.

## Why

- The Tesla K80 is a Kepler GPU with compute capability 3.7 and 12 GB of memory per GPU. Its last supported CUDA family is 11.x and its last driver branch is R470. [NVIDIA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
- A CUDA 11.3 PyTorch wheel can run under a compatible CUDA 11.4 driver. PyTorch publishes the 1.12.1 CUDA 11.3 build. [NVIDIA CUDA 11.4 release notes](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-toolkit-release-notes/), [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- The repository's original training defaults target newer 24 GB GPUs. The K80 needs float32 and a smaller model to keep memory use practical.

## Implementation in this repo

1. Install the versions in [`requirements-k80.txt`](../requirements-k80.txt) after installing `torch==1.12.1+cu113`. Follow the commands in the [README](../README.md#run-inference-on-a-centos-7-k80-vps).
2. Run [`check_k80.py`](../scripts/infra/check_k80.py) on each K80 device. It checks SM 3.7 support, executes a float32 CUDA kernel, and can run a Whisper inference smoke test.
3. Use `DEFAULT_MODEL=whisper-small` for the inference server. Point `WHISPER_SMALL_PATH` at a fine-tuned checkpoint when one is available.
4. Use `train_whisper_turbo.py --model_id openai/whisper-small --batch_size 1 --grad_accum 32 --freeze_encoder --no_push` for the first training run. Supply the dataset and output paths as shown in the README.

## Acceptance checks on the CentOS 7 host

- `python -m pip check` succeeds in the Conda environment.
- `python scripts/infra/check_k80.py --model openai/whisper-tiny` passes.
- The inference server starts with `DEFAULT_MODEL=whisper-small`, `/health` reports the K80, and a short WAV upload to `/transcribe` returns text.
- A short Whisper Small training run completes without out-of-memory or unsupported-kernel errors before attempting the full dataset.

## VPS deployment runbook

The API below listens on `127.0.0.1:8080`. Reach it through an SSH tunnel during testing. If it must be public, put an authenticated HTTPS reverse proxy in front of it before opening the port; the API itself has no authentication.

### 1. Prepare the VPS

1. Confirm CentOS 7 x86_64, NVIDIA R470 driver, and the K80 with `cat /etc/centos-release`, `uname -m`, and `nvidia-smi`.
2. Use a non-root service account with the repository cloned under its home directory. Install Conda or Miniforge for that account.
3. From the repository root, run:

```bash
conda create -n myanmar-asr-k80 python=3.10 -y
conda activate myanmar-asr-k80
python -m pip install 'torch==1.12.1+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements-k80.txt
python -m pip check
python scripts/infra/check_k80.py --model openai/whisper-tiny
```

`check_k80.py` downloads Whisper Tiny for its end-to-end test. Keep the resulting model cache available to the service account. The `requirements-demo.txt` and Vast.ai setup script target different environments and are not the VPS install path.

### 2. Start the API and test it locally

```bash
SERVE_HOST=127.0.0.1 DEFAULT_MODEL=whisper-small python scripts/deploy/inference_server.py
```

In a second shell on the VPS:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/models
curl -fsS -F 'file=@/path/to/short-burmese.wav' http://127.0.0.1:8080/transcribe
```

The health response should report `"gpu": true`, a K80 name, and `"whisper-small"` in `models_loaded`. The transcription response should contain `text`, `model: "whisper-small"`, and an inference time. Check the text against the known speech manually; a successful HTTP response alone does not establish ASR quality.

### 3. Keep the API running

After the local test passes, create `/etc/systemd/system/myanmar-asr.service` using the template below. Replace the `asr` username and both `/home/asr/...` paths with the actual service account and `command -v python` path from its activated Conda environment.

```ini
[Unit]
Description=Myanmar ASR inference API
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=asr
WorkingDirectory=/home/asr/myanmar-asr
Environment=SERVE_HOST=127.0.0.1
Environment=SERVE_PORT=8080
Environment=DEFAULT_MODEL=whisper-small
ExecStart=/home/asr/miniconda3/envs/myanmar-asr-k80/bin/python scripts/deploy/inference_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now myanmar-asr
sudo systemctl status myanmar-asr
sudo journalctl -u myanmar-asr -n 100 --no-pager
```

Repeat the health and transcription requests after restarting the service. From a workstation, use `ssh -N -L 8080:127.0.0.1:8080 asr@VPS_HOST` and send requests to `http://127.0.0.1:8080`.

## VPS test record

Fill this in on the target VPS before marking the deployment verified. Record the command output or log path for any failure.

| Check | Expected result | Result | Evidence / notes |
| --- | --- | --- | --- |
| OS and driver | CentOS 7 x86_64; R470 driver; K80 visible | Pending | |
| Dependency resolution | `python -m pip check` exits 0 | Pending | |
| CUDA and model smoke test | `check_k80.py --model openai/whisper-tiny` prints both PASS lines | Pending | |
| API startup | `myanmar-asr` remains active after restart | Pending | |
| Health endpoint | GPU true; K80; Whisper Small loaded | Pending | |
| Audio request | HTTP 200; JSON text and timing fields | Pending | |
| Known Burmese sample | Output reviewed against expected speech | Pending | |
| Memory | `nvidia-smi` shows no OOM during repeated requests | Pending | |

**Go / no-go:** Mark the VPS deployment verified only when all rows pass. If a row fails, keep the API bound to loopback, record the failing output, and fix that specific blocker before exposing the service to users.

## Limits

These checks require the actual CentOS 7/K80 host. The 809M–2.3B parameter training experiments and NeMo Canary are not validated for one 12 GB K80. Their original performance and training-time figures should not be used as K80 estimates.
