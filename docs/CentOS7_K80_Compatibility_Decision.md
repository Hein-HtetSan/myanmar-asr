# CentOS 7 / CUDA 11.4 / Tesla K80 compatibility decision

## Decision

Run GPU inference and limited fine-tuning in a native Python 3.10 Conda environment using PyTorch 1.12.1 with its CUDA 11.3 wheel. Keep the existing presentation container CPU-based. Start K80 training with Whisper Small, batch size 1, gradient accumulation 32, a frozen encoder, and float32 precision.

## Why

- The Tesla K80 is a Kepler GPU with compute capability 3.7 and 12 GB of memory per GPU. Its last supported CUDA family is 11.x and its last driver branch is R470. [NVIDIA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
- A CUDA 11.3 PyTorch wheel can run under a compatible CUDA 11.4 driver. PyTorch publishes the 1.12.1 CUDA 11.3 build. [NVIDIA CUDA 11.4 release notes](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-toolkit-release-notes/), [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- The repository's original training defaults target newer 24 GB GPUs. The K80 needs float32 and a smaller model to keep memory use practical.

## Implementation in this repo

1. Install the versions in [`requirements-k80.txt`](../requirements-k80.txt) after installing `torch==1.12.1+cu113`. Follow the commands in the [README](../README.md#centos-7--cuda-114--tesla-k80).
2. Run [`check_k80.py`](../scripts/infra/check_k80.py) on each K80 device. It checks SM 3.7 support, executes a float32 CUDA kernel, and can run a Whisper inference smoke test.
3. Use `DEFAULT_MODEL=whisper-small` for the inference server. Point `WHISPER_SMALL_PATH` at a fine-tuned checkpoint when one is available.
4. Use `train_whisper_turbo.py --model_id openai/whisper-small --batch_size 1 --grad_accum 32 --freeze_encoder --no_push` for the first training run. Supply the dataset and output paths as shown in the README.

## Acceptance checks on the CentOS 7 host

- `python -m pip check` succeeds in the Conda environment.
- `python scripts/infra/check_k80.py --model openai/whisper-tiny` passes.
- The inference server starts with `DEFAULT_MODEL=whisper-small`, `/health` reports the K80, and a short WAV upload to `/transcribe` returns text.
- A short Whisper Small training run completes without out-of-memory or unsupported-kernel errors before attempting the full dataset.

## Limits

These checks require the actual CentOS 7/K80 host. The 809M–2.3B parameter training experiments and NeMo Canary are not validated for one 12 GB K80. Their original performance and training-time figures should not be used as K80 estimates.
