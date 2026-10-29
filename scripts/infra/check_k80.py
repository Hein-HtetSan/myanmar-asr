#!/usr/bin/env python3
"""Fail early when the CentOS 7 / K80 CUDA environment cannot execute PyTorch kernels."""

import argparse
import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional Whisper model/path for an end-to-end inference smoke test")
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers

    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}; wheel CUDA: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__}; NumPy: {np.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: PyTorch cannot see CUDA. Check the NVIDIA driver and CUDA wheel.")
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}; compute capability: {capability}")
    print(f"Wheel architectures: {torch.cuda.get_arch_list()}")
    if capability != (3, 7):
        raise SystemExit("FAIL: selected CUDA device is not a Tesla K80 (SM 3.7). Set CUDA_VISIBLE_DEVICES.")
    if torch.version.cuda != "11.3":
        raise SystemExit("FAIL: install the CUDA 11.3 PyTorch wheel for this driver.")
    if "sm_37" not in torch.cuda.get_arch_list():
        raise SystemExit("FAIL: this PyTorch wheel has no SM 3.7 kernels.")
    x = torch.ones((64, 64), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    assert (x @ x)[0, 0].item() == 64.0
    print("PASS: float32 CUDA matrix multiplication")

    if args.model:
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=args.model,
                        device=0, torch_dtype=torch.float32,
                        generate_kwargs={"language": "my", "task": "transcribe"})
        result = pipe({"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000})
        assert "text" in result
        print(f"PASS: Whisper inference ({args.model})")


if __name__ == "__main__":
    main()
