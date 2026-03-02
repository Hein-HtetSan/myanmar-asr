#!/usr/bin/env python3
"""Remote setup script — runs on Vast.ai instance."""
import subprocess, sys, os

def run(cmd):
    print(f">>> {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(f"STDERR: {r.stderr.strip()}")
    return r.returncode

print("=== Installing packages ===")
run("pip install -q 'transformers>=4.44.0' 'datasets>=3.0.0' 'accelerate>=0.34.0' "
    "'evaluate>=0.4.0' 'jiwer>=3.0.0' soundfile librosa mlflow boto3 "
    "tensorboard tqdm safetensors sentencepiece protobuf huggingface_hub")

print("\n=== Setting up HF token ===")
token = os.environ.get("HF_TOKEN", "")
if token:
    from huggingface_hub import login
    login(token=token)
    print("HF login OK")
else:
    print("No HF_TOKEN found in env")

print("\n=== GPU info ===")
run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")

print("\n=== MLflow tunnel ===")
run("curl -s -o /dev/null -w 'HTTP %{http_code}' http://localhost:5050/ || echo 'not reachable'")

print("\n=== Python packages ===")
run("python3 --version")
import torch, transformers, datasets
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")

print("\n=== Scripts check ===")
run("ls -la /workspace/scripts/")

print("\n=== Data check ===")
run("du -sh /workspace/data/myanmar_asr/ 2>/dev/null || echo 'No data yet'")

print("\n=== Setup complete! ===")
